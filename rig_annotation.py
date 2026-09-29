"""
rig_annotation.py -- geometry and file I/O for hand-annotated electrode geometry (any rig).

A rig annotation is made once per camera/rig POSITION with `rig_annotate.py` (napari) and saved
as JSON. It holds, in raw image coordinates (row, col; pycine orientation, frame 0 = top row):

  electrode_A, electrode_B : polylines traced along each electrode's METAL edge facing the gap
  gap_line                 : two points drawn across the MINIMUM gap, metal edge to metal edge
  gap_mm                   : the known physical minimum gap (from the folder name or --gap-mm)

Everything downstream is derived here, so the JSON never stores a number it could recompute:
  px_per_mm        = |gap_line| / gap_mm                       (primary calibration)
  gap_from_edges   = closest approach of the two traced edges  (consistency check on the gap line)
  bisector         = angle bisector of the two edge directions, through the gap midpoint
  electrode_edges  = densified edge point clouds + gap ends, the npz analyze_blade --electrode-edges reads
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import electrode_geometry as eg

DENSIFY_STEP_PX = 0.25      # edge polylines sampled every 0.25 px for the closest-approach search
SCHEMA = "rig_annotation/1"


def densify(path, step: float = DENSIFY_STEP_PX) -> np.ndarray:
    """Points every ~step px along a polyline (N,2) [row, col], endpoints included."""
    path = np.asarray(path, float)
    if len(path) < 2:
        return path.reshape(-1, 2)
    out = []
    for a, b in zip(path[:-1], path[1:]):
        n = max(int(np.ceil(np.hypot(*(b - a)) / step)), 1)
        out.append(a + np.linspace(0, 1, n, endpoint=False)[:, None] * (b - a))
    out.append(path[-1:])
    return np.vstack(out)


def edge_direction(path) -> np.ndarray:
    """Unit direction of an edge polyline: principal axis of its densified points (SVD),
    signed to point from the polyline's first vertex towards its last."""
    P = densify(path)
    if len(P) < 2:
        raise ValueError("an edge needs at least two vertices")
    _, _, vt = np.linalg.svd(P - P.mean(0))
    d = vt[0]
    if (P[-1] - P[0]) @ d < 0:
        d = -d
    return d / np.linalg.norm(d)


def geometry(ann: dict) -> dict:
    """Derived geometry of an annotation dict (see module docstring). Raises on missing parts."""
    A = densify(ann["electrode_A"]) if ann.get("electrode_A") else None
    B = densify(ann["electrode_B"]) if ann.get("electrode_B") else None
    g = ann.get("gap_line")
    gap_mm = float(ann["gap_mm"])
    out: dict = {"gap_mm": gap_mm}
    if g is not None:
        g = np.asarray(g, float)
        if g.shape != (2, 2):
            raise ValueError("gap_line must be exactly two points")
        out["gap_px"] = float(np.hypot(*(g[1] - g[0])))
        out["gap_ends"] = g.tolist()
    if A is not None and B is not None:
        ta, tb, dmin = eg.nearest_facing_tips(A, B)
        out["gap_from_edges_px"] = float(dmin)
        out["edge_closest_points"] = [ta.tolist(), tb.tolist()]
        dA, dB = edge_direction(ann["electrode_A"]), edge_direction(ann["electrode_B"])
        # orient both directions AWAY from the gap (from the closest point outward) before bisecting
        if (A.mean(0) - ta) @ dA < 0:
            dA = -dA
        if (B.mean(0) - tb) @ dB < 0:
            dB = -dB
        opening = float(np.degrees(np.arccos(np.clip(dA @ dB, -1, 1))))
        bis = dA + dB
        bis = bis / np.linalg.norm(bis) if np.linalg.norm(bis) > 1e-9 else np.array([-1.0, 0.0])
        mid = 0.5 * (np.asarray(out.get("gap_ends", [ta, tb]))[0] + np.asarray(out.get("gap_ends", [ta, tb]))[1])
        out.update(opening_deg=opening, bisector_point=mid.tolist(), bisector_dir=bis.tolist())
    if "gap_px" not in out and "gap_from_edges_px" in out:
        out["gap_px"] = out["gap_from_edges_px"]
        out["gap_ends"] = out["edge_closest_points"]
        out["gap_source"] = "closest approach of the traced edges (no gap line drawn)"
    else:
        out["gap_source"] = "drawn gap line"
    if "gap_px" not in out:
        raise ValueError("draw a gap line or both electrode edges")
    out["px_per_mm"] = out["gap_px"] / gap_mm
    out["gap_row"] = float(np.mean(np.asarray(out["gap_ends"])[:, 0]))
    return out


def straight_blades(ann: dict, shape) -> dict:
    """STRAIGHT-EDGED electrodes (the BLADE rig): the drawn polylines only fix each edge's LINE (principal
    axis through the drawn points); how far along the blade they were drawn does not matter.

    Each blade's edge runs from the CUTOFF -- the foot of the perpendicular from its end of the drawn gap line
    onto its line -- outward, away from the other blade, to the image border. A = the LEFT blade (smaller
    column), whichever layer it was drawn in. Requires a gap line. Returns
      A, B          : (M,2) [row, col] points every DENSIFY_STEP_PX along each extended edge
      A_tip, B_tip  : the cutoffs (on the lines)
      gap_end_offsets_px : perpendicular distance of each drawn gap end from its line [A, B]
      opening_deg, ab_swapped (True if the layer called 'electrode A' was the RIGHT blade)
    """
    if not ann.get("gap_line") or not (ann.get("electrode_A") and ann.get("electrode_B")):
        raise ValueError("straight blades need both edges and a gap line")
    H, W = shape
    lines = []
    for key in ("electrode_A", "electrode_B"):
        lines.append((densify(ann[key]).mean(0), edge_direction(ann[key])))
    swapped = lines[0][0][1] > lines[1][0][1]
    if swapped:
        lines = lines[::-1]
    ends = np.asarray(ann["gap_line"], float)

    def perp(e, c, d):
        rel = e - c
        return abs(rel[0] * d[1] - rel[1] * d[0])

    # pair each gap end with the line it lies nearer to; both ends on one line is an annotation error
    dA = [perp(e, *lines[0]) for e in ends]
    ia = int(np.argmin(dA))
    eA, eB = ends[ia], ends[1 - ia]
    if perp(eB, *lines[1]) > perp(eB, *lines[0]):
        raise ValueError("both gap-line ends lie nearer the same blade")
    feet = [c + ((e - c) @ d) * d for e, (c, d) in zip((eA, eB), lines)]
    out = {}
    dirs = []
    for k, (f, f_other, (_, d)) in enumerate(zip(feet, feet[::-1], lines)):
        u = d if d @ (f - f_other) > 0 else -d        # outward = away from the other blade's cutoff
        dirs.append(u)
        # largest t with f + t u inside the image
        ts = []
        for ax, hi in ((0, H - 1), (1, W - 1)):
            if u[ax] > 1e-12:
                ts.append((hi - f[ax]) / u[ax])
            elif u[ax] < -1e-12:
                ts.append((0 - f[ax]) / u[ax])
        t_max = max(min(ts), 0.0)
        out["AB"[k]] = densify(np.array([f, f + t_max * u]))
    out.update(A_tip=feet[0], B_tip=feet[1], gap_end_A=eA, gap_end_B=eB, dir_A=dirs[0], dir_B=dirs[1],
               gap_end_offsets_px=[float(perp(eA, *lines[0])), float(perp(eB, *lines[1]))],
               opening_deg=float(np.degrees(np.arccos(np.clip(dirs[0] @ dirs[1], -1, 1)))),
               cutoff_sep_px=float(np.hypot(*(feet[0] - feet[1]))), ab_swapped=bool(swapped))
    return out


AVG_SEGMENT_PX = 50.0   # length of the synthetic edge segment written for an averaged annotation (only its
                        # direction is used downstream; straight_blades extends it to the border)


def average_straight(anns: list, shape) -> dict:
    """Average several annotations of ONE fixed straight-blade rig into one annotation.

    Per member (after straight_blades: A = left blade): the drawn gap-line end on each blade and each blade's
    outward direction. Averaged: each gap end = mean of the members' ends; each direction = normalised mean of
    the members' unit directions. The result is an annotation dict whose gap line joins the mean ends and whose
    edges are short segments from those ends along the mean directions, so straight_blades(result) cuts off at
    exactly the mean ends. Members must share image shape and gap_mm (checked by the caller)."""
    if not anns:
        raise ValueError("nothing to average")
    sbs = [straight_blades(a, shape) for a in anns]
    eA = np.mean([s["gap_end_A"] for s in sbs], 0)
    eB = np.mean([s["gap_end_B"] for s in sbs], 0)
    uA = np.sum([s["dir_A"] for s in sbs], 0)
    uB = np.sum([s["dir_B"] for s in sbs], 0)
    uA, uB = uA / np.linalg.norm(uA), uB / np.linalg.norm(uB)
    gaps = np.array([np.hypot(*(s["gap_end_B"] - s["gap_end_A"])) for s in sbs])
    return dict(schema=SCHEMA, gap_mm=anns[0]["gap_mm"], image_shape=list(shape),
                electrode_A=[eA.tolist(), (eA + AVG_SEGMENT_PX * uA).tolist()],
                electrode_B=[eB.tolist(), (eB + AVG_SEGMENT_PX * uB).tolist()],
                gap_line=[eA.tolist(), eB.tolist()],
                members=dict(n=len(anns), gap_px=gaps.tolist(),
                             gap_px_mean=float(gaps.mean()),
                             gap_px_sd=float(gaps.std(ddof=1)) if len(gaps) > 1 else float("nan"),
                             gap_px_sem=float(gaps.std(ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else float("nan"),
                             opening_deg=[s["opening_deg"] for s in sbs]))


def edges_npz(ann: dict, path, shape=None) -> Path:
    """Write the electrode-edges npz that analyze_blade --electrode-edges reads (A, B, A_tip, B_tip).
    With `shape` (H, W): straight blades extended from the gap to the image border (straight_blades).
    Without: the drawn polylines as they are, tips = the gap ends."""
    if shape is not None:
        sb = straight_blades(ann, shape)
        path = Path(path)
        np.savez(path, A=sb["A"], B=sb["B"], A_tip=sb["A_tip"], B_tip=sb["B_tip"])
        return path
    geo = geometry(ann)
    A, B = densify(ann["electrode_A"]), densify(ann["electrode_B"])
    ends = np.asarray(geo["gap_ends"], float)
    # A_tip must belong to electrode A: assign the gap end nearer to A's trace
    dA0 = np.min(np.hypot(*(A - ends[0]).T))
    dB0 = np.min(np.hypot(*(B - ends[0]).T))
    a_tip, b_tip = (ends[0], ends[1]) if dA0 <= dB0 else (ends[1], ends[0])
    path = Path(path)
    np.savez(path, A=A, B=B, A_tip=a_tip, B_tip=b_tip)
    return path


def save(ann: dict, path) -> dict:
    ann = dict(ann, schema=SCHEMA)
    ann["derived"] = geometry(ann) if (ann.get("gap_line") or (ann.get("electrode_A") and ann.get("electrode_B"))) else None
    Path(path).write_text(json.dumps(ann, indent=1))
    return ann


def load(path) -> dict:
    ann = json.loads(Path(path).read_text())
    if ann.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} file")
    ann["derived"] = geometry(ann)          # always recomputed from the drawn primitives
    return ann
