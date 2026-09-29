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


def edges_npz(ann: dict, path) -> Path:
    """Write the electrode-edges npz that analyze_blade --electrode-edges reads (A, B, A_tip, B_tip).
    The tips are the gap ends so the asymmetry midline is the perpendicular bisector of the gap."""
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
