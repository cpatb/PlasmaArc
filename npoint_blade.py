#!/usr/bin/env python3
"""
npoint_blade.py -- the 10-segment (whole-arc, N-point) analysis for ONE BLADE-upwards run directory.

Inputs (all written by run_blade_upwards for the run):
  npoint_xy.npz                  N points per detected frame, RAW (row, col), from analyze_blade's own skeleton
                                 (--npoint-n; merged from the chunks with global frame numbers)
  measurements_all_events.csv    frame, arc_detected, is_flash, arc_tip_row, event_id, local_vel_ms
  re_strikes_qa.csv              the TIP detector's re-strikes (qa_restrike), for the tip-invisible flag
  run_metadata.json              H, fps, calibration (px_per_mm, gap_row)

Coordinates. The upward rig has the electrodes at the BOTTOM of the frame. Every quantity here uses the
convention arc_tip_row already uses for arc_direction 'up': r = H-1-row, so r INCREASES away from the
electrodes, exactly as a downward-arc ('down') recording's raw row does. The J-electrode 10-segment code
(npoint_restrike, segment_velocity) was written in that convention, so it is reused unchanged:
  depth_i = r_i - r_gap,  r_gap = H-1-gap_row;  point 0 = the foot on blade A (see ORIENTATION).

Outputs (in the run directory):
  npoint_restrikes.csv      npoint_restrike.detect_events on the N-point depths (initial Mode A/B cut
                            MODE_B_FRAC x L_eff, as registered for the J suite) + tip_invisible flag
  segment_velocities.csv    per-segment local velocity (segment_velocity.compute_arrays)
  npoint_summary.json       counts, the two anchors below, and the dead-zone valley DIAGNOSTIC

Anchors that do not pass through this module:
  * reconstruction: deepest point vs the CSV's arc_tip_row on the same frames (a skeleton ends ~half an arc
    width inside the outermost lit pixel, so a few px is expected, tens is a bug);
  * velocity: the deepest segment's axial velocity vs the same-method tip velocity (segment_velocity's gate).

The J suite's valley re-classification (valley_reclassify.find_valley) has J-rig mm constants; here it is only
REPORTED (found / not found, where), never applied -- the Mode A/B split on this rig is not established.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from npoint_restrike import detect_events, DEDUP_FRAMES, MODE_B_FRAC
from segment_velocity import compute_arrays
from valley_reclassify import find_valley

N_DEFAULT = 10


def load(rd: Path):
    meta = json.loads((rd / "run_metadata.json").read_text())
    z = np.load(rd / "npoint_xy.npz")
    return meta, z


def run(rd: Path, n: int = N_DEFAULT) -> dict:
    meta, z = load(rd)
    if int(z["n"]) != n:
        raise ValueError(f"npoint_xy.npz has N={int(z['n'])}, asked for {n}")
    H = int(meta["H"])
    if int(z["H"]) != H:
        raise ValueError(f"npoint_xy.npz H={int(z['H'])} != run H={H}")
    cal = meta["calibration"]
    pxmm, fps = float(cal["px_per_mm"]), float(meta["fps"])
    r_gap = (H - 1) - float(cal["gap_row"])

    frames_all = z["frames"].astype(np.int64)
    raw = z["pts"].astype(np.float64)                        # (M, n, 2) raw (row, col)
    pts = raw.copy()
    pts[:, :, 0] = (H - 1) - raw[:, :, 0]                    # r = H-1-row: increases away from the electrodes

    # the frame set the tip detector uses: detected & not flash (the CSV's is_flash, as everywhere in BLADE)
    meas = pd.read_csv(rd / "measurements_all_events.csv",
                       usecols=["frame", "arc_detected", "is_flash", "arc_tip_row", "event_id", "local_vel_ms"])
    sel = meas[(meas["arc_detected"] == 1) & (meas["is_flash"] == 0)].dropna(subset=["arc_tip_row"])
    keep = np.isin(frames_all, sel["frame"].to_numpy())
    n_in, n_keep = len(frames_all), int(keep.sum())
    frames, pts = frames_all[keep], pts[keep]
    order = np.argsort(frames, kind="stable")
    frames, pts = frames[order], pts[order]
    # ORIENTATION. analyze_blade resamples along the skeleton route FROM the foot on electrode A TO the foot on
    # B, so point 0 is on blade A (the LEFT blade, rig_annotation.straight_blades) by construction. (The J
    # suite's orient_arc -- point 0 = the end with smaller r -- is not usable here: a BLADE arch's two feet sit
    # at nearly the same height, median |r_0 - r_N| 4 px vs |c_0 - c_N| 41 px on Air Re2250 t1, and it chose
    # the left foot in only 42 % of frames.) Checked, not assumed: point 0 left of point N-1 on every frame.
    n_right = int((pts[:, 0, 1] > pts[:, -1, 1]).sum()) if len(pts) else 0
    if n_right:
        raise ValueError(f"{n_right} frames have point 0 to the RIGHT of point N-1: not A-to-B routed")
    depths = pts[:, :, 0] - r_gap                            # (M, n) px, + away from the gap

    # anchor 1: deepest point vs the pipeline's arc_tip_row (same frames)
    tip = sel.set_index("frame")["arc_tip_row"].reindex(frames).to_numpy(float)
    d_tip = pts[:, :, 0].max(1) - tip
    recon = dict(n_frames=len(frames), median_signed_px=float(np.median(d_tip)) if len(d_tip) else None,
                 p90_abs_px=float(np.percentile(np.abs(d_tip), 90)) if len(d_tip) else None)

    # detector (J suite, unchanged) + tip-invisible flag
    res = detect_events(frames, depths)
    events = res["events"]
    rs = rd / "re_strikes_qa.csv"
    tipf = pd.read_csv(rs)["frame"].to_numpy(np.int64) if rs.exists() else np.zeros(0, np.int64)
    for e in events:
        near = np.min(np.abs(tipf - e["frame"])) if len(tipf) else 1e9
        e["tip_invisible"] = bool(near > DEDUP_FRAMES)
        e["breakpoint_depth_mm"] = round(e["breakpoint_depth_px"] / pxmm, 3)
        e["deepest_depth_mm"] = round(e["deepest_depth_px"] / pxmm, 3)
    cols = ["file", "frame", "mode", "kind", "s", "point_idx", "is_endpoint", "breakpoint_depth_mm",
            "deepest_depth_mm", "strength_px", "cycle_frames", "tip_invisible"]
    stem = rd.name
    ev = pd.DataFrame([dict(file=stem, **{k: e[k] for k in cols if k != "file"}) for e in events], columns=cols)
    ev.to_csv(rd / "npoint_restrikes.csv", index=False)

    # dead-zone valley: DIAGNOSTIC only (J-rig mm constants, not applied)
    valley = find_valley(ev["deepest_depth_mm"].to_numpy(float)) if len(ev) else None

    # per-segment velocities (segment_velocity core, unchanged)
    sv, sv_info = compute_arrays(stem, frames, pts, meas[["frame", "event_id", "arc_tip_row", "local_vel_ms"]],
                                 fps, pxmm, n=n)
    sv.to_csv(rd / "segment_velocities.csv", index=False)

    n_ev = len(ev)
    summ = dict(file=stem, N=n, frames_with_points=n_in, frames_detected_noflash_with_points=n_keep,
                n_events=n_ev, frac_B_initial_cut=(float((ev["mode"] == "B").mean()) if n_ev else None),
                mode_cut=f"deepest depth > {MODE_B_FRAC} x L_eff (L_eff = p90 deepest depth = {res['L_eff']:.2f} px)",
                tip_invisible_frac=(float(ev["tip_invisible"].mean()) if n_ev else None),
                anchor_reconstruction=recon,
                anchor_velocity=dict(deepest_seg_vs_tip_median_abs_ms=sv_info.get("gate_median_abs_diff"),
                                     corr=sv_info.get("gate_corr"), n_rows=sv_info.get("n_rows")),
                valley_diagnostic_J_constants=valley)
    (rd / "npoint_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    a = ap.parse_args()
    s = run(Path(a.run_dir), a.n)
    print(json.dumps(s, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
