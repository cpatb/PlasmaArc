"""Aggregate the BLADE-upwards batch: one row per recording, then per condition (gas x Re, over trials).

usage: aggregate_upwards.py <runs_root> <out_dir>
Every number is read from the per-file outputs; N is printed before and after every filter.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
UPWARD_JUMP_MAX_PX = 10     # V-H4': p99 of frame-to-frame tip GROWTH above this = tip jumping


def file_row(d: Path):
    m = json.loads((d / "run_metadata.json").read_text())
    r = dict(stem=d.name, gas=m["gas"], Re=m["Re"], trial=m["trial"], status=m.get("status"),
             n_frames=m.get("n_frames"), fps=m.get("fps"))
    if m.get("status") != "DONE":
        return r
    det, cal = m["detector"], m["calibration"]
    fps = m["fps"]
    pxmm = cal["px_per_mm"]
    r.update(N=det["N"], smear_S=det["smear_S"], T_low=det["T_low"], T_high=det["T_high"],
             sat_frames_of_2000=det["sat_frames"], b10_max=det["b10_max"],
             gap_px=cal["gap_px"], px_per_mm=pxmm, opening_deg=cal["full_opening_angle_deg"],
             tipL_r=cal["fits"]["L"]["tip"][0], tipL_c=cal["fits"]["L"]["tip"][1],
             tipR_r=cal["fits"]["R"]["tip"][0], tipR_c=cal["fits"]["R"]["tip"][1],
             clean_arches=cal["clean_arches"])
    df = pd.read_csv(d / "measurements.csv", usecols=["frame", "arc_detected", "arc_complete", "area_px",
                                                        "arc_length_px", "arc_width_px", "arc_tip_row"])
    assert len(df) == m["n_frames"], (d, len(df))
    comp = df["arc_complete"] == 1
    r.update(frac_complete=float(comp.mean()))
    c = df[comp]
    # arc_tip_row ('up') = H-1-row, i.e. distance from the BOTTOM EDGE; the gap sits at the mean
    # calibrated blade-tip row, so height above the gap = arc_tip_row - (H-1-gap_row)
    gap_from_bottom = (m["H"] - 1) - 0.5 * (cal["fits"]["L"]["tip"][0] + cal["fits"]["R"]["tip"][0])
    h = c["arc_tip_row"] - gap_from_bottom
    r.update(arc_len_mm_med=float(np.median(c["arc_length_px"]) / pxmm) if len(c) else np.nan,
             eff_width_px_med=float(np.median(c["area_px"] / c["arc_length_px"].clip(lower=1))) if len(c) else np.nan,
             tip_height_above_gap_mm_med=float(np.median(h) / pxmm) if len(c) else np.nan,
             tip_height_above_gap_mm_p90=float(np.percentile(h, 90) / pxmm) if len(c) else np.nan)
    # V-H4': frame-to-frame tip GROWTH (arc_tip_row is distance from the bottom edge for 'up')
    tip = df["arc_tip_row"].where(comp).to_numpy(float)
    dt = np.diff(tip)
    grow = dt[np.isfinite(dt) & (dt > 0)]
    r.update(tip_growth_p99_px=float(np.percentile(grow, 99)) if grow.size else np.nan,
             VH4p_pass=bool(grow.size and np.percentile(grow, 99) < UPWARD_JUMP_MAX_PX))
    ev_p = d / "event_summary.csv"
    if ev_p.exists():
        ev = pd.read_csv(ev_p)
        dur_s = ev["duration_frames"] / fps
        r.update(n_events=int(len(ev)), event_rate_hz=len(ev) / (m["n_frames"] / fps),
                 event_dur_ms_med=float(np.median(dur_s) * 1e3) if len(ev) else np.nan,
                 event_vel_ms_med=float(np.nanmedian(ev["vel_ms"])) if len(ev) else np.nan,
                 local_vel_ms_med=float(np.nanmedian(ev["local_vel_median"])) if "local_vel_median" in ev else np.nan,
                 extension_mm_med=float(np.nanmedian(ev["extension_px"]) / pxmm) if len(ev) else np.nan)
    rs_p = d / "re_strikes_qa.csv"
    if rs_p.exists():
        rs = pd.read_csv(rs_p)
        cyc = rs["cycle_frames"][rs["cycle_frames"] > 0] / fps * 1e3
        r.update(n_restrikes=int(len(rs)), frac_modeB=float((rs["mode"] == "B").mean()) if len(rs) else np.nan,
                 cycle_ms_mean=float(cyc.mean()) if len(cyc) else np.nan,
                 cycle_ms_med=float(cyc.median()) if len(cyc) else np.nan)
    return r


dirs = sorted({p.parent for p in ROOT.rglob("run_metadata.json")})
rows = [file_row(d) for d in dirs]
F = pd.DataFrame(rows).sort_values(["gas", "Re", "trial"])
F.to_csv(OUT / "per_recording.csv", index=False)
print(f"recordings found: {len(F)}; DONE: {(F['status'] == 'DONE').sum()}")
D = F[F["status"] == "DONE"].copy()
print(f"after status==DONE filter: {len(D)}")

metrics = ["frac_complete", "arc_len_mm_med", "tip_height_above_gap_mm_med", "tip_height_above_gap_mm_p90", "event_rate_hz", "event_dur_ms_med",
           "event_vel_ms_med", "local_vel_ms_med", "n_restrikes", "frac_modeB", "cycle_ms_mean",
           "px_per_mm", "opening_deg"]
metrics = [k for k in metrics if k in D]
g = D.groupby(["gas", "Re"])
C = g[metrics].agg(["mean", "std", "count"])
C.columns = [f"{a}_{b}" for a, b in C.columns]
C.to_csv(OUT / "per_condition.csv")
print(C.round(3).to_string())
