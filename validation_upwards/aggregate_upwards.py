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
POOLED_PX_PER_MM = 23.00 / 2.4   # OVERNIGHT_LOG "POOLED SCALE": median of 26 calibration gaps


def file_row(d: Path):
    m = json.loads((d / "run_metadata.json").read_text())
    post = json.loads((d / "POST.json").read_text()) if (d / "POST.json").exists() else {}
    r = dict(stem=d.name, gas=m["gas"], Re=m["Re"], trial=m["trial"], status=m.get("status"), tag=m.get("tag", ""),
             post_status=post.get("status"), n_frames=m.get("n_frames"), fps=m.get("fps"))
    reg = m.get("regime") or {}
    r.update(detected_frames_cal=reg.get("detected"), clean_arches_cal=reg.get("clean_arches"),
             arch_frac_of_frames=reg.get("arch_frac_of_frames"))
    if m.get("status") != "DONE":
        return r
    det, cal = m["detector"], m["calibration"]
    fps = m["fps"]
    # ONE pooled scale for every mm/m/s number. event_velocity ran at px_ev (the file's own scale, or the pooled
    # one if post-staged); its m/s outputs are exactly linear in 1/px_per_mm (verified 2026-09-29: ratios 1.0435 vs
    # 1.0435 expected, max dev 7e-4 = print rounding), so they are rescaled by px_ev / POOLED.
    pxmm = POOLED_PX_PER_MM
    px_ev = post.get("px_per_mm_used", cal["px_per_mm"])
    vscale = px_ev / POOLED_PX_PER_MM
    summ = (d / "re_strikes_qa_summary.txt").read_text(encoding="utf-8") if (d / "re_strikes_qa_summary.txt").exists() else ""
    r["physical_gap_anchor"] = "PHYSICAL gap" in summ
    r.update(N=det["N"], smear_S=det["smear_S"], T_low=det["T_low"], T_high=det["T_high"],
             sat_frames_of_2000=det["sat_frames"], b10_max=det["b10_max"],
             gap_px=cal["gap_px"], px_per_mm=pxmm, px_per_mm_file=cal["px_per_mm"], opening_deg=cal["full_opening_angle_deg"],
             gapA_r=cal["gap_ends"][0][0], gapA_c=cal["gap_ends"][0][1],
             gapB_r=cal["gap_ends"][1][0], gapB_c=cal["gap_ends"][1][1],
             gap_ransac_tips_px=cal["gap_ransac_tips_px"], cloud_sep_min_px=cal["cloud_sep_min_px"],
             clean_arches=cal["clean_arches"],
             sep_gate_pass=bool(cal["cloud_sep_min_px"] - 2 * 4 >= 10))   # review 2 (3): sep - 2 x ELEC_TOL >= ARCH_MIN_FOOT_SEP
    df = pd.read_csv(d / "measurements.csv", usecols=["frame", "arc_detected", "arc_complete", "is_flash", "area_px",
                                                        "arc_length_px", "arc_width_px", "arc_tip_row"])
    assert len(df) == m["n_frames"], (d, len(df))
    comp = (df["arc_complete"] == 1) & (df["is_flash"] == 0)    # arc_filter.valid_frames criterion
    # direct test of the root-spot defect the separation gate guards against (review 2, item 3): complete arcs
    # whose skeleton is shorter than a gap-spanning arch can be (ARCH_MIN_FOOT_SEP_PX = 10)
    r["short_complete_frac"] = float((df.loc[comp, "arc_length_px"] < 10).mean()) if comp.any() else np.nan
    r.update(frac_complete=float(comp.mean()), n_flash_complete=int(((df["arc_complete"] == 1) & (df["is_flash"] == 1)).sum()))
    c = df[comp]
    # arc_tip_row ('up') = H-1-row, i.e. distance from the BOTTOM EDGE; the gap sits at the mean
    # calibrated blade-tip row, so height above the gap = arc_tip_row - (H-1-gap_row)
    gap_from_bottom = (m["H"] - 1) - cal["gap_row"]
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
                 event_vel_ms_med=float(np.nanmedian(ev["vel_ms"]) * vscale) if len(ev) else np.nan,
                 local_vel_ms_med=float(np.nanmedian(ev["local_vel_median"]) * vscale) if "local_vel_median" in ev else np.nan,
                 extension_mm_med=float(np.nanmedian(ev["extension_px"]) / pxmm) if len(ev) else np.nan)
    rs_p = d / "re_strikes_qa.csv"
    if rs_p.exists():
        rs = pd.read_csv(rs_p)
        cyc = rs["cycle_frames"][rs["cycle_frames"] > 0] / fps * 1e3
        low = rs[rs["norm_pos"] <= 0.9]       # shallow dips AT the extension peak excluded (possible jitter)
        # A/B threshold sensitivity straight from norm_pos (re-strikes cluster just above 0.10 at the base)
        r.update(frac_B_thr005=float((rs["norm_pos"] > 0.05).mean()) if len(rs) else np.nan,
                 frac_B_thr020=float((rs["norm_pos"] > 0.20).mean()) if len(rs) else np.nan)
        r.update(frac_modeB_excl_peak=float((low["mode"] == "B").mean()) if len(low) else np.nan,
                 n_restrikes_at_peak=int((rs["norm_pos"] > 0.9).sum()))
        r.update(n_restrikes=int(len(rs)), frac_modeB=float((rs["mode"] == "B").mean()) if len(rs) else np.nan,
                 cycle_ms_mean=float(cyc.mean()) if len(cyc) else np.nan,
                 cycle_ms_med=float(cyc.median()) if len(cyc) else np.nan)
    return r


dirs = sorted({p.parent for p in ROOT.rglob("run_metadata.json")})
rows = [file_row(d) for d in dirs]
F = pd.DataFrame(rows).sort_values(["gas", "Re", "trial"])
F.to_csv(OUT / "per_recording.csv", index=False)
R = F[F["tag"] == ""].groupby(["gas", "Re"]).agg(n=("stem", "size"),
                                                n_no_gliding=("status", lambda x: int((x == "NO_GLIDING").sum())),
                                                arch_frac_mean=("arch_frac_of_frames", "mean"))
R.to_csv(OUT / "regime_by_condition.csv")
print("regime (clean arches per scanned frame; NO_GLIDING = < 50 in 10 000 frames):")
print(R.round(4).to_string())
print(f"recordings found: {len(F)}; DONE: {(F['status'] == 'DONE').sum()}")
D = F[(F["status"] == "DONE") & (F["tag"] == "")].copy()
print(f"after status==DONE and nominal-tag filter: {len(D)}")
pg = D["physical_gap_anchor"].fillna(False).astype(bool)
print("  re-strike apex NOT on the physical gap row:", D.loc[~pg, "stem"].tolist())
D = D[pg]
print(f"after physical-gap-anchor filter: {len(D)}")
sg = D["sep_gate_pass"].fillna(False).astype(bool)
print("  failing the GEOMETRIC separation bound (sep - 8 < 10 px) [kept for the record]:", D.loc[~sg, "stem"].tolist())
dg = D["short_complete_frac"].fillna(1.0) < 0.001
print("  failing the DIRECT test (>= 0.1 % of complete arcs shorter than 10 px):", D.loc[~dg, "stem"].tolist())
D = D[dg]
print(f"after direct short-arc filter: {len(D)}")

metrics = ["frac_complete", "arc_len_mm_med", "tip_height_above_gap_mm_med", "tip_height_above_gap_mm_p90", "event_rate_hz", "event_dur_ms_med",
           "event_vel_ms_med", "local_vel_ms_med", "n_restrikes", "frac_modeB", "frac_B_thr005", "frac_B_thr020", "frac_modeB_excl_peak", "cycle_ms_mean",
           "px_per_mm", "opening_deg"]
metrics = [k for k in metrics if k in D]
g = D.groupby(["gas", "Re"])
C = g[metrics].agg(["mean", "std", "count"])
C.columns = [f"{a}_{b}" for a, b in C.columns]
C.to_csv(OUT / "per_condition.csv")
print(C.round(3).to_string())
