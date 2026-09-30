"""Centroid re-strike detector for the BLADE-upwards annotated runs (author's choice, 2026-09-29).

Observable. Centroid height z_c = mean height above the gap of the 10 points equally spaced along the arc
(npoint_xy.npz; the arc's centre of mass per unit length), in mm with the run's px/mm; frames = complete arcs.

Re-strike. A breakdown replaces the arc by a lower one within one 5 us frame, while gliding moves it ~0.03 px
per frame. So a re-strike is a DROP of z_c between successive complete arcs larger than THR_MM.

Threshold, not chosen by eye. An arc cannot jump UP by breakdown, so RISES of z_c larger than the threshold
estimate the false detections at that threshold (assumes detection errors move z_c up as often as down --
UNVERIFIED). Rule: false-detection estimate rises/drops <= FDR_MAX. THR_MM = 1.5 mm is the smallest common
threshold meeting it for Air, Ar-Air, Ar-CO2 and CO2 on the first 37 runs (per-run means at 1.5 mm:
rises/drops 0.4/47.5, 1.7/42.0, 0.6/27.2, 0.9/23.4). Argon does NOT meet it (59/125 at 1.5 mm; drops and
rises are almost symmetric below 1 mm), so for Argon the estimate to use is drops - rises, which stays
~60-75 per run between 0.5 and 2 mm. Both are written for every run and gas; the FDR is printed beside each.

Outputs:
  <run>/centroid_restrikes.csv          every drop > THR_MM: frames, gap, drop, z_c before/after
  <out>/centroid_restrikes_by_run.csv   per run: n_drops, n_rises (false-detection estimate), excess,
                                        rate per s, at THR_MM and at the curve thresholds
  <out>/centroid_restrikes_by_condition.csv   mean/sd/n over trials per (gas, Re) at THR_MM

usage: python centroid_restrikes.py <runs_root> <out_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THR_MM = 1.5
FDR_MAX = 0.05
CURVE_MM = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def run_one(rd: Path, m: dict) -> dict:
    z = np.load(rd / "npoint_xy.npz")
    H = int(z["H"])
    cal = m["calibration"]
    pxmm = float(cal["px_per_mm"])
    r_gap = (H - 1) - float(cal["gap_row"])
    f = z["frames"]
    zc = (((H - 1) - z["pts"][:, :, 0].astype(float)).mean(1) - r_gap) / pxmm      # mm above the gap
    dz = np.diff(zc)
    k = np.flatnonzero(dz < -THR_MM)
    pd.DataFrame(dict(frame_before=f[k], frame_after=f[k + 1], gap_frames=np.diff(f)[k], drop_mm=-dz[k],
                      zc_before_mm=zc[k], zc_after_mm=zc[k + 1])).to_csv(rd / "centroid_restrikes.csv",
                                                                         index=False)
    dur = m["n_frames"] / m["fps"]
    row = dict(run=rd.name, gas=m["gas"], Re=m["Re"], trial=m["trial"], complete_frames=len(f), duration_s=dur)
    for t in CURVE_MM:
        nd, nr = int((dz < -t).sum()), int((dz > t).sum())
        row.update({f"drops_{t}": nd, f"rises_{t}": nr, f"excess_{t}": nd - nr})
    row.update(n_restrikes=row[f"drops_{THR_MM}"], n_false_est=row[f"rises_{THR_MM}"],
               excess=row[f"excess_{THR_MM}"], rate_per_s=row[f"drops_{THR_MM}"] / dur,
               excess_rate_per_s=row[f"excess_{THR_MM}"] / dur)
    return row


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows, n_skip = [], 0
    for npz in sorted(root.rglob("npoint_xy.npz")):
        rd = npz.parent
        mp = rd / "run_metadata.json"
        if not mp.exists():                           # per-window copies under chunks/
            continue
        m = json.loads(mp.read_text())
        if m.get("status") != "DONE":
            n_skip += 1
            continue
        rows.append(run_one(rd, m))
    R = pd.DataFrame(rows).sort_values(["gas", "Re", "trial"])
    R.to_csv(out / "centroid_restrikes_by_run.csv", index=False)
    print(f"runs: {len(R)} DONE ({n_skip} not DONE skipped)")
    print(f"\nfalse-detection estimate (sum rises / sum drops) at {THR_MM} mm, per gas (rule: <= {FDR_MAX}):")
    for g, grp in R.groupby("gas"):
        fdr = grp["n_false_est"].sum() / max(grp["n_restrikes"].sum(), 1)
        print(f"  {g:14s} runs {len(grp):3d}  FDR {fdr:.3f}  {'meets' if fdr <= FDR_MAX else 'FAILS -> use excess'}")
    C = R.groupby(["gas", "Re"]).agg(n=("run", "size"), restrikes_mean=("n_restrikes", "mean"),
                                     restrikes_sd=("n_restrikes", "std"), false_est_mean=("n_false_est", "mean"),
                                     excess_mean=("excess", "mean"), excess_sd=("excess", "std"),
                                     rate_per_s_mean=("rate_per_s", "mean"),
                                     excess_rate_per_s_mean=("excess_rate_per_s", "mean")).reset_index()
    C.to_csv(out / "centroid_restrikes_by_condition.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print("\n" + C.round(2).to_string(index=False))
    print(f"\nwrote {out / 'centroid_restrikes_by_run.csv'} and {out / 'centroid_restrikes_by_condition.csv'}")


if __name__ == "__main__":
    main()
