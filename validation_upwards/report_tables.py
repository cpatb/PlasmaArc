"""Markdown tables for REPORT.md, printed from the aggregate / sensitivity CSVs (no hand-typed numbers).

usage: report_tables.py <agg_dir> <sensitivity_summary_csv> > tables.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AGG, SENS = Path(sys.argv[1]), Path(sys.argv[2])
F = pd.read_csv(AGG / "per_recording.csv")
D = F[(F["status"] == "DONE") & (F["tag"].fillna("") == "") & (F["short_complete_frac"].fillna(1.0) < 0.001)
      & F["physical_gap_anchor"].fillna(False).astype(bool)]


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else
                                      (f"{v:.3g}" if isinstance(v, float) else str(v)) for v in r.values) + " |")
    return "\n".join(out)


print("### Recordings\n")
st = F[F["tag"].fillna("") == ""].groupby("status").size().rename("n").reset_index()
print(md(st), "\n")
print(f"Analysed and passing every gate: **{len(D)}** recordings.\n")

print("### Regime: clean arches across the gap per scanned frame (calibration pass, 10 000 frames)\n")
R = F[F["tag"].fillna("") == ""].pivot_table(index="gas", columns="Re", values="arch_frac_of_frames", aggfunc="mean")
print(md(R.round(3).reset_index()), "\n")

print("### Headline metrics per condition (mean over trials; n = trials)\n")
keys = [("local_vel_ms_med", "local tip vel (m/s)"), ("arc_len_mm_med", "arc length (mm)"),
        ("tip_height_above_gap_mm_p90", "tip height p90 (mm)"), ("frac_complete", "complete frac"),
        ("cycle_ms_mean", "re-strike cycle (ms)"), ("n_restrikes", "re-strikes"),
        ("frac_B_thr005", "B @0.05"), ("frac_modeB", "B @0.10"), ("frac_B_thr020", "B @0.20")]
T = D.groupby(["gas", "Re"]).agg(n=("stem", "size"), **{lab: (k, "mean") for k, lab in keys}).reset_index()
print(md(T.round(3)), "\n")

print("### Trial-to-trial reproducibility (conditions with 3 trials): range over trials\n")
rep = []
for (g, re_), d in D.groupby(["gas", "Re"]):
    if len(d) < 2:
        continue
    row = {"gas": g, "Re": re_, "n": len(d)}
    for k, lab in keys:
        v = d[k].dropna()
        row[lab] = f"{v.min():.3g}-{v.max():.3g} (+-{100 * (v.max() - v.min()) / 2 / v.mean():.0f} %)" if len(v) else ""
    rep.append(row)
print(md(pd.DataFrame(rep)) if rep else "(none)", "\n")

S = pd.read_csv(SENS)
print("### Detector sensitivity: max |relative change| over arms (first 20 000 frames, calibration fixed)\n")
cols = [c for c in S.columns if c.endswith("_rel")]
M = S.groupby("stem")[cols].agg(lambda x: np.nanmax(np.abs(x))).reset_index()
M["stem"] = M["stem"].str.replace("BLADE_ADCB_", "").str.replace("_06mmNozzle_trial1", "")
M.columns = [c.replace("_rel", "") for c in M.columns]
print(md(M.round(3)), "\n")

print("### Calibration per recording\n")
C = F[F["gap_px"].notna()][["stem", "gap_px", "gap_ransac_tips_px", "opening_deg", "cloud_sep_min_px", "short_complete_frac"]].copy()
C["stem"] = C["stem"].str.replace("BLADE_ADCB_", "").str.replace("_06mmNozzle", "")
print(md(C.round(2)), "\n")
print(f"gap median {F['gap_px'].median():.2f} px over {F['gap_px'].notna().sum()} recordings; "
      f"opening angle median {F['opening_deg'].median():.1f} deg (folder: 45 deg per blade = 90 deg full)\n")
