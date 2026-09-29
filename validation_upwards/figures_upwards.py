"""Figures for the BLADE-upwards campaign from aggregate_upwards.py outputs.

usage: figures_upwards.py <agg_dir>
Writes <agg_dir>/fig_metrics_vs_Re.png/.pdf and fig_calibration.png/.pdf, plus the plotted data CSVs.
Minimal text on the axes; the explanation lives in REPORT.md. Linear axes start at 0.
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AGG = Path(sys.argv[1])
F = pd.read_csv(AGG / "per_recording.csv")
D = F[(F["status"] == "DONE") & (F["tag"].fillna("") == "") & (F["short_complete_frac"].fillna(1.0) < 0.001) & F["physical_gap_anchor"].fillna(False)]
GASES = sorted(F["gas"].unique())
COL = dict(zip(GASES, plt.get_cmap("tab10").colors))

panels = [("event_vel_ms_med", "event velocity (m/s)"), ("local_vel_ms_med", "local velocity (m/s)"),
          ("cycle_ms_mean", "re-strike cycle (ms)"), ("frac_modeB", "Mode B fraction"),
          ("frac_complete", "complete-arc fraction"), ("tip_height_above_gap_mm_p90", "tip height p90 (mm)")]
fig, axs = plt.subplots(2, 3, figsize=(12, 7))
plotted = []
for ax, (k, lab) in zip(axs.ravel(), panels):
    for g in GASES:
        d = D[D["gas"] == g]
        if d.empty or k not in d:
            continue
        ax.plot(d["Re"], d[k], "o", ms=3, alpha=0.35, color=COL[g])
        m = d.groupby("Re")[k].mean()
        ax.plot(m.index, m.values, "-o", ms=4, color=COL[g], label=g)
        for re_, v in m.items():
            plotted.append(dict(metric=k, gas=g, Re=re_, mean=v, n=int((d["Re"] == re_).sum())))
    ax.set_xlabel("Re")
    ax.set_ylabel(lab)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
axs[0, 0].legend(fontsize=7, frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(AGG / f"fig_metrics_vs_Re.{ext}", dpi=150)
pd.DataFrame(plotted).to_csv(AGG / "fig_metrics_vs_Re_data.csv", index=False)

C = F[F["gap_px"].notna()] if "gap_px" in F else F.iloc[0:0]
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for g in GASES:
    d = C[C["gas"] == g]
    axs[0].plot(d["Re"], d["gap_px"], "o", ms=4, color=COL[g], label=g)
    axs[1].plot(d["Re"], d["opening_deg"], "o", ms=4, color=COL[g])
axs[0].set_xlabel("Re"); axs[0].set_ylabel("gap (px)"); axs[0].set_xlim(0, None); axs[0].set_ylim(0, None)
axs[1].set_xlabel("Re"); axs[1].set_ylabel("opening angle (deg)"); axs[1].set_xlim(0, None); axs[1].set_ylim(0, None)
axs[1].axhline(90, color="k", lw=0.8, ls="--")
axs[0].legend(fontsize=7, frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(AGG / f"fig_calibration.{ext}", dpi=150)
C[["stem", "gas", "Re", "trial", "gap_px", "opening_deg"]].to_csv(AGG / "fig_calibration_data.csv", index=False)
print("figures written to", AGG)
