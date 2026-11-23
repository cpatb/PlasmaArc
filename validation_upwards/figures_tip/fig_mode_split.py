"""Mode A/B figures for the two rules of mode_split.py (author, 2026-11-23). One PNG/PDF per figure, each with its
plotted data CSV and a caption .md, in the style of the other BLADE leading-tip figures (_common).

  ms_restrike_height        re-strike height per gas (log counts), fixed 2.4 mm cut and each accepted valley
  ms_modeB_share            Mode B share of DETECTED re-strikes vs Re: fixed cut (top), valley (bottom)
  ms_discharge_time_<rule>  mean discharge time (UPPER BOUND) after Mode A (top) and Mode B (bottom) re-strikes
  ms_glide_velocity_<rule>  median vertical glide velocity of arcs started by Mode A (top) and Mode B (bottom)
Small open grey markers: single trials. Gas markers: mean +/- sd over trials.

usage: python fig_mode_split.py <mode_split_out_dir>
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as C

RULES = [("fixed_cut_2p4mm", "fixed 2.4 mm cut"), ("valley_cut", "per-condition valley")]
CAVEAT = ("Re-strikes are the 1.8 mm leading-tip rule: those found are real (blind check 79/80), but 0.47-0.90 of "
          "re-strikes seen by eye are found where the arc glides and 1/32 in low-Re Ar-CO2. Discharge times are "
          "upper bounds; Mode B share is of detected re-strikes; Mode B is barely detectable in CO2 (see README).")


def stacked(R, cols, ylabels, legend_loc="upper left", legend_panel=0):
    fig, axs = plt.subplots(len(cols), 1, figsize=(3.4, 2.4 * len(cols)), sharex=True)
    rows = []
    for ax, col, yl in zip(axs, cols, ylabels):
        rows.append(C.vs_re_panel(ax, R, col, yl))
    top = max(np.nanmax(R[c].to_numpy(float)) for c in cols if R[c].notna().any())
    for ax in axs:                       # one explicit limit for all panels (sharey trap: the last panel wins)
        ax.set_ylim(0, 1.08 * top)
    for ax in axs[:-1]:
        ax.set_xlabel("")
    C.legend(axs[legend_panel], handles=C.gas_handles(), loc=legend_loc, ncol=2)
    fig.tight_layout()
    return fig, pd.concat(rows, ignore_index=True)


def main():
    ms = Path(sys.argv[1])
    C.OUT = ms / "figures"
    C.style()
    prov = "validation_upwards/figures_tip/fig_mode_split.py on mode_split_2026-11-23"

    # 1. heights
    E = pd.read_csv(ms / "fixed_cut_2p4mm" / "events.csv")
    Vl = pd.read_csv(ms / "valleys.csv")
    Vl = Vl[(Vl["level"] == "gas x Re") & Vl["accepted"]]
    bins = np.arange(-0.1, 18.3, 0.2)
    fig, axs = plt.subplots(len(C.GASES), 1, figsize=(3.4, 7.0), sharex=True)
    rows = []
    for ax, gas in zip(axs, C.GASES):
        h, _ = np.histogram(E.loc[E.gas == gas, "restrike_height_mm"], bins)
        ax.stairs(h, bins, fill=True, color=C.COLOR[gas], ec=C.EDGE, lw=0.6)
        v = Vl.loc[Vl.gas == gas, "valley_mm"]
        for x in v:                       # one thin grey line per accepted gas x Re valley
            ax.axvline(x, color="0.55", lw=0.6, zorder=0)
        ax.axvline(2.4, color=C.EDGE, ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(0.8, 3e3)
        C.tag(ax, f"{C.LABEL[gas]}, N = {int(h.sum())}")
        rows.append(pd.DataFrame({"gas": gas, "bin_lo_mm": bins[:-1], "bin_hi_mm": bins[1:], "count": h,
                                  "valley_min_mm": v.min() if len(v) else np.nan,
                                  "valley_max_mm": v.max() if len(v) else np.nan}))
    axs[-1].set_xlabel("Re-strike Height Above Gap (mm)")
    axs[len(axs) // 2].set_ylabel("Count")
    axs[-1].set_xlim(-0.2, 18.2)
    fig.tight_layout()
    C.save(fig, "ms_restrike_height", pd.concat(rows, ignore_index=True), prov,
           "Re-strike height above the minimum gap, all Re pooled, 0.2 mm bins, log counts. Dashed line: fixed "
           "2.4 mm cut. Thin grey lines: the accepted per-condition (gas x Re) valleys for that gas (none for "
           "Ar-CO2 and CO2, and only Re 8594 for Ar-Air). " + CAVEAT)

    # 2. Mode B share, both rules
    Rf = pd.read_csv(ms / "fixed_cut_2p4mm" / "runs.csv")
    Rv = pd.read_csv(ms / "valley_cut" / "runs.csv")
    R2 = Rf[["recording", "gas", "Re", "trial"]].copy()
    R2["fracB_fixed"] = Rf["fracB_of_detected"]
    R2["fracB_valley"] = Rv.set_index("recording").loc[Rf["recording"], "fracB_of_detected"].to_numpy()
    fig, data = stacked(R2, ["fracB_fixed", "fracB_valley"],
                        ["Mode B Share, Fixed 2.4 mm", "Mode B Share, Valley"])
    for ax in fig.axes:
        ax.set_ylim(0, 1)
    C.save(fig, "ms_modeB_share", data, prov,
           "Share of detected re-strikes that are Mode B, vs Re. Top: fixed 2.4 mm cut (every condition). Bottom: "
           "per-condition valley, only the 11 conditions with an accepted valley. " + CAVEAT)

    # 3-4. discharge time and glide velocity, per rule
    for rule, name in RULES:
        R = pd.read_csv(ms / rule / "runs.csv")
        tag = "fixed" if rule.startswith("fixed") else "valley"
        fig, data = stacked(R, ["modeA_mean_discharge_time_ms_UPPER_BOUND", "modeB_mean_discharge_time_ms_UPPER_BOUND"],
                            ["After Mode A (ms)", "After Mode B (ms)"], legend_loc="upper right", legend_panel=1)
        fig.axes[0].set_title(f"Mean discharge time, upper bound ({name})")
        C.save(fig, f"ms_discharge_time_{tag}", data, prov,
               f"Mean time from a re-strike to the next counted re-strike, after Mode A (top) and Mode B (bottom) "
               f"re-strikes, {name}. UPPER BOUNDS. Part of the A-B difference is the detector: after a Mode A "
               f"re-strike the tip must climb 1.8 mm before the next re-strike can be counted. " + CAVEAT)
        fig, data = stacked(R, ["modeA_median_glide_velocity_m_s", "modeB_median_glide_velocity_m_s"],
                            ["Mode A Arcs (m/s)", "Mode B Arcs (m/s)"])
        fig.axes[0].set_title(f"Vertical glide velocity ({name})")
        C.save(fig, f"ms_glide_velocity_{tag}", data, prov,
               f"Median over a video of the vertical rise speed of each arc (Theil-Sen slope of tip height from its "
               f"re-strike to its highest point), for arcs started by Mode A (top) and Mode B (bottom) re-strikes, "
               f"{name}. Vertical only (tip x is biased). " + CAVEAT)


if __name__ == "__main__":
    main()
