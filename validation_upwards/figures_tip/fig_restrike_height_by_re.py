"""Re-strike height distribution per gas AND per Re (trials pooled): one row per gas, one column per Re.

Companion to fig_restrike_height.py (which pools all Re); same events, same bins, same log count axis. Kept
leading-tip re-strikes only (fall >= T_TIP). Counts are NUMBERS OF COUNTED RE-STRIKES in the condition's 3 trials;
the pipeline's completeness differs with Re (restrike_windows), so the height of the bars must not be compared across
panels as a rate. The SHAPE (where re-strikes happen) is what the figure shows.

usage: python fig_restrike_height_by_re.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as C
from fig_restrike_height import BIN_MM


def main():
    C.style()
    E = pd.read_csv(C.THR / "tip_restrike_events_kept.csv")
    F = pd.read_csv(C.ROOT / "aggregate_annotated" / "per_recording.csv").set_index("stem")
    n0 = len(E)
    E = E[E["kept"]].copy()
    E["gas"], E["Re"] = E["run"].map(F["gas"]), E["run"].map(F["Re"])
    if E[["gas", "Re"]].isna().any().any():
        raise SystemExit("kept re-strikes whose run is not in per_recording.csv")
    print(f"events: {n0}, kept {len(E)}")
    res = sorted(E["Re"].unique())
    hmax = float(np.ceil(E["tip_after_mm"].max() / BIN_MM) * BIN_MM)
    bins = np.arange(0, hmax + BIN_MM / 2, BIN_MM)
    fig, axs = plt.subplots(len(C.GASES), len(res), figsize=(C.DOUBLE_W + 0.6, 7.2), sharex=True, sharey=True)
    rows = []
    for i, gas in enumerate(C.GASES):
        for j, Re in enumerate(res):
            ax = axs[i, j]
            h, _ = np.histogram(E.loc[(E["gas"] == gas) & (E["Re"] == Re), "tip_after_mm"], bins)
            if h.sum() == 0:
                ax.text(0.5, 0.5, "—", transform=ax.transAxes, ha="center", va="center", color="0.5")
            else:
                ax.bar(bins[:-1], h, width=BIN_MM, align="edge", color=C.COLOR[gas], edgecolor=C.EDGE,
                       linewidth=0.3)
                ax.text(0.95, 0.93, f"N = {int(h.sum())}", transform=ax.transAxes, ha="right", va="top",
                        fontsize=6)
            rows.append(pd.DataFrame({"gas": gas, "Re": Re, "bin_lo_mm": bins[:-1], "bin_hi_mm": bins[1:],
                                      "count": h}))
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.set_title(f"Re: {Re}", fontsize=8)
            if j == 0:
                ax.set_ylabel(C.LABEL[gas], fontsize=8)
                C.panel(ax, "abcde"[i], x=0.06, y=0.95)
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_ylim(0.8, 500)
    axs[0, 0].set_xlim(0, hmax)
    fig.supxlabel("Re-strike Height Above Gap (mm)", fontsize=10)
    fig.supylabel("Count", fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.08, hspace=0.12)
    T = pd.concat(rows, ignore_index=True)
    over = T[T["count"] > 500]
    if len(over):
        raise SystemExit(f"{len(over)} bins exceed the 500 axis limit; raise it")
    C.save(fig, "fig_restrike_height_by_re", T,
           f"validation_upwards/figures_tip/fig_restrike_height_by_re.py; bin {BIN_MM} mm; "
           "input threshold/tip_restrike_events_kept.csv",
           f"""# Re-strike height distribution by Re

Height of the new leading tip above the minimum gap at each kept re-strike (fall ≥ T_TIP), one row per gas, one
column per Re, 3 trials pooled per panel, log counts, {BIN_MM} mm bins (the same events and bins as
fig_restrike_height, which pools all Re). N = counted re-strikes in the panel; — = no recordings at that Re. Counted
re-strikes are verified real, but the share of real re-strikes the pipeline counts changes with Re, so bar heights
are not comparable between panels as rates; read where the re-strikes happen, not how many.""")


if __name__ == "__main__":
    main()
