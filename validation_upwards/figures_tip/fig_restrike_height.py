"""Distribution of the re-strike height (new tip height above the gap) of kept leading-tip re-strikes, per gas.
No Mode A/B boundary is drawn (author, 2026-10-01: show the height only). Log counts: the near-gap peak holds ~70 % of events.

usage: python fig_restrike_height.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as C

BIN_MM = 0.32   # ~3 px at the setups' 8.9-9.7 px/mm (2.85-3.11 px): close to a whole number of tip-row quanta, so
                # bins do not alternate 2 and 3 pixel levels (review 2026-09-30: 0.25 mm did)


def main():
    C.style()
    E = pd.read_csv(C.THR / "tip_restrike_events_kept.csv")
    F = pd.read_csv(C.ROOT / "aggregate_annotated" / "per_recording.csv").set_index("stem")
    n0 = len(E)
    E = E[E["kept"]].copy()
    E["gas"] = E["run"].map(F["gas"])
    print(f"events: {n0}, kept {len(E)}")
    hmax = float(np.ceil(E["tip_after_mm"].max() / BIN_MM) * BIN_MM)
    bins = np.arange(0, hmax + BIN_MM / 2, BIN_MM)
    fig, axs = plt.subplots(len(C.GASES), 1, figsize=(C.SINGLE_W, 7.4), sharex=True)
    rows = []
    for ax, gas in zip(axs, C.GASES):
        h, _ = np.histogram(E.loc[E["gas"] == gas, "tip_after_mm"], bins)
        ax.bar(bins[:-1], h, width=BIN_MM, align="edge", color=C.COLOR[gas], edgecolor=C.EDGE, linewidth=0.5)
        ax.set_yscale("log")
        ax.set_ylim(0.8, 2e3)
        C.tag(ax, f"{C.LABEL[gas]}, N = {int(h.sum())}")
        C.panel(ax, "abcde"[len(rows)])
        rows.append(pd.DataFrame({"gas": gas, "bin_lo_mm": bins[:-1], "bin_hi_mm": bins[1:], "count": h}))
    axs[-1].set_xlabel("Re-strike Height Above Gap (mm)")
    axs[len(axs) // 2].set_ylabel("Count")
    axs[-1].set_xlim(0, hmax)
    fig.tight_layout()
    C.save(fig, "fig_restrike_height", pd.concat(rows, ignore_index=True),
           f"validation_upwards/figures_tip/fig_restrike_height.py; bin {BIN_MM} mm; "
           "input threshold/tip_restrike_events_kept.csv",
           f"""# Re-strike height distribution

Height of the new leading tip above the minimum gap at each kept re-strike (fall ≥ T_TIP), all Re pooled, one panel
per gas, log counts, {BIN_MM} mm bins. Re-strikes at the gap form one peak at 0.2–0.5 mm; re-strikes partway up the
blades spread over 2–12 mm, with almost nothing between 1 and 2 mm. CO₂ and Ar–CO₂ re-strike almost only at the
gap.""")


if __name__ == "__main__":
    main()
