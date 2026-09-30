"""Local tip velocity against tip height above the gap: one panel per gas, one line per Re.

Per height bin: median over each TRIAL's frames, then the mean of the trial medians (line) and their min-max (band),
so the band is trial-to-trial scatter, not frame scatter. A bin is drawn only if every trial of the condition has
>= MIN_FRAMES velocity frames in it. Frames from frames_cache.parquet (advancing-front frames only).

Selection caveat, stated in the caption: the local velocity exists only on advancing-front frames, so low heights
are sampled early in each glide and large heights late; the profile is velocity AT a height, not one arc's history.

usage: python fig_velocity_vs_height.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as C

EDGES = np.arange(0.0, 16.01, 1.0)
MIN_FRAMES = 100
RE_SHOW = [2250, 3156, 4969, 6781, 8594]      # every other Re step (lowest available per gas included)


def main():
    C.style()
    D = pd.read_parquet(C.CACHE, columns=["gas", "Re", "trial", "h_mm", "v_ms"])
    D = D[D["v_ms"].notna()]
    D["hb"] = pd.cut(D["h_mm"], EDGES)
    fig, axs = plt.subplots(1, len(C.GASES), figsize=(13, 3.0), sharey=True)
    cmap = plt.get_cmap("viridis")
    rows, ymax = [], 0.0
    for ax, gas in zip(axs, C.GASES):
        for Re in RE_SHOW:
            d = D[(D["gas"] == gas) & (D["Re"] == Re)]
            if d.empty:
                continue
            ntr = d["trial"].nunique()
            t = d.groupby(["hb", "trial"], observed=True)["v_ms"].agg(["median", "size"]).reset_index()
            ok = t[t["size"] >= MIN_FRAMES].groupby("hb", observed=True).filter(lambda x: len(x) == ntr)
            s = ok.groupby("hb", observed=True)["median"].agg(["mean", "min", "max"])
            if s.empty:
                continue
            x = np.array([b.mid for b in s.index])
            col = cmap((Re - 2250) / (8594 - 2250))
            ax.fill_between(x, s["min"], s["max"], color=col, alpha=0.25, lw=0)
            ax.plot(x, s["mean"], "o-", color=col, ms=3, lw=1.2, label=f"{Re}")
            ymax = max(ymax, s["max"].max())
            rows.append(pd.DataFrame({"gas": gas, "Re": Re, "h_mid_mm": x, "v_mean_of_trials": s["mean"].values,
                                      "v_min_trial": s["min"].values, "v_max_trial": s["max"].values, "n_trials": ntr}))
        ax.set_title(C.LABEL[gas], fontsize=10)
        ax.set_xlabel("tip height above gap (mm)")
        ax.set_xlim(0, EDGES[-1])
    for ax in axs:
        ax.set_ylim(0, 1.05 * ymax)
    axs[0].set_ylabel("local tip velocity (m/s)")
    axs[0].legend(title="Re", fontsize=7, title_fontsize=8, frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    C.save(fig, "fig_velocity_vs_height", pd.concat(rows, ignore_index=True),
           f"validation_upwards/figures_tip/fig_velocity_vs_height.py; 1 mm bins; >= {MIN_FRAMES} frames per trial "
           "per bin, all trials required; input frames_cache.parquet",
           f"""# Tip velocity against height

Local tip velocity against the leading tip's height above the minimum gap, one panel per gas, one line per Re (every
other Re step). Line: mean over trials of each trial's median in 1 mm height bins; band: min–max of the trial
medians (trial-to-trial scatter). A bin is drawn only where every trial has ≥ {MIN_FRAMES} velocity frames.
The velocity exists only on advancing-front frames, so this is the velocity observed AT a height, pooled over many
glides, not the history of one arc. The lowest bin of each line also holds the arc's first growth after a re-strike
at the gap, so read the trend from the bins above it.""")


if __name__ == "__main__":
    main()
