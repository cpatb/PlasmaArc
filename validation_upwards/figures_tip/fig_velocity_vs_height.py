"""Local tip velocity against tip height above the gap: one column per gas, one line per Re.

Top row: median over MOVING frames (|v| > C.ZERO_EPS). Bottom row: the share of STALLED frames (v exactly 0: the
whole-pixel tip did not move over the ±5-point fit). They are separated because the stalled share changes with
height, and folding it into one median manufactured a trend (code review 2026-09-30: Argon Re2250's apparent
0.69 -> 1.34 m/s rise came from the zero share falling 0.45 -> 0.08 with height; without zeros it rises at most ~12 %).

Per height bin: each TRIAL's statistic, then the mean over trials (line) and their min-max (band = trial scatter).
A bin is drawn only if every trial of the condition has >= MIN_FRAMES velocity frames in it. Frames above
EDGES[-1] are not drawn and are counted in the printout.

Selection caveat (caption): velocities exist only on advancing-front frames, and none are near the gap (Argon
Re8594: 1st percentile of velocity-frame heights 3.75 mm), so the lowest drawn bin is the floor of sampling, not the
gap. The profile is the velocity observed AT a height, pooled over many glides.

usage: python fig_velocity_vs_height.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as C

EDGES = np.arange(0.0, 16.01, 1.0)
MIN_FRAMES = 100
RE_SHOW = [2250, 3156, 4969, 6781, 8594]      # every other Re step (lowest available per gas included)


def per_bin(d, ntr, stat):
    t = d.groupby(["hb", "trial"], observed=True)["v_ms"].agg([stat, "size"]).reset_index()
    ok = t[t["size"] >= MIN_FRAMES].groupby("hb", observed=True).filter(lambda x: len(x) == ntr)
    return ok.groupby("hb", observed=True)[stat].agg(["mean", "min", "max"])


def main():
    C.style()
    D = pd.read_parquet(C.CACHE, columns=["gas", "Re", "trial", "h_mm", "v_ms"])
    D = D[D["v_ms"].notna()].copy()
    print(f"velocity frames {len(D)}; above {EDGES[-1]} mm (not drawn): {int((D['h_mm'] > EDGES[-1]).sum())}")
    D["hb"] = pd.cut(D["h_mm"], EDGES)
    D["stall"] = (D["v_ms"].abs() <= C.ZERO_EPS).astype(float)
    fig, axs2 = plt.subplots(len(C.GASES), 2, figsize=(C.DOUBLE_W, 8.6), sharex=True, sharey="col",
                             gridspec_kw={"width_ratios": [2, 1]})
    axs = axs2.T                       # axs[0, j] velocity, axs[1, j] stalled share, one ROW per gas
    rows, ymax = [], 0.0
    for j, gas in enumerate(C.GASES):
        for Re in RE_SHOW:
            d = D[(D["gas"] == gas) & (D["Re"] == Re)]
            if d.empty:
                continue
            ntr = d["trial"].nunique()
            col = C.re_color(Re)
            # stalled share uses every velocity frame; the median uses moving frames only
            dz = d.assign(v_ms=d["stall"])
            sz = per_bin(dz, ntr, "mean")
            mv = d[d["stall"] == 0]
            sv = per_bin(mv, mv["trial"].nunique(), "median") if mv["trial"].nunique() == ntr else pd.DataFrame()
            for ax, s, name in ((axs[0, j], sv, "v_moving_median"), (axs[1, j], sz, "stalled_share")):
                if s.empty:
                    continue
                x = np.array([b.mid for b in s.index])
                ax.fill_between(x, s["min"], s["max"], facecolor=col, alpha=0.45, edgecolor="0.55", lw=0.4)
                ax.plot(x, s["mean"], "o-", color=col, ms=3.5, mec=C.EDGE, mew=0.6,
                        label=f"{Re}" if name != "stalled_share" else None, **C.outlined(1.1))
                rows.append(pd.DataFrame({"gas": gas, "Re": Re, "quantity": name, "h_mid_mm": x,
                                          "mean_of_trials": s["mean"].values, "min_trial": s["min"].values,
                                          "max_trial": s["max"].values, "n_trials": ntr}))
                if name == "v_moving_median":
                    ymax = max(ymax, s["max"].max())
        C.tag(axs[0, j], C.LABEL[gas], loc="upper right")
        C.panel(axs[0, j], "acegi"[j])
        C.panel(axs[1, j], "bdfhj"[j])
        axs[1, j].set_xlim(0, EDGES[-1])
    axs[0, 0].set_ylim(0, 1.05 * ymax)
    axs[1, 0].set_ylim(0, 1)
    for ax in axs2[-1]:
        ax.set_xlabel("Tip Height Above Gap (mm)")
    axs2[2, 0].set_ylabel("Tip Velocity, Moving Frames (m/s)")
    axs2[2, 1].set_ylabel("Stalled Share")
    C.legend(axs[0, -1], title="Re", loc="upper center", ncol=3, columnspacing=0.8,
             handlelength=1.6)
    fig.tight_layout()
    C.save(fig, "fig_velocity_vs_height", pd.concat(rows, ignore_index=True),
           f"validation_upwards/figures_tip/fig_velocity_vs_height.py; 1 mm bins; >= {MIN_FRAMES} frames per trial "
           "per bin, all trials required; moving = |v| > 1e-9; input frames_cache.parquet",
           f"""# Tip velocity against height

Left column: median local tip velocity over moving frames against the leading tip's height above the minimum gap, one row
per gas, one line per Re (navy low Re → pale yellow high Re) (every other Re step). Right column: the share of stalled frames (the whole-pixel tip did not move
over the ±5-point fit), which is shown separately because it changes with height and would otherwise create a trend
in a median. Lines: mean over trials of each trial's statistic in 1 mm bins; bands: min–max over trials. A bin is
drawn only where every trial has ≥ {MIN_FRAMES} velocity frames. Velocities exist only on advancing-front frames and
none are near the gap, so the lowest drawn bin is where sampling starts, not the gap; this is the velocity observed
AT a height, pooled over many glides, not the history of one arc.""")


if __name__ == "__main__":
    main()
