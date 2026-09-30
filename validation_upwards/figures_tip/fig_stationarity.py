"""Within-recording stationarity: does a recording drift over its 325 ms more than repeat recordings differ?

Per recording: median of a quantity over the last third of the record divided by its median over the first third
(frames_cache.parquet). NULL, matched in sample size: the same-third medians of two different trials of the same
condition (every trial pair, first and last third). Both are summarised as |ln ratio|; a Mann-Whitney U test
compares them. (A first version compared thirds against the WHOLE-record trial scatter, which has 3x the data and
so is not a like-for-like null; it made every recording look non-stationary. Replaced 2026-09-30.)

Panels (a, b): last/first ratio per recording (open) and mean +/- sd over trials (filled) against Re, with the band
exp(+/- median |ln ratio| of the null). A recording is used only if both thirds hold >= MIN_FRAMES frames.

usage: python fig_stationarity.py
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import _common as C

MIN_FRAMES = 200
QTY = [("v_ms", "tip velocity, last/first third"), ("L_mm", "arc length, last/first third")]


def main():
    C.style()
    D = pd.read_parquet(C.CACHE, columns=["run", "gas", "Re", "trial", "t_frac", "v_ms", "L_mm"])
    D["third"] = np.where(D["t_frac"] < 1 / 3, 0, np.where(D["t_frac"] > 2 / 3, 2, 1))
    D = D[D["third"] != 1]
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.3))
    out, lines = [], []
    for ax, (col, lab) in zip(axs, QTY):
        g = D.dropna(subset=[col]).groupby(["run", "gas", "Re", "trial", "third"])[col]
        M = pd.DataFrame({"med": g.median(), "n": g.size()}).reset_index()
        M.loc[M["n"] < MIN_FRAMES, "med"] = np.nan
        P = M.pivot_table(index=["run", "gas", "Re", "trial"], columns="third", values="med").reset_index()
        P["ratio"] = P[2] / P[0]
        P.loc[~(P[0] > 0), "ratio"] = np.nan
        within = np.abs(np.log(P["ratio"].dropna().to_numpy()))
        across = []
        for _, d in P.groupby(["gas", "Re"]):
            for (_, r1), (_, r2) in itertools.combinations(d.iterrows(), 2):
                for th in (0, 2):
                    if r1[th] > 0 and r2[th] > 0:
                        across.append(abs(np.log(r1[th] / r2[th])))
        across = np.array(across)
        p = mannwhitneyu(within, across).pvalue
        m = float(np.median(across))
        lines.append(f"{col}: within-recording median |ln(last/first)| {np.median(within):.3f} (n {len(within)}); "
                     f"between trials, same third {m:.3f} (n {len(across)}); Mann-Whitney p {p:.3g}")
        ax.axhspan(np.exp(-m), np.exp(m), color="0.9", lw=0)
        ax.axhline(1, color="0.5", lw=0.8)
        out.append(C.vs_re_panel(ax, P.rename(columns={"run": "stem"}), "ratio", lab)
                   .assign(quantity=col, null_median_abs_ln=m, mwu_p=p))
        ax.set_ylim(0, 2)
    print("\n".join(lines))
    for ax, t in zip(axs, "ab"):
        ax.text(0.03, 0.97, f"({t})", transform=ax.transAxes, va="top")
    axs[0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    C.save(fig, "fig_stationarity", pd.concat(out, ignore_index=True),
           f"validation_upwards/figures_tip/fig_stationarity.py; thirds of the record; >= {MIN_FRAMES} frames per "
           "third; null = same third between trials; input frames_cache.parquet",
           f"""# Within-recording stationarity

Median of the last third of each 325 ms recording divided by the median of its first third, for the local tip
velocity (a) and the arc length (b); per recording (open) and mean ± sd over trials (filled). Grey band: the
typical difference between two REPEAT recordings of the same condition over the same third (exp(± median |ln ratio|)),
i.e. a null matched in sample size. {' '.join(lines)}. Drift within a recording is not resolved beyond the
difference between repeat recordings. A recording is used only if both thirds hold ≥ {MIN_FRAMES} frames.""")


if __name__ == "__main__":
    main()
