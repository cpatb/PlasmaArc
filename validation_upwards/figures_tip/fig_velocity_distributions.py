"""Distributions of the per-frame local tip velocity, one panel per gas, one line per Re (trials pooled).

Per-frame local velocity (event_velocity.py, axial component, advancing-front frames only) from frames_cache.parquet.
Histograms on common 0.1 m/s bins, normalised to unit area; a line is drawn only for conditions with >= MIN_FRAMES
velocity frames.

usage: python fig_velocity_distributions.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as C

BIN, VMAX, MIN_FRAMES = 0.1, 10.0, 500


def main():
    C.style()
    D = pd.read_parquet(C.CACHE, columns=["gas", "Re", "trial", "v_ms"])
    n0 = len(D)
    D = D[D["v_ms"].notna()]
    print(f"frames {n0}, with a local velocity {len(D)}; above {VMAX} m/s (outside the axis): "
          f"{int((D['v_ms'] > VMAX).sum())}")
    # first bin centred on 0 so stalled frames (exactly 0, stored as +/-1e-13) are counted, not dropped (review)
    D.loc[D["v_ms"].abs() <= C.ZERO_EPS, "v_ms"] = 0.0
    bins = np.arange(-BIN / 2, VMAX + BIN, BIN)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    fig, axs = plt.subplots(1, len(C.GASES), figsize=(13, 2.9), sharey=True)
    cmap = plt.get_cmap("viridis")
    res = sorted(D["Re"].unique())
    rows, ymax = [], 0.0
    for ax, gas in zip(axs, C.GASES):
        for Re in res:
            v = D.loc[(D["gas"] == gas) & (D["Re"] == Re), "v_ms"].to_numpy()
            if len(v) < MIN_FRAMES:
                continue
            cnt, _ = np.histogram(v, bins)
            h = cnt / (len(v) * BIN)             # normalised by ALL frames, so out-of-range frames are not rescaled in
            outside = len(v) - int(cnt.sum())
            ymax = max(ymax, h.max())
            ax.plot(ctr, h, color=cmap((Re - res[0]) / (res[-1] - res[0])), lw=1.1, label=f"{Re}")
            rows.append(pd.DataFrame({"gas": gas, "Re": Re, "v_ms": ctr, "density": h, "n_frames": len(v),
                                      "n_outside": outside, "zero_share": float((v == 0).mean())}))
        ax.set_title(C.LABEL[gas], fontsize=10)
        ax.set_xlabel("local tip velocity (m/s)")
        ax.set_xlim(-BIN / 2, VMAX)
    axs[0].set_ylabel("probability density (s/m)")
    for ax in axs:
        ax.set_ylim(0, 1.05 * ymax)     # one explicit limit for all shared panels (lessons 2026-08-30)
    axs[-1].legend(title="Re", fontsize=7, title_fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    C.save(fig, "fig_velocity_distributions", pd.concat(rows, ignore_index=True),
           f"validation_upwards/figures_tip/fig_velocity_distributions.py; bin {BIN} m/s; min {MIN_FRAMES} frames; "
           "input frames_cache.parquet",
           f"""# Local tip velocity distributions

Per-frame local tip velocity (axial, advancing-front frames only), one panel per gas, one line per Re (colour, low →
high Re), trials pooled, {BIN} m/s bins, each normalised by all of its velocity frames (frames beyond the axis are left out of the area, not rescaled in). Conditions with fewer than {MIN_FRAMES}
velocity frames are not drawn. The first bin is centred on 0 and holds the frames where the whole-pixel tip did not
move over the 5-frame fit (a stalled or slow tip); its share is in data/fig_velocity_distributions.csv (up to ~0.5 at
the lowest Re of the CO₂ gases). The velocity is quantised, so the fine structure of each curve is not physical.""")


if __name__ == "__main__":
    main()
