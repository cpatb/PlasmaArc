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
    bins = np.arange(0, VMAX + BIN / 2, BIN)
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
            h, _ = np.histogram(v, bins, density=True)
            ymax = max(ymax, h.max())
            ax.plot(ctr, h, color=cmap((Re - res[0]) / (res[-1] - res[0])), lw=1.1, label=f"{Re}")
            rows.append(pd.DataFrame({"gas": gas, "Re": Re, "v_ms": ctr, "density": h, "n_frames": len(v)}))
        ax.set_title(C.LABEL[gas], fontsize=10)
        ax.set_xlabel("local tip velocity (m/s)")
        ax.set_xlim(0, VMAX)
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
high Re), trials pooled, {BIN} m/s bins, each normalised to unit area. Conditions with fewer than {MIN_FRAMES}
velocity frames are not drawn. The distributions shift and broaden with Re in every gas; at the lowest Re of the CO₂
mixtures most velocity frames are near zero (the spot regime).""")


if __name__ == "__main__":
    main()
