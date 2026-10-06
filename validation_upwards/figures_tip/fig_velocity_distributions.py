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
YMAX = 1.6    # s/m: shows the moving-frame body; the 0 bin (stalled frames) can exceed it, listed in the caption


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
    fig, axs = plt.subplots(len(C.GASES), 1, figsize=(C.SINGLE_W, 8.6), sharex=True, sharey=True)
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
            ax.plot(ctr, h, color=C.re_color(Re), label=f"{Re}", **C.outlined(1.2))
            rows.append(pd.DataFrame({"gas": gas, "Re": Re, "v_ms": ctr, "density": h, "n_frames": len(v),
                                      "n_outside": outside, "zero_share": float((v == 0).mean())}))
        C.tag(ax, C.LABEL[gas])
        C.panel(ax, "abcde"[C.GASES.index(gas)])
        ax.set_xlim(-BIN / 2, VMAX)
    axs[-1].set_xlabel("Local Tip Velocity (m/s)")
    axs[len(axs) // 2].set_ylabel("Probability Density (s/m)")
    for ax in axs:
        ax.set_ylim(0, YMAX)            # one explicit limit for all shared panels (lessons 2026-08-30)
    R = pd.concat(rows, ignore_index=True)
    top = R[(R["v_ms"].abs() < BIN / 2) & (R["density"] > YMAX)]
    off_txt = "; ".join(f"{C.LABEL[g].replace('$_2$', '₂')} Re {r} ({d:.2f})" for g, r, d in
                        top[["gas", "Re", "density"]].values) or "none"
    others = R[(R["v_ms"].abs() >= BIN / 2) & (R["density"] > YMAX)]
    if len(others):
        raise SystemExit(f"{len(others)} non-zero bins exceed YMAX = {YMAX}; raise it")
    print(f"zero bins above the axis (density s/m): {off_txt}; overall max {ymax:.2f}")
    C.legend(axs[0], title="Re", loc="center right", ncol=2, columnspacing=0.8, handlelength=1.6)
    fig.tight_layout()
    C.save(fig, "fig_velocity_distributions", R,
           f"validation_upwards/figures_tip/fig_velocity_distributions.py; bin {BIN} m/s; min {MIN_FRAMES} frames; "
           "input frames_cache.parquet",
           f"""# Local tip velocity distributions

Per-frame local tip velocity (axial, advancing-front frames only), one panel per gas, one line per Re (colour, navy at low Re →
pale yellow at high Re, as in the J-electrode paper), trials pooled, {BIN} m/s bins, each normalised by all of its velocity frames (frames beyond the axis are left out of the area, not rescaled in). Conditions with fewer than {MIN_FRAMES}
velocity frames are not drawn. The first bin is centred on 0 and holds the frames where the whole-pixel tip did not
move over the ±5-point fit (a stalled or slow tip); its share is in data/fig_velocity_distributions.csv (up to ~0.5 at
the lowest Re of the CO₂ gases). The y axis stops at {YMAX} s/m; only the zero bin exceeds it, in: {off_txt}. The velocity is quantised, so
the fine structure of each curve is not physical.""")


if __name__ == "__main__":
    main()
