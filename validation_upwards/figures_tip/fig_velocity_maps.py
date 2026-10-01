"""Velocity maps: where on the arc, and how fast, the discharge moves. One row per gas, one column per Re.

Per recording, build_velocity_map.py gives vmap[pixel] = mean over advancing-front frames of that frame's local tip
velocity, over the frames in which the arc covered the pixel (velocity_map_local.npy, mm/s), and the frame count per
pixel (velocity_map_count.npy). This script:
- maps each pixel to millimetres relative to the recording's own hand-annotated gap centre (x across, y = height
  above the gap), with the recording's own px/mm. Trials of one condition can sit at different pixel positions
  (Argon's image moved between set-ups), so they are combined in mm, never in pixels;
- bins pixels onto a common BIN_MM grid by scatter-adding sum(v * count) and sum(count) (exact count weighting, no
  interpolation), over all trials of the condition;
- masks bins with fewer than MIN_COUNT frames (the rim where the arc rarely reached).
The MEAN is what the quantised 5-frame velocity needs (it includes stalled frames as 0). The velocity at a pixel is
the TIP velocity of the frames in which the arc covered it, not a local speed of that piece of arc (the J-suite
definition).

Colour: one scale per gas (row), shared across its Re, set at the 97th percentile of that row's unmasked bins, so the
within-gas Re trend is visible; rows are NOT comparable by colour, the colour bar per row gives the numbers.

usage: python fig_velocity_maps.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as C

BIN_MM = 0.2
X_RANGE, Y_RANGE = (-12.0, 12.0), (-3.0, 26.0)   # the frame reaches ~25 mm above the gap; Argon fills it
MIN_COUNT = 20
PCT = 97


def run_bins(rd):
    """(sum v*count, sum count) on the common grid for one recording, or None if its map is missing."""
    vp, cp = rd / "velocity_map_local.npy", rd / "velocity_map_count.npy"
    if not (vp.exists() and cp.exists()):
        return None
    m = json.loads((rd / "run_metadata.json").read_text(encoding="utf-8"))
    cal = m["calibration"]
    pxmm = float(cal["px_per_mm"])
    (ra, ca), (rb, cb) = cal["gap_ends"]
    r0, c0 = 0.5 * (ra + rb), 0.5 * (ca + cb)                 # gap centre, image (row, col)
    v = np.load(vp) / 1000.0                                   # m/s
    n = np.load(cp)
    if v.shape != n.shape:
        raise SystemExit(f"{rd.name}: map {v.shape} vs count {n.shape}")
    rr, cc = np.nonzero(n > 0)
    x = (cc - c0) / pxmm
    y = (r0 - rr) / pxmm                                       # image rows grow downward; height grows upward
    ex = np.arange(X_RANGE[0], X_RANGE[1] + BIN_MM / 2, BIN_MM)
    ey = np.arange(Y_RANGE[0], Y_RANGE[1] + BIN_MM / 2, BIN_MM)
    w = n[rr, cc]
    s, _, _ = np.histogram2d(y, x, [ey, ex], weights=v[rr, cc] * w)
    c, _, _ = np.histogram2d(y, x, [ey, ex], weights=w)
    return s, c, ex, ey


def main():
    C.style()
    D = C.per_run()
    res = sorted(D["Re"].unique())
    fig, axs = plt.subplots(len(C.GASES), len(res), figsize=(2.0 * len(res), 2.1 * len(C.GASES)),
                            sharex=True, sharey=True, squeeze=False)
    rows, missing = [], []
    for i, gas in enumerate(C.GASES):
        grids = {}
        for Re in res:
            runs = D[(D["gas"] == gas) & (D["Re"] == Re)]
            S = Cn = None
            nt = 0
            for stem in runs["stem"]:
                rd = next((C.ROOT / "runs_annotated").rglob(f"{stem}/run_metadata.json")).parent
                b = run_bins(rd)
                if b is None:
                    missing.append(stem)
                    continue
                s, c, ex, ey = b
                S, Cn = (s, c) if S is None else (S + s, Cn + c)
                nt += 1
            if S is not None:
                V = np.where(Cn >= MIN_COUNT, S / np.maximum(Cn, 1), np.nan)
                grids[Re] = (V, ex, ey, nt, int(len(runs)))
        vals = np.concatenate([g[0][np.isfinite(g[0])] for g in grids.values()]) if grids else np.array([])
        vmax = float(np.percentile(vals, PCT)) if vals.size else 1.0
        for j, Re in enumerate(res):
            ax = axs[i, j]
            ax.set_facecolor("0.93")
            if Re not in grids:
                ax.text(0.5, 0.5, "—", transform=ax.transAxes, ha="center", va="center", color="0.5")
            else:
                V, ex, ey, nt, ntot = grids[Re]
                im = ax.imshow(V, origin="lower", extent=[ex[0], ex[-1], ey[0], ey[-1]], cmap="plasma", vmin=0,
                               vmax=vmax, interpolation="nearest")
                ok = np.isfinite(V)
                rows.append(dict(gas=gas, Re=Re, trials_used=nt, trials=ntot, bins=int(ok.sum()),
                                 v_mean_over_bins=float(np.nanmean(V)) if ok.any() else np.nan, row_vmax=vmax))
                if nt < ntot:
                    ax.text(0.03, 0.03, f"{nt}/{ntot} trials", transform=ax.transAxes, fontsize=6)
            ax.axhline(0, color="0.5", lw=0.5, ls=":")
            if i == 0:
                ax.set_title(f"Re {Re}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{C.LABEL[gas]}\nheight (mm)", fontsize=8)
            if i == len(C.GASES) - 1:
                ax.set_xlabel("x (mm)", fontsize=8)
        if grids:
            cb = fig.colorbar(im, ax=axs[i, :].tolist(), fraction=0.012, pad=0.01)
            cb.set_label("m/s", fontsize=7)
            cb.ax.tick_params(labelsize=6)
    axs[0, 0].set_xlim(*X_RANGE), axs[0, 0].set_ylim(*Y_RANGE)
    for ax in axs.ravel():
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
    print(f"maps missing for {len(missing)} recordings" + (f": {missing}" if missing else ""))
    C.save(fig, "fig_velocity_maps", pd.DataFrame(rows),
           f"validation_upwards/figures_tip/fig_velocity_maps.py; {BIN_MM} mm bins; min {MIN_COUNT} frames per bin; "
           f"colour per gas at p{PCT}; inputs runs_annotated/*/velocity_map_local.npy + velocity_map_count.npy",
           f"""# Velocity maps

Mean leading-tip velocity over the frames in which the arc covered each point, one row per gas, one column per Re,
trials pooled (weighted by frame count) on a common {BIN_MM} mm grid centred on each recording's own hand-annotated
gap (dotted line: the gap height). The value at a point is the TIP velocity of the frames in which the arc covered it
(the J-suite definition), not the local speed of that piece of arc. The mean includes stalled frames as zero, which is
what the pixel-quantised 5-frame velocity needs. Grey: fewer than {MIN_COUNT} frames. **Colour scale per row** (0 to
the row's 97th percentile, colour bar on the right): compare Re within a gas by colour; compare gases by the colour-bar
numbers, not by colour. At the lowest Re of the CO₂ gases the underlying velocities are not resolved (fig_tip_kinematics).""")


if __name__ == "__main__":
    main()
