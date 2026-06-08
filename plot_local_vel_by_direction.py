#!/usr/bin/env python3
"""Direction-only comparison of a per-frame metric (downward vs upward).

Complements ``mode_velocity_compare.py``'s 4-group (mode x direction)
``mode_vel_local_allgroups.png`` by collapsing the mode split — the cleaner
figure when the message is purely about propagation direction.

Plotting convention matches ``mode_velocity_compare.hist_kde_groups`` so the
figures are visually consistent:
  * use only advancing-front frames with the metric > 0,
  * clip the top 1% (99th-percentile cap) instead of an overflow bin,
  * density histogram + Gaussian-KDE overlay, dashed MEAN lines (not median).

``--metric`` selects the per-frame quantity:
  * ``vel``        -> local_vel_ms  -> local_vel_by_direction.png
  * ``tortuosity`` -> tortuosity    -> local_tortuosity_by_direction.png
                      (well-formed frames only: tortuosity >= 1.0)

Optional ``--n-compare`` (vel only) recomputes the local velocity with a second
sliding-window width (default 10) via ``event_velocity.compute_local_velocities``
and writes ``local_vel_N5_vs_N10.png`` — a window-width sensitivity check.
Larger N smooths the per-frame velocity: it trims the fast tail and lowers the
mean by ~0.5-0.9 m/s, while the direction ordering (up >= down) is preserved.

Usage:
    python plot_local_vel_by_direction.py \
        --down  ".../_downward_results/measurements_all_events.csv" \
        --up    ".../_upward_results/measurements_all_events.csv" \
        --out-dir ".../mode_compare" --px-per-mm 8.8 --fps 200000 [--n-compare 10]
        [--metric tortuosity]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

DIR_COLOR = {"downward": "#26A69A", "upward": "#AB47BC"}

# metric -> (csv column, lower-bound filter, x-axis label, output stem, unit)
METRICS = {
    "vel":        ("local_vel_ms", 0.0, "Local arc velocity (m/s)",
                   "local_vel_by_direction", "m/s"),
    "tortuosity": ("tortuosity",   1.0, "Local tortuosity (path / chord)",
                   "local_tortuosity_by_direction", ""),
}


def resolve(p):
    p = Path(p)
    return p if p.suffix == ".csv" else p / "measurements_all_events.csv"


def load_stored(path, col, lo):
    v = pd.read_csv(resolve(path), usecols=[col])[col].to_numpy(float)
    return v[np.isfinite(v) & (v > lo)]


def kde_hist(ax, v, color, label, xcap, bins=25, ls="-"):
    v = v[v <= xcap]
    ax.hist(v, bins=bins, range=(0, xcap), density=True, color=color,
            alpha=0.20, edgecolor="none")
    x = np.linspace(0, xcap, 400)
    ax.plot(x, gaussian_kde(v)(x), color=color, lw=2.0, ls=ls, label=label)
    # dashed vertical line at the MEAN (not median)
    ax.axvline(np.mean(v), color=color, ls="--", lw=1.2, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--down", required=True)
    ap.add_argument("--up", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--px-per-mm", type=float, default=8.8)
    ap.add_argument("--fps", type=float, default=200000)
    ap.add_argument("--metric", choices=list(METRICS), default="vel")
    ap.add_argument("--n-compare", type=int, default=0,
                    help="vel only: if >0, also write a stored-vs-N=<value> figure")
    ap.add_argument("--stored-n", type=int, default=5,
                    help="Window width the STORED local_vel_ms column was computed "
                         "with (used only to label the sensitivity figure). Set to "
                         "10 after rerunning the pipeline with --local-vel-n 10.")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    col, lo, xlabel, stem, unit = METRICS[args.metric]
    usuf = (" " + unit) if unit else ""

    paths = {"downward": args.down, "upward": args.up}
    stored = {d: load_stored(p, col, lo) for d, p in paths.items()}

    # ---- Figure 1: direction comparison (house convention) ----
    allv = np.concatenate(list(stored.values()))
    xcap = float(np.percentile(allv, 99))
    fig, ax = plt.subplots(figsize=(8, 5))
    for d, v in stored.items():
        kde_hist(ax, v, DIR_COLOR[d], f"{d} (mean {np.mean(v):.2f}{usuf})", xcap)
    ax.plot([], [], color="gray", ls="--", lw=1.2, label="dashed = mean")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability density")
    ax.set_xlim(0, xcap)
    ax.legend(frameon=False)
    ax.set_title(f"Per-frame {args.metric} by direction  (top 1% clipped)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"{stem}.png", dpi=150)
    print(f"wrote {out / (stem + '.png')}")

    # ---- Figure 2 (optional, vel only): window-width sensitivity ----
    if args.metric == "vel" and args.n_compare and args.n_compare > 0:
        import event_velocity as ev
        ev.FPS = args.fps; ev.PX_PER_MM = args.px_per_mm
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, d in zip(axes, paths):
            df = pd.read_csv(resolve(paths[d]))
            ev.LOCAL_VEL_N = args.n_compare
            fvN, _, _ = ev.compute_local_velocities(df)
            vN = np.array(list(fvN.values()), float)
            vN = vN[np.isfinite(vN) & (vN > 0)]
            v5 = stored[d]
            xc = float(np.percentile(np.concatenate([v5, vN]), 99))
            kde_hist(ax, v5, "#1f77b4",
                     f"N={args.stored_n} stored (mean {np.mean(v5):.2f})", xc)
            kde_hist(ax, vN, "#d62728",
                     f"N={args.n_compare} (mean {np.mean(vN):.2f})", xc)
            ax.plot([], [], color="gray", ls="--", lw=1.2, label="dashed = mean")
            ax.set_title(d); ax.set_xlabel(xlabel)
            ax.legend(frameon=False)
        axes[0].set_ylabel("Probability density")
        fig.suptitle(f"Sliding-window width sensitivity: N={args.stored_n} (stored) "
                     f"vs N={args.n_compare}", fontsize=11)
        fig.tight_layout()
        fig.savefig(out / f"local_vel_N{args.stored_n}_vs_N{args.n_compare}.png", dpi=150)
        print(f"wrote {out / f'local_vel_N{args.stored_n}_vs_N{args.n_compare}.png'}")


if __name__ == "__main__":
    main()
