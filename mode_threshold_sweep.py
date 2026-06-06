#!/usr/bin/env python3
"""
mode_threshold_sweep.py  -  Compare Mode A / Mode B classification across
four norm_pos threshold candidates (0.01, 0.05, 0.10, 0.15).

For each threshold the script calls mode_common.analyze_recording() for both
directions, pools the per-re-strike DataFrames, and produces five figures:

  thresh_normpos_distribution.png   KDE + rug of norm_pos with vertical lines
  thresh_mode_fractions.png         Mode-A fraction vs threshold (down/up/pooled)
  thresh_velocity_comparison.png    2x2 KDE+hist of advance velocity per threshold
  thresh_duration_comparison.png    2x2 KDE+hist of propagation duration per threshold
  thresh_qa_trajectories.png        2x2 scatter: norm_pos vs advance vel, coloured A/B

Usage
-----
  python mode_threshold_sweep.py \\
      --down PATH/downward_results --up PATH/upward_results \\
      --px-per-mm 8.8 --fps 200000 \\
      --out-dir OUTPUT_DIR
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, gaussian_kde

from mode_common import analyze_recording

THRESHOLDS = [0.01, 0.05, 0.10, 0.15]


# -- helpers ------------------------------------------------------------------

def resolve(p):
    """Accept either a directory (appends measurements_all_events.csv) or a CSV path."""
    p = Path(p)
    return p if p.suffix == ".csv" else p / "measurements_all_events.csv"


def mw(a, b):
    """Mann-Whitney U p-value; returns nan when either group has < 5 finite values."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    try:
        return mannwhitneyu(a, b, alternative="two-sided").pvalue
    except Exception:
        return np.nan


def ps(p):
    """Format p-value for display."""
    return "%.3g" % p if not np.isnan(p) else "n/a"


def clean_vel(v):
    """Return finite, positive values as a numpy array (no velocity cap)."""
    v = np.asarray(v, float)
    return v[np.isfinite(v) & (v > 0)]


def remove_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# -- figure 1 -----------------------------------------------------------------

def fig_normpos_distribution(all_norm_pos, out_dir):
    """KDE + rug of norm_pos with vertical threshold lines."""
    x_all = np.asarray(all_norm_pos, float)
    x_all = x_all[np.isfinite(x_all)]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")

    xs = np.linspace(max(x_all.min() - 0.05, -0.05), min(x_all.max() + 0.05, 1.1), 500)
    kde = gaussian_kde(x_all)
    ax.plot(xs, kde(xs), color="#444444", lw=2, zorder=3)
    ax.fill_between(xs, kde(xs), alpha=0.12, color="#444444")

    # Rug: small ticks at the bottom of the axes
    ax.vlines(x_all, -0.01, 0.0, color="#444444", alpha=0.25, lw=0.5, zorder=2,
              transform=ax.get_xaxis_transform())

    colors = ["#1B5E20", "#2E7D32", "#388E3C", "#43A047"]
    ymax_val = kde(xs).max()
    for i, (thr, col) in enumerate(zip(THRESHOLDS, colors)):
        ax.axvline(thr, color=col, lw=1.5, ls="--", zorder=4,
                   label="thresh=%.2f" % thr)
        ax.axvspan(0, thr, alpha=0.05 + 0.03 * i, color=col, zorder=1)
        ax.text(thr + 0.003, ymax_val * (0.90 - i * 0.10),
                "%.2f" % thr, color=col, fontsize=8, va="top")

    ax.set_xlabel("Normalised re-strike position (norm_pos)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    remove_spines(ax)
    fig.tight_layout()
    out = Path(out_dir) / "thresh_normpos_distribution.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved %s" % out)


# -- figure 2 -----------------------------------------------------------------

def fig_mode_fractions(records, out_dir):
    """Mode A fraction vs threshold, separate lines for down / up / pooled."""
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")

    def frac_a(df):
        total = len(df)
        if total == 0:
            return np.nan, 0, 0
        n_a = int((df["mode"] == "A").sum())
        return n_a / total, n_a, total

    thrs = [r[0] for r in records]
    for direction, color, marker in [("down", "#1565C0", "o"),
                                      ("up",   "#C62828", "s"),
                                      ("pooled", "#555555", "^")]:
        fracs = []
        for thr, df in records:
            sub = df if direction == "pooled" else df[df["direction"] == direction]
            f, _, _ = frac_a(sub)
            fracs.append(f)
        ax.plot(thrs, fracs, color=color, marker=marker, lw=1.8, ms=7, label=direction)

    # Annotate pooled N_A / N_total above each marker
    for thr, df in records:
        f, n_a, n_tot = frac_a(df)
        if not np.isnan(f):
            ax.text(thr, f + 0.025, "%d/%d" % (n_a, n_tot),
                    ha="center", va="bottom", fontsize=7, color="#555555")

    ax.set_xlabel("Classification threshold (norm_pos)")
    ax.set_ylabel("Fraction classified as Mode A")
    ax.set_xticks(thrs)
    ax.set_xticklabels(["%.2f" % t for t in thrs])
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9, frameon=False)
    remove_spines(ax)
    fig.tight_layout()
    out = Path(out_dir) / "thresh_mode_fractions.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved %s" % out)


# -- figure 3 -----------------------------------------------------------------

def fig_velocity_comparison(records, out_dir):
    """2x2 KDE + histogram of advance_vel_ms for Mode A vs B per threshold."""
    all_vel = []
    for _, df in records:
        all_vel.extend(clean_vel(df["advance_vel_ms"].values).tolist())
    xcap = float(np.percentile(all_vel, 99)) if all_vel else 10.0

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor="white")
    axes = axes.ravel()

    for idx, (thr, df) in enumerate(records):
        ax = axes[idx]
        va = clean_vel(df.loc[df["mode"] == "A", "advance_vel_ms"].values)
        vb = clean_vel(df.loc[df["mode"] == "B", "advance_vel_ms"].values)
        p_val = mw(va, vb)

        n_bins = 25
        for arr, color, label in [(va, "#1565C0", "A"), (vb, "#C62828", "B")]:
            if len(arr) == 0:
                continue
            med = np.mean(arr)
            leg = "%s  n=%d  med=%.2f m/s" % (label, len(arr), med)
            ax.hist(arr, bins=n_bins, range=(0, xcap), density=True,
                    color=color, alpha=0.25, edgecolor="none")
            if len(arr) >= 5:
                x = np.linspace(0, xcap, 400)
                ax.plot(x, gaussian_kde(arr)(x), color=color, lw=1.8, label=leg)
            else:
                ax.plot([], [], color=color, lw=1.8, label=leg)

        ax.set_xlim(0, xcap)
        ax.set_xlabel("Advance velocity (m/s)")
        ax.set_ylabel("Probability density")
        ax.legend(fontsize=7, frameon=False)
        remove_spines(ax)

    fig.tight_layout()
    out = Path(out_dir) / "thresh_velocity_comparison.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved %s" % out)


# -- figure 4 -----------------------------------------------------------------

def fig_duration_comparison(records, fps, out_dir):
    """2x2 KDE + histogram of propagation duration (ms) for Mode A vs B."""
    all_dur = []
    for _, df in records:
        d = df["duration_frames"].values.astype(float) / fps * 1000.0
        d = d[np.isfinite(d) & (d > 0)]
        all_dur.extend(d.tolist())
    xcap = float(np.percentile(all_dur, 99)) if all_dur else 1.0

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor="white")
    axes = axes.ravel()

    for idx, (thr, df) in enumerate(records):
        ax = axes[idx]
        va = df.loc[df["mode"] == "A", "duration_frames"].values.astype(float) / fps * 1000.0
        vb = df.loc[df["mode"] == "B", "duration_frames"].values.astype(float) / fps * 1000.0
        va_c = va[np.isfinite(va) & (va > 0)]
        vb_c = vb[np.isfinite(vb) & (vb > 0)]
        p_val = mw(va_c, vb_c)

        n_bins = 25
        for arr, color, label in [(va_c, "#1565C0", "A"), (vb_c, "#C62828", "B")]:
            if len(arr) == 0:
                continue
            med = np.mean(arr)
            leg = "%s  n=%d  med=%.2f ms" % (label, len(arr), med)
            ax.hist(arr, bins=n_bins, range=(0, xcap), density=True,
                    color=color, alpha=0.25, edgecolor="none")
            if len(arr) >= 5:
                x = np.linspace(0, xcap, 400)
                ax.plot(x, gaussian_kde(arr)(x), color=color, lw=1.8, label=leg)
            else:
                ax.plot([], [], color=color, lw=1.8, label=leg)

        ax.set_xlim(0, xcap)
        ax.set_xlabel("Propagation duration (ms)")
        ax.set_ylabel("Probability density")
        ax.legend(fontsize=7, frameon=False)
        remove_spines(ax)

    fig.tight_layout()
    out = Path(out_dir) / "thresh_duration_comparison.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved %s" % out)


# -- figure 5 -----------------------------------------------------------------

def fig_qa_trajectories(records, out_dir):
    """
    2x2 scatter: norm_pos (x) vs advance_vel_ms (y), coloured by Mode A/B.
    Each subplot corresponds to one threshold value.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), facecolor="white")
    axes = axes.ravel()

    for idx, (thr, df) in enumerate(records):
        ax = axes[idx]
        df_a = df[df["mode"] == "A"]
        df_b = df[df["mode"] == "B"]

        # Plot B first (behind), then A on top
        ax.scatter(df_b["norm_pos"], df_b["advance_vel_ms"],
                   color="#C62828", alpha=0.4, s=6, label="B",
                   zorder=2)
        ax.scatter(df_a["norm_pos"], df_a["advance_vel_ms"],
                   color="#1565C0", alpha=0.4, s=6, label="A",
                   zorder=3)
        ax.axvline(thr, color="#333333", lw=1.2, ls="--", zorder=4)

        ax.set_xlim(0, 1.0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("norm_pos")
        ax.set_ylabel("advance vel (m/s)")
        ax.legend(fontsize=7, frameon=False)
        remove_spines(ax)

    fig.tight_layout()
    out = Path(out_dir) / "thresh_qa_trajectories.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved %s" % out)


# -- main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Sweep Mode A/B classification thresholds and compare results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--down", required=True,
                    help="Path to downward_results dir (or measurements_all_events.csv)")
    ap.add_argument("--up", required=True,
                    help="Path to upward_results dir (or measurements_all_events.csv)")
    ap.add_argument("--px-per-mm", type=float, default=8.8)
    ap.add_argument("--fps", type=int, default=200_000)
    ap.add_argument("--out-dir", required=True,
                    help="Directory for output figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    down_csv = resolve(args.down)
    up_csv   = resolve(args.up)

    for p in [down_csv, up_csv]:
        if not p.exists():
            raise FileNotFoundError("Input CSV not found: %s" % p)

    print("Down CSV : %s" % down_csv)
    print("Up CSV   : %s" % up_csv)
    print("Out dir  : %s" % out_dir)
    print()

    # -- collect per-threshold DataFrames -------------------------------------
    records = []  # list of (threshold, df_combined)

    for thr in THRESHOLDS:
        print("--- threshold = %.2f ---" % thr)
        pr_down, _ = analyze_recording(
            csv_path=str(down_csv), direction="down",
            px_per_mm=args.px_per_mm, fps=args.fps,
            method="abs", threshold=thr)
        pr_up, _ = analyze_recording(
            csv_path=str(up_csv), direction="up",
            px_per_mm=args.px_per_mm, fps=args.fps,
            method="abs", threshold=thr)

        df = pd.concat([pr_down, pr_up], ignore_index=True)
        records.append((thr, df))
        n_a = int((df["mode"] == "A").sum())
        n_b = int((df["mode"] == "B").sum())
        print("  total re-strikes: %d  (A=%d  B=%d)" % (len(df), n_a, n_b))
    print()

    # norm_pos does not change with threshold; use threshold=0.10 data (index 2)
    _, df_ref = records[2]
    all_norm_pos = df_ref["norm_pos"].dropna().values

    # -- produce figures ------------------------------------------------------
    print("Producing figures ...")
    fig_normpos_distribution(all_norm_pos, out_dir)
    fig_mode_fractions(records, out_dir)
    fig_velocity_comparison(records, out_dir)
    fig_duration_comparison(records, args.fps, out_dir)
    fig_qa_trajectories(records, out_dir)
    print("\nAll figures written to %s" % out_dir)


if __name__ == "__main__":
    main()
