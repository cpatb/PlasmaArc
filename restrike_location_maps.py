#!/usr/bin/env python3
"""
restrike_location_maps.py  —  2-D heatmaps of WHERE Mode A vs Mode B re-strikes
initiate, in the style of the 2-D velocity maps.

For each recording we histogram the initiation location of every re-strike into a
spatial grid and show three panels on a shared spatial extent (mm) and shared
colour scale:  Mode A  |  Mode B  |  (B - A) difference.  This reveals whether the
two modes ignite from different regions (e.g. Mode A near the apex gap, Mode B
further out / off to one leg).

Two detectors (run both with --source both):
  * tip    — ``re_strikes_qa.csv`` gives the initiation pixel directly
             (``tip_row``, ``tip_col``); authoritative.
  * npoint — ``npoint_restrikes.csv`` gives the breakpoint ``point_idx`` per event;
             its (row,col) is looked up in the N-point reconstruction
             (``npoint_geometry``) at the event frame — this places leg-break
             Mode-B initiations that the tip detector cannot localise laterally.

House style (matches build_velocity_map): ``cmap='plasma'`` for the per-mode
density, ``cmap='RdBu_r'`` (symmetric) for the difference, ``extent`` in mm,
``origin='upper'``, ``aspect='equal'``.  A cross-recording panel compares the
calibration-free initiation DEPTH (``norm_pos``) of A vs B across all recordings.

USAGE
-----
    python restrike_location_maps.py --all --source both
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle
import plot_config
from npoint_restrike import RECORDINGS, N_HEADLINE, load_metadata, detector_params
from npoint_geometry import points_for

DATA_ROOT_DEFAULT = Path(r"C:\Users\compl\Desktop\Plasma Arc\J Electrode")
BIN_PX = 3            # spatial histogram bin (px)
SMOOTH_BINS = 1.0     # gaussian smoothing (in bins) for a heatmap look


def frame_shape(rd: Path) -> "tuple[int, int]":
    """(H, W) from an existing map or default 256x256 (these recordings are 256^2)."""
    for cand in ("velocity_map_local.npy", "velocity_map_segment.npy"):
        p = rd / cand
        if p.exists():
            a = np.load(p)
            if a.ndim == 2:
                return a.shape
    return (256, 256)


def initiation_points(stem: str, data_root: Path, source: str, n: int = N_HEADLINE):
    """Return (row, col, mode) arrays of re-strike initiation locations."""
    rd = data_root / f"{stem}_results"
    if source == "tip":
        df = pd.read_csv(rd / "re_strikes_qa.csv")
        df = df[df["mode"].isin(["A", "B"])].dropna(subset=["tip_row", "tip_col"])
        return (df["tip_row"].to_numpy(float), df["tip_col"].to_numpy(float),
                df["mode"].to_numpy())
    # npoint: breakpoint (row,col) from the reconstruction at each event frame
    df = pd.read_csv(rd / "npoint_restrikes.csv")
    df = df[df["mode"].isin(["A", "B"])].copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    df["point_idx"] = pd.to_numeric(df["point_idx"], errors="coerce")
    frames_xy, pts_xy = points_for(rd, N=n)
    idx = {int(f): i for i, f in enumerate(frames_xy)}
    rows, cols, modes = [], [], []
    for _, r in df.iterrows():
        f = int(r["frame"]); pi = int(r["point_idx"])
        if f in idx and 0 <= pi < n:
            pt = pts_xy[idx[f], pi]
            rows.append(float(pt[0])); cols.append(float(pt[1])); modes.append(r["mode"])
    return np.array(rows), np.array(cols), np.array(modes)


def hist2d(rows, cols, H, W):
    rbins = np.arange(0, H + BIN_PX, BIN_PX)
    cbins = np.arange(0, W + BIN_PX, BIN_PX)
    h, _, _ = np.histogram2d(rows, cols, bins=[rbins, cbins])
    if SMOOTH_BINS > 0:
        h = gaussian_filter(h, SMOOTH_BINS)
    return h, rbins, cbins


def make_maps(stem: str, data_root: Path, source: str, n: int = N_HEADLINE):
    rd = data_root / f"{stem}_results"
    meta = load_metadata(rd)
    pxmm = detector_params(meta)["px_per_mm"]
    gap_row = detector_params(meta)["gap_row"]
    H, W = frame_shape(rd)
    rows, cols, modes = initiation_points(stem, data_root, source, n)
    if len(rows) == 0:
        print(f"    [{stem}] no initiation points ({source})")
        return
    rA, cA = rows[modes == "A"], cols[modes == "A"]
    rB, cB = rows[modes == "B"], cols[modes == "B"]
    HA, rb, cb = hist2d(rA, cA, H, W)
    HB, _, _ = hist2d(rB, cB, H, W)
    ext = [0, W / pxmm, H / pxmm, 0]
    vmax = max(float(np.percentile(np.concatenate([HA[HA > 0], HB[HB > 0]]), 99))
               if (HA > 0).any() or (HB > 0).any() else 1.0, 1e-6)

    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.subplots_adjust(wspace=0.45)
    for j, (ax, dat, title) in enumerate([(axes[0], HA, f"Mode A  (n={len(rA)})"),
                                          (axes[1], HB, f"Mode B  (n={len(rB)})")]):
        im = ax.imshow(dat, cmap="plasma", vmin=0, vmax=vmax, origin="upper",
                       extent=ext, aspect="equal")
        ax.axhline(gap_row / pxmm, color="cyan", lw=0.8, ls="--")   # apex gap row
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (mm)")
        if j == 0:
            ax.set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("re-strike count")
    # difference of AREA-NORMALISED densities (shape, not count)
    dA = HA / HA.sum() if HA.sum() else HA
    dB = HB / HB.sum() if HB.sum() else HB
    diff = dB - dA
    dlim = max(float(np.percentile(np.abs(diff[diff != 0]), 99)) if (diff != 0).any() else 1e-3, 1e-6)
    im = axes[2].imshow(diff, cmap="RdBu_r", vmin=-dlim, vmax=dlim, origin="upper",
                        extent=ext, aspect="equal")
    axes[2].axhline(gap_row / pxmm, color="k", lw=0.8, ls="--")
    axes[2].set_title("B - A  (normalised density)", fontsize=10)
    axes[2].set_xlabel("x (mm)")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04).set_label("delta density (B-A)")

    cap = figstyle.caption_block(
        recording=stem, fps=meta["fps"], px_per_mm=pxmm,
        bright_thresh=int(meta["bright_thresh"]), n=n,
        sigma_pxmm_rel=0.0, sigma_fps_rel=0.0,
        extra=(f"Re-strike initiation-location heatmaps [{source}]. Mode A (n={len(rA)}) "
               f"| Mode B (n={len(rB)}) | B-A normalised-density difference. Cyan/black "
               f"dashed = apex gap row. Bin {BIN_PX}px, gaussian {SMOOTH_BINS} bins."))
    figstyle.save(fig, f"restrike_location_map_{source}", cap, out_dir=str(rd))
    print(f"    [{stem}] {source}: A={len(rA)} B={len(rB)} -> restrike_location_map_{source}.png")


def cross_recording_depth(data_root: Path, source: str, stems: list):
    """Calibration-free cross-recording comparison of A vs B initiation DEPTH
    (norm_pos: 0 = apex gap, 1 = max extent). tip source only (norm_pos in CSV)."""
    if source != "tip":
        return
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    rows = []
    for stem in stems:
        rd = data_root / f"{stem}_results"
        f = rd / "re_strikes_qa.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        for mode, ls in [("A", "-"), ("B", "--")]:
            v = pd.to_numeric(df[df["mode"] == mode]["norm_pos"], errors="coerce").dropna()
            if len(v) < 3:
                continue
            rows.append(dict(stem=stem, mode=mode, median_norm_pos=float(v.median()),
                             n=len(v)))
    summ = pd.DataFrame(rows)
    # grouped bars: median norm_pos for A vs B per recording
    labels = summ["stem"].unique()
    x = np.arange(len(labels))
    wpb = 0.38
    for i, mode in enumerate(["A", "B"]):
        vals = [summ[(summ["stem"] == s) & (summ["mode"] == mode)]["median_norm_pos"].mean()
                for s in labels]
        ax.bar(x + (i - 0.5) * wpb, vals, wpb,
               color=plot_config.CIVIDIS_DISCRETE[0 if mode == "A" else 4],
               label=f"Mode {mode}")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("HS_", "").replace("_(CO2)", "").replace("_results", "")
                        for s in labels], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("median initiation depth  norm_pos\n(0 = apex gap, 1 = max extent)")
    ax.legend(frameon=False)
    out = data_root / "Reynolds_comparison"
    out.mkdir(exist_ok=True)
    cap = ("Cross-recording A vs B re-strike initiation depth (calibration-free "
           "norm_pos). Mode A initiates near the apex gap (small norm_pos); Mode B "
           "initiates from a more extended arc (larger norm_pos) by construction of "
           "the A/B cut, shown here per recording.")
    figstyle.save(fig, "restrike_initiation_depth_xrecording", cap, out_dir=str(out))
    summ.to_csv(out / "restrike_initiation_depth.csv", index=False)
    print(f"  cross-recording initiation-depth panel -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    ap.add_argument("--source", choices=["tip", "npoint", "both"], default="both")
    ap.add_argument("--n", type=int, default=N_HEADLINE)
    args = ap.parse_args()
    data_root = Path(args.data_root)
    stems = list(RECORDINGS.keys()) if (args.all or not args.recording) else [args.recording]
    sources = ["tip", "npoint"] if args.source == "both" else [args.source]

    for src in sources:
        print(f"=== source={src} ===")
        for stem in stems:
            make_maps(stem, data_root, src, n=args.n)
        cross_recording_depth(data_root, src, stems)


if __name__ == "__main__":
    main()
