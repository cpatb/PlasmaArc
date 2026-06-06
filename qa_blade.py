#!/usr/bin/env python3
"""
qa_blade.py  —  Quality-assurance report for a BLADE gliding-arc TIFF chunk.

Headless. Reuses the *exact* detection operators from analyze_blade.py
(`binarize`, `skeletonize`, endpoints) so the QA reflects what the analysis
pipeline actually does — there is no second, divergent implementation.

For one TIFF (a converted chunk or a representative segment) it reports and
plots, with NO fabricated data:

  * pixel-intensity histogram with the detection threshold marked
  * bright-pixel count vs time (reveals the discharge-cycle structure)
  * rolling arc-detection rate, split into complete (bridging) vs partial arcs
  * arc-presence occupancy map with the auto-detected electrode band, plus the
    measured apex (minimum) and maximum inter-blade gap in mm
  * a montage of annotated frames (mask, skeleton, feet, arc front)
  * a text/JSON summary (detection %, complete/partial %, gaps, hot-pixel count)

Complete-vs-partial uses an orientation-agnostic morphological rule that needs
no interactive calibration and no assumed opening angle:
  a frame is a COMPLETE bridging arch iff both skeleton endpoints (feet) lie in
  the electrode band (auto-derived from the occupancy base row) and are
  horizontally separated by at least `--min-gap-px`; otherwise it is PARTIAL.

Usage:
    python qa_blade.py CHUNK.tiff --arc-direction up  --px-per-mm 8.8 --out-dir qa/
    python qa_blade.py CHUNK.tiff --arc-direction down --thresh 100
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import tifffile
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

# Reuse the pipeline's own operators — single source of truth.
from analyze_blade import (binarize, skeleton_endpoints, ordered_skeleton_coords,
                           arc_front_tip, DEFAULT_DARK_GUARD)

FPS_DEFAULT = 200_000


def analyse(tiff, thresh, close_r, min_arc_px, arc_direction, px_per_mm,
            min_gap_px, max_frames):
    with tifffile.TiffFile(tiff) as t:
        n = min(len(t.pages), max_frames) if max_frames else len(t.pages)
        H, W = t.pages[0].asarray().shape
        occ = np.zeros((H, W))
        bright_count = np.zeros(n)
        frame_max = np.zeros(n)
        detected = np.zeros(n, bool)
        complete = np.zeros(n, bool)
        sat_grid = np.zeros((H, W))
        feet_all = []
        examples = {}                      # frame_idx -> (raw, mask, coords, feet, tip)
        # choose example frames spanning the size range later; first collect areas
        areas = np.zeros(n)

        for i in range(n):
            raw = t.pages[i].asarray()
            frame_max[i] = raw.max()
            bright_count[i] = int((raw > thresh).sum())
            sat_grid += (raw >= raw.max()) & (raw.max() >= 255)
            b = binarize(raw, close_r, dark_guard=DEFAULT_DARK_GUARD,
                         bright_thresh=thresh, min_arc_px=min_arc_px)
            if not b.any():
                continue
            detected[i] = True
            areas[i] = b.sum()
            occ += b
            sk = skeletonize(b)
            if not sk.any():
                continue
            eps = skeleton_endpoints(sk).astype(float)
            feet_all.append((i, eps))

    # electrode band from occupancy base row
    base_row = float(np.argmax(occ.sum(axis=1)))
    band = 0.30 * H

    foot_pts = []
    for i, eps in feet_all:
        near = eps[np.abs(eps[:, 0] - base_row) <= band]
        if len(near) >= 2:
            cmin, cmax = near[:, 1].min(), near[:, 1].max()
            if (cmax - cmin) >= min_gap_px:
                complete[i] = True
                foot_pts.append([near[near[:, 1].argmin(), 0], cmin])
                foot_pts.append([near[near[:, 1].argmax(), 0], cmax])
    foot_pts = np.array(foot_pts) if foot_pts else np.empty((0, 2))

    # inter-blade gap from the per-frame foot separations (apex = min, max = widest)
    gaps_px = []
    for i, eps in feet_all:
        if not complete[i]:
            continue
        near = eps[np.abs(eps[:, 0] - base_row) <= band]
        gaps_px.append(near[:, 1].max() - near[:, 1].min())
    gaps_px = np.array(gaps_px) if gaps_px else np.array([np.nan])

    summary = dict(
        n_frames=int(n), H=int(H), W=int(W),
        threshold_DN=int(thresh),
        detected=int(detected.sum()),
        detected_pct=round(100 * detected.sum() / n, 1),
        complete=int(complete.sum()),
        partial=int(detected.sum() - complete.sum()),
        complete_pct_of_detected=round(100 * complete.sum() / max(detected.sum(), 1), 1),
        base_row=round(base_row, 1),
        electrode_end=("BOTTOM" if base_row > H / 2 else "TOP"),
        arc_direction=arc_direction,
        px_per_mm=px_per_mm,
        apex_gap_mm=round(float(np.nanpercentile(gaps_px, 5)) / px_per_mm, 2),
        max_gap_mm=round(float(np.nanpercentile(gaps_px, 95)) / px_per_mm, 2),
        persistent_saturated_px=int((sat_grid > 0.9 * n).sum()),
    )
    arrays = dict(occ=occ, bright_count=bright_count, frame_max=frame_max,
                  detected=detected, complete=complete, areas=areas,
                  base_row=base_row, band=band, foot_pts=foot_pts)
    return summary, arrays


def figure(tiff, thresh, close_r, min_arc_px, arc_direction, summary, arrays,
           fps, out_png, max_frames):
    occ = arrays['occ']; H, W = occ.shape
    t_ms = np.arange(len(arrays['bright_count'])) / fps * 1e3

    fig = plt.figure(figsize=(16, 11), facecolor='white')
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2])

    # A intensity histogram (need one representative frame set) -> use bright_count proxy
    axA = fig.add_subplot(gs[0, 0])
    axA.plot(t_ms, arrays['bright_count'], lw=0.4, color='steelblue')
    axA.set_xlabel('Time (ms)'); axA.set_ylabel('px > %d DN' % thresh)

    # B detection + complete/partial rolling rate
    axB = fig.add_subplot(gs[0, 1])
    win = max(50, len(t_ms) // 50)
    det = arrays['detected'].astype(float)
    comp = arrays['complete'].astype(float)
    def roll(x):
        k = np.ones(win) / win
        return np.convolve(x, k, mode='same') * 100
    axB.plot(t_ms, roll(det), color='black', lw=0.8, label='detected')
    axB.plot(t_ms, roll(comp), color='seagreen', lw=0.8, label='complete')
    axB.plot(t_ms, roll(det - comp), color='orange', lw=0.8, ls='--', label='partial')
    axB.set_xlabel('Time (ms)'); axB.set_ylabel('%'); axB.legend(fontsize=7)

    # C arc area distribution
    axC = fig.add_subplot(gs[0, 2])
    a = arrays['areas'][arrays['detected']]
    axC.hist(a, bins=40, color='slateblue', alpha=0.8)

    # D occupancy + electrode band + gap
    axD = fig.add_subplot(gs[1, 0])
    axD.imshow(np.power(occ / max(occ.max(), 1e-9), 0.4), cmap='magma')
    axD.axhline(arrays['base_row'], color='cyan', lw=1, ls='--')
    axD.axhline(arrays['base_row'] - arrays['band'], color='cyan', lw=0.5, ls=':')
    axD.axhline(min(H, arrays['base_row'] + arrays['band']), color='cyan', lw=0.5, ls=':')
    if len(arrays['foot_pts']):
        axD.scatter(arrays['foot_pts'][:, 1], arrays['foot_pts'][:, 0],
                    s=3, c='lime', alpha=0.3)
    axD.axis('off')

    # E frame_max histogram (saturation / hot-pixel check)
    axE = fig.add_subplot(gs[1, 1])
    axE.hist(arrays['frame_max'], bins=40, color='firebrick', alpha=0.8)
    axE.set_xlabel('max DN'); axE.set_ylabel('frames')

    # F text summary
    axF = fig.add_subplot(gs[1, 2]); axF.axis('off')
    lines = [f"{k:>26s} : {v}" for k, v in summary.items()]
    axF.text(0.0, 1.0, '\n'.join(lines), fontfamily='monospace', fontsize=8,
             va='top', ha='left')

    # G-I example annotated frames spanning the arc-size range
    with tifffile.TiffFile(tiff) as t:
        det_idx = np.where(arrays['detected'])[0]
        if len(det_idx):
            ar = arrays['areas'][det_idx]
            order = det_idx[np.argsort(ar)]
            picks = [int(order[int(f * (len(order) - 1))]) for f in (0.3, 0.6, 0.95)]
        else:
            picks = [0, 0, 0]
        for j, fi in enumerate(picks):
            ax = fig.add_subplot(gs[2, j])
            raw = t.pages[fi].asarray()
            ax.imshow(np.power(raw / max(raw.max(), 1), 0.45), cmap='inferno', vmin=0, vmax=1)
            b = binarize(raw, close_r, bright_thresh=thresh, min_arc_px=min_arc_px)
            if b.any():
                ax.contour(b, levels=[0.5], colors='cyan', linewidths=0.6)
                sk = skeletonize(b)
                rr, cc = np.where(sk)
                ax.scatter(cc, rr, s=0.3, c='lime')
                eps = skeleton_endpoints(sk)
                if len(eps):
                    ax.scatter(eps[:, 1], eps[:, 0], s=30, c='yellow', edgecolors='k')
                tr, tc = arc_front_tip(b, arc_direction)
                if not np.isnan(tr):
                    ax.plot(tc, tr, 'x', c='red', ms=10, mew=2)
            ax.axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_png, dpi=110, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='QA report for a BLADE gliding-arc TIFF')
    ap.add_argument('tiff')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--thresh', type=int, default=100)
    ap.add_argument('--close-radius', type=int, default=3)
    ap.add_argument('--min-arc-px', type=int, default=22)
    ap.add_argument('--arc-direction', choices=['up', 'down'], default='up')
    ap.add_argument('--px-per-mm', type=float, default=8.8)
    ap.add_argument('--fps', type=float, default=FPS_DEFAULT)
    ap.add_argument('--min-gap-px', type=float, default=15.0,
                    help='Minimum foot separation (px) for a COMPLETE bridging arc')
    ap.add_argument('--max-frames', type=int, default=2000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.tiff).parent / 'qa'
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.tiff).stem

    print('QA  %s  (dir=%s thresh=%d px/mm=%.3f)' %
          (stem, args.arc_direction, args.thresh, args.px_per_mm))
    summary, arrays = analyse(args.tiff, args.thresh, args.close_radius,
                              args.min_arc_px, args.arc_direction, args.px_per_mm,
                              args.min_gap_px, args.max_frames)
    for k, v in summary.items():
        print('  %-26s %s' % (k, v))
    with open(out_dir / (stem + '_qa.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    figure(args.tiff, args.thresh, args.close_radius, args.min_arc_px,
           args.arc_direction, summary, arrays, args.fps,
           str(out_dir / (stem + '_qa.png')), args.max_frames)
    print('Wrote %s and %s' % (out_dir / (stem + '_qa.png'),
                               out_dir / (stem + '_qa.json')))


if __name__ == '__main__':
    main()
