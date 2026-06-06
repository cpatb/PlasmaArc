#!/usr/bin/env python3
"""
qa_flash.py  —  Flash explorer: rank detected frames by any metric to find the
signature of the brief bright bursts ("flashes") that corrupt min-gap velocities.

Pick a --metric; the tool shows the frames with the most extreme values of it
(candidate flashes) plus a reference row of median frames, and the metric's
histogram. Flip through metrics to see which one cleanly isolates the flashes you
recognize, then use that metric+threshold as the flash filter.

Available metrics (computed from measurements_all_events.csv columns):
  area_px               mask pixel count (flashes are large; >3000 already rejected)
  integrated_intensity  summed normalised brightness (flashes are bright)
  mean_intensity        mean normalised brightness over the mask
  peak_intensity        peak normalised brightness
  eff_width             area_px / arc_length_px   (also flags broken skeletons)
  fill                  area_px / bbox_area        (compactness; blob=high)
  bbox_diag             bounding-box diagonal (px)
  dist_from_apex        arc_tip_row - p5(arc_tip_row)  (how far from the gap)

Usage:
    python qa_flash.py --csv "<results>/measurements_all_events.csv" \
        --tiff-dir "<results>/tiff" --frames-per-tiff 10000 \
        --metric integrated_intensity --top 16 --out "<results>/qa_flash_intensity.png"
    # add --threshold T to draw a cut line and split flagged/normal rows
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from analyze_blade import binarize
from qa_restrike import FrameSource

NUM = ['area_px', 'arc_length_px', 'integrated_intensity', 'mean_intensity',
       'peak_intensity', 'bbox_r0', 'bbox_c0', 'bbox_r1', 'bbox_c1', 'arc_tip_row']


def add_metrics(df):
    for c in NUM:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['eff_width'] = df['area_px'] / df['arc_length_px'].clip(lower=1)
    bh = (df['bbox_r1'] - df['bbox_r0']).abs()
    bw = (df['bbox_c1'] - df['bbox_c0']).abs()
    df['fill'] = df['area_px'] / (bh * bw).clip(lower=1)
    df['bbox_diag'] = np.hypot(bh, bw)
    apex = df.loc[df['arc_detected'] == 1, 'arc_tip_row'].quantile(0.05)
    df['dist_from_apex'] = df['arc_tip_row'] - apex
    return df


def draw(ax, e, src, thresh_dn, metric, color, H, W):
    raw = src.get(int(e['frame']))
    if raw is None:
        ax.text(0.5, 0.5, 'f%d n/a' % int(e['frame']), ha='center', va='center'); ax.axis('off'); return
    ax.imshow(np.power(raw / max(raw.max(), 1), 0.45), cmap='inferno', vmin=0, vmax=1)
    b = binarize(raw, 3, bright_thresh=thresh_dn)
    if b.any():
        ax.contour(b, levels=[0.5], colors='cyan', linewidths=0.5)
    ax.add_patch(mpatches.Rectangle((1, 1), W - 2, H - 2, fill=False, edgecolor=color, linewidth=3))
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')


def main():
    ap = argparse.ArgumentParser(description='Flash explorer — rank frames by a metric')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--tiff'); ap.add_argument('--tiff-dir')
    ap.add_argument('--frames-per-tiff', type=int, default=None)
    ap.add_argument('--frame-offset', type=int, default=0)
    ap.add_argument('--thresh', type=int, default=100, help='binarization DN threshold')
    ap.add_argument('--metric', default='integrated_intensity',
                    help='area_px|integrated_intensity|mean_intensity|peak_intensity|'
                         'eff_width|fill|bbox_diag|dist_from_apex|arc_length_px')
    ap.add_argument('--order', choices=['high', 'low'], default='high',
                    help='show the highest (default) or lowest values of the metric')
    ap.add_argument('--top', type=int, default=16, help='number of candidate frames to show')
    ap.add_argument('--threshold', type=float, default=None,
                    help='draw a cut line on the histogram; split flagged vs normal rows')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    df = add_metrics(pd.read_csv(args.csv))
    det = df[df['arc_detected'] == 1].dropna(subset=[args.metric]).copy()
    m = args.metric
    asc = (args.order == 'low')
    ranked = det.sort_values(m, ascending=asc)
    cand = ranked.head(args.top)
    ref = det.iloc[(det[m] - det[m].median()).abs().argsort()[:args.top]]  # median frames
    print('detected=%d  metric=%s  range [%.2f, %.2f]  mean=%.2f'
          % (len(det), m, det[m].min(), det[m].max(), det[m].mean()))
    if args.threshold is not None:
        n_fl = int((det[m] > args.threshold).sum())
        print('  %d frames %s %.2f (%.1f%%)' % (n_fl, '>', args.threshold, 100 * n_fl / len(det)))

    src = FrameSource(tiff=args.tiff, tiff_dir=args.tiff_dir,
                      frames_per_tiff=args.frames_per_tiff, frame_offset=args.frame_offset)
    H, W = src.H, src.W
    cols = min(8, args.top)
    rows_c = int(np.ceil(args.top / cols))
    fig = plt.figure(figsize=(2.3 * cols, 2.0 + 2.7 * (rows_c + 1)), facecolor='white')
    gs = fig.add_gridspec(1 + rows_c + 1, cols, height_ratios=[1.0] + [1] * rows_c + [1])

    axh = fig.add_subplot(gs[0, :])
    v = det[m].clip(upper=np.nanpercentile(det[m], 99.5))
    axh.hist(v, bins=60, color='#607D8B')
    if args.threshold is not None:
        axh.axvline(args.threshold, color='red', lw=2, ls='--', label='cut=%.2f' % args.threshold)
        axh.legend(fontsize=9)
    axh.set_yscale('log'); axh.set_xlabel(m); axh.set_ylabel('frames (log)')

    # candidate (extreme-metric) frames
    cand_list = list(cand.iterrows())
    for i, (_, e) in enumerate(cand_list):
        r, c = divmod(i, cols)
        draw(fig.add_subplot(gs[1 + r, c]), e, src, args.thresh, m, '#C62828', H, W)
    for i in range(len(cand_list), rows_c * cols):
        r, c = divmod(i, cols); fig.add_subplot(gs[1 + r, c]).axis('off')
    # reference (median) frames
    for j, (_, e) in enumerate(list(ref.iterrows())[:cols]):
        draw(fig.add_subplot(gs[1 + rows_c, j]), e, src, args.thresh, m, '#2E7D32', H, W)
    src.close()

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=120, facecolor='white')
    plt.close(fig)
    print('Saved %s' % args.out)


if __name__ == '__main__':
    main()
