#!/usr/bin/env python3
"""
qa_restrike_frames.py  —  Per-re-strike mode-inspection gallery.

Ground-truths the Mode A/B classifier by showing, for EVERY re-strike, the exact
initiation frame it was classified from, labelled and border-coloured by its
assigned mode. Each cell overlays the apex (min-gap) line and the recorded arc
tip, so you can confirm by eye:
  * Mode A frames  -> arc rooted AT / very near the yellow apex line (short arc)
  * Mode B frames  -> arc re-struck ABOVE the apex (re-ignited away from the gap)

Reads re_strikes_qa.csv (written by qa_restrike.py) so the frames shown are
exactly those used for classification — no re-detection.

Outputs paginated PNGs: <out-dir>/restrike_frames_p01.png, _p02.png, ...

Usage:
    python qa_restrike_frames.py \
        --restrikes "<results>/re_strikes_qa.csv" \
        --tiff      "<results>/tiff/<stem>_0001.tiff" \
        --arc-direction down --out-dir "<results>/restrike_gallery"
    # inspect only one mode, biggest grid:
    python qa_restrike_frames.py ... --mode B --per-page 30 --cols 6
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from analyze_blade import binarize

MODE_COLOR = {'A': '#1565C0', 'B': '#C62828'}
FPS_DEFAULT = 200_000


def main():
    ap = argparse.ArgumentParser(description='Per-re-strike Mode A/B inspection gallery')
    ap.add_argument('--restrikes', required=True, help='re_strikes_qa.csv from qa_restrike.py')
    ap.add_argument('--tiff', required=True)
    ap.add_argument('--arc-direction', choices=['up', 'down'], required=True)
    ap.add_argument('--thresh', type=int, default=100, help='binarization DN threshold')
    ap.add_argument('--fps', type=float, default=FPS_DEFAULT)
    ap.add_argument('--frame-offset', type=int, default=0)
    ap.add_argument('--mode', choices=['A', 'B', 'all'], default='all',
                    help='inspect only this mode (default all)')
    ap.add_argument('--sort', choices=['mode', 'norm_pos', 'frame'], default='mode',
                    help='ordering of cells')
    ap.add_argument('--per-page', type=int, default=24)
    ap.add_argument('--cols', type=int, default=6)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    rs = pd.read_csv(args.restrikes)
    if args.mode != 'all':
        rs = rs[rs['mode'] == args.mode]
    if args.sort == 'mode':
        rs = rs.sort_values(['mode', 'norm_pos'])
    elif args.sort == 'norm_pos':
        rs = rs.sort_values('norm_pos')
    else:
        rs = rs.sort_values('frame')
    rs = rs.reset_index(drop=True)
    if len(rs) == 0:
        raise SystemExit('No re-strikes to show (check --mode filter / CSV).')

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    n_a = int((rs['mode'] == 'A').sum()); n_b = int((rs['mode'] == 'B').sum())
    print('Re-strikes to inspect: %d  (A=%d  B=%d)' % (len(rs), n_a, n_b))

    import tifffile
    tif = tifffile.TiffFile(args.tiff)
    n_pages = len(tif.pages)
    H, W = tif.pages[0].asarray().shape

    cols = args.cols
    rows = int(np.ceil(args.per_page / cols))
    per_page = rows * cols
    n_total_pages = int(np.ceil(len(rs) / per_page))

    for pg in range(n_total_pages):
        sub = rs.iloc[pg * per_page:(pg + 1) * per_page]
        fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.7 * rows),
                                 facecolor='white', squeeze=False)
        for ax in axes.ravel():
            ax.axis('off')
        for cell, (_, e) in enumerate(sub.iterrows()):
            ax = axes.ravel()[cell]
            mode = e['mode']; color = MODE_COLOR.get(mode, '#555')
            page = int(e['frame']) - args.frame_offset
            apex = float(e['apex_row']); tipr = float(e['tip_row'])
            raw_apex = (H - 1 - apex) if args.arc_direction == 'up' else apex
            raw_tip = (H - 1 - tipr) if args.arc_direction == 'up' else tipr
            ok = 0 <= page < n_pages
            if ok:
                try:
                    raw = tif.pages[page].asarray()
                except Exception:
                    ok = False
            if ok:
                ax.imshow(np.power(raw / max(raw.max(), 1), 0.45),
                          cmap='inferno', vmin=0, vmax=1)
                b = binarize(raw, 3, bright_thresh=args.thresh)
                if b.any():
                    ax.contour(b, levels=[0.5], colors='cyan', linewidths=0.5)
                ax.axhline(raw_apex, color='yellow', lw=1, ls='--')
                tc = e.get('tip_col', np.nan)
                if pd.notna(tc):
                    ax.plot(float(tc), raw_tip, 'x', c='red', ms=9, mew=2)
                ax.set_xlim(0, W); ax.set_ylim(H, 0)
                # mode-coloured border
                ax.add_patch(mpatches.Rectangle((1, 1), W - 2, H - 2, fill=False,
                             edgecolor=color, linewidth=3))
                rf = e.get('retreat_frac', np.nan)
                t_ms = int(e['frame']) / args.fps * 1e3
            else:
                ax.text(0.5, 0.5, 'frame %d\nnot in TIFF' % int(e['frame']),
                        ha='center', va='center', fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        outp = out_dir / ('restrike_frames_p%02d.png' % (pg + 1))
        fig.savefig(outp, dpi=110, facecolor='white')
        plt.close(fig)
        print('Saved %s' % outp)
    tif.close()
    print('Inspect: Mode A cells (blue border) should show the arc at the yellow apex; '
          'Mode B cells (red border) should show it re-struck above it. Mislabels jump out.')


if __name__ == '__main__':
    main()
