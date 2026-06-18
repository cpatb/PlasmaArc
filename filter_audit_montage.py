"""
filter_audit_montage.py — visual evidence that the unified arc_filter criteria
reject only genuine junk and accept only genuine arcs.

For a given target recording:
  * Sample N frames from each of three buckets:
      ACCEPTED        arc_filter.valid_frames(df)                    (criteria 1∧2∧3)
      REJECTED-flash  is_flash == 1                                  (criterion 3)
      REJECTED-incomp arc_detected == 1 & arc_complete == 0 & ~flash (criterion 2 only)
  * Render a 3 × N montage of raw frames with arc overlays + the criterion that
    made the call printed under each frame.
  * Save filter_audit_montage.{pdf,png,txt}.

Random sampling uses a fixed seed recorded in the figure caption (Plan agent
pushback: no silent stochasticity).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import arc_filter
import figstyle


def _fetch_frame(tiff_dir, frame_idx, frames_per_tiff=10000):
    """Pull a raw frame by global index from numbered chunk TIFFs."""
    import tifffile
    tiff_dir = Path(tiff_dir)
    tiffs = sorted(tiff_dir.glob('*.tiff'))
    if not tiffs:
        return None
    k    = int(frame_idx) // frames_per_tiff
    page = int(frame_idx) %  frames_per_tiff
    if k >= len(tiffs):
        return None
    try:
        with tifffile.TiffFile(str(tiffs[k])) as t:
            if page >= len(t.pages):
                return None
            return t.pages[page].asarray()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True, type=str,
                    help='Target <stem>_results dir.')
    ap.add_argument('--n-each',  type=int, default=5,
                    help='Frames sampled per bucket.')
    ap.add_argument('--seed',    type=int, default=20260618,
                    help='Random seed for the sample (recorded in caption).')
    ap.add_argument('--frames-per-tiff', type=int, default=10000)
    args = ap.parse_args()

    results = Path(args.results)
    meta    = json.loads((results / 'run_metadata.json').read_text())
    df      = pd.read_csv(results / 'measurements_all_events.csv')

    df  = arc_filter.add_flash_flag(df)
    det = pd.to_numeric(df['arc_detected'], errors='coerce').fillna(0).astype(int) == 1
    cmp = pd.to_numeric(df['arc_complete'], errors='coerce').fillna(0).astype(int) == 1
    fls = df['is_flash'].astype(bool)

    accept = df[det & cmp & ~fls]
    flash  = df[fls]                      # criterion 3 fired
    incomp = df[det & ~cmp & ~fls]        # detected but skeleton doesn't bridge

    rng = np.random.default_rng(args.seed)
    def pick(g, n):
        if len(g) == 0:
            return []
        idx = rng.choice(len(g), size=min(n, len(g)), replace=False)
        return g.iloc[sorted(idx)]

    pa = pick(accept, args.n_each)
    pf = pick(flash,  args.n_each)
    pi = pick(incomp, args.n_each)

    rows_spec = [(pa, 'ACCEPTED', 'criteria 1∧2∧3', len(accept))]
    if len(pf):
        rows_spec.append((pf, 'REJECTED', 'criterion 3 (is_flash)', len(flash)))
    if len(pi):
        rows_spec.append((pi, 'REJECTED', 'criterion 2 (incomplete)', len(incomp)))

    figstyle.apply()
    nrows = len(rows_spec)
    ncols = args.n_each
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.3),
                             squeeze=False)

    tiff_dir = results / 'tiff'

    def draw_row(row_idx, rows, label, criterion, bucket_n):
        for j in range(ncols):
            ax = axes[row_idx, j]
            if j >= len(rows):
                ax.axis('off'); continue
            r = rows.iloc[j]
            img = _fetch_frame(tiff_dir, int(r['frame']), args.frames_per_tiff)
            if img is None:
                ax.text(0.5, 0.5, 'NO TIFF', ha='center', va='center',
                        transform=ax.transAxes); ax.axis('off'); continue
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            if 'arc_tip_row' in r and 'arc_tip_col' in r and \
               np.isfinite(r['arc_tip_row']) and np.isfinite(r['arc_tip_col']):
                ax.plot(r['arc_tip_col'], r['arc_tip_row'],
                        marker='o', mfc='none', mec='#fde725', ms=8, mew=1.4)
            ax.set_xticks([]); ax.set_yticks([])
            tag = (f"f={int(r['frame'])} "
                   f"area={int(r.get('area_px', 0))} "
                   f"w={float(r.get('arc_width_px', np.nan)):.1f}px")
            ax.set_title(tag, fontsize=7)
            if j == 0:
                ax.set_ylabel(f'{label}\n{criterion}\n(n={bucket_n} total)',
                              fontsize=8, rotation=0, ha='right', va='center')

    for i, (rows, label, criterion, n) in enumerate(rows_spec):
        draw_row(i, rows, label, criterion, n)
    fig.suptitle(f'arc_filter audit — {results.name}', fontsize=10)
    fig.tight_layout()

    cap = figstyle.caption_block(
        recording=meta['recording'],
        fps=meta['fps'], px_per_mm=meta['px_per_mm'],
        bright_thresh=meta['bright_thresh'],
        n=args.n_each * 3,
        sigma_pxmm_rel=meta.get('sigma_pxmm_rel') or 0.05,
        sigma_fps_rel=meta.get('sigma_fps_rel')  or 1e-3,
        seed=args.seed,
        extra=(
            'Rows: ACCEPTED — valid_frames() (criteria 1+2+3 hold).\n'
            'REJECTED-flash — is_flash==1 (eff_width or arc_width_px above '
            'threshold).\nREJECTED-incomplete — arc_detected==1 but '
            'arc_complete==0 (skeleton fails to bridge both electrodes).\n'
            'Tip-row/col, area_px and arc_width_px shown per frame. '
            'Random seed recorded so the sample is reproducible.'
        ))
    stem = results / 'filter_audit_montage'
    figstyle.save(fig, str(stem), cap)
    figstyle.verify(str(stem))
    print(f"Saved: {stem}.pdf, .png, .txt")


if __name__ == '__main__':
    main()
