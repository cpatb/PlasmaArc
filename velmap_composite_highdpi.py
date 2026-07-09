#!/usr/bin/env python3
"""
velmap_composite_highdpi.py  -  high-DPI local-tip-velocity maps for the four Re
recordings, in the paper montage style (hs_figures.fig_velocity_map_montage).

Reuses the reference renderer + geometry from the desktop paper_figures/hs_figures.py
so the output matches `fig_velocity_map_montage_noelectrodes.png` exactly, and adds:
  * `fig_velocity_map_montage_noelectrodes_highdpi`  - the 2x2 montage (colorbar on
    the right), at INDIVIDUAL_DPI
  * velocity_maps/Re<re>_velmap.{png,pdf}  - each panel on its own, all IDENTICAL
    pixel dimensions (fixed canvas, common mm window), no colorbar
  * velocity_maps/velmap_colorbar.{png,pdf}  - the shared colorbar alone (place on
    the right of the tiled panels)
  * velocity_maps/Re<re>_velmap.csv  - every populated pixel (row,col,x_mm,
    y_from_gap_mm,local_tip_velocity_ms), apex-anchored to match the plots

All panels share one colour scale (VMIN..vmax, vmax = 98th pct pooled), plasma cmap,
gap apex = midpoint of the two electrode tips, y=0 at the gap.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(r'C:\Users\compl\Desktop\Plasma Arc\J Electrode')
PAPER_FIGS = ROOT / 'paper_figures'
OUT_DIR = ROOT / 'mode_by_re_comparison'
IND_DIR = OUT_DIR / 'velocity_maps'
INDIVIDUAL_DPI = 1200
VMIN = 0.25   # m/s; matches hs_figures._render_velmap_montage
CBAR_LABEL = 'Local tip velocity (m s$^{-1}$)'

# Reuse the reference renderer + geometry loader so styling is identical.
sys.path.insert(0, str(PAPER_FIGS))
import hs_figures as H   # noqa: E402


def cmap():
    c = plt.get_cmap('plasma').copy()
    c.set_bad('0.12')
    return c


def main():
    IND_DIR.mkdir(parents=True, exist_ok=True)

    # High-DPI paper montage (exact reference style, colorbar on the right).
    H.OUT = str(OUT_DIR)
    H.DPI = INDIVIDUAL_DPI
    H.SAVE_PDF = True
    H.set_style()
    items, vmax, xlim, ylim = H._velmap_items()
    H._render_velmap_montage(items, vmax, xlim, ylim,
                             'fig_velocity_map_montage_noelectrodes_highdpi', False)

    cm = cmap()

    # Individual panels: fixed figure size + common mm window + bbox_inches=None
    # => every saved PNG has IDENTICAL pixel dimensions, so they tile cleanly.
    for m, re, A, B, ppm, ar, ac, extent, valid, bbox in items:
        disp = np.ma.masked_where(~valid, m)
        fig, ax = plt.subplots(figsize=(4.4, 5.2))
        ax.imshow(disp, cmap=cm, vmin=VMIN, vmax=vmax, origin='upper',
                  extent=extent, aspect='equal', interpolation='nearest')
        ax.set_title('Re %d' % re)
        ax.set_facecolor('0.12')
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.tick_params(axis='y', which='major', direction='out', length=5,
                       left=True, right=False, labelleft=True)
        ax.set_ylabel('Y from gap (mm)')
        for ext in ('png', 'pdf'):
            fig.savefig(IND_DIR / ('Re%d_velmap.%s' % (re, ext)),
                        dpi=INDIVIDUAL_DPI, facecolor='white', bbox_inches=None)
        plt.close(fig)
        print('Saved Re%d_velmap.png/.pdf' % re)

        # per-pixel CSV, apex-anchored to match the plot's Y-from-gap axis
        rows, cols = np.where(valid)
        pd.DataFrame({
            'row': rows, 'col': cols,
            'x_mm': (cols - ac) / ppm,
            'y_from_gap_mm': (rows - ar) / ppm,
            'local_tip_velocity_ms': m[rows, cols],
        }).sort_values(['row', 'col']).to_csv(
            IND_DIR / ('Re%d_velmap.csv' % re), index=False)
        print('  wrote Re%d_velmap.csv (%d pixels)' % (re, len(rows)))

    # Standalone colorbar (vertical; place on the right of the tiled panels).
    fig, ax = plt.subplots(figsize=(1.4, 5.2))
    sm = plt.cm.ScalarMappable(plt.Normalize(VMIN, vmax), cm)
    fig.colorbar(sm, cax=ax).set_label(CBAR_LABEL)
    for ext in ('png', 'pdf'):
        fig.savefig(IND_DIR / ('velmap_colorbar.%s' % ext),
                    dpi=INDIVIDUAL_DPI, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print('Saved velmap_colorbar.png/.pdf  (shared vmax=%.1f m/s, DPI=%d)'
          % (vmax, INDIVIDUAL_DPI))


if __name__ == '__main__':
    main()
