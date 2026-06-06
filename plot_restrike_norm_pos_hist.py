#!/usr/bin/env python3
"""
plot_restrike_norm_pos_hist.py  -  histogram of re-strike (reinitiation)
normalized position, to eyeball where the Mode A / Mode B threshold belongs.

norm_pos is where along the previous arc extent a reinitiation nucleates:
0 = at the electrode gap (Mode A, gap initiation), 1 = at the apex / far end
(Mode B, reconnection). A good threshold sits in the valley between the two
populations. The histogram is intentionally NOT colored by the existing 'mode'
column, because that column is just a hard cut at the current threshold and
would be circular for choosing a new one.

Produces:
  --out          one stacked panel per CSV (raw counts)
  --overlay-out  (optional) a single axes with every direction overlaid

Usage:
    python plot_restrike_norm_pos_hist.py \
        --csv ".../<stem>_results/re_strikes_qa.csv" [more csvs ...] \
        --labels downward upward \
        --bins 100 \
        --out ".../mode_compare/restrike_norm_pos_hist.png" \
        --overlay-out ".../mode_compare/restrike_norm_pos_overlay.png"
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']


def _load(csv):
    df = pd.read_csv(csv)
    return pd.to_numeric(df['norm_pos'], errors='coerce').dropna().values


def main():
    ap = argparse.ArgumentParser(description='Re-strike norm_pos histogram for threshold selection')
    ap.add_argument('--csv', nargs='+', required=True, help='one or more re_strikes_qa.csv files')
    ap.add_argument('--labels', nargs='*', default=None, help='label per csv (default: parent dir name)')
    ap.add_argument('--bins', type=int, default=50)
    ap.add_argument('--out', required=True, help='stacked one-panel-per-direction figure')
    ap.add_argument('--overlay-out', default=None,
                    help='also write a single-axes figure with all directions overlaid')
    args = ap.parse_args()

    csvs = [Path(c) for c in args.csv]
    labels = args.labels if args.labels and len(args.labels) == len(csvs) \
        else [c.parent.name.replace('_results', '') for c in csvs]
    bins = np.linspace(0, 1, args.bins + 1)

    # ---- stacked: one panel per direction (raw counts) ----
    n = len(csvs)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3.2 * n), squeeze=False)
    for ax, csv, label in zip(axes[:, 0], csvs, labels):
        x = _load(csv)
        ax.hist(x, bins=bins, color='tab:gray', edgecolor='white', linewidth=0.4)
        ax.set_xlim(0, 1)
        ax.set_xlabel('normalized reinitiation position  (0 = gap / Mode A,  1 = apex / Mode B)')
        ax.set_ylabel('re-strike count')
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close(fig)
    print('Saved: %s' % out)

    # ---- overlay: all directions on one axes ----
    if args.overlay_out:
        figo, axo = plt.subplots(figsize=(9, 5))
        for i, (csv, label) in enumerate(zip(csvs, labels)):
            x = _load(csv)
            color = _COLORS[i % len(_COLORS)]
            axo.hist(x, bins=bins, histtype='stepfilled', alpha=0.35, color=color)
            axo.hist(x, bins=bins, histtype='step', linewidth=1.8, color=color,
                     label=label)
        axo.set_xlim(0, 1)
        axo.set_xlabel('normalized reinitiation position  (0 = gap / Mode A,  1 = apex / Mode B)')
        axo.set_ylabel('re-strike count')
        axo.legend()
        outo = Path(args.overlay_out); outo.parent.mkdir(parents=True, exist_ok=True)
        figo.tight_layout(); figo.savefig(outo, dpi=150, bbox_inches='tight'); plt.close(figo)
        print('Saved: %s' % outo)


if __name__ == '__main__':
    main()
