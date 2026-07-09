#!/usr/bin/env python3
"""
threshold_overlay_j.py  —  threshold-overlay strips for the J recordings.

For a few arc frames per recording, show the original grayscale frame and the
pixels kept above each candidate threshold (red overlay), so you can confirm the
arc stays intact and the surrounding plume drops out. Reads the already-converted
TIFF chunks directly.

Usage:
    python threshold_overlay_j.py --all "<...>/J Electrode"
    python threshold_overlay_j.py --all "<...>/J Electrode" --thresholds 200 210 220
"""
import argparse
from pathlib import Path
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_THRESHOLDS = [200, 210, 220]
N_FRAMES = 4


def sample_stack(results_dir, n_sample):
    tiffs = sorted((Path(results_dir) / 'tiff').glob('*.tiff'))
    if not tiffs:
        return None
    handles = [tifffile.TiffFile(str(tp)) for tp in tiffs]
    counts = [len(h.pages) for h in handles]
    starts = np.cumsum([0] + counts)
    total = sum(counts)
    want = np.linspace(0, total - 1, min(n_sample, total), dtype=int)
    out = []
    for gi in want:
        c = int(np.searchsorted(starts, gi, 'right') - 1)
        out.append(handles[c].pages[int(gi - starts[c])].asarray())
    for h in handles:
        h.close()
    return np.stack(out, 0)


def red_overlay(ax, frame, thresh=None):
    ax.imshow(frame, cmap='gray', vmin=0, vmax=255)
    if thresh is not None:
        mask = frame > thresh
        ov = np.zeros((*frame.shape, 4), float)
        ov[mask] = [1, 0, 0, 0.55]
        ax.imshow(ov)
        ax.text(3, 12, 'n=%d' % int(mask.sum()), color='yellow', fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser(description='Threshold-overlay strips (J TIFFs)')
    ap.add_argument('dirs', nargs='*')
    ap.add_argument('--all', default=None, help='a J Electrode folder; scans *_results with tiffs')
    ap.add_argument('--thresholds', type=int, nargs='+', default=DEFAULT_THRESHOLDS)
    ap.add_argument('--n-sample', type=int, default=4000)
    ap.add_argument('--bright', type=int, default=150, help='DN for ranking arc frames')
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dirs]
    if args.all:
        dirs += sorted(Path(args.all).glob('*_results'))
    dirs = [d for d in dict.fromkeys(dirs) if (d / 'tiff').exists()]
    if not dirs:
        raise SystemExit('no *_results dirs with a tiff/ subfolder found')

    cols = 1 + len(args.thresholds)
    for d in dirs:
        name = d.name.replace('_results', '')
        stack = sample_stack(d, args.n_sample)
        if stack is None:
            continue
        nb = (stack > args.bright).sum(axis=(1, 2))
        top = np.argsort(nb)[::-1][:30]
        idx = sorted(top[np.linspace(0, len(top) - 1, N_FRAMES, dtype=int)])
        fig, axs = plt.subplots(N_FRAMES, cols, figsize=(2.4 * cols, 2.4 * N_FRAMES))
        for r, fi in enumerate(idx):
            red_overlay(axs[r, 0], stack[fi])
            axs[r, 0].set_ylabel('frame %d' % fi, fontsize=8)
            for c, t in enumerate(args.thresholds, start=1):
                red_overlay(axs[r, c], stack[fi], t)
        for c, lab in enumerate(['original'] + ['threshold %d' % t for t in args.thresholds]):
            axs[0, c].set_title(lab, fontsize=10)
        fig.suptitle('%s  — red = pixels kept above threshold' % name, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = d / 'threshold_overlay.png'
        fig.savefig(str(out), dpi=140)
        plt.close(fig)
        print('wrote %s' % out)


if __name__ == '__main__':
    main()
