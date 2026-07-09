#!/usr/bin/env python3
"""
qa_intensity_j.py  —  simple pixel-intensity histograms for the converted J
TIFF chunks, to help pick a detection threshold.

Instead of histogramming the whole video (which is dominated by plume / between-
strike frames), this picks a few frames where the ARC is clearly visible and
histograms only those. One clean histogram per recording, plus a small montage
of the frames that were used (so you can confirm the arc is really there), plus
a combined overlay of all recordings.

Usage:
    python qa_intensity_j.py --all "<...>/J Electrode"
    python qa_intensity_j.py "<...>/HS_0.4mm_1500mL_(CO2)_trig_results"
    python qa_intensity_j.py --all "<...>/J Electrode" --n-arc-frames 12 --bright 150
"""
import argparse
from pathlib import Path
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sample_stack(results_dir, n_sample):
    tiffs = sorted((Path(results_dir) / 'tiff').glob('*.tiff'))
    if not tiffs:
        return None
    handles = [tifffile.TiffFile(str(tp)) for tp in tiffs]
    counts = [len(h.pages) for h in handles]
    starts = np.cumsum([0] + counts)
    total = sum(counts)
    want = np.linspace(0, total - 1, min(n_sample, total), dtype=int)
    frames = []
    for gi in want:
        c = int(np.searchsorted(starts, gi, 'right') - 1)
        frames.append(handles[c].pages[int(gi - starts[c])].asarray())
    for h in handles:
        h.close()
    return np.stack(frames, 0)


def pick_arc_frames(stack, k, bright):
    """k frames with the most bright (arc) pixels; plume-only frames have few."""
    n_bright = (stack > bright).sum(axis=(1, 2))
    return np.argsort(n_bright)[::-1][:k]


def histogram(values, name, n_frames, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=256, range=(0, 255), color='#3b7dd8')
    ax.set_yscale('log')
    ax.set_xlim(0, 255)
    ax.set_xlabel('pixel brightness (0 = black, 255 = brightest)')
    ax.set_ylabel('number of pixels (log scale)')
    ax.set_title('%s\nfrom %d frames with a visible arc' % (name, n_frames))
    ax.grid(True, color='#dddddd', lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def montage(stack, idx, name, out_path):
    k = len(idx)
    cols = min(6, k)
    rows = int(np.ceil(k / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.1 * rows))
    axs = np.atleast_1d(axs).ravel()
    for a in axs:
        a.axis('off')
    for ax, fi in zip(axs, idx):
        ax.imshow(stack[fi], cmap='gray', vmin=0, vmax=255)
        ax.set_title('frame %d' % fi, fontsize=7)
    fig.suptitle('%s — frames used for the histogram' % name, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out_path), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Simple arc-frame intensity histograms (J TIFFs)')
    ap.add_argument('dirs', nargs='*')
    ap.add_argument('--all', default=None, help='a J Electrode folder; scans *_results with tiffs')
    ap.add_argument('--n-sample', type=int, default=4000, help='frames scanned per recording')
    ap.add_argument('--n-arc-frames', type=int, default=12, help='arc frames to histogram')
    ap.add_argument('--bright', type=int, default=150,
                    help='DN above which a pixel counts as "bright arc" when ranking frames')
    ap.add_argument('--combined-out', default=None)
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dirs]
    if args.all:
        dirs += sorted(Path(args.all).glob('*_results'))
    dirs = [d for d in dict.fromkeys(dirs) if (d / 'tiff').exists()]
    if not dirs:
        raise SystemExit('no *_results dirs with a tiff/ subfolder found')

    overlay = []
    for d in dirs:
        name = d.name.replace('_results', '')
        stack = sample_stack(d, args.n_sample)
        if stack is None:
            print('  %s: no tiffs; skip' % name); continue
        idx = pick_arc_frames(stack, args.n_arc_frames, args.bright)
        vals = stack[idx].ravel()
        histogram(vals, name, len(idx), d / 'intensity_histogram.png')
        montage(stack, idx, name, d / 'intensity_histogram_frames.png')
        counts, edges = np.histogram(vals, bins=256, range=(0, 255))
        overlay.append((name, edges[:-1], counts / counts.sum()))
        print('wrote %s  (arc frames: %s)' % (d / 'intensity_histogram.png',
                                              ', '.join(map(str, sorted(idx)))))

    if overlay:
        combined = Path(args.combined_out) if args.combined_out \
            else dirs[0].parent / 'intensity_histogram_combined.png'
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for name, x, h in overlay:
            ax.plot(x, h, lw=1.6, label=name)
        ax.set_yscale('log')
        ax.set_xlim(0, 255)
        ax.set_xlabel('pixel brightness (0 = black, 255 = brightest)')
        ax.set_ylabel('share of pixels (log scale)')
        ax.set_title('Pixel-brightness distribution (arc frames) — all recordings')
        ax.grid(True, color='#dddddd', lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(str(combined), dpi=150)
        plt.close(fig)
        print('wrote %s' % combined)


if __name__ == '__main__':
    main()
