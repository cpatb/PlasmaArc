"""
threshold_qa_11may26.py
=======================
Threshold-selection QA for the BLADE 11MAY26 recordings.

Converts a short segment of a .cine file to 8-bit (using the same Drop Impact
convert_cine pipeline as run_11may26.py), then for a range of candidate
thresholds plots:

  Panel A  Pixel-intensity histogram: all frames stacked, log-scale
  Panel B  n_above_thresh vs frame number (reveals arc on/off periods)
  Panel C  Precision-recall tradeoff over candidate thresholds
           (requires manual labelling of ~50 frames — see instructions)
  Panel D  Frame-strip of sample frames at the chosen threshold

Usage
-----
    cd "C:\\Users\\compl\\Github\\Plasma Arc"

    # Quick mode — no labels needed, shows histogram + n_above_thresh trace
    python threshold_qa_11may26.py

    # Label mode — after viewing Panel B, edit LABELED_ARCS below, re-run
    python threshold_qa_11may26.py --label

    # Use a pre-converted TIFF chunk instead of re-converting
    python threshold_qa_11may26.py --tiff "path/to/chunk.tiff"

Outputs
-------
    threshold_qa_11may26.png   4-panel diagnostic figure
"""

import sys, os, argparse, struct as _struct
import numpy as np
import tifffile
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

BLADE_11MAY26 = Path.home() / 'OneDrive' / 'Desktop' / 'BLADE' / 'BLADE 11MAY26'
DROP_IMPACT   = Path(r'C:\Users\compl\Github\Drop Impact')
CINE_PATH     = BLADE_11MAY26 / '0.6mm_1.5SLPM-N2_25deg_OA_downward.cine'

# ── Conversion parameters (must match run_11may26.py) ────────────────────────

N_FRAMES_QA          = 2_000   # frames to convert for QA
N_PERCENTILE_SAMPLE  = 300
SATURATED            = 0.35

# ── Candidate thresholds to sweep ─────────────────────────────────────────────

THRESH_CANDIDATES = list(range(20, 120, 5))   # 20, 25, 30, … 115

# ── Manual arc-frame labels (edit after viewing Panel B) ──────────────────────
# Set arc_frame_indices to a list of frame indices (0-based within the QA
# segment) that clearly contain a fully-bridging arc.  Leave empty to skip
# the precision-recall panel.
# Example: LABELED_ARCS = list(range(120, 180)) + list(range(340, 420))
LABELED_ARCS: list[int] = []

# ── Drop Impact imports ───────────────────────────────────────────────────────

if str(DROP_IMPACT) not in sys.path:
    sys.path.insert(0, str(DROP_IMPACT))

from pycine.raw import read_frames, create_raw_array   # type: ignore
from pycine.file import read_header                    # type: ignore
import pycine.raw as _pycine_raw                       # type: ignore
from cine_to_tiff import correct_bad_pixels_stack      # type: ignore

def _patched_frame_reader(cine_file, header, start_frame=1, count=None):
    frame = start_frame
    if not count:
        count = header["cinefileheader"].ImageCount
    with open(cine_file, "rb") as f:
        while count:
            f.seek(header["pImage"][frame - 1])
            ann_size  = _struct.unpack("I", f.read(4))[0]
            ann_bytes = ann_size - 8
            if ann_bytes > 0:
                if ann_bytes > 16 * 1024 * 1024:
                    raise IOError("Suspicious annotation_size at frame %d" % frame)
                f.seek(ann_bytes, 1)
            img_size = _struct.unpack("I", f.read(4))[0]
            yield create_raw_array(f.read(img_size), header)
            frame += 1
            count -= 1

_pycine_raw.frame_reader = _patched_frame_reader


def _read_frames_at(cine_path, indices, header):
    with open(str(cine_path), "rb") as f:
        for idx in indices:
            f.seek(header["pImage"][idx])
            ann_size  = _struct.unpack("I", f.read(4))[0]
            ann_bytes = ann_size - 8
            if ann_bytes > 0:
                f.seek(ann_bytes, 1)
            img_size = _struct.unpack("I", f.read(4))[0]
            yield create_raw_array(f.read(img_size), header)


# ── Conversion ────────────────────────────────────────────────────────────────

def convert_segment(cine_path, n_frames):
    """Convert the first n_frames to 8-bit using Drop Impact normalisation."""
    print('Converting first %d frames of %s ...' % (n_frames, Path(cine_path).name))

    # Sample for lo/hi
    header = read_header(str(cine_path))
    sample_idx = np.linspace(0, n_frames - 1, N_PERCENTILE_SAMPLE, dtype=int).tolist()
    sample_raw = list(_read_frames_at(cine_path, sample_idx, header))
    sample_corr = correct_bad_pixels_stack(np.stack(sample_raw, 0))
    lo = float(np.percentile(sample_corr, SATURATED / 2))
    hi = float(np.percentile(sample_corr, 100.0 - SATURATED / 2))
    if hi <= lo: hi = lo + 1.0
    scale = 255.0 / (hi - lo)
    del sample_raw, sample_corr
    print('  lo=%.1f  hi=%.1f' % (lo, hi))

    # Stream first n_frames
    raw_gen, _, _ = read_frames(str(cine_path))
    raw_buf = []
    for i, frame in enumerate(raw_gen):
        if i >= n_frames: break
        raw_buf.append(frame)

    corrected = correct_bad_pixels_stack(np.stack(raw_buf, 0))
    del raw_buf
    frames_8bit = np.clip((corrected - lo) * scale, 0, 255).astype(np.uint8)
    del corrected
    print('  Done. Stack shape: %s' % str(frames_8bit.shape))
    return frames_8bit


def load_tiff_stack(tiff_path, max_frames=None):
    print('Loading TIFF: %s' % tiff_path)
    with tifffile.TiffFile(str(tiff_path)) as tif:
        n = len(tif.pages)
        if max_frames:
            n = min(n, max_frames)
        stack = np.stack([tif.pages[i].asarray() for i in range(n)], axis=0)
    print('  Loaded %d frames, shape %s' % (n, str(stack.shape)))
    return stack


# ── Analysis ──────────────────────────────────────────────────────────────────

def run_qa(stack, out_path, label_mode=False):
    N, H, W = stack.shape
    print('Running threshold QA on %d frames ...' % N)

    # Per-frame stats
    frame_max    = stack.max(axis=(1, 2))
    frame_mean   = stack.mean(axis=(1, 2))

    # n_above_thresh for each candidate
    n_above = {t: (stack > t).sum(axis=(1, 2)) for t in THRESH_CANDIDATES}

    fig = plt.figure(figsize=(16, 12), facecolor='#1a1a1a')
    axes = fig.subplots(2, 2)

    clr_bg = '#1a1a1a'
    for ax in axes.ravel():
        ax.set_facecolor('#111111')
        for sp in ax.spines.values(): sp.set_color('#444')
        ax.tick_params(colors='#aaa', labelsize=8)
        ax.yaxis.label.set_color('#ccc')
        ax.xaxis.label.set_color('#ccc')
        ax.title.set_color('#ddd')

    # ── Panel A: pixel intensity histogram (all frames) ───────────────────────
    ax = axes[0, 0]
    counts, edges = np.histogram(stack.ravel(), bins=128, range=(0, 255))
    ax.bar(edges[:-1], counts, width=np.diff(edges), color='steelblue', alpha=0.85)
    ax.set_yscale('log')
    ax.set_xlabel('8-bit pixel value (DN)')
    ax.set_ylabel('Count')
    for t in [50, 60, 70, 80, 90, 100]:
        ax.axvline(t, color='tomato' if t == 80 else 'gold', lw=0.8, alpha=0.7,
                   linestyle='--', label=('DN=%d' % t) if t in [80] else None)
    ax.legend(fontsize=7, labelcolor='white', facecolor='#222')

    # ── Panel B: n_above_thresh trace — reveals arc on/off periods ───────────
    ax = axes[0, 1]
    t_ms = np.arange(N) / 200_000 * 1000
    for t in [60, 70, 80, 90, 100]:
        alpha = 1.0 if t == 80 else 0.55
        lw    = 1.2 if t == 80 else 0.7
        ax.plot(t_ms, n_above[t], lw=lw, alpha=alpha, label='DN>%d' % t)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Pixel count above threshold')
    ax.legend(fontsize=7, labelcolor='white', facecolor='#222')
    if LABELED_ARCS:
        for fi in LABELED_ARCS:
            ax.axvline(fi / 200_000 * 1000, color='lime', lw=0.4, alpha=0.3)

    # ── Panel C: precision-recall sweep (only if labels given) ───────────────
    ax = axes[1, 0]
    if LABELED_ARCS and label_mode:
        gt = np.zeros(N, bool)
        for fi in LABELED_ARCS:
            if 0 <= fi < N: gt[fi] = True
        precisions, recalls, f1s = [], [], []
        for t in THRESH_CANDIDATES:
            # Predict arc when n_above > min_count (use 10 as a loose gate)
            pred = n_above[t] > 10
            tp = int((pred & gt).sum())
            fp = int((pred & ~gt).sum())
            fn = int((~pred & gt).sum())
            p  = tp / max(tp + fp, 1)
            r  = tp / max(tp + fn, 1)
            f1 = 2 * p * r / max(p + r, 1e-9)
            precisions.append(p); recalls.append(r); f1s.append(f1)
        ax.plot(THRESH_CANDIDATES, precisions, 'o-', color='steelblue',  lw=1.2, label='Precision')
        ax.plot(THRESH_CANDIDATES, recalls,    's-', color='tomato',     lw=1.2, label='Recall')
        ax.plot(THRESH_CANDIDATES, f1s,        'd-', color='gold',       lw=1.2, label='F1')
        best_t = THRESH_CANDIDATES[int(np.argmax(f1s))]
        ax.axvline(best_t, color='lime', lw=1.0, linestyle='--',
                   label='Best F1: DN=%d' % best_t)
        ax.set_xlabel('Threshold (DN)')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, labelcolor='white', facecolor='#222')
        print('\nBest threshold by F1: DN>%d  (F1=%.3f, P=%.3f, R=%.3f)' % (
              best_t, max(f1s), precisions[int(np.argmax(f1s))],
              recalls[int(np.argmax(f1s))]))
    else:
        # Show false-positive proxy: mean n_above during the apparent dark periods
        # (frames where frame_max < 40 are almost certainly no-arc)
        dark_mask = frame_max < 40
        n_dark = int(dark_mask.sum())
        if n_dark > 0:
            for t in THRESH_CANDIDATES:
                fp_rate = float((n_above[t][dark_mask] > 10).mean())
                ax.scatter(t, fp_rate, color='tomato', s=20, alpha=0.8)
            ax.set_xlabel('Threshold (DN)')
            ax.set_ylabel('FP rate on dark frames (frame_max < 40)')
            ax.axhline(0, color='#555', lw=0.5)
        else:
            ax.text(0.5, 0.5, 'No dark frames found\n(frame_max < 40)',
                    transform=ax.transAxes, ha='center', va='center', color='#aaa')
        ax.text(0.02, 0.96,
                'Add arc frame indices to LABELED_ARCS\nand re-run with --label\nto get full precision-recall curve.',
                transform=ax.transAxes, va='top', color='#aaa', fontsize=8)

    # ── Panel D: sample frame strip at DN=80 ─────────────────────────────────
    ax = axes[1, 1]
    THRESH_SHOW = 80
    # Pick 6 evenly-spaced frames to show
    sample_fi = np.linspace(0, N - 1, 6, dtype=int)
    strip = np.zeros((H, 6 * W), np.uint8)
    for k, fi in enumerate(sample_fi):
        strip[:, k*W:(k+1)*W] = stack[fi]
    ax.imshow(strip, cmap='gray', vmin=0, vmax=255, aspect='auto')
    # Overlay threshold mask in red channel
    overlay = np.zeros((*strip.shape, 4), np.float32)
    for k, fi in enumerate(sample_fi):
        mask = stack[fi] > THRESH_SHOW
        overlay[:, k*W:(k+1)*W, 0] = mask.astype(float)
        overlay[:, k*W:(k+1)*W, 3] = mask.astype(float) * 0.45
    ax.imshow(overlay, aspect='auto')
    for k, fi in enumerate(sample_fi):
        ax.text(k*W + 3, 5, 't=%d' % fi, color='yellow', fontsize=6, va='top')
        n_px = int((stack[fi] > THRESH_SHOW).sum())
        ax.text(k*W + 3, H - 5, 'n=%d' % n_px, color='cyan', fontsize=6, va='bottom')
    ax.axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(str(out_path), dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print('\nSaved: %s' % out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Threshold QA for BLADE 11MAY26')
    ap.add_argument('--tiff',    default=None,
                    help='Pre-converted TIFF chunk (skips .cine conversion)')
    ap.add_argument('--cine',    default=str(CINE_PATH),
                    help='Source .cine file (default: downward)')
    ap.add_argument('--n-frames', type=int, default=N_FRAMES_QA,
                    help='Frames to convert (default %d)' % N_FRAMES_QA)
    ap.add_argument('--label',   action='store_true',
                    help='Run precision-recall panel (requires LABELED_ARCS)')
    ap.add_argument('--out',     default=str(BLADE_11MAY26 / 'threshold_qa_11may26.png'))
    args = ap.parse_args()

    if args.tiff:
        stack = load_tiff_stack(args.tiff, max_frames=args.n_frames)
    else:
        stack = convert_segment(args.cine, args.n_frames)

    run_qa(stack, args.out, label_mode=args.label)


if __name__ == '__main__':
    main()
