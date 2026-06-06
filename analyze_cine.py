"""
analyze_cine.py  —  J-electrode CINE arc analysis
==================================================
Reads a Phantom .cine file and extracts per-frame arc measurements.

Code recycled from existing repos
----------------------------------
  pycine.raw.read_frames          — CINE reader (same as Drop Impact)
  cine_to_tiff._correct_frame     — bad-pixel correction (same as Drop Impact)
  cine_to_tiff.correct_bad_pixels_stack  — parallel bad-pixel stack fix
  cine_to_tiff percentile normalization  — lo/hi from sampled frames, not
                                           per-frame min/max, so the scale
                                           is consistent across the recording
  plasma_utils.image.binarize     — Otsu arc mask (Plasma Arc pipeline)
  plasma_utils.image.measure_frame — skeleton, tips, length, width, etc.

The only new code here is:
  • Slicing the generator to the first max_frames (so we don't OOM on 100 GB)
  • binarize_cine() — bright-nucleation clip before Otsu
  • The per-frame analysis loop + CSV/QA output

Usage
-----
  python analyze_cine.py                           # defaults in CONFIG
  python analyze_cine.py path/to/file.cine out_dir --max-gb=2
"""

import sys
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage as ndi
import os
import platform

# ── Repos on sys.path ────────────────────────────────────────────────────────

# Drop Impact repo — for bad-pixel correction and the pycine reader
DROP_IMPACT = Path(r'C:\Users\compl\Github\Drop Impact')
sys.path.insert(0, str(DROP_IMPACT))

# This repo — for arc-image utilities
sys.path.insert(0, str(Path(__file__).parent))

# ── Imports from Drop Impact's cine_to_tiff.py ───────────────────────────────
from pycine.raw import read_frames                    # correct CINE reader
from cine_to_tiff import (                            # type: ignore[import]
    _correct_frame,
    correct_bad_pixels_stack,
)

# ── Monkey-patch pycine.raw.frame_reader ────────────────────────────────────
# Upstream bug: line 25 of pycine/raw.py has
#   struct.unpack("{}B".format(annotation_size - 8),
#                 f.read((annotation_size - 8) // 8))
# The format string and the read length disagree by a factor of 8, and when
# annotation_size is large (or the file pointer drifts to garbage on some
# frame), f.read() tries to allocate hundreds of MB / GB → MemoryError.
# The annotation block is unused by this analyzer, so we just seek past it.
import struct as _struct
import pycine.raw as _pycine_raw
from pycine.raw import create_raw_array as _create_raw_array

def _patched_frame_reader(cine_file, header, start_frame=1, count=None):
    frame = start_frame
    if not count:
        count = header["cinefileheader"].ImageCount
    with open(cine_file, "rb") as f:
        while count:
            frame_index = frame - 1
            f.seek(header["pImage"][frame_index])
            annotation_size = _struct.unpack("I", f.read(4))[0]
            ann_bytes = annotation_size - 8
            if ann_bytes > 0:
                # Bounded sanity check so a corrupt offset can't blow up RAM
                if ann_bytes > 16 * 1024 * 1024:
                    raise IOError(f"Suspicious annotation_size={annotation_size}"
                                  f" at frame {frame}; pImage offset is likely"
                                  f" misaligned")
                f.seek(ann_bytes, 1)
            image_size = _struct.unpack("I", f.read(4))[0]
            data = f.read(image_size)
            yield _create_raw_array(data, header)
            frame += 1
            count -= 1

_pycine_raw.frame_reader = _patched_frame_reader


def _read_frames_at(cine_path, frame_indices, header):
    """
    Yield raw images for specific 0-based frame indices, opening the file
    once and seeking directly via header['pImage'] — no full sequential scan.
    Used by Pass 1 sampling so we don't decode 873k frames just to keep 300.
    """
    with open(cine_path, "rb") as f:
        for idx in frame_indices:
            f.seek(header["pImage"][idx])
            annotation_size = _struct.unpack("I", f.read(4))[0]
            ann_bytes = annotation_size - 8
            if ann_bytes > 0:
                if ann_bytes > 16 * 1024 * 1024:
                    raise IOError(f"Suspicious annotation_size={annotation_size}"
                                  f" at frame index {idx}")
                f.seek(ann_bytes, 1)
            image_size = _struct.unpack("I", f.read(4))[0]
            data = f.read(image_size)
            yield _create_raw_array(data, header)

# ── Imports from Plasma Arc pipeline ─────────────────────────────────────────
from plasma_utils.image import binarize, measure_frame

# =============================================================================
# CONFIG  —  edit before running
# =============================================================================

if platform.system() == 'Windows':
    DEFAULT_CINE = Path(r'C:\Users\compl\OneDrive\Desktop\J electrode arc'
                        r'\AD_Cer_1.2mmNozz_1500mL_(CO2).cine')
    DEFAULT_OUT  = Path(r'C:\Users\compl\OneDrive\Desktop\J electrode arc'
                        r'\analysis')
else:
    DEFAULT_CINE = Path('/sessions/epic-kind-sagan/mnt/BLADE/../'
                        'J electrode arc/AD_Cer_1.2mmNozz_1500mL_(CO2).cine')
    DEFAULT_OUT  = Path('/sessions/epic-kind-sagan/mnt/BLADE/../'
                        'J electrode arc/analysis')

MAX_GB         = None       # None = full file; set e.g. 2.0 for a partial run
CHUNK_GB       = 2.0        # RAM used per processing chunk (tune to your machine)
FPS_FALLBACK   = 200_000    # used only if pycine cannot read FPS from header

# Physical calibration — set once you measure a known feature
PX_PER_MM      = None       # e.g. 9.167  (all mm outputs are nan until set)

# --- Normalization (identical logic to cine_to_tiff.convert_cine) -----------
# These keep the brightness scale consistent across the recording so
# thresholding behaves the same way throughout the clip.
SATURATED        = 0.35     # % of pixels clipped at each end (PCC default)
N_SAMPLE         = 300      # frames sampled to estimate the lo/hi limits

# --- Binarization for bright-nucleation J-electrode arc ---------------------
# Nucleation spots are much brighter than the arc body.  Clipping at
# BRIGHT_CLIP_PCTILE before Otsu prevents those spots from pulling the
# threshold so high that the dimmer arc body is missed.
# clip=99 / gate=50 gives the cleanest arc body on bright frames (QA-confirmed).
BRIGHT_CLIP_PCTILE = 99     # clip above this percentile before Otsu
OTSU_GATE_PCTILE   = 50     # Otsu runs on pixels above this percentile

CLOSE_R = 3                 # morphological closing radius (bridges arc gaps)
OPEN_R  = 2                 # morphological opening radius (removes speckle)

MIN_ARC_AREA_PX =     50    # anything smaller is just a noise speck
MAX_ARC_AREA_PX = 150_000   # J-electrode arcs are large; raised from 25k

# Brightness gate — frames darker than this are rejected before binarization.
# clip=99 + per-frame renormalization stretches dark frames into fake arcs;
# this guard catches frames where the 99th-percentile pixel is just noise.
# Tune upward if dark background frames still slip through.
MIN_FRAME_SIGNAL = 0.45    # 99th-pctile of globally-normalised frame

# arc_complete proxy: skeleton endpoints span > this fraction of frame width
ARC_COMPLETE_WIDTH_FRAC = 0.2

# QA
QA_EVERY_N       = 200      # save annotated frame every N frames processed
QA_THRESH_N      = 20       # first N frames get the 6-panel threshold grid
FRAME_DPI        = 110

# =============================================================================
# BINARIZATION  (adapted for bright-nucleation)
# =============================================================================

def binarize_cine(gray: np.ndarray,
                  clip_pctile: float | None = None,
                  gate_pctile: float | None = None) -> np.ndarray | None:
    """
    Clip nucleation hotspots then run the standard Otsu pipeline.

    gray must already be float32 in [0, 1] (after the global lo/hi scaling
    from convert_cine's normalization step, NOT per-frame min/max).
    """
    cp = clip_pctile if clip_pctile is not None else BRIGHT_CLIP_PCTILE
    gp = gate_pctile if gate_pctile is not None else OTSU_GATE_PCTILE

    # Brightness gate: reject dark / no-arc frames before binarization.
    # clip=99 + per-frame renormalization would otherwise stretch background
    # noise to fill [0, 1], letting Otsu produce a fake "arc" mask.
    # On a frame with a visible arc, 1-5% of pixels are bright arc body pixels,
    # which push the 99th percentile well above background.
    clip_val = float(np.percentile(gray, cp))
    if clip_val < MIN_FRAME_SIGNAL:
        return None

    # Clip hotspots and renormalize so the arc body fills [0, 1]
    gray_c = np.clip(gray, 0.0, clip_val) / clip_val

    # Standard Otsu + morphological cleanup
    mask = binarize(gray_c, close_r=CLOSE_R, open_r=OPEN_R,
                    otsu_percentile=gp)

    # Keep only the largest connected component.
    # This trims scattered background noise on dim frames where clip=99 can
    # pick up a halo around the nucleation spot — those stray blobs become
    # separate components and are discarded here.
    labeled, n_comp = ndi.label(mask)  # type: ignore[misc]
    if n_comp == 0:
        return None
    if n_comp > 1:
        comp_sizes = ndi.sum(mask, labeled, range(1, n_comp + 1))
        largest_label = int(np.argmax(comp_sizes)) + 1
        mask = (labeled == largest_label)

    area = int(mask.sum())
    if area < MIN_ARC_AREA_PX or area > MAX_ARC_AREA_PX:
        return None
    return mask


# =============================================================================
# ARC TIP  (bottommost pixel — furthest from electrodes)
# =============================================================================

def arc_tip(mask: np.ndarray) -> tuple[float, float]:
    rows, cols = np.where(mask)
    tip_row = float(rows.max())
    tip_col = float(cols[rows == int(tip_row)].mean())
    return tip_row, tip_col


def arc_complete_flag(m, frame_shape: tuple) -> bool:
    W = frame_shape[1]
    if not (np.isnan(m.tip_a_col) or np.isnan(m.tip_b_col)):
        return bool(abs(m.tip_b_col - m.tip_a_col) > ARC_COMPLETE_WIDTH_FRAC * W)
    return False


# =============================================================================
# QA IMAGES
# =============================================================================

def render_qa_frame(gray, frame_i, mask, tip_row, tip_col,
                    detected, arc_len_mm, area_px, out_path, category=''):
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8),
                             gridspec_kw={'wspace': 0.04})
    for ax in axes:
        ax.imshow(gray, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    if mask is not None and mask.any():
        ov = np.zeros((*gray.shape, 4), dtype=np.float32)
        ov[mask] = [0.0, 0.85, 0.15, 0.45] if detected else [1.0, 0.5, 0.0, 0.40]
        axes[1].imshow(ov)
    if detected and not (np.isnan(tip_row) or np.isnan(tip_col)):
        for ax in axes:
            ax.plot(tip_col, tip_row, '+', color='yellow',
                    markersize=16, markeredgewidth=2)
    det_str = 'DETECTED' if detected else 'NOT DETECTED'
    arc_str = (f'  area={area_px}px  len={arc_len_mm:.1f}mm'
               if detected and not np.isnan(arc_len_mm) else '')
    fig.savefig(out_path, dpi=FRAME_DPI, bbox_inches='tight')
    plt.close(fig)


def render_thresh_comparison(gray, frame_i, out_path):
    variants = [
        (99, 50, 'clip=99 gate=50  ← current'),
        (99, 60, 'clip=99 gate=60'),
        (97, 50, 'clip=97 gate=50'),
        (95, 50, 'clip=95 gate=50'),
        (90, 40, 'clip=90 gate=40'),
        (95, 40, 'clip=95 gate=40'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, (cp, gp, lbl) in zip(axes.ravel(), variants):
        ax.imshow(gray, cmap='gray', vmin=0, vmax=1)
        mask = binarize_cine(gray, clip_pctile=cp, gate_pctile=gp)
        if mask is not None and mask.any():
            ov = np.zeros((*gray.shape, 4), dtype=np.float32)
            ov[mask] = [0.0, 0.85, 0.15, 0.45]
            ax.imshow(ov)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=FRAME_DPI, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# VELOCITY  (frame-to-frame Δ=1)
# =============================================================================

def compute_velocity(df: pd.DataFrame, fps: int) -> pd.DataFrame:
    df = df.copy()
    df['arc_tip_vel_mm_s'] = float('nan')
    if PX_PER_MM is None:
        return df
    det = df[df['arc_detected'] == 1].sort_values('frame')
    frames = det['frame'].values.astype(float)
    rows   = det['arc_tip_row'].values.astype(float)
    for k in range(1, len(frames)):
        if int(frames[k]) - int(frames[k-1]) == 1:
            dr_mm = (rows[k] - rows[k-1]) / PX_PER_MM
            vel   = dr_mm * fps * 1000          # mm/s
            df.loc[det.index[k], 'arc_tip_vel_mm_s'] = round(vel, 2)
    return df


# =============================================================================
# MAIN
# =============================================================================

def _analyse_frame(fi, raw_frame, lo, hi, file_stem, fps,
                   out_dir, n_det_ref, qa_thresh_done):
    """Normalise, binarize, and measure one frame. Returns a record dict."""
    gray = np.clip((raw_frame - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    mask     = binarize_cine(gray)
    detected = mask is not None

    rec = {'file': file_stem, 'frame': fi, 'arc_detected': int(detected)}

    if detected:
        n_det_ref[0] += 1
        m = measure_frame(gray, mask, px_per_mm=PX_PER_MM, fast=True)
        tip_row, tip_col = arc_tip(mask)
        complete         = arc_complete_flag(m, gray.shape)
        rec.update({
            'arc_tip_row'      : round(tip_row, 2),
            'arc_tip_col'      : round(tip_col, 2),
            'arc_length_px'    : round(m.arc_length_px, 1),
            'arc_length_mm'    : (round(m.arc_length_mm, 3)
                                  if PX_PER_MM else float('nan')),
            'arc_area_px'      : m.area_px,
            'centroid_row'     : round(m.centroid_row, 2),
            'centroid_col'     : round(m.centroid_col, 2),
            'mean_intensity'   : round(m.mean_intensity, 4),
            'peak_intensity'   : round(m.peak_intensity, 4),
            'tip_a_row'        : round(m.tip_a_row, 2),
            'tip_a_col'        : round(m.tip_a_col, 2),
            'tip_b_row'        : round(m.tip_b_row, 2),
            'tip_b_col'        : round(m.tip_b_col, 2),
            'electrode_gap_px' : round(m.electrode_gap_px, 2),
            'arc_width_px_mean': round(m.arc_width_px_mean, 2),
            'arc_complete'     : int(complete),
            'skeleton_px'      : m.skeleton_n_pixels,
        })
    else:
        tip_row = tip_col = float('nan')

    # QA — threshold grid for first QA_THRESH_N frames (once each)
    if fi not in qa_thresh_done and fi < QA_THRESH_N:
        render_thresh_comparison(
            gray, fi,
            out_dir / 'qa_thresh' / f'thresh_frame_{fi:07d}.png')
        qa_thresh_done.add(fi)

    # QA — annotated frame every QA_EVERY_N
    if fi % QA_EVERY_N == 0:
        render_qa_frame(
            gray, fi, mask if detected else None,
            tip_row, tip_col, detected,
            rec.get('arc_length_mm', float('nan')),
            rec.get('arc_area_px', 0),
            out_dir / 'qa_frames' / f'frame_{fi:07d}.png',
            category='full')

    return rec


def main():
    # ── args ─────────────────────────────────────────────────────────────────
    cine_path = DEFAULT_CINE
    out_dir   = DEFAULT_OUT
    max_gb    = MAX_GB
    positional = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags      = {a.split('=')[0].lstrip('-'): a.split('=')[1]
                  for a in sys.argv[1:] if a.startswith('--') and '=' in a}
    if len(positional) >= 1: cine_path = Path(positional[0])
    if len(positional) >= 2: out_dir   = Path(positional[1])
    if 'max-gb' in flags:    max_gb    = float(flags['max-gb'])

    if not cine_path.exists():
        print(f"[ERROR] Not found: {cine_path}"); sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'qa_frames').mkdir(exist_ok=True)
    (out_dir / 'qa_thresh').mkdir(exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  J-electrode CINE analysis  (lazy / chunked)")
    print(f"  File    : {cine_path.name}  ({cine_path.stat().st_size/1e9:.1f} GB)")
    print(f"  Output  : {out_dir}")
    print(f"  Limit   : {'full file' if max_gb is None else f'{max_gb} GB'}")
    print(f"  Chunk   : {CHUNK_GB} GB")
    print(f"{'='*65}\n")

    # ── Step 1: probe header — fps, bit-depth, frame size ────────────────────
    print("  Probing CINE header ...")
    raw_probe, setup, bpp = read_frames(str(cine_path))
    fps = int(getattr(setup, 'FrameRate', FPS_FALLBACK))

    # Read one frame to get bytes_per_frame, then immediately close the generator
    first_frame = next(iter(raw_probe))
    bytes_per_frame = first_frame.nbytes
    frame_shape     = first_frame.shape

    # Total frames: prefer header field, fall back to file-size estimate
    total_from_header = int(getattr(setup, 'ImageCount',
                            getattr(setup, 'TotalImageCount', 0)))
    if total_from_header > 0:
        total_frames_file = total_from_header
    else:
        total_frames_file = int(cine_path.stat().st_size / bytes_per_frame)

    if max_gb is not None:
        max_frames = int(max_gb * 1e9 / bytes_per_frame)
        total_frames = min(total_frames_file, max_frames)
    else:
        total_frames = total_frames_file

    chunk_frames = max(100, int(CHUNK_GB * 1e9 / bytes_per_frame))

    print(f"    FPS              : {fps:,}")
    print(f"    Bit depth        : {bpp}")
    print(f"    Frame shape      : {frame_shape}")
    print(f"    Bytes / frame    : {bytes_per_frame:,}  "
          f"({bytes_per_frame/1e6:.1f} MB)")
    print(f"    Frames in file   : {total_frames_file:,}")
    print(f"    Frames to process: {total_frames:,}")
    print(f"    Chunk size       : {chunk_frames:,} frames  "
          f"({chunk_frames*bytes_per_frame/1e9:.2f} GB)")
    print(f"    Duration         : {total_frames/fps*1e3:.1f} ms  "
          f"@ {fps/1000:.0f} kfps")

    # ── Step 2: normalization pass — sample N_SAMPLE frames across the file ──
    # We need lo/hi from across the WHOLE recording before analysis begins.
    # Seek directly to the sample frames (via header['pImage']) instead of
    # decoding all 873k frames just to keep 300.
    print(f"\n  Pass 1 / 2 — sampling {N_SAMPLE} frames for normalisation ...")
    from pycine.file import read_header
    header = read_header(str(cine_path))
    sample_indices = np.linspace(0, total_frames - 1, N_SAMPLE, dtype=int).tolist()
    sample_step = sample_indices[1] - sample_indices[0] if len(sample_indices) > 1 else 1
    sample_raw  = []
    for k, frame in enumerate(_read_frames_at(str(cine_path), sample_indices, header)):
        sample_raw.append(_correct_frame(frame))
        if (k + 1) % 50 == 0 or k + 1 == len(sample_indices):
            print(f"    sampled {k+1}/{len(sample_indices)}")

    sample_stack = np.stack(sample_raw)
    del sample_raw
    lo = float(np.percentile(sample_stack, SATURATED / 2))
    hi = float(np.percentile(sample_stack, 100.0 - SATURATED / 2))
    if hi <= lo:
        hi = lo + 1.0
    del sample_stack
    print(f"    lo={lo:.1f}  hi={hi:.1f}  (saturated={SATURATED}%,  "
          f"sampled every ~{sample_step} frames)")

    # Write metadata
    with open(out_dir / 'metadata.txt', 'w') as mf:
        mf.write(f"cine_path         : {cine_path}\n")
        mf.write(f"fps               : {fps}\n")
        mf.write(f"bpp               : {bpp}\n")
        mf.write(f"total_frames      : {total_frames}\n")
        mf.write(f"bytes_per_frame   : {bytes_per_frame}\n")
        mf.write(f"chunk_frames      : {chunk_frames}\n")
        mf.write(f"lo                : {lo}\n")
        mf.write(f"hi                : {hi}\n")
        mf.write(f"SATURATED         : {SATURATED}\n")
        mf.write(f"px_per_mm         : {PX_PER_MM}\n")
        mf.write(f"BRIGHT_CLIP_PCTILE: {BRIGHT_CLIP_PCTILE}\n")
        mf.write(f"OTSU_GATE_PCTILE  : {OTSU_GATE_PCTILE}\n")
        mf.write(f"MIN_FRAME_SIGNAL  : {MIN_FRAME_SIGNAL}\n")

    # ── Step 3: analysis pass — stream through in chunks ─────────────────────
    # Each chunk: read → bad-pixel correct → normalize → binarize → measure.
    # Records are appended to CSV after each chunk so a crash doesn't lose work.
    print(f"\n  Pass 2 / 2 — analysing {total_frames:,} frames in chunks "
          f"of {chunk_frames:,} ...")

    csv_path     = out_dir / 'measurements.csv'
    file_stem    = cine_path.stem
    n_det        = [0]          # mutable so _analyse_frame can increment it
    qa_thresh_done = set()
    first_chunk  = True
    n_chunks     = (total_frames + chunk_frames - 1) // chunk_frames

    raw_gen, _, _ = read_frames(str(cine_path))   # fresh generator for pass 2
    frame_buf  = []
    fi_global  = 0             # global frame index across all chunks
    chunk_idx  = 0

    def flush_chunk(buf, chunk_start):
        nonlocal first_chunk
        stack     = np.stack(buf, axis=0)
        corrected = correct_bad_pixels_stack(stack)
        del stack

        recs = []
        for fi_local, raw_frame in enumerate(corrected):
            fi = chunk_start + fi_local
            rec = _analyse_frame(fi, raw_frame, lo, hi, file_stem, fps,
                                 out_dir, n_det, qa_thresh_done)
            recs.append(rec)

            if fi % 5_000 == 0 and fi > 0:
                pct = 100 * fi / total_frames
                print(f"    Frame {fi:>8,}/{total_frames:,}  "
                      f"({pct:5.1f}%)  detected so far: {n_det[0]:,}")
        del corrected

        chunk_df = pd.DataFrame(recs)
        chunk_df.to_csv(csv_path,
                        mode='w' if first_chunk else 'a',
                        header=first_chunk,
                        index=False)
        first_chunk = False
        return len(recs)

    for frame in raw_gen:
        if fi_global >= total_frames:
            break
        frame_buf.append(frame)

        if len(frame_buf) >= chunk_frames:
            chunk_start = fi_global - len(frame_buf) + 1
            chunk_idx  += 1
            print(f"\n  Chunk {chunk_idx}/{n_chunks}  "
                  f"(frames {chunk_start:,}–{fi_global:,})")
            flush_chunk(frame_buf, chunk_start)
            frame_buf = []

        fi_global += 1

    # Flush the last partial chunk
    if frame_buf:
        chunk_start = fi_global - len(frame_buf)
        chunk_idx  += 1
        print(f"\n  Chunk {chunk_idx}/{n_chunks}  "
              f"(frames {chunk_start:,}–{fi_global-1:,})")
        flush_chunk(frame_buf, chunk_start)
        del frame_buf

    # ── Step 4: post-pass velocity column (requires full CSV in RAM) ─────────
    # Only computed when PX_PER_MM is set; otherwise the column stays nan.
    print("\n  Computing velocity ...")
    df = pd.read_csv(csv_path)
    df = compute_velocity(df, fps)
    df.to_csv(csv_path, index=False)

    det = df[df['arc_detected'] == 1]
    n_processed = len(df)
    n_detected  = n_det[0]
    print(f"\n{'─'*55}")
    print(f"  Done.")
    print(f"  Frames processed : {n_processed:,}")
    print(f"  Detected         : {n_detected:,}  "
          f"({100*n_detected/max(n_processed,1):.1f}%)")
    if len(det):
        print(f"  Arc length mean: {det['arc_length_px'].mean():.1f} px"
              + (f" / {det['arc_length_mm'].mean():.2f} mm" if PX_PER_MM else ''))
        if 'arc_complete' in det.columns:
            nc = int(det['arc_complete'].sum())
            print(f"  Complete arcs    : {nc:,}  ({100*nc/max(len(det),1):.1f}%)")
    print(f"\n  CSV      : {csv_path}")
    print(f"  QA       : {out_dir / 'qa_frames'}")
    print(f"  Thresh QA: {out_dir / 'qa_thresh'}")
    print(f"\n  Next: set PX_PER_MM, then run event analysis:")
    print(f"    python event_velocity.py {csv_path} {out_dir}")
    print(f"{'─'*55}\n")


if __name__ == '__main__':
    main()
