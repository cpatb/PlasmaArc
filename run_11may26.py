"""
run_11may26.py  —  BLADE 11MAY26 analysis pipeline
====================================================

Stage 1  convert    chunked .cine → numbered 8-bit TIFF files
                    Uses Drop Impact's bad-pixel correction and percentile
                    normalisation (same approach as convert_cine in
                    cine_to_tiff.py, but streaming so the whole file is
                    never held in RAM at once).

Stage 2  analyze    analyze_blade.py on every TIFF chunk

Stage 3  merge      concatenate per-chunk measurements.csv files

Stage 4  events     event_velocity.py  →  Mode A / Mode B classification
Stage 5  fourier    fourier_analysis.py → frequency-domain spectra
Stage 6  visualize  visualize_blade.py → paper-ready figures

Usage
-----
    cd "C:\\Users\\compl\\Github\\Plasma Arc"

    python run_11may26.py                        # both files, all stages
    python run_11may26.py --file downward        # one file
    python run_11may26.py --from-stage analyze   # skip conversion
    python run_11may26.py --max-frames 5000      # smoke-test
    python run_11may26.py --from-stage fourier   # re-run fourier + visualize only
    python run_11may26.py --velmap-vmax 5        # cap velocity map at 5 m/s

Outputs (under BLADE 11MAY26 / <stem>_results/)
    tiff/               numbered 8-bit TIFF chunks  (<stem>_0001.tiff …)
    measurements.csv    merged per-frame metrics
    event_summary.csv   one row per arc event, Mode A/B column
"""

import sys, os, subprocess, struct as _struct, argparse, time
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

PLASMA_ARC   = Path(__file__).parent
BLADE_11MAY26 = Path.home() / 'OneDrive' / 'Desktop' / 'BLADE' / 'BLADE 11MAY26'
DROP_IMPACT   = Path(r'C:\Users\compl\Github\Drop Impact')

CINE_FILES = {
    'downward': BLADE_11MAY26 / '0.6mm_1.5SLPM-N2_25deg_OA_downward.cine',
    'upward':   BLADE_11MAY26 / '0.6mm_1.5SLPM-N2_25deg_OA_upward.cine',
}

# ── Calibration ───────────────────────────────────────────────────────────────

PX_PER_MM     = 8.8    # Calibrate/calibration.txt: 44 px = 5 mm
BRIGHT_THRESH = 100    # determined from threshold_qa_11may26.py on 11MAY26 data

# Blade-line geometry is per-recording (the reactor is flipped between downward
# and upward videos, so each needs its own annotation).  The file is stored
# inside each recording's results folder:
#   <stem>_results/blade_lines.npy
# Create it by running:
#   python calibrate_electrodes.py \
#       --tiff "<stem>_results/tiff/<stem>_0001.tiff" \
#       --out  "<stem>_results/blade_lines.npy"
# Until it exists the pipeline falls back to the widest-pair heuristic, but
# arc_complete accuracy will be lower.

# Arc propagation direction per recording.
# 'downward' → arc front is the BOTTOMMOST skeleton pixel (row argmax).
# 'upward'   → arc front is the TOPMOST pixel (row argmin) because the reactor
#               is flipped.  Pass --arc-direction up to analyze_blade.py.
ARC_DIRECTION = {
    'downward': 'down',
    'upward':   'up',
}

# Electrode-gap flash cutoff per recording (pixel row of the minimum electrode
# gap).  Frames whose arc mask crosses past this row into the electrode/flash
# zone are rejected as flashes (analyze_blade.py --gap-row).  None = disabled.
# Read the row off the QA frames / blade_lines geometry for each recording.
# CLI flags --gap-row-down / --gap-row-up override these.
GAP_ROW = {
    'downward': None,
    'upward':   None,
}

# ── Conversion parameters ────────────────────────────────────────────────────
# FRAMES_PER_TIFF controls the conversion chunk size (frames per TIFF file).
# It bounds BOTH the on-disk TIFF size AND the peak RAM during bad-pixel
# correction (which stacks the chunk as uint16 + float32):
#     frames   TIFF(8-bit)   peak RAM (convert)
#     10 000   0.61 GiB      ~7 GiB        <- default, safe on 16 GB machines
#     20 000   1.22 GiB      ~15 GiB
#     50 000   3.05 GiB      ~37 GiB       <- OOMs most machines; near 4 GiB TIFF cap
# Override with --frames-per-tiff. Keep TIFFs < 4 GiB (classic-TIFF hard limit,
# bigtiff=False) and peak RAM under your machine's memory.
FRAMES_PER_TIFF   = 10_000
N_PERCENTILE_SAMPLE = 300    # frames sampled for lo/hi estimation
SATURATED           = 0.35   # % clipped each end (matches PCC / convert_cine default)

# ── Drop Impact imports ───────────────────────────────────────────────────────

if str(DROP_IMPACT) not in sys.path:
    sys.path.insert(0, str(DROP_IMPACT))

from pycine.raw import read_frames, create_raw_array   # type: ignore
from pycine.file import read_header                    # type: ignore
import pycine.raw as _pycine_raw                       # type: ignore
from cine_to_tiff import correct_bad_pixels_stack      # type: ignore
import tifffile

# Monkey-patch: upstream pycine annotation_size MemoryError on large files
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
                    raise IOError(f"Suspicious annotation_size={ann_size} at frame {frame}")
                f.seek(ann_bytes, 1)
            img_size = _struct.unpack("I", f.read(4))[0]
            yield create_raw_array(f.read(img_size), header)
            frame += 1
            count -= 1

_pycine_raw.frame_reader = _patched_frame_reader


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_frames_at(cine_path, frame_indices, header):
    """Yield raw frames at specific 0-based indices (direct seek — no full scan)."""
    with open(str(cine_path), "rb") as f:
        for idx in frame_indices:
            f.seek(header["pImage"][idx])
            ann_size  = _struct.unpack("I", f.read(4))[0]
            ann_bytes = ann_size - 8
            if ann_bytes > 0:
                f.seek(ann_bytes, 1)
            img_size = _struct.unpack("I", f.read(4))[0]
            yield create_raw_array(f.read(img_size), header)


def probe_cine(cine_path):
    """Return (n_frames, fps, H, W)."""
    raw_gen, setup, _ = read_frames(str(cine_path))
    first = next(iter(raw_gen))
    H, W  = first.shape
    fps   = int(getattr(setup, 'FrameRate', 200_000))
    n     = int(getattr(setup, 'ImageCount', getattr(setup, 'TotalImageCount', 0)))
    if n == 0:
        n = int(os.path.getsize(cine_path) / first.nbytes)
    return n, fps, H, W


def run(cmd, label=''):
    print('\n>>> %s' % (label or ' '.join(str(c) for c in cmd)))
    r = subprocess.run([sys.executable] + [str(c) for c in cmd])
    if r.returncode != 0:
        print('[ERROR] failed (exit %d)' % r.returncode)
        sys.exit(r.returncode)


# ── Stage 1: chunked TIFF conversion ─────────────────────────────────────────

def convert_chunked(cine_path, tiff_dir, max_frames=None,
                    frames_per_tiff=FRAMES_PER_TIFF):
    """
    Convert .cine → numbered 8-bit TIFF chunks using the same approach as
    Drop Impact's convert_cine():
      • percentile-based lo/hi from N_PERCENTILE_SAMPLE frames (no hot-pixel bias)
      • correct_bad_pixels_stack per chunk
      • 8-bit = clip((frame - lo) / (hi - lo) * 255, 0, 255)

    frames_per_tiff bounds both the TIFF file size and the peak conversion RAM;
    it MUST match the value passed to merge_csvs so global frame numbering is
    consistent.

    Returns a list of Path objects for the written TIFF files.
    """
    tiff_dir = Path(tiff_dir)
    tiff_dir.mkdir(parents=True, exist_ok=True)

    n_frames, fps, H, W = probe_cine(cine_path)
    if max_frames:
        n_frames = min(n_frames, max_frames)

    print('  Frames  : %d' % n_frames)
    print('  Shape   : %d×%d' % (H, W))
    print('  FPS     : %d' % fps)

    # ── Pass 1: sample frames for percentile normalisation ────────────────────
    # Identical to convert_cine(): sample evenly across the file, correct
    # bad pixels, then compute lo/hi percentiles.
    print('\n  [Pass 1/2]  Sampling %d frames for normalisation ...' % N_PERCENTILE_SAMPLE)
    header = read_header(str(cine_path))
    sample_idx = np.linspace(0, n_frames - 1, N_PERCENTILE_SAMPLE, dtype=int).tolist()
    sample_raw = [f for f in _read_frames_at(cine_path, sample_idx, header)]
    sample_corrected = correct_bad_pixels_stack(np.stack(sample_raw, 0))
    del sample_raw
    lo = float(np.percentile(sample_corrected, SATURATED / 2))
    hi = float(np.percentile(sample_corrected, 100.0 - SATURATED / 2))
    if hi <= lo:
        hi = lo + 1.0
    del sample_corrected
    scale = 255.0 / (hi - lo)
    print('    lo=%.1f  hi=%.1f  scale=%.4f' % (lo, hi, scale))

    # ── Pass 2: stream frames in chunks, write numbered TIFFs ─────────────────
    print('\n  [Pass 2/2]  Converting %d frames in chunks of %d ...' % (
          n_frames, frames_per_tiff))

    raw_gen, _, _ = read_frames(str(cine_path))

    tiff_paths = []
    chunk_buf  = []
    fi         = 0
    chunk_num  = 0
    t0         = time.time()

    def flush(buf, num):
        corrected = correct_bad_pixels_stack(np.stack(buf, 0))
        out_path  = tiff_dir / ('%s_%04d.tiff' % (Path(cine_path).stem, num + 1))
        with tifffile.TiffWriter(str(out_path), bigtiff=False) as tw:
            for frame in corrected:
                frame_8bit = np.clip((frame - lo) * scale, 0, 255).astype(np.uint8)
                tw.write(frame_8bit, photometric='minisblack', contiguous=True)
        del corrected
        tiff_paths.append(out_path)
        elapsed = time.time() - t0
        rate    = (fi + 1) / elapsed if elapsed > 0 else 1
        eta_s   = (n_frames - fi - 1) / rate if rate > 0 else 0
        print('    chunk %d → %s  (%d fr, %.0f fr/s, ETA %.0f min)' % (
              num + 1, out_path.name, len(buf), rate, eta_s / 60))

    for raw16 in raw_gen:
        if fi >= n_frames:
            break
        chunk_buf.append(raw16)
        if len(chunk_buf) >= frames_per_tiff:
            flush(chunk_buf, chunk_num)
            chunk_buf = []
            chunk_num += 1
        fi += 1

    if chunk_buf:
        flush(chunk_buf, chunk_num)

    print('\n  Conversion complete: %d TIFF file(s) in %s' % (
          len(tiff_paths), tiff_dir))
    return tiff_paths


# ── Stage 2: analyze each TIFF chunk ─────────────────────────────────────────

def analyze_chunks(tiff_paths, px_per_mm=PX_PER_MM, arc_direction='down',
                   blade_lines_npy=None, gap_row=None):
    """Run analyze_blade.py on every TIFF, return per-chunk out_dirs.

    Parameters
    ----------
    arc_direction    : 'down' (bottommost pixel = arc front) or
                       'up'   (topmost pixel = arc front; reactor flipped)
    blade_lines_npy  : Path to recording-specific blade_lines.npy, or None
    gap_row          : Electrode-gap cutoff row (px) or None. Frames whose arc
                       mask crosses past it are rejected as flashes.
    """
    out_dirs = []
    for tp in tiff_paths:
        tp       = Path(tp)
        stem     = tp.stem
        out_dir  = tp.parent.parent / (stem + '_results')
        ckpt     = out_dir / 'measurements.csv'
        if ckpt.exists() and ckpt.stat().st_size > 100:
            print('  [skip] %s (measurements.csv exists)' % tp.name)
        else:
            cmd = [PLASMA_ARC / 'analyze_blade.py', str(tp),
                   '--out-dir',       str(out_dir),
                   '--px-per-mm',     str(px_per_mm),
                   '--bright-thresh', str(BRIGHT_THRESH),
                   '--arc-direction', arc_direction]
            if gap_row is not None:
                cmd += ['--gap-row', str(gap_row)]
            if blade_lines_npy and Path(blade_lines_npy).exists():
                cmd += ['--blade-lines', str(blade_lines_npy)]
            else:
                print('  [WARN] blade_lines.npy not found — using widest-pair heuristic')
                print('         Run calibrate_electrodes.py for this recording first.')
            run(cmd, label='analyze_blade.py  %s' % tp.name)
        out_dirs.append(out_dir)
    return out_dirs


# ── Stage 3: merge per-chunk CSVs ────────────────────────────────────────────

def merge_csvs(out_dirs, merged_csv, frames_per_tiff, recording_stem):
    """
    Concatenate per-chunk measurements.csv files into one recording.

    Two normalisations make the chunks behave as a single continuous recording
    downstream (event detection, apex estimation, re-strike/Mode A-B, figures):
      • 'frame'/'time_s' in chunk k are offset by k * frames_per_tiff so frame
        numbers are globally unique and contiguous.
      • 'file' is set to recording_stem for EVERY chunk, so event_velocity.py,
        qa_restrike.py and visualize_blade.py (which all group by 'file') treat
        the chunks as one recording rather than separate "runs".
    """
    import pandas as pd
    dfs = []
    for k, d in enumerate(out_dirs):
        csv = Path(d) / 'measurements.csv'
        if not csv.exists():
            print('  [WARN] missing %s — skipping chunk %d' % (csv, k))
            continue
        df = pd.read_csv(csv)
        df['file']   = recording_stem
        df['frame']  = df['frame']  + k * frames_per_tiff
        df['time_s'] = df['time_s'] + k * frames_per_tiff / 200_000
        dfs.append(df)
    if not dfs:
        print('[ERROR] no chunk CSVs found to merge'); sys.exit(1)
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(merged_csv, index=False)
    print('  Merged %d chunks → %s  (%d frames)' % (
          len(dfs), merged_csv, len(merged)))


# ── Main ──────────────────────────────────────────────────────────────────────

STAGES = ['convert', 'analyze', 'merge', 'events', 'fourier', 'visualize']

def main():
    ap = argparse.ArgumentParser(
        description='BLADE 11MAY26 full pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--file', choices=['downward', 'upward', 'both'],
                    default='both')
    ap.add_argument('--from-stage', choices=STAGES, default='convert',
                    help='Resume from this stage (earlier stages are skipped)')
    ap.add_argument('--max-frames', type=int, default=None,
                    help='Process only the first N frames (smoke-test)')
    ap.add_argument('--px-per-mm', type=float, default=PX_PER_MM)
    ap.add_argument('--frames-per-tiff', type=int, default=FRAMES_PER_TIFF,
                    help='Frames per TIFF chunk written by the conversion stage')
    # Re-strike Mode A/B characterization (runs automatically in Stage 4).
    ap.add_argument('--restrike-method', choices=['abs', 'rel'], default='abs',
                    help='Mode A/B cut: abs=norm_pos, rel=retreat fraction')
    ap.add_argument('--restrike-threshold', type=float, default=0.10,
                    help='abs method: norm_pos above which a re-strike is Mode B')
    ap.add_argument('--restrike-prominence-frac', type=float, default=0.10,
                    help='min extension-peak prominence (fraction of L_eff); lower '
                         'to detect more/shallower cycles')
    ap.add_argument('--velmap-vmax', type=float, default=None,
                    help='Upper velocity bound for 2-D velocity map (m/s). '
                         'Default: auto (95th percentile of arc body, excluding '
                         'electrode-gap rows).  Pass e.g. 5 to suppress flash-arc '
                         'noise at the electrode top.')
    ap.add_argument('--gap-row-down', type=int, default=GAP_ROW['downward'],
                    help='Electrode-gap cutoff row for the downward recording '
                         '(analyze_blade.py --gap-row). Overrides GAP_ROW["downward"].')
    ap.add_argument('--gap-row-up', type=int, default=GAP_ROW['upward'],
                    help='Electrode-gap cutoff row for the upward recording. '
                         'Overrides GAP_ROW["upward"].')
    args = ap.parse_args()

    files = list(CINE_FILES.items()) if args.file == 'both' \
            else [(args.file, CINE_FILES[args.file])]

    from_idx = STAGES.index(args.from_stage)
    def active(stage): return STAGES.index(stage) >= from_idx

    for name, cine_path in files:
        if not cine_path.exists():
            print('[SKIP] not found: %s' % cine_path); continue

        stem     = cine_path.stem
        out_root = BLADE_11MAY26 / (stem + '_results')
        tiff_dir = out_root / 'tiff'
        out_root.mkdir(parents=True, exist_ok=True)

        print('\n' + '=' * 60)
        print('  File : %s' % cine_path.name)
        print('  Out  : %s' % out_root)
        print('=' * 60)

        # ── Stage 1: convert ─────────────────────────────────────────
        if active('convert'):
            print('\n[STAGE 1]  Convert .cine → TIFF chunks')
            tiff_paths = convert_chunked(cine_path, tiff_dir,
                                         max_frames=args.max_frames,
                                         frames_per_tiff=args.frames_per_tiff)
        else:
            tiff_paths = sorted(tiff_dir.glob('*.tiff'))
            print('[skip] convert  (%d TIFFs already in %s)' % (
                  len(tiff_paths), tiff_dir))

        if not tiff_paths:
            print('[ERROR] no TIFF files found in %s' % tiff_dir); sys.exit(1)

        # ── Stage 2: analyze ─────────────────────────────────────────
        arc_dir       = ARC_DIRECTION.get(name, 'down')
        blade_lines   = out_root / 'blade_lines.npy'
        if active('analyze'):
            print('\n[STAGE 2]  Analyze TIFF chunks')
            print('  Arc direction : %s' % arc_dir)
            # Auto-generate blade_lines.npy (headless) if it does not exist, so
            # arc_complete uses the perpendicular-distance-to-blade-line test
            # rather than the weaker widest-pair fallback. The interactive
            # calibrate_electrodes.py can still be used to override it.
            if not blade_lines.exists():
                print('  Blade lines   : not found — auto-detecting from first chunk')
                run([PLASMA_ARC / 'detect_blade_lines.py', str(tiff_paths[0]),
                     '--arc-direction', arc_dir,
                     '--thresh', str(BRIGHT_THRESH),
                     '--px-per-mm', str(args.px_per_mm),
                     '--out', str(blade_lines)],
                    label='detect_blade_lines.py  %s' % Path(tiff_paths[0]).name)
            if blade_lines.exists():
                print('  Blade lines   : %s' % blade_lines)
            else:
                print('  Blade lines   : auto-detection failed — falling back to '
                      'widest-pair heuristic (lower arc_complete accuracy).')
            gap_row = args.gap_row_down if name == 'downward' else args.gap_row_up
            if gap_row is not None:
                print('  Gap-row cutoff: %d px (frames crossing past it rejected as flashes)' % gap_row)
            chunk_out_dirs = analyze_chunks(tiff_paths,
                                            px_per_mm=args.px_per_mm,
                                            arc_direction=arc_dir,
                                            blade_lines_npy=blade_lines,
                                            gap_row=gap_row)
        else:
            chunk_out_dirs = [tiff_dir.parent / (Path(tp).stem + '_results')
                              for tp in tiff_paths]
            print('[skip] analyze')

        # ── Stage 3: merge ───────────────────────────────────────────
        merged_csv = out_root / 'measurements.csv'
        if active('merge'):
            print('\n[STAGE 3]  Merge per-chunk CSVs')
            merge_csvs(chunk_out_dirs, merged_csv, args.frames_per_tiff, stem)
        else:
            print('[skip] merge  (%s)' % merged_csv)

        if not merged_csv.exists():
            print('[ERROR] merged CSV missing: %s' % merged_csv); sys.exit(1)

        # ── Stage 4: events + Mode A/B ───────────────────────────────
        if active('events'):
            print('\n[STAGE 4]  Event detection + velocity (+ legacy per-event mode)')
            run([PLASMA_ARC / 'event_velocity.py',
                 str(merged_csv), str(out_root),
                 '--px-per-mm', str(args.px_per_mm)],
                label='event_velocity.py')

            # Stage 4b: trustworthy, non-circular Mode A/B from re-strikes
            # (minima of the arc-tip trajectory). This is the authoritative
            # Mode A/B record (re_strikes_qa.csv); the per-event 'mode' column
            # in event_summary.csv is the legacy circular classifier.
            event_csv = out_root / 'measurements_all_events.csv'
            if event_csv.exists():
                run([PLASMA_ARC / 'qa_restrike.py',
                     '--csv', str(event_csv),
                     '--tiff-dir', str(tiff_dir),
                     '--frames-per-tiff', str(args.frames_per_tiff),
                     '--arc-direction', arc_dir,
                     '--px-per-mm', str(args.px_per_mm),
                     '--method', args.restrike_method,
                     '--threshold', str(args.restrike_threshold),
                     '--ext-prominence-frac', str(args.restrike_prominence_frac),
                     '--out', str(out_root / 'qa_restrike.png')],
                    label='qa_restrike.py (Mode A/B re-strike characterization)')
            else:
                print('  [skip] re-strike Mode A/B — events CSV missing')

        # ── Stage 5: Fourier analysis ─────────────────────────────
        if active('fourier'):
            print('\n[STAGE 5]  Fourier analysis')
            fourier_csv = out_root / 'measurements_all_events.csv'
            if not fourier_csv.exists():
                fourier_csv = merged_csv
            run([PLASMA_ARC / 'fourier_analysis.py',
                 str(fourier_csv), str(out_root),
                 '--px-per-mm', str(args.px_per_mm)],
                label='fourier_analysis.py')

        # ── Stage 6: visualization ────────────────────────────────
        if active('visualize'):
            print('\n[STAGE 6]  Visualization')
            event_csv = out_root / 'measurements_all_events.csv'
            viz_csv   = event_csv if event_csv.exists() else merged_csv

            # 6a. Rebuild the 2-D velocity map from the LOCAL (time-resolved)
            # velocity — the physically meaningful map. Loops over ALL chunks in
            # the tiff dir (global frame -> chunk/page via --frames-per-tiff) so
            # the whole recording is covered. The central-difference map from
            # Stage 2 is retained for comparison.
            if event_csv.exists():
                run([PLASMA_ARC / 'build_velocity_map.py',
                     '--csv', str(event_csv),
                     '--tiff-dir', str(tiff_dir),
                     '--frames-per-tiff', str(args.frames_per_tiff),
                     '--out-dir', str(out_root),
                     '--px-per-mm', str(args.px_per_mm)],
                    label='build_velocity_map.py (local-velocity 2-D map)')
            else:
                print('  [skip] local velocity map — no events CSV (run Stage 4 first)')

            # 6b. Paper figures (velmap figure now prefers velocity_map_local.npy)
            viz_cmd = [PLASMA_ARC / 'visualize_blade.py',
                       str(viz_csv), str(out_root),
                       '--px-per-mm', str(args.px_per_mm)]
            if args.velmap_vmax is not None:
                viz_cmd += ['--velmap-vmax', str(args.velmap_vmax)]
            run(viz_cmd, label='visualize_blade.py')

        print('\nDone.  Results: %s' % out_root)


if __name__ == '__main__':
    main()
