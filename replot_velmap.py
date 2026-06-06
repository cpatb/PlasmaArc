"""
replot_velmap.py  —  Recompute 2-D velocity maps using sliding-window OLS velocity

Paints each arc-mask pixel with the per-frame local_vel_ms produced by
event_velocity.py (sliding-window OLS over LOCAL_VEL_N_FRAMES=10 advancing-
front neighbors).  This preserves genuine within-event velocity gradients
(e.g. deceleration as the arc extends) that are erased by the global
event-regression scalar.

Falls back to event_vel_ms for frames where local_vel_ms is NaN
(non-advancing-front frames, or events too short for the OLS window).

Reads:
  measurements_all_events.csv   (output of event_velocity.py)
  TIFF files                    (to re-binarize each detected frame)

Writes:
  <stem>_velmap_<type>.png / .npy    per run × arc type (full / partial / both)

Usage:
    python replot_velmap.py [measurements_all_events.csv] [output_dir]
"""

import sys, os
import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Pull image processing from plasma_utils so we stay consistent with the pipeline
sys.path.insert(0, str(Path(__file__).parent))
from plasma_utils.image import normalize_frame, binarize
from blade_run2 import PX_PER_MM

# =============================================================================
# CONFIG
# =============================================================================

# Where the TIFF files live (same as blade_run2.py)
import platform
if platform.system() == 'Windows':
    BLADE_DIR = Path.home() / 'OneDrive' / 'Desktop' / 'BLADE'
else:
    BLADE_DIR = Path('/sessions/epic-kind-sagan/mnt/BLADE')

TIFF_FILES = [
    BLADE_DIR / 'ADCB_0.8mmNoz_1000mL_N2_1.tiff',
    BLADE_DIR / 'ADCB_0.8mmNoz_1000mL_N2_2.tiff',
    BLADE_DIR / 'ADCB_0.8mmNoz_1000mL_N2_3.tiff',
]
STEMS = [
    'ADCB_0.8mmNoz_1000mL_N2_1',
    'ADCB_0.8mmNoz_1000mL_N2_2',
    'ADCB_0.8mmNoz_1000mL_N2_3',
]
RUN_LABELS = ['Run 1', 'Run 2', 'Run 3']

# Only use frames whose painting velocity exceeds this (filters un-assigned frames).
# Applied to local_vel_ms when available, otherwise to event_vel_ms.
MIN_VEL_MS    = 0.0    # m/s

# Per-frame mask area bounds (applied after re-binarizing each TIFF frame).
# Must match blade_run2.py / plasma_utils.image pipeline values.
# Full arcs: a few hundred to ~3000 px; noise blobs: < 50 px.
MIN_AREA_PX   = 100    # pixels  — skip sparse / noise detections
MAX_AREA_PX   = 3000   # pixels  — skip electrode-flash / saturated frames

# Only render pixels that were hit by at least this fraction of total qualifying
# frames for the run.  Adapts to dataset size; typically keeps true arc pixels
# (high hit-rate) while masking background junk (low hit-rate).
MIN_HIT_FRAC  = 0.02   # 2 % of qualifying frames
MIN_HIT_COUNT = 5      # absolute floor (for very short recordings)

# =============================================================================
# BLADE LINES  (inverted-V electrode geometry, annotated in napari)
# =============================================================================
# Shape (2, 2, 2): [blade_A, blade_B] × [point_0, point_1] × [row, col]
# Loaded once at import time; None if the file is absent.
_BLADE_LINES_PATH = BLADE_DIR / 'blade_lines.npy'
_BLADE_LINES      = None
if _BLADE_LINES_PATH.exists():
    _BLADE_LINES = np.load(_BLADE_LINES_PATH)
    print(f'[velmap] blade_lines loaded: '
          f'A ({_BLADE_LINES[0,0,0]:.0f},{_BLADE_LINES[0,0,1]:.0f})'
          f'→({_BLADE_LINES[0,1,0]:.0f},{_BLADE_LINES[0,1,1]:.0f})  '
          f'B ({_BLADE_LINES[1,0,0]:.0f},{_BLADE_LINES[1,0,1]:.0f})'
          f'→({_BLADE_LINES[1,1,0]:.0f},{_BLADE_LINES[1,1,1]:.0f})')
else:
    print(f'[velmap] WARNING: blade_lines.npy not found at {_BLADE_LINES_PATH}'
          f' — electrode overlay will be skipped.')

# =============================================================================
# BUILD VELMAP FOR ONE FILE
# =============================================================================

def build_velmap(tiff_path, frames_df):
    """
    Single-pass TIFF read that accumulates velocity into three separate maps:
      full    — frames where arc bridges both electrodes (arc_complete == True)
      partial — all other detected frames (leaders / streamers)
      both    — sum of the above two

    Returns a dict:
        {
          'full':    (vsum, vcnt),
          'partial': (vsum, vcnt),
          'both':    (vsum, vcnt),   ← derived by adding full + partial
        }
    Any accumulator that collected zero frames has (None, None).
    """
    # Resolve painting velocity: prefer per-frame sliding-window OLS (local_vel_ms),
    # fall back to global event regression scalar (event_vel_ms) when NaN.
    frames_df = frames_df.copy()
    if 'local_vel_ms' in frames_df.columns:
        frames_df['_vel'] = pd.to_numeric(
            frames_df['local_vel_ms'], errors='coerce').fillna(
            pd.to_numeric(frames_df['event_vel_ms'], errors='coerce'))
        n_local   = frames_df['local_vel_ms'].notna().sum()
        n_fallback = frames_df['_vel'].notna().sum() - n_local
        print(f"  Velocity source: {n_local:,} local OLS  +  "
              f"{n_fallback:,} event-regression fallback")
    else:
        frames_df['_vel'] = pd.to_numeric(
            frames_df['event_vel_ms'], errors='coerce')
        print(f"  [WARN] 'local_vel_ms' not found — using event_vel_ms for all frames. "
              f"Re-run event_velocity.py to add local_vel_ms.")

    frames_df = frames_df.dropna(subset=['_vel'])
    frames_df = frames_df[frames_df['_vel'] > MIN_VEL_MS]

    # Drop small blobs — these are noise detections, not real arcs
    if 'area_px' in frames_df.columns:
        before = len(frames_df)
        frames_df = frames_df[pd.to_numeric(
            frames_df['area_px'], errors='coerce').fillna(0) >= MIN_AREA_PX]
        print(f"  Area filter (>= {MIN_AREA_PX} px): "
              f"{before} → {len(frames_df)} frames")
    else:
        print(f"  [WARN] 'area_px' column not found — area filter skipped. "
              f"Re-run event_velocity.py to add it.")

    # Build per-frame lookup: frame_index → (velocity_m_s, is_complete)
    has_complete = 'arc_complete' in frames_df.columns
    if not has_complete:
        print(f"  [WARN] 'arc_complete' column not found — "
              f"all frames treated as 'both' only.")

    frame_records = {}
    for _, row in frames_df.iterrows():
        fi  = int(row['frame'])
        vel = float(row['_vel'])
        is_complete = bool(row['arc_complete']) if has_complete else None
        frame_records[fi] = (vel, is_complete)

    if not frame_records:
        print(f"  [WARN] No qualifying frames for {tiff_path.name}")
        return {'full': (None, None), 'partial': (None, None), 'both': (None, None)}

    n_qualifying = len(frame_records)
    print(f"  {n_qualifying:,} frames to process ...")

    # Accumulators — initialised lazily on first valid frame
    acc = {k: {'vsum': None, 'vcnt': None, 'done': 0}
           for k in ('full', 'partial')}
    done_total  = 0
    skipped_area = 0

    with tifffile.TiffFile(tiff_path) as tif:
        for frame_i, (vel, is_complete) in sorted(frame_records.items()):
            raw  = tif.pages[frame_i].asarray()

            # CRITICAL: normalize to float32 [0,1] before binarizing.
            # binarize() runs Otsu on normalized data; passing raw 16-bit
            # values directly produces garbage thresholds and corrupted masks.
            gray = normalize_frame(raw)
            mask = binarize(gray, close_r=3, open_r=0)

            if not mask.any():
                continue

            # Area bounds — identical to the pipeline's _binarize_raw()
            area = int(mask.sum())
            if area < MIN_AREA_PX or area > MAX_AREA_PX:
                skipped_area += 1
                continue

            # Determine which bucket(s) this frame belongs to
            if is_complete is None:
                buckets = []           # no arc_complete column → skip full/partial
            elif is_complete:
                buckets = ['full']
            else:
                buckets = ['partial']

            for key in buckets:
                a = acc[key]
                if a['vsum'] is None:
                    a['vsum'] = np.zeros(raw.shape, np.float64)
                    a['vcnt'] = np.zeros(raw.shape, np.float64)
                a['vsum'][mask] += vel
                a['vcnt'][mask] += 1
                a['done']       += 1

            done_total += 1
            if done_total % 2000 == 0:
                print(f"    {done_total}/{n_qualifying} frames")

    if skipped_area:
        print(f"  Area-bounds filter: {skipped_area} frames skipped "
              f"(outside {MIN_AREA_PX}–{MAX_AREA_PX} px)")

    for key in ('full', 'partial'):
        n = acc[key]['done']
        print(f"  {key:>8}: {n:,} frames accumulated")

    # Build the 'both' accumulator by adding full + partial arrays
    def _sum_pair(a, b):
        vs_a, vc_a = a['vsum'], a['vcnt']
        vs_b, vc_b = b['vsum'], b['vcnt']
        if vs_a is None and vs_b is None:
            return None, None
        if vs_a is None:
            return vs_b.copy(), vc_b.copy()
        if vs_b is None:
            return vs_a.copy(), vc_a.copy()
        return vs_a + vs_b, vc_a + vc_b

    vsum_both, vcnt_both = _sum_pair(acc['full'], acc['partial'])

    return {
        'full':    (acc['full']['vsum'],    acc['full']['vcnt']),
        'partial': (acc['partial']['vsum'], acc['partial']['vcnt']),
        'both':    (vsum_both,              vcnt_both),
    }

# =============================================================================
# PLOT ONE VELMAP
# =============================================================================

def _draw_blade_lines(ax):
    """
    Overlay the napari-annotated inverted-V electrode lines onto a velmap axes.

    blade_lines shape (2, 2, 2): [blade_A, blade_B] × [pt0, pt1] × [row, col]
    Velmap axes: x = col / PX_PER_MM,  y (depth) = row / PX_PER_MM.
    """
    if _BLADE_LINES is None:
        return
    for blade in _BLADE_LINES:
        r0, c0 = blade[0]
        r1, c1 = blade[1]
        x0, d0 = c0 / PX_PER_MM, r0 / PX_PER_MM
        x1, d1 = c1 / PX_PER_MM, r1 / PX_PER_MM
        ax.plot([x0, x1], [d0, d1],
                color='white', linewidth=1.4, linestyle='--',
                alpha=0.85, zorder=5)


def save_velmap(vsum, vcnt, stem, run_label, arc_type, out_dir):
    """
    arc_type : 'full' | 'partial' | 'both'
        Used in the filename and plot title so the three maps are distinct.
    Electrode blade lines (inverted-V geometry) are overlaid from
    blade_lines.npy if the file is present.
    """
    if vsum is None or vcnt is None:
        print(f"  [{arc_type}] No data — skipping.")
        return

    # Adaptive hit-count threshold: a pixel must be hit by at least MIN_HIT_FRAC
    # of all qualifying frames — adapts to recording length.  True arc pixels
    # accumulate hits consistently; sporadic noise blobs do not.
    threshold = max(MIN_HIT_COUNT, int(vcnt.max() * MIN_HIT_FRAC))
    print(f"  [{arc_type}] Hit-count threshold: {threshold} "
          f"({MIN_HIT_FRAC*100:.0f}% of max={int(vcnt.max())})")

    # Use np.divide with 'where' to avoid divide-by-zero RuntimeWarning
    mask = vcnt >= threshold
    vmap = np.divide(vsum, vcnt, where=mask, out=np.full_like(vsum, np.nan))
    vmap[~mask] = np.nan
    vmax = np.nanpercentile(vmap, 97)

    # Convert axes to mm
    H, W   = vmap.shape
    extent = (0, W / PX_PER_MM, H / PX_PER_MM, 0)   # (left, right, bottom, top)

    type_label = {
        'full':    'full (bridging) arcs',
        'partial': 'partial arcs / leaders',
        'both':    'full + partial arcs',
    }.get(arc_type, arc_type)

    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(vmap, cmap='plasma',
                   vmin=0, vmax=max(vmax, 0.1),
                   origin='upper', extent=extent,
                   aspect='equal')
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('Velocity (m/s)')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Depth (mm)')

    # Overlay the napari-annotated inverted-V electrode edges
    _draw_blade_lines(ax)

    fig.tight_layout()

    out = out_dir / f'{stem}_velmap_{arc_type}.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out}")

    # Save raw array for further analysis
    np.save(out_dir / f'{stem}_velmap_{arc_type}.npy', vmap)

# =============================================================================
# MAIN
# =============================================================================

def main():
    script_dir = Path(__file__).parent.resolve()
    csv_path   = (Path(sys.argv[1]) if len(sys.argv) > 1
                  else script_dir / 'measurements_all_events.csv')
    out_dir    = Path(sys.argv[2]) if len(sys.argv) > 2 else csv_path.parent

    if not csv_path.exists():
        print(f"[ERROR] Not found: {csv_path}")
        print("Run event_velocity.py first to generate measurements_all_events.csv")
        sys.exit(1)
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Replot velmaps — sliding-window OLS velocity (N=10)")
    print(f"  CSV    : {csv_path}")
    print(f"  Output : {out_dir}")
    print(f"{'='*55}\n")

    df = pd.read_csv(csv_path)
    df['event_vel_ms'] = pd.to_numeric(df['event_vel_ms'], errors='coerce')
    if 'local_vel_ms' in df.columns:
        df['local_vel_ms'] = pd.to_numeric(df['local_vel_ms'], errors='coerce')
    df['frame'] = pd.to_numeric(df['frame'], errors='coerce')

    for tiff_path, stem, label in zip(TIFF_FILES, STEMS, RUN_LABELS):
        print(f"\n--- {label} ---")
        if not tiff_path.exists():
            print(f"  [SKIP] TIFF not found: {tiff_path}")
            continue

        file_df = df[df['file'] == stem].copy()
        if len(file_df) == 0:
            # Try matching by stem substring in case 'file' column differs
            file_df = df[df['file'].str.contains(stem, na=False)].copy()
        if len(file_df) == 0:
            print(f"  [SKIP] No rows found for {stem}")
            continue

        maps = build_velmap(tiff_path, file_df)
        for arc_type in ('full', 'partial', 'both'):
            vsum, vcnt = maps[arc_type]
            save_velmap(vsum, vcnt, stem, label, arc_type, out_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
