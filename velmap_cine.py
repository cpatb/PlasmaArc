"""
velmap_cine.py  --  2-D velocity map for J-electrode CINE recordings

Equivalent of replot_velmap.py but reads frames from a .cine file instead
of TIFF stacks, and uses the J-electrode binarization from analyze_cine.py.

All spatial axes are in PIXELS (no px/mm calibration assumed).
Velocity color scale uses the same m/s values written by event_velocity.py
(which inherits BLADE's px/mm calibration until J-electrode is calibrated;
the spatial pattern is correct regardless).

Reads:
  measurements_events.csv   (output of event_velocity.py)
  metadata.txt              (written by analyze_cine.py -- lo/hi/fps/cine_path)

Writes:
  velmap_full.png / .npy    -- frames where arc bridges both electrodes
  velmap_partial.png / .npy -- all other detected frames (leaders/streamers)
  velmap_both.png / .npy    -- full + partial combined

Usage:
    python velmap_cine.py [measurements_events.csv] [output_dir]
"""

import sys
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# =============================================================================
# IMPORT CINE INFRASTRUCTURE FROM analyze_cine.py
# =============================================================================
# We load analyze_cine as a module so we get binarize_cine(), _read_frames_at(),
# and _correct_frame() without duplicating the pycine monkey-patch logic.

def _load_analyze_cine():
    script_dir = Path(__file__).parent
    ac_path    = script_dir / 'analyze_cine.py'
    if not ac_path.exists():
        raise FileNotFoundError(f"analyze_cine.py not found at {ac_path}")
    spec = importlib.util.spec_from_file_location('analyze_cine', ac_path)
    ac   = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ac)
    return ac

_ac             = _load_analyze_cine()
binarize_cine   = _ac.binarize_cine
_read_frames_at = _ac._read_frames_at
_correct_frame  = _ac._correct_frame

# =============================================================================
# CONFIG
# =============================================================================

# Area bounds matching analyze_cine.py (J-electrode values)
MIN_AREA_PX   =      50
MAX_AREA_PX   = 150_000

# Only paint pixels whose velocity exceeds this floor (filters unassigned frames)
MIN_VEL_MS    = 0.0

# Hit-count threshold for the final map (same adaptive logic as replot_velmap.py)
MIN_HIT_FRAC  = 0.02   # 2% of max hit count
MIN_HIT_COUNT = 5      # absolute floor

# =============================================================================
# READ metadata.txt
# =============================================================================

def read_metadata(meta_path):
    """
    Parse the key:value metadata.txt written by analyze_cine.py.
    Returns a dict of string values; caller casts as needed.
    """
    meta = {}
    with open(meta_path) as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                meta[k.strip()] = v.strip()
    return meta

# =============================================================================
# BUILD VELMAP
# =============================================================================

def build_velmap(cine_path, lo, hi, frames_df):
    """
    Stream through the CINE file for qualifying frames and accumulate
    per-pixel velocity into three maps: full / partial / both.

    Parameters
    ----------
    cine_path : Path or str
    lo, hi    : float   -- global normalisation limits from analyze_cine Pass 1
    frames_df : DataFrame -- rows from measurements_events.csv with valid velocity

    Returns
    -------
    dict  {  'full':    (vsum | None, vcnt | None),
             'partial': (vsum | None, vcnt | None),
             'both':    (vsum | None, vcnt | None)  }
    """
    from pycine.file import read_header

    cine_path = Path(cine_path)

    # --- Resolve painting velocity -------------------------------------------
    frames_df = frames_df.copy()
    if 'local_vel_ms' in frames_df.columns:
        frames_df['_vel'] = pd.to_numeric(
            frames_df['local_vel_ms'], errors='coerce').fillna(
            pd.to_numeric(frames_df['event_vel_ms'], errors='coerce'))
        n_local    = frames_df['local_vel_ms'].notna().sum()
        n_fallback = frames_df['_vel'].notna().sum() - n_local
        print(f"  Velocity source: {n_local:,} local OLS  +  "
              f"{n_fallback:,} event-regression fallback")
    else:
        frames_df['_vel'] = pd.to_numeric(
            frames_df['event_vel_ms'], errors='coerce')
        print("  [WARN] 'local_vel_ms' not found -- using event_vel_ms for all frames.")

    frames_df = frames_df.dropna(subset=['_vel'])
    frames_df = frames_df[frames_df['_vel'] > MIN_VEL_MS]

    # Area filter on arc_area_px (J-electrode column name)
    if 'arc_area_px' in frames_df.columns:
        area_col = 'arc_area_px'
    elif 'area_px' in frames_df.columns:
        area_col = 'area_px'
    else:
        area_col = None

    if area_col:
        before = len(frames_df)
        frames_df = frames_df[
            pd.to_numeric(frames_df[area_col], errors='coerce')
            .fillna(0)
            .between(MIN_AREA_PX, MAX_AREA_PX)]
        print(f"  Area filter ({MIN_AREA_PX}-{MAX_AREA_PX} px): "
              f"{before} -> {len(frames_df)} frames")
    else:
        print("  [WARN] No area column found -- area filter skipped.")

    has_complete = 'arc_complete' in frames_df.columns
    if not has_complete:
        print("  [WARN] 'arc_complete' not found -- all frames treated as 'both' only.")

    # Build lookup: frame_index -> (velocity, is_complete)
    frame_records = {}
    for _, row in frames_df.iterrows():
        fi  = int(row['frame'])
        vel = float(row['_vel'])
        is_complete = bool(row['arc_complete']) if has_complete else None
        frame_records[fi] = (vel, is_complete)

    if not frame_records:
        print("  [WARN] No qualifying frames.")
        return {'full': (None, None), 'partial': (None, None), 'both': (None, None)}

    n_qualifying = len(frame_records)
    print(f"  {n_qualifying:,} qualifying frames -- reading CINE ...")

    # --- Stream through CINE at qualifying frame indices only ----------------
    header        = read_header(str(cine_path))
    sorted_indices = sorted(frame_records.keys())

    acc = {k: {'vsum': None, 'vcnt': None, 'done': 0}
           for k in ('full', 'partial')}
    done_total   = 0
    skipped_area = 0

    for raw_frame, fi in zip(
            _read_frames_at(str(cine_path), sorted_indices, header),
            sorted_indices):

        vel, is_complete = frame_records[fi]

        # Correct bad pixels, normalise to float32 [0, 1]
        corrected = _correct_frame(raw_frame)
        gray = np.clip((corrected - lo) / max(hi - lo, 1.0),
                       0.0, 1.0).astype(np.float32)

        mask = binarize_cine(gray)
        if mask is None or not mask.any():
            continue

        # Re-check area after live binarization (the CSV area may differ slightly
        # due to post-processing; this keeps the velmap consistent with the live mask)
        area = int(mask.sum())
        if area < MIN_AREA_PX or area > MAX_AREA_PX:
            skipped_area += 1
            continue

        # Bucket assignment
        if is_complete is None:
            buckets = []
        elif is_complete:
            buckets = ['full']
        else:
            buckets = ['partial']

        for key in buckets:
            a = acc[key]
            if a['vsum'] is None:
                a['vsum'] = np.zeros(gray.shape, np.float64)
                a['vcnt'] = np.zeros(gray.shape, np.float64)
            a['vsum'][mask] += vel
            a['vcnt'][mask] += 1
            a['done']       += 1

        done_total += 1
        if done_total % 1000 == 0:
            print(f"    {done_total}/{n_qualifying} frames processed")

    if skipped_area:
        print(f"  Live area filter: {skipped_area} frames re-skipped "
              f"(mask outside {MIN_AREA_PX}-{MAX_AREA_PX} px)")

    for key in ('full', 'partial'):
        print(f"  {key:>8}: {acc[key]['done']:,} frames accumulated")

    # Combine full + partial into 'both'
    def _add(a, b):
        va, ca = a['vsum'], a['vcnt']
        vb, cb = b['vsum'], b['vcnt']
        if va is None and vb is None: return None, None
        if va is None: return vb.copy(), cb.copy()
        if vb is None: return va.copy(), ca.copy()
        return va + vb, ca + cb

    vsum_both, vcnt_both = _add(acc['full'], acc['partial'])

    return {
        'full':    (acc['full']['vsum'],    acc['full']['vcnt']),
        'partial': (acc['partial']['vsum'], acc['partial']['vcnt']),
        'both':    (vsum_both,              vcnt_both),
    }

# =============================================================================
# SAVE ONE VELMAP
# =============================================================================

def save_velmap(vsum, vcnt, stem, arc_type, out_dir):
    """
    Render and save one velocity map.

    Spatial axes are in PIXELS (no px/mm calibration).
    arc_type : 'full' | 'partial' | 'both'
    """
    if vsum is None or vcnt is None:
        print(f"  [{arc_type}] No data -- skipping.")
        return

    threshold = max(MIN_HIT_COUNT, int(vcnt.max() * MIN_HIT_FRAC))
    print(f"  [{arc_type}] Hit-count threshold: {threshold} "
          f"({MIN_HIT_FRAC*100:.0f}% of max={int(vcnt.max())})")

    mask = vcnt >= threshold
    vmap = np.divide(vsum, vcnt, where=mask, out=np.full_like(vsum, np.nan))
    vmap[~mask] = np.nan
    vmax = np.nanpercentile(vmap, 97)

    H, W = vmap.shape

    type_label = {
        'full':    'full (bridging) arcs',
        'partial': 'partial arcs / leaders',
        'both':    'full + partial arcs',
    }.get(arc_type, arc_type)

    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(vmap, cmap='plasma',
                   vmin=0, vmax=max(vmax, 0.1),
                   origin='upper',
                   extent=(0, W, H, 0),
                   aspect='equal')
    cb = plt.colorbar(im, ax=ax)
    cb.set_label('Velocity (m/s)*')
    ax.set_xlabel('x (px)')
    ax.set_ylabel('Depth (px)')
    ax.text(0.01, 0.01,
            '* velocity scale uses BLADE px/mm calibration\n'
            '  until J-electrode calibration is set',
            transform=ax.transAxes,
            fontsize=6, color='white', alpha=0.7,
            verticalalignment='bottom')

    fig.tight_layout()

    png_out = out_dir / f'{stem}_velmap_{arc_type}.png'
    npy_out = out_dir / f'{stem}_velmap_{arc_type}.npy'
    fig.savefig(png_out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    np.save(npy_out, vmap)
    print(f"  Saved: {png_out.name}  ({png_out.stat().st_size/1024:.0f} kB)")

# =============================================================================
# MAIN
# =============================================================================

def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    out_dir  = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if csv_path is None or not csv_path.exists():
        print("[ERROR] Usage: python velmap_cine.py measurements_events.csv output_dir")
        print("        measurements_events.csv is produced by event_velocity.py")
        sys.exit(1)

    if out_dir is None:
        out_dir = csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Locate metadata.txt (written by analyze_cine.py next to measurements.csv) -
    meta_path = out_dir / 'metadata.txt'
    if not meta_path.exists():
        # Also try sibling of the events CSV
        meta_path = csv_path.parent / 'metadata.txt'
    if not meta_path.exists():
        print(f"[ERROR] metadata.txt not found in {out_dir} or {csv_path.parent}")
        print("        Run analyze_cine.py first to generate it.")
        sys.exit(1)

    meta = read_metadata(meta_path)
    lo   = float(meta.get('lo', 0))
    hi   = float(meta.get('hi', 65535))

    # Locate CINE file
    cine_path_str = meta.get('cine_path', '')
    cine_path     = Path(cine_path_str) if cine_path_str else None
    if cine_path is None or not cine_path.exists():
        # Fall back: search for *.cine in the J-electrode folder
        candidates = list(out_dir.parent.glob('*.cine'))
        if not candidates:
            candidates = list(csv_path.parent.parent.glob('*.cine'))
        if candidates:
            cine_path = candidates[0]
            print(f"  [INFO] metadata.txt cine_path not found; "
                  f"using auto-discovered: {cine_path.name}")
        else:
            print(f"[ERROR] CINE file not found.  "
                  f"Expected: {cine_path_str or '(not set in metadata.txt)'}")
            sys.exit(1)

    stem = cine_path.stem

    print(f"\n{'='*55}")
    print(f"  CINE velmap  --  J electrode")
    print(f"  CINE   : {cine_path.name}")
    print(f"  CSV    : {csv_path.name}")
    print(f"  Output : {out_dir}")
    print(f"  lo={lo:.1f}  hi={hi:.1f}  (from metadata.txt)")
    print(f"  Axes in PIXELS (no px/mm calibration)")
    print(f"{'='*55}\n")

    df = pd.read_csv(csv_path)
    df['event_vel_ms'] = pd.to_numeric(df['event_vel_ms'], errors='coerce')
    if 'local_vel_ms' in df.columns:
        df['local_vel_ms'] = pd.to_numeric(df['local_vel_ms'], errors='coerce')
    df['frame'] = pd.to_numeric(df['frame'], errors='coerce').dropna()

    # Filter to detected frames only
    if 'arc_detected' in df.columns:
        df = df[pd.to_numeric(df['arc_detected'], errors='coerce') == 1]

    # The J electrode has only one file in the CSV; if 'file' column is present
    # use the stem to filter, otherwise use all rows
    if 'file' in df.columns:
        file_df = df[df['file'].str.contains(stem, na=False, regex=False)]
        if len(file_df) == 0:
            print(f"  [INFO] No exact stem match for '{stem}' in 'file' column; "
                  f"using all {len(df)} rows.")
            file_df = df
    else:
        file_df = df

    print(f"  {len(file_df):,} rows after filtering to stem '{stem}'")

    maps = build_velmap(cine_path, lo, hi, file_df)

    print()
    for arc_type in ('full', 'partial', 'both'):
        vsum, vcnt = maps[arc_type]
        save_velmap(vsum, vcnt, stem, arc_type, out_dir)

    print("\nDone.")


if __name__ == '__main__':
    main()
