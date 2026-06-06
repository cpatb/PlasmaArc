"""
qa_frames.py  —  Frame-level binarisation and tip-detection QA

Samples frames uniformly from each TIFF and from the detected-arc frames in
measurements_all.csv, then renders annotated images so you can verify that:

  • The arc mask covers the right region (not background noise, not missing arc)
  • The yellow tip marker lands at the correct arc-tip pixel
  • The 'arc_complete' flag is plausible (bridging vs leader)
  • Dark / non-detected frames are correctly rejected

Output structure:
  qa_frames/
    Run1_uniform/          — uniformly sampled frames across the full recording
      frame_NNNNNNN.png
    Run1_detected/         — randomly sampled arc-detected frames
      frame_NNNNNNN.png
    Run1_boundaries/       — frames near arc-on / arc-off transitions
      frame_NNNNNNN.png
    ...  (same for Run2, Run3)
    qa_frames_summary.csv  — one row per rendered frame with key metrics

Usage:
    python qa_frames.py [measurements_all.csv] [out_dir]
"""

import sys
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from plasma_utils.image import normalize_frame, binarize
from blade_run2 import PX_PER_MM

import platform
if platform.system() == 'Windows':
    BLADE_DIR = Path.home() / 'OneDrive' / 'Desktop' / 'BLADE'
else:
    BLADE_DIR = Path('/sessions/epic-kind-sagan/mnt/BLADE')

TIFF_MAP = {
    'ADCB_0.8mmNoz_1000mL_N2_1': BLADE_DIR / 'ADCB_0.8mmNoz_1000mL_N2_1.tiff',
    'ADCB_0.8mmNoz_1000mL_N2_2': BLADE_DIR / 'ADCB_0.8mmNoz_1000mL_N2_2.tiff',
    'ADCB_0.8mmNoz_1000mL_N2_3': BLADE_DIR / 'ADCB_0.8mmNoz_1000mL_N2_3.tiff',
}

# =============================================================================
# CONFIG
# =============================================================================

N_UNIFORM    = 40    # uniformly-spaced frames from the full recording
N_DETECTED   = 30    # randomly-sampled arc-detected frames
N_BOUNDARY   = 10    # arc-on/off transition pairs (N_BOUNDARY each side)
N_BEFORE_ARC = 2     # extra frames before first detected arc in a transition
N_AFTER_ARC  = 2     # extra frames after last detected arc in a transition
FRAME_DPI    = 110

MIN_AREA_PX  = 5
MAX_AREA_PX  = 10_000

# =============================================================================
# HELPERS
# =============================================================================

def safe_binarize(raw):
    gray = normalize_frame(raw)
    mask = binarize(gray, close_r=3, open_r=0)
    area = int(mask.sum())
    return mask if MIN_AREA_PX <= area <= MAX_AREA_PX else None


def render_qa_frame(raw, frame_i, csv_row, out_path, category):
    """
    Two-panel image: raw frame (left) and arc-mask overlay + tip (right).
    csv_row may be None for non-detected / context frames.
    """
    gray = normalize_frame(raw)
    mask = safe_binarize(raw)

    detected     = bool(csv_row['arc_detected']) if csv_row is not None else False
    complete     = bool(csv_row.get('arc_complete', False)) if detected else False
    tip_r        = float(csv_row['arc_tip_row']) if detected else float('nan')
    tip_c        = float(csv_row['arc_tip_col']) if detected else float('nan')
    area_px      = int(float(csv_row.get('area_px', 0) or 0)) if detected else 0
    arc_len_mm   = (float(csv_row.get('arc_length_mm', float('nan')) or float('nan'))
                    if csv_row is not None else float('nan'))

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4),
                             gridspec_kw={'wspace': 0.04})
    for ax in axes:
        ax.imshow(gray, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')

    # Overlay on right panel
    if mask is not None and mask.any():
        ov = np.zeros((*gray.shape, 4), dtype=np.float32)
        if detected:
            ov[mask] = [0.0, 0.85, 0.15, 0.45]   # green — arc present
        else:
            ov[mask] = [1.0, 0.5, 0.0, 0.40]      # orange — binarizer fires but CSV says no arc
        axes[1].imshow(ov)

    if detected and not (np.isnan(tip_r) or np.isnan(tip_c)):
        for ax in axes:
            ax.plot(tip_c, tip_r, '+',
                    color='yellow', markersize=16, markeredgewidth=2)
        # Mark electrode tips if available
        for tip_key in [('tip_a_col', 'tip_a_row'), ('tip_b_col', 'tip_b_row')]:
            tc_k, tr_k = tip_key
            if csv_row is not None and tc_k in csv_row:
                tc_v = float(csv_row.get(tc_k, float('nan')) or float('nan'))
                tr_v = float(csv_row.get(tr_k, float('nan')) or float('nan'))
                if not (np.isnan(tc_v) or np.isnan(tr_v)):
                    axes[1].plot(tc_v, tr_v, 'cx',
                                markersize=10, markeredgewidth=1.5)


    det_str  = 'DETECTED' if detected else 'NOT DETECTED'
    comp_str = ('  complete' if complete else '  partial') if detected else ''
    arc_str  = (f'  area={area_px}px  len={arc_len_mm:.1f}mm'
                if detected else '')
    title_color = '#00C853' if detected else '#757575'

    fig.savefig(out_path, dpi=FRAME_DPI, bbox_inches='tight')
    plt.close(fig)

    return {
        'category'    : category,
        'frame'       : frame_i,
        'detected'    : int(detected),
        'arc_complete': int(complete),
        'area_px'     : area_px,
        'arc_len_mm'  : arc_len_mm,
        'tip_row'     : tip_r,
        'tip_col'     : tip_c,
        'output_file' : str(out_path),
    }

# =============================================================================
# FRAME SELECTION
# =============================================================================

def select_frames(file_df, n_frames_tiff):
    """
    Return dict of category → sorted list of frame indices.
    """
    selected = {}

    # 1. Uniform sample across the full recording
    uniform = np.round(np.linspace(0, n_frames_tiff - 1, N_UNIFORM)).astype(int)
    selected['uniform'] = sorted(set(uniform.tolist()))

    # 2. Random sample from detected frames
    det = file_df[file_df['arc_detected'] == 1]['frame'].dropna().astype(int)
    if len(det) > 0:
        rng   = np.random.default_rng(0)
        picks = rng.choice(det.values, size=min(N_DETECTED, len(det)),
                           replace=False)
        selected['detected'] = sorted(picks.tolist())
    else:
        selected['detected'] = []

    # 3. Arc-on / arc-off boundaries (transitions)
    all_det = file_df[['frame', 'arc_detected']].copy()
    all_det = all_det.dropna(subset=['frame']).sort_values('frame')
    all_det['arc_detected'] = pd.to_numeric(
        all_det['arc_detected'], errors='coerce').fillna(0).astype(int)

    transitions = all_det[all_det['arc_detected'].diff().abs() == 1]['frame'].astype(int)
    n_trans     = min(N_BOUNDARY, len(transitions))
    if n_trans > 0:
        rng_t  = np.random.default_rng(7)
        t_pick = rng_t.choice(transitions.values, size=n_trans, replace=False)
        boundary_frames = set()
        for t in t_pick:
            for offset in range(-N_BEFORE_ARC, N_AFTER_ARC + 1):
                f = int(t) + offset
                if 0 <= f < n_frames_tiff:
                    boundary_frames.add(f)
        selected['boundaries'] = sorted(boundary_frames)
    else:
        selected['boundaries'] = []

    return selected

# =============================================================================
# PER-FILE QA
# =============================================================================

def qa_file(stem, run_label, file_df, out_root):
    tiff_path = TIFF_MAP.get(stem)
    if tiff_path is None or not tiff_path.exists():
        print(f"  [SKIP] TIFF not found: {stem}")
        return []

    # Frame-number → CSV row lookup (only detected rows for efficiency)
    frame_lookup = (file_df.set_index('frame')
                           .to_dict(orient='index'))

    with tifffile.TiffFile(tiff_path) as tif:
        n_pages = len(tif.pages)
        print(f"  {run_label}: {n_pages:,} total frames")

        categories = select_frames(file_df, n_pages)
        summary_rows = []

        for category, frame_list in categories.items():
            cat_dir = out_root / f'{run_label}_{category}'
            cat_dir.mkdir(parents=True, exist_ok=True)
            print(f"    {category}: {len(frame_list)} frames")

            for fi in frame_list:
                raw    = tif.pages[fi].asarray()
                row    = frame_lookup.get(fi)

                # Some context frames may not be in the CSV (non-detected)
                if row is not None and int(row.get('arc_detected', 0)) == 0:
                    row = None   # treat as non-detected for rendering

                out_fp = cat_dir / f'frame_{fi:07d}.png'
                result = render_qa_frame(raw, fi, row, out_fp, category)
                result['run'] = run_label
                summary_rows.append(result)

    return summary_rows

# =============================================================================
# MAIN
# =============================================================================

def main():
    script_dir = Path(__file__).parent.resolve()

    csv_path = (Path(sys.argv[1]) if len(sys.argv) > 1
                else script_dir / 'measurements_all.csv')
    out_root = (Path(sys.argv[2]) if len(sys.argv) > 2
                else script_dir / 'qa_frames')

    if not csv_path.exists():
        print(f"[ERROR] Not found: {csv_path}"); sys.exit(1)

    out_root.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Frame-level QA  —  binarisation and tip-detection check")
    print(f"  CSV      : {csv_path}")
    print(f"  Output   : {out_root}")
    print(f"{'='*60}")
    print(f"  Uniform: {N_UNIFORM} frames / run")
    print(f"  Detected sample: {N_DETECTED} frames / run")
    print(f"  Boundary transitions: up to {N_BOUNDARY} / run\n")

    df = pd.read_csv(csv_path)
    df['frame']        = pd.to_numeric(df['frame'],        errors='coerce')
    df['arc_detected'] = pd.to_numeric(df['arc_detected'], errors='coerce')

    all_summary = []
    for stem, tiff_path in TIFF_MAP.items():
        run_label = f'Run{list(TIFF_MAP.keys()).index(stem) + 1}'
        print(f"\n--- {run_label}  ({stem}) ---")
        file_df = df[df['file'] == stem].copy()
        if len(file_df) == 0:
            file_df = df[df['file'].str.contains(stem, na=False)].copy()
        if len(file_df) == 0:
            print(f"  [SKIP] No CSV rows for {stem}")
            continue
        rows = qa_file(stem, run_label, file_df, out_root)
        all_summary.extend(rows)

    if all_summary:
        sumdf = pd.DataFrame(all_summary)
        sumdf.to_csv(out_root / 'qa_frames_summary.csv', index=False)
        print(f"\n  Summary CSV: {out_root / 'qa_frames_summary.csv'}")

    print(f"\nDone.  {len(all_summary)} frames rendered in: {out_root}\n")


if __name__ == '__main__':
    main()
