#!/usr/bin/env python3
"""
mode_sample_frames.py  —  Panel of annotated sample video frames by Mode A/B category.

For a given threshold, selects ~N examples from each of 4 categories:
  A-far   : Mode A, norm_pos <= threshold * 0.40  (nucleates very close to electrode)
  A-close : Mode A, norm_pos in [threshold*0.65, threshold)  (just below cutoff)
  B-close : Mode B, norm_pos in (threshold, threshold*1.80]  (just above cutoff)
  B-far   : Mode B, norm_pos >= threshold * 5.0  (well into arc body)

Each panel cell shows the raw grayscale frame with:
  - Red dashed horizontal line   : threshold position  (erow + threshold * leff)
  - Yellow dashed horizontal line: electrode edge (erow)
  - Cyan circle marker           : arc-tip position (if arc_tip_col is not NaN)
  - Title: "Mode {mode}  p={norm_pos:.3f}"  (blue for A, red for B)

Usage:
    python mode_sample_frames.py \\
        --down  PATH_TO_DOWN_RESULTS \\
        --up    PATH_TO_UP_RESULTS \\
        --out-dir OUTPUT_DIR
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile

# ---------------------------------------------------------------------------
# Import qa_restrike helpers (must be on sys.path)
# ---------------------------------------------------------------------------
try:
    from qa_restrike import apex_and_range
except ImportError:
    # Fallback implementation matching qa_restrike.apex_and_range exactly
    def apex_and_range(tip, complete=None):
        if complete is not None and np.asarray(complete).sum() >= 20:
            erow = float(np.percentile(tip[np.asarray(complete)], 5))
        else:
            erow = float(np.percentile(tip, 5))
        l_eff = max(float(np.percentile(tip, 90)) - erow, 1.0)
        return erow, l_eff


MODE_COLOR = {'A': '#1565C0', 'B': '#C62828'}

CATEGORIES = [
    ('A-far',   'A', 'far'),
    ('A-close', 'A', 'close'),
    ('B-close', 'B', 'close'),
    ('B-far',   'B', 'far'),
]

CAT_LABELS = {
    'A-far':   'A-far\n(near electrode)',
    'A-close': 'A-close\n(just below thr)',
    'B-close': 'B-close\n(just above thr)',
    'B-far':   'B-far\n(deep in arc body)',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_csv(p: Path) -> Path:
    """Accept either a directory (appends measurements_all_events.csv) or a CSV path."""
    p = Path(p)
    if p.suffix == '.csv':
        return p
    cand = p / 'measurements_all_events.csv'
    if cand.exists():
        return cand
    cand2 = p / 're_strikes_qa.csv'
    if cand2.exists():
        return cand2
    raise FileNotFoundError('No CSV found in %s' % p)


def load_frame(tiff_dir: Path, stem: str, global_frame: int, frames_per_tiff: int):
    """Load a single page from the correct chunk TIFF. Returns (ndarray, path) or (None, path)."""
    chunk_idx = global_frame // frames_per_tiff
    page_idx  = global_frame % frames_per_tiff
    chunk_path = tiff_dir / ('%s_%04d.tiff' % (stem, chunk_idx + 1))
    if not chunk_path.exists():
        return None, chunk_path
    try:
        with tifffile.TiffFile(str(chunk_path)) as tf:
            if page_idx >= len(tf.pages):
                return None, chunk_path
            return tf.pages[page_idx].asarray(), chunk_path
    except Exception:
        return None, chunk_path


def detect_tiff_stem(tiff_dir: Path) -> str:
    """Infer the file stem from the TIFFs present (everything before _NNNN.tiff)."""
    tiffs = sorted(tiff_dir.glob('*.tiff'))
    if not tiffs:
        raise FileNotFoundError('No TIFF files found in %s' % tiff_dir)
    name = tiffs[0].stem
    parts = name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return name


def build_restrikes_from_csv(csv_path, direction, threshold,
                              px_per_mm=8.8, fps=200000,
                              ext_prom_frac=0.10, min_cycle=5, smooth=3):
    """
    Re-detect re-strikes from a measurements_all_events.csv.
    Returns (DataFrame, dict{fname: (erow, leff)}).
    DataFrame columns: file, frame, tip_row, arc_tip_col, norm_pos, mode, erow, leff, direction
    """
    from scipy.signal import find_peaks
    from scipy.ndimage import uniform_filter1d

    df = pd.read_csv(csv_path)
    for c in ['arc_tip_row', 'arc_tip_col', 'arc_complete']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    det = (df[(df['arc_detected'] == 1)]
           .dropna(subset=['arc_tip_row'])
           .sort_values(['file', 'frame']))

    def _detect(tip, erow, l_eff):
        sig = tip.astype(float)
        if smooth > 1:
            sig = uniform_filter1d(sig, size=int(smooth))
        ext_prom = max(ext_prom_frac * l_eff, 1.0)
        peaks, _ = find_peaks(sig, prominence=ext_prom,
                               distance=max(int(min_cycle), 1))
        out = []
        for i in range(1, len(peaks)):
            a, b = peaks[i - 1], peaks[i]
            k = a + int(np.argmin(sig[a:b + 1]))
            out.append(dict(idx=k, prev_ext=float(sig[a])))
        return out

    recs = []
    file_params = {}

    for fname, g in det.groupby('file'):
        g = g.reset_index(drop=True)
        frames = g['frame'].values.astype(int)
        tip    = g['arc_tip_row'].values.astype(float)
        col    = (g['arc_tip_col'].values.astype(float)
                  if 'arc_tip_col' in g.columns
                  else np.full(len(g), np.nan))
        comp   = ((g['arc_complete'].values == 1)
                  if 'arc_complete' in g.columns else None)

        erow, leff = apex_and_range(tip, comp)
        file_params[fname] = (erow, leff)

        rk = _detect(tip, erow, leff)
        for r in rk:
            k    = r['idx']
            npos = float(np.clip((tip[k] - erow) / leff, 0, 1))
            mode = 'B' if npos > threshold else 'A'
            tip_col_val = float(col[k]) if not np.isnan(col[k]) else np.nan
            recs.append(dict(
                file        = fname,
                frame       = int(frames[k]),
                tip_row     = float(tip[k]),
                arc_tip_col = tip_col_val,
                norm_pos    = round(npos, 4),
                mode        = mode,
                erow        = round(erow, 2),
                leff        = round(leff, 2),
                direction   = direction,
            ))

    return pd.DataFrame(recs), file_params


def load_restrikes_precomputed(csv_path, direction, threshold):
    """
    Load a re_strikes_qa.csv that already has mode/norm_pos/apex_row/l_eff_px.
    Reclassifies with the supplied threshold.
    """
    rs = pd.read_csv(csv_path)
    rs['erow'] = pd.to_numeric(rs.get('apex_row'), errors='coerce')
    rs['leff'] = pd.to_numeric(rs.get('l_eff_px'), errors='coerce')
    if 'arc_tip_col' not in rs.columns:
        rs['arc_tip_col'] = np.nan
    if 'tip_row' not in rs.columns and 'arc_tip_row' in rs.columns:
        rs['tip_row'] = rs['arc_tip_row']
    elif 'tip_row' not in rs.columns:
        rs['tip_row'] = rs['erow']
    rs['mode'] = rs['norm_pos'].apply(lambda p: 'B' if p > threshold else 'A')
    rs['direction'] = direction
    return rs


def categorise(rs, threshold):
    """Return dict: category_name -> DataFrame of candidates."""
    thr = threshold
    return {
        'A-far':   rs[(rs['mode'] == 'A') & (rs['norm_pos'] <= thr * 0.40)].copy(),
        'A-close': rs[(rs['mode'] == 'A') & (rs['norm_pos'] >= thr * 0.65) &
                      (rs['norm_pos'] <  thr)].copy(),
        'B-close': rs[(rs['mode'] == 'B') & (rs['norm_pos'] >  thr) &
                      (rs['norm_pos'] <= thr * 1.80)].copy(),
        'B-far':   rs[(rs['mode'] == 'B') & (rs['norm_pos'] >= thr * 5.0)].copy(),
    }


def pick_examples(candidates, threshold, n_examples):
    """
    Sort by distance from threshold boundary; pick n_examples spread across
    that sorted list (first, middle, last) to show variety.
    """
    if len(candidates) == 0:
        return candidates
    df = candidates.copy()
    df['_dist'] = (df['norm_pos'] - threshold).abs()
    df = df.sort_values('_dist').reset_index(drop=True)
    n = len(df)
    if n <= n_examples:
        return df.drop(columns=['_dist'])
    indices = np.round(np.linspace(0, n - 1, n_examples)).astype(int)
    indices = sorted(set(indices))
    return df.iloc[indices].drop(columns=['_dist']).reset_index(drop=True)


def annotate_axis(ax, img, erow, leff, threshold,
                  tip_row, tip_col, mode, norm_pos, is_leftmost=False,
                  col_header=None):
    """Draw the annotated frame on ax."""
    ax.imshow(img, cmap='gray', aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Yellow dashed line: electrode edge (erow)
    ax.axhline(erow, color='yellow', lw=1.2, ls='--', zorder=3)

    # Red dashed line: threshold position
    thr_row = erow + threshold * leff
    ax.axhline(thr_row, color='red', lw=1.2, ls='--', zorder=3)

    # Label threshold line only on leftmost column to avoid clutter
    if is_leftmost:
        ax.text(3, thr_row - 2, 'threshold',
                color='red', fontsize=6, va='bottom', ha='left', zorder=5,
                bbox=dict(boxstyle='square,pad=0.1', fc='none', ec='none'))

    # Cyan open circle at arc tip (skip if column is NaN)
    if not (tip_col != tip_col):  # NaN check without numpy import at function level
        ax.plot(tip_col, tip_row, 'o',
                color='cyan', ms=7, mew=1.5, fillstyle='none', zorder=4)

    color = MODE_COLOR.get(mode, '#555')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Panel of annotated sample video frames by Mode A/B category.')
    ap.add_argument('--down',  required=True,
                    help='Downward results dir (or CSV path)')
    ap.add_argument('--up',    required=True,
                    help='Upward results dir (or CSV path)')
    ap.add_argument('--tiff-dir-down', default=None,
                    help='TIFF directory for downward data (default: {down}/tiff/)')
    ap.add_argument('--tiff-dir-up',   default=None,
                    help='TIFF directory for upward data (default: {up}/tiff/)')
    ap.add_argument('--threshold',       type=float, default=0.05,
                    help='Mode A/B norm_pos threshold (default: 0.05)')
    ap.add_argument('--px-per-mm',       type=float, default=8.8)
    ap.add_argument('--fps',             type=int,   default=200000)
    ap.add_argument('--frames-per-tiff', type=int,   default=10000)
    ap.add_argument('--out-dir',         required=True)
    ap.add_argument('--n-examples',      type=int,   default=3,
                    help='Number of examples per category (default: 3)')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Resolve paths
    # -----------------------------------------------------------------------
    down_path = Path(args.down)
    up_path   = Path(args.up)

    tiff_dir_down = (Path(args.tiff_dir_down) if args.tiff_dir_down
                     else down_path / 'tiff')
    tiff_dir_up   = (Path(args.tiff_dir_up)   if args.tiff_dir_up
                     else up_path   / 'tiff')

    # -----------------------------------------------------------------------
    # Load per-restrike data for each direction
    # -----------------------------------------------------------------------
    all_rs_list = []

    for direction, base_path, tiff_dir in [
            ('down', down_path, tiff_dir_down),
            ('up',   up_path,   tiff_dir_up)]:

        print('[%s] loading data from %s' % (direction, base_path))

        # Prefer a pre-computed re_strikes_qa.csv; fall back to measurements CSV
        rs_csv = None
        for cand_name in ['re_strikes_qa.csv', 'restrikes.csv']:
            p = (base_path / cand_name if base_path.is_dir()
                 else base_path.parent / cand_name)
            if p.exists():
                rs_csv = p
                break

        if rs_csv is not None:
            print('  Found pre-computed re-strikes CSV: %s' % rs_csv)
            rs = load_restrikes_precomputed(rs_csv, direction, args.threshold)
        else:
            meas_csv = resolve_csv(base_path)
            print('  Deriving re-strikes from: %s' % meas_csv)
            rs, _ = build_restrikes_from_csv(
                meas_csv, direction, args.threshold,
                args.px_per_mm, args.fps)

        if len(rs) == 0:
            print('  WARNING: no re-strikes found for direction=%s' % direction)
            continue

        rs['_tiff_dir'] = str(tiff_dir)
        all_rs_list.append(rs)
        n_a = int((rs['mode'] == 'A').sum())
        n_b = int((rs['mode'] == 'B').sum())
        print('  %d re-strikes loaded  (A=%d, B=%d)' % (len(rs), n_a, n_b))

    if not all_rs_list:
        raise SystemExit('ERROR: no re-strike data found in either direction.')

    all_rs = pd.concat(all_rs_list, ignore_index=True)
    n_a_tot = int((all_rs['mode'] == 'A').sum())
    n_b_tot = int((all_rs['mode'] == 'B').sum())
    print('Total re-strikes: %d  (A=%d, B=%d)' % (len(all_rs), n_a_tot, n_b_tot))

    # -----------------------------------------------------------------------
    # Categorise and pick examples
    # -----------------------------------------------------------------------
    cats   = categorise(all_rs, args.threshold)
    picked = {}
    for cat_name, _mode, _dist in CATEGORIES:
        cands = cats[cat_name]
        picked[cat_name] = pick_examples(cands, args.threshold, args.n_examples)
        print('  %s: %d candidates -> %d selected'
              % (cat_name, len(cands), len(picked[cat_name])))

    # -----------------------------------------------------------------------
    # Build figure:  n_rows x 4 columns
    # -----------------------------------------------------------------------
    n_cols = 4
    n_rows = args.n_examples

    cell_w, cell_h = 3.2, 3.2
    fig_w = cell_w * n_cols + 0.8
    fig_h = cell_h * n_rows + 1.2   # extra for suptitle + column headers

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(fig_w, fig_h),
                              facecolor='white',
                              squeeze=False)

    # Blank all cells
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c].axis('off')

    # Column headers are injected into the top-row cell title via col_header param

    # Fill cells row by row
    for c_idx, (cat_name, mode, _dist_class) in enumerate(CATEGORIES):
        examples = picked[cat_name]
        row_idx  = 0

        for _, ex in examples.iterrows():
            if row_idx >= n_rows:
                break

            ax = axes[row_idx, c_idx]
            tiff_dir = Path(str(ex['_tiff_dir']))

            # Detect TIFF stem
            try:
                stem = detect_tiff_stem(tiff_dir)
            except FileNotFoundError:
                ax.text(0.5, 0.5,
                        'No TIFFs in\n%s' % tiff_dir.name,
                        ha='center', va='center', fontsize=7,
                        transform=ax.transAxes)
                ax.axis('off')
                row_idx += 1
                continue

            global_frame = int(ex['frame'])
            img, chunk_path = load_frame(tiff_dir, stem, global_frame,
                                          args.frames_per_tiff)
            if img is None:
                ax.text(0.5, 0.5,
                        'frame %d\nnot found\n(%s)' % (global_frame, chunk_path.name),
                        ha='center', va='center', fontsize=7,
                        transform=ax.transAxes)
                ax.axis('off')
                row_idx += 1
                continue

            erow    = float(ex['erow'])
            leff    = float(ex['leff'])
            tip_row = float(ex.get('tip_row', erow))
            tip_col_raw = ex.get('arc_tip_col', float('nan'))
            try:
                tip_col = float(tip_col_raw)
            except (ValueError, TypeError):
                tip_col = float('nan')
            norm_pos = float(ex['norm_pos'])

            is_leftmost = (c_idx == 0)
            hdr = CAT_LABELS[cat_name] if row_idx == 0 else None
            annotate_axis(ax, img, erow, leff, args.threshold,
                          tip_row, tip_col, mode, norm_pos,
                          is_leftmost=is_leftmost, col_header=hdr)

            row_idx += 1

    fig.tight_layout(rect=[0, 0, 1, 0.985])

    out_name = 'mode_sample_frames_thr%.2f.png' % args.threshold
    out_path = out_dir / out_name
    fig.savefig(str(out_path), dpi=150, facecolor='white')
    plt.close(fig)
    print('Saved: %s' % out_path)


if __name__ == '__main__':
    main()
