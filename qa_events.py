"""
qa_events.py  —  Visual QA for discharge event velocity analysis

For each selected event this script outputs a dedicated sub-folder containing:

  regression.png      — arc-tip depth vs frame, advancing-front frames
                        highlighted, regression line overlaid, key stats annotated
  contact_sheet.png   — thumbnail grid of every advancing-front frame
                        (up to MAX_SHEET_FRAMES, evenly subsampled if needed)
  frame_NNNNNNN_<status>.png
                      — individual frames with arc-mask overlay and tip marker:
                          green  = advancing-front frame used in regression
                          red    = retreating frame within the event (excluded)
                          grey   = context frame before/after the event

Events selected:
  • Top N_TOP_VEL events by regression velocity  (the ones we most want to QA)
  • N_STRATIFIED events drawn from each lower-velocity quartile band
    (gives baseline context — if fast events LOOK different, that is informative)

A qa_summary.csv is also written to the output root with one row per QA event.

Usage:
    python qa_events.py [measurements_all_events.csv] [event_summary.csv] [out_dir]

Defaults (if no arguments):
    measurements_all_events.csv   in same folder as this script
    event_summary.csv             in same folder as this script
    qa_output/                    sub-folder next to this script
"""

import sys, math
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

FPS = 200_000

# =============================================================================
# CONFIG
# =============================================================================

N_TOP_VEL       = 10    # events with the highest regression velocity
N_STRATIFIED    = 3     # events drawn from each lower-velocity quartile band
N_CONTEXT       = 5     # context frames before / after the event
MAX_SHEET_FRAMES = 36   # max thumbnails on contact sheet (subsampled if needed)
FRAME_DPI       = 110   # DPI for individual frame images (balance quality/size)
SHEET_DPI       = 90    # DPI for contact sheet

# Area bounds for re-binarisation inside this script
# (identical to replot_velmap.py / pipeline)
MIN_AREA_PX = 5
MAX_AREA_PX = 10_000   # generous — we want to see whatever the binarizer finds

# =============================================================================
# HELPERS
# =============================================================================

def advancing_front_mask(frames, rows):
    """Boolean array — True where the frame is on the advancing front."""
    is_front = np.zeros(len(frames), dtype=bool)
    runmax   = -np.inf
    for i, r in enumerate(rows):
        if r >= runmax:
            is_front[i] = True
            runmax = r
    return is_front


def safe_binarize(raw):
    """Normalize then binarize; return None if nothing passes area bounds."""
    gray = normalize_frame(raw)
    mask = binarize(gray, close_r=3, open_r=0)
    area = int(mask.sum())
    if area < MIN_AREA_PX or area > MAX_AREA_PX:
        return None
    return mask

# =============================================================================
# EVENT SELECTION
# =============================================================================

def select_events(events_df):
    """
    Return list of (rank_label, event_row Series) tuples.

    rank_label encodes why the event was selected:
      top01 … top10  — highest velocity events
      q25_1 … q25_3  — events near the 25th percentile (typical slow events)
      q50_1 … q50_3  — events near the median
      q75_1 … q75_3  — events near the 75th percentile
    """
    down = events_df[events_df['vel_ms'] > 0].sort_values(
        'vel_ms', ascending=False).reset_index(drop=True)
    if len(down) == 0:
        return []

    selected   = []
    seen_eids  = set()

    def _add(label, row):
        eid = int(row['event_id'])
        if eid not in seen_eids:
            selected.append((label, row))
            seen_eids.add(eid)

    # Top N
    for rank, (_, row) in enumerate(down.head(N_TOP_VEL).iterrows(), start=1):
        _add(f'top{rank:02d}', row)

    # Stratified sample — fixed seed for reproducibility
    rng = np.random.default_rng(42)
    q25 = down['vel_ms'].quantile(0.25)
    q50 = down['vel_ms'].quantile(0.50)
    q75 = down['vel_ms'].quantile(0.75)

    bands = [
        ('q25', down[down['vel_ms'] <= q25]),
        ('q50', down[(down['vel_ms'] > q25) & (down['vel_ms'] <= q50)]),
        ('q75', down[(down['vel_ms'] > q50) & (down['vel_ms'] <= q75)]),
    ]
    for band_label, band_df in bands:
        n_pick = min(N_STRATIFIED, len(band_df))
        if n_pick == 0:
            continue
        picks = band_df.sample(n=n_pick, random_state=int(rng.integers(0, 9999)))
        for i, (_, row) in enumerate(picks.iterrows(), start=1):
            _add(f'{band_label}_{i}', row)

    return selected

# =============================================================================
# REGRESSION PLOT
# =============================================================================

def render_regression(event_row, det_df, front_mask, out_path):
    """
    Tip depth vs frame — shows advancing front, regression line, and R².
    Depth increases downward (arc moves away from electrode gap).
    """
    frames = det_df['frame'].values.astype(float)
    rows   = pd.to_numeric(det_df['arc_tip_row'], errors='coerce').values
    cols   = pd.to_numeric(det_df['arc_tip_col'], errors='coerce').values  # noqa (for completeness)

    f_adv = frames[front_mask];  r_adv = rows[front_mask]
    f_ret = frames[~front_mask]; r_ret = rows[~front_mask]

    vel_ms  = float(event_row['vel_ms'])
    r2      = float(event_row['r2']) if pd.notna(event_row['r2']) else float('nan')
    n_front = int(event_row['n_frames_front'])
    n_total = int(event_row['n_frames_total'])
    ext_mm  = float(event_row['extension_px']) / PX_PER_MM

    fig, ax = plt.subplots(figsize=(9, 4))

    if len(f_ret):
        ax.scatter(f_ret, r_ret / PX_PER_MM, s=10, c='#B0BEC5',
                   alpha=0.6, zorder=2, label='Retreating')
    if len(f_adv):
        ax.scatter(f_adv, r_adv / PX_PER_MM, s=14, c='#1565C0',
                   alpha=0.9, zorder=3,
                   label='Advancing front (used in regression)')

    # Regression line (recomputed here for transparency)
    if len(f_adv) >= 2:
        slope, intercept = np.polyfit(f_adv, r_adv, 1)
        x_fit = np.array([f_adv.min(), f_adv.max()])
        ax.plot(x_fit, (slope * x_fit + intercept) / PX_PER_MM,
                color='#E53935', lw=2.0, zorder=4, label='Regression fit')

    ax.invert_yaxis()   # row 0 = top of image = electrode gap
    ax.set_xlabel('Frame number')
    ax.set_ylabel('Arc tip depth (mm)  [0 = electrode gap]')
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# INDIVIDUAL FRAME IMAGE
# =============================================================================

STATUS_COLORS = {
    'advancing' : ([0.00, 0.85, 0.15, 0.45], '#00C853'),
    'retreating': ([1.00, 0.20, 0.00, 0.35], '#D50000'),
    'context'   : ([0.55, 0.55, 0.55, 0.25], '#9E9E9E'),
}

def render_frame_image(raw, frame_i, tip_r, tip_c,
                       status, event_info, out_path):
    """
    Two-panel image: left = raw grayscale, right = mask overlay + tip marker.

    Colour coding:
      green  = advancing-front frame (used in regression)
      red    = retreating frame within event (excluded from regression)
      grey   = context frame before/after event
    """
    gray = normalize_frame(raw)
    mask = safe_binarize(raw)
    overlay_rgba, title_color = STATUS_COLORS[status]

    fig, axes = plt.subplots(1, 2, figsize=(7, 4),
                             gridspec_kw={'wspace': 0.04})
    for ax in axes:
        ax.imshow(gray, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
        if not (np.isnan(tip_r) or np.isnan(tip_c)):
            ax.plot(tip_c, tip_r, '+',
                    color='yellow', markersize=16, markeredgewidth=2.0)

    # Right panel: arc-mask overlay
    if mask is not None and mask.any():
        ov = np.zeros((*gray.shape, 4), dtype=np.float32)
        ov[mask] = overlay_rgba
        axes[1].imshow(ov)


    fig.savefig(out_path, dpi=FRAME_DPI, bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# CONTACT SHEET
# =============================================================================

def render_contact_sheet(raw_list, frame_idx, mask_list, tip_list,
                         event_row, out_path):
    """
    Grid of advancing-front frame thumbnails. Up to MAX_SHEET_FRAMES shown;
    if the event has more, frames are evenly subsampled.
    """
    n = len(raw_list)
    if n == 0:
        return

    # Even subsample
    if n > MAX_SHEET_FRAMES:
        pick = np.round(np.linspace(0, n - 1, MAX_SHEET_FRAMES)).astype(int)
        raw_list  = [raw_list[i]  for i in pick]
        frame_idx = [frame_idx[i] for i in pick]
        mask_list = [mask_list[i] for i in pick]
        tip_list  = [tip_list[i]  for i in pick]
        n = MAX_SHEET_FRAMES

    ncols = min(6, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 1.9, nrows * 2.3))
    axes_flat = np.array(axes).reshape(-1)

    for i, (raw, fi, mask, (tr, tc)) in enumerate(
            zip(raw_list, frame_idx, mask_list, tip_list)):
        ax   = axes_flat[i]
        gray = normalize_frame(raw)
        ax.imshow(gray, cmap='gray', vmin=0, vmax=1)

        if mask is not None and mask.any():
            ov = np.zeros((*gray.shape, 4), dtype=np.float32)
            ov[mask] = [0.0, 0.85, 0.15, 0.55]
            ax.imshow(ov)

        if not (np.isnan(tr) or np.isnan(tc)):
            ax.plot(tc, tr, 'y+', markersize=8, markeredgewidth=1.5)

        ax.axis('off')

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')

    vel_ms = float(event_row['vel_ms'])
    r2     = float(event_row['r2']) if pd.notna(event_row['r2']) else float('nan')
    fig.tight_layout()
    fig.savefig(out_path, dpi=SHEET_DPI, bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# PER-EVENT QA ORCHESTRATION
# =============================================================================

def qa_event(rank_label, event_row, all_df, out_root):
    eid     = int(event_row['event_id'])
    fname   = str(event_row['file'])
    vel_ms  = float(event_row['vel_ms'])
    f_start = int(event_row['frame_start'])
    f_end   = int(event_row['frame_end'])

    tiff_path = TIFF_MAP.get(fname)
    if tiff_path is None or not tiff_path.exists():
        print(f"  [SKIP] TIFF not found for {fname}")
        return

    # All detected frames belonging to this event
    det = all_df[
        (all_df['event_id'] == eid) &
        (all_df['arc_detected'] == 1)
    ].copy().sort_values('frame').reset_index(drop=True)

    if len(det) == 0:
        print(f"  [SKIP] No detected frames for event {eid}")
        return

    frames = det['frame'].values.astype(float)
    rows   = pd.to_numeric(det['arc_tip_row'], errors='coerce').fillna(-1).values
    cols   = pd.to_numeric(det['arc_tip_col'], errors='coerce').values
    tip_r  = pd.to_numeric(det['arc_tip_row'], errors='coerce').values
    tip_c  = pd.to_numeric(det['arc_tip_col'], errors='coerce').values

    front  = advancing_front_mask(frames, rows)

    # Output folder: rank_label + event stats in name for easy sorting
    dir_name = f'{rank_label}_event{eid}_v{vel_ms:.3f}ms_R2{event_row["r2"]:.2f}'
    out_dir  = out_root / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [{rank_label}] event {eid}  file={fname}  "
          f"v={vel_ms:.3f} m/s  R²={event_row['r2']:.3f}  "
          f"front={int(event_row['n_frames_front'])}/{int(event_row['n_frames_total'])}")

    # ── 1. Regression plot ────────────────────────────────────────────────────
    render_regression(event_row, det, front, out_dir / 'regression.png')

    # ── 2. Build frame render list ────────────────────────────────────────────
    # Context frames before and after event
    frame_status: dict[int, str] = {}
    for f in range(max(0, f_start - N_CONTEXT), f_start):
        frame_status[f] = 'context'
    for f in range(f_end + 1, f_end + 1 + N_CONTEXT):
        frame_status[f] = 'context'

    # Event frames
    for i, f in enumerate(frames.astype(int)):
        frame_status[f] = 'advancing' if front[i] else 'retreating'

    # Tip lookup for event frames
    tip_lookup = {int(f): (r, c)
                  for f, r, c in zip(frames, tip_r, tip_c)}
    event_info = {'event_id': eid, 'vel_ms': vel_ms}

    # ── 3. Read TIFF once, render everything ──────────────────────────────────
    sheet_raws, sheet_idx, sheet_masks, sheet_tips = [], [], [], []

    with tifffile.TiffFile(tiff_path) as tif:
        n_pages = len(tif.pages)
        for fi in sorted(frame_status.keys()):
            if fi < 0 or fi >= n_pages:
                continue
            raw    = tif.pages[fi].asarray()
            status = frame_status[fi]
            tr, tc = tip_lookup.get(fi, (float('nan'), float('nan')))

            out_fp = out_dir / f'frame_{fi:07d}_{status}.png'
            render_frame_image(raw, fi, tr, tc, status, event_info, out_fp)

            if status == 'advancing':
                sheet_raws.append(raw)
                sheet_idx.append(fi)
                sheet_masks.append(safe_binarize(raw))
                sheet_tips.append((tr, tc))

    # ── 4. Contact sheet ──────────────────────────────────────────────────────
    render_contact_sheet(sheet_raws, sheet_idx, sheet_masks, sheet_tips,
                         event_row, out_dir / 'contact_sheet.png')

    return {
        'rank_label'     : rank_label,
        'event_id'       : eid,
        'file'           : fname,
        'vel_ms'         : vel_ms,
        'r2'             : float(event_row['r2']),
        'n_front'        : int(event_row['n_frames_front']),
        'n_total'        : int(event_row['n_frames_total']),
        'extension_mm'   : float(event_row['extension_px']) / PX_PER_MM,
        'duration_ms'    : int(event_row['duration_frames']) / FPS * 1000,
        'output_dir'     : str(out_dir),
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    script_dir = Path(__file__).parent.resolve()

    csv_path   = (Path(sys.argv[1]) if len(sys.argv) > 1
                  else script_dir / 'measurements_all_events.csv')
    evsum_path = (Path(sys.argv[2]) if len(sys.argv) > 2
                  else script_dir / 'event_summary.csv')
    out_root   = (Path(sys.argv[3]) if len(sys.argv) > 3
                  else script_dir / 'qa_output')

    for p in [csv_path, evsum_path]:
        if not p.exists():
            print(f"[ERROR] Not found: {p}"); sys.exit(1)

    out_root.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Event QA  —  visual verification of velocity estimates")
    print(f"  Frames CSV   : {csv_path}")
    print(f"  Event summary: {evsum_path}")
    print(f"  Output root  : {out_root}")
    print(f"{'='*60}\n")
    print(f"  Selection: top {N_TOP_VEL} by velocity  +  "
          f"{N_STRATIFIED}×3 stratified sample\n")

    all_df    = pd.read_csv(csv_path)
    all_df['frame']        = pd.to_numeric(all_df['frame'],        errors='coerce')
    all_df['event_id']     = pd.to_numeric(all_df['event_id'],     errors='coerce')
    all_df['arc_detected'] = pd.to_numeric(all_df['arc_detected'], errors='coerce')

    events_df = pd.read_csv(evsum_path)
    events_df['vel_ms'] = pd.to_numeric(events_df['vel_ms'], errors='coerce')
    events_df['r2']     = pd.to_numeric(events_df['r2'],     errors='coerce')

    selected = select_events(events_df)
    print(f"  {len(selected)} events selected\n")

    summary_rows = []
    for rank_label, event_row in selected:
        result = qa_event(rank_label, event_row, all_df, out_root)
        if result:
            summary_rows.append(result)

    # Write summary CSV
    if summary_rows:
        sumdf = pd.DataFrame(summary_rows)
        sumdf.to_csv(out_root / 'qa_summary.csv', index=False)
        print(f"\n  Summary: {out_root / 'qa_summary.csv'}")

    print(f"\nDone.  {len(summary_rows)} events QA'd in: {out_root}\n")


if __name__ == '__main__':
    main()
