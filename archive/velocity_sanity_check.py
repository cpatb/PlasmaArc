"""
velocity_sanity_check.py  —  Independent velocity validation

Reads measurements_all.csv and re-derives arc-tip velocity using three
strict quality gates:

  1. FULL ARC:   both electrode tips detected + arc spans electrode gap
  2. SINGLE EVENT: pairs drawn only from within a single discharge event
                   (no cross-event jumps where the tip teleports back up)
  3. POSITIVE DISPLACEMENT: tip must move away from electrodes (downward)

Then repeats the analysis at four temporal spacings to check for bias:
  ~1 frame   (nearest-neighbour, same as main pipeline)
  ~50 frames
  ~100 frames
  ~500 frames

This script does NOT modify any existing data — it is read-only.

Usage:
    python velocity_sanity_check.py [measurements_all.csv] [output_dir]
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIG  —  edit these if needed
# =============================================================================

FPS           = 200_000      # frames per second
PX_PER_MM     = 9.167        # calibration

# Full-arc filter: arc must span at least this fraction of the max observed
# electrode gap in each file (filters out partial / side arcs).
MIN_GAP_FRAC  = 0.70         # 0.70 = arc spans >= 70% of the full electrode gap

# Brightness filter: only use arcs in the top N % by integrated intensity.
BRIGHT_PCTILE = 75           # 75 means keep only the top 25% brightest

# Spatial reset threshold: if the arc tip row drops by more than this many
# pixels from the running maximum within an event, the arc has extinguished
# and re-nucleated — that is the physical event boundary.
# At 9.167 px/mm, 25 px ≈ 2.7 mm ≈ one electrode gap.
RESET_ROWS    = 25        # pixels

# Secondary guard: a detection gap this large also forces a new event.
MAX_FRAME_GAP = 100       # frames

# Temporal spacings to test (in frames).
SPACINGS      = [1, 50, 100, 500]

# Tolerance for matching pairs at each spacing (fraction of spacing).
# E.g. 0.25 means a target spacing of 100 accepts pairs 75–125 frames apart.
SPACING_TOL   = 0.25

# =============================================================================
# LOAD DATA
# =============================================================================

def load(csv_path):
    df = pd.read_csv(csv_path)
    det = df[df['arc_detected'] == 1].copy()

    numeric_cols = ['tip_a_row', 'tip_a_col', 'tip_b_row', 'tip_b_col',
                    'arc_tip_row', 'arc_tip_col',
                    'electrode_gap_px', 'integrated_intensity']
    for col in numeric_cols:
        det[col] = pd.to_numeric(det[col], errors='coerce')

    det = det.dropna(subset=['arc_tip_row', 'arc_tip_col'])
    det = det.sort_values(['file', 'frame']).reset_index(drop=True)

    print(f"Loaded {len(det):,} detected frames from {det['file'].nunique()} file(s)")
    return det

# =============================================================================
# QUALITY FILTERS
# =============================================================================

def filter_full_arcs(det):
    """
    Keep only frames where:
      - Both electrode tips are detected (tip_a and tip_b not NaN)
      - The electrode gap is at least MIN_GAP_FRAC of the per-file maximum
        (ensures the arc is spanning fully across, not a partial discharge)
    """
    has_tips = det['tip_a_row'].notna() & det['tip_b_row'].notna()

    gap_ok = pd.Series(False, index=det.index)
    for fname, grp in det.groupby('file'):
        max_gap = grp['electrode_gap_px'].max()
        thresh  = MIN_GAP_FRAC * max_gap
        gap_ok[grp.index] = grp['electrode_gap_px'] >= thresh

    full = det[has_tips & gap_ok].copy()
    print(f"Full-arc filter : {len(full):,} frames  "
          f"({100 * len(full) / len(det):.1f}% of detected)")
    return full


def filter_bright(full):
    """
    Keep only the top (100 - BRIGHT_PCTILE)% brightest arcs by integrated
    intensity.  This selects the most well-defined, cleanest discharges.
    """
    thresh = full['integrated_intensity'].quantile(BRIGHT_PCTILE / 100)
    bright = full[full['integrated_intensity'] >= thresh].copy()
    print(f"Brightness filter (top {100 - BRIGHT_PCTILE:.0f}%): "
          f"{len(bright):,} frames")
    return bright

# =============================================================================
# DISCHARGE EVENT GROUPING
# =============================================================================

def assign_events(df):
    """
    Define discharge events by spatial trajectory, not arbitrary time gaps.

    A new event starts when either:
      (a) the arc tip row drops > RESET_ROWS pixels from the running maximum
          within an event — the arc extinguished and re-nucleated at the
          electrodes (the physically meaningful boundary), OR
      (b) there is a detection gap > MAX_FRAME_GAP frames (complete dropout).
    """
    df = df.copy()
    df['event_id'] = -1
    eid = 0

    for fname, grp in df.groupby('file'):
        grp    = grp.sort_values('frame')
        frames = grp['frame'].values
        rows   = grp['arc_tip_row'].values
        idx    = grp.index

        running_max = rows[0]
        df.loc[idx[0], 'event_id'] = eid

        for k in range(1, len(frames)):
            frame_gap = frames[k] - frames[k - 1]
            row_drop  = running_max - rows[k]

            if row_drop > RESET_ROWS or frame_gap > MAX_FRAME_GAP:
                eid        += 1
                running_max = rows[k]
            else:
                running_max = max(running_max, rows[k])

            df.loc[idx[k], 'event_id'] = eid
            if k == len(frames) - 1:
                eid += 1

    n_events = df['event_id'].nunique()
    med_len  = df.groupby('event_id').size().mean()
    print(f"Discharge events : {n_events}  (mean length {med_len:.0f} frames)")
    return df

# =============================================================================
# VELOCITY COMPUTATION
# =============================================================================

def compute_velocity(df, spacing):
    """
    For each qualifying pair (frame_i, frame_j) within the same discharge event
    where frame_j - frame_i ≈ spacing:

      velocity = displacement_mm / elapsed_time_s   (in m/s)

    Only pairs where the tip moves DOWNWARD (row increases) are kept.
    This enforces the 'positive displacement' constraint — the arc tip should
    advance away from the electrodes during a single discharge, not jump back.
    """
    tol  = 0 if spacing == 1 else max(1, int(spacing * SPACING_TOL))
    lo   = spacing - tol   # for spacing=1: lo=hi=1  → exact match only
    hi   = spacing + tol
    vels = []

    for eid, grp in df.groupby('event_id'):
        if len(grp) < 2:
            continue
        grp    = grp.sort_values('frame')
        frames = grp['frame'].values
        rows   = grp['arc_tip_row'].values
        cols   = grp['arc_tip_col'].values

        for k in range(len(frames)):
            for j in range(k + 1, len(frames)):
                gap = int(frames[j]) - int(frames[k])
                if gap < lo:
                    continue
                if gap > hi:
                    break   # frames are sorted, no need to look further

                dr = rows[j] - rows[k]
                if dr <= 0:
                    continue   # tip did not advance downward — skip

                dc       = cols[j] - cols[k]
                disp_mm  = np.hypot(dr, dc) / PX_PER_MM
                dt_s     = gap / FPS
                vels.append(disp_mm / dt_s / 1000)   # convert mm/s → m/s

    return np.array(vels)

# =============================================================================
# OUTPUT
# =============================================================================

def print_summary(results):
    print(f"\n{'─' * 65}")
    print(f"{'Spacing (f)':>12}  {'N pairs':>8}  {'Mean':>8}  "
          f"{'p25':>7}  {'p75':>7}  {'p95':>7}  (m/s)")
    print(f"{'─' * 65}")
    for s in SPACINGS:
        d = results[s]
        if len(d) == 0:
            print(f"{s:>12}  {'0':>8}  {'—':>8}  {'—':>7}  {'—':>7}  {'—':>7}")
        else:
            print(f"{s:>12}  {len(d):>8}  {np.mean(d):>8.2f}  "
                  f"{np.percentile(d, 25):>7.2f}  "
                  f"{np.percentile(d, 75):>7.2f}  "
                  f"{np.percentile(d, 95):>7.2f}")
    print(f"{'─' * 65}")
    print("Interpretation: if the mean is stable across spacings, the\n"
          "main pipeline velocity is not contaminated by cross-event jumps.\n")


def plot_results(results, out_dir):
    """
    Figure 1 — violin plot of velocity distributions at each spacing.
    Figure 2 — median + IQR vs spacing (coarse-graining stability check).
    """
    spacings_with_data = [s for s in SPACINGS if len(results[s]) > 0]
    if not spacings_with_data:
        print("[WARN] No velocity data — skipping plots.")
        return

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    # ── Figure 1: violin comparison ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    data    = [results[s] for s in SPACINGS]
    parts   = ax.violinplot(data, positions=range(len(SPACINGS)),
                            showmedians=True, showextrema=False)
    for pc, col in zip(parts['bodies'], colors):  # type: ignore[arg-type]
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(2)

    ax.set_xticks(range(len(SPACINGS)))
    ax.set_xticklabels([f'Δ = {s} frames' for s in SPACINGS])
    ax.set_ylabel('Arc-tip velocity (m/s)')
    ax.grid(axis='y', alpha=0.3)

    # Annotate median and count on each violin
    ymax = ax.get_ylim()[1]
    for k, s in enumerate(SPACINGS):
        d = results[s]
        if len(d):
            ax.text(k, ymax * 0.97,
                    f'n = {len(d)}\nmed = {np.mean(d):.1f}',
                    ha='center', va='top', fontsize=8)

    fig.tight_layout()
    out = out_dir / 'sanity_velocity_distributions.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")

    # ── Figure 2: coarse-graining stability curve ─────────────────────────────
    medians = [np.mean(results[s]) for s in spacings_with_data]
    q25     = [np.percentile(results[s], 25) for s in spacings_with_data]
    q75     = [np.percentile(results[s], 75) for s in spacings_with_data]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(spacings_with_data, medians, 'o-', color='#2196F3',
            linewidth=2, markersize=7, label='Mean')
    ax.fill_between(spacings_with_data, q25, q75,
                    alpha=0.25, color='#2196F3', label='IQR (25-75%)')
    if len(medians) > 0:
        ax.axhline(medians[0], color='gray', linestyle='--', linewidth=1,
                   label=f'delta=1 baseline  ({means[0]:.1f} m/s)')
    ax.set_xscale('log')
    ax.set_xlabel('Frame spacing  (log scale)')
    ax.set_ylabel('Arc-tip velocity (m/s)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / 'sanity_velocity_coarsegraining.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    script_dir = Path(__file__).parent.resolve()
    csv_path   = Path(sys.argv[1]) if len(sys.argv) > 1 else script_dir / 'measurements_all.csv'
    out_dir    = Path(sys.argv[2]) if len(sys.argv) > 2 else script_dir

    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 55}")
    print(f"  Velocity sanity check")
    print(f"  CSV    : {csv_path}")
    print(f"  Output : {out_dir}")
    print(f"{'=' * 55}\n")

    det    = load(csv_path)
    full   = filter_full_arcs(det)
    bright = filter_bright(full)
    bright = assign_events(bright)

    print(f"\nComputing velocity at frame spacings: {SPACINGS} ...\n")
    results = {}
    for spacing in SPACINGS:
        v = compute_velocity(bright, spacing)
        results[spacing] = v
        if len(v):
            print(f"  Δ = {spacing:4d} f  →  {len(v):5d} pairs,  "
                  f"median = {np.mean(v):.2f} m/s")
        else:
            print(f"  Δ = {spacing:4d} f  →  0 pairs  (not enough qualifying frames)")

    print_summary(results)
    plot_results(results, out_dir)
    print("Done.\n")


if __name__ == '__main__':
    main()
