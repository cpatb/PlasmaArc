#!/usr/bin/env python3
"""
arc_filter.py - THE single source of truth for "what counts as a valid plasma-arc
frame" in the BLADE pipeline.

Every stage that produces scientific results (measurement, event/velocity, the
2-D velocity map, Mode A/B comparisons, paper figures, Fourier analysis) imports
the criteria and predicates from THIS module, so that one definition is applied
everywhere. Do NOT redefine these thresholds or re-implement these predicates in
any other file.

===========================================================================
                        UNIFIED VALIDITY CRITERIA
===========================================================================
A frame contributes to scientific results if and only if ALL of criteria 1-3
hold. They are evaluated on the per-frame measurements written by
analyze_blade.py (one row per frame in measurements_all_events.csv).

  1. DETECTED        arc_detected == 1
       A bright connected region survived binarisation:
         - pixel DN > bright_thresh            (per-recording calibration)
         - total area in [min_arc_px, max_arc_px]
         - frame not dark-guarded (max DN >= dark_guard)
       AND the region did NOT cross the electrode-gap row: if --gap-row is set,
       a mask that reaches past it into the electrode/flash zone sets
       arc_detected = 0 at the source (analyze_blade.measure_frame).

  2. COMPLETE        arc_complete == 1
       The arc skeleton bridges BOTH electrodes - each endpoint within
       ELEC_TOL_PX of its blade line (blade_lines.npy), or the widest-pair
       fallback when no blade lines are available.

  3. NOT A FLASH     is_flash == 0, where
       is_flash = (eff_width > FLASH_EFF_WIDTH) OR (arc_width_px > FLASH_ARC_WIDTH)
         eff_width    = area_px / arc_length_px   - saturated blob / collapsed
                        skeleton (ratio explodes).
         arc_width_px = mean FWHM perpendicular to the skeleton - a genuinely
                        thick arc that did not blob enough to break the skeleton.
       The canonical is_flash is computed once in analyze_blade.measure_frame and
       written to the CSV; downstream code reads that column (see add_flash_flag).

NO ARBITRARY VELOCITY CAP.
  Flash artefacts are removed at the FRAME level by criteria 1-3 (the width-based
  is_flash flag plus the electrode-gap cutoff), NOT by clipping velocity values at
  some magnitude.  Velocities are reported exactly as measured for every valid
  frame and the distribution is never truncated.  (A 16 m/s cap previously stood
  in for flash filtering; it was an unprincipled band-aid and has been removed now
  that flashes are rejected at the source.)

NOTE ON CALIBRATION vs CRITERIA
  Dataset-specific CALIBRATION - px/mm, bright_thresh, dark_guard, the electrode
  gap row - is per-recording and is owned by run_11may26.py (the GAP_ROW /
  PX_PER_MM / BRIGHT_THRESH config + CLI flags), NOT by this module. arc_filter
  owns only the UNIVERSAL validity logic and thresholds below.
===========================================================================
"""
import numpy as np
import pandas as pd

# ---- Universal validity thresholds (dataset-independent) --------------------
# These two widths define the only flash test (criterion 3).  There is NO
# velocity cap: flashes are rejected per-frame, velocities are never clipped.
FLASH_EFF_WIDTH = 20.0   # px ; area_px / arc_length_px above this  -> flash
FLASH_ARC_WIDTH = 16.0   # px ; FWHM arc width (arc_width_px) above this -> flash


def add_flash_flag(df, eff_width=FLASH_EFF_WIDTH, arc_width=FLASH_ARC_WIDTH):
    """Return a copy of df with 'eff_width' and 'is_flash' columns.

    is_flash = (eff_width > eff_width) | (arc_width_px > arc_width).  NaN
    comparisons are False, so a frame missing a width is not flagged by that leg.
    Pass eff_width=0/None or arc_width=0/None to disable that leg.

    If the CSV already carries an 'is_flash' column (written at the source by
    analyze_blade.measure_frame), that canonical value is PRESERVED rather than
    recomputed, guaranteeing every stage sees the identical flag.
    """
    df = df.copy()
    a = pd.to_numeric(df.get('area_px'), errors='coerce')
    L = pd.to_numeric(df.get('arc_length_px'), errors='coerce').clip(lower=1)
    w = pd.to_numeric(df.get('arc_width_px'), errors='coerce')
    df['eff_width'] = a / L
    if 'is_flash' in df.columns:
        df['is_flash'] = (pd.to_numeric(df['is_flash'], errors='coerce')
                          .fillna(0).astype(bool))
        return df
    flag = pd.Series(False, index=df.index)
    if eff_width:
        flag = flag | (df['eff_width'] > eff_width)
    if arc_width:
        flag = flag | (w > arc_width)
    df['is_flash'] = flag
    return df


def is_flash(area_px, arc_length_px, arc_width_px,
             eff_width=FLASH_EFF_WIDTH, arc_width=FLASH_ARC_WIDTH):
    """Scalar form of criterion 3 - used by analyze_blade.measure_frame to write
    the canonical is_flash column at the source.  Returns a Python bool."""
    L = max(float(arc_length_px), 1.0)
    ew = float(area_px) / L
    flag = False
    if eff_width and ew > eff_width:
        flag = True
    if arc_width and not np.isnan(float(arc_width_px)) and float(arc_width_px) > arc_width:
        flag = True
    return bool(flag)


def flash_mask(df, eff_width=FLASH_EFF_WIDTH, arc_width=FLASH_ARC_WIDTH):
    """Boolean Series: True where the frame is a flash (criterion 3)."""
    return add_flash_flag(df, eff_width, arc_width)['is_flash']


def detected_mask(df):
    """Bright, non-flash arc present (criteria 1 & 3; completeness NOT required).
    Use for detection-rate / partial-arc diagnostics."""
    det = pd.to_numeric(df.get('arc_detected'), errors='coerce').fillna(0).astype(int) == 1
    return det & ~flash_mask(df)


def valid_frames(df):
    """THE canonical 'real propagating arc' predicate: detected & complete &
    not-flash (criteria 1-3).  Every result-producing stage must select frames
    with this and nothing else."""
    comp = pd.to_numeric(df.get('arc_complete'), errors='coerce').fillna(0).astype(int) == 1
    return detected_mask(df) & comp
