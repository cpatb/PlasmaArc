"""
event_velocity.py  —  event-averaged and local arc-tip velocity

For each discharge event, fits a line to the arc tip's advancing-front
trajectory (monotonic envelope, retreats excluded) to produce a
noise-resistant event-averaged velocity.  A per-frame sliding-window OLS
velocity is also computed, resolving within-event acceleration while
remaining noise-resistant.

Event detection is physics-based: a new primary event is declared only when
the arc tip returns to near the electrode gap (its nucleation zone).  Within
a primary event, intermediate retreats that exceed SUBEVENT_RETREAT_FRAC of
the current extension are labelled sub-events (partial re-ignition without
full reset).  Regression uses complete-arc frames (arc_complete == True)
where the tip position is unambiguous; falls back to all detected frames when
too few complete frames exist.

Mode A / Mode B classification follows Czernichowski (1994) and Zhang et al.
(2020 IET HVE): Mode A events nucleate fresh at the electrode gap; Mode B
events reconnect (back-breakdown) at an intermediate position.  Classification
uses norm_nucleation_pos = (tip_row_start − electrode_row) / L_eff_px, a
dimensionless ratio that requires no spatial calibration.  The threshold
equals NUCLEATION_SLACK_FRAC so that the event detector and the mode
classifier share a self-consistent definition of "at the gap."

Usage:
    python event_velocity.py [measurements_all.csv] [output_dir]

Outputs:
    event_summary.csv             — one row per event
    measurements_all_events.csv   — frame-level data with event IDs merged in
    events/                       — per-event CSVs (frames + dynamics)
    event_velocity_dist.png       — velocity histogram, Mode A/B stratified
    event_velocity_scatter.png    — velocity vs duration coloured by R²
    event_velocity_comparison.png — old Δ=1 vs regression method comparison
    event_dynamics_density.png    — hexbin: local velocity vs normalised position
    event_dynamics_heatmap.png    — per-event velocity heatmap sorted by speed
    mode_nucleation_distribution.png — nucleation position histogram + scatter
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import arc_filter  # single source of truth for flash / validity criteria

# ---
# Configuration
# ---

FPS    = 200_000   # frames per second
PX_PER_MM = 9.167  # px/mm, BLADE calibration
# No velocity cap: flashes are rejected at the frame level via arc_filter
# (width-based is_flash + electrode-gap cutoff), not by clipping velocities.

# Gliding arc cycle: arc nucleates at the electrode gap, propagates
# downstream, extinguishes, and re-nucleates.  A new primary event is
# declared when the tip returns to within NUCLEATION_SLACK_FRAC of the
# full arc range above the electrode row.  This fraction is data-driven:
# it scales with the measured arc range per recording rather than using a
# hard-coded pixel threshold.
NUCLEATION_SLACK_FRAC = 0.15   # 15% of full range → "at the gap"

# Within a primary event, retreats larger than this fraction of the
# current extension are labelled sub-events (intermediate re-attachment).
SUBEVENT_RETREAT_FRAC = 0.50

# Detection dropout longer than this forces a new primary event — long
# enough to bridge genuine inter-frame dropouts without merging events.
MAX_FRAME_GAP = 500   # frames (~2.5 ms at 200 kfps)

# Minimum advancing-front frames required to fit an event.
MIN_EVENT_FRAMES = 5

# Sliding-window width (in advancing-front neighbors) for local OLS velocity.
# Centered on each advancing-front frame; 5 -> +/-2 neighbors each side.
LOCAL_VEL_N = 5

# Position bins for dynamics plots (normalised arc position 0→1).
N_POSITION_BINS = 10

# Minimum arc extension for a dynamics record to be included in plots.
MIN_EXTENSION_PX = 10

# ---
# Load
# ---

def load(csv_path):
    df = pd.read_csv(csv_path)
    for col in ['arc_tip_row', 'arc_tip_col', 'arc_tip_vel_mm_s']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"Loaded {len(df):,} rows  |  {(df['arc_detected']==1).sum():,} detected")
    return df

# ---
# Event assignment
# ---

def estimate_electrode_row(grp):
    """
    Pixel row of the electrode gap for one file's detected frames.

    Uses the 5th percentile of arc_tip_row from complete-arc frames
    (where the arc bridges both electrodes and the tip is closest to the
    gap).  Falls back to all detected frames when fewer than 20 complete
    rows are available.  arc_tip_row is the bottommost arc pixel, so
    smaller row ≈ tip near the gap, larger row ≈ arc extended.
    """
    rows = pd.to_numeric(grp['arc_tip_row'], errors='coerce').dropna()
    if len(rows) == 0:
        return np.nan
    if 'arc_complete' in grp.columns:
        mask = pd.to_numeric(grp['arc_complete'], errors='coerce').fillna(0).astype(bool)
        complete_rows = rows[mask.reindex(rows.index, fill_value=False)]
        if len(complete_rows) >= 20:
            return float(np.percentile(complete_rows, 5))
    return float(np.percentile(rows, 5))


def assign_events(df):
    """
    Group detected frames into discharge events.

    Primary event boundary: tip row returns to within NUCLEATION_SLACK_FRAC
    of the full arc range above the electrode row, OR detection drops for
    more than MAX_FRAME_GAP frames.

    Sub-event boundary (within a primary event): tip retreats more than
    SUBEVENT_RETREAT_FRAC of the current extension without returning to
    the gap — corresponds to intermediate re-attachment.

    Returns the DataFrame with event_id and sub_event_id columns added,
    and a dict of per-file electrode geometry for downstream use.
    """
    df = df.copy()
    df['event_id']     = -1
    df['sub_event_id'] = -1
    eid = 0
    electrode_info = {}

    for fname, fgrp in df[df['arc_detected'] == 1].groupby('file'):
        fgrp   = fgrp.sort_values('frame')
        frames = fgrp['frame'].values.astype(float)
        rows   = fgrp['arc_tip_row'].values.astype(float)
        idx    = fgrp.index

        electrode_row   = estimate_electrode_row(fgrp)
        if np.isnan(electrode_row):
            electrode_row = float(np.nanpercentile(rows, 5))

        typical_max    = float(np.nanpercentile(rows, 90))
        full_range     = max(typical_max - electrode_row, 1.0)
        nucl_thresh    = electrode_row + NUCLEATION_SLACK_FRAC * full_range

        electrode_info[fname] = {
            'electrode_row': electrode_row,
            'l_eff_px':      full_range,
            'nucl_thresh':   nucl_thresh,
        }

        print(f"  {fname}:  electrode_row={electrode_row:.1f} px  "
              f"typical_max={typical_max:.1f} px  "
              f"nucleation_thresh={nucl_thresh:.1f} px  "
              f"(slack {NUCLEATION_SLACK_FRAC:.0%} × {full_range:.1f} px range)")

        running_max = rows[0]
        sub_eid     = 0
        df.loc[idx[0], 'event_id']     = eid
        df.loc[idx[0], 'sub_event_id'] = sub_eid

        for k in range(1, len(frames)):
            gap = int(frames[k]) - int(frames[k - 1])
            tip = rows[k]

            if gap > MAX_FRAME_GAP or tip <= nucl_thresh:
                eid        += 1
                sub_eid     = 0
                running_max = tip
            else:
                retreat   = running_max - tip
                extension = max(running_max - electrode_row, 0.0)
                if (extension > MIN_EXTENSION_PX
                        and retreat > SUBEVENT_RETREAT_FRAC * extension):
                    sub_eid += 1
                running_max = max(running_max, tip)

            df.loc[idx[k], 'event_id']     = eid
            df.loc[idx[k], 'sub_event_id'] = sub_eid

        eid += 1

    n = df[df['event_id'] >= 0]['event_id'].nunique()
    print(f"Discharge events (primary): {n}")
    return df, electrode_info

# ---
# Advancing-front filter
# ---

def advancing_front(frames, rows):
    """
    Return indices of frames where the arc tip reaches a new maximum row
    (i.e. genuinely advancing, not retreating).  Used to strip retreats
    before regression so the slope reflects forward progression only.
    """
    keep   = []
    runmax = -np.inf
    for i, r in enumerate(rows):
        if r >= runmax:
            keep.append(i)
            runmax = r
    return np.array(keep, dtype=int)

# ---
# Event regression
# ---

def fit_events(df, electrode_info=None):
    """
    Fit a line to the advancing-front tip trajectory for each primary event:
        tip_row = slope × frame + intercept
        velocity = slope [px/frame] × FPS / PX_PER_MM / 1000  →  m/s

    Preferentially uses complete-arc frames (arc bridges both electrodes,
    tip position unambiguous); falls back to all detected frames when fewer
    than MIN_EVENT_FRAMES complete frames are available.

    Returns one row per qualifying event.
    """
    detected = df[df['event_id'] >= 0].dropna(subset=['arc_tip_row'])
    detected = detected[~arc_filter.flash_mask(detected)]   # criterion 3: drop flashes
    records  = []
    n_complete_used = 0

    for eid, grp in detected.groupby('event_id'):
        grp       = grp.sort_values('frame')
        sub_count = int(grp['sub_event_id'].max()) if 'sub_event_id' in grp.columns else 0

        used_complete = False
        if 'arc_complete' in grp.columns:
            mask  = pd.to_numeric(grp['arc_complete'], errors='coerce').fillna(0).astype(bool)
            fit_g = grp[mask]
            if len(fit_g) >= MIN_EVENT_FRAMES:
                used_complete = True
            else:
                fit_g = grp
        else:
            fit_g = grp

        frames = fit_g['frame'].values.astype(float)
        rows   = fit_g['arc_tip_row'].values.astype(float)
        front  = advancing_front(frames, rows)

        if len(front) < MIN_EVENT_FRAMES:
            continue

        f_front = frames[front]
        r_front = rows[front]

        slope, intercept = np.polyfit(f_front, r_front, 1)
        predicted = slope * f_front + intercept
        ss_res = np.sum((r_front - predicted) ** 2)
        ss_tot = np.sum((r_front - r_front.mean()) ** 2)
        r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        vel_ms = slope * FPS / PX_PER_MM / 1000
        # No velocity cap: flashes are rejected at the frame level (arc_filter).

        if used_complete:
            n_complete_used += 1

        fname = grp['file'].iloc[0]
        ei    = (electrode_info or {}).get(fname, {})
        records.append({
            'event_id':        int(eid),
            'file':            fname,
            'frame_start':     int(grp['frame'].iloc[0]),
            'frame_end':       int(grp['frame'].iloc[-1]),
            'duration_frames': int(grp['frame'].iloc[-1] - grp['frame'].iloc[0]),
            'n_frames_total':  len(grp),
            'n_frames_front':  len(front),
            'used_complete':   int(used_complete),
            'sub_event_count': sub_count,
            'vel_ms':          round(vel_ms, 4),
            'r2':              round(r2, 4),
            'tip_row_start':   round(float(r_front[0]), 2),
            'tip_row_end':     round(float(r_front[-1]), 2),
            'extension_px':    round(float(r_front.max() - r_front.min()), 2),
            'electrode_row':   round(ei.get('electrode_row', np.nan), 2),
            'l_eff_px':        round(ei.get('l_eff_px', np.nan), 2),
        })

    events = pd.DataFrame(records)
    if len(events) == 0:
        print("[WARN] No events fitted.")
        return events

    n_down = (events['vel_ms'] > 0).sum()
    print(f"Events fitted: {len(events)}  |  "
          f"downward (vel > 0): {n_down} ({100*n_down/max(len(events),1):.0f}%)  |  "
          f"using complete-arc frames: {n_complete_used} "
          f"({100*n_complete_used/max(len(events),1):.0f}%)")
    return events

# ---
# Mode A / Mode B classification
# ---

def classify_modes(events, threshold=NUCLEATION_SLACK_FRAC):
    """
    Label each event as Mode A (fresh gap initiation) or Mode B
    (reconnection / back-breakdown at an intermediate position).

    Primary criterion — norm_nucleation_pos, a dimensionless depth in [0, 1]:
        norm_nucleation_pos = (tip_row_start − electrode_row) / L_eff_px
    where L_eff_px = p90(tip_rows) − electrode_row is the effective arc
    range for that recording (same quantity used by the event detector).

    The threshold defaults to NUCLEATION_SLACK_FRAC, making the classifier
    self-consistent with the event detector: any event the detector considers
    "returned to the gap" has norm_nucleation_pos ≤ threshold by construction.

    Also adds inter_event_gap_frames (frames since the prior event in the same
    file ended) as a secondary observable.  Mode B events are expected to have
    shorter gaps, but this column is stored for validation and is not used for
    primary classification.
    """
    if events.empty:
        print("[WARN] classify_modes: no events to classify.")
        return events
    events = events.copy().sort_values(['file', 'frame_start']).reset_index(drop=True)

    has_geom = 'electrode_row' in events.columns and 'l_eff_px' in events.columns
    if has_geom:
        l_eff = events['l_eff_px'].clip(lower=1.0)
        events['norm_nucleation_pos'] = (
            (events['tip_row_start'] - events['electrode_row']) / l_eff
        ).clip(0.0, 1.0).round(4)
    else:
        events['norm_nucleation_pos'] = np.nan
        print("[WARN] classify_modes: electrode geometry missing — run fit_events() "
              "with electrode_info.")

    events['inter_event_gap_frames'] = np.nan
    for fname, grp in events.groupby('file'):
        grp = grp.sort_values('frame_start')
        if len(grp) < 2:
            continue
        gaps = grp['frame_start'].values[1:] - grp['frame_end'].values[:-1]
        events.loc[grp.index[1:], 'inter_event_gap_frames'] = gaps

    if has_geom:
        events['mode'] = np.where(events['norm_nucleation_pos'] > threshold, 'B', 'A')
    else:
        events['mode'] = 'unknown'

    n_a   = (events['mode'] == 'A').sum()
    n_b   = (events['mode'] == 'B').sum()
    n_tot = len(events)
    print(f"\nMode classification  (threshold = {threshold:.2f} × L_eff_px):")
    print(f"  Mode A (gap initiation) : {n_a:4d}  ({100*n_a/max(n_tot,1):.0f}%)")
    print(f"  Mode B (reconnection)   : {n_b:4d}  ({100*n_b/max(n_tot,1):.0f}%)")
    if has_geom:
        med_a = events.loc[events['mode'] == 'A', 'norm_nucleation_pos'].mean()
        med_b = events.loc[events['mode'] == 'B', 'norm_nucleation_pos'].mean()
        print(f"  Mean norm_nucleation — A: {med_a:.3f}   B: {med_b:.3f}")
    return events

# ---
# Sliding-window local velocity
# ---

def compute_local_velocities(df):
    """
    Sliding-window OLS velocity for every advancing-front frame.

    For each advancing-front frame k, fit a line to the LOCAL_VEL_N nearest
    advancing-front neighbors centered on k:
        window = f_front[ max(0, k-N//2) : min(n, k+N//2+1) ]

    This is equivalent to a Savitzky-Golay first-derivative filter: it gives
    a time-resolved velocity that remains resistant to noise.  The same
    complete-arc / fallback selection logic as fit_events() is applied so that
    the local velocity is consistent with the event regression.

    Returns:
        frame_vel : dict {(file, frame) → local_vel_ms}
            Only advancing-front frames are present; all others map to NaN in CSV.
        dyn : DataFrame with columns [event_id, norm_pos, vel_ms, extension, file]
            Used for density and heatmap plots.
    """
    detected    = df[df['event_id'] >= 0].dropna(subset=['arc_tip_row'])
    detected    = detected[~arc_filter.flash_mask(detected)]   # criterion 3: drop flashes
    frame_vel   = {}
    dyn_records = []
    half        = LOCAL_VEL_N // 2

    for eid, grp in detected.groupby('event_id'):
        grp   = grp.sort_values('frame')
        fname = grp['file'].iloc[0]

        if 'arc_complete' in grp.columns:
            mask  = pd.to_numeric(grp['arc_complete'], errors='coerce').fillna(0).astype(bool)
            fit_g = grp[mask]
            if len(fit_g) < MIN_EVENT_FRAMES:
                fit_g = grp
        else:
            fit_g = grp

        frames      = fit_g['frame'].values.astype(float)
        rows        = fit_g['arc_tip_row'].values.astype(float)
        orig_frames = fit_g['frame'].values

        front = advancing_front(frames, rows)
        if len(front) < MIN_EVENT_FRAMES:
            continue

        f_front    = frames[front]
        r_front    = rows[front]
        orig_front = orig_frames[front]

        ext = r_front[-1] - r_front[0]
        n   = len(f_front)

        for k in range(n):
            lo = max(0, k - half)
            hi = min(n, k + half + 1)
            if hi - lo < 3:
                continue

            slope, _ = np.polyfit(f_front[lo:hi], r_front[lo:hi], 1)
            vel_ms   = slope * FPS / PX_PER_MM / 1000
            # No velocity cap: flashes are rejected at the frame level (arc_filter).

            frame_vel[(fname, int(orig_front[k]))] = vel_ms

            if ext >= MIN_EXTENSION_PX:
                norm_pos = float(np.clip((r_front[k] - r_front[0]) / ext, 0.0, 1.0))
                dyn_records.append({
                    'event_id':  int(eid),
                    'norm_pos':  norm_pos,
                    'vel_ms':    float(vel_ms),
                    'extension': float(ext),
                    'file':      fname,
                })

    print(f"Local velocities computed: {len(frame_vel):,} advancing-front frames  "
          f"(window N={LOCAL_VEL_N})")
    return frame_vel, pd.DataFrame(dyn_records)

# ---
# Merge and output
# ---

def merge_back(df, events, frame_vel=None):
    """
    Attach event_vel_ms (event regression scalar) and local_vel_ms
    (per-frame sliding-window OLS) to every row of df.

    local_vel_ms is NaN for retreating frames, non-detected frames, and
    events that were too short to fit.  Downstream velocity maps should use
    local_vel_ms where available and fall back to event_vel_ms otherwise.
    """
    df = df.copy()
    if events.empty:
        df['event_vel_ms'] = np.nan
        df['local_vel_ms'] = np.nan
        return df
    vel_map = events.set_index('event_id')['vel_ms'].to_dict()
    df = df.copy()
    df['event_vel_ms'] = df['event_id'].map(vel_map)

    if frame_vel:
        lv = pd.DataFrame(
            [{'file': f, 'frame': int(fr), 'local_vel_ms': v}
             for (f, fr), v in frame_vel.items()]
        )
        df['frame']  = pd.to_numeric(df['frame'], errors='coerce')
        lv['frame']  = lv['frame'].astype(df['frame'].dtype)
        df = df.merge(lv, on=['file', 'frame'], how='left')
    else:
        df['local_vel_ms'] = np.nan

    return df


def add_local_velocity_stats(events, df_merged):
    """Attach per-event time-resolved (local sliding-window) velocity statistics
    to the events table, alongside the whole-event average ``vel_ms``.

    The local-velocity *profile* is the primary characterization of an event;
    these are its summary statistics, derived from the per-frame ``local_vel_ms``
    on advancing-front frames (``local_vel_ms`` is NaN elsewhere). The full
    per-frame profile for each event is in events/event_NNNN_dynamics.csv.

    Added columns:
        local_vel_median / mean   central tendency of the local velocity
        local_vel_p10 / p90       within-event spread (deceleration→acceleration)
        local_vel_peak (p95)/max  fastest instantaneous propagation
        local_vel_std             variability (steadiness of the glide)
        n_local_frames            advancing-front frames contributing
    """
    cols = ['local_vel_median', 'local_vel_mean', 'local_vel_p10', 'local_vel_p90',
            'local_vel_peak', 'local_vel_max', 'local_vel_std', 'n_local_frames']
    if events.empty or 'local_vel_ms' not in df_merged.columns:
        for c in cols:
            events[c] = np.nan
        return events
    g = df_merged.dropna(subset=['local_vel_ms'])
    g = g[g['local_vel_ms'] > 0]
    if len(g) == 0:
        for c in cols:
            events[c] = np.nan
        return events
    grp = g.groupby('event_id')['local_vel_ms']
    stats = grp.agg(
        local_vel_median='median',
        local_vel_mean='mean',
        local_vel_p10=lambda x: x.quantile(0.10),
        local_vel_p90=lambda x: x.quantile(0.90),
        local_vel_peak=lambda x: x.quantile(0.95),
        local_vel_max='max',
        local_vel_std='std',
        n_local_frames='count',
    ).reset_index()
    for c in ['local_vel_median', 'local_vel_mean', 'local_vel_p10', 'local_vel_p90',
              'local_vel_peak', 'local_vel_max', 'local_vel_std']:
        stats[c] = stats[c].round(4)
    events = events.merge(stats, on='event_id', how='left')
    n = int(events['n_local_frames'].notna().sum())
    print(f"Per-event local-velocity stats added for {n} events "
          f"(median/mean/p10/p90/peak/max/std).")
    return events


def save_per_event_data(df_merged, events, dyn, out_dir):
    """
    Write per-event CSVs to out_dir/events/.

        events_index.csv         — full events table (authoritative)
        event_NNNN_frames.csv    — all rows for event NNNN
        event_NNNN_dynamics.csv  — norm_pos / local velocity for event NNNN
    """
    ev_dir = out_dir / 'events'
    ev_dir.mkdir(exist_ok=True)

    events.rename(columns={'mode': 'mode_eventstart_legacy'}).to_csv(
        ev_dir / 'events_index.csv', index=False)
    print(f"Saved: {ev_dir / 'events_index.csv'}  ({len(events)} events)")

    detected = df_merged[df_merged['event_id'] >= 0]
    n_frames = 0
    for eid, grp in detected.groupby('event_id'):
        grp.to_csv(ev_dir / f'event_{int(eid):04d}_frames.csv', index=False)
        n_frames += 1
    print(f"Saved per-event frames: {n_frames} files → {ev_dir}/")

    if dyn is not None and len(dyn) > 0:
        n_dyn = 0
        for eid, grp in dyn.groupby('event_id'):
            grp.to_csv(ev_dir / f'event_{int(eid):04d}_dynamics.csv', index=False)
            n_dyn += 1
        print(f"Saved per-event dynamics: {n_dyn} files → {ev_dir}/")


def print_summary(events):
    down = events[events['vel_ms'] > 0]['vel_ms']
    print(f"\n{'─' * 60}")
    print(f"  Event velocity summary  (downward events only)")
    print(f"{'─' * 60}")
    if len(down) == 0:
        print("  No downward-moving events found.")
        print(f"{'─' * 60}\n")
        return

    print(f"  N primary events   : {len(events)}")
    print(f"  N downward events  : {len(down)}")
    print(f"  Mean vel.        : {np.mean(down):.3f} m/s")
    print(f"  Mean vel.          : {np.mean(down):.3f} m/s")
    print(f"  25th – 75th pct    : "
          f"{np.percentile(down,25):.3f} – {np.percentile(down,75):.3f} m/s")
    print(f"  95th pct           : {np.percentile(down,95):.3f} m/s")

    r2_good = (events['r2'] >= 0.5).sum()
    print(f"  Events R²≥0.5      : {r2_good} ({100*r2_good/max(len(events),1):.0f}%)")

    if 'sub_event_count' in events.columns:
        sub      = events['sub_event_count']
        with_sub = (sub > 0).sum()
        print(f"  Events with sub-ev.: {with_sub} "
              f"({100*with_sub/max(len(events),1):.0f}%),  "
              f"mean {sub[sub>0].mean():.1f} sub-events each")

    if 'used_complete' in events.columns:
        n_comp = events['used_complete'].sum()
        print(f"  Fitted on complete : {n_comp} ({100*n_comp/max(len(events),1):.0f}%) events")

    if 'mode' in events.columns:
        n_a = (events['mode'] == 'A').sum()
        n_b = (events['mode'] == 'B').sum()
        print(f"  Mode A (gap init)  : {n_a} ({100*n_a/max(len(events),1):.0f}%)")
        print(f"  Mode B (reconnect) : {n_b} ({100*n_b/max(len(events),1):.0f}%)")
        if n_a > 0 and n_b > 0:
            va = events.loc[(events['mode'] == 'A') & (events['vel_ms'] > 0), 'vel_ms']
            vb = events.loc[(events['mode'] == 'B') & (events['vel_ms'] > 0), 'vel_ms']
            if len(va) and len(vb):
                print(f"  Mean vel A / B   : "
                      f"{np.mean(va):.3f} / {np.mean(vb):.3f} m/s")

    print(f"{'─' * 60}\n")

# ---
# Figures
# ---

def plot_distribution_by_mode(events, out_dir):
    """
    Velocity histogram stratified by Mode A / Mode B.
    Saves two separate single-panel figures:
      event_vel_hist_all.png     — all events with per-mode step overlays
      event_vel_hist_by_mode.png — Mode A vs B filled histograms (if modes present)
    """
    down = events[events['vel_ms'] > 0].copy()
    if len(down) == 0:
        return

    has_mode = 'mode' in down.columns and down['mode'].nunique() > 1
    vmax     = float(np.percentile(down['vel_ms'], 99))
    bins     = np.linspace(0, vmax, 35)
    colors   = {'A': '#1565C0', 'B': '#C62828'}

    # Panel 1: all-events histogram with per-mode step overlays
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(down['vel_ms'].clip(upper=vmax), bins=bins,
            color='#90A4AE', alpha=0.6, label='All')
    if has_mode:
        for mode, grp in down.groupby('mode'):
            v = grp['vel_ms'].clip(upper=vmax)
            ax.hist(v, bins=bins, histtype='step', lw=2,
                    color=colors.get(mode, 'grey'),
                    label=f'Mode {mode}')
    ax.set_xlabel('Arc-tip velocity (m/s)')
    ax.set_ylabel('Number of events')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    out1 = out_dir / 'event_vel_hist_all.png'
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out1}")

    # Panel 2: Mode A vs B filled distributions (only when modes are present)
    if has_mode:
        fig, ax = plt.subplots(figsize=(7, 4))
        for mode, grp in down.groupby('mode'):
            v = grp['vel_ms'].clip(upper=vmax)
            ax.hist(v, bins=bins, alpha=0.65,
                    color=colors.get(mode, 'grey'),
                    label=f'Mode {mode}')
        ax.axvline(np.mean(down['vel_ms']), color='#37474F', lw=1.5, ls=':',
                   label='Overall')
        ax.set_xlabel('Arc-tip velocity (m/s)')
        ax.set_ylabel('Number of events')
        ax.legend(frameon=False, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        out2 = out_dir / 'event_vel_hist_by_mode.png'
        fig.savefig(out2, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out2}")


def plot_scatter(events, out_dir):
    """
    Saves two single-panel figures:
      event_vel_scatter.png — velocity vs event duration, coloured by R²
      event_r2_hist.png     — histogram of R² values
    """
    down = events[events['vel_ms'] > 0].copy()
    if len(down) == 0:
        return

    # Panel 1: scatter
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sc = ax.scatter(down['duration_frames'], down['vel_ms'],
                    c=down['r2'], cmap='viridis',
                    alpha=0.6, s=20, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label='R2')
    ax.set_xlabel('Event duration (frames)')
    ax.set_ylabel('Velocity (m/s)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out1 = out_dir / 'event_vel_scatter.png'
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out1}")

    # Panel 2: R² histogram
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(down['r2'].dropna(), bins=30, color='#4CAF50', alpha=0.8)
    ax.set_xlabel('R2 of advancing-front regression')
    ax.set_ylabel('Number of events')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out2 = out_dir / 'event_r2_hist.png'
    fig.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out2}")


def plot_comparison(df_merged, out_dir):
    """
    Saves two single-panel figures:
      vel_method_framediff.png  — frame-to-frame Δ=1 velocity histogram
      vel_method_regression.png — event advancing-front regression histogram
    """
    if 'arc_tip_vel_mm_s' not in df_merged.columns:
        return
    old = df_merged['arc_tip_vel_mm_s'].dropna() / 1000
    new = df_merged['event_vel_ms'].dropna()
    new = new[new > 0]
    if len(old) == 0 or len(new) == 0:
        return

    cap = np.percentile(old, 97)

    # Panel 1: frame-to-frame
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(old.clip(upper=cap), bins=60, color='#FF5722', alpha=0.75)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Count')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out1 = out_dir / 'vel_method_framediff.png'
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out1}")

    # Panel 2: event regression
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(new, bins=40, color='#2196F3', alpha=0.75)
    ax.set_xlabel('Velocity (m/s)')
    ax.set_ylabel('Count')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out2 = out_dir / 'vel_method_regression.png'
    fig.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out2}")


def plot_dynamics_density(dyn, out_dir):
    """
    Hexbin density of local velocity vs normalised arc position, one panel
    per recording file.  Overlaid orange curves show p10/p50/p90 percentiles
    across events in each position bin.

    Y-axis is clipped at the 99th percentile to suppress extreme outliers
    without removing them from the density count.  A flat p50 curve indicates
    constant-velocity propagation; a rising curve indicates acceleration with
    arc extension; a wide p10–p90 band indicates high event-to-event variability.
    """
    dyn = dyn[dyn['vel_ms'] > 0].copy()
    if len(dyn) == 0:
        print("[WARN] No dynamics data — skipping density plot.")
        return

    files  = list(dyn['file'].unique())
    vcap   = float(np.percentile(dyn['vel_ms'], 99))
    bins   = np.linspace(0, 1, N_POSITION_BINS + 1)
    bctrs  = 0.5 * (bins[:-1] + bins[1:])

    for run_idx, fname in enumerate(files):
        sub    = dyn[dyn['file'] == fname].copy()
        n_evts = sub['event_id'].nunique()
        run_label = f'Run{run_idx + 1}'

        fig, ax = plt.subplots(figsize=(5, 5))
        hb = ax.hexbin(sub['norm_pos'], sub['vel_ms'].clip(upper=vcap),
                       gridsize=40, cmap='Blues', bins='log',
                       extent=[0, 1, 0, vcap])
        plt.colorbar(hb, ax=ax, label='log10(count)')

        for pct, lw, ls, lbl in [(10, 1.0, '--', 'p10'),
                                  (50, 2.2, '-',  'p50'),
                                  (90, 1.0, '--', 'p90')]:
            pct_vals = [
                np.percentile(sub[(sub['norm_pos'] >= lo) &
                                  (sub['norm_pos'] <  hi)]['vel_ms'], pct)
                if len(sub[(sub['norm_pos'] >= lo) &
                           (sub['norm_pos'] <  hi)]) >= 3 else np.nan
                for lo, hi in zip(bins[:-1], bins[1:])
            ]
            pct_vals = np.array(pct_vals)
            ok = ~np.isnan(pct_vals)
            if ok.any():
                ax.plot(bctrs[ok], np.clip(pct_vals[ok], 0, vcap),
                        color='#FF6F00', lw=lw, ls=ls, label=lbl)

        ax.set_xlabel('Normalised arc position  (0 = nucleation,  1 = max extent)')
        ax.set_ylabel('Local arc-tip velocity (m/s)')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, vcap)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        out = out_dir / f'event_dynamics_density_{run_label.lower()}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out}")


def plot_dynamics_heatmap(dyn, events, out_dir):
    """
    Per-event velocity heatmap: rows = individual events sorted by regression
    velocity, columns = normalised position bins, colour = mean local velocity
    in that bin.  White cells mean no advancing-front frames fell in that bin.

    Useful for spotting whether velocity depends on arc position (horizontal
    gradient), event identity (vertical banding), or both.
    """
    dyn = dyn[dyn['vel_ms'] > 0].copy()
    if len(dyn) == 0:
        print("[WARN] No dynamics data — skipping heatmap.")
        return

    vel_lookup = events.set_index('event_id')['vel_ms'].to_dict()
    files      = list(dyn['file'].unique())
    vcap       = float(np.percentile(dyn['vel_ms'], 99))
    bins       = np.linspace(0, 1, N_POSITION_BINS + 1)
    bctrs      = 0.5 * (bins[:-1] + bins[1:])

    for run_idx, fname in enumerate(files):
        sub   = dyn[dyn['file'] == fname]
        eids  = sorted(sub['event_id'].unique(),
                       key=lambda e: vel_lookup.get(e, 0))
        run_label = f'Run{run_idx + 1}'

        mat = np.full((len(eids), N_POSITION_BINS), np.nan)
        for row_i, eid in enumerate(eids):
            grp = sub[sub['event_id'] == eid]
            for col_j, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
                cell = grp[(grp['norm_pos'] >= lo) & (grp['norm_pos'] < hi)]['vel_ms']
                if len(cell) >= 1:
                    mat[row_i, col_j] = float(cell.mean())

        mat  = np.clip(mat, 0, vcap)
        cmap = plt.get_cmap('plasma').copy()
        cmap.set_bad('white')

        fig, ax = plt.subplots(figsize=(5, 6))
        im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=0, vmax=vcap,
                       extent=[0, 1, 0, len(eids)], origin='lower')
        plt.colorbar(im, ax=ax, label='Mean local velocity (m/s)')

        ax.set_xticks(bctrs[::2])
        ax.set_xticklabels([f'{b:.1f}' for b in bctrs[::2]])
        ax.set_xlabel('Normalised arc position')
        ax.set_ylabel('Events  (sorted by regression velocity up)')

        fig.tight_layout()
        out = out_dir / f'event_dynamics_heatmap_{run_label.lower()}.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {out}")


def plot_nucleation_distribution(events, out_dir, threshold=NUCLEATION_SLACK_FRAC):
    """
    Two-panel diagnostic for the Mode A/B classification.

    Left: histogram of norm_nucleation_pos with the A/B threshold marked.
    A bimodal distribution supports the A/B separation; a unimodal one
    suggests most events are the same mode.

    Right: scatter of nucleation position vs inter-event gap, coloured by
    mode.  Mode B events are expected to cluster at high position AND short
    gap (the previous channel has not fully cooled before back-breakdown).
    """
    if 'norm_nucleation_pos' not in events.columns:
        return

    down = events[events['vel_ms'] > 0].copy()
    nn   = down['norm_nucleation_pos'].dropna()
    if len(nn) == 0:
        return

    mode_colors = {'A': '#1565C0', 'B': '#C62828', 'unknown': '#546E7A'}

    # Panel 1: nucleation position histogram
    n_a = (nn <= threshold).sum()
    n_b = (nn >  threshold).sum()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(nn, bins=25, color='#546E7A', alpha=0.85, edgecolor='white', lw=0.5)
    ax.axvline(threshold, color='#E53935', lw=2, ls='--',
               label=f'A/B threshold = {threshold:.2f}')
    ax.set_xlabel('Normalised nucleation position\n'
                  '(0 = electrode gap,  1 = typical max arc extent)',
                  fontsize=10)
    ax.set_ylabel('Number of events')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    out1 = out_dir / 'nucleation_hist.png'
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out1}")

    # Panel 2: nucleation position vs inter-event gap scatter
    if 'inter_event_gap_frames' in down.columns and 'mode' in down.columns:
        both = down.dropna(subset=['inter_event_gap_frames', 'norm_nucleation_pos'])
        if len(both) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            for mode, grp in both.groupby('mode'):
                ax.scatter(grp['norm_nucleation_pos'],
                           grp['inter_event_gap_frames'],
                           c=mode_colors.get(mode, '#546E7A'),
                           s=20, alpha=0.55, label=f'Mode {mode}')
            ax.axvline(threshold, color='#E53935', lw=1.5, ls='--',
                       label=f'threshold = {threshold:.2f}')
            ax.set_xlabel('Normalised nucleation position', fontsize=10)
            ax.set_ylabel('Inter-event gap (frames)')
            ax.legend(frameon=False, fontsize=9)
            ax.grid(alpha=0.25)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            out2 = out_dir / 'nucleation_scatter.png'
            fig.savefig(out2, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {out2}")

# ---
# Main
# ---

def main():
    # Declared global up front: PX_PER_MM and FPS are read in the argparse help
    # strings below (and reassigned after parsing), so the declaration must
    # precede their first use in this function.
    global PX_PER_MM, FPS
    import argparse
    ap = argparse.ArgumentParser(description='Event velocity — advancing-front regression')
    ap.add_argument('csv',         nargs='?', default=None,
                    help='measurements CSV (default: measurements_all.csv in script dir)')
    ap.add_argument('out_dir',     nargs='?', default=None,
                    help='Output directory (default: same folder as CSV)')
    ap.add_argument('--px-per-mm', type=float, default=None,
                    help='Spatial calibration in px/mm.  REQUIRED to match the recording; '
                         'default %.4f (ADCB).  Pass 8.8 for 11MAY26.' % PX_PER_MM)
    ap.add_argument('--fps',       type=float, default=None,
                    help='Frame rate in fps.  Default %.0f.' % FPS)
    args = ap.parse_args()

    # Override module-level calibration if supplied on the command line.
    # These globals are used by fit_events(), compute_local_velocities(), and
    # every velocity-to-speed conversion below, so setting them here propagates
    # everywhere without touching function signatures. (Declared global at the
    # top of main(), since they are also read in the argparse help strings.)
    if args.px_per_mm is not None:
        PX_PER_MM = args.px_per_mm
    if args.fps is not None:
        FPS = args.fps

    script_dir = Path(__file__).parent.resolve()
    csv_path   = Path(args.csv)     if args.csv     else script_dir / 'measurements_all.csv'
    out_dir    = Path(args.out_dir) if args.out_dir else csv_path.parent

    if not csv_path.exists():
        print(f"[ERROR] Not found: {csv_path}")
        sys.exit(1)
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'=' * 55}")
    print(f"  Event velocity — advancing-front regression")
    print(f"  CSV        : {csv_path}")
    print(f"  Output     : {out_dir}")
    print(f"  px/mm      : {PX_PER_MM:.4f}")
    print(f"  FPS        : {FPS:.0f}")
    print(f"{'=' * 55}\n")

    df                 = load(csv_path)
    df, electrode_info = assign_events(df)
    events             = fit_events(df, electrode_info=electrode_info)
    events             = classify_modes(events)
    frame_vel, dyn     = compute_local_velocities(df)
    df_merged          = merge_back(df, events, frame_vel)
    events             = add_local_velocity_stats(events, df_merged)

    # The per-event 'mode' here is the LEGACY event-start classifier, which is
    # Mode-A-biased by construction (a new event begins when the tip returns to
    # the gap). Write it under an explicit legacy name so it is not confused with
    # the authoritative re-strike Mode A/B in re_strikes_qa.csv (qa_restrike.py).
    events_out = events.rename(columns={'mode': 'mode_eventstart_legacy'})
    events_out.to_csv(out_dir / 'event_summary.csv', index=False)
    print(f"Saved: {out_dir / 'event_summary.csv'}")
    print("NOTE: 'mode_eventstart_legacy' is the legacy (Mode-A-biased) per-event "
          "classifier. Authoritative Mode A/B is per re-strike in re_strikes_qa.csv.")

    df_merged.to_csv(out_dir / 'measurements_all_events.csv', index=False)
    print(f"Saved: {out_dir / 'measurements_all_events.csv'}")

    save_per_event_data(df_merged, events, dyn, out_dir)
    print_summary(events)

    plot_distribution_by_mode(events, out_dir)
    plot_scatter(events, out_dir)
    plot_comparison(df_merged, out_dir)
    plot_dynamics_density(dyn, out_dir)
    plot_dynamics_heatmap(dyn, events, out_dir)
    plot_nucleation_distribution(events, out_dir)

    print("Done.\n")


if __name__ == '__main__':
    main()
