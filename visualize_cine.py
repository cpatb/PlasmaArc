"""
visualize_cine.py  —  Figures for J-electrode CINE arc analysis
================================================================
Reads:  measurements.csv   (produced by analyze_cine.py)
        metadata.txt       (optional — supplies FPS, lo/hi, px_per_mm)

Usage:
    python visualize_cine.py                              # defaults in CONFIG
    python visualize_cine.py path/to/measurements.csv out_dir
    python visualize_cine.py path/to/measurements.csv out_dir --px-per-mm=8.5

Figures produced
----------------
  fig01  Kymograph               arc-tip row vs time (complete / partial)
  fig02  Arc-tip 2D map          tip column vs tip row — where the arc travels
  fig03  Lateral drift           arc-tip column vs time
  fig04  Arc length over time    rolling median arc length
  fig05  Arc area over time      rolling median mask area
  fig06  Electrode gap over time rolling median tip-to-tip span
  fig07  Arc intensity over time mean + peak normalised intensity
  fig08  Detection rate          rolling arc detection fraction
  fig09  Complete arc fraction   rolling complete vs partial fraction
  fig10  Arc length distribution histogram / KDE
  fig11  Arc area distribution   histogram
  fig12  Arc width distribution  histogram
  fig13  Velocity time-series    frame-to-frame tip velocity
  fig14  Velocity distribution   KDE + CDF
  fig15  Velocity vs depth       median + IQR in depth bins
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================

_J_ARC_DIR   = Path(r'C:\Users\compl\OneDrive\Desktop\J electrode arc')
DEFAULT_CSV  = _J_ARC_DIR / 'analysis' / 'measurements.csv'
DEFAULT_OUT  = _J_ARC_DIR / 'analysis' / 'figures'

FPS_FALLBACK = 200_000      # used if metadata.txt not found
PX_PER_MM    = None         # override with --px-per-mm=X  or set here once known

# Rolling-window width (frames) for smoothed time-series plots
ROLL_WIN = 500

# Velocity outlier clip (percentile) — removes rare tracking glitches
VEL_CLIP_PCT = 99.0

# Colour palette
ARC_COLOR     = '#1f77b4'   # complete arc
PARTIAL_COLOR = '#ff7f0e'   # partial / leader arc

SINGLE_W = 3.5
DOUBLE_W = 7.0
PANEL_H  = 2.8
DPI      = 300

plt.rcParams.update({
    'font.size':          9,
    'axes.labelsize':    10,
    'axes.titlesize':    10,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    8,
    'axes.linewidth':     0.8,
    'lines.linewidth':    1.0,
    'font.family':        'sans-serif',
    'figure.dpi':         DPI,
    'savefig.dpi':        DPI,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.05,
})

# =============================================================================
# DATA LOADING
# =============================================================================

def read_metadata(csv_path):
    """Try to read FPS and px_per_mm from the sibling metadata.txt."""
    meta = {}
    meta_path = Path(csv_path).parent / 'metadata.txt'
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            if ':' in line:
                k, v = line.split(':', 1)
                meta[k.strip()] = v.strip()
    fps = int(meta.get('fps', FPS_FALLBACK))
    try:
        px_mm = float(meta.get('px_per_mm', 'None'))
    except (ValueError, TypeError):
        px_mm = None
    return fps, px_mm


def load_data(csv_path, fps, px_per_mm):
    df = pd.read_csv(csv_path)

    numeric_cols = [
        'arc_tip_row', 'arc_tip_col',
        'arc_length_px', 'arc_length_mm',
        'arc_area_px', 'arc_width_px_mean',
        'centroid_row', 'centroid_col',
        'mean_intensity', 'peak_intensity',
        'tip_a_row', 'tip_a_col', 'tip_b_row', 'tip_b_col',
        'electrode_gap_px', 'skeleton_px',
        'arc_tip_vel_mm_s', 'arc_complete',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.sort_values('frame').reset_index(drop=True)

    # Time axis
    df['time_ms'] = df['frame'] / fps * 1e3

    # Detected subset
    det = df[df['arc_detected'] == 1].copy()
    det['is_complete'] = det['arc_complete'].fillna(0).astype(bool)
    det['is_partial']  = ~det['is_complete']

    # Pixel velocity from consecutive detected frames
    # (used when PX_PER_MM is not set; also the source of truth when it is)
    det = det.copy()
    det['vel_px_per_frame'] = float('nan')
    frames = det['frame'].values
    rows   = det['arc_tip_row'].values
    idx    = det.index.values
    for k in range(1, len(frames)):
        if int(frames[k]) - int(frames[k-1]) == 1:
            det.loc[idx[k], 'vel_px_per_frame'] = float(rows[k] - rows[k-1])

    # Convert to m/s when calibration is available
    if px_per_mm is not None:
        det['vel_m_s'] = det['vel_px_per_frame'] * fps / (px_per_mm * 1000.0)
        det['tip_depth_mm'] = det['arc_tip_row'] / px_per_mm
        det['arc_length_mm_calc'] = det['arc_length_px'] / px_per_mm
        det['arc_width_mm_calc']  = det['arc_width_px_mean'] / px_per_mm
        det['electrode_gap_mm']   = det['electrode_gap_px'] / px_per_mm
        vel_col   = 'vel_m_s'
        vel_label = 'Arc-tip velocity (m/s)'
        len_col   = 'arc_length_mm_calc'
        len_label = 'Arc length (mm)'
        wid_col   = 'arc_width_mm_calc'
        wid_label = 'Arc width (mm)'
        dep_col   = 'tip_depth_mm'
        dep_label = 'Tip depth (mm)'
        gap_col   = 'electrode_gap_mm'
        gap_label = 'Electrode gap (mm)'
    else:
        det['vel_m_s'] = float('nan')
        dep_col   = 'arc_tip_row'
        dep_label = 'Tip row (px)'
        vel_col   = 'vel_px_per_frame'
        vel_label = 'Arc-tip velocity (px/frame)'
        len_col   = 'arc_length_px'
        len_label = 'Arc length (px)'
        wid_col   = 'arc_width_px_mean'
        wid_label = 'Arc width (px)'
        gap_col   = 'electrode_gap_px'
        gap_label = 'Electrode gap (px)'

    units = dict(
        vel_col=vel_col, vel_label=vel_label,
        len_col=len_col, len_label=len_label,
        wid_col=wid_col, wid_label=wid_label,
        dep_col=dep_col, dep_label=dep_label,
        gap_col=gap_col, gap_label=gap_label,
    )
    return df, det, units


# =============================================================================
# HELPERS
# =============================================================================

def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    plt.close(fig)
    print('  ' + name)


def clean_vel(series, pct_hi=VEL_CLIP_PCT):
    """Drop NaN, zero, and extreme-outlier velocities."""
    v = series.dropna()
    # For m/s, keep positive (downstream) only; for px/frame allow negative
    # (arc retreating) but clip extremes in both directions.
    if len(v) < 5:
        return v
    lo = np.percentile(v, 100 - pct_hi)
    hi = np.percentile(v, pct_hi)
    return v[(v >= lo) & (v <= hi)]


def rolling_on_allframes(df_all, det, flag_col, win=ROLL_WIN):
    """Rolling fraction of all frames that match flag_col in det."""
    flags = df_all['frame'].isin(
        det.loc[det[flag_col], 'frame']
    ).astype(float)
    return flags.rolling(win, min_periods=win // 5, center=True).mean() * 100


# =============================================================================
# FIGURES
# =============================================================================

def fig_kymograph(det, df_all, units, out_dir):
    """Fig 01 — arc-tip depth vs time, coloured by arc type."""
    dep_col   = units['dep_col']
    dep_label = units['dep_label']

    complete = det[det['is_complete']]
    partial  = det[det['is_partial']]

    fig, ax = plt.subplots(figsize=(DOUBLE_W, PANEL_H))
    ax.scatter(complete['time_ms'], complete[dep_col],
               s=0.4, c=ARC_COLOR, alpha=0.4, linewidths=0, rasterized=True,
               label='Complete arc')
    ax.scatter(partial['time_ms'], partial[dep_col],
               s=0.4, c=PARTIAL_COLOR, alpha=0.4, linewidths=0, rasterized=True,
               label='Partial / leader')

    handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=ARC_COLOR,
               ms=4, label='Complete arc'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=PARTIAL_COLOR,
               ms=4, label='Partial / leader'),
    ]
    ax.legend(handles=handles, loc='lower right', framealpha=0.7)
    ax.invert_yaxis()
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(dep_label)
    save(fig, out_dir, 'fig01_kymograph.png')


def fig_tip_2d_map(det, units, out_dir):
    """Fig 02 — spatial map of arc-tip position (col vs row)."""
    dep_col   = units['dep_col']
    dep_label = units['dep_label']

    complete = det[det['is_complete']]
    partial  = det[det['is_partial']]

    fig, ax = plt.subplots(figsize=(PANEL_H + 0.5, PANEL_H + 0.5))
    ax.scatter(complete['arc_tip_col'], complete[dep_col],
               s=0.5, c=ARC_COLOR, alpha=0.3, linewidths=0, rasterized=True,
               label='Complete')
    ax.scatter(partial['arc_tip_col'], partial[dep_col],
               s=0.5, c=PARTIAL_COLOR, alpha=0.3, linewidths=0, rasterized=True,
               label='Partial')

    # Mark mean electrode gap as a horizontal reference line
    if det['electrode_gap_px'].notna().any():
        mean_gap_row = det['arc_tip_row'].min() if dep_col == 'arc_tip_row' else \
                       det['tip_depth_mm'].min() if 'tip_depth_mm' in det.columns else None
        if mean_gap_row is not None:
            ax.axhline(mean_gap_row, color='gray', lw=0.7, ls='--',
                       alpha=0.6, label='Nucleation zone')

    handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=ARC_COLOR,
               ms=4, label='Complete'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=PARTIAL_COLOR,
               ms=4, label='Partial'),
    ]
    ax.legend(handles=handles, framealpha=0.7, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Tip column (px)')
    ax.set_ylabel(dep_label)
    save(fig, out_dir, 'fig02_tip_2d_map.png')


def fig_lateral_drift(det, out_dir):
    """Fig 03 — arc-tip column position over time."""
    fig, ax = plt.subplots(figsize=(DOUBLE_W, PANEL_H))
    ax.scatter(det['time_ms'], det['arc_tip_col'],
               s=0.4, c='#9467bd', alpha=0.3, linewidths=0, rasterized=True)
    roll = (det.set_index('time_ms')['arc_tip_col']
               .rolling(ROLL_WIN, min_periods=ROLL_WIN // 5, center=True)
               .mean())
    ax.plot(roll.index, roll.values, color='#9467bd', lw=1.0, alpha=0.9,
            label=f'Mean ({ROLL_WIN}-frame roll)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Tip column (px)')
    ax.legend(framealpha=0.7)
    save(fig, out_dir, 'fig03_lateral_drift.png')


def fig_arc_length_time(det, units, out_dir):
    """Fig 04 — arc length over time."""
    col   = units['len_col']
    label = units['len_label']
    sub   = det[det[col].notna()].sort_values('time_ms')
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(DOUBLE_W, PANEL_H))
    ax.scatter(sub['time_ms'], sub[col],
               s=0.4, c=ARC_COLOR, alpha=0.2, linewidths=0, rasterized=True)
    roll = (sub.set_index('time_ms')[col]
               .rolling(ROLL_WIN, min_periods=ROLL_WIN // 5, center=True)
               .mean())
    ax.plot(roll.index, roll.values, color=ARC_COLOR, lw=1.0, alpha=0.9,
            label=f'Mean ({ROLL_WIN}-frame roll)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(label)
    ax.legend(framealpha=0.7)
    save(fig, out_dir, 'fig04_arc_length_time.png')


def fig_arc_area_time(det, out_dir):
    """Fig 05 — arc mask area over time."""
    sub = det[det['arc_area_px'].notna()].sort_values('time_ms')
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(DOUBLE_W, PANEL_H))
    ax.scatter(sub['time_ms'], sub['arc_area_px'],
               s=0.4, c='#2ca02c', alpha=0.2, linewidths=0, rasterized=True)
    roll = (sub.set_index('time_ms')['arc_area_px']
               .rolling(ROLL_WIN, min_periods=ROLL_WIN // 5, center=True)
               .mean())
    ax.plot(roll.index, roll.values, color='#2ca02c', lw=1.0, alpha=0.9,
            label=f'Mean ({ROLL_WIN}-frame roll)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Arc area (px2)')
    ax.legend(framealpha=0.7)
    save(fig, out_dir, 'fig05_arc_area_time.png')


def fig_electrode_gap_time(det, units, out_dir):
    """Fig 06 — electrode gap (tip-to-tip span) over time."""
    col   = units['gap_col']
    label = units['gap_label']
    sub   = det[det[col].notna()].sort_values('time_ms')
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(DOUBLE_W, PANEL_H))
    ax.scatter(sub['time_ms'], sub[col],
               s=0.4, c='#d62728', alpha=0.25, linewidths=0, rasterized=True)
    roll = (sub.set_index('time_ms')[col]
               .rolling(ROLL_WIN, min_periods=ROLL_WIN // 5, center=True)
               .mean())
    ax.plot(roll.index, roll.values, color='#d62728', lw=1.0, alpha=0.9,
            label=f'Mean ({ROLL_WIN}-frame roll)')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(label)
    ax.legend(framealpha=0.7)
    save(fig, out_dir, 'fig06_electrode_gap_time.png')


def fig_intensity_time(det, out_dir):
    """Fig 07 — mean and peak arc intensity over time."""
    sub = det[det['mean_intensity'].notna()].sort_values('time_ms')
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(DOUBLE_W, PANEL_H))
    for col, color, label in [
        ('mean_intensity',  '#1f77b4', 'Mean intensity'),
        ('peak_intensity',  '#ff7f0e', 'Peak intensity'),
    ]:
        if col not in sub.columns:
            continue
        roll = (sub.set_index('time_ms')[col]
                   .rolling(ROLL_WIN, min_periods=ROLL_WIN // 5, center=True)
                   .mean())
        ax.plot(roll.index, roll.values, color=color, lw=1.0, alpha=0.9, label=label)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Normalised intensity')
    ax.set_ylim(0, 1.05)
    ax.legend(framealpha=0.7)
    save(fig, out_dir, 'fig07_intensity_time.png')


def fig_detection_rate(df_all, out_dir):
    """Fig 08 — rolling arc detection rate."""
    roll = (df_all['arc_detected'].astype(float)
                .rolling(ROLL_WIN, min_periods=ROLL_WIN // 5, center=True)
                .mean() * 100)
    fig, ax = plt.subplots(figsize=(DOUBLE_W, PANEL_H))
    ax.plot(df_all['time_ms'], roll.values, color=ARC_COLOR, lw=0.8, alpha=0.9)
    ax.fill_between(df_all['time_ms'], roll.values, alpha=0.15, color=ARC_COLOR)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Detection rate (%)')
    ax.set_ylim(0, 100)
    save(fig, out_dir, 'fig08_detection_rate.png')


def fig_complete_fraction(df_all, det, out_dir):
    """Fig 09 — rolling complete vs partial arc fraction."""
    roll_c = rolling_on_allframes(df_all, det, 'is_complete')
    roll_p = rolling_on_allframes(df_all, det, 'is_partial')

    fig, ax = plt.subplots(figsize=(DOUBLE_W, PANEL_H))
    ax.plot(df_all['time_ms'], roll_c.values,
            color=ARC_COLOR,     lw=0.8, alpha=0.9, label='Complete (bridging)')
    ax.plot(df_all['time_ms'], roll_p.values,
            color=PARTIAL_COLOR, lw=0.8, alpha=0.9, ls='--', label='Partial / leader')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frame fraction (%)')
    ax.set_ylim(0)
    ax.legend(framealpha=0.7)
    save(fig, out_dir, 'fig09_complete_fraction.png')


def fig_length_dist(det, units, out_dir):
    """Fig 10 — arc length distribution."""
    col   = units['len_col']
    label = units['len_label']
    sub   = det[det[col].notna()][col].dropna()
    if len(sub) < 10:
        return

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, PANEL_H))

    # Histogram
    axes[0].hist(sub, bins=50, color=ARC_COLOR, alpha=0.7,
                 density=True, histtype='stepfilled')
    axes[0].hist(sub, bins=50, color=ARC_COLOR, alpha=1.0,
                 density=True, histtype='step', lw=0.8)
    axes[0].set_xlabel(label)
    axes[0].set_ylabel('Probability density')

    # Complete vs partial split
    for is_c, color, lbl in [
        (True,  ARC_COLOR,     'Complete'),
        (False, PARTIAL_COLOR, 'Partial'),
    ]:
        v = det[det['is_complete'] == is_c][col].dropna()
        if len(v) < 10:
            continue
        kde = gaussian_kde(v, bw_method='scott')
        x   = np.linspace(sub.min(), sub.max(), 400)
        axes[1].fill_between(x, kde(x), alpha=0.2, color=color)
        axes[1].plot(x, kde(x), color=color, lw=1.2, label=lbl)
    axes[1].set_xlabel(label)
    axes[1].set_ylabel('Probability density')
    axes[1].legend(framealpha=0.7)

    fig.tight_layout()
    save(fig, out_dir, 'fig10_length_dist.png')


def fig_area_dist(det, out_dir):
    """Fig 11 — arc area distribution."""
    sub = det['arc_area_px'].dropna()
    if len(sub) < 10:
        return
    hi = np.percentile(sub, 99)
    sub = sub[sub <= hi]

    fig, ax = plt.subplots(figsize=(SINGLE_W, PANEL_H))
    ax.hist(sub, bins=60, color='#2ca02c', alpha=0.7,
            density=True, histtype='stepfilled')
    ax.hist(sub, bins=60, color='#2ca02c', alpha=1.0,
            density=True, histtype='step', lw=0.8)
    ax.set_xlabel('Arc area (px2)')
    ax.set_ylabel('Probability density')
    save(fig, out_dir, 'fig11_area_dist.png')


def fig_width_dist(det, units, out_dir):
    """Fig 12 — arc width distribution."""
    col   = units['wid_col']
    label = units['wid_label']
    sub   = det[det[col].notna()][col].dropna()
    if len(sub) < 10:
        return
    hi = np.percentile(sub, 99)
    sub = sub[sub <= hi]

    fig, ax = plt.subplots(figsize=(SINGLE_W, PANEL_H))
    ax.hist(sub, bins=40, color='#8c564b', alpha=0.7,
            density=True, histtype='stepfilled')
    ax.hist(sub, bins=40, color='#8c564b', alpha=1.0,
            density=True, histtype='step', lw=0.8)
    ax.set_xlabel(label)
    ax.set_ylabel('Probability density')
    save(fig, out_dir, 'fig12_width_dist.png')


def fig_velocity_timeseries(det, units, fps, out_dir):
    """Fig 13 — frame-to-frame arc-tip velocity over time."""
    col   = units['vel_col']
    label = units['vel_label']
    sub   = det[det[col].notna()].copy()
    v     = clean_vel(sub[col])
    sub   = sub.loc[v.index]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(DOUBLE_W, PANEL_H))
    ax.scatter(sub['time_ms'], sub[col],
               s=0.5, c=ARC_COLOR, alpha=0.2, linewidths=0, rasterized=True)
    roll = (sub.set_index('time_ms')[col]
               .rolling(50, min_periods=5, center=True)
               .mean())
    ax.plot(roll.index, roll.values, color=ARC_COLOR, lw=1.0, alpha=0.9,
            label='50-frame rolling mean')
    ax.axhline(0, color='gray', lw=0.6, ls='--')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel(label)
    ax.legend(framealpha=0.7)
    save(fig, out_dir, 'fig13_velocity_timeseries.png')


def fig_velocity_dist_cdf(det, units, out_dir):
    """Fig 14 — velocity distribution (KDE) and CDF, side by side."""
    col   = units['vel_col']
    label = units['vel_label']
    v     = clean_vel(det[col])
    if len(v) < 10:
        return

    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_W, PANEL_H))

    # KDE
    kde = gaussian_kde(v, bw_method='scott')
    x   = np.linspace(v.min(), v.max(), 500)
    axes[0].fill_between(x, kde(x), alpha=0.25, color=ARC_COLOR)
    axes[0].plot(x, kde(x), color=ARC_COLOR, lw=1.3)
    axes[0].axvline(float(v.mean()), color='gray', lw=0.8, ls='--',
                    label=f'Mean: {v.mean():.2f}')
    axes[0].set_xlabel(label)
    axes[0].set_ylabel('Probability density')
    axes[0].legend(framealpha=0.7, fontsize=7)

    # CDF
    sv = np.sort(v)
    axes[1].plot(sv, np.arange(1, len(sv)+1) / len(sv),
                 color=ARC_COLOR, lw=1.3)
    for p, ls in [(0.50, '--'), (0.95, ':')]:
        val = float(np.percentile(sv, p * 100))
        axes[1].axvline(val, color='gray', lw=0.7, ls=ls)
        axes[1].text(val, p + 0.02, f'p{int(p*100)}: {val:.2f}',
                     fontsize=7, color='gray', ha='left')
    axes[1].set_xlabel(label)
    axes[1].set_ylabel('Cumulative fraction')
    axes[1].set_ylim(0, 1)

    fig.tight_layout()
    save(fig, out_dir, 'fig14_velocity_dist_cdf.png')


def fig_velocity_vs_depth(det, units, out_dir):
    """Fig 15 — velocity vs arc-tip depth (median + IQR per bin)."""
    vcol  = units['vel_col']
    dcol  = units['dep_col']
    vlbl  = units['vel_label']
    dlbl  = units['dep_label']

    sub = det[det[vcol].notna() & det[dcol].notna()].copy()
    v   = clean_vel(sub[vcol])
    sub = sub.loc[v.index]
    if sub.empty:
        return

    n_bins = 15
    d_min  = sub[dcol].quantile(0.01)
    d_max  = sub[dcol].quantile(0.99)
    bins   = np.linspace(d_min, d_max, n_bins + 1)
    mids   = (bins[:-1] + bins[1:]) / 2

    grp = sub.groupby(pd.cut(sub[dcol], bins=bins, labels=mids), observed=False)[vcol]
    med = grp.mean()
    q25 = grp.quantile(0.25)
    q75 = grp.quantile(0.75)
    x   = med.index.astype(float)

    fig, ax = plt.subplots(figsize=(PANEL_H + 0.3, PANEL_H + 0.5))
    ax.scatter(sub[vcol], sub[dcol],
               s=0.6, c=ARC_COLOR, alpha=0.1, linewidths=0, rasterized=True)
    ax.plot(med.values, x, color=ARC_COLOR, lw=1.5, label='Mean')
    ax.fill_betweenx(x, q25.values, q75.values,
                     color=ARC_COLOR, alpha=0.2, label='IQR')
    ax.axvline(0, color='gray', lw=0.6, ls='--')
    ax.invert_yaxis()
    ax.set_xlabel(vlbl)
    ax.set_ylabel(dlbl)
    ax.legend(framealpha=0.7, fontsize=7)
    save(fig, out_dir, 'fig15_velocity_vs_depth.png')


# =============================================================================
# SUMMARY STATS
# =============================================================================

def print_summary(df, det, units, fps):
    n_total  = len(df)
    n_det    = len(det)
    n_comp   = int(det['is_complete'].sum())
    n_part   = int(det['is_partial'].sum())
    dur_ms   = df['time_ms'].max() - df['time_ms'].min()

    print(f"\n{'─'*55}")
    print(f"  Recording duration : {dur_ms:.1f} ms  ({n_total:,} frames @ {fps/1000:.0f} kfps)")
    print(f"  Frames detected    : {n_det:,}  ({100*n_det/max(n_total,1):.1f}%)")
    print(f"  Complete arcs      : {n_comp:,}  ({100*n_comp/max(n_det,1):.1f}% of detected)")
    print(f"  Partial arcs       : {n_part:,}  ({100*n_part/max(n_det,1):.1f}% of detected)")

    lcol = units['len_col']
    if lcol in det.columns and det[lcol].notna().any():
        v = det[lcol].dropna()
        print(f"  Arc length mean  : {v.mean():.1f}  ({units['len_label'].split('(')[1].rstrip(')')})")

    vcol = units['vel_col']
    if vcol in det.columns:
        v = clean_vel(det[vcol])
        if len(v) > 5:
            print(f"  Velocity mean    : {v.mean():.3f}  ({units['vel_label'].split('(')[1].rstrip(')')})")
            print(f"  Velocity p95       : {np.percentile(v, 95):.3f}")
    print(f"{'─'*55}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    csv_path  = DEFAULT_CSV
    out_dir   = DEFAULT_OUT
    px_per_mm = PX_PER_MM

    positional = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags      = {a.split('=')[0].lstrip('-'): a.split('=')[1]
                  for a in sys.argv[1:] if a.startswith('--') and '=' in a}
    if len(positional) >= 1: csv_path  = positional[0]
    if len(positional) >= 2: out_dir   = positional[1]
    if 'px-per-mm' in flags: px_per_mm = float(flags['px-per-mm'])

    if not Path(csv_path).exists():
        print(f"[ERROR] CSV not found: {csv_path}"); sys.exit(1)

    os.makedirs(out_dir, exist_ok=True)

    # Read FPS from metadata.txt if present
    fps, px_from_meta = read_metadata(csv_path)
    if px_per_mm is None and px_from_meta is not None:
        px_per_mm = px_from_meta

    print(f"\n{'='*55}")
    print(f"  J-electrode CINE visualizer")
    print(f"  CSV    : {csv_path}")
    print(f"  Output : {out_dir}")
    print(f"  FPS    : {fps:,}")
    print(f"  px/mm  : {px_per_mm if px_per_mm else 'not set — using pixels'}")
    print(f"{'='*55}\n")

    df, det, units = load_data(csv_path, fps, px_per_mm)
    print_summary(df, det, units, fps)

    print("  Saving figures ...")
    fig_kymograph(det, df, units, out_dir)
    fig_tip_2d_map(det, units, out_dir)
    fig_lateral_drift(det, out_dir)
    fig_arc_length_time(det, units, out_dir)
    fig_arc_area_time(det, out_dir)
    fig_electrode_gap_time(det, units, out_dir)
    fig_intensity_time(det, out_dir)
    fig_detection_rate(df, out_dir)
    fig_complete_fraction(df, det, out_dir)
    fig_length_dist(det, units, out_dir)
    fig_area_dist(det, out_dir)
    fig_width_dist(det, units, out_dir)
    fig_velocity_timeseries(det, units, fps, out_dir)
    fig_velocity_dist_cdf(det, units, out_dir)
    fig_velocity_vs_depth(det, units, out_dir)

    n_saved = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
    print(f"\n  Done — {n_saved} figures saved to {out_dir}")
    if px_per_mm is None:
        print("  [NOTE] PX_PER_MM not set — spatial/velocity axes are in pixels.")
        print("         Re-run with --px-per-mm=X once you have a calibration image.")


if __name__ == '__main__':
    main()
