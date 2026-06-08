"""
visualize_blade.py  -  Paper-ready individual-panel figures for plasma arc analysis.

Reads:  measurements_all.csv  (combined, produced by blade_run2.py combine)
        *_velocity_map.npy    (one per run, in *_results/ subdirs)

Usage:
    python visualize_blade.py [measurements_all.csv] [output_dir]
"""

import os, sys, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

import arc_filter as af  # single source of truth for flash / validity criteria

PX_PER_MM    = 9.167
FPS          = 200_000
FRAME_MM     = 256 / PX_PER_MM
# Flash criteria come from arc_filter (do not redefine here). No velocity cap:
# flashes are rejected per-frame, velocities are reported as measured.
FLASH_EFF_WIDTH = af.FLASH_EFF_WIDTH
FLASH_ARC_WIDTH = af.FLASH_ARC_WIDTH

VELMAP_VMIN =  0   # m/s — pixels below this are hidden
VELMAP_VMAX = None    # m/s — None = auto (95th-percentile across all runs)

RUN_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']
RUN_LABELS = ['Run 1', 'Run 2', 'Run 3']

SINGLE_W = 3.5
DOUBLE_W = 7.0
PANEL_H  = 2.8
DPI      = 300

plt.rcParams.update({
    'font.size':        9,
    'axes.labelsize':   10,
    'axes.titlesize':   10,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  8,
    'axes.linewidth':   0.8,
    'lines.linewidth':  1.0,
    'font.family':      'sans-serif',
    'figure.dpi':       DPI,
    'savefig.dpi':      DPI,
    'savefig.bbox':     'tight',
    'savefig.pad_inches': 0.05,
})


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    for col in ['arc_tip_row', 'arc_tip_col', 'arc_tip_vel_mm_s',
                'arc_length_mm', 'arc_length_px', 'arc_width_mm',
                'arc_width_px', 'area_px',
                'mean_intensity', 'peak_intensity', 'bbox_r0',
                'electrode_gap_px', 'curvature_mean',
                'tip_a_row', 'tip_a_col', 'tip_b_row', 'tip_b_col',
                'tortuosity', 'total_curvature_rad', 'arclen_asym', 'tort_asym']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    stems = sorted(df['file'].unique())
    stem_to_run = {s: 'Run %d' % (i+1) for i, s in enumerate(stems)}
    df['run'] = df['file'].map(stem_to_run)
    df['time_ms'] = df['time_s'] * 1e3
    det = df[df['arc_detected'] == 1].copy()
    # Width-based flash flag (see FLASH_EFF_WIDTH / FLASH_ARC_WIDTH at top).
    # NaN > x returns False in pandas, so missing arc_width_px doesn't flag.
    det = af.add_flash_flag(det)   # eff_width + is_flash (uses the baked is_flash if present)
    det['tip_depth_mm'] = det['arc_tip_row'] / PX_PER_MM
    det['vel_ms']       = det['arc_tip_vel_mm_s'] / 1000.0  # reported as measured (no cap)
    det['time_ms']      = det['time_s'] * 1e3

    # Arc completeness: purely morphological, no position threshold.
    # Complete (bridging) arc: skeleton endpoints are separated more horizontally
    # than vertically — both tips are at electrode height on opposite sides.
    # Partial arc (leader/streamer): one tip is floating below — vertical
    # separation dominates. Robust to any electrode gap or V-angle.
    if 'arc_complete' in det.columns:
        det['is_complete_arc'] = (
            det['arc_complete'].fillna(False).astype(bool) & ~det['is_flash']
        )
    else:
        det['is_complete_arc'] = (
            det['tip_a_row'].notna() & det['tip_a_col'].notna() &
            det['tip_b_row'].notna() & det['tip_b_col'].notna() &
            ((det['tip_b_col'] - det['tip_a_col']).abs() >
             (det['tip_b_row'] - det['tip_a_row']).abs()) &
            ~det['is_flash']
        )
    det['is_partial_arc'] = ~det['is_complete_arc'] & ~det['is_flash']

    return df, det, stems, stem_to_run


def load_velmaps(stems, base_dir):
    vmaps = {}
    for stem in stems:
        # analyze_blade.py writes "velocity_map.npy"; the older blade_run2 layout
        # wrote "<stem>_velocity_map.npy". Accept either, and also look directly
        # in base_dir (the 11MAY26 layout passes the results dir itself).
        candidates = [
            # Primary: the local (time-resolved) velocity map built post-Stage-4
            # by build_velocity_map.py (stored in mm/s, like the others).
            os.path.join(base_dir, 'velocity_map_local.npy'),
            # Fallbacks: the central-difference map from analyze_blade.py.
            os.path.join(base_dir, stem + '_results', stem + '_velocity_map.npy'),
            os.path.join(base_dir, stem + '_results', 'velocity_map.npy'),
            os.path.join(base_dir, stem + '_velocity_map.npy'),
            os.path.join(base_dir, 'velocity_map.npy'),
        ]
        for npy in candidates:
            if os.path.exists(npy):
                vmaps[stem] = np.load(npy)
                break
    return vmaps


def save(fig, out_dir, name):
    fig.savefig(os.path.join(out_dir, name))
    plt.close(fig)
    print('  ' + name)


def clean_vel(series, pct_hi=99.5):
    v = series.dropna()
    v = v[v > 0]
    if len(v) < 5:
        return v
    return v[v <= np.percentile(v, pct_hi)]


def fig_kymograph(det, run_label, color, out_dir):
    sub      = det[det['run'] == run_label]
    complete = sub[ sub['is_complete_arc']]
    partial  = sub[ sub['is_partial_arc']]
    flash    = sub[ sub['is_flash']]
    fig, ax = plt.subplots(figsize=(DOUBLE_W, PANEL_H))
    ax.scatter(complete['time_ms'], complete['tip_depth_mm'],
               s=0.4, c=color, alpha=0.4, linewidths=0, rasterized=True)
    ax.scatter(partial['time_ms'], partial['tip_depth_mm'],
               s=0.4, c='gold', alpha=0.4, linewidths=0, rasterized=True)
    ax.scatter(flash['time_ms'], flash['tip_depth_mm'],
               s=0.6, c='red', alpha=0.5, linewidths=0, rasterized=True)
    handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=color,  ms=4, label='Complete arc'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='gold',  ms=4, label='Partial arc'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='red',   ms=4, label='Flash'),
    ]
    ax.legend(handles=handles, loc='lower right', framealpha=0.7)
    ax.invert_yaxis()
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Tip depth (mm)')
    tag = run_label.replace(' ', '').lower()
    save(fig, out_dir, 'fig01_kymograph_' + tag + '.png')


def fig_tip_depth(df_full, det, out_dir):
    fig, ax = plt.subplots(figsize=(DOUBLE_W, PANEL_H))
    for run, color in zip(sorted(det['run'].unique()), RUN_COLORS):
        sub = det[(det['run'] == run) & ~det['is_flash']].sort_values('time_ms')
        sub = sub.copy()
        sub['t_bin'] = (sub['time_ms'] / 0.05).round() * 0.05
        binned = sub.groupby('t_bin')['tip_depth_mm'].mean()
        ax.plot(binned.index, binned.values, color=color, lw=0.7, alpha=0.85, label=run)
    ax.invert_yaxis()
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Tip depth (mm)')
    ax.legend(framealpha=0.7)
    save(fig, out_dir, 'fig02_tip_depth.png')


def fig_arc_length(det, out_dir):
    fig, ax = plt.subplots(figsize=(DOUBLE_W, PANEL_H))
    for run, color in zip(sorted(det['run'].unique()), RUN_COLORS):
        sub = det[(det['run'] == run) & ~det['is_flash'] &
                  det['arc_length_mm'].notna()].sort_values('time_ms').copy()
        sub['t_bin'] = (sub['time_ms'] / 0.05).round() * 0.05
        binned = sub.groupby('t_bin')['arc_length_mm'].mean()
        ax.plot(binned.index, binned.values, color=color, lw=0.7, alpha=0.85, label=run)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Arc length (mm)')
    ax.legend(framealpha=0.7)
    save(fig, out_dir, 'fig03_arc_length.png')


def fig_velocity_timeseries(det, out_dir):
    fig, ax = plt.subplots(figsize=(DOUBLE_W, PANEL_H))
    for run, color in zip(sorted(det['run'].unique()), RUN_COLORS):
        sub = det[(det['run'] == run) & ~det['is_flash'] &
                  det['vel_ms'].notna() & (det['vel_ms'] > 0)].sort_values('time_ms')
        if sub.empty:
            continue
        ax.scatter(sub['time_ms'], sub['vel_ms'],
                   s=0.5, c=color, alpha=0.25, linewidths=0, rasterized=True)
        roll = sub.set_index('time_ms')['vel_ms'].rolling(50, min_periods=5, center=True).mean()
        ax.plot(roll.index, roll.values, color=color, lw=1.0, alpha=0.9, label=run)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Arc-tip velocity (m/s)')
    ax.set_ylim(bottom=0)
    legend_h = [Line2D([0],[0], color=c, lw=1.5, label=r)
                for r, c in zip(RUN_LABELS, RUN_COLORS)]
    ax.legend(handles=legend_h, framealpha=0.7)
    save(fig, out_dir, 'fig04_velocity_timeseries.png')


def fig_velocity_dist(det, out_dir):
    fig, ax = plt.subplots(figsize=(SINGLE_W, PANEL_H))
    for run, color in zip(sorted(det['run'].unique()), RUN_COLORS):
        v = clean_vel(det[(det['run'] == run) & ~det['is_flash']]['vel_ms'])
        if len(v) < 10:
            continue
        # Histogram always plotted; KDE only when data has sufficient spread
        vmax = float(v.max())
        ax.hist(v, bins=40, range=(0, vmax), density=True,
                color=color, alpha=0.2, edgecolor='none')
        if float(v.std()) > 1e-6 * float(v.mean() + 1e-10):
            try:
                kde = gaussian_kde(v, bw_method='scott')
                x = np.linspace(0, vmax, 500)
                ax.plot(x, kde(x), color=color, lw=1.2,
                        label=run)
            except Exception:
                ax.plot([], [], color=color, lw=1.2,
                        label=run)
        else:
            ax.plot([], [], color=color, lw=1.2,
                    label=run)
    ax.set_xlabel('Arc-tip velocity (m/s)')
    ax.set_ylabel('Probability density')
    ax.set_xlim(left=0)
    ax.legend(framealpha=0.7)
    save(fig, out_dir, 'fig05_velocity_dist.png')


def fig_velocity_cdf(det, out_dir):
    fig, ax = plt.subplots(figsize=(SINGLE_W, PANEL_H))
    for run, color in zip(sorted(det['run'].unique()), RUN_COLORS):
        v = clean_vel(det[(det['run'] == run) & ~det['is_flash']]['vel_ms'])
        if len(v) < 10:
            continue
        sv = np.sort(v)
        ax.plot(sv, np.arange(1, len(sv)+1) / len(sv), color=color, lw=1.2, label=run)
    ax.axhline(0.50, ls='--', lw=0.6, color='gray')
    ax.axhline(0.95, ls='--', lw=0.6, color='gray')
    ax.text(0.02, 0.52, 'p50', transform=ax.get_yaxis_transform(), fontsize=7, color='gray')
    ax.text(0.02, 0.97, 'p95', transform=ax.get_yaxis_transform(), fontsize=7, color='gray')
    ax.set_xlabel('Arc-tip velocity (m/s)')
    ax.set_ylabel('Cumulative fraction')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend(framealpha=0.7)
    save(fig, out_dir, 'fig06_velocity_cdf.png')


def fig_velocity_vs_depth(det, out_dir):
    fig, ax = plt.subplots(figsize=(PANEL_H + 0.3, DOUBLE_W * 0.55))
    bins = np.arange(0, FRAME_MM + 1.5, 1.5).tolist()
    mids = ((np.array(bins[:-1]) + np.array(bins[1:])) / 2).tolist()
    for run, color in zip(sorted(det['run'].unique()), RUN_COLORS):
        sub = det[(det['run'] == run) & ~det['is_flash'] &
                  det['vel_ms'].notna() & (det['vel_ms'] > 0) &
                  det['tip_depth_mm'].notna()]
        if sub.empty:
            continue
        ax.scatter(sub['vel_ms'], sub['tip_depth_mm'],
                   s=0.8, c=color, alpha=0.12, linewidths=0, rasterized=True)
        grp = sub.groupby(pd.cut(sub['tip_depth_mm'], bins=bins, labels=mids),
                          observed=False)['vel_ms']
        med = grp.mean()
        q25 = grp.quantile(0.25)
        q75 = grp.quantile(0.75)
        x   = med.index.astype(float)
        ax.plot(med.values, x, color=color, lw=1.3, label=run)
        ax.fill_betweenx(x, q25.values, q75.values, color=color, alpha=0.15)
    ax.set_xlabel('Arc-tip velocity (m/s)')
    ax.set_ylabel('Tip depth (mm)')
    ax.set_ylim(0, FRAME_MM)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.legend(framealpha=0.7)
    save(fig, out_dir, 'fig07_velocity_vs_depth.png')


def fig_arc_area(det, out_dir):
    fig, ax = plt.subplots(figsize=(SINGLE_W, PANEL_H))
    bins = np.linspace(0, 1600, 40).tolist()
    for run, color in zip(sorted(det['run'].unique()), RUN_COLORS):
        sub = det[(det['run'] == run) & ~det['is_flash'] & det['area_px'].notna()]
        ax.hist(sub['area_px'], bins=bins, color=color, alpha=0.45,
                density=True, histtype='stepfilled', label=run)
        ax.hist(sub['area_px'], bins=bins, color=color, alpha=1.0,
                density=True, histtype='step', lw=1.0)
    ax.set_xlabel('Arc area (px^2)')
    ax.set_ylabel('Probability density')
    ax.legend(framealpha=0.7)
    save(fig, out_dir, 'fig08_arc_area.png')


def fig_arc_width(det, out_dir):
    fig, ax = plt.subplots(figsize=(SINGLE_W, PANEL_H))
    bins = np.linspace(0, 4, 30).tolist()
    for run, color in zip(sorted(det['run'].unique()), RUN_COLORS):
        sub = det[(det['run'] == run) & ~det['is_flash'] & det['arc_width_mm'].notna()]
        ax.hist(sub['arc_width_mm'], bins=bins, color=color, alpha=0.45,
                density=True, histtype='stepfilled', label=run)
        ax.hist(sub['arc_width_mm'], bins=bins, color=color, alpha=1.0,
                density=True, histtype='step', lw=1.0)
    ax.set_xlabel('Arc width (mm)')
    ax.set_ylabel('Probability density')
    ax.legend(framealpha=0.7)
    save(fig, out_dir, 'fig09_arc_width.png')


def fig_detection_rate(df_full, out_dir):
    fig, ax = plt.subplots(figsize=(DOUBLE_W, PANEL_H))
    for run, color in zip(sorted(df_full['run'].unique()), RUN_COLORS):
        sub = df_full[df_full['run'] == run].sort_values('frame')
        roll = sub['arc_detected'].rolling(500, min_periods=100, center=True).mean() * 100
        ax.plot(sub['time_ms'], roll.values, color=color, lw=0.8, alpha=0.85, label=run)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Detection rate (%)')
    ax.set_ylim(0, 100)
    ax.legend(framealpha=0.7)
    save(fig, out_dir, 'fig10_detection_rate.png')


def fig_arc_type_rate(df_full, det, out_dir):
    """Rolling fraction of complete vs partial arcs over time."""
    fig, ax = plt.subplots(figsize=(DOUBLE_W, PANEL_H))
    for run, color in zip(sorted(df_full['run'].unique()), RUN_COLORS):
        sub_all = df_full[df_full['run'] == run].sort_values('frame')
        sub_det = det[(det['run'] == run)].copy()

        # rolling window on all frames: fraction that are complete / partial
        n_win = 500
        complete_flag = sub_all['frame'].isin(
            sub_det.loc[sub_det['is_complete_arc'], 'frame']
        ).astype(float)
        partial_flag = sub_all['frame'].isin(
            sub_det.loc[sub_det['is_partial_arc'], 'frame']
        ).astype(float)
        t = sub_all['time_ms'].values
        roll_c = complete_flag.rolling(n_win, min_periods=50, center=True).mean() * 100
        roll_p = partial_flag.rolling(n_win, min_periods=50, center=True).mean() * 100
        ax.plot(t, roll_c.values, color=color, lw=0.8, alpha=0.9,
                label=f'{run} complete')
        ax.plot(t, roll_p.values, color=color, lw=0.8, alpha=0.5,
                ls='--', label=f'{run} partial')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frame fraction (%)')
    ax.set_ylim(0, 100)
    ax.legend(framealpha=0.7, fontsize=7, ncol=2)
    save(fig, out_dir, 'fig10b_arc_type_rate.png')


def fig_velmap(vmap_mm_s, run_label, fig_num, out_dir, roi_mask=None, vmax=None):
    vm = vmap_mm_s / 1000.0
    if roi_mask is not None:
        vm = vm.copy()
        vm[~roi_mask] = np.nan
    vmin_km_s = VELMAP_VMIN / 1000.0
    vm[vm < vmin_km_s] = np.nan
    if vmax is None:
        # Exclude electrode-gap rows (~2 mm) — arc flash dominates percentile otherwise.
        gap_rows = max(1, int(round(2.0 * PX_PER_MM)))
        body = vm[gap_rows:, :] if vm.shape[0] > gap_rows else vm
        pos  = body[np.isfinite(body) & (body > 0)]
        vmax = float(np.nanpercentile(pos, 95)) if len(pos) > 0 else 1.0
    px = np.arange(vm.shape[1]) / PX_PER_MM
    py = np.arange(vm.shape[0]) / PX_PER_MM
    fig, ax = plt.subplots(figsize=(SINGLE_W, SINGLE_W))
    im = ax.pcolormesh(px, py, vm, cmap='plasma',
                       vmin=vmin_km_s, vmax=vmax, rasterized=True)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Velocity (m/s)')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('Depth (mm)  [0 = electrode gap]')
    ax.invert_yaxis()
    ax.set_aspect('equal')
    tag = run_label.replace(' ', '').lower()
    save(fig, out_dir, 'fig%02d_velocity_map_%s.png' % (fig_num, tag))


def fig_flash_scatter(det, out_dir):
    """Diagnostic scatter of the two flash discriminators: eff_width vs
    arc_width_px (FWHM). Dashed lines show the OR cut. Frames in the upper
    OR right region are flagged as flashes."""
    fig, ax = plt.subplots(figsize=(SINGLE_W, PANEL_H))
    for run, color in zip(sorted(det['run'].unique()), RUN_COLORS):
        sub   = det[det['run'] == run]
        clean = sub[~sub['is_flash']]
        flash = sub[ sub['is_flash']]
        ax.scatter(clean['arc_width_px'], clean['eff_width'],
                   s=1.5, c=color, alpha=0.3, linewidths=0, rasterized=True)
        ax.scatter(flash['arc_width_px'], flash['eff_width'],
                   s=3.0, c='red',   alpha=0.6, linewidths=0, rasterized=True)
    ax.axvline(FLASH_ARC_WIDTH, ls='--', lw=0.8, color='gray')
    ax.axhline(FLASH_EFF_WIDTH, ls='--', lw=0.8, color='gray')
    handles = [Line2D([0],[0], marker='o', color='w', markerfacecolor=c,
                      ms=4, label=r) for r, c in zip(RUN_LABELS, RUN_COLORS)]
    handles.append(Line2D([0],[0], marker='o', color='w',
                          markerfacecolor='red', ms=4, label='Flash'))
    ax.legend(handles=handles, framealpha=0.7, fontsize=7)
    ax.set_xlabel('arc_width_px (FWHM, px)')
    ax.set_ylabel('eff_width = area_px / arc_length_px (px)')
    save(fig, out_dir, 'fig14_flash_scatter.png')


def fig_detection_counts(det, out_dir):
    runs    = sorted(det['run'].unique())
    n_clean = [len(det[(det['run'] == r) & ~det['is_flash']]) for r in runs]
    n_flash = [len(det[(det['run'] == r) &  det['is_flash']]) for r in runs]
    x = np.arange(len(runs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(SINGLE_W, PANEL_H))
    ax.bar(x - w/2, n_clean, w, color=RUN_COLORS[:len(runs)], alpha=0.85, label='Clean arc')
    ax.bar(x + w/2, n_flash, w, color='#e15759', alpha=0.85, label='Flash')
    ax.set_xticks(x)
    ax.set_xticklabels(runs)
    ax.set_ylabel('Frames detected')
    ax.legend(framealpha=0.7)
    save(fig, out_dir, 'fig15_detection_counts.png')


def fig_intensity(det, out_dir):
    fig, ax = plt.subplots(figsize=(SINGLE_W, PANEL_H))
    bins = np.linspace(0, 1.05, 35).tolist()
    for run, color in zip(sorted(det['run'].unique()), RUN_COLORS):
        sub = det[(det['run'] == run) & ~det['is_flash'] & det['peak_intensity'].notna()]
        ax.hist(sub['peak_intensity'], bins=bins, color=color, alpha=0.45,
                density=True, histtype='stepfilled', label=run)
        ax.hist(sub['peak_intensity'], bins=bins, color=color, alpha=1.0,
                density=True, histtype='step', lw=1.0)
    ax.set_xlabel('Normalised peak intensity')
    ax.set_ylabel('Probability density')
    ax.legend(framealpha=0.7)
    save(fig, out_dir, 'fig16_intensity_dist.png')


def fig_tortuosity_dist(det, out_dir):
    """Distribution of arc tortuosity (path length / electrode-gap chord) per run.
    Only well-formed arcs (tortuosity >= 1) are shown; values < 1 indicate a
    fragmented skeleton that did not bridge both roots."""
    fig, ax = plt.subplots(figsize=(SINGLE_W, PANEL_H))
    for run, color in zip(sorted(det['run'].unique()), RUN_COLORS):
        v = det[(det['run'] == run) & ~det['is_flash'] &
                det['tortuosity'].notna() & (det['tortuosity'] >= 1.0)]['tortuosity']
        if len(v) < 10:
            continue
        vmax = float(np.percentile(v, 99.5))
        ax.hist(v, bins=40, range=(1.0, vmax), density=True,
                color=color, alpha=0.35, histtype='stepfilled')
        if float(v.std()) > 1e-6:
            try:
                kde = gaussian_kde(v)
                x = np.linspace(1.0, vmax, 400)
                ax.plot(x, kde(x), color=color, lw=1.2,
                        label=run)
            except Exception:
                ax.plot([], [], color=color, lw=1.2, label=run)
    ax.axvline(1.0, ls='--', lw=0.6, color='gray')
    ax.set_xlabel('Tortuosity (path / chord)')
    ax.set_ylabel('Probability density')
    ax.set_xlim(left=1.0)
    ax.legend(framealpha=0.7)
    save(fig, out_dir, 'fig17_tortuosity_dist.png')


def fig_tortuosity_vs_depth(det, out_dir):
    """Median tortuosity vs arc-tip depth — does the arc get twistier as it
    propagates away from the electrodes?"""
    fig, ax = plt.subplots(figsize=(DOUBLE_W, PANEL_H))
    bins = np.arange(0, FRAME_MM + 1.5, 1.5).tolist()
    mids = ((np.array(bins[:-1]) + np.array(bins[1:])) / 2).tolist()
    for run, color in zip(sorted(det['run'].unique()), RUN_COLORS):
        sub = det[(det['run'] == run) & ~det['is_flash'] &
                  det['tortuosity'].notna() & (det['tortuosity'] >= 1.0) &
                  det['tip_depth_mm'].notna()]
        if len(sub) < 20:
            continue
        grp = sub.groupby(pd.cut(sub['tip_depth_mm'], bins=bins, labels=mids),
                          observed=False)['tortuosity']
        med, q25, q75 = grp.mean(), grp.quantile(0.25), grp.quantile(0.75)
        x = med.index.astype(float)
        ax.plot(x, med.values, color=color, lw=1.3, label=run)
        ax.fill_between(x, q25.values, q75.values, color=color, alpha=0.15)
    ax.axhline(1.0, ls='--', lw=0.6, color='gray')
    ax.set_xlabel('Tip depth (mm)')
    ax.set_ylabel('Tortuosity (path / chord)')
    ax.set_ylim(bottom=1.0)
    ax.legend(framealpha=0.7)
    save(fig, out_dir, 'fig18_tortuosity_vs_depth.png')


def fig_asymmetry_dist(det, out_dir):
    """Signed asymmetry of arc length and tortuosity about the electrode midline
    (+ = blade-A side).  A reference line at 0 marks perfect symmetry."""
    cols = [c for c in ['arclen_asym', 'tort_asym'] if c in det.columns]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(SINGLE_W * len(cols), PANEL_H),
                             squeeze=False)
    titles = {'arclen_asym': 'Arc-length asymmetry', 'tort_asym': 'Tortuosity asymmetry'}
    for ax, col in zip(axes[0], cols):
        any_data = False
        for run, color in zip(sorted(det['run'].unique()), RUN_COLORS):
            v = det[(det['run'] == run) & ~det['is_flash'] & det[col].notna()][col]
            v = v[np.abs(v) <= 1.0]
            if len(v) < 10:
                continue
            any_data = True
            ax.hist(v, bins=41, range=(-1, 1), density=True,
                    color=color, alpha=0.30, histtype='stepfilled')
            if float(v.std()) > 1e-9:
                try:
                    x = np.linspace(-1, 1, 400)
                    ax.plot(x, gaussian_kde(v)(x), color=color, lw=1.2,
                            label=run)
                except Exception:
                    ax.plot([], [], color=color, lw=1.2, label=run)
        ax.axvline(0.0, ls='--', lw=0.7, color='gray')
        ax.set_xlabel('%s  (+ = blade-A side)' % titles.get(col, col))
        ax.set_ylabel('Probability density')
        ax.set_xlim(-1, 1)
        if any_data:
            ax.legend(framealpha=0.7, fontsize=7)
    save(fig, out_dir, 'fig19_asymmetry_dist.png')


def main():
    # Declared global up front: PX_PER_MM is read in the argparse help string
    # below (and PX_PER_MM/FRAME_MM are reassigned after parsing), so the
    # declaration must precede their first use in this function.
    global PX_PER_MM, FRAME_MM, VELMAP_VMAX
    import argparse
    ap = argparse.ArgumentParser(description='Paper-ready figures for plasma arc analysis')
    ap.add_argument('csv',         nargs='?', default=None,
                    help='measurements CSV (default: dated BLADE dir)')
    ap.add_argument('out_dir',     nargs='?', default=None,
                    help='Output directory')
    ap.add_argument('velmap_dir',  nargs='?', default=None,
                    help='Directory containing *_results/*_velocity_map.npy '
                         '(default: same as out_dir)')
    ap.add_argument('--px-per-mm', type=float, default=None,
                    help='Spatial calibration in px/mm.  Default %.4f (ADCB). '
                         'Pass 8.8 for 11MAY26.' % PX_PER_MM)
    ap.add_argument('--velmap-vmax', type=float, default=None,
                    help='Upper velocity bound for 2-D velocity map colour scale '
                         '(m/s).  Default: auto (95th percentile of arc body, '
                         'excluding electrode-gap rows).  Set a fixed value to '
                         'suppress flash-arc noise at the electrode top.')
    ap.add_argument('--arc-direction', choices=['down', 'up'], default='down',
                    help="Reactor orientation for this recording. The 2-D velocity "
                         "map is stored in raw image rows (electrode at TOP for "
                         "'down', at BOTTOM for 'up'). For 'up' the map is flipped "
                         "vertically so depth 0 (the electrode gap) is at the top of "
                         "the figure and the depth axis is correct for both.")
    args = ap.parse_args()

    # Override module-level calibration constants so every downstream function
    # uses the correct scale for this recording. (Declared global at the top of
    # main(), since PX_PER_MM is also read in the argparse help string.)
    if args.px_per_mm is not None:
        PX_PER_MM = args.px_per_mm
        FRAME_MM  = 256.0 / PX_PER_MM
    global VELMAP_VMAX
    if args.velmap_vmax is not None:
        VELMAP_VMAX = args.velmap_vmax

    _date      = datetime.date.today().isoformat()
    _blade_dir = os.path.join(os.path.expanduser('~'), 'OneDrive', 'Desktop', 'BLADE')
    _dated_dir = os.path.join(_blade_dir, _date)
    csv_path   = args.csv       or os.path.join(_dated_dir, 'measurements_all.csv')
    out_dir    = args.out_dir   or _dated_dir
    velmap_dir = args.velmap_dir or out_dir

    print('px/mm:  %.4f  (frame = %.1f mm × %.1f mm)' % (
          PX_PER_MM, FRAME_MM, FRAME_MM))

    os.makedirs(out_dir, exist_ok=True)
    print('Input:  ' + csv_path)
    print('Output: ' + out_dir + '\n')

    df_full, det, stems, _ = load_data(csv_path)
    vmaps = load_velmaps(stems, velmap_dir)

    # Load ROI mask if define_roi.py has been run
    _roi_path = os.path.join(_blade_dir, 'roi_polygon.npy')
    if os.path.exists(_roi_path):
        from plasma_utils.roi import load_roi_mask
        _sample_vmap = next(iter(vmaps.values())) if vmaps else None
        _shape = _sample_vmap.shape[:2] if _sample_vmap is not None else (256, 256)
        roi_mask = load_roi_mask(_roi_path, _shape)
        assert roi_mask is not None
        print(f'ROI mask applied ({roi_mask.sum()} / {roi_mask.size} px)\n')
    else:
        roi_mask = None

    n_runs     = len(stems)
    n_det      = len(det)
    n_flash    = int(det['is_flash'].sum())
    n_complete = int(det['is_complete_arc'].sum())
    n_partial  = int(det['is_partial_arc'].sum())
    print('%d frames total | %d detected (%.1f%%)' % (
        len(df_full), n_det, 100.0*n_det/len(df_full)))
    print('  complete (bridging) arcs : %d (%.1f%% of detected)' % (
        n_complete, 100.0*n_complete/max(n_det,1)))
    print('  partial arcs (leader)    : %d (%.1f%% of detected)' % (
        n_partial, 100.0*n_partial/max(n_det,1)))
    print('  flash frames             : %d (%.1f%% of detected)\n' % (
        n_flash, 100.0*n_flash/max(n_det,1)))

    for run, color in zip(RUN_LABELS[:n_runs], RUN_COLORS[:n_runs]):
        fig_kymograph(det, run, color, out_dir)

    fig_tip_depth(df_full, det, out_dir)
    fig_arc_length(det, out_dir)
    fig_velocity_timeseries(det, out_dir)
    fig_velocity_dist(det, out_dir)
    fig_velocity_cdf(det, out_dir)
    fig_velocity_vs_depth(det, out_dir)
    fig_arc_area(det, out_dir)
    fig_arc_width(det, out_dir)
    fig_detection_rate(df_full, out_dir)
    fig_arc_type_rate(df_full, det, out_dir)

    # Shape metrics — only present on CSVs from the updated analyze_blade.py.
    if 'tortuosity' in det.columns and det['tortuosity'].notna().any():
        fig_tortuosity_dist(det, out_dir)
        fig_tortuosity_vs_depth(det, out_dir)
    if any(c in det.columns and det[c].notna().any()
           for c in ['arclen_asym', 'tort_asym']):
        fig_asymmetry_dist(det, out_dir)

    # ── 2-D velocity maps ──────────────────────────────────────────────────────
    # The maps are stored in raw image-row coordinates.  For the UPWARD recording
    # the electrode gap sits at the BOTTOM of the frame, so flip the map
    # vertically: like the downward recording, depth 0 (the electrode gap) is then
    # at the TOP of the figure and both the gap-row exclusion below and the
    # 'Depth (mm) [0 = electrode gap]' axis in fig_velmap are correct.
    if args.arc_direction == 'up':
        vmaps = {k: v[::-1, :] for k, v in vmaps.items()}
        if roi_mask is not None:
            roi_mask = roi_mask[::-1, :]

    # Shared vmax across all runs so the colourmap is comparable.
    if VELMAP_VMAX is not None:
        shared_vmax = VELMAP_VMAX / 1000.0
    else:
        _gap = max(1, int(round(2.0 * PX_PER_MM)))
        _pos_maps = [v / 1000.0 for v in vmaps.values()]
        _pos_maps = [p[_gap:, :] if p.shape[0] > _gap else p for p in _pos_maps]
        if _pos_maps:
            _allpos = np.concatenate(
                [p[np.isfinite(p) & (p > 0)].ravel() for p in _pos_maps])
            shared_vmax = float(np.nanpercentile(_allpos, 95)) if _allpos.size else 1.0
        else:
            shared_vmax = 1.0

    for fig_i, (run_label, stem) in enumerate(zip(RUN_LABELS[:n_runs], stems)):
        vm = vmaps.get(stem)
        if vm is None:
            continue
        fig_velmap(vm, run_label, 17 + fig_i, out_dir,
                   roi_mask=roi_mask, vmax=shared_vmax)

    print('\nDone. Figures written to: %s' % out_dir)


if __name__ == '__main__':
    main()