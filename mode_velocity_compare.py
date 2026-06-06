#!/usr/bin/env python3
"""
mode_velocity_compare.py  -  Arc velocity by Mode (A/B) and direction (up/down).

Figures saved
-------------
mode_vel_local_allgroups.png  -- per-FRAME local velocity, 4 groups (histogram+KDE)
mode_vel_advance_AB.png       -- per-CYCLE advance velocity, A vs B
mode_vel_advance_dir.png      -- per-CYCLE advance velocity, down vs up
mode_vel_boxplot.png          -- per-CYCLE advance velocity, 4-group box+jitter
mode_dur_AB.png               -- per-CYCLE propagation duration (ms), A vs B
mode_dur_dir.png              -- per-CYCLE propagation duration, down vs up
mode_dur_boxplot.png          -- per-CYCLE propagation duration, 4-group box+jitter

Advance velocity = linear regression slope of arc-tip position from the
re-strike frame to the first extension peak, converted to m/s.

Propagation duration = number of frames from re-strike to first extension peak,
converted to ms at the recording fps.

Usage (PowerShell):
    python mode_velocity_compare.py ^
        --down PATH/downward_results --up PATH/upward_results
        --px-per-mm 8.8 --out-dir OUTPUT_DIR
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, gaussian_kde

from mode_common import analyze_recording
import arc_filter  # single source of truth for flash / velocity-cap criteria

GROUP_ORDER = [('down', 'A'), ('down', 'B'), ('up', 'A'), ('up', 'B')]
GLABEL = {('down', 'A'): 'A dn', ('down', 'B'): 'B dn',
          ('up',   'A'): 'A up', ('up',   'B'): 'B up'}
GCOLOR = {('down', 'A'): '#26A69A', ('down', 'B'): '#00695C',
          ('up',   'A'): '#AB47BC', ('up',   'B'): '#6A1B9A'}


def resolve(p):
    p = Path(p)
    return p if p.suffix == '.csv' else p / 'measurements_all_events.csv'


def mw(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    try:
        return mannwhitneyu(a, b, alternative='two-sided').pvalue
    except Exception:
        return np.nan


def ps(p):
    return '%.3g' % p if not np.isnan(p) else 'n/a'


def hist_kde_groups(ax, groups, xcap, xlabel, ylabel, title, n_bins=25):
    """Normalised histogram (density) with KDE curve overlaid per group."""
    for label, color, v in groups:
        v = np.asarray(v, float)
        v = v[np.isfinite(v) & (v > 0) & (v <= xcap)]
        if len(v) < 5:
            continue
        ax.hist(v, bins=n_bins, range=(0, xcap), density=True,
                color=color, alpha=0.25, edgecolor='none')
        x = np.linspace(0, xcap, 400)
        ax.plot(x, gaussian_kde(v)(x), color=color, lw=1.8,
                label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(0, xcap)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def box4_generic(ax, data, labels, colors, cap, ylabel, title):
    bp = ax.boxplot(data, positions=range(len(data)), widths=0.6,
                    showfliers=False, patch_artist=True,
                    showmeans=True, meanline=True,
                    meanprops=dict(color='black'),
                    medianprops=dict(linewidth=0))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    for i, v in enumerate(data):
        if len(v):
            jit = np.random.default_rng(0).normal(0, 0.05, len(v))
            ax.scatter(np.full(len(v), i) + jit, v, s=6,
                       c='k', alpha=0.25, zorder=3)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, cap)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print('Saved %s' % path)


def main():
    ap = argparse.ArgumentParser(description='Velocity by Mode and direction')
    ap.add_argument('--down', required=True)
    ap.add_argument('--up',   required=True)
    ap.add_argument('--px-per-mm',           type=float, default=8.8)
    ap.add_argument('--fps',                 type=float, default=200_000)
    ap.add_argument('--method',              choices=['abs', 'rel'], default='abs')
    ap.add_argument('--threshold',           type=float, default=0.10)
    ap.add_argument('--rel-threshold',       type=float, default=0.70)
    ap.add_argument('--ext-prominence-frac', type=float, default=0.10)
    ap.add_argument('--flash-eff-width',     type=float, default=arc_filter.FLASH_EFF_WIDTH,
                    help='flash if area_px/arc_length_px > this (px). 0 disables.')
    ap.add_argument('--flash-arc-width',     type=float, default=arc_filter.FLASH_ARC_WIDTH,
                    help='flash if arc_width_px (FWHM) > this (px). 0 disables.')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--out',     default=None, help='legacy; parent dir used')
    args = ap.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.out:
        out_dir = Path(args.out).parent
    else:
        raise SystemExit('Supply --out-dir or --out')
    out_dir.mkdir(parents=True, exist_ok=True)

    prs, pfs = [], []
    for direction, p in [('down', args.down), ('up', args.up)]:
        csv = resolve(p)
        if not csv.exists():
            print('[WARN] missing %s -- skipping %s' % (csv, direction))
            continue
        pr, pf = analyze_recording(
            csv, direction, args.px_per_mm, args.fps,
            args.method, args.threshold, args.rel_threshold,
            args.ext_prominence_frac,
            flash_eff_width=args.flash_eff_width,
            flash_arc_width=args.flash_arc_width)
        prs.append(pr); pfs.append(pf)
    if not prs:
        raise SystemExit('No data.')
    PR = pd.concat(prs, ignore_index=True)
    PF = pd.concat(pfs, ignore_index=True)
    PF['local_vel_ms'] = pd.to_numeric(PF['local_vel_ms'], errors='coerce')  # no velocity cap
    PR.to_csv(out_dir / 'mode_velocity_restrikes.csv', index=False)

    # --- advance velocity (reported as measured; flashes already removed) ---
    av = PR.dropna(subset=['advance_vel_ms'])
    A  = av[av['mode'] == 'A']['advance_vel_ms']
    B  = av[av['mode'] == 'B']['advance_vel_ms']
    UP = av[av['direction'] == 'up'  ]['advance_vel_ms']
    DN = av[av['direction'] == 'down']['advance_vel_ms']
    def g_av(dr, m):
        return av[(av['direction'] == dr) & (av['mode'] == m)]['advance_vel_ms']
    p_ab_v  = mw(A, B)
    p_ud_v  = mw(UP, DN)
    p_Aud_v = mw(g_av('up', 'A'), g_av('down', 'A'))
    p_Bud_v = mw(g_av('up', 'B'), g_av('down', 'B'))
    print('\nAdvance velocity (m/s):')
    print('  A vs B p=%s   up vs dn p=%s   A-up vs A-dn p=%s   B-up vs B-dn p=%s'
          % (ps(p_ab_v), ps(p_ud_v), ps(p_Aud_v), ps(p_Bud_v)))

    # --- propagation duration ---
    has_dur = ('duration_frames' in PR.columns and PR['duration_frames'].notna().any())
    if has_dur:
        dv  = PR.dropna(subset=['duration_frames'])
        fps = args.fps
        dA  = dv[dv['mode'] == 'A']['duration_frames'] / fps * 1000.0
        dB  = dv[dv['mode'] == 'B']['duration_frames'] / fps * 1000.0
        dUP = dv[dv['direction'] == 'up'  ]['duration_frames'] / fps * 1000.0
        dDN = dv[dv['direction'] == 'down']['duration_frames'] / fps * 1000.0
        def g_dur_ms(dr, m):
            return dv[(dv['direction'] == dr) & (dv['mode'] == m)]['duration_frames'] / fps * 1000.0
        p_ab_d  = mw(dA, dB)
        p_ud_d  = mw(dUP, dDN)
        p_Aud_d = mw(g_dur_ms('up', 'A'), g_dur_ms('down', 'A'))
        p_Bud_d = mw(g_dur_ms('up', 'B'), g_dur_ms('down', 'B'))
        print('\nPropagation duration (ms):')
        print('  A vs B p=%s   up vs dn p=%s   A-up vs A-dn p=%s   B-up vs B-dn p=%s'
              % (ps(p_ab_d), ps(p_ud_d), ps(p_Aud_d), ps(p_Bud_d)))
        all_dur = pd.concat([dA, dB])
        dur_cap = float(np.nanpercentile(all_dur[all_dur > 0], 99)) if (all_dur > 0).any() else 1.0

    lv_cap = (float(np.nanpercentile(PF['local_vel_ms'].dropna(), 99))
              if PF['local_vel_ms'].notna().any() else 1.0)
    av_cap = float(np.nanpercentile(av['advance_vel_ms'], 99)) if len(av) else 1.0

    flash_note = 'flashes removed (eff_width>%.0f OR arc_width_px>%.0f)' % (
        args.flash_eff_width, args.flash_arc_width)

    # Figure 1: per-frame local velocity, all 4 groups
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [(GLABEL[k], GCOLOR[k],
               PF[(PF['direction'] == k[0]) & (PF['cycle_mode'] == k[1])]['local_vel_ms'])
              for k in GROUP_ORDER]
    hist_kde_groups(ax, groups, lv_cap,
                    xlabel='Per-frame local arc-tip velocity (m/s)',
                    ylabel='Probability density',
                    title='Local velocity distribution by mode x direction\n'
                          '(per detected frame, OLS sliding window; %s)' % flash_note)
    savefig(fig, out_dir / 'mode_vel_local_allgroups.png')

    # Figure 2: advance velocity, A vs B
    fig, ax = plt.subplots(figsize=(6, 4))
    hist_kde_groups(ax,
                    [('Mode A', '#1565C0', A), ('Mode B', '#C62828', B)],
                    av_cap,
                    xlabel='Advance velocity (m/s)',
                    ylabel='Probability density',
                    title='Arc advance velocity per re-strike cycle: A vs B\n'
                          '(linear regression: nucleation to extension peak)\n'
                          'Mann-Whitney p = %s' % ps(p_ab_v))
    savefig(fig, out_dir / 'mode_vel_advance_AB.png')

    # Figure 3: advance velocity, up vs down
    fig, ax = plt.subplots(figsize=(6, 4))
    hist_kde_groups(ax,
                    [('down', '#00897B', DN), ('up', '#8E24AA', UP)],
                    av_cap,
                    xlabel='Advance velocity (m/s)',
                    ylabel='Probability density',
                    title='Arc advance velocity per re-strike cycle: down vs up\n'
                          '(linear regression: nucleation to extension peak)\n'
                          'Mann-Whitney p = %s' % ps(p_ud_v))
    savefig(fig, out_dir / 'mode_vel_advance_dir.png')

    # Figure 4: advance velocity, 4-group boxplot
    fig, ax = plt.subplots(figsize=(6, 4))
    data4, labels4, colors4 = [], [], []
    for key in GROUP_ORDER:
        v = np.asarray(g_av(*key), float)
        v = v[np.isfinite(v) & (v > 0) & (v <= av_cap)]
        data4.append(v); colors4.append(GCOLOR[key])
        med = np.mean(v) if len(v) else float('nan')
        labels4.append('%s\nn=%d\nmed=%.2f' % (GLABEL[key], len(v), med))
    box4_generic(ax, data4, labels4, colors4, av_cap,
                 ylabel='Advance velocity (m/s)',
                 title='Arc advance velocity by mode x direction\n'
                       'A-up vs A-dn p=%s   B-up vs B-dn p=%s'
                       % (ps(p_Aud_v), ps(p_Bud_v)))
    savefig(fig, out_dir / 'mode_vel_boxplot.png')

    if has_dur:
        # Figure 5: duration, A vs B
        fig, ax = plt.subplots(figsize=(6, 4))
        hist_kde_groups(ax,
                        [('Mode A', '#1565C0', dA), ('Mode B', '#C62828', dB)],
                        dur_cap,
                        xlabel='Propagation duration (ms)',
                        ylabel='Probability density',
                        title='Arc propagation duration per re-strike cycle: A vs B\n'
                              '(frames from nucleation to extension peak at %.0f fps)\n'
                              'Mann-Whitney p = %s' % (args.fps, ps(p_ab_d)))
        savefig(fig, out_dir / 'mode_dur_AB.png')

        # Figure 6: duration, up vs down
        fig, ax = plt.subplots(figsize=(6, 4))
        hist_kde_groups(ax,
                        [('down', '#00897B', dDN), ('up', '#8E24AA', dUP)],
                        dur_cap,
                        xlabel='Propagation duration (ms)',
                        ylabel='Probability density',
                        title='Arc propagation duration per re-strike cycle: down vs up\n'
                              'Mann-Whitney p = %s' % ps(p_ud_d))
        savefig(fig, out_dir / 'mode_dur_dir.png')

        # Figure 7: duration, 4-group boxplot
        fig, ax = plt.subplots(figsize=(6, 4))
        data7, labels7, colors7 = [], [], []
        for key in GROUP_ORDER:
            v = np.asarray(g_dur_ms(*key), float)
            v = v[np.isfinite(v) & (v > 0) & (v <= dur_cap)]
            data7.append(v); colors7.append(GCOLOR[key])
            med = np.mean(v) if len(v) else float('nan')
            labels7.append('%s\nn=%d\nmed=%.2f ms' % (GLABEL[key], len(v), med))
        box4_generic(ax, data7, labels7, colors7, dur_cap,
                     ylabel='Propagation duration (ms)',
                     title='Arc propagation duration by mode x direction\n'
                           'A-up vs A-dn p=%s   B-up vs B-dn p=%s'
                           % (ps(p_Aud_d), ps(p_Bud_d)))
        savefig(fig, out_dir / 'mode_dur_boxplot.png')
    else:
        print('[WARN] duration_frames not available -- skipping duration figures.')


if __name__ == '__main__':
    main()
