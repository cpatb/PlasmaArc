#!/usr/bin/env python3
"""
mode_arclength_compare.py  —  Arc-length distributions by Mode (A/B) and
propagation direction (up/down). Flash frames are excluded (see mode_common.py).

What is being plotted
---------------------
Figure 1  mode_arclen_perframe_allgroups.png
    Per-FRAME arc length (mm) — the skeleton length of the arc measured from
    every arc-detected, non-flash frame, tagged with the mode of the re-strike
    cycle it belongs to.  One bin = one frame.
    Binned histogram (density) with KDE curve overlaid.
    Four groups: Mode A ↓, Mode B ↓, Mode A ↑, Mode B ↑.

Figure 2  mode_arclen_maxpercycle_AB.png
    Per-CYCLE maximum arc length (mm) — the largest arc length (mm) reached
    within each re-strike cycle.  One bin = one re-strike cycle.
    Groups: Mode A vs Mode B, pooled across directions.
    Mann-Whitney U p-value shown in title.

Figure 3  mode_arclen_maxpercycle_dir.png
    Same per-CYCLE maximum arc length, grouped by direction (down vs up),
    pooled across modes.

Figure 4  mode_arclen_boxplot.png
    Per-CYCLE maximum arc length: 4-group box + jitter (mode × direction).
    Provides n, median, and inter-quartile range for each group.

Usage:
    python mode_arclength_compare.py \\
        --down "<...downward_results>" --up "<...upward_results>" \\
        --px-per-mm 8.8 --out-dir "<...results_dir>"
--out is also accepted for backward compatibility (parent dir is used).
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
GLABEL = {('down', 'A'): 'A down', ('down', 'B'): 'B down',
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
    """
    For each group: draw a normalised histogram (density) then overlay
    a KDE smoothing curve.  Histograms show the actual data; the KDE
    curve traces the envelope.
    """
    for label, color, v in groups:
        v = np.asarray(v, float)
        v = v[np.isfinite(v) & (v > 0) & (v <= xcap)]
        if len(v) < 5:
            continue
        # Histogram — bars (density normalised so area = 1)
        ax.hist(v, bins=n_bins, range=(0, xcap), density=True,
                color=color, alpha=0.25, edgecolor='none')
        # KDE curve overlaid
        x = np.linspace(0, xcap, 400)
        ax.plot(x, gaussian_kde(v)(x), color=color, lw=1.8,
                label=label)
        # dashed vertical line at the group MEAN (not median)
        ax.axvline(float(np.mean(v)), color=color, ls='--', lw=1.2, alpha=0.9)
    ax.plot([], [], color='gray', ls='--', lw=1.2, label='dashed = mean')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(0, xcap)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def hist_kde_signed(ax, groups, xabs, xlabel, ylabel, n_bins=31):
    """Like hist_kde_groups but for SIGNED quantities (e.g. asymmetry indices in
    [-1, 1]).  Plots over [-xabs, xabs] with a reference line at 0; does not drop
    negative values."""
    for label, color, v in groups:
        v = np.asarray(v, float)
        v = v[np.isfinite(v) & (np.abs(v) <= xabs)]
        if len(v) < 5:
            continue
        ax.hist(v, bins=n_bins, range=(-xabs, xabs), density=True,
                color=color, alpha=0.25, edgecolor='none')
        if np.std(v) > 1e-9:
            x = np.linspace(-xabs, xabs, 400)
            ax.plot(x, gaussian_kde(v)(x), color=color, lw=1.8,
                    label=label)
        else:
            ax.plot([], [], color=color, lw=1.8,
                    label=label)
        # dashed vertical line at the group MEAN (not median)
        ax.axvline(float(np.mean(v)), color=color, ls='--', lw=1.2, alpha=0.9)
    ax.axvline(0.0, ls=':', lw=0.8, color='gray')  # zero reference (dotted)
    ax.plot([], [], color='gray', ls='--', lw=1.2, label='dashed = mean')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(-xabs, xabs)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def box4(ax, getter, cap, ylabel, title):
    data, labels, colors = [], [], []
    for key in GROUP_ORDER:
        v = np.asarray(getter(*key), float)
        v = v[np.isfinite(v) & (v > 0) & (v <= cap)]
        data.append(v)
        colors.append(GCOLOR[key])
        labels.append(GLABEL[key])
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
    ap = argparse.ArgumentParser(
        description='Arc length by Mode (A/B) and direction (up/down)')
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
    # --out-dir preferred; --out kept for backward compat (parent dir is used)
    ap.add_argument('--out-dir', default=None,
                    help='Directory for output PNGs')
    ap.add_argument('--out',     default=None,
                    help='(legacy) single output path; parent dir is used instead')
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
            print('[WARN] missing %s — skipping %s' % (csv, direction)); continue
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
    PR.to_csv(out_dir / 'mode_arclength_restrikes.csv', index=False)

    # per-re-strike maximum arc length (one value per cycle)
    mx  = PR.dropna(subset=['cycle_max_arclen_mm'])
    A   = mx[mx['mode'] == 'A']['cycle_max_arclen_mm']
    B   = mx[mx['mode'] == 'B']['cycle_max_arclen_mm']
    UP  = mx[mx['direction'] == 'up'  ]['cycle_max_arclen_mm']
    DN  = mx[mx['direction'] == 'down']['cycle_max_arclen_mm']
    def g_pr(dr, m):
        return mx[(mx['direction'] == dr) & (mx['mode'] == m)]['cycle_max_arclen_mm']
    p_ab  = mw(A, B)
    p_ud  = mw(UP, DN)
    p_Aud = mw(g_pr('up', 'A'), g_pr('down', 'A'))
    p_Bud = mw(g_pr('up', 'B'), g_pr('down', 'B'))
    print('\nPer-re-strike MAX arc length (mm):')
    print('  A vs B p=%s   up vs down p=%s   A↑vsA↓ p=%s   B↑vsB↓ p=%s'
          % (ps(p_ab), ps(p_ud), ps(p_Aud), ps(p_Bud)))

    cap_pr = float(np.nanpercentile(mx['cycle_max_arclen_mm'], 99)) \
             if len(mx) else 1.0
    cap_pf = float(np.nanpercentile(PF['arc_length_mm'].dropna(), 99)) \
             if PF['arc_length_mm'].notna().any() else 1.0

    flash_note = 'flashes removed (eff_width>%.0f OR arc_width_px>%.0f)' % (
        args.flash_eff_width, args.flash_arc_width)

    # ── Figure 1: per-frame arc length — all 4 groups ────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [(GLABEL[k], GCOLOR[k],
               PF[(PF['direction'] == k[0]) & (PF['cycle_mode'] == k[1])]['arc_length_mm'])
              for k in GROUP_ORDER]
    hist_kde_groups(ax, groups, cap_pf,
                    xlabel='Per-frame arc length (mm)',
                    ylabel='Probability density',
                    title='Arc-length distribution by mode × direction\n'
                          '(per detected frame; %s)' % flash_note)
    savefig(fig, out_dir / 'mode_arclen_perframe_allgroups.png')

    # ── Figure 2: per-cycle max arc length — A vs B ──────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    hist_kde_groups(ax,
                    [('Mode A', '#1565C0', A), ('Mode B', '#C62828', B)],
                    cap_pr,
                    xlabel='Per-cycle maximum arc length (mm)',
                    ylabel='Probability density',
                    title='Max arc length per re-strike cycle: A vs B\n'
                          'Mann-Whitney p = %s' % ps(p_ab))
    savefig(fig, out_dir / 'mode_arclen_maxpercycle_AB.png')

    # ── Figure 3: per-cycle max arc length — up vs down ──────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    hist_kde_groups(ax,
                    [('down', '#00897B', DN), ('up', '#8E24AA', UP)],
                    cap_pr,
                    xlabel='Per-cycle maximum arc length (mm)',
                    ylabel='Probability density',
                    title='Max arc length per re-strike cycle: down vs up\n'
                          'Mann-Whitney p = %s' % ps(p_ud))
    savefig(fig, out_dir / 'mode_arclen_maxpercycle_dir.png')

    # ── Figure 4: per-cycle max arc length — 4-group boxplot ─────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    box4(ax, g_pr, cap_pr,
         ylabel='Per-cycle maximum arc length (mm)',
         title='Max arc length by mode × direction\n'
               'A↑vsA↓ p=%s   B↑vsB↓ p=%s' % (ps(p_Aud), ps(p_Bud)))
    savefig(fig, out_dir / 'mode_arclen_boxplot.png')

    # ── Shape metrics by mode (only on re-analysed CSVs with the new columns) ──
    # tortuosity = path length / electrode-gap chord; arclen_asym / tort_asym are
    # signed about the electrode midline (+ = blade-A side, see analyze_blade.py).
    def _has(col):
        return col in PR.columns and PR[col].notna().any()

    if _has('cycle_mean_tortuosity'):
        to = PR.dropna(subset=['cycle_mean_tortuosity'])
        Ato = to[to['mode'] == 'A']['cycle_mean_tortuosity']
        Bto = to[to['mode'] == 'B']['cycle_mean_tortuosity']
        p_to = mw(Ato, Bto)
        print('\nPer-cycle mean TORTUOSITY:  A vs B p=%s' % ps(p_to))
        cap_to = float(np.nanpercentile(to['cycle_mean_tortuosity'], 99))
        fig, ax = plt.subplots(figsize=(6, 4))
        hist_kde_groups(ax,
                        [('Mode A', '#1565C0', Ato), ('Mode B', '#C62828', Bto)],
                        cap_to,
                        xlabel='Per-cycle mean tortuosity (path / chord)',
                        ylabel='Probability density',
                        title='Tortuosity per re-strike cycle: A vs B\n'
                              'Mann-Whitney p = %s' % ps(p_to))
        savefig(fig, out_dir / 'mode_tortuosity_AB.png')

    if _has('cycle_mean_arclen_asym') or _has('cycle_mean_tort_asym'):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, col, name in [
                (axes[0], 'cycle_mean_arclen_asym', 'Arc-length asymmetry'),
                (axes[1], 'cycle_mean_tort_asym',   'Tortuosity asymmetry')]:
            if not _has(col):
                ax.set_visible(False); continue
            sub = PR.dropna(subset=[col])
            A = sub[sub['mode'] == 'A'][col]; B = sub[sub['mode'] == 'B'][col]
            p = mw(A, B)
            print('Per-cycle mean %-22s A vs B p=%s' % (name + ':', ps(p)))
            xabs = float(np.nanpercentile(np.abs(sub[col]), 99)) or 1.0
            hist_kde_signed(ax,
                            [('Mode A', '#1565C0', A), ('Mode B', '#C62828', B)],
                            xabs,
                            xlabel='%s   (+ = blade-A side)' % name,
                            ylabel='Probability density')
        savefig(fig, out_dir / 'mode_asymmetry_AB.png')


if __name__ == '__main__':
    main()
