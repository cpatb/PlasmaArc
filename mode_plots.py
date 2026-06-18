#!/usr/bin/env python3
"""
mode_plots.py  -  shared plotting helpers for mode (A/B) x direction (up/down)
comparison figures.

Centralises the primitives that mode_velocity_compare.py and
mode_arclength_compare.py used to duplicate, plus two "figure-set" emitters that
produce the full comparison panel for any metric:

    emit_pos_set(...)        positive metric (velocity, arc length, tortuosity)
                             -> {stub}_AB.png        Mode A vs Mode B (pooled dir)
                                {stub}_dir.png       down vs up      (pooled mode)
                                {stub}_allgroups.png all 4 groups overlaid
                                {stub}_boxplot.png   all 4 groups, box (+ jitter)

    emit_signed_panels(...)  signed metric(s) in [-x, x] (asymmetry indices),
                             rendered as a multi-panel figure, same four views.

Every figure now carries a title with the relevant Mann-Whitney U p-value(s).
"""
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, gaussian_kde

# Canonical group ordering / styling shared by every comparison figure.
GROUP_ORDER = [('down', 'A'), ('down', 'B'), ('up', 'A'), ('up', 'B')]
GLABEL = {('down', 'A'): 'A dn', ('down', 'B'): 'B dn',
          ('up',   'A'): 'A up', ('up',   'B'): 'B up'}
GCOLOR = {('down', 'A'): '#26A69A', ('down', 'B'): '#00695C',
          ('up',   'A'): '#AB47BC', ('up',   'B'): '#6A1B9A'}

# Two-class overlay colours.
COL_A, COL_B = '#1565C0', '#C62828'    # Mode A / Mode B
COL_DN, COL_UP = '#00897B', '#8E24AA'  # down / up


def mw(a, b):
    """Two-sided Mann-Whitney U p-value; NaN if either group has < 5 points."""
    a = pd.to_numeric(pd.Series(a), errors='coerce').to_numpy(float)
    b = pd.to_numeric(pd.Series(b), errors='coerce').to_numpy(float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    try:
        return mannwhitneyu(a, b, alternative='two-sided').pvalue
    except Exception:
        return np.nan


def ps(p):
    return '%.3g' % p if not np.isnan(p) else 'n/a'


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print('Saved %s' % path)


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def hist_kde_groups(ax, groups, xcap, xlabel, ylabel, title=None, n_bins=25):
    """Normalised histogram (density) with a KDE curve overlaid per group.
    A dashed vertical line marks each group MEAN. Positive support [0, xcap]."""
    for label, color, v in groups:
        v = pd.to_numeric(pd.Series(v), errors='coerce').to_numpy(float)
        v = v[np.isfinite(v) & (v > 0) & (v <= xcap)]
        if len(v) < 5:
            continue
        ax.hist(v, bins=n_bins, range=(0, xcap), density=True,
                color=color, alpha=0.25, edgecolor='none')
        x = np.linspace(0, xcap, 400)
        ax.plot(x, gaussian_kde(v)(x), color=color, lw=1.8, label=label)
        ax.axvline(float(np.mean(v)), color=color, ls='--', lw=1.2, alpha=0.9)
    ax.plot([], [], color='gray', ls='--', lw=1.2, label='dashed = mean')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(0, xcap)
    if title:
        ax.set_title(title, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def hist_kde_signed(ax, groups, xabs, xlabel, ylabel, title=None, n_bins=31):
    """Like hist_kde_groups but for SIGNED quantities (asymmetry indices in
    [-xabs, xabs]); keeps negatives and draws a zero reference line."""
    for label, color, v in groups:
        v = pd.to_numeric(pd.Series(v), errors='coerce').to_numpy(float)
        v = v[np.isfinite(v) & (np.abs(v) <= xabs)]
        if len(v) < 5:
            continue
        ax.hist(v, bins=n_bins, range=(-xabs, xabs), density=True,
                color=color, alpha=0.25, edgecolor='none')
        if np.std(v) > 1e-9:
            x = np.linspace(-xabs, xabs, 400)
            ax.plot(x, gaussian_kde(v)(x), color=color, lw=1.8, label=label)
        else:
            ax.plot([], [], color=color, lw=1.8, label=label)
        ax.axvline(float(np.mean(v)), color=color, ls='--', lw=1.2, alpha=0.9)
    ax.axvline(0.0, ls=':', lw=0.8, color='gray')
    ax.plot([], [], color='gray', ls='--', lw=1.2, label='dashed = mean')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(-xabs, xabs)
    if title:
        ax.set_title(title, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def box4_getter(ax, getter, cap, ylabel, title=None, signed=False, jitter=True):
    """Four-group (mode x direction) box plot built from getter(direction, mode).
    signed=True keeps negatives, draws a 0 line and a symmetric y-range; the
    raw-point jitter overlay is optional (off for dense per-frame data)."""
    data, labels, colors = [], [], []
    for key in GROUP_ORDER:
        v = pd.to_numeric(pd.Series(getter(*key)), errors='coerce').to_numpy(float)
        if signed:
            v = v[np.isfinite(v) & (np.abs(v) <= cap)]
        else:
            v = v[np.isfinite(v) & (v > 0) & (v <= cap)]
        data.append(v)
        colors.append(GCOLOR[key])
        mean = float(np.mean(v)) if len(v) else float('nan')
        labels.append('%s\nn=%d\nmean=%.2f' % (GLABEL[key], len(v), mean))
    bp = ax.boxplot(data, positions=range(len(data)), widths=0.6,
                    showfliers=False, patch_artist=True,
                    showmeans=True, meanline=True,
                    meanprops=dict(color='black'),
                    medianprops=dict(linewidth=0))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    if jitter:
        for i, v in enumerate(data):
            if len(v):
                jit = np.random.default_rng(0).normal(0, 0.05, len(v))
                ax.scatter(np.full(len(v), i) + jit, v, s=6,
                           c='k', alpha=0.25, zorder=3)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if signed:
        ax.axhline(0.0, ls=':', lw=0.8, color='gray')
        ax.set_ylim(-cap, cap)
    else:
        ax.set_ylim(0, cap)
    if title:
        ax.set_title(title, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ---------------------------------------------------------------------------
# helpers for the emitters
# ---------------------------------------------------------------------------
def _series(getter, dr, m):
    return pd.to_numeric(pd.Series(getter(dr, m)), errors='coerce')


def _cap_pos(getter):
    allv = pd.concat([_series(getter, *k) for k in GROUP_ORDER], ignore_index=True)
    allv = allv[np.isfinite(allv) & (allv > 0)]
    return float(np.nanpercentile(allv, 99)) if len(allv) else 1.0


def _cap_abs(getter):
    allv = pd.concat([_series(getter, *k) for k in GROUP_ORDER], ignore_index=True)
    allv = allv[np.isfinite(allv)]
    return (float(np.nanpercentile(np.abs(allv), 99)) or 1.0) if len(allv) else 1.0


def _pvals(getter):
    A  = pd.concat([_series(getter, 'down', 'A'), _series(getter, 'up', 'A')], ignore_index=True)
    B  = pd.concat([_series(getter, 'down', 'B'), _series(getter, 'up', 'B')], ignore_index=True)
    DN = pd.concat([_series(getter, 'down', 'A'), _series(getter, 'down', 'B')], ignore_index=True)
    UP = pd.concat([_series(getter, 'up', 'A'),   _series(getter, 'up', 'B')], ignore_index=True)
    return dict(A=A, B=B, DN=DN, UP=UP,
                pAB=mw(A, B), pdir=mw(DN, UP),
                pAud=mw(_series(getter, 'up', 'A'), _series(getter, 'down', 'A')),
                pBud=mw(_series(getter, 'up', 'B'), _series(getter, 'down', 'B')))


# ---------------------------------------------------------------------------
# figure-set emitters
# ---------------------------------------------------------------------------
def emit_pos_set(out_dir, stub, xlabel, title_base, getter,
                 box_name=None, jitter=True):
    """Four comparison figures for a positive metric. Returns the p-value dict
    so the caller can log a console summary."""
    cap = _cap_pos(getter)
    P = _pvals(getter)

    fig, ax = plt.subplots(figsize=(6, 4))
    hist_kde_groups(ax, [('Mode A', COL_A, P['A']), ('Mode B', COL_B, P['B'])],
                    cap, xlabel, 'Probability density',
                    '%s: Mode A vs Mode B\nMann-Whitney p = %s' % (title_base, ps(P['pAB'])))
    savefig(fig, out_dir / ('%s_AB.png' % stub))

    fig, ax = plt.subplots(figsize=(6, 4))
    hist_kde_groups(ax, [('down', COL_DN, P['DN']), ('up', COL_UP, P['UP'])],
                    cap, xlabel, 'Probability density',
                    '%s: down vs up\nMann-Whitney p = %s' % (title_base, ps(P['pdir'])))
    savefig(fig, out_dir / ('%s_dir.png' % stub))

    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [(GLABEL[k], GCOLOR[k], getter(*k)) for k in GROUP_ORDER]
    hist_kde_groups(ax, groups, cap, xlabel, 'Probability density',
                    '%s by mode x direction\n'
                    'A up vs A dn p=%s   B up vs B dn p=%s'
                    % (title_base, ps(P['pAud']), ps(P['pBud'])))
    savefig(fig, out_dir / ('%s_allgroups.png' % stub))

    fig, ax = plt.subplots(figsize=(6, 4))
    box4_getter(ax, getter, cap, xlabel,
                '%s by mode x direction\n'
                'A up vs A dn p=%s   B up vs B dn p=%s'
                % (title_base, ps(P['pAud']), ps(P['pBud'])),
                jitter=jitter)
    savefig(fig, out_dir / (box_name or ('%s_boxplot.png' % stub)))
    return P


def emit_signed_panels(out_dir, stub, panels, title_base, jitter=True):
    """Four comparison figures for one or more SIGNED metrics, each rendered as
    a 1 x len(panels) grid. `panels` is a list of (getter, name) tuples.
    Returns a list of p-value dicts (one per panel)."""
    n = len(panels)
    caps = [_cap_abs(g) for g, _ in panels]
    Ps = [_pvals(g) for g, _ in panels]

    def _new():
        fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4), squeeze=False)
        return fig, axes[0]

    # A vs B
    fig, axes = _new()
    for ax, (g, name), P, cap in zip(axes, panels, Ps, caps):
        hist_kde_signed(ax, [('Mode A', COL_A, P['A']), ('Mode B', COL_B, P['B'])],
                        cap, '%s   (+ = blade-A side)' % name, 'Probability density',
                        '%s: A vs B   p = %s' % (name, ps(P['pAB'])))
    fig.suptitle('%s: Mode A vs Mode B' % title_base, fontsize=10)
    savefig(fig, out_dir / ('%s_AB.png' % stub))

    # down vs up
    fig, axes = _new()
    for ax, (g, name), P, cap in zip(axes, panels, Ps, caps):
        hist_kde_signed(ax, [('down', COL_DN, P['DN']), ('up', COL_UP, P['UP'])],
                        cap, '%s   (+ = blade-A side)' % name, 'Probability density',
                        '%s: down vs up   p = %s' % (name, ps(P['pdir'])))
    fig.suptitle('%s: down vs up' % title_base, fontsize=10)
    savefig(fig, out_dir / ('%s_dir.png' % stub))

    # all 4 groups
    fig, axes = _new()
    for ax, (g, name), P, cap in zip(axes, panels, Ps, caps):
        groups = [(GLABEL[k], GCOLOR[k], g(*k)) for k in GROUP_ORDER]
        hist_kde_signed(ax, groups, cap,
                        '%s   (+ = blade-A side)' % name, 'Probability density',
                        '%s\nA up/dn p=%s  B up/dn p=%s'
                        % (name, ps(P['pAud']), ps(P['pBud'])))
    fig.suptitle('%s by mode x direction' % title_base, fontsize=10)
    savefig(fig, out_dir / ('%s_allgroups.png' % stub))

    # boxplot
    fig, axes = _new()
    for ax, (g, name), P, cap in zip(axes, panels, Ps, caps):
        box4_getter(ax, g, cap, '%s  (+ = blade-A side)' % name,
                    '%s\nA up/dn p=%s  B up/dn p=%s'
                    % (name, ps(P['pAud']), ps(P['pBud'])),
                    signed=True, jitter=jitter)
    fig.suptitle('%s by mode x direction' % title_base, fontsize=10)
    savefig(fig, out_dir / ('%s_boxplot.png' % stub))
    return Ps
