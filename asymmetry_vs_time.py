"""
asymmetry_vs_time.py — left-vs-right arc-length asymmetry, dynamic + summary.

The midline that bisects the two electrodes (apex bisector) is already
defined by analyze_blade.measure_frame, and the per-frame quantities are
written into measurements.csv:

    arclen_a_px / arclen_b_px : path length of each half of the skeleton (px)
    arclen_asym               : (L_A - L_B) / (L_A + L_B)
                                  -> in [-1, +1]; 0 = symmetric;
                                     +1 = entirely on A side; -1 = entirely on B.

This script:
  1. Loads measurements.csv per recording, filters to arc_filter.valid_frames
     (detected + complete + not flash), and computes the time-resolved
     arclen_asym signal.
  2. Plots arclen_asym vs time (running median band) and the corresponding
     log10(arclen_a_px / arclen_b_px) — same information, different scaling.
  3. Per recording, computes summary statistics (mean, median, IQR, std,
     fraction of time |asym| > 0.2 = "biased") and a Spearman rank-correlation
     vs time to detect monotonic drift.
  4. Across recordings, plots median|asym| and bias-fraction vs Re to look
     for any systematic Re dependence.

Outputs:
    asymmetry_timeseries_<recording>.{pdf,png,txt}  per-recording figure
    asymmetry_per_recording.csv                     summary stats per recording
    asymmetry_vs_re.{pdf,png,txt}                   cross-recording vs Re

Usage:
    python asymmetry_vs_time.py --root "C:\\Users\\compl\\Desktop\\Plasma Arc\\J Electrode"
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import arc_filter
import figstyle
import plot_config
import reynolds_comparison as rc
import sweep_utils

BIAS_THRESHOLD = 0.2          # |arclen_asym| above this counts as "biased"
RUNNING_WINDOW_FRAMES = 2001  # ~10 ms at 200 kHz; broad enough to smooth out
                              # per-event jitter but narrow enough to track
                              # any real drift across the recording.


def load_valid(results_dir):
    """Return DataFrame of valid frames with [time_ms, arclen_asym, log_AB]."""
    meta_path = results_dir / 'run_metadata.json'
    meas_path = results_dir / 'measurements.csv'
    if not meta_path.exists() or not meas_path.exists():
        return None, None
    meta = json.loads(meta_path.read_text())
    fps  = float(meta.get('fps', 200000))
    df   = pd.read_csv(meas_path)
    df   = arc_filter.add_flash_flag(df)
    keep = arc_filter.valid_frames(df)
    df   = df[keep].copy()
    if len(df) == 0:
        return None, None
    if 'arclen_asym' not in df.columns:
        return None, None
    df['time_ms'] = df['frame'].astype(float) / fps * 1e3
    a = pd.to_numeric(df.get('arclen_a_px'), errors='coerce')
    b = pd.to_numeric(df.get('arclen_b_px'), errors='coerce')
    # log10(a/b) with a small-count epsilon
    eps = 1.0
    df['log_AB'] = np.log10((a + eps) / (b + eps))
    df['arclen_asym'] = pd.to_numeric(df['arclen_asym'], errors='coerce')
    df = df.dropna(subset=['arclen_asym', 'time_ms']).reset_index(drop=True)
    return df, meta


def summary_stats(df):
    """Per-recording summary of arclen_asym + Spearman rho vs time."""
    s = df['arclen_asym'].to_numpy(dtype=float)
    t = df['time_ms'].to_numpy(dtype=float)
    n = len(s)
    if n == 0:
        return None
    abs_s = np.abs(s)
    out = dict(
        n_valid_frames    = int(n),
        mean_asym         = float(np.mean(s)),
        median_asym       = float(np.median(s)),
        std_asym          = float(np.std(s, ddof=1)) if n > 1 else float('nan'),
        median_abs_asym   = float(np.median(abs_s)),
        p10_asym          = float(np.percentile(s, 10)),
        p90_asym          = float(np.percentile(s, 90)),
        frac_biased       = float((abs_s > BIAS_THRESHOLD).mean()),
        median_log_AB     = float(np.median(df['log_AB'])),
    )
    if n >= 30:
        from scipy.stats import spearmanr
        rho = spearmanr(t, s)
        out['spearman_rho']   = float(rho.statistic)
        out['spearman_pval']  = float(rho.pvalue)
    else:
        out['spearman_rho']   = float('nan')
        out['spearman_pval']  = float('nan')
    return out


def plot_one(results_dir, df, meta, stats, out_dir):
    figstyle.apply()
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.6), sharex=True,
                              gridspec_kw=dict(height_ratios=[1.3, 1.0]))
    ax1, ax2 = axes

    t = df['time_ms'].to_numpy(dtype=float)
    y = df['arclen_asym'].to_numpy(dtype=float)
    yl = df['log_AB'].to_numpy(dtype=float)

    # Scatter underlay — heavy alpha for density.
    ax1.scatter(t, y, s=1.6, color='#1d4f7a', alpha=0.10)
    # Running median (and p10/p90 band) over a frame window.
    wf = min(RUNNING_WINDOW_FRAMES, max(51, len(df) // 30))
    if wf % 2 == 0:
        wf += 1
    s = pd.Series(y, index=t).sort_index()
    rm = s.rolling(wf, center=True, min_periods=max(20, wf // 5)).median()
    r10 = s.rolling(wf, center=True, min_periods=max(20, wf // 5)).quantile(0.10)
    r90 = s.rolling(wf, center=True, min_periods=max(20, wf // 5)).quantile(0.90)
    ax1.fill_between(rm.index, r10.values, r90.values,
                      color='#cfc28f', alpha=0.55, label=f'p10-p90 (window {wf} f)')
    ax1.plot(rm.index, rm.values, color='#00204d', lw=1.6,
              label=f'running median (window {wf} f)')
    ax1.axhline(0.0, color='#000000', lw=0.6)
    ax1.axhline(+BIAS_THRESHOLD, color='#979c91', lw=0.8, ls=':',
                 label=f'|asym| = {BIAS_THRESHOLD}')
    ax1.axhline(-BIAS_THRESHOLD, color='#979c91', lw=0.8, ls=':')
    ax1.set_ylabel('arclen_asym  =  (L_A − L_B) / (L_A + L_B)')
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False, fontsize=8, loc='upper right')
    rho_txt = (f'Spearman ρ={stats["spearman_rho"]:+.3f}  '
               f'p={stats["spearman_pval"]:.2g}'
               if stats['spearman_rho'] == stats['spearman_rho'] else
               'Spearman: insufficient n')
    ax1.set_title(f'Arc asymmetry vs time — {results_dir.name}\n'
                   f'median = {stats["median_asym"]:+.3f}   '
                   f'|asym| median = {stats["median_abs_asym"]:.3f}   '
                   f'fraction biased (|asym|>{BIAS_THRESHOLD}) = '
                   f'{stats["frac_biased"]:.2%}   {rho_txt}', fontsize=10)

    # Lower panel: log10(L_A/L_B) — same info, log scale for intuition.
    s2 = pd.Series(yl, index=t).sort_index()
    rm2 = s2.rolling(wf, center=True, min_periods=max(20, wf // 5)).median()
    ax2.scatter(t, yl, s=1.6, color='#5d7588', alpha=0.10)
    ax2.plot(rm2.index, rm2.values, color='#00204d', lw=1.4,
              label=f'running median (window {wf} f)')
    ax2.axhline(0.0, color='#000000', lw=0.6)
    ax2.set_ylabel('log10( L_A / L_B )')
    ax2.set_xlabel('Time (ms)')
    ax2.grid(alpha=0.3)
    ax2.legend(frameon=False, fontsize=8, loc='upper right')

    fig.tight_layout()
    cap = figstyle.caption_block(
        recording=meta['recording'], fps=meta['fps'],
        px_per_mm=meta['px_per_mm'], bright_thresh=meta['bright_thresh'],
        n=stats['n_valid_frames'],
        sigma_pxmm_rel=float(meta.get('sigma_pxmm_rel') or 0.05),
        sigma_fps_rel=float(meta.get('sigma_fps_rel')  or 1e-3),
        extra=(
            'Top panel: per-frame arclen_asym (defined on the apex-bisector '
            'midline) for every valid frame (arc_filter.valid_frames). '
            'Scatter at low alpha; thick line is a centered rolling-median '
            'over a frame window chosen for the recording length. '
            'Dotted lines at +/-{0:.2f} bound the "biased" classification.\n\n'
            'Bottom panel: same data shown as log10(L_A / L_B). 0 = symmetric.\n\n'
            'Summary stats (recording-wide, valid frames only):\n'
            '   mean(arclen_asym)    = {1:+.3f}\n'
            '   median(arclen_asym)  = {2:+.3f}\n'
            '   std(arclen_asym)     = {3:.3f}\n'
            '   median(|arclen_asym|) = {4:.3f}\n'
            '   p10 / p90            = {5:+.3f} / {6:+.3f}\n'
            '   fraction |asym|>{7:.2f}: {8:.2%}\n'
            '   Spearman rho(time)   = {9}  (p={10})\n\n'
            'Calibration sigma_pxmm + sigma_fps CANCEL: arclen_asym is a '
            'pixel-ratio quantity, and time on the x-axis uses fps but the '
            'within-recording rolling shape does not depend on it. The '
            'reported uncertainties are statistical / counting only.'
        ).format(BIAS_THRESHOLD,
                  stats['mean_asym'],   stats['median_asym'],
                  stats['std_asym'],    stats['median_abs_asym'],
                  stats['p10_asym'],    stats['p90_asym'],
                  BIAS_THRESHOLD,       stats['frac_biased'],
                  (f"{stats['spearman_rho']:+.3f}"
                   if stats['spearman_rho'] == stats['spearman_rho']
                   else 'n/a'),
                  (f"{stats['spearman_pval']:.2g}"
                   if stats['spearman_pval'] == stats['spearman_pval']
                   else 'n/a')))
    stem = out_dir / f'asymmetry_timeseries_{results_dir.name}'
    figstyle.save(fig, str(stem), cap)
    figstyle.verify(str(stem))


def cross_recording_plot(rows, out_dir):
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'asymmetry_per_recording.csv', index=False)
    if df.empty:
        return
    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    ax1, ax2, ax3 = axes

    def _plot(ax, ycol, yerr_col, ylab, title, show_zero=False):
        for d_mm, g in df.groupby('d_mm'):
            col = plot_config.nozzle_color(d_mm)
            yerr = g[yerr_col] if yerr_col in g.columns else None
            ax.errorbar(g['Re'], g[ycol],
                         xerr=g['sigma_Re'], yerr=yerr,
                         fmt='o', ms=8, capsize=4,
                         color=col, mec='#000000', mew=0.6,
                         label=f'd = {d_mm:g} mm')
        ax.set_xlabel('Reynolds number')
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        if show_zero:
            ax.axhline(0.0, color='#000000', lw=0.6)

    _plot(ax1, 'median_asym', 'std_se_median',
          ylab='median( arclen_asym )',
          title='Signed asymmetry vs Re',
          show_zero=True)
    _plot(ax2, 'median_abs_asym', 'std_se_median',
          ylab='median( |arclen_asym| )',
          title='|asymmetry| vs Re')
    _plot(ax3, 'frac_biased', 'frac_biased_SE',
          ylab=f'fraction frames |asym| > {BIAS_THRESHOLD}',
          title=f'Biased-frame fraction vs Re')
    ax3.legend(frameon=False, fontsize=8, loc='best', title='Nozzle d',
                title_fontsize=8)
    fig.suptitle('Arc asymmetry — cross-recording summary vs Re', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    cap = ('Cross-recording aggregation of the per-frame arclen_asym = '
           '(L_A - L_B) / (L_A + L_B) signal vs Reynolds number.\n\n'
           'Left: signed median per recording (a non-zero value means the arc '
           'is systematically biased toward one electrode across the entire '
           'recording -- likely electrode-specific geometry or wear, not Re).\n'
           'Middle: |arclen_asym| median per recording (insensitive to which '
           'side is dominant, so a Re-dependent trend here is the cleanest '
           'fluid-dynamic signal).\n'
           'Right: fraction of valid frames with |arclen_asym| > '
           f'{BIAS_THRESHOLD} (an event-time-style metric).\n\n'
           'Error bars: counting SE on counts (frac_biased) and per-recording '
           'SE of the median for the medians. Calibration systematics cancel '
           'in arclen_asym (pixel-ratio quantity).\n\n'
           'CONFOUND: changing nozzle diameter changes both Re AND reactor '
           'geometry, so any trend is conflated with electrode geometry / '
           'positioning. Per-recording median is also confounded with '
           'electrode-pair mounting accuracy (the "A" and "B" labels are '
           'arbitrary; a recording-specific bias does not imply physics).')
    figstyle.save(fig, str(out_dir / 'asymmetry_vs_re'), cap)
    figstyle.verify(str(out_dir / 'asymmetry_vs_re'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, type=str)
    ap.add_argument('--out',  type=str, default=None)
    args = ap.parse_args()
    root    = Path(args.root)
    out_dir = Path(args.out) if args.out else (root / 'Reynolds_comparison')
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in sweep_utils.iter_sweep_dirs(root):
        df, meta = load_valid(d)
        if df is None or meta is None:
            print(f"  [skip] {d.name}: no measurements / metadata / asymmetry col")
            continue
        stats = summary_stats(df)
        if stats is None:
            print(f"  [skip] {d.name}: no valid frames")
            continue
        cond = rc.parse_condition(meta.get('recording', d.name))
        if cond is None:
            print(f"  [skip] {d.name}: cannot parse Re from filename")
            continue
        d_mm, q_mLpm, is_rep = cond
        re_out = rc.compute_re(d_mm, q_mLpm)

        # SE of the median ~= 1.253 * std / sqrt(n)  for a Gaussian; for the
        # actual distribution we use the bootstrap-friendly approximation.
        n = stats['n_valid_frames']
        SE_median = 1.253 * stats['std_asym'] / np.sqrt(n) if n > 1 else float('nan')
        # SE on fraction: Wilson-style binomial sqrt(p(1-p)/n)
        p = stats['frac_biased']
        SE_frac   = float(np.sqrt(p * (1 - p) / n)) if n > 1 else float('nan')

        row = dict(
            recording=meta['recording'],
            d_mm=d_mm, q_mLpm=q_mLpm, is_repeat=int(is_rep),
            Re=re_out['Re'], sigma_Re=re_out['sigma_Re'], Ma=re_out['Ma'],
            **stats,
            std_se_median=SE_median,
            frac_biased_SE=SE_frac,
        )
        rows.append(row)
        rho_str = (f"rho={stats['spearman_rho']:+.3f}"
                   if stats['spearman_rho'] == stats['spearman_rho'] else 'rho=n/a')
        print(f"  [ok]   {d.name}: n={n} "
              f"median={stats['median_asym']:+.3f}  "
              f"|med|={stats['median_abs_asym']:.3f}  "
              f"biased_frac={stats['frac_biased']:.2%}  "
              f"{rho_str}  Re={re_out['Re']:.0f}")
        plot_one(d, df, meta, stats, d)

    cross_recording_plot(rows, out_dir)
    print(f"\nWrote: asymmetry_per_recording.csv + asymmetry_vs_re.{{pdf,png,txt}} in {out_dir}")


if __name__ == '__main__':
    main()
