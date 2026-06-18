"""
warmup_time.py — characteristic warm-up time per recording.

Per-event median local velocity vs event start time is fit to an
exponential approach to steady state:

    v(t) = v_inf - (v_inf - v_0) * exp(-t / tau)

where:
  v_0    = cold-start velocity (event at t=0)
  v_inf  = steady-state velocity (t -> infinity)
  tau    = characteristic warm-up time (same units as t — here, milliseconds)

Each event provides one (t, v) data point weighted by its own statistical SE
(local_vel_std / sqrt(n_local_frames)) — events with steady sliding-window
velocity get more weight than noisy ones. Calibration systematics (px/mm,
fps) are common to all events within the recording and DROP OUT of the fit;
this is a within-recording analysis, NOT a cross-recording one.

Outputs:
    warmup_fit_<recording>.{pdf,png,txt}
    warmup_per_recording.csv         (one row per recording)
    warmup_summary.{pdf,png,txt}     (cross-recording panel)

Usage:
    # one recording:
    python warmup_time.py --results "<stem>_results"
    # whole sweep:
    python warmup_time.py --root "C:\\Users\\compl\\Desktop\\Plasma Arc\\J Electrode"
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import figstyle


def _expo(t, v_inf, dv, tau):
    """v(t) = v_inf - dv * exp(-t / tau)   with dv = v_inf - v_0 (>=0 if warming up)."""
    return v_inf - dv * np.exp(-t / tau)


def fit_one(t_ms, v_ms, v_se=None,
            tau_min=0.05, tau_max=None):
    """Robust exponential fit. Returns dict with params, errors, R^2, and a
    flag describing whether the fit is well-conditioned for THIS recording."""
    t = np.asarray(t_ms, dtype=float)
    v = np.asarray(v_ms, dtype=float)
    good = np.isfinite(t) & np.isfinite(v) & (v > 0)
    t = t[good]; v = v[good]
    if v_se is not None:
        v_se = np.asarray(v_se, dtype=float)[good]
        v_se = np.where(np.isfinite(v_se) & (v_se > 0), v_se,
                         max(np.median(v_se[np.isfinite(v_se)]) if np.any(np.isfinite(v_se)) else 1.0, 0.3))
    if len(t) < 8:
        return dict(ok=False, reason=f'only {len(t)} events',
                    n=len(t), tau_ms=np.nan, sigma_tau_ms=np.nan,
                    v_0=np.nan, v_inf=np.nan, r2=np.nan,
                    t=t, v=v, v_se=v_se)

    t_span = float(t.max() - t.min())
    if tau_max is None:
        # 5x the recording duration — past this the fit is "no decay"
        tau_max = max(5 * t_span, 200.0)

    # Initial guesses: v_inf = median of late half, v_0 = median of first 10%,
    # tau = ~10% of span. dv = v_inf - v_0 must be >=0 in our convention; if
    # the data look like the arc SLOWS down over time (cooling rather than
    # warming), allow a negative dv by relaxing the bound (but mark the fit
    # so the report can flag it).
    half      = max(int(len(t) * 0.5), 3)
    first_q   = max(int(len(t) * 0.1), 3)
    v_inf_g   = float(np.median(np.sort(v)[-half:]))
    v_0_g     = float(np.median(np.sort(np.column_stack([t, v]),
                                          axis=0)[:first_q, 1]))
    dv_g      = v_inf_g - v_0_g
    tau_g     = max(t_span * 0.1, 0.5)

    # Bounds: v_inf in [0, 30], dv in [-25, 25], tau in [tau_min, tau_max].
    p0     = [v_inf_g, dv_g, tau_g]
    bounds = ([0.0, -25.0, tau_min],
              [40.0, +25.0, tau_max])

    try:
        popt, pcov = curve_fit(_expo, t, v, p0=p0, bounds=bounds,
                                sigma=v_se, absolute_sigma=v_se is not None,
                                maxfev=20000)
    except Exception as e:
        return dict(ok=False, reason=f'curve_fit failed: {e}',
                    n=len(t), tau_ms=np.nan, sigma_tau_ms=np.nan,
                    v_0=np.nan, v_inf=np.nan, r2=np.nan,
                    t=t, v=v, v_se=v_se)

    v_inf, dv, tau = popt
    v_0            = v_inf - dv
    perr           = np.sqrt(np.diag(pcov))
    sigma_v_inf, sigma_dv, sigma_tau = perr

    # R^2
    pred = _expo(t, *popt)
    ss_res = float(np.sum((v - pred) ** 2))
    ss_tot = float(np.sum((v - v.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Conditioning flags
    flags = []
    if tau > 0.5 * t_span:
        flags.append('tau > 0.5 * recording duration (lower bound only)')
    if tau < tau_min * 5:
        flags.append('tau near the lower bound — likely <1 event resolution')
    if abs(dv) < 2 * sigma_dv:
        flags.append('dv consistent with zero — no significant warm-up signal')
    if sigma_tau > 0.5 * tau:
        flags.append('sigma_tau > 50% of tau — poorly constrained')
    if not np.isfinite(r2) or r2 < 0.1:
        flags.append(f'R²={r2:.2f} — exponential model captures little of the variance')

    return dict(ok=True, reason='', n=len(t),
                tau_ms=float(tau), sigma_tau_ms=float(sigma_tau),
                v_0=float(v_0), v_inf=float(v_inf),
                sigma_v_0=float(np.sqrt(sigma_v_inf ** 2 + sigma_dv ** 2)),
                sigma_v_inf=float(sigma_v_inf),
                dv=float(dv), sigma_dv=float(sigma_dv),
                r2=float(r2), flags=', '.join(flags) if flags else '',
                t=t, v=v, v_se=v_se, popt=popt)


def _per_event_series(results_dir, min_r2=0.5, min_ext_px=30.0):
    """Return (t_ms, v_ms, v_se, recording_name, fps). v_ms is the per-event
    median local velocity (preferred) or vel_ms (event regression) fallback."""
    es = results_dir / 'event_summary_with_unc.csv'
    if not es.exists():
        es = results_dir / 'event_summary.csv'
    if not es.exists():
        return None
    meta = json.loads((results_dir / 'run_metadata.json').read_text())
    df   = pd.read_csv(es)
    df   = df[(df.get('r2', 1) >= min_r2)
            & (df.get('extension_px', 0) >= min_ext_px)
            & (df.get('vel_ms', 0) > 0)].copy()
    if df.empty:
        return None
    df = df.sort_values('frame_start').reset_index(drop=True)
    fps = float(meta.get('fps', 200000))
    t_ms = df['frame_start'].astype(float) / fps * 1e3

    if 'local_vel_median' in df.columns:
        v_ms = df['local_vel_median'].astype(float).to_numpy()
    else:
        v_ms = df['vel_ms'].astype(float).to_numpy()

    if 'local_vel_std' in df.columns and 'n_local_frames' in df.columns:
        n_lf = pd.to_numeric(df['n_local_frames'], errors='coerce').clip(lower=1)
        v_se = (pd.to_numeric(df['local_vel_std'], errors='coerce')
                / np.sqrt(n_lf)).to_numpy()
    else:
        v_se = np.full_like(v_ms, np.nan)

    return dict(t_ms=t_ms.to_numpy(dtype=float), v_ms=v_ms, v_se=v_se,
                recording=meta.get('recording', results_dir.name),
                fps=fps, meta=meta, n_qualifying=len(df))


def plot_one(results_dir, series, fit, out_dir):
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    t = series['t_ms']; v = series['v_ms']; v_se = series['v_se']
    if v_se is not None and np.isfinite(v_se).any():
        ax.errorbar(t, v, yerr=v_se, fmt='o', ms=5, capsize=2,
                     color='#5d7588', mec='#00204d', mew=0.5, alpha=0.7,
                     label='per-event median local velocity')
    else:
        ax.scatter(t, v, s=18, color='#5d7588', alpha=0.7,
                    label='per-event median local velocity')

    if fit.get('ok'):
        tt = np.linspace(0.0, float(t.max()) * 1.05, 400)
        ax.plot(tt, _expo(tt, *fit['popt']), color='#00204d', lw=2.2,
                label=(r'exp. fit: $\tau$ = '
                       f"{fit['tau_ms']:.2f}±{fit['sigma_tau_ms']:.2f} ms"))
        ax.axhline(fit['v_inf'], color='#fde725', lw=1.0, ls='--',
                    label=f"v∞ = {fit['v_inf']:.2f}±{fit['sigma_v_inf']:.2f} m/s")
        # 3-tau "steady-state reached" marker
        t3 = 3 * fit['tau_ms']
        if t3 < t.max():
            ax.axvline(t3, color='#cfc28f', lw=1.0, ls=':',
                        label=f'3τ = {t3:.2f} ms')

    ax.set_xlabel('Event start time (ms from start of recording)')
    ax.set_ylabel('Per-event median local velocity (m/s)')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9, loc='best')
    title = f'Warm-up fit — {series["recording"]}'
    if fit.get('flags'):
        title += '\n[flags] ' + fit['flags']
    ax.set_title(title, fontsize=10)
    fig.tight_layout()

    meta = series['meta']
    cap = figstyle.caption_block(
        recording=meta['recording'], fps=meta['fps'],
        px_per_mm=meta['px_per_mm'], bright_thresh=meta['bright_thresh'],
        n=fit.get('n', 0),
        sigma_pxmm_rel=float(meta.get('sigma_pxmm_rel') or 0.05),
        sigma_fps_rel=float(meta.get('sigma_fps_rel')  or 1e-3),
        extra=(
            'Per-event median sliding-window OLS velocity (m/s) as a function '
            'of event start time (ms from recording start). Solid curve is the '
            'least-squares fit to v(t) = v_inf - (v_inf - v_0) * exp(-t/tau). '
            'Weighted by per-event SE = local_vel_std / sqrt(n_local_frames) '
            'where available. Calibration systematics (px/mm + fps) are common '
            'to every point and DROP OUT of the within-recording fit, so the '
            'reported sigma_tau is the statistical fit error only.\n\n'
            f'tau           = {fit.get("tau_ms", float("nan")):.3f} +- '
            f'{fit.get("sigma_tau_ms", float("nan")):.3f} ms\n'
            f'v_inf         = {fit.get("v_inf", float("nan")):.3f} +- '
            f'{fit.get("sigma_v_inf", float("nan")):.3f} m/s\n'
            f'v_0           = {fit.get("v_0", float("nan")):.3f} m/s\n'
            f'R^2           = {fit.get("r2", float("nan")):.3f}\n'
            f'n_events      = {fit.get("n", 0)}\n'
            f'fit flags     : {fit.get("flags", "") or "(none)"}\n\n'
            'Caveat: a recording shorter than ~3*tau cannot see the asymptote '
            'cleanly, so the v_inf fit is a lower bound there. Flags surface '
            'this and other degeneracies.'
        ))
    stem = out_dir / f'warmup_fit_{results_dir.name}'
    figstyle.save(fig, str(stem), cap)
    figstyle.verify(str(stem))


def cross_recording_plot(rows, out_dir):
    """tau per recording (left) and tau vs Re where available (right)."""
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'warmup_per_recording.csv', index=False)

    figstyle.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                    gridspec_kw=dict(width_ratios=[1.2, 1.0]))

    short = (df['recording'].str.replace('.cine', '', regex=False)
                              .str.replace('HS_', '', regex=False)
                              .str.replace('(CO2)_trig', '', regex=False)
                              .str.replace('_(2)', '_2', regex=False)
                              .str.strip('_'))
    x = np.arange(len(df))
    ax1.bar(x, df['tau_ms'],
             yerr=df['sigma_tau_ms'], color='#5d7588',
             edgecolor='black', linewidth=0.5, capsize=4,
             error_kw=dict(elinewidth=1.0))
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, rotation=20, ha='right', fontsize=8)
    ax1.set_ylabel('Warm-up time  τ  (ms)')
    ax1.set_title('Characteristic warm-up time per recording', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Right panel: tau vs Re if we can recover Re from the filename.
    try:
        import reynolds_comparison as rc
        Re_vals = []; Re_err = []
        for rec in df['recording']:
            cond = rc.parse_condition(rec.replace('.cine', ''))
            if cond is None:
                Re_vals.append(np.nan); Re_err.append(np.nan); continue
            d_mm, q_mLpm, _ = cond
            re_out = rc.compute_re(d_mm, q_mLpm)
            Re_vals.append(re_out['Re']); Re_err.append(re_out['sigma_Re'])
        df['Re']        = Re_vals
        df['sigma_Re']  = Re_err
        ok = df['Re'].notna() & df['tau_ms'].notna()
        if ok.any():
            ax2.errorbar(df.loc[ok, 'Re'], df.loc[ok, 'tau_ms'],
                          xerr=df.loc[ok, 'sigma_Re'],
                          yerr=df.loc[ok, 'sigma_tau_ms'],
                          fmt='o', ms=8, capsize=3,
                          color='#1d4f7a', mec='#00204d')
            for _, r in df.loc[ok].iterrows():
                ax2.annotate(r['recording']
                              .replace('.cine', '')
                              .replace('HS_', '')
                              .replace('(CO2)_trig', '')
                              .replace('_(2)', '_2')
                              .strip('_'),
                              (r['Re'], r['tau_ms']),
                              xytext=(6, 4), textcoords='offset points',
                              fontsize=7, color='#444')
            ax2.set_xlabel('Reynolds number (CO₂ at 295 K, 101 kPa)')
            ax2.set_ylabel('Warm-up time  τ  (ms)')
            ax2.grid(alpha=0.3)
            ax2.set_title('τ vs Re', fontsize=10)
        else:
            ax2.axis('off')
    except Exception:
        ax2.axis('off')

    fig.suptitle('Warm-up time across the J-electrode sweep', fontsize=11, y=1.02)
    fig.tight_layout()
    cap = ('Left: per-recording exponential-warm-up time tau (ms), with '
           'statistical 1-sigma SE from the fit. Right: tau vs incompressible '
           'Reynolds number (same Re as reynolds_comparison.py). A real '
           'thermal/fluid-dynamic warm-up should scale with the dominant '
           'physical timescale (gas residence time tau_res ~ d/v_bulk for the '
           'gas, or an electrode-conduction time for the metal). With this '
           'small sample (5 recordings) we are not claiming a functional form -- '
           'just plotting it.')
    figstyle.save(fig, str(out_dir / 'warmup_summary'), cap)
    figstyle.verify(str(out_dir / 'warmup_summary'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--results', type=str, help='Single <stem>_results directory')
    g.add_argument('--root',    type=str, help='Root containing *_results dirs')
    ap.add_argument('--out',    type=str, default=None,
                     help='Where to write cross-recording summary (default: '
                          'when --root, <root>/Reynolds_comparison; when '
                          '--results, the per-recording dir.)')
    ap.add_argument('--min-r2', type=float, default=0.5)
    ap.add_argument('--min-ext-px', type=float, default=30.0)
    args = ap.parse_args()

    if args.results:
        targets = [Path(args.results)]
        out_dir = Path(args.out) if args.out else Path(args.results)
    else:
        root = Path(args.root)
        targets = sorted(d for d in root.glob('*_results') if d.is_dir())
        out_dir = Path(args.out) if args.out else (root / 'Reynolds_comparison')
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in targets:
        series = _per_event_series(d, min_r2=args.min_r2,
                                       min_ext_px=args.min_ext_px)
        if series is None:
            print(f"  [skip] {d.name}: no event_summary.csv / run_metadata.json")
            continue
        fit = fit_one(series['t_ms'], series['v_ms'], v_se=series['v_se'])
        rows.append(dict(
            recording=series['recording'],
            n_qualifying=series['n_qualifying'],
            n_used=fit.get('n', 0),
            tau_ms=fit.get('tau_ms', np.nan),
            sigma_tau_ms=fit.get('sigma_tau_ms', np.nan),
            v_0=fit.get('v_0', np.nan),
            v_inf=fit.get('v_inf', np.nan),
            sigma_v_inf=fit.get('sigma_v_inf', np.nan),
            dv=fit.get('dv', np.nan),
            sigma_dv=fit.get('sigma_dv', np.nan),
            r2=fit.get('r2', np.nan),
            flags=fit.get('flags', ''),
            ok=fit.get('ok', False),
            reason=fit.get('reason', ''),
        ))
        # Per-recording figure goes into the recording's own _results dir.
        plot_one(d, series, fit, d)
        tag = (f"tau={fit.get('tau_ms', float('nan')):.3f}+-"
               f"{fit.get('sigma_tau_ms', float('nan')):.3f}ms  "
               f"v_inf={fit.get('v_inf', float('nan')):.2f}m/s  "
               f"R2={fit.get('r2', float('nan')):.3f}  "
               f"flags=[{fit.get('flags', '') or '-'}]")
        print(f"  [ok]   {series['recording']}: {tag}")
    if args.root:
        cross_recording_plot(rows, out_dir)
        print(f"\nWrote: warmup_per_recording.csv + warmup_summary.{{pdf,png,txt}} in {out_dir}")


if __name__ == '__main__':
    main()
