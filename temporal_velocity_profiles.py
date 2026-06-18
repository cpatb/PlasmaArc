"""
temporal_velocity_profiles.py — first-events vs last-events local velocity
profiles, as a function of normalised arc position.

Tests for temporal drift in the per-event propagation dynamics (heating,
electrode conditioning, gas-flow stabilisation, repeated-arc memory) by
contrasting the early-recording events with the late-recording events:

  * Per-event ``vel_ms`` versus ``norm_pos`` is already written by
    event_velocity.py into ``events/event_NNNN_dynamics.csv``.
  * events_index.csv orders events by ``frame_start``.

Outputs (one panel pair per recording supplied):
  temporal_profiles.{pdf,png,txt}        — main comparison figure
  temporal_profiles_summary.csv          — per-group binned mean / band

Usage:
    python temporal_velocity_profiles.py --results "<stem>_results"
    python temporal_velocity_profiles.py --results "..." --n-events 5
    python temporal_velocity_profiles.py --results "..." --n-events 10 --min-r2 0.7
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import figstyle
import uncertainty as U


# Bin grid for the velocity-vs-position curves (0 .. 1, inclusive endpoints).
N_BINS = 12
GROUP_COLORS = {'first': '#00204d', 'last': '#fde725'}
GROUP_FACES  = {'first': '#1d4f7a', 'last': '#cfc28f'}


def _load_dynamics(results_dir, event_ids):
    """Load and concatenate per-event dynamics CSVs for the given event_ids."""
    ev_dir = Path(results_dir) / 'events'
    frames = []
    for eid in event_ids:
        p = ev_dir / f'event_{int(eid):04d}_dynamics.csv'
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df['event_id'] = int(eid)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _bin_curve(df, n_bins=N_BINS):
    """Return (bin centres, p10, p50, p90, n_per_bin)."""
    df = df[df['vel_ms'] > 0]
    if len(df) == 0:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ctrs  = 0.5 * (edges[:-1] + edges[1:])
    p10 = np.full(n_bins, np.nan)
    p50 = np.full(n_bins, np.nan)
    p90 = np.full(n_bins, np.nan)
    n   = np.zeros(n_bins, dtype=int)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (df['norm_pos'] >= lo) & (df['norm_pos'] < hi)
        if i == n_bins - 1:                          # last bin: include 1.0
            m = (df['norm_pos'] >= lo) & (df['norm_pos'] <= hi)
        cell = df.loc[m, 'vel_ms'].to_numpy(dtype=float)
        if cell.size >= 2:                           # >=2 advancing-front frames
            p10[i], p50[i], p90[i] = np.percentile(cell, [10, 50, 90])
            n[i] = int(cell.size)
    return ctrs, p10, p50, p90, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True, type=str,
                    help='Target <stem>_results directory.')
    ap.add_argument('--n-events', type=int, default=5,
                    help='How many of the FIRST and LAST events to compare.')
    ap.add_argument('--min-r2',   type=float, default=0.5,
                    help='Filter events to those with regression r2 >= this.')
    ap.add_argument('--min-extension-px', type=float, default=30.0,
                    help='Skip events whose arc never extended beyond this many '
                         'pixels (too short to fit a curve).')
    args = ap.parse_args()
    results = Path(args.results)

    meta = json.loads((results / 'run_metadata.json').read_text())
    es_unc = results / 'event_summary_with_unc.csv'
    es_path = es_unc if es_unc.exists() else results / 'event_summary.csv'
    events = pd.read_csv(es_path)

    # Quality filter — events with a real advancing-front regression and a
    # non-trivial extent. Sort by frame_start so "first/last" is temporal.
    q = events.copy()
    q = q[(q.get('r2',         1) >= args.min_r2)
        & (q.get('extension_px', 0) >= args.min_extension_px)
        & (q.get('vel_ms',     0) > 0)]
    q = q.sort_values('frame_start').reset_index(drop=True)
    if len(q) < 2 * args.n_events:
        print(f"[WARN] only {len(q)} events pass quality filters; "
              f"reducing n_events to {len(q) // 2}.")
        args.n_events = max(1, len(q) // 2)

    first_ids = q.head(args.n_events)['event_id'].tolist()
    last_ids  = q.tail(args.n_events)['event_id'].tolist()
    print(f"FIRST events ({args.n_events}): {first_ids}")
    print(f"LAST  events ({args.n_events}): {last_ids}")

    first_df = _load_dynamics(results, first_ids)
    last_df  = _load_dynamics(results, last_ids)
    if first_df.empty or last_df.empty:
        print("[ERROR] missing per-event dynamics CSVs. Did event_velocity.py "
              "run with save_per_event_data()?")
        return

    first_bins = _bin_curve(first_df); last_bins = _bin_curve(last_df)

    # ── Figure: 1×2 panels ───────────────────────────────────────────────────
    figstyle.apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                    gridspec_kw=dict(width_ratios=[1.2, 1.0]))

    # Panel A: per-event individual curves (faint) + group p10/p50/p90 (bold)
    for group_label, group_df, group_ids in [('first', first_df, first_ids),
                                              ('last',  last_df,  last_ids)]:
        for eid, grp in group_df.groupby('event_id'):
            grp = grp[grp['vel_ms'] > 0].sort_values('norm_pos')
            ax1.plot(grp['norm_pos'], grp['vel_ms'],
                     color=GROUP_COLORS[group_label], alpha=0.30, lw=0.9)
        if group_label == 'first':
            ctrs, p10, p50, p90, n = first_bins
        else:
            ctrs, p10, p50, p90, n = last_bins
        ax1.fill_between(ctrs, p10, p90,
                          color=GROUP_FACES[group_label], alpha=0.35,
                          label=f'{group_label}-{args.n_events} p10-p90')
        ax1.plot(ctrs, p50,
                  color=GROUP_COLORS[group_label], lw=2.4,
                  label=f'{group_label}-{args.n_events} median (n events={len(group_ids)})')
    ax1.set_xlabel('Normalised arc position  (0 = nucleation, 1 = max extent)')
    ax1.set_ylabel('Local arc-tip velocity (m/s)')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(bottom=0)
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False, fontsize=8, loc='upper left')
    ax1.set_title(f'First {args.n_events} events vs last {args.n_events} events  '
                   f'(of {len(q)} qualifying)', fontsize=10)

    # Panel B: per-event median local velocity vs event time, highlight first/last
    if 'local_vel_median' in q.columns and 'frame_start' in q.columns:
        t_ms = q['frame_start'].astype(float) / float(meta.get('fps', 200000)) * 1e3
        ax2.scatter(t_ms, q['local_vel_median'],
                    s=18, color='#5d7588', alpha=0.55, label='all qualifying events')
        mask_f = q['event_id'].isin(first_ids)
        mask_l = q['event_id'].isin(last_ids)
        ax2.scatter(t_ms[mask_f], q.loc[mask_f, 'local_vel_median'],
                    s=46, color=GROUP_COLORS['first'], edgecolor='black', lw=0.7,
                    label=f'first {args.n_events}')
        ax2.scatter(t_ms[mask_l], q.loc[mask_l, 'local_vel_median'],
                    s=46, color=GROUP_COLORS['last'], edgecolor='black', lw=0.7,
                    label=f'last  {args.n_events}')
        ax2.set_xlabel('Event start time (ms from start of recording)')
        ax2.set_ylabel('Per-event median local velocity (m/s)')
        ax2.grid(alpha=0.3)
        ax2.legend(frameon=False, fontsize=8, loc='best')
        ax2.set_title('Drift across recording', fontsize=10)
    else:
        ax2.axis('off')

    # KS-style overlap report (printed + caption)
    from scipy.stats import ks_2samp
    fv = first_df.loc[first_df['vel_ms'] > 0, 'vel_ms'].to_numpy()
    lv = last_df.loc[last_df['vel_ms']  > 0, 'vel_ms'].to_numpy()
    ks = ks_2samp(fv, lv)
    delta_med = float(np.median(lv) - np.median(fv))

    pxmm_rel = float(meta.get('sigma_pxmm_rel') or U.SIGMA_PXMM_REL)
    fps_rel  = float(meta.get('sigma_fps_rel')  or U.SIGMA_FPS_REL)
    median_overall = float(np.median(np.concatenate([fv, lv])))
    sigma_sys = float(median_overall * np.sqrt(pxmm_rel ** 2 + fps_rel ** 2))

    flag = ('|delta_median| <= 1*sigma_sys'  if abs(delta_med) <= sigma_sys else
            '|delta_median| <= 2*sigma_sys'  if abs(delta_med) <= 2 * sigma_sys else
            'EXCEEDS 2*sigma_sys')
    print(f"\n  median(first) = {np.median(fv):.3f} m/s   "
          f"median(last) = {np.median(lv):.3f} m/s   "
          f"delta = {delta_med:+.3f} m/s")
    print(f"  sigma_sys (at median v): {sigma_sys:.3f} m/s   ->  {flag}")
    print(f"  KS test on raw local-vel samples: D={ks.statistic:.3f}  "
          f"p={ks.pvalue:.3g}  "
          f"(small p => significantly different distributions)")

    fig.suptitle(f'Temporal effects on propagation velocity — {results.name}',
                  fontsize=11, y=1.02)
    fig.tight_layout()

    cap = figstyle.caption_block(
        recording=meta['recording'], fps=meta['fps'],
        px_per_mm=meta['px_per_mm'], bright_thresh=meta['bright_thresh'],
        n=int(len(first_df) + len(last_df)),
        sigma_pxmm_rel=pxmm_rel, sigma_fps_rel=fps_rel,
        extra=(
            f'Group sizes: first={args.n_events}, last={args.n_events} '
            f'(of {len(q)} events with r2>={args.min_r2} and '
            f'extension>={args.min_extension_px:.0f} px).\n'
            f'Left panel: thin lines are per-event local-velocity curves; '
            f'thick line is the across-event median per norm_pos bin; band '
            f'is p10-p90 across events in that bin.\n'
            f'Right panel: per-event median local velocity vs event start time '
            f'across the whole recording (drift check).\n'
            f'Statistics: median(first)={np.median(fv):.3f} m/s, '
            f'median(last)={np.median(lv):.3f} m/s, delta={delta_med:+.3f} m/s. '
            f'sigma_sys at the recording-median velocity = {sigma_sys:.3f} m/s -> {flag}. '
            f'2-sample KS on raw advancing-front vel_ms: D={ks.statistic:.3f}, '
            f'p={ks.pvalue:.3g}.\n'
            f'Caveat: the systematic (px/mm + fps) is identical for both groups '
            f'within a recording, so it CANCELS in the first-vs-last difference. '
            f'For drift, use the statistical / repeat spread, NOT sigma_sys.'
        ))
    stem = results / 'temporal_profiles'
    figstyle.save(fig, str(stem), cap)
    figstyle.verify(str(stem))

    # CSV summary of the two binned curves
    def _to_df(group_label, bins, ids):
        ctrs, p10, p50, p90, n = bins
        return pd.DataFrame({
            'group':       group_label,
            'event_ids':   ', '.join(map(str, ids)),
            'norm_pos':    ctrs,
            'p10_vel_ms':  p10,
            'p50_vel_ms':  p50,
            'p90_vel_ms':  p90,
            'n_obs':       n,
        })
    out_csv = results / 'temporal_profiles_summary.csv'
    pd.concat([_to_df('first', first_bins, first_ids),
               _to_df('last',  last_bins,  last_ids)],
              ignore_index=True).to_csv(out_csv, index=False)
    print(f"\nWrote: {stem}.pdf, .png, .txt")
    print(f"       {out_csv}")


if __name__ == '__main__':
    main()
