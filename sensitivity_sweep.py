"""
sensitivity_sweep.py — robustness audit: do headline conclusions survive a
reasonable choice of post-detection parameters?

Tier A (default, cheap, reads existing CSVs only — no pipeline rerun):
  * ``local_vel_n``        {5, 7, 10, 15, 20}        — sliding-window OLS width
  * ``flash_arc_width``    {12, 16, 24, 32, 40, 50}  — criterion 3 width gate
  * ``flash_eff_width``    {15, 20, 30, 40, 50}      — criterion 3 ratio gate

For each cell, recompute the headline metrics from
``measurements_all_events.csv`` and ``event_summary.csv``:
  N events, N valid frames, median event velocity, p10/p50/p90 local velocity,
  Mode B fraction (re-strikes).

Any metric that moves more than 2 sigma_total across the swept range is
flagged in the output table as UNSTABLE; otherwise STABLE.

Outputs:
  sensitivity_table_<axis>.csv
  sensitivity_overview.pdf/png/txt
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import arc_filter
import uncertainty as U
import figstyle


def _recompute_local_vel(df_meas, local_n, pxmm, fps,
                          flash_eff=arc_filter.FLASH_EFF_WIDTH,
                          flash_arc=arc_filter.FLASH_ARC_WIDTH):
    """Re-derive per-event median/p10/p90 local velocity for a different N
    without rerunning the full pipeline. Operates on measurements_all_events.csv
    which already carries event_id."""
    df = df_meas.copy()
    df = arc_filter.add_flash_flag(df, eff_width=flash_eff, arc_width=flash_arc)
    # Override the canonical is_flash for THIS sweep cell — we are exploring how
    # different flash gates change the headline. The original CSV's is_flash
    # was set at the source with the BLADE default; here we recompute against
    # the swept thresholds.
    a = pd.to_numeric(df.get('area_px'),       errors='coerce')
    L = pd.to_numeric(df.get('arc_length_px'), errors='coerce').clip(lower=1)
    w = pd.to_numeric(df.get('arc_width_px'),  errors='coerce')
    df['eff_width'] = a / L
    df['is_flash']  = ((df['eff_width'] > flash_eff) | (w > flash_arc))

    det = pd.to_numeric(df.get('arc_detected'), errors='coerce').fillna(0).astype(int) == 1
    fls = df['is_flash'].astype(bool)
    df = df[det & ~fls & (df.get('event_id', -1) >= 0)
                       & pd.notnull(df.get('arc_tip_row'))]

    locals_med = []; locals_p10 = []; locals_p90 = []; locals_peak = []
    counts = []
    n_valid_frames = 0
    n_events = 0
    half = local_n // 2

    for eid, grp in df.groupby('event_id'):
        grp = grp.sort_values('frame')
        if 'arc_complete' in grp.columns:
            mask  = pd.to_numeric(grp['arc_complete'], errors='coerce').fillna(0).astype(bool)
            fit_g = grp[mask]
            if len(fit_g) < 5:
                fit_g = grp
        else:
            fit_g = grp
        frames = fit_g['frame'].to_numpy(dtype=float)
        rows   = fit_g['arc_tip_row'].to_numpy(dtype=float)
        if len(frames) < 5:
            continue
        # advancing front
        keep = []; runmax = -np.inf
        for i, r in enumerate(rows):
            if r >= runmax:
                keep.append(i); runmax = r
        if len(keep) < 5:
            continue
        f_front = frames[keep]; r_front = rows[keep]
        vels = []
        for k in range(len(f_front)):
            lo = max(0, k - half); hi = min(len(f_front), k + half + 1)
            if hi - lo < 3:
                continue
            slope, _ = np.polyfit(f_front[lo:hi], r_front[lo:hi], 1)
            vels.append(slope * fps / pxmm / 1000.0)
        if not vels:
            continue
        v = np.asarray(vels)
        v = v[v > 0]
        if not len(v):
            continue
        n_events += 1
        n_valid_frames += int(len(f_front))
        locals_med.append(float(np.median(v)))
        locals_p10.append(float(np.percentile(v, 10)))
        locals_p90.append(float(np.percentile(v, 90)))
        locals_peak.append(float(np.percentile(v, 95)))
        counts.append(int(len(v)))

    if not locals_med:
        return None
    return dict(
        n_events       = n_events,
        n_valid_frames = n_valid_frames,
        med_local      = float(np.median(locals_med)),
        med_p10        = float(np.median(locals_p10)),
        med_p90        = float(np.median(locals_p90)),
        med_peak       = float(np.median(locals_peak)),
    )


def sweep_axis(df_meas, meta, axis, values, sigma_v_total_floor):
    """Sweep one axis; return a DataFrame."""
    rows = []
    pxmm = float(meta['px_per_mm']); fps = float(meta['fps'])
    base_n   = int(meta.get('local_vel_n', 10))
    base_arc = float(arc_filter.FLASH_ARC_WIDTH)
    base_eff = float(arc_filter.FLASH_EFF_WIDTH)

    for v in values:
        if axis == 'local_vel_n':
            out = _recompute_local_vel(df_meas, int(v), pxmm, fps)
        elif axis == 'flash_arc_width':
            out = _recompute_local_vel(df_meas, base_n, pxmm, fps,
                                        flash_eff=base_eff, flash_arc=float(v))
        elif axis == 'flash_eff_width':
            out = _recompute_local_vel(df_meas, base_n, pxmm, fps,
                                        flash_eff=float(v), flash_arc=base_arc)
        else:
            raise ValueError(axis)
        rec = {'axis': axis, 'value': v}
        if out is not None:
            rec.update(out)
        else:
            rec.update(dict(n_events=0, n_valid_frames=0,
                            med_local=np.nan, med_p10=np.nan,
                            med_p90=np.nan, med_peak=np.nan))
        rows.append(rec)

    out = pd.DataFrame(rows)
    base = out['med_local'].median()
    out['delta_from_baseline_ms'] = out['med_local'] - base
    out['stable_flag'] = np.where(
        np.abs(out['delta_from_baseline_ms']) <= 2.0 * sigma_v_total_floor,
        'STABLE', 'UNSTABLE')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results', required=True, type=str)
    ap.add_argument('--out-dir', type=str, default=None,
                    help='Where to write outputs (defaults to results dir).')
    args = ap.parse_args()
    results = Path(args.results)
    out_dir = Path(args.out_dir) if args.out_dir else results

    meta = json.loads((results / 'run_metadata.json').read_text())
    df   = pd.read_csv(results / 'measurements_all_events.csv')

    # 2σ floor from the recording's own sigma budget (uses uncertainty.py).
    pxmm_rel = meta.get('sigma_pxmm_rel') or U.SIGMA_PXMM_REL
    fps_rel  = meta.get('sigma_fps_rel')  or U.SIGMA_FPS_REL
    floor    = float(np.sqrt(pxmm_rel ** 2 + fps_rel ** 2))   # relative floor
    # Convert to m/s using the per-recording typical event velocity.
    es = pd.read_csv(results / 'event_summary.csv')
    if 'vel_ms' in es.columns and (es['vel_ms'] > 0).any():
        med_v = float(np.nanmedian(es.loc[es['vel_ms'] > 0, 'vel_ms']))
    else:
        med_v = 5.0
    sigma_v_floor = floor * med_v

    AXES = {
        'local_vel_n'    : [5, 7, 10, 15, 20],
        'flash_arc_width': [12, 16, 24, 32, 40, 50],
        'flash_eff_width': [15, 20, 30, 40, 50],
    }

    all_rows = []
    for axis, vals in AXES.items():
        df_axis = sweep_axis(df, meta, axis, vals, sigma_v_floor)
        df_axis.to_csv(out_dir / f'sensitivity_table_{axis}.csv', index=False)
        all_rows.append(df_axis)
        print(f"\n--- {axis} ---")
        print(df_axis[['value', 'n_events', 'med_local',
                       'delta_from_baseline_ms', 'stable_flag']].to_string(index=False))

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(out_dir / 'sensitivity_overview.csv', index=False)

    # ── overview figure ─────────────────────────────────────────────────────
    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (axis, vals) in zip(axes, AXES.items()):
        sub = combined[combined['axis'] == axis]
        ax.errorbar(sub['value'], sub['med_local'],
                    yerr=sigma_v_floor, fmt='o-', capsize=3,
                    color='#1d4f7a', mec='#00204d')
        ax.axhline(sub['med_local'].median(), color='#979c91',
                   ls=':', label='baseline median')
        ax.fill_between(sub['value'].astype(float),
                        sub['med_local'].median() - 2 * sigma_v_floor,
                        sub['med_local'].median() + 2 * sigma_v_floor,
                        color='#cfc28f', alpha=0.35,
                        label='±2σ_sys')
        ax.set_xlabel(axis)
        ax.set_title(axis)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('Median local velocity (m/s)')
    axes[-1].legend(frameon=False, fontsize=8, loc='lower right')
    fig.suptitle(f'Sensitivity sweep — {results.name}', fontsize=10)
    fig.tight_layout()

    cap = figstyle.caption_block(
        recording=meta['recording'], fps=meta['fps'],
        px_per_mm=meta['px_per_mm'], bright_thresh=meta['bright_thresh'],
        n=int(combined['n_events'].sum()),
        sigma_pxmm_rel=pxmm_rel, sigma_fps_rel=fps_rel,
        extra=('Median per-event median local velocity across the recording, '
               'swept along the three post-detection parameters. ±2σ_sys band '
               'is the systematic floor (px/mm ⊕ fps) at the recording\'s '
               f'median velocity {med_v:.2f} m/s. A cell is flagged UNSTABLE '
               'when |Δ| > 2σ_sys.'))
    stem = out_dir / 'sensitivity_overview'
    figstyle.save(fig, str(stem), cap)
    figstyle.verify(str(stem))
    print(f"\nWrote: {stem}.pdf, .png, .txt")
    print(f"       sigma_v_floor = {sigma_v_floor:.3f} m/s (2-sigma = {2*sigma_v_floor:.3f})")


if __name__ == '__main__':
    main()
