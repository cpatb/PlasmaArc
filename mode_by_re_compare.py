#!/usr/bin/env python3
"""
mode_by_re_compare.py  -  Per-cycle max arc extent and advance velocity by Mode
(A/B), POOLED across all Reynolds numbers.

This is the "all Re, by-mode" version of mode_arclength_compare.py /
mode_velocity_compare.py.  Those scripts compare one recording's up/down runs;
here we instead pool the four sweep recordings (one per Re) and split only by
mode.  Re-strike detection, flash filtering and mode classification are the
SAME code path (mode_common.analyze_recording) used by the original figures, so
the distributions are directly comparable to the per-recording ones.

Per-recording px/mm is read from each results dir's j_geometry.json: arc length
is already stored in mm in the CSV (per-recording calibration baked in), but the
advance-velocity regression needs the correct px/mm per recording.

Outputs (into --out-dir):
  mode_by_re_percycle.csv        raw data, one row per re-strike cycle
                                 (recording, Re, mode, cycle_max_arclen_mm,
                                  advance_vel_ms, duration_ms, norm_pos, frame)
  mode_by_re_arclen_AB.png/.pdf  per-cycle max arc length, Mode A vs B
  mode_by_re_advvel_AB.png/.pdf  advance velocity, Mode A vs B
  mode_by_re_summary.txt         n / median / mean / MWU p per metric & mode
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, mannwhitneyu

from mode_common import analyze_recording

FPS = 200_000

# The four sweep recordings (one per Re), matching the velocity-map figure
# labels.  Repeats are excluded (as in the Reynolds sweep).  Re values are the
# paper's LOCKED numbers (as in hs_figures.py), not the recalculated sweep
# values (9759/6506/4879/3253) — all shared CSVs use the paper numbers.
RECORDINGS = [
    ('HS_0.4mm_1500mL_(CO2)_trig',    9717),
    ('HS_0.4mm_1000mL_(CO2)_trig',    6478),
    ('HS_0.8mm_1500mL_(CO2)_trig(2)', 4859),
    ('HS_1.2mm_1500mL_(CO2)',         3239),
]

COL_A, COL_B = '#1565C0', '#C62828'   # Mode A (blue) / Mode B (red)


def px_per_mm_for(results_dir):
    geo = results_dir / 'j_geometry.json'
    return float(json.loads(geo.read_text())['px_per_mm'])


def _clean(s):
    s = pd.to_numeric(pd.Series(s), errors='coerce').to_numpy(float)
    return s[np.isfinite(s) & (s > 0)]


def mwu(a, b):
    a, b = _clean(a), _clean(b)
    if len(a) < 5 or len(b) < 5:
        return np.nan
    return mannwhitneyu(a, b, alternative='two-sided').pvalue


def plot_AB(a_vals, b_vals, xlabel, out_stub, dpi, n_bins=25):
    """Mode A vs Mode B: density histogram + KDE, legend carries n and median.
    Reproduces the collaborator's poster figure style."""
    a, b = _clean(a_vals), _clean(b_vals)
    cap = float(np.nanpercentile(np.concatenate([a, b]), 99))
    x = np.linspace(0, cap, 400)

    fig, ax = plt.subplots(figsize=(6, 4))
    for v, color, name in [(a, COL_A, 'Mode A'), (b, COL_B, 'Mode B')]:
        vc = v[v <= cap]
        ax.hist(vc, bins=n_bins, range=(0, cap), density=True,
                color=color, alpha=0.22, edgecolor='none')
        kde = gaussian_kde(vc)(x)
        ax.plot(x, kde, color=color, lw=1.8,
                label='%s  n=%d  med=%.2f' % (name, len(v), np.median(v)))
        ax.fill_between(x, kde, color=color, alpha=0.12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability density')
    ax.set_xlim(0, cap)
    ax.legend(frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig('%s.%s' % (out_stub, ext), dpi=dpi,
                    facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print('Saved %s.png / .pdf' % out_stub)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=r'C:\Users\compl\Desktop\Plasma Arc\J Electrode')
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--dpi', type=int, default=600)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else root / 'mode_by_re_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    for stem, re_val in RECORDINGS:
        rdir = root / ('%s_results' % stem)
        csv = rdir / 'measurements_all_events.csv'
        if not csv.exists():
            print('[WARN] missing %s -- skipping' % csv)
            continue
        pxmm = px_per_mm_for(rdir)
        print('\n%s  (Re=%d, px/mm=%.4f)' % (stem, re_val, pxmm))
        pr, _pf = analyze_recording(csv, 'down', px_per_mm=pxmm, fps=FPS)
        pr = pr.copy()

        # Replace the (superseded norm_pos>0.10) classifier label with the
        # AUTHORITATIVE valley-cut mode from the tip detector (re_strikes_qa.csv).
        # Re-strike detection is the same tip path, so frames align 1:1; we join
        # nearest-frame with a tight tolerance and keep the valley mode.
        pr['mode_normpos'] = pr['mode']
        rs = rdir / 're_strikes_qa.csv'
        if rs.exists():
            v = (pd.read_csv(rs)[['frame', 'mode']]
                 .rename(columns={'mode': 'mode_valley'})
                 .sort_values('frame'))
            pr = pd.merge_asof(pr.sort_values('frame'), v, on='frame',
                               direction='nearest', tolerance=3)
            matched = pr['mode_valley'].notna().mean()
            print('    valley-mode match: %.0f%%' % (100 * matched))
            pr['mode'] = pr['mode_valley'].fillna(pr['mode_normpos'])
        else:
            print('[WARN] no re_strikes_qa.csv -- keeping norm_pos mode')

        pr.insert(0, 'recording', stem)
        pr.insert(1, 'Re', re_val)
        pr['duration_ms'] = pr['duration_frames'] * 1000.0 / FPS
        parts.append(pr)

    if not parts:
        raise SystemExit('No data found.')
    PR = pd.concat(parts, ignore_index=True)

    # ---- two minimal raw-data CSVs, one per mode definition ----------------
    # Same rows/metrics in both; only the `mode` column differs. Every row keeps
    # `Re` + `recording` so any single-Re (or pooled) by-mode comparison can be
    # regenerated by filtering. `frame` is the re-strike (cycle-start) frame.
    METRIC_COLS = ['cycle_max_arclen_mm', 'advance_vel_ms', 'cycle_mean_local_vel',
                   'duration_ms', 'norm_pos', 'frame']
    METRIC_COLS = [c for c in METRIC_COLS if c in PR.columns]

    def write_minimal(mode_col, path):
        out = PR[['recording', 'Re']].copy()
        out['mode'] = PR[mode_col]
        for c in METRIC_COLS:
            out[c] = PR[c]
        out = out.sort_values(['Re', 'mode', 'frame'],
                              ascending=[False, True, True])
        out.to_csv(path, index=False)
        print('Saved %s  (%d cycles)' % (path, len(out)))
        return out

    raw = write_minimal('mode', out_dir / 'mode_by_re_percycle_VALLEY.csv')
    write_minimal('mode_normpos', out_dir / 'mode_by_re_percycle_LEGACY.csv')

    A = PR[PR['mode'] == 'A']
    B = PR[PR['mode'] == 'B']

    plot_AB(A['cycle_max_arclen_mm'], B['cycle_max_arclen_mm'],
            'Per-cycle maximum arc length (mm)',
            str(out_dir / 'mode_by_re_arclen_AB'), args.dpi)
    plot_AB(A['advance_vel_ms'], B['advance_vel_ms'],
            'Advance velocity (m/s)',
            str(out_dir / 'mode_by_re_advvel_AB'), args.dpi)
    plot_AB(A['cycle_mean_local_vel'], B['cycle_mean_local_vel'],
            'Local tip velocity (m/s)',
            str(out_dir / 'mode_by_re_localvel_AB'), args.dpi)

    # ---- per-Re individual A-vs-B figures (valley modes) -------------------
    per_re_dir = out_dir / 'per_Re'
    per_re_dir.mkdir(exist_ok=True)
    for stem, re_val in RECORDINGS:
        g = PR[PR['recording'] == stem]
        ga, gb = g[g['mode'] == 'A'], g[g['mode'] == 'B']
        for col, xlab, tag in [('cycle_max_arclen_mm',
                                'Per-cycle maximum arc length (mm)', 'arclen'),
                               ('advance_vel_ms', 'Advance velocity (m/s)', 'advvel'),
                               ('cycle_mean_local_vel',
                                'Local tip velocity (m/s)', 'localvel')]:
            try:
                plot_AB(ga[col], gb[col], '%s  (Re %d)' % (xlab, re_val),
                        str(per_re_dir / ('Re%d_%s_AB' % (re_val, tag))), args.dpi)
            except Exception as e:
                print('[WARN] Re %d %s: %s' % (re_val, tag, e))

    # summary sidecar
    lines = ['Per-cycle max arc extent & advance velocity by mode, pooled over all Re',
             'Recordings pooled (Re): ' + ', '.join('%s (%d)' % (s, r) for s, r in RECORDINGS),
             'fps=%d ; per-recording px/mm from j_geometry.json ; flash filter as in' % FPS,
             'mode_arclength_compare / mode_velocity_compare.',
             'Mode A/B = AUTHORITATIVE dead-zone valley cut (tip detector, re_strikes_qa.csv);',
             'raw CSV also keeps the superseded norm_pos>0.10 label as mode_normpos.',
             'All cycles from all four Re pooled equally (1.2mm dominates the count).',
             '']
    for label, col in [('Per-cycle max arc length (mm)', 'cycle_max_arclen_mm'),
                       ('Advance velocity (m/s) [per-cycle nucleation->peak regression]',
                        'advance_vel_ms'),
                       ('Local tip velocity (m/s) [per-cycle mean of sliding-window OLS]',
                        'cycle_mean_local_vel')]:
        a, b = _clean(A[col]), _clean(B[col])
        lines.append(label)
        lines.append('  Mode A: n=%d  median=%.3f  mean=%.3f' % (len(a), np.median(a), np.mean(a)))
        lines.append('  Mode B: n=%d  median=%.3f  mean=%.3f' % (len(b), np.median(b), np.mean(b)))
        lines.append('  Mann-Whitney U (two-sided) p = %.3g' % mwu(a, b))
        lines.append('')
    # per-recording breakdown
    lines.append('Per-recording cycle counts (mode A / B, valid arc-length):')
    for stem, re_val in RECORDINGS:
        sub = PR[PR['recording'] == stem]
        na = len(_clean(sub[sub['mode'] == 'A']['cycle_max_arclen_mm']))
        nb = len(_clean(sub[sub['mode'] == 'B']['cycle_max_arclen_mm']))
        lines.append('  Re %-5d %-32s A=%-4d B=%-4d' % (re_val, stem, na, nb))
    (out_dir / 'mode_by_re_summary.txt').write_text('\n'.join(lines))
    print('\n' + '\n'.join(lines))


if __name__ == '__main__':
    main()
