#!/usr/bin/env python3
"""
mode_arclength_compare.py  -  Arc shape distributions by Mode (A/B) and
propagation direction (up/down). Flash frames are excluded (see mode_common.py).

For every metric that exists in BOTH a per-FRAME (local) and a per-CYCLE
(per-event) form, the full comparison panel is produced:
    {stub}_AB.png         Mode A vs Mode B   (pooled over direction)
    {stub}_dir.png        down vs up         (pooled over mode)
    {stub}_allgroups.png  all four mode x direction groups overlaid
    {stub}_boxplot.png    all four groups as a box plot

Metrics
-------
Arc length        local  PF['arc_length_mm']         stub mode_arclen_perframe
                  event  PR['cycle_max_arclen_mm']    stub mode_arclen_maxpercycle
                                                       (box -> mode_arclen_boxplot.png)
Tortuosity        local  PF['tortuosity'] (>=1)       stub mode_tortuosity_local
                  event  PR['cycle_mean_tortuosity']  stub mode_tortuosity
                                                       (AB  -> mode_tortuosity_AB.png)
Asymmetry         local  PF['arclen_asym'],           stub mode_asymmetry_local
(signed, 2-panel)        PF['tort_asym']
                  event  PR['cycle_mean_arclen_asym'],stub mode_asymmetry
                         PR['cycle_mean_tort_asym']    (AB  -> mode_asymmetry_AB.png)

Tortuosity = path length / electrode-gap chord. Asymmetry indices are signed
about the electrode midline (+ = blade-A side, see analyze_blade.py). Shape
columns are present only on CSVs produced by the updated analyze_blade.py;
metrics whose columns are missing are skipped.

Usage:
    python mode_arclength_compare.py \
        --down "<...downward_results>" --up "<...upward_results>" \
        --px-per-mm 8.8 --out-dir "<...results_dir>"
--out is also accepted for backward compatibility (parent dir is used).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from mode_common import analyze_recording
import arc_filter  # single source of truth for flash / velocity-cap criteria
from mode_plots import emit_pos_set, emit_signed_panels, ps


def resolve(p):
    p = Path(p)
    return p if p.suffix == '.csv' else p / 'measurements_all_events.csv'


def _pf_getter(PF, col, floor=None):
    """Closure returning per-frame `col` for a (direction, mode) group."""
    def g(dr, m):
        s = PF[(PF['direction'] == dr) & (PF['cycle_mode'] == m)][col]
        s = pd.to_numeric(s, errors='coerce')
        return s[s >= floor] if floor is not None else s
    return g


def _pr_getter(df, col):
    """Closure returning per-cycle `col` for a (direction, mode) group."""
    def g(dr, m):
        return pd.to_numeric(
            df[(df['direction'] == dr) & (df['mode'] == m)][col], errors='coerce')
    return g


def _log(tag, P):
    print('  %-26s A vs B p=%s   down vs up p=%s   A up/dn p=%s   B up/dn p=%s'
          % (tag, ps(P['pAB']), ps(P['pdir']), ps(P['pAud']), ps(P['pBud'])))


def main():
    ap = argparse.ArgumentParser(
        description='Arc shape by Mode (A/B) and direction (up/down)')
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
    ap.add_argument('--out-dir', default=None, help='Directory for output PNGs')
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
            print('[WARN] missing %s -- skipping %s' % (csv, direction)); continue
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

    def pf_has(col):
        return col in PF.columns and PF[col].notna().any()

    def pr_has(col):
        return col in PR.columns and PR[col].notna().any()

    # Arc length
    print('\nArc length (mm):')
    if pf_has('arc_length_mm'):
        P = emit_pos_set(out_dir, 'mode_arclen_perframe',
                         'Per-frame arc length (mm)', 'Per-frame arc length',
                         _pf_getter(PF, 'arc_length_mm'), jitter=False)
        _log('per-frame (local):', P)
    if pr_has('cycle_max_arclen_mm'):
        P = emit_pos_set(out_dir, 'mode_arclen_maxpercycle',
                         'Per-cycle max arc length (mm)', 'Per-cycle max arc length',
                         _pr_getter(PR.dropna(subset=['cycle_max_arclen_mm']),
                                    'cycle_max_arclen_mm'),
                         box_name='mode_arclen_boxplot.png', jitter=True)
        _log('per-cycle max:', P)

    # Tortuosity (path / chord)
    print('\nTortuosity (path / chord):')
    if pf_has('tortuosity'):
        P = emit_pos_set(out_dir, 'mode_tortuosity_local',
                         'Per-frame tortuosity (path / chord)', 'Per-frame tortuosity',
                         _pf_getter(PF, 'tortuosity', floor=1.0), jitter=False)
        _log('per-frame (local):', P)
    if pr_has('cycle_mean_tortuosity'):
        P = emit_pos_set(out_dir, 'mode_tortuosity',
                         'Per-cycle mean tortuosity (path / chord)',
                         'Per-cycle mean tortuosity',
                         _pr_getter(PR.dropna(subset=['cycle_mean_tortuosity']),
                                    'cycle_mean_tortuosity'), jitter=True)
        _log('per-cycle mean:', P)

    # Asymmetry (signed about electrode midline)
    print('\nAsymmetry (+ = blade-A side):')
    if pf_has('arclen_asym') or pf_has('tort_asym'):
        panels = []
        if pf_has('arclen_asym'):
            panels.append((_pf_getter(PF, 'arclen_asym'), 'Arc-length asymmetry'))
        if pf_has('tort_asym'):
            panels.append((_pf_getter(PF, 'tort_asym'), 'Tortuosity asymmetry'))
        Ps = emit_signed_panels(out_dir, 'mode_asymmetry_local', panels,
                                'Per-frame asymmetry', jitter=False)
        for (g, name), P in zip(panels, Ps):
            _log('local ' + name + ':', P)
    if pr_has('cycle_mean_arclen_asym') or pr_has('cycle_mean_tort_asym'):
        panels = []
        if pr_has('cycle_mean_arclen_asym'):
            panels.append((_pr_getter(PR, 'cycle_mean_arclen_asym'), 'Arc-length asymmetry'))
        if pr_has('cycle_mean_tort_asym'):
            panels.append((_pr_getter(PR, 'cycle_mean_tort_asym'), 'Tortuosity asymmetry'))
        Ps = emit_signed_panels(out_dir, 'mode_asymmetry', panels,
                                'Per-cycle mean asymmetry', jitter=True)
        for (g, name), P in zip(panels, Ps):
            _log('per-cycle ' + name + ':', P)


if __name__ == '__main__':
    main()
