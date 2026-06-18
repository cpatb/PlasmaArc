#!/usr/bin/env python3
"""
mode_velocity_compare.py  -  Arc velocity / propagation timing by Mode (A/B) and
direction (up/down). Flash frames are excluded (see mode_common.py).

For every metric the full comparison panel is produced:
    {stub}_AB.png         Mode A vs Mode B   (pooled over direction)
    {stub}_dir.png        down vs up         (pooled over mode)
    {stub}_allgroups.png  all four mode x direction groups overlaid
    {stub}_boxplot.png    all four groups as a box plot

Metrics
-------
Local velocity   local  PF['local_vel_ms']      stub mode_vel_local
                        per detected frame, OLS sliding window
                        (allgroups -> mode_vel_local_allgroups.png,
                         box       -> mode_vel_local_boxplot.png)
Advance velocity event  PR['advance_vel_ms']    stub mode_vel_advance
                        per-cycle linear-regression slope of arc-tip position
                        from nucleation (re-strike) to the first extension peak
                        (AB  -> mode_vel_advance_AB.png,
                         dir -> mode_vel_advance_dir.png,
                         box -> mode_vel_boxplot.png)
Duration         event  PR['duration_frames']   stub mode_dur
                        frames from nucleation to extension peak, in ms at --fps

Local velocity and advance velocity are the per-frame and per-cycle views of the
same propagation. Local-velocity box plots omit the raw-point jitter (dense).

Usage (PowerShell):
    python mode_velocity_compare.py ^
        --down PATH/downward_results --up PATH/upward_results
        --px-per-mm 8.8 --out-dir OUTPUT_DIR
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from mode_common import analyze_recording
import arc_filter  # single source of truth for flash / velocity-cap criteria
from mode_plots import emit_pos_set, ps


def resolve(p):
    p = Path(p)
    return p if p.suffix == '.csv' else p / 'measurements_all_events.csv'


def _pf_getter(PF, col):
    def g(dr, m):
        return pd.to_numeric(
            PF[(PF['direction'] == dr) & (PF['cycle_mode'] == m)][col], errors='coerce')
    return g


def _pr_getter(df, col, scale=1.0):
    def g(dr, m):
        s = pd.to_numeric(
            df[(df['direction'] == dr) & (df['mode'] == m)][col], errors='coerce')
        return s * scale
    return g


def _log(tag, P):
    print('  %-22s A vs B p=%s   down vs up p=%s   A up/dn p=%s   B up/dn p=%s'
          % (tag, ps(P['pAB']), ps(P['pdir']), ps(P['pAud']), ps(P['pBud'])))


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

    def pr_has(col):
        return col in PR.columns and PR[col].notna().any()

    # Local arc-tip velocity (per detected frame)
    print('\nLocal velocity (m/s, per frame):')
    P = emit_pos_set(out_dir, 'mode_vel_local',
                     'Per-frame local arc-tip velocity (m/s)', 'Local arc-tip velocity',
                     _pf_getter(PF, 'local_vel_ms'), jitter=False)
    _log('per-frame (local):', P)

    # Advance velocity (per re-strike cycle; regression nucleation -> peak)
    print('\nAdvance velocity (m/s, per cycle):')
    av = PR.dropna(subset=['advance_vel_ms'])
    P = emit_pos_set(out_dir, 'mode_vel_advance',
                     'Advance velocity (m/s)', 'Advance velocity (regression)',
                     _pr_getter(av, 'advance_vel_ms'),
                     box_name='mode_vel_boxplot.png', jitter=True)
    _log('per-cycle advance:', P)

    # Propagation duration (per cycle; frames -> ms)
    if pr_has('duration_frames'):
        print('\nPropagation duration (ms, per cycle):')
        dv = PR.dropna(subset=['duration_frames'])
        P = emit_pos_set(out_dir, 'mode_dur',
                         'Propagation duration (ms)', 'Propagation duration',
                         _pr_getter(dv, 'duration_frames', scale=1000.0 / args.fps),
                         jitter=True)
        _log('per-cycle duration:', P)
    else:
        print('[WARN] duration_frames not available -- skipping duration figures.')


if __name__ == '__main__':
    main()
