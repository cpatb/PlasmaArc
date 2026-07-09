#!/usr/bin/env python3
"""
mode_common.py  -  shared helpers for mode (A/B) x direction (up/down) comparisons.

Used by mode_velocity_compare.py and mode_arclength_compare.py so the flash
filter, re-strike detection and cycle tagging are defined in ONE place.

Flash filter (width-based): is_flash = (eff_width > flash_eff_width) OR
                                       (arc_width_px > flash_arc_width)

  eff_width    = area_px / arc_length_px  — coarse, also flags broken skeletons
                 where the arc saturates into a blob (skeleton collapses, ratio
                 explodes). Default cut: 20 px.
  arc_width_px = FWHM perpendicular to skeleton, averaged across samples
                 (fwhm_perpendicular in analyze_blade.py). Default cut: 16 px.
                 Catches genuinely thick arcs that didn't blob enough to break
                 the skeleton.

The two criteria target different failure modes; OR catches both. Justified by
the BLADE width-distribution analysis (~4-5% of detected frames are flagged).

analyze_recording() returns:
  per_restrike : one row per re-strike -- mode, advance_vel_ms,
                 duration_frames, cycle_mean_local_vel, cycle_max_arclen_mm,
                 and (when the CSV was produced by the updated analyze_blade.py)
                 cycle_mean_tortuosity, cycle_max_tortuosity,
                 cycle_mean_total_curv, cycle_mean_arclen_asym,
                 cycle_mean_tort_asym.  These shape columns are NaN for older
                 CSVs that predate the tortuosity/asymmetry metrics.
  per_frame    : one row per detected non-flash frame, tagged with cycle mode
                 (plus tortuosity/arclen_asym/tort_asym when available)
"""
import numpy as np
import pandas as pd

from qa_restrike import apex_and_range, detect_restrikes, classify
# Flash / velocity-cap criteria come from the single source of truth (arc_filter).
from arc_filter import (add_flash_flag,
                        FLASH_EFF_WIDTH as FLASH_EFF_WIDTH_DEFAULT,
                        FLASH_ARC_WIDTH as FLASH_ARC_WIDTH_DEFAULT)

FPS_DEFAULT = 200_000


def analyze_recording(csv_path, direction, px_per_mm=8.8, fps=FPS_DEFAULT,
                      method='abs', threshold=0.10, rel_threshold=0.70,
                      ext_prom_frac=0.10, min_cycle=5, smooth=3,
                      flash_eff_width=FLASH_EFF_WIDTH_DEFAULT,
                      flash_arc_width=FLASH_ARC_WIDTH_DEFAULT):
    df = pd.read_csv(csv_path)
    df = add_flash_flag(df, flash_eff_width, flash_arc_width)
    for c in ['arc_tip_row', 'arc_complete', 'local_vel_ms', 'arc_length_mm',
              'tortuosity', 'total_curvature_rad', 'arclen_asym', 'tort_asym']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    n_det = int((df['arc_detected'] == 1).sum())
    n_flash = int(((df['arc_detected'] == 1) & df['is_flash']).sum())
    det = df[(df['arc_detected'] == 1) & (~df['is_flash'])].dropna(
        subset=['arc_tip_row']).sort_values(['file', 'frame'])

    per_restrike, per_frame = [], []
    for fname, g in det.groupby('file'):
        g = g.reset_index(drop=True)
        frames = g['frame'].values.astype(float)
        tip = g['arc_tip_row'].values.astype(float)
        comp = (g['arc_complete'].values == 1) if 'arc_complete' in g.columns else None
        erow, leff, _src = apex_and_range(tip, comp)
        rk, peaks, _ = detect_restrikes(tip, erow, leff, ext_prom_frac, min_cycle, smooth)
        peaks = np.asarray(peaks)
        ks = [r['idx'] for r in rk]
        for i, r in enumerate(rk):
            k = r['idx']
            k_next = ks[i + 1] if i + 1 < len(ks) else len(g) - 1
            nxt = peaks[peaks > k]
            vel = np.nan
            dur = np.nan
            if len(nxt):
                pk = int(nxt[0])
                if pk - k >= 3:
                    slope = np.polyfit(frames[k:pk + 1], tip[k:pk + 1], 1)[0]
                    v = slope * fps / px_per_mm / 1000.0
                    vel = v if v > 0 else np.nan
                dur = float(pk - k)
            npos = float(np.clip((tip[k] - erow) / leff, 0, 1))
            denom = max(r['prev_ext'] - erow, 1e-6)
            rfrac = float(np.clip((r['prev_ext'] - tip[k]) / denom, 0, 1))
            mode = classify(npos, rfrac, method, threshold, rel_threshold)
            cyc = g.iloc[k:k_next + 1]
            lv = (pd.to_numeric(cyc['local_vel_ms'], errors='coerce')
                  if 'local_vel_ms' in cyc.columns else pd.Series([], dtype=float))
            lv = lv[lv > 0]
            ln = (pd.to_numeric(cyc['arc_length_mm'], errors='coerce')
                  if 'arc_length_mm' in cyc.columns else pd.Series([], dtype=float))

            # Shape metrics over the cycle (present only on re-analysed CSVs).
            def _cyccol(name):
                return (pd.to_numeric(cyc[name], errors='coerce')
                        if name in cyc.columns else pd.Series([], dtype=float))
            tort  = _cyccol('tortuosity'); tort = tort[tort >= 1.0]   # well-formed only
            tcurv = _cyccol('total_curvature_rad').dropna()
            la    = _cyccol('arclen_asym').dropna()
            ta_   = _cyccol('tort_asym').dropna()

            per_restrike.append({
                'direction': direction,
                'mode': mode,
                'frame': int(frames[k]),
                'norm_pos': round(npos, 4),
                'advance_vel_ms': round(vel, 4) if not np.isnan(vel) else np.nan,
                'duration_frames': dur,
                'cycle_mean_local_vel': round(float(lv.mean()), 4) if len(lv) else np.nan,
                'cycle_max_arclen_mm': round(float(ln.max()), 4) if ln.notna().any() else np.nan,
                'cycle_mean_tortuosity': round(float(tort.mean()), 4) if len(tort) else np.nan,
                'cycle_max_tortuosity':  round(float(tort.max()), 4) if len(tort) else np.nan,
                'cycle_mean_total_curv': round(float(tcurv.mean()), 4) if len(tcurv) else np.nan,
                'cycle_mean_arclen_asym': round(float(la.mean()), 4) if len(la) else np.nan,
                'cycle_mean_tort_asym':   round(float(ta_.mean()), 4) if len(ta_) else np.nan,
            })
            sub = cyc[['frame']].copy()
            sub['direction'] = direction
            sub['cycle_mode'] = mode
            sub['local_vel_ms'] = pd.to_numeric(cyc.get('local_vel_ms'), errors='coerce').values
            sub['arc_length_mm'] = pd.to_numeric(cyc.get('arc_length_mm'), errors='coerce').values
            for nm in ['tortuosity', 'arclen_asym', 'tort_asym']:
                if nm in cyc.columns:
                    sub[nm] = pd.to_numeric(cyc[nm], errors='coerce').values
            per_frame.append(sub)

    pr = pd.DataFrame(per_restrike)
    pf = (pd.concat(per_frame, ignore_index=True) if per_frame
          else pd.DataFrame(columns=['frame', 'direction', 'cycle_mode',
                                     'local_vel_ms', 'arc_length_mm']))
    print('  %s: detected=%d  flashes removed=%d (%.1f%%)  re-strikes=%d'
          % (direction, n_det, n_flash, 100.0 * n_flash / max(n_det, 1), len(pr)))
    return pr, pf
