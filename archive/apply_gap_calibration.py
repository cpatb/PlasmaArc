"""
apply_gap_calibration.py — retro-correct existing J-electrode results to the
apex-gap calibration (px/mm = min_gap_px / min_gap_mm).

Several recordings were processed with a px/mm that did not match their clicked
apex gap (the only valid calibration for this archival CO2 data). Because px/mm
enters the pipeline ONLY as a final scale on velocities and mm-lengths — detection,
event boundaries, and the calibration-free Mode A/B classification are all
unaffected — the correction is an exact linear rescale, no reprocessing needed:

    factor = px_per_mm_used / px_per_mm_gap        (velocity, mm-length) *= factor

Per recording this script:
  * reads min_gap_px / min_gap_mm from j_geometry.json -> the correct px/mm,
  * reads the px/mm actually used from run_metadata.json,
  * backs up the affected CSVs to __pre_calib_rescale_backup__/,
  * rescales the whitelisted velocity (m/s, mm/s) and length (mm) columns,
  * rewrites px_per_mm in run_metadata.json and j_geometry.json to the gap value.

Idempotent: a recording whose run_metadata already carries the gap-derived px/mm
(within 1e-4) is skipped, so re-running can't double-rescale. Use --dry-run first.

    python apply_gap_calibration.py --root "C:\\Users\\compl\\Desktop\\Plasma Arc\\J Electrode" --dry-run
    python apply_gap_calibration.py --root "C:\\Users\\compl\\Desktop\\Plasma Arc\\J Electrode"
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import pandas as pd

BACKUP_DIRNAME = '__pre_calib_rescale_backup__'

# Columns that scale linearly with 1/px_per_mm (velocity in m/s or mm/s, and
# lengths in mm). Everything else — px lengths, ratios, frames, rows, modes,
# norm_pos — is left untouched.
RESCALE_COLUMNS = {
    'event_summary.csv': [
        'vel_ms', 'vel_ms_err',
        'local_vel_median', 'local_vel_mean', 'local_vel_p10', 'local_vel_p90',
        'local_vel_peak', 'local_vel_max', 'local_vel_std',
        'local_speed_median', 'local_speed_mean', 'local_speed_p90',
        'local_speed_peak', 'local_speed_max',
    ],
    'event_summary_with_unc.csv': [
        'vel_ms', 'vel_ms_err', 'vel_ms_err_stat', 'vel_ms_err_sys', 'vel_ms_err_total',
        'local_vel_median', 'local_vel_mean', 'local_vel_p10', 'local_vel_p90',
        'local_vel_peak', 'local_vel_max', 'local_vel_std',
        'local_speed_median', 'local_speed_mean', 'local_speed_p90',
        'local_speed_peak', 'local_speed_max',
    ],
    'measurements_all_events.csv': [
        'arc_length_mm', 'arc_width_mm', 'arc_tip_vel_mm_s', 'electrode_gap_mm',
        'event_vel_ms', 'local_vel_ms', 'local_vx_ms', 'local_vy_ms', 'local_speed_ms',
    ],
    're_strikes_qa.csv': [
        'dist_from_apex_mm', 'prev_extension_mm',
    ],
}


def gap_px_per_mm(geom):
    gpx = geom.get('min_gap_px')
    gmm = geom.get('min_gap_mm')
    if not gpx or gpx <= 0 or not gmm or gmm <= 0:
        return None
    return float(gpx) / float(gmm)


def rescale_dir(d, dry_run):
    geom_path = d / 'j_geometry.json'
    meta_path = d / 'run_metadata.json'
    if not geom_path.exists() or not meta_path.exists():
        return f'[skip] {d.name}: no geometry/metadata'
    geom = json.loads(geom_path.read_text())
    meta = json.loads(meta_path.read_text())
    new_px = gap_px_per_mm(geom)
    used_px = meta.get('px_per_mm')
    if new_px is None or not used_px:
        return f'[skip] {d.name}: missing px/mm'
    if abs(used_px - new_px) < 1e-4:
        return f'[ok]   {d.name}: already at gap calibration ({new_px:.4f})'

    factor = used_px / new_px
    backup = d / BACKUP_DIRNAME
    changed = []
    for fname, cols in RESCALE_COLUMNS.items():
        fp = d / fname
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        hit = [c for c in cols if c in df.columns]
        if not hit:
            continue
        if not dry_run:
            backup.mkdir(exist_ok=True)
            if not (backup / fname).exists():
                shutil.copy2(fp, backup / fname)
            for c in hit:
                df[c] = df[c] * factor
            df.to_csv(fp, index=False)
        changed.append(f'{fname}({len(hit)})')

    if not dry_run:
        if not (backup / 'run_metadata.json').exists():
            backup.mkdir(exist_ok=True)
            shutil.copy2(meta_path, backup / 'run_metadata.json')
            shutil.copy2(geom_path, backup / 'j_geometry.json')
        meta['px_per_mm'] = round(new_px, 5)
        meta['px_per_mm_source'] = 'apex-gap (min_gap_px/min_gap_mm)'
        meta['px_per_mm_before_rescale'] = used_px
        meta_path.write_text(json.dumps(meta, indent=2))
        geom['px_per_mm'] = round(new_px, 4)
        geom_path.write_text(json.dumps(geom, indent=2))

    verb = 'WOULD rescale' if dry_run else 'rescaled'
    return (f'[fix]  {d.name}: used={used_px:.4f} -> gap={new_px:.4f}  '
            f'x{factor:.4f}  {verb} {", ".join(changed) or "(no csv cols)"}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    root = Path(args.root)
    for d in sorted(root.glob('*_results')):
        if d.is_dir():
            print(rescale_dir(d, args.dry_run))


if __name__ == '__main__':
    main()
