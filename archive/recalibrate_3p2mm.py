"""
recalibrate_3p2mm.py — switch the J-electrode calibration reference gap from the
old 2.4 mm assumption to the CONFIRMED 3.2 mm true minimum electrode gap.

px/mm = min_gap_px / gap_mm, and velocity = slope*FPS/px_per_mm/1000, so velocity
and every mm-length scale by exactly gap_mm_new/gap_mm_old = 3.2/2.4 = 4/3, uniformly
for every recording. Frame-counted quantities (Mode-B fraction, cycle time, frequency,
event counts, pixel lengths, ratios) are unchanged.

Per <stem>_results dir this script:
  * backs up the affected CSVs + j_geometry.json + run_metadata.json to __pre_3p2_backup__/,
  * multiplies the whitelisted velocity (m/s, mm/s) and length (mm) columns by 4/3,
  * sets j_geometry.json min_gap_mm = 3.2 and px_per_mm = min_gap_px/3.2,
  * sets run_metadata.json px_per_mm + provenance (keeps prior value).
Idempotent: a dir whose run_metadata already carries calib_gap_mm == 3.2 is skipped.

    python recalibrate_3p2mm.py --root "<J Electrode>" --dry-run
    python recalibrate_3p2mm.py --root "<J Electrode>" --apply
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import pandas as pd

OLD_GAP_MM = 2.4
NEW_GAP_MM = 3.2
BACKUP = "__pre_3p2_backup__"

# columns that scale linearly with gap_mm (i.e. with 1/px_per_mm): velocities in
# m/s or mm/s, and lengths in mm. Pixel lengths, ratios, frames, modes untouched.
RESCALE_COLUMNS = {
    "event_summary.csv": [
        "vel_ms", "vel_ms_err",
        "local_vel_median", "local_vel_mean", "local_vel_p10", "local_vel_p90",
        "local_vel_peak", "local_vel_max", "local_vel_std",
        "local_speed_median", "local_speed_mean", "local_speed_p90",
        "local_speed_peak", "local_speed_max",
    ],
    "event_summary_with_unc.csv": [
        "vel_ms", "vel_ms_err", "vel_ms_err_stat", "vel_ms_err_sys", "vel_ms_err_total",
        "local_vel_median", "local_vel_mean", "local_vel_p10", "local_vel_p90",
        "local_vel_peak", "local_vel_max", "local_vel_std",
        "local_speed_median", "local_speed_mean", "local_speed_p90",
        "local_speed_peak", "local_speed_max",
    ],
    "measurements_all_events.csv": [
        "arc_length_mm", "arc_width_mm", "arc_tip_vel_mm_s", "electrode_gap_mm",
        "event_vel_ms", "local_vel_ms", "local_vx_ms", "local_vy_ms", "local_speed_ms",
    ],
    "re_strikes_qa.csv": [
        "dist_from_apex_mm", "prev_extension_mm",
    ],
}


def recalibrate_dir(d, dry):
    gp, mp = d / "j_geometry.json", d / "run_metadata.json"
    if not gp.exists() or not mp.exists():
        return f"[skip] {d.name}: no geometry/metadata"
    geom = json.loads(gp.read_text())
    meta = json.loads(mp.read_text())
    if abs(float(meta.get("calib_gap_mm", 0)) - NEW_GAP_MM) < 1e-9:
        return f"[ok]   {d.name}: already at {NEW_GAP_MM} mm"
    gpx = geom.get("min_gap_px")
    if not gpx:
        return f"[skip] {d.name}: no min_gap_px"
    old_px = meta.get("px_per_mm")            # current (2.4 mm) px/mm
    new_px = float(gpx) / NEW_GAP_MM          # 3.2 mm px/mm
    factor = NEW_GAP_MM / OLD_GAP_MM          # = 4/3 on velocity & mm-length

    bak = d / BACKUP
    changed = []
    if not dry:
        bak.mkdir(exist_ok=True)
        for f in (gp, mp):
            if not (bak / f.name).exists():
                shutil.copy2(f, bak / f.name)
    for fname, cols in RESCALE_COLUMNS.items():
        fp = d / fname
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        hit = [c for c in cols if c in df.columns]
        if not hit:
            continue
        if not dry:
            if not (bak / fname).exists():
                shutil.copy2(fp, bak / fname)
            for c in hit:
                df[c] = df[c] * factor
            df.to_csv(fp, index=False)
        changed.append(f"{fname}({len(hit)})")

    if not dry:
        geom["min_gap_mm"] = NEW_GAP_MM
        geom["px_per_mm"] = round(new_px, 4)
        geom["px_per_mm_source"] = f"hand-authoritative 25JUN26 (min_gap_px/{NEW_GAP_MM}mm)"
        gp.write_text(json.dumps(geom, indent=2))
        if old_px:
            meta["px_per_mm_before_3p2"] = old_px
        meta["px_per_mm"] = round(new_px, 5)
        meta["px_per_mm_source"] = f"hand-authoritative 25JUN26 (gap={NEW_GAP_MM}mm confirmed)"
        meta["calib_gap_mm"] = NEW_GAP_MM
        mp.write_text(json.dumps(meta, indent=2))

    verb = "WOULD rescale" if dry else "rescaled"
    return (f"[fix]  {d.name}: px/mm {old_px:.4f} -> {new_px:.4f}  "
            f"x{factor:.4f}  {verb} {', '.join(changed) or '(no csv)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = not args.apply
    root = Path(args.root)
    print(f"{'recording':50s}  transition 2.4mm -> 3.2mm (x{NEW_GAP_MM/OLD_GAP_MM:.4f} on v & mm)\n")
    for d in sorted(root.glob("*_results")):
        if d.is_dir():
            print(recalibrate_dir(d, dry))
    print("\n" + ("DRY RUN — nothing written." if dry
                  else "APPLIED. Backups in each <results>/__pre_3p2_backup__/."))


if __name__ == "__main__":
    main()
