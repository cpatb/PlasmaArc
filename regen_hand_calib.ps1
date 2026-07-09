# regen_hand_calib.ps1
# Regenerate ALL J-electrode outputs + figures on the hand-authoritative
# calibration (hand_calibration_authoritative_25JUN26.txt).
#
# The calibration has ALREADY been injected into every j_geometry.json +
# run_metadata.json by apply_hand_calibration.py (originals backed up to each
# <results>/__hand_calib_backup__/). This script just regenerates the data.
#
# WHY --from-stage merge (not analyze): px/mm enters the pipeline ONLY as a final
# scale on velocities and mm-lengths; detection, event boundaries and Mode A/B are
# calibration-free (see apply_gap_calibration.py). Restarting from the merge stage
# reuses the existing per-chunk detections and reproduces a full re-detect EXACTLY,
# while skipping the hours-long convert+detect (esp. the 873k-frame 1.2 mm file).

$ErrorActionPreference = "Stop"
$repo = "C:\Users\compl\Github\Plasma Arc"
$root = "C:\Users\compl\Desktop\Plasma Arc\J Electrode"
Set-Location $repo

# --- 1. per-recording: regenerate CSVs + figures from existing detections -------
# px/mm is auto-read from the hand-edited j_geometry.json; pass each recording's
# original bright_thresh (not stored in j_geometry.json).
$rec = [ordered]@{
  "HS_0.4mm_1000mL_(CO2)_trig.cine"                 = 160
  "HS_0.4mm_1500mL_(CO2)_trig.cine"                 = 220
  "HS_0.4mm_1500mL_(CO2)_trig_repeat.cine"          = 235
  "HS_0.8mm_1500mL_(CO2)_trig(2).cine"              = 160
  "HS_0.8mm_1500mL_(CO2)_trig_repeat_weird(2).cine" = 210
  "HS_1.2mm_1500mL_(CO2).cine"                      = 175
}
foreach ($k in $rec.Keys) {
  Write-Host "`n=== merge-regen: $k  (bright_thresh $($rec[$k])) ===" -ForegroundColor Cyan
  python run_j_co2.py --cine $k --from-stage merge --bright-thresh $rec[$k]
}

# --- 2. per-recording uncertainty columns (event_summary_with_unc.csv) ----------
python recompute_uncertainty.py --root $root

# --- 3. cross-recording comparison figures (repeat sentinel still honored) -------
# reynolds_replot.py = clean-seaborn style; also rewrites reynolds_*_.csv.
python "$root\Reynolds_comparison\reynolds_replot.py"
python mode_ratio_vs_re.py  --root $root
python asymmetry_vs_time.py --root $root

Write-Host "`nDONE. Fresh CSVs + figures regenerated on the hand calibration." -ForegroundColor Green
Write-Host "Spot-check: diff one recording's event_summary.csv median vel before/after" -ForegroundColor Yellow
Write-Host "should equal old_median * (old_pxmm/new_pxmm) from apply_hand_calibration's table." -ForegroundColor Yellow
