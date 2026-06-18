<#
  rerun_localvel_N10.ps1
  ----------------------------------------------------------------------------
  Re-run ONLY the velocity-dependent stages with the local-velocity window set
  to N = 10 (was N = 5).  FAST: it reuses the existing TIFF chunks and per-frame
  measurements.csv from last night, so the multi-hour convert + analyze stages
  are SKIPPED.  Expect ~30-60 min (dominated by the 2-D velocity-map TIFF
  re-read).

  N affects only the local sliding-window velocity (local_vel_ms), so this
  re-runs from the events stage onward:
     events (event_velocity.py, N=10) -> re-strike -> fourier ->
     2-D velocity map (build_velocity_map.py) -> figures (visualize_blade.py)
  then refreshes the mode x direction comparison figures.

  NOT changed by N (left as-is): the per-frame measurements (arc length, width,
  tortuosity, gap), the whole-event advancing-front velocity vel_ms, and the
  re-strike Mode A/B classification (those read arc_tip_row, not local_vel_ms).

  Nothing is deleted.  Outputs are overwritten in place.

  LAUNCH:
    powershell -ExecutionPolicy Bypass -File "C:\Users\compl\Github\Plasma Arc\rerun_localvel_N10.ps1"
#>

$ErrorActionPreference = 'Continue'

$repo    = 'C:\Users\compl\Github\Plasma Arc'
$blade   = 'C:\Users\compl\OneDrive\Desktop\BLADE\BLADE 11MAY26'
$down    = Join-Path $blade '0.6mm_1.5SLPM-N2_25deg_OA_downward_results'
$up      = Join-Path $blade '0.6mm_1.5SLPM-N2_25deg_OA_upward_results'
$modecmp = Join-Path $blade 'mode_compare'

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log   = Join-Path $blade ("rerun_localvel_N10_$stamp.log")

Set-Location $repo
Start-Transcript -Path $log -Append | Out-Null
Write-Host "================================================================"
Write-Host " BLADE 11MAY26  LOCAL-VELOCITY RERUN  N=10   $stamp"
Write-Host "================================================================"

# --- sanity: the inputs this rerun reuses must exist -----------------------
$needed = @(
    (Join-Path $down 'measurements.csv'),
    (Join-Path $up   'measurements.csv'),
    (Join-Path $down 'tiff'),
    (Join-Path $up   'tiff')
)
foreach ($n in $needed) {
    if (-not (Test-Path $n)) {
        Write-Host "[FATAL] Missing required input from the full run: $n"
        Write-Host "        Run run_full_overnight.ps1 first (it produces these)."
        Stop-Transcript | Out-Null
        exit 1
    }
}
Write-Host "Found existing TIFFs + measurements.csv for both recordings - reusing them."

# --- 0) provenance: commit the N-parameter change --------------------------
Write-Host ""
Write-Host "[0/3] Committing code to git (provenance)..."
if (-not (git -C $repo config user.email)) {
    git -C $repo config user.email 'plasma-arc@local'
    git -C $repo config user.name  'Plasma Arc Pipeline'
}
git -C $repo add -A
git -C $repo commit -m "Local-velocity window N=10 (was 5): --local-vel-n param, run_11may26 default 10, methods + sensitivity-figure labels [$stamp]" 2>&1 | Write-Host
Write-Host ("HEAD = " + (git -C $repo rev-parse --short HEAD))

# --- 0.5) syntax-check the modules this rerun touches ----------------------
Write-Host ""
Write-Host "[0.5] Syntax-checking modules..."
$mods = @(
    'run_11may26.py','event_velocity.py','qa_restrike.py','arc_filter.py',
    'mode_common.py','build_velocity_map.py','fourier_analysis.py',
    'visualize_blade.py','mode_velocity_compare.py','mode_plots.py','mode_arclength_compare.py',
    'plot_local_vel_by_direction.py'
)
python -m py_compile @mods
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FATAL] A module failed to compile - run not started. Fix and re-launch."
    Stop-Transcript | Out-Null
    exit 1
}
Write-Host "  all modules compile OK"

# --- 1) re-run from the events stage with N=10 (reuses TIFFs + CSV) ---------
Write-Host ""
Write-Host "[1/3] Re-running events -> restrike -> fourier -> velmap -> figures with N=10..."
python run_11may26.py --file both --from-stage events --local-vel-n 10
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] run_11may26.py exited $LASTEXITCODE - stopping before comparisons."
    Stop-Transcript | Out-Null
    exit 1
}

# --- 2) refresh mode x direction comparisons (cycle local-vel uses N) -------
Write-Host ""
Write-Host "[2/3] Refreshing mode x direction comparison figures..."
python mode_velocity_compare.py       --down "$down" --up "$up" --px-per-mm 8.8 --out-dir "$modecmp"
python mode_arclength_compare.py      --down "$down" --up "$up" --px-per-mm 8.8 --out-dir "$modecmp"
python plot_local_vel_by_direction.py --down "$down" --up "$up" --out-dir "$modecmp" --px-per-mm 8.8 --fps 200000
python plot_local_vel_by_direction.py --down "$down" --up "$up" --out-dir "$modecmp" --px-per-mm 8.8 --fps 200000 --metric tortuosity

# --- 3) N=10-vs-N=5 sensitivity figure (stored is now N=10) ----------------
Write-Host ""
Write-Host "[3/3] Writing N=10 (stored) vs N=5 sensitivity figure..."
python plot_local_vel_by_direction.py --down "$down" --up "$up" --out-dir "$modecmp" --px-per-mm 8.8 --fps 200000 --stored-n 10 --n-compare 5

Write-Host ""
Write-Host "================================================================"
Write-Host (" DONE   {0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
Write-Host " local_vel_ms and the 2-D velocity map are now N=10."
Write-Host " Sensitivity: $modecmp\local_vel_N10_vs_N5.png"
Write-Host " Log: $log"
Write-Host "================================================================"
Stop-Transcript | Out-Null
