<#
  run_full_overnight.ps1
  ----------------------------------------------------------------------------
  BLADE 11MAY26 - FULL-RECORDING, FROM-SCRATCH regeneration.

  Runs BOTH complete recordings (downward 291,353 frames + upward 459,446
  frames - NOT a smoke test) end to end with the audited code, then rebuilds
  the cross-recording comparison and shape-overlay figures.

  What it does, in order:
    0.   Commits the current code to git (provenance for this run).
    0.5  Syntax-checks every module; aborts before deleting if anything fails.
    1.   DELETES stale derived outputs so nothing old is reused
         (the two *_results dirs, mode_compare, shape_overlays).
         The raw *.cine source files are NEVER touched.
    2.   Runs run_11may26.py --file both   (convert -> analyze -> merge ->
         events+restrike -> fourier -> velmap+visualize) on the WHOLE files.
    3.   Mode x direction comparison figures (needs both recordings done).
    4.   Shape-metric overlays (per recording).

  Everything is logged to run_full_overnight_<timestamp>.log in the BLADE dir.

  LAUNCH (from any PowerShell window):
    powershell -ExecutionPolicy Bypass -File "C:\Users\compl\Github\Plasma Arc\run_full_overnight.ps1"

  Expect several hours. Needs ~60 GB free for the full-resolution TIFFs.
#>

$ErrorActionPreference = 'Continue'

$repo    = 'C:\Users\compl\Github\Plasma Arc'
$blade   = 'C:\Users\compl\OneDrive\Desktop\BLADE\BLADE 11MAY26'
$cineDn  = Join-Path $blade '0.6mm_1.5SLPM-N2_25deg_OA_downward.cine'
$cineUp  = Join-Path $blade '0.6mm_1.5SLPM-N2_25deg_OA_upward.cine'
$down    = Join-Path $blade '0.6mm_1.5SLPM-N2_25deg_OA_downward_results'
$up      = Join-Path $blade '0.6mm_1.5SLPM-N2_25deg_OA_upward_results'
$modecmp = Join-Path $blade 'mode_compare'
$overlay = Join-Path $blade 'shape_overlays'

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$log   = Join-Path $blade ("run_full_overnight_$stamp.log")

Set-Location $repo
Start-Transcript -Path $log -Append | Out-Null
Write-Host "================================================================"
Write-Host " BLADE 11MAY26  FULL-RECORDING REGEN   $stamp"
Write-Host "================================================================"

# --- sanity: the raw sources must exist before we delete anything ----------
if ((-not (Test-Path $cineDn)) -or (-not (Test-Path $cineUp))) {
    Write-Host "[FATAL] One or both .cine source files are missing - aborting."
    Write-Host "        down: $cineDn"
    Write-Host "        up  : $cineUp"
    Stop-Transcript | Out-Null
    exit 1
}

# --- free space check ------------------------------------------------------
$drive = (Get-Item $blade).PSDrive.Name
$free  = [math]::Round((Get-PSDrive $drive).Free / 1GB, 1)
Write-Host ("Free space on {0}: {1} GB  (need ~60 GB for full-res TIFFs)" -f $drive, $free)
if ($free -lt 60) { Write-Host "[WARN] Less than 60 GB free - the run may fill the disk." }

# --- 0) provenance: commit the exact code being run ------------------------
Write-Host ""
Write-Host "[0/4] Committing code to git (provenance)..."
if (-not (git -C $repo config user.email)) {
    git -C $repo config user.email 'plasma-arc@local'
    git -C $repo config user.name  'Plasma Arc Pipeline'
}
git -C $repo add -A
git -C $repo commit -m "Audit fixes before full run: blade-bisector midline (no tip fallback), detect_blade_lines sign fix, visualize_blade repair + up/down velmap orientation, min_arc_px doc [$stamp]" 2>&1 | Write-Host
$head = (git -C $repo rev-parse --short HEAD)
Write-Host "HEAD = $head"

# --- 0.5) syntax-check every module BEFORE we delete anything --------------
# Guarantees we never wipe data or start a multi-hour run on broken code.
Write-Host ""
Write-Host "[0.5] Syntax-checking pipeline modules..."
$mods = @(
    'run_11may26.py','analyze_blade.py','detect_blade_lines.py','event_velocity.py',
    'qa_restrike.py','arc_filter.py','mode_common.py','visualize_blade.py',
    'build_velocity_map.py','fourier_analysis.py','mode_velocity_compare.py',
    'mode_arclength_compare.py','plot_local_vel_by_direction.py','overlay_shape_metrics.py'
)
python -m py_compile @mods
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FATAL] A module failed to compile - nothing deleted, run not started."
    Write-Host "        Fix the syntax error above and re-launch."
    Stop-Transcript | Out-Null
    exit 1
}
Write-Host "  all modules compile OK"

# --- 1) wipe stale derived outputs (raw .cine untouched) -------------------
Write-Host ""
Write-Host "[1/4] Deleting stale derived outputs in 20 s (Ctrl-C now to abort)..."
Write-Host "      removing: *_results, mode_compare, shape_overlays"
Write-Host "      keeping : the .cine files, Calibrate/, ASSESSMENT/METHODS docs"
Start-Sleep -Seconds 20
foreach ($d in @($down, $up, $modecmp, $overlay)) {
    if (Test-Path $d) { Write-Host "  removing $d"; Remove-Item -Recurse -Force $d }
}

# --- 2) full pipeline, BOTH recordings, WHOLE files ------------------------
# No --max-frames => every frame of each recording is processed.
Write-Host ""
Write-Host "[2/4] Full pipeline on the WHOLE recordings (this is the long part)..."
python run_11may26.py --file both
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] run_11may26.py exited $LASTEXITCODE - stopping before comparisons."
    Stop-Transcript | Out-Null
    exit 1
}

# --- 3) cross-recording comparison figures ---------------------------------
Write-Host ""
Write-Host "[3/4] Mode x direction comparison figures..."
python mode_velocity_compare.py       --down "$down" --up "$up" --px-per-mm 8.8 --out-dir "$modecmp"
python mode_arclength_compare.py      --down "$down" --up "$up" --px-per-mm 8.8 --out-dir "$modecmp"
python plot_local_vel_by_direction.py --down "$down" --up "$up" --out-dir "$modecmp" --px-per-mm 8.8 --fps 200000 --n-compare 10
python plot_local_vel_by_direction.py --down "$down" --up "$up" --out-dir "$modecmp" --px-per-mm 8.8 --fps 200000 --metric tortuosity

# --- 4) shape-metric overlays (per recording, separate subfolders) ---------
Write-Host ""
Write-Host "[4/4] Shape-metric overlays..."
python overlay_shape_metrics.py --results "$down" --out-dir (Join-Path $overlay 'down') --direction down --gap-row 26  --px-per-mm 8.8
python overlay_shape_metrics.py --results "$up"   --out-dir (Join-Path $overlay 'up')   --direction up   --gap-row 241 --px-per-mm 8.8

Write-Host ""
Write-Host "================================================================"
Write-Host (" ALL DONE   {0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
Write-Host " Results : $down"
Write-Host "           $up"
Write-Host " Figures : $modecmp , $overlay"
Write-Host " Log     : $log"
Write-Host "================================================================"
Stop-Transcript | Out-Null
