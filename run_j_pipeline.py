"""
run_j_pipeline.py  --  J-electrode CINE analysis, full pipeline
======================================================================
Run from the "Plasma Arc" folder in any VSCode terminal:

    python run_j_pipeline.py               # run everything
    python run_j_pipeline.py --from=event  # skip to event analysis
    python run_j_pipeline.py --from=velmap # skip to velmap only
    python run_j_pipeline.py --force       # re-run everything ignoring checkpoints

Stages (in order):
  1. analyze    -- analyze_cine.py       per-frame arc measurement (chunked)
  2. event      -- event_velocity.py     discharge event segmentation + velocity
  3. velmap     -- velmap_cine.py        2-D velocity maps (full / partial / both)
  4. fourier    -- fourier_analysis.py   frequency-domain analysis
  5. visualize  -- visualize_cine.py     15 summary figures
  6. sanity     -- velocity_sanity_check.py  multi-spacing velocity validation

Notes:
  - nucleation_tracker.py is skipped (J-electrode is not a V-shaped geometry)
  - qa_events.py is skipped (requires TIFF stacks; J electrode uses CINE)
  - All spatial axes are in PIXELS (PX_PER_MM is not set for J electrode)
  - Velocity values from event_velocity.py use BLADE px/mm calibration (9.167)
    until a J-electrode calibration is measured; the spatial pattern is correct.

Checkpoint logic:
  Completed stages are detected by the presence of their output file.
  Ctrl+C is safe -- re-run to resume.

Edit the CONFIG section below if your paths differ.
"""

import os, sys, subprocess, pathlib, datetime

# =============================================================================
# CONFIG  --  edit these if your folder layout is different
# =============================================================================

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()

import platform
if platform.system() == 'Windows':
    J_DIR = pathlib.Path.home() / 'OneDrive' / 'Desktop' / 'J electrode arc'
else:
    J_DIR = pathlib.Path('/sessions/epic-kind-sagan/mnt/BLADE/../J electrode arc')

# Output folder: dated subfolder inside the J electrode directory
_date   = datetime.date.today().isoformat()
OUT_DIR = J_DIR / 'analysis' / _date

# Input CINE file (will auto-discover *.cine in J_DIR if this doesn't exist)
CINE_FILE = J_DIR / 'AD_Cer_1.2mmNozz_1500mL_(CO2).cine'

# Downstream scripts
ANALYZE_SCRIPT  = SCRIPT_DIR / 'analyze_cine.py'
EVENT_SCRIPT    = SCRIPT_DIR / 'event_velocity.py'
VELMAP_SCRIPT   = SCRIPT_DIR / 'velmap_cine.py'
FOURIER_SCRIPT  = SCRIPT_DIR / 'fourier_analysis.py'
VIZ_SCRIPT      = SCRIPT_DIR / 'visualize_cine.py'
SANITY_SCRIPT   = SCRIPT_DIR / 'velocity_sanity_check.py'

# Checkpoint output files
MEASUREMENTS_CSV = OUT_DIR / 'measurements.csv'
EVENTS_CSV       = OUT_DIR / 'measurements_events.csv'
EVENT_SUM_CSV    = OUT_DIR / 'event_summary.csv'

# =============================================================================
# HELPERS
# =============================================================================

PYTHON = sys.executable
_FORCE = False

def already_done(path, min_bytes=100):
    if _FORCE:
        return False
    p = pathlib.Path(path)
    return p.exists() and p.stat().st_size > min_bytes

def run_script(script, *args, label=None):
    cmd = [PYTHON, str(script)] + [str(a) for a in args]
    lbl = label or pathlib.Path(script).name
    print(f"\n>>> {lbl}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] '{lbl}' failed (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)

def run_script_if_needed(checkpoint_path, script, *args, label=None):
    if already_done(checkpoint_path):
        lbl = label or pathlib.Path(script).name
        print(f"    [skip] {lbl}  (output exists: "
              f"{pathlib.Path(checkpoint_path).name})")
        return
    run_script(script, *args, label=label)

# =============================================================================
# MAIN
# =============================================================================

def main():
    from_stage = 'analyze'
    force      = False

    for arg in sys.argv[1:]:
        if arg.startswith('--from='):
            from_stage = arg.split('=', 1)[1].lower()
        elif arg in ('--force', '-f'):
            force = True

    STAGES = ['analyze', 'event', 'velmap', 'fourier', 'visualize', 'sanity']
    if from_stage not in STAGES:
        print(f"[ERROR] Unknown --from= stage '{from_stage}'. "
              f"Choose from: {', '.join(STAGES)}")
        sys.exit(1)

    def skip(stage):
        return STAGES.index(stage) < STAGES.index(from_stage)

    if force:
        global _FORCE
        _FORCE = True
        print("[FORCE] Checkpoint files ignored -- all active stages will re-run.")

    # Auto-discover CINE file if configured path doesn't exist
    cine_path = CINE_FILE
    if not cine_path.exists():
        candidates = sorted(J_DIR.glob('*.cine'))
        if candidates:
            cine_path = candidates[0]
            print(f"  [INFO] Default CINE not found; "
                  f"using auto-discovered: {cine_path.name}")
        else:
            print(f"\n[ERROR] No .cine file found in:\n  {J_DIR}")
            print("Edit the CINE_FILE line in the CONFIG section above.")
            sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nPython     : {PYTHON}")
    print(f"CINE file  : {cine_path}")
    print(f"Output dir : {OUT_DIR}")
    if from_stage != 'analyze':
        print(f"Starting from stage: {from_stage}")

    # =========================================================================
    # STAGE 1 -- Per-frame arc measurement (analyze_cine.py)
    # =========================================================================
    if not skip('analyze'):
        print(f"\n{'='*60}")
        print("  ANALYZE  --  per-frame arc measurement from CINE")
        print(f"{'='*60}")
        if not ANALYZE_SCRIPT.exists():
            print(f"[ERROR] analyze_cine.py not found at:\n  {ANALYZE_SCRIPT}")
            sys.exit(1)
        run_script_if_needed(
            MEASUREMENTS_CSV, ANALYZE_SCRIPT,
            str(cine_path), str(OUT_DIR),
            label="analyze_cine.py")
    else:
        print("\n[skip] Stage 1 -- analyze_cine.py")

    # =========================================================================
    # STAGE 2 -- Discharge event segmentation + velocity (event_velocity.py)
    # =========================================================================
    if not skip('event'):
        print(f"\n{'='*60}")
        print("  EVENT VELOCITY  --  event segmentation + regression velocity")
        print(f"{'='*60}")
        if not MEASUREMENTS_CSV.exists():
            print(f"[ERROR] measurements.csv not found at:\n  {MEASUREMENTS_CSV}")
            print("Run stage 1 first (or without --from=).")
            sys.exit(1)
        run_script_if_needed(
            EVENTS_CSV, EVENT_SCRIPT,
            str(MEASUREMENTS_CSV), str(OUT_DIR),
            label="event_velocity.py")
    else:
        print("\n[skip] Stage 2 -- event_velocity.py")

    # =========================================================================
    # STAGE 3 -- 2-D velocity maps (velmap_cine.py)
    # =========================================================================
    if not skip('velmap'):
        print(f"\n{'='*60}")
        print("  VELOCITY MAPS  --  2-D per-pixel velocity maps (CINE)")
        print(f"{'='*60}")
        stem       = cine_path.stem
        velmap_check = OUT_DIR / f'{stem}_velmap_both.png'
        if not EVENTS_CSV.exists():
            print(f"[ERROR] measurements_events.csv not found at:\n  {EVENTS_CSV}")
            print("Run stage 2 first (or use --from=event).")
            sys.exit(1)
        run_script_if_needed(
            velmap_check, VELMAP_SCRIPT,
            str(EVENTS_CSV), str(OUT_DIR),
            label="velmap_cine.py")
    else:
        print("\n[skip] Stage 3 -- velmap_cine.py")

    # =========================================================================
    # STAGE 4 -- Fourier analysis (fourier_analysis.py)
    # =========================================================================
    if not skip('fourier'):
        print(f"\n{'='*60}")
        print("  FOURIER  --  frequency-domain analysis of arc dynamics")
        print(f"{'='*60}")
        fourier_check = OUT_DIR / 'fourier_detection_allruns.png'
        if not MEASUREMENTS_CSV.exists():
            print(f"[ERROR] measurements.csv not found; run stage 1 first.")
            sys.exit(1)
        run_script_if_needed(
            fourier_check, FOURIER_SCRIPT,
            str(MEASUREMENTS_CSV), str(OUT_DIR),
            label="fourier_analysis.py")
    else:
        print("\n[skip] Stage 4 -- fourier_analysis.py")

    # =========================================================================
    # STAGE 5 -- Visualize (visualize_cine.py)
    # =========================================================================
    if not skip('visualize'):
        print(f"\n{'='*60}")
        print("  VISUALIZE  --  summary figures (kymograph, velocity, etc.)")
        print(f"{'='*60}")
        viz_check = OUT_DIR / 'fig01_kymograph.png'
        if not VIZ_SCRIPT.exists():
            print(f"  [WARN] visualize_cine.py not found at {VIZ_SCRIPT} -- skipping.")
        elif not MEASUREMENTS_CSV.exists():
            print(f"  [WARN] measurements.csv missing -- skipping visualize.")
        else:
            run_script_if_needed(
                viz_check, VIZ_SCRIPT,
                str(MEASUREMENTS_CSV), str(OUT_DIR),
                label="visualize_cine.py")
    else:
        print("\n[skip] Stage 5 -- visualize_cine.py")

    # =========================================================================
    # STAGE 6 -- Velocity sanity check (velocity_sanity_check.py)
    # =========================================================================
    if not skip('sanity'):
        print(f"\n{'='*60}")
        print("  SANITY CHECK  --  multi-spacing velocity validation")
        print(f"{'='*60}")
        sanity_check = OUT_DIR / 'sanity_velocity_distributions.png'
        if not SANITY_SCRIPT.exists():
            print(f"  [WARN] velocity_sanity_check.py not found -- skipping.")
        else:
            run_script_if_needed(
                sanity_check, SANITY_SCRIPT,
                str(MEASUREMENTS_CSV), str(OUT_DIR),
                label="velocity_sanity_check.py")
    else:
        print("\n[skip] Stage 6 -- velocity_sanity_check.py")

    # =========================================================================
    print(f"\n{'='*60}")
    print("  All done!")
    print(f"  Output folder : {OUT_DIR}")
    print(f"  CINE file     : {cine_path.name}")
    print()
    print("  NOTE: velocity axes are in m/s using BLADE px/mm calibration.")
    print("  Spatial maps are in pixels.  Once a J-electrode px/mm value")
    print("  is measured, update PX_PER_MM in analyze_cine.py and")
    print("  event_velocity.py and re-run from --from=event.")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
