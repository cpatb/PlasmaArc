"""
run_pipeline.py  —  Plasma arc analysis, full pipeline
======================================================================
Run from the "Plasma Arc" folder in any VSCode terminal:

    python run_pipeline.py               # run everything
    python run_pipeline.py --from=event  # skip to event analysis
    python run_pipeline.py --from=qa     # skip to QA only

Stages (in order):
  1. blade_run2.py     — per-frame arc measurement (checkpointed)
  2. event_velocity.py — discharge event segmentation + velocity
  3. replot_velmap.py  — 2-D velocity maps (full, partial, both)
  4. fourier_analysis.py — frequency-domain analysis
  5. velocity_sanity_check.py — independent velocity validation
  6. qa_events.py      — visual QA for top-N velocity events
  7. qa_frames.py      — frame-level binarisation QA

Completed stages are skipped automatically (checkpoint files or
output CSVs already present). Ctrl+C is safe — re-run to resume.

Edit the CONFIG section below if your paths differ.
"""

import os, sys, subprocess, pathlib, datetime

# =============================================================================
# CONFIG  —  edit these if your folder layout is different
# =============================================================================

SCRIPT_DIR   = pathlib.Path(__file__).parent.resolve()
BLADE_SCRIPT = SCRIPT_DIR / "blade_run2.py"
BLADE_DIR    = pathlib.Path.home() / "OneDrive" / "Desktop" / "BLADE"
STATE_DIR    = BLADE_DIR / "pipeline_state"
_date        = datetime.date.today().isoformat()
OUT_DIR      = BLADE_DIR / _date          # dated output folder for this run
VISUALIZE    = SCRIPT_DIR / "visualize_blade.py"
MEASUREMENTS = OUT_DIR / "measurements_all.csv"

# Downstream analysis scripts (all in the same folder as this script)
EVENT_VEL_SCRIPT   = SCRIPT_DIR / "event_velocity.py"
VELMAP_SCRIPT      = SCRIPT_DIR / "replot_velmap.py"
FOURIER_SCRIPT     = SCRIPT_DIR / "fourier_analysis.py"
SANITY_SCRIPT      = SCRIPT_DIR / "velocity_sanity_check.py"
QA_EVENTS_SCRIPT    = SCRIPT_DIR / "qa_events.py"
QA_FRAMES_SCRIPT    = SCRIPT_DIR / "qa_frames.py"
NUCLEATION_SCRIPT   = SCRIPT_DIR / "nucleation_tracker.py"

# Downstream output files used as skip-checkpoints
EVENTS_CSV    = OUT_DIR / "measurements_all_events.csv"
EVENT_SUM_CSV = OUT_DIR / "event_summary.csv"

N_FILES  = 3      # number of TIFF files
N_CHUNKS = 7      # measure / velmap chunks per file (each = 4000 frames)

# =============================================================================
# HELPERS
# =============================================================================

PYTHON = sys.executable   # use whatever Python is running this script

# Pass the output directory to every blade_run2.py subprocess so they all
# write to the same folder regardless of when (or what day) they execute.
os.environ['BLADE_OUT_BASE'] = str(OUT_DIR)

def run(*args):
    """Run one blade_run2.py phase and stream its output live."""
    cmd = [PYTHON, str(BLADE_SCRIPT)] + [str(a) for a in args]
    label = " ".join(str(a) for a in args)
    print(f"\n>>> blade_run2.py {label}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] '{label}' failed (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)

def run_script(script, *args, label=None):
    """Run any analysis script with optional positional arguments."""
    cmd = [PYTHON, str(script)] + [str(a) for a in args]
    lbl = label or script.name
    print(f"\n>>> {lbl}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[ERROR] '{lbl}' failed (exit {result.returncode}). Stopping.")
        sys.exit(result.returncode)

_FORCE = False   # set to True by --force flag; disables all checkpoint skipping

def already_done(path, min_bytes=100):
    """Return True if a checkpoint file exists and looks complete.
    Always returns False when --force is active."""
    if _FORCE:
        return False
    p = pathlib.Path(path)
    return p.exists() and p.stat().st_size > min_bytes

def run_if_needed(checkpoint_path, *args):
    """Run a blade_run2.py phase only if its checkpoint is missing or empty."""
    if already_done(checkpoint_path):
        label = " ".join(str(a) for a in args)
        print(f"    [skip] {label}  (checkpoint exists)")
        return
    run(*args)

def run_script_if_needed(checkpoint_path, script, *args, label=None):
    """Run an analysis script only if its output file is missing or empty."""
    if already_done(checkpoint_path):
        lbl = label or script.name
        print(f"    [skip] {lbl}  (output exists: {pathlib.Path(checkpoint_path).name})")
        return
    run_script(script, *args, label=label)

# =============================================================================
# PIPELINE
# =============================================================================

def process_file(fi):
    """Run all phases for a single TIFF file."""
    name = {0: "Run 1", 1: "Run 2", 2: "Run 3"}[fi]
    print(f"\n{'='*60}")
    print(f"  FILE {fi}  —  {name}")
    print(f"{'='*60}")

    # 1. Calibration (fast, always run — just prints px/mm)
    run("calibrate", fi)

    # 2. Measure: process frames in chunks, save arc measurements to CSV
    print("\n-- Measuring arc geometry --")
    for chunk in range(N_CHUNKS):
        checkpoint = STATE_DIR / f"f{fi}_meas_chunk{chunk}.csv"
        run_if_needed(checkpoint, "measure", fi, chunk)

    # 3. Compute frame-to-frame arc tip velocity
    print("\n-- Computing velocity --")
    run("velocity", fi)

    # 4. Velmap: accumulate per-pixel velocity across all frames
    print("\n-- Building velocity map --")
    for chunk in range(N_CHUNKS):
        # velmap chunks don't produce a CSV, so we use a small .done marker
        checkpoint = STATE_DIR / f"f{fi}_velmap_chunk{chunk}.done"
        if already_done(checkpoint):
            print(f"    [skip] velmap {fi} {chunk}  (checkpoint exists)")
        else:
            run("velmap", fi, chunk)
            checkpoint.touch()   # mark this chunk as done

    # 5. Merge chunks into a single measurements.csv for this file
    print("\n-- Finalizing --")
    run("finalize", fi)


def main():
    # ── parse flags ───────────────────────────────────────────────────────────
    # --from=<stage>  skip all stages before <stage>
    # --force / -f    ignore checkpoint files; re-run every active stage
    #
    # Examples:
    #   python run_pipeline.py --from=event           resume from event stage
    #   python run_pipeline.py --from=event --force   re-run event + all after
    from_stage = 'pipeline'
    force      = False
    for arg in sys.argv[1:]:
        if arg.startswith('--from='):
            from_stage = arg.split('=', 1)[1].lower()
        elif arg in ('--force', '-f'):
            force = True

    STAGES = ['pipeline', 'event', 'velmap', 'fourier', 'nucleation', 'sanity', 'qa']
    if from_stage not in STAGES:
        print(f"[ERROR] Unknown --from= stage '{from_stage}'. "
              f"Choose from: {', '.join(STAGES)}")
        sys.exit(1)

    def skip(stage):
        """Return True if this stage should be skipped based on --from=."""
        return STAGES.index(stage) < STAGES.index(from_stage)

    if force:
        global _FORCE
        _FORCE = True
        print("[FORCE] Checkpoint files ignored — all active stages will re-run.")

    STATE_DIR.mkdir(exist_ok=True)

    print(f"\nPython     : {PYTHON}")
    print(f"Script     : {BLADE_SCRIPT}")
    print(f"State dir  : {STATE_DIR}")
    print(f"Output dir : {OUT_DIR}")
    if from_stage != 'pipeline':
        print(f"Starting from stage: {from_stage}")

    if not BLADE_SCRIPT.exists():
        print(f"\n[ERROR] blade_run2.py not found at:\n  {BLADE_SCRIPT}")
        sys.exit(1)
    if not BLADE_DIR.exists():
        print(f"\n[ERROR] BLADE folder not found at:\n  {BLADE_DIR}")
        print("Edit the BLADE_DIR line in the CONFIG section above.")
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # STAGE 1 — Core per-frame measurement (blade_run2.py)
    # =========================================================================
    if not skip('pipeline'):
        for fi in range(N_FILES):
            process_file(fi)

        print(f"\n{'='*60}")
        print("  COMBINE  —  merging all three measurement files")
        print(f"{'='*60}")
        run("combine")

        print(f"\n{'='*60}")
        print("  VISUALIZE  —  per-run figures from measurements_all.csv")
        print(f"{'='*60}")
        if VISUALIZE.exists() and MEASUREMENTS.exists():
            cmd = [PYTHON, str(VISUALIZE),
                   str(MEASUREMENTS), str(OUT_DIR), str(OUT_DIR)]
            print(">>> visualize_blade.py")
            subprocess.run(cmd)
        else:
            print("[WARN] visualize_blade.py or measurements_all.csv missing, skipping.")
    else:
        print("\n[skip] Stage 1 — blade_run2.py measurement")

    # =========================================================================
    # STAGE 2 — Discharge event segmentation + event-regression velocity
    # =========================================================================
    if not skip('event'):
        print(f"\n{'='*60}")
        print("  EVENT VELOCITY  —  event segmentation + regression velocity")
        print(f"{'='*60}")
        if not MEASUREMENTS.exists():
            print(f"[ERROR] measurements_all.csv not found at:\n  {MEASUREMENTS}")
            print("Run stage 1 first (or without --from=).")
            sys.exit(1)
        run_script_if_needed(
            EVENTS_CSV, EVENT_VEL_SCRIPT,
            str(MEASUREMENTS), str(OUT_DIR),
            label="event_velocity.py")
    else:
        print("\n[skip] Stage 2 — event_velocity.py")

    # =========================================================================
    # STAGE 3 — 2-D velocity maps (full / partial / both)
    # =========================================================================
    if not skip('velmap'):
        print(f"\n{'='*60}")
        print("  VELOCITY MAPS  —  full, partial, and combined arc maps")
        print(f"{'='*60}")
        # Checkpoint on one of the output PNGs
        velmap_check = OUT_DIR / 'ADCB_0.8mmNoz_1000mL_N2_1_velmap_both.png'
        run_script_if_needed(
            velmap_check, VELMAP_SCRIPT,
            str(EVENTS_CSV), str(OUT_DIR),
            label="replot_velmap.py")
    else:
        print("\n[skip] Stage 3 — replot_velmap.py")

    # =========================================================================
    # STAGE 4 — Fourier analysis
    # =========================================================================
    if not skip('fourier'):
        print(f"\n{'='*60}")
        print("  FOURIER  —  frequency-domain analysis of arc dynamics")
        print(f"{'='*60}")
        fourier_check = OUT_DIR / 'fourier_detection_allruns.png'
        run_script_if_needed(
            fourier_check, FOURIER_SCRIPT,
            str(MEASUREMENTS), str(OUT_DIR),
            label="fourier_analysis.py")
    else:
        print("\n[skip] Stage 4 — fourier_analysis.py")

    # =========================================================================
    # STAGE 5 — Nucleation tracker (along-blade propagation analysis)
    # =========================================================================
    if not skip('nucleation'):
        print(f"\n{'='*60}")
        print("  NUCLEATION  —  along-blade attachment propagation")
        print(f"{'='*60}")
        nucleation_check = OUT_DIR / 'nucleation_positions.csv'
        run_script_if_needed(
            nucleation_check, NUCLEATION_SCRIPT,
            str(EVENTS_CSV), str(OUT_DIR),
            label="nucleation_tracker.py")
    else:
        print("\n[skip] Stage 5 — nucleation_tracker.py")

    # =========================================================================
    # STAGE 6 — Independent velocity sanity check
    # =========================================================================
    if not skip('sanity'):
        print(f"\n{'='*60}")
        print("  SANITY CHECK  —  multi-spacing velocity validation")
        print(f"{'='*60}")
        sanity_check = OUT_DIR / 'sanity_velocity_distributions.png'
        run_script_if_needed(
            sanity_check, SANITY_SCRIPT,
            str(MEASUREMENTS), str(OUT_DIR),
            label="velocity_sanity_check.py")
    else:
        print("\n[skip] Stage 5 — velocity_sanity_check.py")

    # =========================================================================
    # STAGE 7 — QA outputs
    # =========================================================================
    if not skip('qa'):
        print(f"\n{'='*60}")
        print("  QA EVENTS  —  annotated frames for top-velocity events")
        print(f"{'='*60}")
        qa_events_dir = OUT_DIR / 'qa_output'
        qa_sum_check  = qa_events_dir / 'qa_summary.csv'
        run_script_if_needed(
            qa_sum_check, QA_EVENTS_SCRIPT,
            str(EVENTS_CSV), str(EVENT_SUM_CSV), str(qa_events_dir),
            label="qa_events.py")

        print(f"\n{'='*60}")
        print("  QA FRAMES  ---  binarisation / tip-detection spot check")
        print(f"{'='*60}")
        qa_frames_dir  = OUT_DIR / 'qa_frames'
        qa_frames_check = qa_frames_dir / 'qa_frames_summary.csv'
        run_script_if_needed(
            qa_frames_check, QA_FRAMES_SCRIPT,
            str(MEASUREMENTS), str(qa_frames_dir),
            label="qa_frames.py")
    else:
        print("\n[skip] Stage 7 --- QA scripts")

    # =========================================================================
    print(f"\n{'='*60}")
    print("  All done!")
    print(f"  Output folder : {OUT_DIR}")
    print(f"  QA events     : {OUT_DIR / 'qa_output'}")
    print(f"  QA frames     : {OUT_DIR / 'qa_frames'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
