#!/bin/bash
# ============================================================
# run_pipeline.sh  --  Full plasma arc analysis pipeline
#
# Run this from Git Bash inside VSCode:
#   1. Open VSCode in the "Plasma Arc" folder
#   2. Terminal → New Terminal → switch to Git Bash
#   3. bash run_pipeline.sh
#
# Checkpoints are saved to ./pipeline_state/ so you can
# safely Ctrl+C and re-run — completed chunks are skipped.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN2="$SCRIPT_DIR/blade_run2.py"
STATE="$SCRIPT_DIR/pipeline_state"

# Prefer 'python' on Windows (Anaconda/standard install),
# fall back to 'python3' (Linux / some Windows setups)
if command -v python &> /dev/null; then
    PY="python"
else
    PY="python3"
fi

echo "Using Python: $($PY --version 2>&1)"
echo "Script     : $RUN2"
echo "State dir  : $STATE"
mkdir -p "$STATE"

run() {
    echo ""
    echo ">>> $@"
    "$PY" "$RUN2" "$@"
}

skip_if_done() {
    local cfile="$1"; shift
    if [ -f "$cfile" ] && [ "$(wc -c < "$cfile")" -gt 100 ]; then
        echo ">>> Skipping (checkpoint exists): $(basename "$cfile")"
        return 0   # skip
    fi
    run "$@"
}

# ---- File 0 (Run 1) ----------------------------------------
echo ""; echo "======================================================="
echo "  FILE 0 — ADCB_0.8mmNoz_1000mL_N2_1"
echo "======================================================="
run calibrate 0
for chunk in 0 1 2 3 4 5 6; do
    skip_if_done "$STATE/f0_meas_chunk${chunk}.csv" measure 0 $chunk
done
run velocity 0
for chunk in 0 1 2 3 4 5 6; do
    skip_if_done "$STATE/f0_velmap_chunk${chunk}.done" velmap 0 $chunk \
    && touch "$STATE/f0_velmap_chunk${chunk}.done" 2>/dev/null || true
done
run finalize 0

# ---- File 1 (Run 2) ----------------------------------------
echo ""; echo "======================================================="
echo "  FILE 1 — ADCB_0.8mmNoz_1000mL_N2_2"
echo "======================================================="
run calibrate 1
for chunk in 0 1 2 3 4 5 6; do
    skip_if_done "$STATE/f1_meas_chunk${chunk}.csv" measure 1 $chunk
done
run velocity 1
for chunk in 0 1 2 3 4 5 6; do
    skip_if_done "$STATE/f1_velmap_chunk${chunk}.done" velmap 1 $chunk \
    && touch "$STATE/f1_velmap_chunk${chunk}.done" 2>/dev/null || true
done
run finalize 1

# ---- File 2 (Run 3) ----------------------------------------
echo ""; echo "======================================================="
echo "  FILE 2 — ADCB_0.8mmNoz_1000mL_N2_3"
echo "======================================================="
run calibrate 2
for chunk in 0 1 2 3 4 5 6; do
    skip_if_done "$STATE/f2_meas_chunk${chunk}.csv" measure 2 $chunk
done
run velocity 2
for chunk in 0 1 2 3 4 5 6; do
    skip_if_done "$STATE/f2_velmap_chunk${chunk}.done" velmap 2 $chunk \
    && touch "$STATE/f2_velmap_chunk${chunk}.done" 2>/dev/null || true
done
run finalize 2

# ---- Combine + Visualize -----------------------------------
echo ""; echo "======================================================="
echo "  COMBINE + VISUALIZE"
echo "======================================================="
run combine

BLADE_DIR="$USERPROFILE/OneDrive/Desktop/BLADE"
# On Linux or if USERPROFILE unset, fall back to ~/OneDrive/Desktop/BLADE
[ -z "$BLADE_DIR" ] && BLADE_DIR="$HOME/OneDrive/Desktop/BLADE"

"$PY" "$SCRIPT_DIR/visualize_blade.py" \
    "$SCRIPT_DIR/measurements_all.csv" \
    "$BLADE_DIR"

echo ""
echo "======================================================="
echo "  Pipeline complete!"
echo "  Figures saved to: $BLADE_DIR"
echo "======================================================="
