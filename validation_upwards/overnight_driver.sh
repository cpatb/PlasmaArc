#!/bin/bash
# One driver for the overnight run (relaunched after a harness low-memory reap, 2026-09-29 ~03:10):
# main batch (3 files x 5 windows) and the trial-2/3 reproducibility files (1 x 4) concurrently.
cd "C:/Users/compl/Github/Plasma Arc"
OUT="E:/BladeReactor_Upwards_analysis"
python -X utf8 run_blade_upwards.py --list-file validation_upwards/repro_files.txt --files-parallel 1 --workers 4 \
  >> "$OUT/batch_repro.log" 2>&1 &
p1=$!
python -X utf8 run_blade_upwards.py --all --files-parallel 3 --workers 5 >> "$OUT/batch_main6.log" 2>&1 &
p2=$!
wait $p1; echo "REPRO rc=$?" >> "$OUT/batch_main6.log"
wait $p2; echo "MAIN rc=$?" >> "$OUT/batch_main6.log"
echo "DRIVER_DONE" >> "$OUT/batch_main6.log"
