#!/bin/bash
# Launch the V-T1 threshold sweep on the 10 pre-registered files in parallel.
cd "C:/Users/compl/Github/Plasma Arc"
OUT="E:/BladeReactor_Upwards_analysis/validation/threshold_sweep"
mkdir -p "$OUT"
pids=()
while IFS= read -r f; do
  f="${f%$'\r'}"
  [ -z "$f" ] && continue
  stem=$(basename "$f" .cine)
  python -X utf8 validation_upwards/threshold_sweep.py "$f" "$OUT" > "$OUT/$stem.log" 2>&1 &
  pids+=($!)
  while [ "$(jobs -rp | wc -l)" -ge 5 ]; do sleep 5; done
done < validation_upwards/sweep_files.txt
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "SWEEP_DONE rc=$rc"
