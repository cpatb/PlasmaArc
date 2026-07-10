# Design: generalized electrode-geometry engine (Option A)

Date: 2026-07-09 · Branch: `cleanup/structural-pass` · Baseline: `c3efc88`

## Goal
Remove the copy-pasted geometry logic spread across the five electrode-geometry
producers by centralizing the genuinely-duplicated **pure computational helpers**
into one module `electrode_geometry.py`. **No change to any numerical output or
on-disk artifact.** The measurement consumer (`analyze_blade.measure_frame`) is
NOT touched in this pass (that is the deferred Option B).

## Scope (in)
Producers that dedup onto the new module:
- `calibrate_electrodes.py` (BLADE interactive infinite-line)
- `detect_blade_lines.py` (BLADE auto TLS fit)
- `calibrate_j_electrodes.py` (J finite-segment)
- `detect_j_hooks.py` (J edge-cloud, auto)
- `segment_electrodes.py` (J edge-cloud, hand-drawn — current default)

## Scope (out — explicitly not done here)
- `analyze_blade.measure_frame` branch cascade (Option B, deferred).
- On-disk `j_geometry.json` / `blade_lines.npy` / `electrode_edges.npz` **schemas,
  field names, values, and rounding** — all unchanged. `run_j_co2.resolve_px_per_mm`
  and its recompute-from-`min_gap_px` path are untouched.

## The shared module `electrode_geometry.py` — public API
Pure functions, each EXTRACTED VERBATIM from the current inline code (no algorithm
change), so behaviour is identical by construction:

- `find_first_arc_frame(tif, dark_guard=35, n_search=1000) -> int`
  (from `calibrate_electrodes` / `calibrate_j_electrodes`, byte-identical copies)
- `skeleton_endpoints(mask) -> (N,2) int array`
  (the `k=ones(3,3); k[1,1]=0` convolution kernel; 3 near-identical copies)
- `gap_row_from_tips(tip_a_row, tip_b_row, arc_direction) -> int`
  down → `floor(mean) - 2`; up → `ceil(mean) + 2`. Identical formula in 3 files.
- `nearest_facing_tips(A, B, subsample=None, seed=0) -> (tip_a, tip_b, gap_px)`
  `subsample=None` → full O(N²) argmin (matches `segment_electrodes.paths_to_edges`);
  `subsample=500, seed=0` → seeded `default_rng(0)` subsample then argmin (matches
  `detect_j_hooks.detect_edges` EXACTLY, preserving determinism).
- `load_guide_stack(results_dir, n_max, across=False) -> (F,H,W) array`
  `across=False` → first n (detect_j_hooks); `across=True` → even across-run sampling
  (segment_electrodes). Preserves each producer's sampling policy.

Interactive matplotlib picker is left in place for this pass (UI, low duplication
value, higher review cost); may be folded later.

## Invariance checkpoints (must hold — from the geometry map §5)
Only producer-side risks apply (consumer untouched):
1. **px/mm rounding**: each producer keeps computing & rounding `min_gap_px` exactly
   as today (calibrate_j: 3 dp; segment: 2 dp). The shared writer, if any, does NOT
   re-round. `resolve_px_per_mm` recompute path unchanged.
2. **gap_row**: `floor/ceil(mean) ± 2` reproduced exactly (the shared helper is the
   verbatim formula).
3. **nearest-pair determinism**: `detect_j_hooks` keeps its `default_rng(0)` 500-pt
   subsample via the `subsample=500, seed=0` args; `segment_electrodes` keeps full.
4. **skeleton kernel**: identical convolution kernel & endpoint rule.
5. **blade-line endpoint order** (`pt0 = tip-end`) and BLADE `finite=False` unchanged
   (not touched — producers still write their own arrays).

## Validation harness
1. `python validate.py` — compile + core-lib import (must stay PASS).
2. Extend `test_j_geometry.py` with unit tests pinning each extracted helper against
   its pre-refactor result on synthetic inputs (golden values captured BEFORE editing).
3. **Golden-artifact test** for `calibrate_j_electrodes` (pure `derive_geometry`):
   fixed `--points`, assert `blade_lines.npy` + `j_geometry.json` byte-identical to a
   snapshot captured on `c3efc88`.
4. For `detect_j_hooks` / `segment_electrodes`: unit-test the extracted helpers
   (`nearest_facing_tips`, `skeleton_endpoints`, `gap_row_from_tips`, `load_guide_stack`)
   directly — extraction is proven inert because the function body IS the original code.
5. Import-graph check: no producer imports another producer; all import the new module.

## Rollout
One commit: add `electrode_geometry.py` + tests; then per-producer edits swapping
inline copies for imports, each validated. All on `cleanup/structural-pass`, revertible.
