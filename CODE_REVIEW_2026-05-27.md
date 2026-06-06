# Plasma Arc - Scientific Code Review

**Date:** 2026-05-27
**Scope:** entire `C:\Users\compl\Github\Plasma Arc` repository (46 Python files + `plasma_utils/` package)
**Reviewer note:** This is a quality/reproducibility audit against scientific-software gold standards. Low-risk fixes were applied directly (see Section 5); anything that could change scientific output was left for your approval (Section 1).

---

## Executive summary

The **scientific core is in good shape.** The current 11MAY26 pipeline
(`run_11may26.py` -> `analyze_blade.py` -> `event_velocity.py` ->
`fourier_analysis.py` -> `visualize_blade.py` / `build_velocity_map.py`) is
well-documented, the units are handled carefully, and the velocity math is
correct. The `plasma_utils/` package is genuinely high quality (type hints,
dataclasses, documented algorithms, robust fallbacks).

The gaps are **engineering/reproducibility hygiene**, not physics:
no tests, no pinned environment, a single git commit with the package left
uncommitted, a calibration constant duplicated (and divergent) across ~12
files, and a large amount of superseded legacy code sitting alongside the
current pipeline with no signposting of which is which.

Severity counts: **1 High**, **5 Medium**, **6 Low/Hygiene**.

---

## 1. HIGH - needs your judgement (not auto-fixed)

### 1.1 `PX_PER_MM` is duplicated in ~12 files and the values disagree
- `9.167 px/mm` (22 px = 2.4 mm gap): `analyze_blade.py:45`, `arc_shape.py:55`,
  `blade_run2.py:45`, `event_velocity.py:54`, `fourier_analysis.py:35`,
  `nucleation_tracker.py:57`, `qa_check.py:37`, `velocity_sanity_check.py:37`,
  `visualize_blade.py:20`, ...
- `8.8 px/mm` (44 px = 5 mm): `run_11may26.py:53`, `build_velocity_map.py` default.

That is a ~4% spatial-scale discrepancy, which propagates linearly into every
mm and m/s number.

**Good news / important nuance:** the 11MAY26 results are NOT corrupted by this.
Every core module accepts `--px-per-mm` and overrides its module-level constant
(`event_velocity.py:992`, `analyze_blade.py:534`, `fourier_analysis.py:281`,
`visualize_blade.py:482`), and `run_11may26.py` passes `8.8` explicitly to all
stages. So a full orchestrated run is internally consistent at 8.8.

**The risk** is running any module *standalone* without `--px-per-mm`: it
silently falls back to 9.167 and produces numbers ~4% off from the pipeline.

**Recommended fix (your call, since it touches results):** make a single source
of truth - e.g. `plasma_utils/calibration.py` exporting `PX_PER_MM`, `FPS`,
`BRIGHT_THRESH` - and have every module import from it. Decide which value is
authoritative (8.8 appears to be the current 11MAY26 calibration). I did not
change any value because that would alter outputs.

---

## 2. MEDIUM

### 2.1 No tests anywhere
No `tests/`, no `test_*.py`, no `conftest.py`. For analysis code that produces
physical numbers, even a handful of unit tests would be high value, e.g.:
- `event_velocity`: a synthetic linear front of known slope -> known m/s.
- `plasma_utils.image.binarize`: a synthetic bright blob -> expected mask/area.
- `_load_map_ms` (build_velocity_map): mm/s vs m/s autodetection.
- A round-trip on `estimate_px_per_mm` with a synthetic gap.

### 2.2 Reproducibility: environment + provenance
- `setup.py` `install_requires` was **missing `scipy` and `tifffile`** even
  though both are imported directly throughout. (Fixed - see 5.)
- Cross-repo dependency: `run_11may26.py:44` and `run_j_pipeline.py` import
  `pycine` and `cine_to_tiff` from a sibling repo hard-coded at
  `C:\Users\compl\Github\Drop Impact`. Not on PyPI, not discoverable - anyone
  cloning this repo cannot run the conversion stage. Recommend vendoring the
  needed function or documenting the dependency (noted in `requirements.txt`).
- Single git commit ("Initial plasma arc analysis project"), and **almost the
  entire analysis codebase is untracked**: `git status` shows only
  `setup.py` and `plasma_utils/{__init__,image,io}.py` as tracked (the latter
  with large uncommitted edits). Every orchestrator and analysis script
  (`run_11may26.py`, `analyze_blade.py`, `event_velocity.py`,
  `build_velocity_map.py`, all `qa_*`/`mode_*`/`visualize_*`, and
  `plasma_utils/{nephrogram,qa,roi}.py`) is NOT in version control at all. There
  is effectively no history for the code that produces the results.

### 2.3 Hard-coded absolute + stale paths
Many files branch on OS and hard-code a Linux path to a **stale Cowork session**
that no longer exists: `/sessions/epic-kind-sagan/mnt/BLADE` in
`analyze_cine.py:128`, `nucleation_tracker.py:62`, `qa_events.py:49`,
`qa_frames.py:44`, `replot_velmap.py:47`, `run_j_pipeline.py:45`. Windows
defaults are also hard-coded (`C:\Users\compl\...`). Recommend a single
`BLADE_DIR` resolved from an env var (e.g. `PLASMA_BLADE_DIR`) with a sensible
default, shared across modules. Not auto-fixed because the correct value is
environment-specific.

### 2.4 Legacy / superseded code is indistinguishable from current code
The repo contains at least three pipeline generations with no "DEPRECATED"
markers:
- **Current (11MAY26):** `run_11may26.py` + `analyze_blade.py` +
  `event_velocity.py` + `fourier_analysis.py` + `visualize_blade.py` +
  `build_velocity_map.py` + `qa_restrike.py` + `detect_blade_lines.py` +
  `calibrate_electrodes.py` + `plasma_utils/`.
- **J-electrode dataset:** `run_j_pipeline.py` + `analyze_cine.py` +
  `visualize_cine.py` (still plausibly in use for that dataset).
- **Apparently superseded:** `run_pipeline.py`, `run_blade_analysis.py`
  (targets the older `ADCB_0.8mmNoz...` 3-run dataset), `blade_run2.py`
  (alt subcommand architecture), `analyze.py`, `analyze_tiff.py`,
  `replot_velmap.py` (hard-codes the old `ADCB_...` filenames at lines 49-58).

Recommend moving superseded scripts to `legacy/` or annotating their docstrings,
so a reader (or future-you) knows the entry points.

### 2.5 `binarize()` / `_f()` re-implemented instead of imported
`binarize` exists in `analyze_blade.py`, `analyze_cine.py`, `arc_shape.py`, and
`plasma_utils/image.py`; `_f()` in `analyze_blade.py`, `blade_run2.py`,
`calibrate_electrodes.py`, `plasma_utils/image.py`. The package already exports
canonical versions - the scripts should import them so the binarization
algorithm (which directly determines what counts as "arc") cannot silently
diverge between scripts.

---

## 3. LOW / hygiene

- **`print()` everywhere, no `logging`.** Acceptable for scripts, but `logging`
  (with a `-v` flag) is the gold standard and lets you silence/redirect output.
- **`FPS = 200_000` duplicated in ~20 files** - consistent everywhere, so not a
  bug, but belongs in the shared calibration module (see 1.1).
- **Magic literal** in `run_11may26.py:308`: `... / 200_000` instead of the
  `FPS` constant in the `time_s` chunk-offset computation.
- **`__pycache__/` committed/present** with mixed interpreter tags
  (`cpython-310` and `cpython-312`) - environment is not pinned to one Python.
- **Two bare `except:`** in the `_f()` helpers (analyze_blade, blade_run2) -
  swallowed `KeyboardInterrupt`/`MemoryError` too. (Fixed - see 5.)
- **No README.** There are good METHODS/PIPELINE_GUIDE docs, but no top-level
  entry-point map. (Not auto-created per repo convention; recommend adding one
  that names `run_11may26.py` as the current entry point.)

---

## 4. What is genuinely good (keep doing)

- Randomness is seeded everywhere it is used (`np.random.default_rng(0/42/7)`)
  - reproducible. (`qa_events.py:132`, `qa_frames.py:170`, `mode_*compare.py`)
- Units are tracked and documented; the local-velocity OLS in
  `event_velocity.py:432-433` (`slope * FPS / PX_PER_MM / 1000`) correctly
  converts px/frame -> m/s, and `np.polyfit` unpacking is correct.
- `plasma_utils/image.py`: bilinear-interpolated perpendicular FWHM, a fast and
  a full skeleton path, NaN-safe calibration - clean and well-documented.
- The 11MAY26 orchestrator has stage resumption, per-chunk checkpointing,
  documented RAM/chunk-size tradeoffs, and per-recording arc-direction handling.
- Divide-by-zero is guarded in the hot paths (e.g. `hi <= lo` checks,
  `total_w > 0`, `max(n_tot, 1)`); integer `//` uses are all legitimate
  chunk/window arithmetic, not accidental physics truncation.

---

## 5. Fixes applied in this review (low-risk, no change to scientific output)

1. **Narrowed two bare excepts** to `except (ValueError, TypeError):` in the
   numeric-coercion helpers `analyze_blade.py:445` and `blade_run2.py:237`
   (CSV-formatting helpers; behaviour for the intended inputs is unchanged, but
   real errors and Ctrl-C are no longer swallowed).
2. **`setup.py`:** added the missing direct dependencies `scipy` and `tifffile`
   to `install_requires`.
3. **Added `requirements.txt`** mirroring `setup.py`, with a note about the
   external "Drop Impact" dependency for the .cine conversion stages.

All edited Python files were re-checked with `python -m py_compile` and pass.

---

## 6. Suggested next steps (priority order)

1. Resolve the `PX_PER_MM` source-of-truth (1.1) - the only item that can affect
   numbers.
2. Commit `plasma_utils/` and start using real git history (2.2).
3. Add a minimal `tests/` with 3-4 synthetic-input unit tests (2.1).
4. Centralize calibration constants + `BLADE_DIR` path resolution (1.1, 2.3).
5. Quarantine legacy scripts into `legacy/` and add a README entry-point map
   (2.4, 3).
