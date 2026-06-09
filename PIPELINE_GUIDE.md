# Plasma Arc — Pipeline Guide & Reference

> **Purpose.** Single reference for the Plasma Arc analysis repo: how to run an
> experiment end-to-end, what every script does, and **why** each non-obvious
> choice was made. Physics detail lives in `METHODS_BLADE.md` (N₂ blade arc) and
> `METHODS.md` (CO₂ J-electrode). This guide is the operational + rationale layer.
>
> Last revised: 2026-05-21. Reflects the post-refactor state of the BLADE 11MAY26
> analysis path.

---

## 0. Experiments / subsets

| Subset | Gas | Geometry | Calibration | FPS | Data folder |
|---|---|---|---|---|---|
| **BLADE 11MAY26** | N₂, 1.5 SLPM, 0.6 mm nozzle | two blade electrodes, 25° **half-angle** wedge (~47–50° full) | **8.8 px/mm** | 200 kHz | `…/BLADE/BLADE 11MAY26/` |
| **J-electrode (ADCB)** | CO₂ | inverted-V electrodes, 2.4 mm gap | 9.167 px/mm | 200 kHz | `…/J electrode arc/`, `…/BLADE/ADCB_*` |

Two BLADE 11MAY26 recordings, identical except reactor orientation:
- **downward** (291,353 frames): apex/electrodes at **top**, arc descends, arc front = bottommost pixel.
- **upward** (459,446 frames): apex/electrodes at **bottom**, arc rises, arc front = topmost pixel.

Both are `.cine`, 256×256, 12-bit, 200,000 fps, 4.5 µs exposure (confirmed from headers).

**The active, maintained path is the BLADE 11MAY26 path below.** The J-electrode
scripts (`run_j_pipeline.py`, `analyze_tiff.py`, `nucleation_tracker.py`, the
`plasma_utils/` package, etc.) are precedent and were intentionally **not**
modified.

---

## 0a. What counts as a valid arc frame — ONE definition (`arc_filter.py`)

Every result-producing stage (measurement, event/velocity, the 2-D velocity map,
Mode A/B, paper figures, Fourier) applies the **same** validity criteria, defined
once in **`arc_filter.py`**. Do not redefine these anywhere else. The full,
authoritative spec is the `arc_filter.py` module docstring; in brief, a frame is a
valid arc iff **all three** hold:

1. **Detected** — `arc_detected == 1`: a bright region survived binarisation
   (DN > `bright_thresh`, area in `[min_arc_px, max_arc_px]`, not dark-guarded)
   **and** did not cross the electrode-gap row (`--gap-row`; a mask reaching into
   the electrode/flash zone sets `arc_detected = 0` at the source).
2. **Complete** — `arc_complete == 1`: the skeleton bridges both electrodes
   (within `ELEC_TOL_PX` of each blade line, or the widest-pair fallback).
3. **Not a flash** — `is_flash == 0`, where
   `is_flash = (area_px/arc_length_px > FLASH_EFF_WIDTH) OR (arc_width_px > FLASH_ARC_WIDTH)`.
   The canonical `is_flash` is computed once in `analyze_blade.measure_frame` and
   written to the CSV; downstream code reads that column.

**There is NO velocity cap.** Flashes are rejected at the frame level (criteria
1–3); velocities are reported exactly as measured for every valid frame and the
distribution is never truncated. (A 16 m/s cap previously stood in for flash
filtering — an unprincipled band-aid, now removed since flashes are rejected at
the source.)

Universal thresholds (`FLASH_EFF_WIDTH=20 px`, `FLASH_ARC_WIDTH=16 px`) live in
`arc_filter.py`. Per-recording **calibration** (`px/mm`, `bright_thresh`, the
electrode `gap_row`) lives in `run_11may26.py` (`PX_PER_MM` / `BRIGHT_THRESH` /
`GAP_ROW` + CLI flags), not in `arc_filter.py`.

---

## 1. How to run BLADE 11MAY26 end-to-end

### 1.1 Clean full run (recommended — current results are inconsistent)

The existing result folders are a mix of stale and partial runs. For a
trustworthy full analysis on the current code, clear them and run from scratch:

```powershell
cd "C:\Users\compl\Github\Plasma Arc"
$base = "C:\Users\compl\OneDrive\Desktop\BLADE\BLADE 11MAY26"
Remove-Item -Recurse -Force "$base\0.6mm_1.5SLPM-N2_25deg_OA_downward_results"
Remove-Item -Recurse -Force "$base\0.6mm_1.5SLPM-N2_25deg_OA_upward_results"

# full both videos, all stages, memory-safe 10k-frame chunks
python run_11may26.py --file both
```

What this does, per recording (`run_11may26.py` orchestrates it):

1. **convert** `.cine` → numbered 8-bit TIFF chunks (10,000 frames each), applying
   Drop Impact's `correct_bad_pixels_stack` (median+MAD) and percentile normalisation.
2. **detect_blade_lines** (auto, headless) → `blade_lines.npy` if missing.
3. **analyze** each chunk with `analyze_blade.py` → per-chunk `measurements.csv`
   + single-pass 2-D velocity map.
4. **merge** per-chunk CSVs → `measurements.csv`, stamping one `file` id per recording.
5. **events** `event_velocity.py` → events, advancing-front velocity, local velocity,
   per-event local-velocity stats, legacy per-event mode.
6. **re-strike Mode A/B** `qa_restrike.py` → `re_strikes_qa.csv` + `qa_restrike.png`
   (the authoritative Mode A/B).
7. **velocity map** `build_velocity_map.py` → `velocity_map_local.npy/.png` (all chunks).
8. **visualize** `visualize_blade.py` → paper figures.

> Budget a few hours; ~50 GB of intermediate TIFFs (deletable after). Tune chunk
> size with `--frames-per-tiff` (10k ≈ 0.6 GB TIFF / ~7 GB RAM; do not exceed
> ~20k or you risk OOM and the 4 GB classic-TIFF limit).

### 1.2 Cheap re-runs

Stages are resumable; the heavy stages (convert/analyze) are independent of the
Mode-A/B threshold, so iterate on classification cheaply:

```powershell
python run_11may26.py --file both --from-stage merge      # re-merge + events + mode + figures
python run_11may26.py --file downward --from-stage events  # just re-run events/mode/viz
```

Tunable knobs on the runner: `--px-per-mm 8.8`, `--frames-per-tiff`,
`--restrike-method abs|rel`, `--restrike-threshold 0.10`, `--restrike-prominence-frac 0.10`.

### 1.3 Comparative analysis (after both recordings are processed)

```powershell
$dn = "$base\0.6mm_1.5SLPM-N2_25deg_OA_downward_results"
$up = "$base\0.6mm_1.5SLPM-N2_25deg_OA_upward_results"
python mode_velocity_compare.py  --down $dn --up $up --px-per-mm 8.8 --out "$base\mode_velocity_compare.png"
python mode_arclength_compare.py --down $dn --up $up --px-per-mm 8.8 --out "$base\mode_arclength_compare.png"
```

### 1.4 Standalone QA (run any time)

```powershell
# per-chunk detection QA
python qa_blade.py "$up\tiff\<chunk>_0001.tiff" --arc-direction up --px-per-mm 8.8
# velocity-regression ground truth
python qa_velocity.py --csv "$up\measurements_all_events.csv" --events "$up\event_summary.csv" --tiff "$up\tiff\<chunk>_0001.tiff" --arc-direction up --out "$up\qa_velocity.png"
# per-re-strike mode gallery
python qa_restrike_frames.py --restrikes "$up\re_strikes_qa.csv" --tiff "$up\tiff\<chunk>_0001.tiff" --arc-direction up --out-dir "$up\restrike_gallery"
# flash explorer
python qa_flash.py --csv "$up\measurements_all_events.csv" --tiff-dir "$up\tiff" --frames-per-tiff 10000 --metric area_px --out "$up\qa_flash_area.png"
```

---

## 1J. J-electrode (CO₂) — analysed the *same way* as BLADE

The J-electrode CO₂ recording (`…/J electrode arc/AD_Cer_1.2mmNozz_1500mL_(CO2).cine`,
873,563 frames, 256×256, 200 kHz) is run through the **maintained BLADE stack**,
not the legacy `run_j_pipeline.py`. The orchestrator `run_j_co2.py` imports and
reuses `run_11may26.py`'s conversion / merge / subprocess helpers verbatim, so
every stage (convert → analyze → merge → events → re-strike Mode A/B → Fourier →
local-velocity map → figures) is the same code. Only the **reactor geometry**
differs, and the J configuration specifies it explicitly:

| Item | BLADE 11MAY26 | J-electrode |
|---|---|---|
| Electrodes | two long blades, ~47° full wedge | inverted-V razor blades, ~15° included |
| Geometry source | `detect_blade_lines.py` (auto, **infinite** lines) | `calibrate_j_electrodes.py` (**interactive**, **finite** segments) |
| Electrode ends | n/a (blades span the frame) | **specified** — tip-end + far-end per blade |
| Minimum gap | n/a | **specified** — ~2.4 mm at the apex |
| px/mm | 8.8 | 9.167 (or derived from the clicked min-gap) |
| `gap_row` | hand-set (26 / 241) | derived from the clicked tip-ends |

**Why a different geometry step.** BLADE blades are effectively infinite (the arc
foot glides across the frame), so an extrapolated line is the electrode
everywhere. J blades are short: a free arc tip can lie on the *extrapolated* blade
line **beyond** the physical electrode and would be falsely called "complete".
So the two clicked points per blade are stored as a **finite segment**
(`pt0` = tip-end near the apex, `pt1` = far-end), and `analyze_blade.py
--finite-segments` requires the arc foot to **project onto that segment**
(± `--segment-margin-px`, default 6) in addition to the perpendicular-distance
test. Clicking the two tip-ends also fixes the **minimum gap** (their pixel
separation = 2.4 mm → px/mm) and the **gap-row** flash cutoff / Mode-A/B apex
anchor (their mean row). All of this is written to `j_geometry.json`, which
`run_j_co2.py` reads to thread `--blade-lines / --finite-segments / --gap-row /
--px-per-mm` through the pipeline.

**Run it:**

```powershell
cd "C:\Users\compl\Github\Plasma Arc"
# 1) make a first chunk to calibrate on (smoke-test sized)
python run_j_co2.py --from-stage convert --max-frames 10000
# 2) click the four electrode ends — TIP-END FIRST per blade (A-tip, A-far, B-tip, B-far)
$res = "$HOME\OneDrive\Desktop\J electrode arc\AD_Cer_1.2mmNozz_1500mL_(CO2)_results"
python calibrate_j_electrodes.py --tiff "$res\tiff\AD_Cer_1.2mmNozz_1500mL_(CO2)_0001.tiff" `
    --gap-mm 2.4 --arc-direction down
# 3) run the rest of the pipeline
python run_j_co2.py --from-stage analyze
```

`--bright-thresh` (default 80) is the one detection knob likely to need tuning for
CO₂ vs N₂; check it with `mode_threshold_sweep.py` before a full run. The arc
propagates **down** from the inverted-V (`--arc-direction down`). Budget: ~57 GB
of intermediate TIFF for the full 873 k frames; use `--max-frames` to smoke-test.

> The geometry math (min-gap → px/mm, gap-row, finite-segment attachment gate, and
> the BLADE-path-unchanged regression) is unit-tested; the BLADE infinite-line
> behaviour is byte-for-byte unchanged when `--finite-segments` is absent.

---

## 2. Script map

### Pipeline (BLADE 11MAY26)
| Script | Role |
|---|---|
| `run_11may26.py` | Orchestrator (5 stages + auto blade-lines, re-strike Mode A/B, velocity map). |
| `detect_blade_lines.py` | **Headless** auto blade-line fit → `blade_lines.npy`. |
| `analyze_blade.py` | Per-frame measurements; single-pass 2-D velocity map. |
| `event_velocity.py` | Events, advancing-front + local velocity, legacy mode, per-event local-vel stats. |
| `build_velocity_map.py` | 2-D velocity map from **local** velocity, all chunks. |
| `visualize_blade.py` | Paper-ready figures. |
| `calibrate_electrodes.py` | **Interactive** blade-line definition (manual override). |

### QA / analysis
| Script | Role |
|---|---|
| `qa_blade.py` | Per-chunk detection QA (detection %, cycles, gaps, annotated frames). |
| `qa_restrike.py` | **Authoritative** Mode A/B from re-strikes → `re_strikes_qa.csv`. |
| `qa_restrike_frames.py` | Per-re-strike inspection gallery (frame each mode was classified from). |
| `qa_flash.py` | Flash explorer — rank frames by any metric. |
| `qa_velocity.py` | Velocity-regression ground-truth QA (fit + 3 independent cross-checks + frames). |
| `mode_common.py` | Shared: flash filter + re-strike/cycle metrics for the comparison scripts. |
| `mode_velocity_compare.py` | Local-velocity distribution shape + avg velocity, by mode × direction. |
| `mode_arclength_compare.py` | Arc-length distribution + per-re-strike max length, by mode × direction. |

### J-electrode (CO₂) — maintained, BLADE-style path
| Script | Role |
|---|---|
| `run_j_co2.py` | J orchestrator; reuses `run_11may26.py` stages, reads `j_geometry.json`. |
| `calibrate_j_electrodes.py` | **Interactive** finite-blade + min-gap calibration (4 clicks, tip-end first). |
| `analyze_blade.py --finite-segments` | Finite-segment attachment gate (J geometry; off = BLADE infinite lines). |

### J-electrode / legacy (superseded by the above; not modified this cycle)
`run_j_pipeline.py`, `run_pipeline.py`, `run_blade_analysis.py`, `blade_run2.py`,
`analyze_tiff.py`, `analyze_cine.py`, `nucleation_tracker.py`, `velmap_cine.py`,
`replot_velmap.py`, `fourier_analysis.py`, `visualize_cine.py`, `arc_shape.py`,
`per_event_figures.py`, `qa_check.py`, `qa_events.py`, `qa_frames.py`,
`define_roi.py`, `plasma_utils/`.

---

## 3. Design decisions & rationale (the "why")

**Calibration is threaded explicitly (8.8 px/mm).** Module defaults are the
ADCB/J-electrode value (9.167); `run_11may26.py` overrides every stage with 8.8.
Standalone invocations must pass `--px-per-mm 8.8` or they silently use the wrong
scale.

**Fixed detection threshold DN>100 (not adaptive Otsu).** The arc is a small
bright object on a near-black field; per-frame Otsu amplifies noise on dark
inter-stroke frames, creating spurious detections that corrupt event detection.
Guards: skip frames with max < 35 DN; accept mask area 22–3000 px.

**Orientation handled once, at the tip.** `arc_tip_row` is stored as
"distance from the electrode" (the upward video's raw row is flipped `H-1-row`)
so all downstream code sees one advancing-front axis regardless of reactor
orientation.

**Blade lines auto-detected, not hand-clicked.** `detect_blade_lines.py` fits the
two blade lines from the loci of arc feet across many clean-arch frames
(deterministic left/right split + total-least-squares with trimming). The
interactive `calibrate_electrodes.py` remains as a manual override. The measured
**full** opening angle (~47°) reconciles with the filename "25 deg" as the
per-blade **half-angle**.

**Chunk size = 10,000 frames.** Bounds both TIFF size (~0.6 GB) and conversion
peak RAM (~7 GB). 50,000 OOMs (~37 GB) and approaches the 4 GB classic-TIFF
limit. (`--frames-per-tiff` was previously a no-op for conversion — fixed so it
controls conversion and the merge offset consistently.)

**One `file` id per recording.** The merge stamps every chunk with the recording
stem, so `event_velocity.py`, `qa_restrike.py`, and `visualize_blade.py` (which
all group by `file`) treat the chunks as one continuous recording. Without this,
each 10k-frame chunk was treated as a separate "run" and the apex was estimated
per chunk, suppressing Mode B.

**Velocity of record = the local, time-resolved velocity.** The central-
difference per-frame velocity is noisy (legacy, kept only for a comparison
figure). `event_velocity.py` computes (a) an advancing-front regression per
event and (b) a sliding-window (Savitzky-Golay-like) **local** velocity per
frame. The 2-D velocity map (`build_velocity_map.py`) is built from the local
velocity, which is cleaner and physical (~5–14 m/s vs the central-difference
map's noise-inflated ~50 m/s).

**Single-pass velocity map.** `analyze_blade.py` accumulates the map during the
one measurement pass (no second TIFF read), and now uses the same threshold as
the measurements (a latent 80-vs-100 DN mismatch was fixed).

**Event vs re-strike — and why Mode A/B is per re-strike.** An "event"
(`event_velocity.py`) starts only when the tip returns to the apex, so events ≈
Mode-A cycles by construction and a Mode B back-breakdown is hidden *inside* an
event. A "re-strike" (`qa_restrike.py`) is **every** re-ignition, detected as a
local minimum of the arc-tip trajectory (anchored on extensions, so shallow
partial retreats are not missed). Therefore Mode A/B is classified **per
re-strike**, not per event. `re_strikes_qa.csv` is authoritative; the per-event
`mode` in `event_summary.csv` is written as `mode_eventstart_legacy` and is
**Mode-A-biased** — do not use it for A/B.

**Mode A/B metric.** Per re-strike, calibration-free: `norm_pos = (tip_min −
apex)/L_eff` (0 = apex, 1 = max extent) or `retreat_frac` (how far back toward
the gap the retreat came). Mode B = re-strike away from the apex.

**Flash filter = large area AND short skeleton.** A flash is a brief bright
*burst* at the gap: `area_px ≥ 2000 AND arc_length_px ≤ 200`. This separates
bursts (area ~2750, len ~60–145) from big *extended* arcs (same area, len 300+)
and normal arcs (small area). Flags ~0.2 % of frames. **Note:** `eff_width =
area/length` was tried first and rejected — `arc_length_px` (skeleton walk)
breaks on a minority of real arcs and inflates it, so `eff_width` flags broken-
skeleton arcs, not flashes. Genuinely huge flashes (>3000 px) are already
rejected upstream by the detector's area cap. Flashes are removed *before*
re-strike detection in `mode_common.py`.

**Two latent `global` bugs fixed.** `event_velocity.py` and `visualize_blade.py`
declared `global PX_PER_MM …` *after* the names were used in argparse help
strings — a compile-time `SyntaxError` that blocked both. The `global` is now at
the top of `main()`.

---

## 4. Output files (per recording results folder)

| File | Meaning |
|---|---|
| `tiff/<stem>_NNNN.tiff` | 8-bit, bad-pixel-corrected TIFF chunks (deletable after analysis). |
| `blade_lines.npy` | Auto-detected blade lines (2×2×2). |
| `measurements.csv` | Merged per-frame metrics, one `file` id. |
| `event_summary.csv` | One row per event; velocity, R², local-vel stats, `mode_eventstart_legacy` (legacy). |
| `measurements_all_events.csv` | Per-frame data + event IDs + local velocity. |
| `events/` | Per-event frame & dynamics CSVs. |
| **`re_strikes_qa.csv`** | **Authoritative Mode A/B**, one row per re-strike. |
| `qa_restrike.png` | Re-strike trajectory + distributions + montage. |
| `velocity_map_local.npy/.png` | 2-D local-velocity map (primary). |
| `velocity_map_comparison.png` | Local vs central-difference map. |
| `qa/`, `timeseries.png`, `fig*.png` | Detection QA and paper figures. |

---

## 5. Known limitations & TODO

- **`arc_length_px` is unreliable on a minority of frames.** The skeleton-ordering
  DFS (`ordered_skeleton_coords`) stops at any break, underestimating length.
  This is why `eff_width` failed as a flash metric and why per-frame arc-length
  has some noise. A more robust length (total skeleton pixel count, or bbox-based)
  would fix it but requires re-running `analyze_blade.py`.
- **Flash filter is in the comparison path only.** `mode_common.py` removes
  flashes before its re-strike detection, but the pipeline's `qa_restrike.py`
  (Stage 4b → `re_strikes_qa.csv`) does **not** yet apply it. *(TODO: propagate
  `area ≥ 2000 & len ≤ 200` into `qa_restrike` so the authoritative Mode A/B is
  flash-clean too.)*
- **Current results are partial/inconsistent** (see §1.1) — do a clean full run.
- **Frame rate / calibration are single-distance**; no lens-distortion correction.
- **Sandbox note (development only):** files edited during this work sometimes
  read back truncated in the Linux sandbox mount; the Windows files are correct.
  Irrelevant to running on the workstation.

---

## 6. Quick reference — what still needs running

1. **Clean full run of both recordings** (§1.1) — neither is currently a full,
   current-code run (downward measurements are stale at ~1k frames; upward is
   ~287k of 459k).
2. After that: `mode_velocity_compare.py` and `mode_arclength_compare.py` (§1.3).
3. Optional: decide whether to propagate the flash filter into `qa_restrike.py`
   (§5 TODO), then re-run `--from-stage events`.
