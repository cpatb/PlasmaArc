# Plasma Arc Analysis — Methods (CO₂ J-electrode)

> **Living document.** Updated in lockstep with the codebase.
> Last revised: 2026-06-21
> **Current pipeline modules:** `run_j_co2.py`, `analyze_blade.py`, `calibrate_j_electrodes.py`,
> `detect_j_hooks.py`, `event_velocity.py`, `arc_shape.py`, `uncertainty.py`,
> `sensitivity_sweep.py`, `reynolds_comparison.py`, `build_velocity_map.py`.
> **Legacy modules (Part II, reference only):** `plasma_utils/*`, `analyze_tiff.py`.
> Companion document: `METHODS_BLADE.md` (N₂ blade-electrode experiment).

---

## 0. Pipeline status & how to read this document

**The J-electrode CO₂ analysis now runs through the maintained BLADE stack.**
Since the original write-up (May 2026), the J data has been migrated onto the
same code path used for the 11 May N₂ blade experiment: `run_j_co2.py` drives
the BLADE conversion/detection/measurement machinery (`analyze_blade.py`,
imported from `run_11may26.py`, not re-implemented) and shares
`event_velocity.py` and `arc_shape.py` with every other dataset. The only
J-specific pieces are the **electrode geometry** (hooked J-segments with a
finite-segment attachment gate) and the **per-recording calibration**.

The earlier J pipeline — per-frame Otsu detection in `plasma_utils/image.py`,
nephrogram + Radon velocity extraction, the Radon velocity-distribution map —
is **superseded** for quantitative work but still present in the repository.
It is preserved in **Part II** of this document for reference and for the
method-comparison figure; do not use it for new measurements.

| Topic | Section | Status |
|---|---|---|
| Experiment & recording sweep | Part I §1 | current |
| Pipeline stages (`run_j_co2.py`) | Part I §2 | current |
| J-hook geometry & calibration | Part I §3 | current |
| Arc detection (fixed DN threshold) | Part I §4 | current |
| Per-frame measurements | Part I §5 | current |
| Events, velocity, Mode A/B | Part I §6 | current |
| Arc-shape / normal velocity | Part I §7 | current |
| Uncertainty budget | Part I §8 | current |
| Robustness: sweep, repeats, drift | Part I §9 | current |
| Reynolds comparison | Part I §10 | current |
| Outputs & provenance | Part I §11 | current |
| Otsu detection, nephrogram, Radon | Part II §§12–17 | **legacy** |
| Known limitations & open risks | §18 | current |
| References | §19 | — |

---

# PART I — Current J-electrode CO₂ pipeline (BLADE-based)

## 1. Experiment & the recording sweep

The dataset is a **sweep over nozzle diameter × CO₂ flow rate, with repeats**.
Each recording is an **independent measurement** with its own dimensions, frame
rate (auto-detected per file — the high-speed `trig` recordings are **not**
assumed 200 kHz) and its own electrode calibration stored in its
`<stem>_results/` folder.

| Parameter | Value | Source |
|---|---|---|
| Gas | CO₂ | filename |
| Nozzle diameter | 0.4 / 0.8 / 1.2 mm (swept) | filename |
| Flow rate | 1000 / 1500 mL·min⁻¹ (swept) | filename |
| Minimum electrode (apex) gap | ≈ 2.4 mm | machined; sets px/mm |
| Electrode opening angle | ≈ 15° half-angle | filename, frames |
| Frame rate | per-recording (≈ 200 000 fps typical) | `.cine` header (auto-detected) |
| Bit depth | 12-bit (0–4095) → 8-bit TIFF | `.cine` header |

Physical picture: the arc breaks down at the apex (minimum gap), the CO₂ jet
and electric field drive it to **extend downward** along the diverging
electrodes, and it then **re-strikes** — producing the cyclic discharge visible
in the bright-pixel-count trace. Two re-nucleation modes are tracked (Mode A /
Mode B; see §6.4).

### Recordings analysed (verification set)

| Recording | nozzle / flow | n events | median v (m/s) | σ_total (m/s) |
|---|---|---:|---:|---:|
| `HS_1.2mm_1500mL_(CO2)` (full, 4.37 s) | 1.2 mm / 1500 | 669 | 4.21 | 0.22 |
| `HS_0.8mm_1500mL_(CO2)_trig(2)` | 0.8 mm / 1500 | 24 | 7.24 | 0.46 |
| `HS_0.4mm_1000mL_(CO2)_trig` | 0.4 mm / 1000 | 53 | 9.47 | 0.59 |
| `HS_0.4mm_1500mL_(CO2)_trig` | 0.4 mm / 1500 | 50 | 10.01 | 0.79 |
| `HS_0.4mm_1500mL_(CO2)_trig_repeat` | 0.4 mm / 1500 | 31 | 6.07 | 0.52 |

The 0.4 mm / 1500 mL pair disagrees by ≈ 3.9 m/s — far more than either
recording's σ_total. This is **real run-to-run variability**, not a measurement
artefact (see §9.2); the `_repeat` is excluded from cross-recording aggregates.

---

## 2. Pipeline overview (`run_j_co2.py`)

```
run_j_co2.py (orchestrator — imports run_11may26 / analyze_blade)
  Stage 1  convert    .cine → 8-bit TIFF chunks
                      • percentile normalisation from sampled frames
                      • per-chunk bad-pixel correction
  Stage 2  analyze    analyze_blade.py per chunk → per-frame measurements.csv
                      • fixed DN threshold (§4), finite-segment attachment (§3)
  Stage 3  merge      concatenate chunk CSVs with global frame numbering
  Stage 4  events     event_velocity.py → events, velocities, Mode A/B (§6)
  Stage 5  fourier    fourier_analysis.py → cycle spectra (peaks carry σ_f, §8)
  Stage 6  visualize  build_velocity_map.py + paper figures
```

**One-time calibration precedes Stage 2** (§3): convert a first chunk, run the
interactive electrode calibration to write `j_geometry.json` (+ `blade_lines.npy`
or `electrode_edges.npz`), then run the remaining stages. All resolved
parameters and provenance are written to `run_metadata.json` (§11).

---

## 3. Electrode geometry & spatial calibration

The J electrodes are **hooked J-segments**: each blade is a *finite* curved
segment, closest to its partner at the apex and diverging away from it. Two
geometry representations are supported, both read by `run_j_co2.py`:

| Representation | File | Producer |
|---|---|---|
| Straight finite blades | `blade_lines.npy` | `calibrate_j_electrodes.py` (interactive) |
| Curved J-hook edges | `electrode_edges.npz` | `detect_j_hooks.py` |

**Finite-segment attachment gate (`--finite-segments`).** Unlike the BLADE
default (which treats electrodes as *infinite* lines), the J path requires each
arc foot to land on the actual finite electrode segment near the apex. This
rejects spurious "bridging" arcs whose feet would only attach to an imaginary
extension of the blade, and it is the gate that keeps incomplete arcs out of
the event stream (the filter audit finds an empty incomplete-arc bucket on the
target recording as a result).

**The apex gap (≈ 2.4 mm) anchors three things:** the px/mm scale, the
electrode-gap **flash-cutoff row** (`--gap-row`, default per-recording), and the
Mode-A/B apex reference position.

### Spatial calibration (px/mm)

px/mm is **per-recording** and written to `j_geometry.json`. The minimum apex
gap in pixels divided by 2.4 mm gives the scale. Resolution order at run time:

```
--px-per-mm (CLI)  >  j_geometry.json  >  tiff/calibration.txt  >  9.167 (fallback)
```

The fallback `J_PX_PER_MM_DEFAULT = 9.167` is a last resort only; every reported
recording uses a calibrated per-file value (e.g. 12.5 px/mm for the
hand-segmented 0.4 mm target).

> **Calibration provenance — known issue (open).** On the 1.2 mm recording,
> `j_geometry.json` stores `px_per_mm = 8.75`, but the stored apex gap
> (`min_gap_px = 26.81`, `min_gap_mm = 2.4`) implies **11.17**. Runs use a
> `--px-per-mm 11.2` CLI override, so the reported numbers are correct and
> internally consistent, but the `8.75` field is a stale leftover from a
> different recording's calibration and should be set to 11.17. It was left
> unchanged because editing it re-triggers a multi-hour reprocess.

---

## 4. Arc detection (fixed DN threshold)

**File:** `analyze_blade.binarize` (same operators as the BLADE path).

Detection uses a **fixed DN threshold on the raw 8-bit image**, *not* the legacy
per-frame Otsu (Part II §12). The arc is a small bright object on a near-black
field; per-frame adaptive thresholds amplify noise on the dark inter-stroke
frames. Steps: dark guard → threshold at `bright_thresh` → morphological closing
→ keep the single largest connected component within `[min_arc_px, max_arc_px]`.

`max_arc_px` is raised above BLADE's 3000-px cap because the wide high-flow CO₂
arc can legitimately exceed it (default: max of the CLI value and 10 % of frame
area).

### Threshold resolution

```
--bright-thresh (CLI)  >  j_geometry.json  >  tiff/threshold_suggested.txt  >  210 (fallback)
```

The fallback `J_BRIGHT_THRESH = 210` was chosen from the arc-vs-plume
local-intensity QA (`qa_arc_local_intensity_j.py`): the arc core saturates near
255 while the diffuse plume sits below ~150, so 210 keeps the whole arc and
rejects the plume across the sweep. Per-recording values are used where
available (e.g. 160 for the 0.4 mm/1000 target; 220 and 235 for the 0.4 mm/1500
pair — the high-flow plume requires the higher value, see §9.2). The resolved
threshold and its **source** are recorded in `run_metadata.json`.

### Flash / saturation filtering (`arc_filter.py`)

`arc_filter.py` defines the unified arc-validity criteria consumed (not
redefined) by the pipeline: the gap-row flash cutoff, an `arc_width_px` leg, and
an effective-width leg (`eff_width = area / length`, which fires on disconnected
double-strikes). The filter audit (`filter_audit_montage.py`) confirms on the
target recording that rejected frames are genuine flash/saturation/blobs and
accepted frames are genuine arcs.

---

## 5. Per-frame measurements

For each detected frame `analyze_blade.measure_frame` records: arc-mask **area**;
**skeleton** (Zhang–Suen) and ordered centreline; **arc length** (skeleton px /
px-per-mm); **FWHM width** sampled perpendicular to the local tangent (a proxy
for channel diameter **plus** motion blur at the few-µs exposure);
**brightness** (mean / peak / integrated); **centroid**; **bounding box**;
**curvature**; the two skeleton **endpoints** (arc feet); and the **arc-front
tip** (extreme pixel in the propagation direction).

**Two conventions matter for correctness:**
- **Intensity is measured on raw DN (0–255)**, never on the per-frame normalised
  image. Cross-frame brightness comparisons must use `integrated_intensity`.
- `arc_tip_row` is stored as **distance from the electrode**, increasing as the
  arc extends, so downstream code sees one consistent advancing-front axis
  regardless of reactor orientation.

A bridging arc must root on **both** electrodes; this is enforced by the
finite-segment attachment gate (§3).

---

## 6. Events, velocity & modes (`event_velocity.py`)

Operates on the merged per-frame CSV. **Shared with the BLADE pipeline**; the J
run supplies `--px-per-mm <calibrated>` and the auto-detected `--fps`.

> **fps is per-recording and must thread through.** It is auto-detected by the
> `.cine` probe and threaded into `event_velocity.py`, `qa_restrike.py`, and
> `fourier_analysis.py`. A past regression that mis-scaled velocity and
> frequency by `true_fps / 200000` is now guarded by a dimensional self-check
> (synthetic 200 kHz vs 100 kHz → velocity ratio exactly 2.0) in
> `test_j_geometry.py`.

### 6.1 Discharge event detection

A new primary event begins when the tip returns to within
`NUCLEATION_SLACK_FRAC` (default **15 %**) of the effective arc range above the
electrode row, or after a detection dropout > 500 frames. The electrode row is
the 5th-percentile `arc_tip_row` of complete-arc frames; `L_eff_px = p90(tip) −
electrode_row`. Intermediate retreats > 50 % of the current extension are
flagged as **sub-events**.

### 6.2 Advancing-front regression

Only monotonic-advancing-envelope frames are fit:

```
velocity = slope [px/frame] × FPS / px_per_mm / 1000   →   m/s
```

R² reports linearity. Complete-arc frames are preferred; all detected frames are
used if too few complete frames exist.

### 6.3 Sliding-window local velocity

A per-frame local velocity uses a centered OLS window of `LOCAL_VEL_N = 10`
advancing-front neighbours (±5; `--local-vel-n`), equivalent to a
Savitzky–Golay first derivative. **N = 10 is canonical**: N = 5 spans only
~25 µs and admits short-baseline spikes (flagged UNSTABLE in §9.1); N = 10
(~50 µs) suppresses them at modest cost in time resolution. The sensitivity
sweep confirms the headline metric is stable for N ≥ 7.

### 6.4 Mode A / Mode B classification

From the dimensionless **normalised nucleation position**:

```
norm_nucleation_pos = (tip_row_start − electrode_row) / L_eff_px
```

Events with `norm_nucleation_pos > 0.15` are **Mode B** (back-breakdown at an
intermediate position); the rest are **Mode A** (full extinction, re-ignition at
the apex). The threshold is self-consistent with the event detector. The
authoritative Mode A/B record is `re_strikes_qa.csv`.

### 6.5 Per-event output

`event_summary.csv` carries both characterisations per event: `vel_ms` (the
whole-event average rate) and the time-resolved `local_vel_median / mean / p10 /
p90 / peak / max / std`. The full per-frame profile is in
`events/event_NNNN_dynamics.csv`. For multi-phase events the two diverge — the
signal that a single average is insufficient.

---

## 7. Arc-shape tracking & normal velocity (`arc_shape.py`)

The arc is a deformable filament, not a rigid body: the tip regression captures
only the leading edge, while the arc body can flex and sweep laterally.
`arc_shape.py` resamples the ordered skeleton to N arc-length-equally-spaced
points (point 0 = electrode-proximal end), then for each consecutive frame pair
within an event decomposes each point's displacement into components **tangent**
and **normal** to the local arc direction:

```
v_normal_i     = dot(p_i(t+1) − p_i(t), normal_i)  / Δt    ← lateral sweep
v_tangential_i = dot(p_i(t+1) − p_i(t), tangent_i) / Δt    ← along-arc motion
```

Only complete-arc frames are used (truncated skeletons make the parameterisation
unreliable). Output: `arc_shape_velocities.csv` with `norm_pos = seg_idx /
(N − 1)`.

---

## 8. Uncertainty budget (`uncertainty.py`)

**Single source of truth** for the systematic budget the event/Fourier stages
previously lacked. The old `vel_ms_err` was only the OLS slope SE (random
fitting error) — on the target recording a misleadingly tight median 0.10 m/s.
Reported velocities must additionally carry the **calibration (px/mm)** and
**fps** systematics.

For `v = slope · FPS / px_per_mm / 1000`, propagating uncorrelated terms:

```
(σ_v / |v|)²  =  (σ_v_stat / |v|)²  +  (σ_fps / FPS)²  +  (σ_pxmm / px_per_mm)²
```

| Quantity | Default | Justification |
|---|---|---|
| `SIGMA_FPS_REL` | 0.1 % | Phantom HS crystal ≈ 50–100 ppm; 0.1 % is conservative for trig-mode |
| `SIGMA_PXMM_REL` | 5 % | √2 × 1 px click localisation / ≈ 26.56 px apex gap ≈ 5.3 % |

Both are relative 1-σ and **overridable per-recording** (`--pxmm-rel-err`,
`--fps-rel-err`), logged in `run_metadata.json`.
`pxmm_uncertainty_from_calibration()` can derive σ_pxmm from the measured
`min_gap_px` instead of the flat default.

**Columns written** by `event_velocity.py`: `vel_ms_err_stat`, `vel_ms_err_sys`,
`vel_ms_err_total` (the legacy `vel_ms_err` is retained = stat only).
**Frequencies** (`f = k · FPS / N`) have no spatial term, so
`σ_f / f = σ_fps / FPS`; `fourier_analysis.py` annotates each peak as `f ± σ_f`.

`recompute_uncertainty.py` derives the three velocity columns on existing
`_results/` directories **without re-running the pipeline** (a pure derived
column — point estimates are unchanged). The net effect on the target recording:
median velocity 9.47 → **9.47 ± 0.59 m/s**, an error bar ≈ 6× the old
slope-SE-only value because the calibration term had been entirely outside the
budget.

---

## 9. Robustness: sensitivity, repeats, temporal drift

### 9.1 Sensitivity sweep (`sensitivity_sweep.py`)

Tier-A (post-hoc; re-reads `measurements_all_events.csv`, no re-detection). A
parameter is flagged **UNSTABLE** if shifting it moves the headline median by
more than the systematic floor `σ_floor = √(σ_pxmm² + σ_fps²) × median_v`.

| Parameter | Range swept | Result |
|---|---|---|
| `local_vel_n` | 5, 7, 10, 15, 20 | stable for **N ≥ 7**; N = 5 flagged UNSTABLE |
| `flash_arc_width` | 12–50 px | all STABLE (max |Δ| 0.026 m/s) |
| `flash_eff_width` | 15–50 | all STABLE |

Tier B (varying `bright_thresh` and re-detecting) is **not** part of the routine
sweep — see §18.

### 9.2 Repeat handling (`sweep_utils.py`)

The 0.4 mm / 1500 mL condition has two recordings whose per-event medians
disagree by 3.94 m/s. A matched-settings diagnostic (reprocessing chunk 1 of
each at identical `bright_thresh`) showed the gap is **+3.35 m/s of genuine
run-to-run physics**, not a threshold or calibration artefact (the high-flow
plume actually *requires* the elevated threshold; dropping to the sweep-typical
175 balloons the connected region and degenerates the skeleton).

`sweep_utils.py` provides an `.excluded_from_sweep` sentinel-file mechanism:
dropping that file (with a written rationale) into a `_results/` directory causes
the four cross-recording aggregators (`reynolds_comparison.py`,
`mode_ratio_vs_re.py`, `asymmetry_vs_time.py`, `warmup_time.py`) to skip it. The
`_repeat` is excluded on this basis; its per-recording figures remain valid, and
per-recording diagnostics still process it.

### 9.3 Temporal drift & long recordings

`temporal_velocity_profiles.py` (first-vs-last events) and `warmup_time.py`
(exponential fit) test whether arc dynamics drift within a recording. The short
(~95 ms) recordings have too few events to resolve a sign; the **1.2 mm
recording is the exception**. It is the only long-duration file (873 563 frames =
**4.37 s**), and the earlier robustness pass had analysed only a ~10 % prefix.
A go/no-go probe (`long_video_probe.py`) flagged drift, so it was reprocessed
end-to-end:

| Metric | 10 % prefix | Full 4.37 s |
|---|---:|---:|
| events fitted | 74 | **669** |
| median v (m/s) | 3.68 | **4.21** |
| \|asym\| median | 0.213 (initial transient) | **0.187** (time-average) |

The full record shows a **stable positive velocity drift** (≈ +0.34 m/s per
second; KS p down to 10⁻⁴¹), the only recording where the sign does not flip
across group sizes. The exponential warm-up fit does **not** converge
(τ = 13.9 ± 16.8 s, R² ≈ 0): 4.37 s is too short relative to the timescale.
**Lesson:** long `.cine` files must be probed before trusting a prefix.

---

## 10. Reynolds comparison (`reynolds_comparison.py`)

Built to relate arc behaviour to the **gas-flow** Reynolds number across the
sweep. Definitions (cited in every figure caption):

- Characteristic length **L = nozzle diameter d**.
- Characteristic velocity **U = bulk gas velocity** `v_bulk = Q / (π/4 · d²)` —
  **not** the arc velocity.
- Fluid: CO₂ at T = 295.15 K, P = 101.325 kPa. ρ = 1.815 kg·m⁻³ (CRC Handbook
  90th ed.); μ = 1.48 × 10⁻⁵ Pa·s (Lemmon & Jacobsen 2004).
- **Re = ρ · v_bulk · d / μ.**
- Default uncertainties: σ_d/d = 5 %, σ_Q/Q = 5 %, σ_T = 2 K, σ_P = 1 kPa,
  σ_μ/μ = 2 % (override at CLI when machine drawings / MFC calibrations exist).

Repeats per condition feed the error bars. Outputs (each 300 DPI PNG + vector
PDF + sidecar caption): `reynolds_arc_velocity`, `reynolds_mode_b_fraction`,
`reynolds_cycle_frequency` (plus `mode_ratio_vs_re`, `asymmetry_vs_re`).

| d (mm) | Q (mL/min) | repeats | Re | σ_Re | Ma | median v_arc (m/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1.2 | 1500 | 1 | 3 253 | 406 | 0.08 | 4.21 |
| 0.8 | 1500 | 1 | 4 879 | 608 | 0.18 | 7.24 |
| 0.4 | 1000 | 1 | 6 506 | 811 | 0.49 | 9.47 |
| 0.4 | 1500 | 1 (repeat excluded) | 9 759 | 1 217 | 0.74 | 10.01 |

> **Confound (in every caption):** changing nozzle diameter changes **both** Re
> and reactor geometry (impingement distance, divergence angle). A visible trend
> therefore conflates fluid-dynamic and geometric effects; isolating Re alone
> needs fixed-d, swept-Q, but the sweep has at most two Q per d.
>
> **Compressibility caveat:** v_bulk reaches ≈ 199 m/s at d = 0.4 mm /
> 1500 mL/min (Ma ≈ 0.74; a_CO₂ ≈ 270 m/s). The reported Re is the
> *incompressible-equivalent* number using ambient ρ; the small-nozzle corner is
> firmly compressible and the nozzle-exit density is not the ambient ρ used here.

---

## 11. Outputs & provenance

| File | Description |
|---|---|
| `tiff/` | numbered 8-bit TIFF chunks |
| `j_geometry.json` | min gap, px/mm, gap_row, finite-segment flags |
| `blade_lines.npy` / `electrode_edges.npz` | finite electrode geometry |
| `measurements.csv` | merged per-frame metrics |
| `event_summary.csv` | one row per event incl. velocity, R², Mode A/B, σ columns |
| `measurements_all_events.csv` | per-frame data with event IDs + local velocity |
| `re_strikes_qa.csv` | authoritative Mode A/B |
| `events/` | per-event frame and dynamics CSVs |
| `velocity_map_local.*` | 2-D local-velocity map (`build_velocity_map.py`) |
| `fourier_*` | cycle spectra with σ_f-annotated peaks |
| `run_metadata.json` | **provenance** (below) |
| `Reynolds_comparison/` | cross-recording figures (§10) |

**`run_metadata.json` provenance** carries, for every run: fps, px/mm + source,
`bright_thresh` + source, geometry mode + finite-segment flags, gap_row,
`local_vel_n`, the σ_pxmm/σ_fps budget, git hash, and UTC timestamp — so every
reported number is auditable end to end.

---

# PART II — Legacy TIFF / Radon pipeline (reference only)

> **Superseded.** The sections below describe the *original* J pipeline
> (`plasma_utils/*`, `analyze_tiff.py`): per-frame Otsu detection, nephrogram
> construction, and Radon-transform velocity extraction. They are retained for
> historical reference and for the legacy method-comparison figure only. For new
> measurements use Part I.

## 12. Arc detection — percentile-filtered Otsu (legacy)

**File:** `plasma_utils/image.py : binarize()`

A global Otsu threshold on the full histogram is biased toward the background.
The legacy path applied Otsu only to pixels above the per-frame median:

```
gray[gray > percentile(gray, 50)]  → bright_pixels
threshold = threshold_otsu(bright_pixels)
binary = gray > threshold
```

Morphological cleanup: closing (disk r = 3) to fill channel gaps, opening
(disk r = 2) to remove speckle, then largest-connected-component selection.

> Replaced in Part I §4 by a fixed DN threshold, which is more stable on the
> dark inter-stroke frames.

## 13. Skeletonisation & path ordering (shared concept)

The mask is thinned to a 1-px skeleton (Zhang–Suen via
`skimage.morphology.skeletonize`), then ordered into a connected path by a
depth-first walk on the 8-connected adjacency graph. Endpoints are degree-1
skeleton pixels; if none exist the walk starts arbitrarily. (The current path
uses the same skeleton concept inside `analyze_blade`.)

## 14. Physical calibration — 5th-percentile gap (legacy)

**File:** `plasma_utils/image.py : estimate_px_per_mm()`

```
gap_px[i]  = ‖tip_b − tip_a‖₂
px_per_mm  = percentile(gap_px, 5) / electrode_gap_mm
```

The 5th percentile (not the absolute minimum) guards against single-frame
outliers. Superseded by the explicit J-hook apex-gap calibration in Part I §3.

## 15. Arc measurements (legacy reference)

`measure_frame()` produced arc length (skeleton px / px-per-mm), FWHM width
(41-px perpendicular profile, every 5 skeleton px), brightness (mean / peak /
integrated), centroid, bounding box, tip A/B, electrode gap, and curvature
(mean |Δθ| along the skeleton). These map directly onto the current per-frame
measurements (Part I §5), the difference being that the current path measures
intensity on **raw DN**, not the normalised image.

## 16. Nephrogram, Radon velocity & velocity-distribution map (legacy)

**Files:** `plasma_utils/nephrogram.py`

- **Nephrogram** — a space–time kymograph: each frame's 2-D field is reduced to a
  1-D profile (`max` / `mean` / `sum`) along the `vertical` or `horizontal`
  axis, stacked into a time × space image. Downward-sloping streaks encode
  propagation; the slope is velocity.
- **Radon velocity** (`radon_velocity()`) — the dominant streak angle θ over
  θ ∈ [−80°, 80°] gives `v = cot(θ) · fps / px_per_mm`. (Using tan θ would
  return the reciprocal — see the original convention note.)
- **Velocity-distribution map** (`velocity_distribution_map()`) — a sliding
  temporal window (50 frames, 10-frame hop) histograms the top-30 Radon
  candidates by magnitude, weighted by spectral energy → a time × velocity map.

> Superseded by the advancing-front regression + sliding-window local velocity
> (Part I §6) and the local-velocity 2-D map (`build_velocity_map.py`). The
> central-difference and Radon estimates are retained only for the
> method-comparison figure.

## 17. Cycle detection & QA frame rendering (legacy)

- **Cycle detection** (`detect_cycles()`) — `scipy.signal.find_peaks` on a 1-D
  temporal signal (mean nephrogram intensity over the central 20 % of the
  spatial axis), min prominence 0.5σ, min spacing ≥ 10 µs. Cycle frequency in
  the current pipeline comes instead from the re-strike interval / Fourier
  spectrum.
- **QA frame rendering** (`plasma_utils/qa.py : render_qa_frame()`) — annotated
  overlays (mask, skeleton, width samples, tips, gap line, centroid, bbox) with
  a monospace text panel and a γ = 0.5 display stretch (display-only). The
  current path uses the BLADE QA montages (`qa_restrike`, `filter_audit_montage`).

---

## 18. Known limitations & open risks

Current-pipeline risks (from the June robustness pass — flagged honestly, not
bugs):

- **No full pipeline re-run on the target recording.** Existing `_results/`
  already had `fps = 200000` threaded everywhere; the uncertainty addition is a
  pure derived recompute, so current `vel_ms` values are byte-identical to a
  re-run. Defensible, but a shortcut.
- **Tier-B threshold sweep not run.** `bright_thresh` stability was argued from
  the arc/plume QA, not a re-detection sweep across `{130…190}`.
- **Mach > 0.3 at the small-nozzle corner.** Re is incompressible-equivalent;
  nozzle-exit density is wrong by O(10–30 %) at Ma ≈ 0.49–0.74. Captioned, not
  corrected.
- **σ_pxmm assumes isotropic 1-px clicks.** For the hand-segmented J-hook the
  localisation is correlated and possibly anisotropic; the 5 % default is a
  floor. Override when better calibration audits exist.
- **σ_fps default 0.1 %.** Conservative for the crystal but possibly
  *un*-conservative for trig-mode line-time rounding. A `.cine`-header
  requested-vs-achievable-fps probe would give a per-recording value.
- **0.4 mm / 1500 mL is now effectively single-recording** after the repeat
  exclusion (§9.2). 2–3 more independent recordings (chamber repurged between
  each) would bracket endogenous run-to-run spread vs setup drift.
- **Calibration field discrepancy** in the 1.2 mm `j_geometry.json` (8.75 vs
  11.17; §3) — correct the stored field on the next reprocess.

Standing measurement limitations:

- **Motion blur** at the few-µs exposure inflates FWHM width ∝ tip speed.
- **Single-distance calibration**; no lens-distortion correction.
- **Per-frame normalisation** (display/legacy) means cross-frame brightness must
  use `integrated_intensity` on raw DN.

---

## 19. References

1. Otsu, N. (1979). "A threshold selection method from gray-level histograms."
   *IEEE Trans. Syst. Man Cybern.* **9**(1), 62–66.
2. Zhang, T. Y., & Suen, C. Y. (1984). "A fast parallel algorithm for thinning
   digital patterns." *Commun. ACM* **27**(3), 236–239.
3. Deans, S. R. (1983). *The Radon Transform and Some of Its Applications.* Wiley.
4. van der Walt, S. et al. (2014). "scikit-image: image processing in Python."
   *PeerJ* **2**, e453.
5. Lieberman, M. A., & Lichtenberg, A. J. (2005). *Principles of Plasma
   Discharges and Materials Processing.* Wiley.
6. Czernichowski, A. (1994). "Gliding arc: applications to engineering and
   environment control." *Pure Appl. Chem.* **66**(6), 1301–1310.
7. Zhang, H., et al. (2020). "Optical study of gliding arc discharge modes."
   *IET High Voltage* **5**(4), 386–392.
8. Fridman, A., Gutsol, A., & Kennedy, L. A. (2005). *Non-Equilibrium Air Plasmas
   at Atmospheric Pressure.* IOP Publishing.
9. Lemmon, E. W., & Jacobsen, R. T. (2004). "Viscosity and thermal conductivity
   equations for nitrogen, oxygen, argon, and air." *Int. J. Thermophys.* **25**,
   21–69. (CO₂ transport properties; see also *J. Phys. Chem. Ref. Data* 33.)
10. CRC Handbook of Chemistry and Physics, 90th ed. (2009). CRC Press. (CO₂ density.)

---

*End of document.*
