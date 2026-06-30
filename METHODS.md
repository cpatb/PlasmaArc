# Plasma Arc Analysis — Methods (CO₂ J-electrode)

> **Living document.** Updated in lockstep with the codebase.
> Last revised: 2026-06-29
> **Current pipeline modules:** `run_j_co2.py`, `analyze_blade.py`, `calibrate_j_electrodes.py`,
> `detect_j_hooks.py`, `event_velocity.py`, `arc_shape.py`, `npoint_restrike.py`,
> `uncertainty.py`, `sensitivity_sweep.py`, `reynolds_comparison.py`, `build_velocity_map.py`.
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
| `HS_1.2mm_1500mL_(CO2)` (full, 4.37 s) | 1.2 mm / 1500 | 669 | 4.22 | 0.24 |
| `HS_0.8mm_1500mL_(CO2)_trig(2)` | 0.8 mm / 1500 | 24 | 6.13 | 0.39 |
| `HS_0.4mm_1000mL_(CO2)_trig` | 0.4 mm / 1000 | 53 | 10.70 | 0.66 |
| `HS_0.4mm_1500mL_(CO2)_trig` | 0.4 mm / 1500 | 50 | 10.35 | 0.82 |
| `HS_0.4mm_1500mL_(CO2)_trig_repeat` | 0.4 mm / 1500 | 31 | 6.22 | 0.53 |

(Velocities are on the per-video apex-gap calibration, px/mm = `min_gap_px / 2.4`.)

The 0.4 mm / 1500 mL pair disagrees by ≈ 4.1 m/s — far more than either
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

The archival CO2 videos have **no ruler in frame**, so the *only* spatial
calibration is the known minimum electrode gap (2.4 mm). px/mm is therefore
**derived per-video** from the clicked tip-end separation:

```
px/mm = min_gap_px / 2.4 mm        (per recording, from j_geometry.json)
```

There is **no hardcoded px/mm and no numeric fallback**: `run_j_co2.py` computes
the scale from `min_gap_px / min_gap_mm` and stops with an error if a recording
has not been gap-calibrated. The `--px-per-mm` CLI flag remains only as an
explicit debug override (it prints a warning). The per-video values across the
sweep span ≈ 8.5–11.2 px/mm (e.g. 11.07 for 0.4 mm/1000, 11.17 for 1.2 mm,
10.33 for 0.8 mm/1500), reflecting that the recordings were taken on different
days at different magnifications.

> **Calibration provenance (resolved 2026-06-25).** The earlier inconsistency —
> `j_geometry.json` storing a `px_per_mm` field (e.g. 8.75) that disagreed with
> its own clicked gap — is gone: the field is now always written as
> `min_gap_px / 2.4` (consistent), and the pipeline ignores the stored field and
> recomputes from the gap regardless. All recordings were fully re-converted and
> reprocessed on this calibration.

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

> **REVISED (authoritative).** The Mode A/B boundary is now the **dead-zone valley**
> of the bimodal initiation-depth distribution, and the **primary detector is the
> 10-segment whole-arc detector** (`npoint_restrike.py` → `npoint_restrikes.csv`),
> not the tip detector. The `norm_pos > 0.10` cut below is the *initial* cut; it
> slices through the near-apex first peak, so the valley cut (auto-detected per
> recording, ~3.4–4.6 mm below the apex; applied by `apply_valley_modes.py`)
> replaces it. Under the revised scheme: Mode B (back-breakdown anywhere on the arc)
> self-sustains (P(B|B)=0.93; first-order Markov, p=5×10⁻⁶⁴); **every reset interval
> is followed by Mode A** (post-reset p_A=1.00 vs 0.08 background, p=7×10⁻⁶⁰); and
> last-arc geometry predicts the next mode (CV-AUC ≈ 0.75–0.82). Full detail and the
> tip-detector cross-check (which shows A↔B alternation) are in
> `MODE_DYNAMICS_ANALYSIS.md`. The text below documents the original tip classifier.

**Initial classifier (`qa_restrike.py` → `re_strikes_qa.csv`).** Mode A/B is
classified **per re-strike**, not per event (a Mode-B back-breakdown re-ignites
*inside* an event, so it is invisible to the event detector). A re-strike is the
local minimum of the arc-tip trajectory between two extension peaks. For each:

```
norm_pos = (tip_min − apex) / L_eff        Mode B if norm_pos > 0.10   [initial cut]
```

Two properties make this **non-circular and flash-clean** (hardened 2026-06-25):

- **apex = the calibrated electrode-gap row** (`gap_row` from `j_geometry.json`,
  mapped into the `arc_tip_row` coordinate), i.e. the *physical* gap — not a
  statistical percentile of where the tip happened to retreat to. If `--gap-row`
  is absent it falls back to the 5th-percentile and the QA says so.
- **flashes are removed before re-strike detection** (`arc_filter` criterion 3),
  so saturation bursts cannot create phantom re-strikes or mislabels.

`qa_restrike.py` emits process QA for every recording: `re_strikes_qa.png`
(trajectory with A/B-coloured re-strikes, the apex line, the cut, and a montage of
the extreme cases) and `re_strikes_qa_summary.txt` (detected frames, % rejected as
flash, apex source, thresholds, and the A/B counts).

**Legacy per-event field (do not use for A/B).** `event_summary.csv` also carries
`mode_eventstart_legacy` from the per-event nucleation position
(`norm_nucleation_pos = (tip_row_start − electrode_row)/L_eff_px > 0.15`). Because
an event starts only when the tip returns near the apex, this is **Mode-A-biased**
and is retained only for provenance.

### 6.5 Per-event output

`event_summary.csv` carries both characterisations per event: `vel_ms` (the
whole-event average rate) and the time-resolved `local_vel_median / mean / p10 /
p90 / peak / max / std`. The full per-frame profile is in
`events/event_NNNN_dynamics.csv`. For multi-phase events the two diverge — the
signal that a single average is insufficient.

### 6.6 Whole-arc (N-point) breakpoint re-strike detector (`npoint_restrike.py`)

**Why a complement is needed.** The authoritative re-strike metric (§6.4) is
built from the **arc-tip** trajectory (`arc_tip_row`, the arc *front* pixel). It
is structurally **blind** to re-strikes where the tip is invariant but one
**leg/side** of the arc breaks and re-initiates: the front pixel never moves, so
no tip cycle is seen. These one-sided "leg" breaks are real Mode-B
(back-breakdown) events, so the tip metric **undercounts** Mode B. The N-point
detector tracks the **whole arc** and recovers them. It is a **complement**: the
tip metric (§6.4) is kept intact as the conservative companion, and both are
reported.

**Frame set (identical to §6.4).** The rows of `measurements_all_events.csv` with
`arc_detected==1 AND is_flash==0`, in `frame` order, grouped by `file` (no
`arc_complete` requirement — the same set the tip detector uses). The N-point arc
is **reconstructed** for those frames only (analyze_blade is **not** re-run): the
raw TIFF page is re-binarised with `analyze_blade.binarize` at the recording's
calibrated `bright_thresh` / `gap_row` / `max_arc_px` (close radius 3, dark-guard
35, `min_arc_px` 22), then `skeletonize` → `arc_shape.ordered_path` →
`resample_arc(N)` → `orient_arc` (point 0 electrode-proximal). Per-point depth
`d_i = row_i − gap_row`; frames whose skeleton/resample fails are skipped.

**Detector (per point, unioned).** For each of the N points, the depth signal is
smoothed (`uniform_filter1d`, size 3) and `find_peaks(prominence = 0.10·(p90−p5
of that point's depth), distance ≥ 5)` locates its extensions; a re-strike is the
`argmin` between consecutive peaks — the **same** collapse-between-extensions
logic as `qa_restrike.detect_restrikes`, applied per point with a per-point
adaptive prominence. Events from all N points are **unioned and de-duplicated
within 5 frames**, keeping the strongest-collapse representative and its
triggering point index *i*.

**Classifier (per event).** `L_eff` = p90 of the **deepest-point** depth (matches
`qa_restrike.apex_and_range`). `Mode B` if the deepest-point depth at the trough
frame > `0.10·L_eff` (arc stayed extended / back-breakdown), else `Mode A` (reset
to the gap) — the faithful generalisation of the tip `norm_pos` cut. Breakpoint
position `s = i/(N−1)` (0 = one foot, 1 = the other); tagged **leg** if `s < 0.2
or s > 0.8`, else **central**. Each event also carries `tip_invisible` (no tip
re-strike within 5 frames), which isolates the genuinely new leg events.

**Outputs.** `<stem>_results/npoint_restrikes.csv` (one row per event:
`frame, mode, kind, s, point_idx, breakpoint_depth_mm, deepest_depth_mm,
strength_px, cycle_frames, tip_invisible`) and the combined
`Reynolds_comparison/npoint_restrike_summary.csv` (N, n_events, frac_B,
leg_share_of_B, mean_cycle_ms, Re, d_mm, q_mLpm, is_repeat). `N=10` is the
headline; `N=15` is used for breakpoint-position localisation.

**Validation (`python npoint_restrike.py validate`).**
- **V1 — reproduce the tip detector (hard gate).** Re-deriving
  `detect_restrikes`+`classify`+`apex_and_range` on `arc_tip_row` reproduces every
  recording's `re_strikes_qa.csv` **exactly** (count + A/B). Reference
  `HS_0.4mm_1000mL`: **92 re-strikes, 53 A / 39 B, 0.981 ms** — matched. If V1
  fails the run stops.
- **V2 — reconstruction fidelity.** On a random sample of detected frames the
  re-binarised mask **front pixel** equals `arc_tip_row` to **0.0 px** median,
  proving the binarisation reproduces the pipeline detection. The N-point
  **deepest centerline** point sits a small, consistent ~2–3 px **above** the
  front pixel (half the local tip width); this offset is physical, not error, and
  **cancels** in the A/B cut because both `deepest_depth` and `L_eff` are
  centerline quantities.
- **V3 — N-sensitivity.** Re-running at N = 10 / 15 / 20 leaves `n_events`,
  `frac_B` and `leg_share` stable (e.g. `HS_0.4mm_1000mL`: Δevents = 3,
  Δfrac_B = 0.008), confirming the headline numbers are not an artefact of N.
- **V4 — visual audit.** `<stem>_results/npoint_leg_event_audit.png` renders a
  seeded sample of leg-break, tip-invisible Mode-B events: raw frames spanning
  each event with the N-point arc overlaid and the triggering point ringed,
  beside the depth-vs-time traces (tip + per-point, event marked). These confirm
  the leg points are real one-sided breakups, not skeleton/resample noise.
- **V5 — current-trace hook.** `crosscorrelate_current_trace()` is a documented
  TODO: true event-count validation needs the oscilloscope current trace (does a
  breakdown pulse coincide with each detected event?), which is not yet available
  for these archival recordings.

**Caveat carried over.** `arc_filter.is_flash`'s `eff_width = area/arc_length_px`
leg uses `arc_length_px = len(ordered_skeleton_coords)`, which follows only one
skeleton branch and so **under-measures forked arcs**; this over-flags ~10–19 % of
real deep **forked** arcs as "flash" and drops them from the frame set. It does
**not** change the tip `frac_B` (the dropped frames carry no tip re-strike there),
but it biases residence/length/velocity stats and removes some frames the N-point
detector would otherwise see — i.e. the leg-share reported here is, if anything, a
**lower bound**.

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
median velocity **10.70 ± 0.66 m/s** (σ_stat 0.30 ⊕ σ_sys 0.54), an error bar
≈ 2× the slope-SE-only value because the calibration term had been entirely
outside the budget.

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
disagree by ≈ 4.1 m/s (10.35 vs 6.22). A matched-settings diagnostic
(reprocessing chunk 1 of each at identical `bright_thresh`) showed the gap is
**genuine run-to-run physics**, not a threshold or calibration artefact (the high-flow
plume actually *requires* the elevated threshold; dropping to the sweep-typical
175 balloons the connected region and degenerates the skeleton).

`sweep_utils.py` provides an `.excluded_from_sweep` sentinel-file mechanism:
dropping that file (with a written rationale) into a `_results/` directory causes
the cross-recording aggregators (`reynolds_comparison.py`,
`mode_ratio_vs_re.py`, `asymmetry_vs_time.py`) to skip it. The
`_repeat` is excluded on this basis; its per-recording figures remain valid, and
per-recording diagnostics still process it.

### 9.3 Temporal drift & long recordings

`temporal_velocity_profiles.py` (first-vs-last events) tests whether arc dynamics
drift within a recording. The short (~95 ms) recordings have too few events to
resolve a sign; the **1.2 mm recording is the exception**. It is the only
long-duration file (873 563 frames = **4.37 s**), and the earlier robustness pass
had analysed only a ~10 % prefix. A go/no-go probe (`long_video_probe.py`)
flagged drift, so it was reprocessed end-to-end:

| Metric | 10 % prefix | Full 4.37 s |
|---|---:|---:|
| events fitted | 74 | **669** |
| median v (m/s) | 3.68 | **4.22** |
| \|asym\| median | 0.213 (initial transient) | **0.187** (time-average) |

The full record shows a **stable positive velocity drift** (≈ +0.34 m/s per
second; KS p down to 10⁻⁴¹), the only recording where the sign does not flip
across group sizes. **Lesson:** long `.cine` files must be probed before trusting
a prefix. (An exponential warm-up *time-constant* fit was dropped: 4.37 s is too
short relative to the timescale, so τ was unresolvable, R² ≈ 0 — the τ
quantification and its figures were removed as uninformative.)

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
`reynolds_cycle_time` (mean re-strike cycle **duration in ms**, period =
1000·cycle_frames/fps) plus `mode_ratio_vs_re`, `asymmetry_vs_re`.

| d (mm) | Q (mL/min) | repeats | Re | σ_Re | Ma | median v_arc (m/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1.2 | 1500 | 1 | 3 253 | 406 | 0.08 | 4.22 |
| 0.8 | 1500 | 2 | 4 879 | 608 | 0.18 | 7.01 |
| 0.4 | 1000 | 1 | 6 506 | 811 | 0.49 | 10.70 |
| 0.4 | 1500 | 1 (repeat excluded) | 9 759 | 1 217 | 0.74 | 10.35 |

The arc velocity rises with Re from 4.22 m/s (1.2 mm) to ≈ 10.7 m/s at the
0.4 mm / 1000 point, then is **flat-to-slightly-lower** (10.35 m/s) at the
highest-Re 0.4 mm / 1500 point — i.e. the trend is monotone up to a plateau, not
monotone across the whole range (the geometry × Re confound below applies).

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

- **Full pipeline re-run done (2026-06-25).** All six recordings were
  re-converted from `.cine` and reprocessed end-to-end on the apex-gap
  calibration, with Mode A/B anchored on the calibrated gap and flash-cleaned.
  The earlier-pass results (and the linear rescale that preceded this) are
  superseded; current outputs are a single consistent pass.
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
