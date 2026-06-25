# BLADE Gliding-Arc Analysis — Methods (11MAY26 N₂ series)

> Companion to `METHODS.md`. That document describes the **CO₂ J-electrode**
> experiment; **this** document describes the **N₂ blade-electrode** experiment
> recorded on 11 May 2026. The two share image-analysis machinery but differ in
> gas, geometry, calibration, and reactor orientation, so their parameters must
> not be cross-applied.
>
> For a single write-up that factors the shared core out once and places the two
> geometries side by side, see **`METHODS_UNIFIED.md`**.
>
> Last revised: 2026-06-21.
> Pipeline modules: `run_11may26.py`, `analyze_blade.py`, `event_velocity.py`
> (shared), `arc_shape.py` (shared), `uncertainty.py` (shared),
> `visualize_blade.py`, `qa_blade.py`.

## 0. What changed since the May write-up

The detection/measurement code is unchanged, but the **shared** velocity stack
gained a systematic-uncertainty budget (`uncertainty.py`) that now applies to
the blade velocities automatically, because the blade path calls the same
`event_velocity.py`. The practical consequences for this dataset:

- Every event velocity now carries `vel_ms_err_stat / _sys / _total` (§5.1), not
  just the OLS slope SE.
- The fps invariant (§5) is now explicitly checked by `test_j_geometry.py`'s
  dimensional self-check, which guards both geometries against the historic
  `true_fps / 200000` mis-scaling.
- The Reynolds *definition* used for the CO₂ sweep also applies here (§9), but
  the 11MAY26 N₂ data is a **single flow/nozzle condition**, so there is no
  cross-condition Reynolds comparison to make — only the single Re value, for
  context.

Nothing in §§1–4 (experiment, pipeline, detection, measurements) changed.

## 1. Experiment

| Parameter | Value | Source |
|---|---|---|
| Gas | N₂ | filename |
| Flow rate | 1.5 SLPM | filename |
| Nozzle | 0.6 mm | filename |
| Electrode opening angle | 25° (blade wedge) | filename, confirmed from frames |
| Frame rate | 200,000 fps (5 µs/frame) | `.cine` header |
| Exposure | 4.5 µs | `.cine` header |
| Frame size | 256 × 256 px | `.cine` header |
| Bit depth | 12-bit (0–4095), converted to 8-bit | `.cine` header |
| Spatial calibration | **8.8 px/mm** (44 px = 5 mm) | `Calibrate/calibration.txt` |
| Field of view | ≈ 29 mm × 29 mm | derived |

Two recordings differ only in reactor orientation:

| File | Frames | Electrode/apex | Arc extends | Arc front pixel |
|---|---|---|---|---|
| `..._downward.cine` | 291,353 | top of frame | downward | bottommost (row argmax) |
| `..._upward.cine` | 459,446 | bottom of frame | upward | topmost (row argmin) |

### Physical picture

Two blade electrodes form a wedge (a "V"), closest at the **apex** (minimum gap
≈ 2.5–5 mm) and diverging to a wider gap. The filename's "25deg" is the per-blade
**half-angle**: the auto-detected **full** opening angle is ≈ 38–47°
(≈ 2 × 25°), consistent with the half-angle convention used in `METHODS.md`. The arc breaks down near
the apex, then **glides** along the diverging blades — driven by the N₂ jet and
the electric field — forming an **inverted-U arch** whose two feet stay rooted on
the blades while the arch grows. The arc then extinguishes and re-nucleates,
giving the periodic discharge cycle visible in the bright-pixel-count trace.

Gliding-arc theory recognises two re-nucleation modes (Czernichowski 1994; Zhang
et al. 2020), both handled by `event_velocity.py`:
- **Mode A** — full extinction and fresh re-ignition at the apex (minimum gap).
- **Mode B** — back-breakdown / reconnection at an intermediate position.

## 2. Pipeline

```
run_11may26.py (orchestrator)
  Stage 1  convert    .cine → 8-bit TIFF chunks
                      • percentile normalisation (0.35% clip) from sampled frames
                      • per-chunk bad-pixel correction (correct_bad_pixels_stack)
  Stage 2  analyze    analyze_blade.py per chunk → per-frame measurements.csv
  Stage 3  merge      concatenate chunk CSVs with global frame numbering
  Stage 4  events     event_velocity.py → events, velocities, Mode A/B
  Stage 5  visualize  visualize_blade.py → paper-ready figures
```

`qa_blade.py` is run **alongside** Stage 2 on representative chunks to verify
detection quality before committing to a full run.

> **Same code as the J pipeline.** `analyze_blade.py`, `event_velocity.py`,
> `arc_shape.py` and `uncertainty.py` are exactly the modules `run_j_co2.py`
> imports. The blade run uses `PX_PER_MM = 8.8`, `BRIGHT_THRESH = 100`, the
> auto-probed per-file fps, and `--local-vel-n 10` (defaults in
> `run_11may26.py`). The J path overrides these with its own calibration and
> adds the finite-segment attachment gate; everything downstream is identical.

### 2.1 Bad-pixel correction is mandatory

The raw sensor has a fixed grid of ~128 stuck/hot pixels that saturate every
frame. `correct_bad_pixels_stack` (Drop Impact) removes them during conversion.
Any path that reads raw `.cine` frames without this step will register false
detections on the hot-pixel grid — verified directly (`qa_blade.py` reports
`persistent_saturated_px`, which is 0 on properly converted chunks).

## 3. Arc detection (`analyze_blade.binarize`)

Detection is a **fixed DN threshold**, not adaptive Otsu, because the arc is a
small bright object on a near-black field and per-frame adaptive thresholds
amplify noise on the dark inter-stroke frames. Steps:

1. Dark guard: skip frames whose max pixel < `dark_guard` (35 DN).
2. Threshold: pixels > `bright_thresh` (**100 DN** for 11MAY26) are arc
   candidates; reject if fewer than `min_arc_px`.
3. Morphological closing (disk radius 3) to bridge small gaps.
4. Keep the single largest connected component; reject if its area is outside
   `[min_arc_px, max_arc_px]` (rejects speckle and noise floods).

On the upward smoke-test chunk this gives ≈ 64 % of frames detected.

> **Threshold provenance.** 100 DN was chosen from `threshold_qa_11may26.py` on
> this dataset. The J path resolves its threshold per-recording (CLI >
> `j_geometry.json` > `threshold_suggested.txt` > 210 fallback); the blade path
> uses the single dataset-wide constant because it is one operating condition.

## 4. Per-frame measurements (`analyze_blade.measure_frame`)

For each detected frame: arc mask area; **skeleton** (Zhang–Suen) and ordered
centreline path; **arc length** (skeleton pixel count / px-per-mm); **FWHM
width** sampled perpendicular to the local tangent (a proxy for channel
diameter **plus motion blur** — see §1, 4.5 µs exposure); **brightness**
(mean / peak / integrated, on the per-frame min–max normalised image);
**centroid**; **bounding box**; **curvature** (mean absolute angular change
along the skeleton); the two **skeleton endpoints** (the arc feet); and the
**arc-front tip** (extreme pixel in the propagation direction).

### Orientation convention

`arc_tip_row` is stored as **distance from the electrode**, increasing as the
arc extends, for **both** recordings. For the upward video (electrodes at the
bottom) the raw row is flipped (`H-1-row`) so that downstream code
(`event_velocity.py`) sees one consistent advancing-front axis regardless of
reactor orientation.

### Complete vs partial arcs

A physically meaningful bridging arc roots on **both** blades, tested by the
perpendicular distance of each skeleton foot to a stored blade line
(`--blade-lines`, tolerance `ELEC_TOL_PX`). `blade_lines.npy` can be produced two
ways:
- **Automatic (default, headless):** `detect_blade_lines.py` fits the two blade
  lines from the loci of arc feet across many clean-arch frames (deterministic
  left/right assignment + total-least-squares with outlier trimming). It runs
  with no user interaction and `run_11may26.py` invokes it automatically before
  Stage 2 whenever `blade_lines.npy` is missing.
- **Manual override:** the interactive `calibrate_electrodes.py` (click two
  points per blade) still writes the same file and takes precedence if present.

If no blade lines are available, the analyzer falls back to the widest-pair
heuristic (lower `arc_complete` accuracy). `qa_blade.py` independently reports a
calibration-free complete/partial split (both feet in the electrode band,
separated by ≥ the apex gap) as a cross-check.

> **Geometry contrast with the J path.** Here the electrodes are **straight
> infinite blade lines** (`blade_lines.npy`); attachment is tested by
> perpendicular distance to those lines. The J path instead uses **finite**
> hooked segments (`blade_lines.npy` from `calibrate_j_electrodes.py`, or curved
> `electrode_edges.npz` from `detect_j_hooks.py`) and an explicit
> `--finite-segments` gate requiring the feet to land on the real segment near
> the apex. This is the single most important methodological difference between
> the two geometries.

## 5. Events, velocity & modes (`event_velocity.py`)

Operates on the merged per-frame CSV. Unchanged from the J-electrode pipeline
except for `--px-per-mm 8.8 --fps 200000`:

> **fps invariant.** fps is per-recording (auto-detected by `probe_cine`) and
> must thread into `event_velocity.py`, `qa_restrike.py`, and
> `fourier_analysis.py`. A past regression mis-scaled every velocity and
> frequency by `true_fps / 200000`; this is now guarded by the dimensional
> self-check in `test_j_geometry.py` (synthetic 200 kHz vs 100 kHz → velocity
> ratio exactly 2.0), which covers both geometries.

- **Event detection** is data-driven: a new primary event begins when the tip
  returns to within 15 % of the effective arc range above the electrode row, or
  after a detection dropout > 500 frames. Intermediate retreats > 50 % of the
  current extension are flagged as sub-events.
- **Velocity** is the slope of a least-squares fit to the **advancing-front**
  (monotonic-envelope) tip trajectory, converted with
  `vel = slope · FPS / px_per_mm / 1000` (m/s). R² reports linearity. A
  sliding-window OLS gives a time-resolved local velocity, using a centered
  window of **N = 10** advancing-front frames (±5 neighbours, `--local-vel-n`).
  N = 10 is the canonical setting for this experiment: the narrower N = 5 window
  spans only ~25 µs and produced short-baseline spurious velocity spikes; N = 10
  (~50 µs) suppresses them at a modest cost in time resolution. This local
  velocity feeds `local_vel_ms`, the per-event `local_vel_*` statistics, the
  mode comparisons, and the 2-D velocity map. (Note: `METHODS.md` also states
  N = 10, and the J sensitivity sweep confirms stability for N ≥ 7.)
- **Mode A / B** from the dimensionless normalised nucleation position
  `(tip_row_start − electrode_row) / L_eff_px`, threshold 0.15 (self-consistent
  with the event detector).

Each event row in `event_summary.csv` carries **both** characterizations:
- `vel_ms` — the whole-event advancing-front regression (the **average** rate;
  useful for Mode A vs B comparisons).
- `local_vel_median / mean / p10 / p90 / peak (p95) / max / std`,
  `n_local_frames` — the **time-resolved** characterization, aggregated from the
  per-frame local velocity. This is the richer record: the full per-frame profile
  is in `events/event_NNNN_dynamics.csv`, and the average is recoverable from it.
  For multi-phase events the two diverge sharply (e.g. a long event with
  mid-glide pauses has a low `vel_ms` but a much higher `local_vel_peak`),
  which is the signal that a single average is insufficient.

### 5.1 Velocity uncertainty (`uncertainty.py`) — applies to the blade data

`event_velocity.py` imports `uncertainty.py` and writes, for every event:

| Column | Meaning |
|---|---|
| `vel_ms_err` | legacy: OLS slope SE only (statistical) |
| `vel_ms_err_stat` | statistical (= the above) |
| `vel_ms_err_sys` | systematic: px/mm ⊕ fps in quadrature |
| `vel_ms_err_total` | `√(stat² + sys²)` — **the value to report** |

For `v = slope · FPS / px_per_mm / 1000`:

```
(σ_v / |v|)²  =  (σ_v_stat / |v|)²  +  (σ_fps / FPS)²  +  (σ_pxmm / px_per_mm)²
```

with relative defaults `SIGMA_FPS_REL = 0.1 %` and `SIGMA_PXMM_REL = 5 %`
(overridable via `--pxmm-rel-err` / `--fps-rel-err`). For the blade calibration
(8.8 px/mm = 44 px / 5 mm), the 5 % px/mm default is the appropriate floor; if a
tighter calibration audit exists, override it. Frequencies obey
`σ_f / f = σ_fps / FPS`, and `fourier_analysis.py` annotates peaks as `f ± σ_f`.
`recompute_uncertainty.py` can add the three columns to an existing
blade `_results/` directory without re-running the pipeline.

### 5.2 Arc-shape / normal velocity (`arc_shape.py`) — shared

The arc is a deformable filament. `arc_shape.py` (the same module the J path
uses) resamples the skeleton to N arc-length-equally-spaced points and
decomposes each point's frame-to-frame displacement into components **tangent**
(along-arc) and **normal** (lateral sweep) to the local arc direction, on
complete-arc frames only. Output `arc_shape_velocities.csv`. This is the natural
way to separate the gliding motion from the lateral jet-driven sweep on the
blade geometry.

## 6. QA (`qa_blade.py`)

Headless. Reuses `analyze_blade`'s exact detection operators. For one chunk or
representative segment it reports detection %, complete/partial %, cycle
structure, the auto-detected electrode band, apex and maximum inter-blade gap in
mm, the persistent-saturated-pixel count, and a montage of annotated frames, plus
a JSON summary. Run it on a few segments spanning each recording before a full
analysis.

## 7. Outputs

| File | Description |
|---|---|
| `measurements.csv` | per-frame metrics (merged across chunks) |
| `event_summary.csv` | one row per discharge event incl. velocity (+ σ columns), R², Mode A/B |
| `measurements_all_events.csv` | per-frame data with event IDs and local velocity |
| `events/` | per-event frame and dynamics CSVs |
| `velocity_map.png` / `.npy` | 2-D mean arc-tip-speed map |
| `timeseries.png`, `event_*.png`, `fig*.png` | diagnostic & paper figures |
| `qa/<chunk>_qa.png` / `.json` | QA report per chunk |
| `run_metadata.json` | provenance (fps, px/mm, threshold, σ budget, git hash, timestamp) when run via the shared orchestrator |

## 8. Robustness tooling (shared; applicability to the blade data)

The June robustness tooling lives in shared modules; what transfers to the
single-condition N₂ dataset:

- **Uncertainty budget** (`uncertainty.py`) — fully applicable; see §5.1.
- **fps-threading audit** (`verify_fps_threading.py`) — applicable; run it on
  the blade `_results/` to confirm the probed fps threads downstream.
- **Filter audit** (`filter_audit_montage.py`) — applicable; produces a
  reproducible accepted/rejected frame montage for the blade detection.
- **Sensitivity sweep** (`sensitivity_sweep.py`, Tier A) — applicable to the
  blade's `measurements_all_events.csv` for `local_vel_n` and flash-width
  stability; not yet run on this dataset.
- **Reynolds comparison** (`reynolds_comparison.py`) — **not applicable as a
  comparison**: it aggregates across the CO₂ nozzle×flow sweep, and the blade
  data is a single condition (see §9).
- **Repeat exclusion** (`sweep_utils.py`) and **long-recording probe**
  (`long_video_probe.py`) — the blade recordings are long (291k / 459k frames);
  if only a prefix has been analysed, run `long_video_probe.py` first to confirm
  the prefix is representative before trusting it (this is exactly the issue that
  bit the J 1.2 mm recording).

## 9. Reynolds number (single condition — context only)

The same definition as the CO₂ sweep applies, with N₂ properties:
characteristic length L = nozzle diameter d = 0.6 mm; characteristic velocity
U = bulk gas velocity `v_bulk = Q / (π/4 · d²)` (**not** arc velocity);
`Re = ρ · v_bulk · d / μ`, with N₂ at ≈ 295 K, 1 atm
(ρ ≈ 1.16 kg·m⁻³, μ ≈ 1.78 × 10⁻⁵ Pa·s).

For Q = 1.5 SLPM through d = 0.6 mm, `v_bulk ≈ 90 m·s⁻¹`, giving **Re ≈ 3 × 10³**
and **Ma ≈ 0.25** (a_N₂ ≈ 350 m·s⁻¹). Two honest caveats: (i) "SLPM" is a
*standard*-condition flow, so the actual nozzle-exit velocity differs from the
ambient-density estimate; and (ii) this is a **single** operating point — there
is no swept-condition Reynolds trend to report for the N₂ blade experiment, only
this number for placing it relative to the CO₂ sweep (whose Re spanned
≈ 3.3 × 10³ – 9.8 × 10³).

## 10. Known limitations

- **Motion blur** at 4.5 µs exposure inflates FWHM width ∝ tip speed.
- **Single-distance calibration** (8.8 px/mm); no lens-distortion correction.
  The 5 % px/mm uncertainty floor (§5.1) reflects this.
- **Per-frame normalisation** means cross-frame brightness comparisons must use
  `integrated_intensity`, not `mean_intensity`.
- Only **representative segments** have been analysed so far; no full-file run.
  Given the recording lengths (291k / 459k frames), run `long_video_probe.py`
  before trusting any prefix (§8).
- **Single operating condition** — no Reynolds or cross-condition comparison is
  possible from this dataset alone (§9).
- "OA" in the filename is **not yet expanded**.

## 11. References

Czernichowski 1994 (*Pure Appl. Chem.* 66:1301); Zhang et al. 2020 (*IET High
Voltage* 5:386); Fridman, Gutsol & Kennedy 2005; Otsu 1979; Zhang & Suen 1984;
Lemmon & Jacobsen 2004 (gas transport properties); CRC Handbook 90th ed. (gas
densities).
