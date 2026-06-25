# Plasma Arc Analysis — Unified Methods (both electrode geometries)

> **Purpose.** A single, paper-oriented description of the gliding-arc image
> analysis used for **both** reactor geometries, with the shared core stated
> once and the geometry-specific pieces placed side by side. Use this to talk
> through the methods end to end or to draft a manuscript Methods section for
> either experiment.
>
> Last revised: 2026-06-21.
> Detailed per-experiment companions: `METHODS.md` (CO₂ J-electrode) and
> `METHODS_BLADE.md` (N₂ blade). Where this document and a companion disagree,
> the companion is authoritative for its experiment.

---

## 1. Scope: two experiments, one analysis stack

Two high-speed discharge experiments are analysed with a **common** image-
analysis pipeline. They differ only in gas, electrode geometry, calibration, and
operating sweep; every quantitative measurement (detection, skeletonisation,
event segmentation, velocity, modes, uncertainty) is computed by the **same**
code.

| | **CO₂ J-electrode** | **N₂ blade** |
|---|---|---|
| Gas | CO₂ | N₂ |
| Electrodes | hooked **finite** J-segments | straight **infinite** blade lines (25° half-angle wedge) |
| Operating points | sweep: nozzle 0.4/0.8/1.2 mm × flow 1000/1500 mL·min⁻¹, with a repeat | single: 0.6 mm, 1.5 SLPM, two reactor orientations |
| Frame rate | per-recording, auto-detected (≈ 200 kHz) | 200 kHz (5 µs/frame) |
| Calibration | per-recording px/mm from ≈ 2.4 mm apex gap | 8.8 px/mm (44 px = 5 mm) |
| Orchestrator | `run_j_co2.py` | `run_11may26.py` |
| Companion doc | `METHODS.md` | `METHODS_BLADE.md` |

**Shared modules** (identical code for both): `analyze_blade.py` (detection +
per-frame measurement), `event_velocity.py` (events, velocity, modes,
uncertainty), `arc_shape.py` (normal/tangential decomposition), `uncertainty.py`
(systematic budget), `build_velocity_map.py`, `fourier_analysis.py`. The two
orchestrators differ only in the geometry/calibration they pass and, for the J
path, the finite-segment attachment gate (§5).

---

## 2. Physical picture (common)

In both reactors two electrodes form a narrow gap (the **apex**) that widens
downstream. The arc breaks down at the apex, then the gas jet and electric field
drive it to **extend/glide** along the diverging electrodes, forming an arch
whose feet stay rooted on the electrodes. The arc then extinguishes and
**re-nucleates**, producing a periodic discharge cycle. Gliding-arc theory
distinguishes two re-nucleation modes, both classified by the pipeline:

- **Mode A** — full extinction, fresh re-ignition at the apex (minimum gap).
- **Mode B** — back-breakdown / reconnection at an intermediate position.

(Czernichowski 1994; Zhang et al. 2020; Fridman et al. 2005.)

---

## 3. Pipeline architecture (common)

```
orchestrator (run_j_co2.py | run_11may26.py)
  Stage 1  convert    .cine → 8-bit TIFF chunks
                      • percentile-clip normalisation from sampled frames
                      • per-chunk bad-pixel correction (mandatory; see §4)
  Stage 2  analyze    analyze_blade.py per chunk → per-frame measurements.csv
  Stage 3  merge      concatenate chunk CSVs with global frame numbering
  Stage 4  events     event_velocity.py → events, velocities, Mode A/B, σ
  Stage 5  fourier    fourier_analysis.py → cycle spectra (peaks ± σ_f)   [J path]
  Stage 6  visualize  build_velocity_map.py + visualize_blade.py → figures
```

QA (`qa_blade.py`) is run on representative chunks before a full run to confirm
detection quality. All resolved parameters and provenance are written to
`run_metadata.json` (§11).

---

## 4. Image conversion & preprocessing (common)

`.cine` frames (12-bit, 0–4095) are converted to 8-bit TIFF chunks with
percentile-clip normalisation estimated from sampled frames. **Bad-pixel
correction is mandatory**: the sensor carries a fixed grid of ~128 stuck/hot
pixels that saturate every frame; `correct_bad_pixels_stack` removes them during
conversion. Any path reading raw frames without this step registers false
detections on the hot-pixel grid (`qa_blade.py` reports `persistent_saturated_px`,
which is 0 on correctly converted chunks).

**Intensity convention.** Quantitative brightness is measured on **raw DN**
(0–255); cross-frame comparisons use `integrated_intensity`, never a per-frame
normalised value.

---

## 5. Electrode geometry & spatial calibration

This is the **principal divergence** between the two experiments. The shared
requirement is that a physically meaningful bridging arc roots on **both**
electrodes; the geometry determines how "rooted on the electrode" is tested and
how px/mm is set.

### Shared: px/mm from a known gap, and the apex anchor

Spatial scale comes from a known physical gap measured in pixels. The apex gap
also anchors the electrode-gap **flash-cutoff row** and the Mode-A/B reference
position. Velocity uncertainty includes a calibration term derived from this gap
(§10).

### 5a. J-electrode (CO₂): finite hooked segments

Each electrode is a **finite** curved J-segment. Two representations are
supported: straight finite blades (`blade_lines.npy` from the interactive
`calibrate_j_electrodes.py`) or curved edges (`electrode_edges.npz` from
`detect_j_hooks.py`). Attachment is enforced by an explicit **finite-segment
gate** (`--finite-segments`): each arc foot must land on the real electrode
segment near the apex, not on an imaginary infinite extension. This gate is what
keeps incomplete arcs out of the event stream.

px/mm is **per-recording**, written to `j_geometry.json`, resolved as
`--px-per-mm (CLI) > j_geometry.json > tiff/calibration.txt > 9.167 (fallback)`,
from the ≈ 2.4 mm apex gap. *Known provenance issue:* the 1.2 mm recording's
`j_geometry.json` stores `px_per_mm = 8.75` while the stored gap implies 11.17;
runs use a `--px-per-mm 11.2` override (numbers correct, stored field stale).

### 5b. Blade (N₂): straight infinite lines

Each electrode is a straight blade modelled as an **infinite line**. Attachment
is tested by the perpendicular distance of each skeleton foot to a stored blade
line (`blade_lines.npy`, tolerance `ELEC_TOL_PX`). The lines are fit
automatically by `detect_blade_lines.py` (total-least-squares on the loci of arc
feet across clean-arch frames) or overridden interactively by
`calibrate_electrodes.py`. px/mm is the single dataset constant **8.8 px/mm**
(44 px = 5 mm). The "25deg" filename label is the per-blade **half-angle**; the
auto-detected full opening angle is ≈ 38–47°.

### Orientation handling (common)

`arc_tip_row` is stored as **distance from the electrode**, increasing as the arc
extends. For inverted (upward) recordings the raw row is flipped (`H-1-row`) so
downstream code sees one consistent advancing-front axis regardless of reactor
orientation.

---

## 6. Arc detection (common: fixed DN threshold)

Detection uses a **fixed DN threshold**, not adaptive Otsu — the arc is a small
bright object on a near-black field, and per-frame adaptive thresholds amplify
noise on dark inter-stroke frames. Steps (`analyze_blade.binarize`): dark guard
(skip frames with max < ~35 DN) → threshold at `bright_thresh` → morphological
closing (disk r = 3) → keep the single largest connected component within
`[min_arc_px, max_arc_px]`.

**Threshold provenance differs by experiment:**

- **J (CO₂):** resolved per-recording,
  `CLI > j_geometry.json > tiff/threshold_suggested.txt > 210 (fallback)`. The
  210 fallback (and per-recording values, e.g. 160 / 220 / 235) come from
  arc-vs-plume local-intensity QA (`qa_arc_local_intensity_j.py`): arc core
  saturates near 255, plume below ~150. `max_arc_px` is raised above the blade
  default because the wide high-flow CO₂ arc can exceed it.
- **Blade (N₂):** a single dataset-wide constant **100 DN**, from
  `threshold_qa_11may26.py`, since it is one operating condition.

Flash/saturation rejection (`arc_filter.py`) applies the unified validity
criteria: a gap-row flash cutoff, an `arc_width_px` leg, and an effective-width
leg (`eff_width = area / length`, which fires on disconnected double-strikes).
`filter_audit_montage.py` produces a reproducible accepted/rejected montage to
confirm the criteria behave.

---

## 7. Per-frame measurements (common, `analyze_blade.measure_frame`)

For each detected frame: arc-mask **area**; **skeleton** (Zhang–Suen) and ordered
centreline (depth-first walk on the 8-connected graph; endpoints = degree-1
pixels); **arc length** (skeleton px / px-per-mm); **FWHM width** sampled
perpendicular to the local tangent (channel-diameter proxy **plus** motion blur
at the few-µs exposure); **brightness** (mean / peak / integrated, raw DN);
**centroid**; **bounding box**; **curvature** (mean |Δθ| along the skeleton); the
two skeleton **endpoints** (arc feet); and the **arc-front tip** (extreme pixel
in the propagation direction).

---

## 8. Events, velocity & modes (common, `event_velocity.py`)

Operates on the merged per-frame CSV; the only per-experiment inputs are
`--px-per-mm` and `--fps`.

> **fps invariant.** fps is per-recording (auto-detected) and must thread into
> `event_velocity.py`, `qa_restrike.py`, and `fourier_analysis.py`. A historic
> regression mis-scaled velocity and frequency by `true_fps / 200000`; this is
> now guarded by a dimensional self-check (`test_j_geometry.py`: synthetic
> 200 kHz vs 100 kHz → ratio exactly 2.0) covering both geometries.

**Event detection.** A new primary event begins when the tip returns to within
15 % of the effective arc range above the electrode row, or after a dropout
> 500 frames. The electrode row is the 5th-percentile `arc_tip_row` of
complete-arc frames; `L_eff_px = p90(tip) − electrode_row`. Retreats > 50 % of
the current extension are flagged as sub-events.

**Velocity.** The slope of an OLS fit to the **advancing-front** (monotonic-
envelope) tip trajectory:

```
v = slope [px/frame] × FPS / px_per_mm / 1000   (m/s)
```

R² reports linearity. A centered sliding-window OLS over **N = 10** advancing-
front frames (±5; `--local-vel-n`) gives a time-resolved local velocity
(equivalent to a Savitzky–Golay first derivative). **N = 10 is canonical**:
N = 5 spans only ~25 µs and admits short-baseline spikes; the J sensitivity
sweep confirms the headline metric is stable for N ≥ 7 (§11).

**Mode A / B.** From the dimensionless normalised nucleation position
`(tip_row_start − electrode_row) / L_eff_px`, threshold 0.15 (self-consistent
with the event detector). The authoritative record is `re_strikes_qa.csv`.

**Output.** `event_summary.csv` carries both `vel_ms` (whole-event average rate)
and time-resolved `local_vel_{median,mean,p10,p90,peak,max,std}`; the full
per-frame profile is in `events/event_NNNN_dynamics.csv`. The two diverge for
multi-phase events — the signal that a single average is insufficient.

---

## 9. Arc-shape tracking & normal velocity (common, `arc_shape.py`)

The arc is a deformable filament: the tip regression captures only the leading
edge. `arc_shape.py` resamples the skeleton to N arc-length-equally-spaced points
(point 0 = electrode-proximal) and, for each consecutive frame pair within an
event, decomposes each point's displacement into **tangential** (along-arc) and
**normal** (lateral sweep) components relative to the local arc direction. Only
complete-arc frames are used. Output `arc_shape_velocities.csv` with
`norm_pos = seg_idx / (N − 1)`. This separates gliding motion from jet-driven
lateral sweep for either geometry.

---

## 10. Uncertainty budget (common, `uncertainty.py`)

A single source of truth for the systematic budget the velocity stack previously
lacked: the OLS slope SE is only the random fitting error; reported velocities
must also carry the **px/mm** and **fps** systematics. For
`v = slope · FPS / px_per_mm / 1000`:

```
(σ_v / |v|)²  =  (σ_v_stat / |v|)²  +  (σ_fps / FPS)²  +  (σ_pxmm / px_per_mm)²
```

| Quantity | Default | Justification |
|---|---|---|
| `SIGMA_FPS_REL` | 0.1 % | Phantom HS crystal ≈ 50–100 ppm; conservative for trig-mode |
| `SIGMA_PXMM_REL` | 5 % | √2 × 1 px click / ≈ 26.56 px apex gap ≈ 5.3 % |

Both are relative 1-σ, overridable per-recording (`--pxmm-rel-err`,
`--fps-rel-err`) and logged in `run_metadata.json`.
`pxmm_uncertainty_from_calibration()` can derive σ_pxmm from the measured gap.
`event_velocity.py` writes `vel_ms_err_stat`, `vel_ms_err_sys`,
`vel_ms_err_total` (**report the total**); the legacy `vel_ms_err` is retained =
stat only. Frequencies obey `σ_f / f = σ_fps / FPS`; Fourier peaks are annotated
`f ± σ_f`. `recompute_uncertainty.py` adds these columns to existing results
without re-running — point estimates are unchanged; only the error bars grow
(≈ 6× on the J target, where the calibration term had been omitted entirely).

This budget applies identically to both geometries.

---

## 11. Aggregate & robustness analyses

These operate across or within recordings; their applicability differs because
only the CO₂ experiment is a multi-condition sweep.

**Sensitivity sweep** (`sensitivity_sweep.py`, Tier A; post-hoc on
`measurements_all_events.csv`). A parameter is UNSTABLE if shifting it moves the
median by more than `σ_floor = √(σ_pxmm² + σ_fps²) × median_v`. On the J target:
`local_vel_n` stable for N ≥ 7 (N = 5 UNSTABLE); `flash_arc_width` and
`flash_eff_width` stable across the swept range. Applicable to the blade data but
not yet run there.

**Reynolds comparison** (`reynolds_comparison.py`) — **CO₂ sweep only.** Defined
with L = nozzle diameter d, U = bulk gas velocity `v_bulk = Q / (π/4 · d²)`
(*not* arc velocity), `Re = ρ v_bulk d / μ`, CO₂ at 295.15 K / 101.325 kPa
(ρ = 1.815 kg·m⁻³, CRC 90th; μ = 1.48 × 10⁻⁵ Pa·s, Lemmon & Jacobsen 2004).
Repeat-aware error bars on both axes. Caveats stated in every caption: (i)
changing d changes Re **and** geometry (confound); (ii) Ma reaches ≈ 0.74 at the
small-nozzle corner, so the reported Re is incompressible-equivalent. Coverage
(median arc velocity): Re ≈ 3.3×10³ → 4.9×10³ → 6.5×10³ → 9.8×10³ giving
4.21 → 7.24 → 9.47 → 10.01 m/s. The N₂ blade is a **single** condition
(Re ≈ 3×10³, Ma ≈ 0.25), so no Reynolds *trend* exists for it — only the single
value, for context.

**Repeat handling** (`sweep_utils.py`). The 0.4 mm/1500 mL CO₂ condition has two
recordings disagreeing by ≈ 3.9 m/s; a matched-settings diagnostic showed
+3.35 m/s of genuine run-to-run physics (not threshold/calibration), so the
repeat is excluded from cross-recording aggregates via an `.excluded_from_sweep`
sentinel while its per-recording outputs remain valid.

**Temporal drift & long recordings.** `temporal_velocity_profiles.py` and
`warmup_time.py` test within-recording drift. Long `.cine` files must be probed
(`long_video_probe.py`) before trusting a prefix — the J 1.2 mm recording showed
a stable +0.34 m/s·s⁻¹ drift only after full reprocessing (669 events vs 74 in
the prefix). The blade recordings are long (291k / 459k frames) and only
representative segments have been analysed — the same probe applies before any
prefix is trusted.

---

## 12. Provenance & reproducibility (common)

`run_metadata.json` records, per run: fps, px/mm + source, `bright_thresh` +
source, geometry mode + finite-segment flags, gap_row, `local_vel_n`, the
σ_pxmm/σ_fps budget, git hash, and UTC timestamp. `test_j_geometry.py` carries
the geometry-derivation and fps dimensional self-checks (pytest-discoverable).
`verify_fps_threading.py` audits that the probed fps threads downstream across
recordings.

---

## 13. Parameter reference (side by side)

| Parameter | CO₂ J-electrode | N₂ blade |
|---|---|---|
| Detection threshold | per-recording (CLI > json > txt > **210**) | **100 DN** (dataset constant) |
| px/mm | per-recording (≈ 2.4 mm apex gap); fallback 9.167 | **8.8** (44 px = 5 mm) |
| Electrode model | finite J-segments + `--finite-segments` gate | infinite blade lines + perp-distance test |
| Geometry file | `j_geometry.json` + `blade_lines.npy`/`electrode_edges.npz` | `blade_lines.npy` |
| fps | per-recording, auto-detected | 200 000 (5 µs/frame) |
| `local_vel_n` | 10 (stable ≥ 7) | 10 |
| Mode A/B threshold | 0.15 | 0.15 |
| σ_pxmm / σ_fps | 5 % / 0.1 % (override per-recording) | 5 % / 0.1 % |
| Operating points | sweep (4 conditions + 1 repeat) | single (2 orientations) |

---

## 14. Known limitations (common unless noted)

- **Motion blur** at the few-µs exposure inflates FWHM width ∝ tip speed.
- **Single-distance calibration**; no lens-distortion correction (the 5 % px/mm
  floor reflects this).
- **Per-frame normalisation** (display/legacy) means cross-frame brightness must
  use raw-DN `integrated_intensity`.
- **CO₂-specific:** Mach > 0.3 at the small-nozzle corner (incompressible-
  equivalent Re); 0.4 mm/1500 mL effectively single-recording after the repeat
  exclusion; the 1.2 mm `j_geometry.json` px/mm field is stale (8.75 vs 11.17).
- **Blade-specific:** single operating condition (no Reynolds trend); only
  representative segments analysed so far; "OA" in the filename not yet expanded.
- **Both:** Tier-B (re-detection) threshold sweep not run.

---

## 15. References

1. Czernichowski, A. (1994). *Pure Appl. Chem.* **66**(6), 1301–1310.
2. Zhang, H., et al. (2020). *IET High Voltage* **5**(4), 386–392.
3. Fridman, A., Gutsol, A., & Kennedy, L. A. (2005). *Non-Equilibrium Air Plasmas
   at Atmospheric Pressure.* IOP Publishing.
4. Otsu, N. (1979). *IEEE Trans. Syst. Man Cybern.* **9**(1), 62–66.
5. Zhang, T. Y., & Suen, C. Y. (1984). *Commun. ACM* **27**(3), 236–239.
6. van der Walt, S. et al. (2014). "scikit-image." *PeerJ* **2**, e453.
7. Lemmon, E. W., & Jacobsen, R. T. (2004). *Int. J. Thermophys.* **25**, 21–69
   (gas transport properties).
8. CRC Handbook of Chemistry and Physics, 90th ed. (2009). CRC Press
   (gas densities).

---

*End of document.*
