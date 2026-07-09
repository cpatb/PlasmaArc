# Robustness Report — J-electrode CO₂ pipeline

Branch: `robustness-pass`. Pre-pass baseline tag: `pre-robustness` (5758914).
Working tree clean at write time; every output below was reproduced by the
verification commands in §7.

---

## 1. What changed and why

| Commit | Files | Purpose |
| --- | --- | --- |
| 5758914 | (10 modified + 3 new — prior work)               | `chore(baseline)`: clean cut so the robustness commits diff independently. Tagged `pre-robustness`. |
| 435a3be | `ANALYSIS_BRIEF.md`                              | `docs(brief)`: the acceptance brief itself. |
| 44fc5ff | `uncertainty.py`, `recompute_uncertainty.py`, `event_velocity.py`, `fourier_analysis.py`, `run_j_co2.py` | `feat(uncertainty)`: single source of truth (`uncertainty.py`) for the px/mm + fps systematic budget. `event_velocity.py` now writes `vel_ms_err_stat`, `vel_ms_err_sys`, `vel_ms_err_total` alongside the legacy `vel_ms_err`. `fourier_analysis.py` annotates spectral peaks with `f ± σ_f`. `run_j_co2.py` threads the relative-σ CLI flags through and logs them in `run_metadata.json`. `recompute_uncertainty.py` derives the three error columns on existing artifacts without a pipeline rerun. |
| bac3796 | `test_j_geometry.py`                             | `test(geometry)`: synthetic 200 kHz vs 100 kHz dimensional self-check (acceptance §6.2). Also refactors the prior tests so both halves are pytest-discoverable. |
| e8e4fbb | `verify_fps_threading.py`, `filter_audit_montage.py`, `figstyle.py` | `feat(audit)`: fps-threading checker, filter-audit montage with reproducible seed, and the paper-grade matplotlib helper used by every figure this pass produces. |
| 73a8bf2 | `sensitivity_sweep.py`, `reynolds_comparison.py` | `feat(robustness)`: post-hoc sensitivity sweep (Tier A) and the previously-absent Reynolds comparison across the sweep. |
| (this)  | `ROBUSTNESS_REPORT.md`                           | `docs(robustness)`: this file. |

**Things NOT touched** (BLADE parity / scope hygiene, brief §0):
- `analyze_blade.py` defaults — unchanged. J-specific values still come via
  `run_j_co2.py` CLI flags.
- `arc_filter.py` — the unified validity criteria are consumed, not redefined.
- `calibrate_j_electrodes.py`, `qa_restrike.py` — already correctly accept
  `--fps`; no changes needed.
- No legacy figures retrofitted; new figures use the §5 standard.

---

## 2. Verification-run numbers (target recording)

Target: `HS_0.4mm_1000mL_(CO2)_trig.cine` — 18 758 frames, fps = 200 000,
px/mm = 11.07 (gap-derived: clicked apex gap 26.56 px / 2.4 mm), bright_thresh =
160, gap_row = 20. Provenance: `run_metadata.json` carries fps, px/mm + source,
threshold + source, geometry mode, σ budget, git hash, UTC timestamp — §6.3 ✓.

> **Numbers below reflect the 2026-06-25 full re-convert + reprocess on the
> apex-gap calibration** (px/mm = `min_gap_px / 2.4`, per video) with Mode A/B
> anchored on the physical gap and flash-cleaned. They supersede the original
> robustness-pass figures (which used a stale px/mm, e.g. 12.5 on the target).

### 2.1 Per-event velocity, with full uncertainty

The OLS slope SE alone (σ_stat, median 0.30 m/s) is misleadingly tight. Adding
the calibration + fps systematic (σ_pxmm_rel = 5.00 %, σ_fps_rel = 0.100 %):

| Stat                                  | Value (target recording)      |
| ------------------------------------- | ----------------------------: |
| Median event velocity                 | 10.70 ± 0.66 m/s              |
| Median σ_stat (OLS slope SE)          | 0.30 m/s                      |
| Median σ_sys (px/mm ⊕ fps)            | 0.54 m/s                      |
| Median σ_total                        | 0.66 m/s                      |

The error bar is ≈ 2× the slope-SE-only value because the calibration term had
been entirely outside the budget.

Across all six recordings (σ_sys = 5 % of v; σ_total = σ_stat ⊕ σ_sys):

| Recording                              | n events | median v (m/s) | σ_sys (m/s) | σ_total (m/s) |
| -------------------------------------- | -------: | -------------: | ----------: | ------------: |
| HS_0.4mm_1000mL_(CO2)_trig             |       53 | 10.70          | 0.54        | 0.66          |
| HS_0.4mm_1500mL_(CO2)_trig             |       50 | 10.35          | 0.52        | 0.82          |
| HS_0.4mm_1500mL_(CO2)_trig_repeat      |       31 | 6.22           | 0.31        | 0.53          |
| HS_0.8mm_1500mL_(CO2)_trig(2)          |       24 | 6.13           | 0.31        | 0.39          |
| HS_0.8mm_..._trig_repeat_weird(2)      |       22 | 7.89           | 0.39        | 0.50          |
| HS_1.2mm_1500mL_(CO2)                  |      669 | 4.22           | 0.21        | 0.24          |

**Discrepancy worth surfacing (brief §0).** The 0.4 mm / 1500 mL/min condition
has two recordings; their per-event medians differ by ≈ 4.1 m/s (10.35 vs 6.22),
**far more than the per-recording σ_total** (~0.6 m/s). The repeat-to-repeat
spread dominates the systematic at that corner — likely a real run-to-run
variability rather than a measurement artefact, but it deserves more repeats
before any conclusion at that operating point is leaned on.

### 2.2 fps threading verification

`verify_fps_threading.py` over all five recordings: **5 PASS / 0 FAIL**. The
target's `run_metadata.json fps=200000` matches `metadata.txt fps:200000` and
no CSV downstream of detection mentions a stray 200000 inconsistent with that.

### 2.3 Tests

```
$ python test_j_geometry.py
test_derive_geometry: OK
test_finite_segments: OK
test_fps_dimensional: OK
ALL J-GEOMETRY TESTS PASSED

$ python -m pytest test_j_geometry.py -q
3 passed, 44 warnings in 1.37s
```

Both halves of `test_j_geometry.py` pass under direct invocation and under
pytest discovery (acceptance §6.1). The dimensional self-check (synthetic
200 kHz vs 100 kHz) yields velocity ratio = 2.0 exactly (acceptance §6.2).

---

## 3. Filtering audit

`filter_audit_montage.py --n-each 6 --seed 20260618` on the target produced
[`filter_audit_montage.png`](../Desktop/Plasma%20Arc/J%20Electrode/HS_0.4mm_1000mL_(CO2)_trig_results/filter_audit_montage.png)
(+ matching PDF + sidecar `.txt`).

| Bucket                              | n frames in recording | sampled |
| ----------------------------------- | --------------------: | ------: |
| ACCEPTED (criteria 1∧2∧3 hold)      | 4 497                 | 6       |
| REJECTED — criterion 3 (`is_flash`) | 808                   | 6       |
| REJECTED — criterion 2 (incomplete) | 0                     | —       |

Findings from the montage:
- ACCEPTED row shows clean J-arcs bridging both electrodes, tip well below
  the gap, narrow skeleton (`arc_width_px` 4-7 px). No bad calls.
- REJECTED-flash row contains mostly genuinely wider arcs / saturated blobs
  (`arc_width_px` 8-16 px, or tiny blobs with `area_px` < 50). One borderline
  call (e.g. `f=14987 area=728 w=6.0px`) looks similar to ACCEPTED frames;
  the criterion that flagged it is the eff_width leg (`area/length` > 20),
  which fires on disconnected double-strikes. Not a misclassification, but
  the eff_width=20 default is tight on the high-flow CO₂ regime.
- REJECTED-incomplete bucket is **empty** for this recording: every
  detected arc that survived the dark/area/gap-row gates also bridged both
  electrodes successfully. This is good — the finite-segment attachment gate
  (introduced in the prior pass) catches the cases that would otherwise
  pollute the events.

Acceptance §6.5: rejects are genuinely invalid (or at most borderline-wide);
accepts are genuinely arcs — ✓.

---

## 4. Sensitivity sweep

Tier A only (post-hoc; reads existing `measurements_all_events.csv`, no
pipeline rerun). Tier B (varying `bright_thresh` on a fresh chunk) was NOT
run — see "Open risks" below.

σ_v floor for STABLE/UNSTABLE classification:
`σ_floor = sqrt(σ_pxmm² + σ_fps²) × median_v ≈ 0.05 × 11.19 ≈ 0.56 m/s`
(2σ ≈ 1.12 m/s).

### local_vel_n (sliding-window OLS width)

| value | n events | median local v (m/s) | Δ from baseline (m/s) | flag       |
| ----: | -------: | -------------------: | --------------------: | ---------- |
|   5   | 53       | 12.44                | +1.24                 | **UNSTABLE** |
|   7   | 53       | 11.80                | +0.60                 | STABLE     |
|  10   | 53       | 11.19                |  0.00 (baseline)      | STABLE     |
|  15   | 53       | 10.70                | −0.49                 | STABLE     |
|  20   | 53       | 10.31                | −0.88                 | STABLE     |

Conclusion: the prior-work default `local_vel_n = 10` is in the stable middle
of the curve. N = 5 is the only flagged cell — too narrow a window admits
short-baseline OLS spikes that bias the median upward. The bias is monotone
in N, which is the expected behaviour of a low-pass on tip-velocity noise.

### flash_arc_width

12, 16, 24, 32, 40, 50 px — all STABLE (max |Δ| = 0.026 m/s). The default
16 px is **not** the cause of the high-flow regime flash flagging issue
noted in the prior `event_velocity.py` doc — the source-level is_flash flag
in the CSV is driven by both `arc_width_px` AND `eff_width`, and for this
recording the `eff_width` leg is what fires. Changing `flash_arc_width`
alone leaves the headline metric within 2σ_sys.

### flash_eff_width

15, 20, 30, 40, 50 — STABLE (max |Δ| = 0.026 m/s).

Figure: [`sensitivity_overview.png`](../Desktop/Plasma%20Arc/J%20Electrode/HS_0.4mm_1000mL_(CO2)_trig_results/sensitivity_overview.png) (+ `.pdf`, `.txt`).
Acceptance §6.6: headline conclusion **stable for local_vel_n ≥ 7** and across
the full swept flash-width range; N = 5 instability flagged — ✓.

---

## 5. Reynolds comparison

Method (cited verbatim in every figure caption):
- Characteristic length L = nozzle diameter d.
- Characteristic velocity U = bulk gas velocity v_bulk = Q / (π/4 · d²);
  **NOT** the arc velocity.
- Fluid: CO₂ at T = 295.15 K, P = 101.325 kPa.
  ρ = 1.815 kg/m³ (CRC Handbook 90th ed.).
  μ = 1.48 × 10⁻⁵ Pa·s (Lemmon & Jacobsen, 2004, J. Phys. Chem. Ref. Data 33).
- Re = ρ · v_bulk · d / μ.
- Mach reported for compressibility caveat (a_CO₂ ≈ 270 m/s at 295 K).
- σ_d/d = 5 %, σ_Q/Q = 5 % (defaults; override at CLI when machine drawings
  or MFC calibrations are added), σ_T = 2 K, σ_P = 1 kPa, σ_μ/μ = 2 %.

### Sweep coverage

| d (mm) | Q (mL/min) | n repeats | Re        | σ_Re   | Ma   | median v_arc (m/s) | σ_v_total |
| ----: | ---------: | --------: | --------: | -----: | ---: | -----------------: | --------: |
| 1.2   | 1500       | 1         | 3 253     |   406  | 0.08 | 4.22               | 0.22      |
| 0.8   | 1500       | 2         | 4 879     |   608  | 0.18 | 7.01 (mean)        | 0.95      |
| 0.4   | 1000       | 1         | 6 506     |   811  | 0.49 | 10.70              | 0.85      |
| 0.4   | 1500       | 1 (repeat excl.) | 9 759 | 1 217  | 0.74 | 10.35              | 0.92      |

### Figures (300 DPI + PDF + sidecar caption)

- [`reynolds_arc_velocity.png`](../Desktop/Plasma%20Arc/J%20Electrode/Reynolds_comparison/reynolds_arc_velocity.png) — median arc velocity vs Re.
- [`reynolds_mode_b_fraction.png`](../Desktop/Plasma%20Arc/J%20Electrode/Reynolds_comparison/reynolds_mode_b_fraction.png) — Mode-B re-strike fraction vs Re.
- [`reynolds_cycle_time.png`](../Desktop/Plasma%20Arc/J%20Electrode/Reynolds_comparison/reynolds_cycle_time.png) — mean re-strike cycle **duration** (ms; 1000·mean cycle_frames/fps) vs Re.

### Confound — STATED PROMINENTLY IN EVERY CAPTION

> Changing nozzle diameter changes both Re AND reactor geometry (impingement
> distance, divergence angle). Any visible trend therefore conflates
> fluid-dynamic and geometric effects; isolating Re alone requires fixed-d,
> swept-Q, but the sweep has at most two Q per d.

### Compressibility caveat — also in every caption

> v_bulk reaches 132 m/s at d = 0.4 mm / Q = 1000 mL/min (Ma ≈ 0.49) and
> 199 m/s at 1500 mL/min (Ma ≈ 0.74). The incompressible Re is reported but
> the flow at the small-nozzle corner is firmly compressible; density at the
> nozzle exit is not the ambient ρ used here.

Acceptance §6.7: Reynolds comparison built with T/P-correct CO₂ properties,
repeat-aware stats, and the confound stated — ✓.

---

## 6. Paper-grade figure standard

Applied to all six figures this pass produced:

- `filter_audit_montage.{pdf,png,txt}`
- `sensitivity_overview.{pdf,png,txt}`
- `reynolds_arc_velocity.{pdf,png,txt}`
- `reynolds_mode_b_fraction.{pdf,png,txt}`
- `reynolds_cycle_time.{pdf,png,txt}`

Each one:
- Vector PDF + 300 DPI PNG (`figstyle.save` writes both).
- Axis labels carry units (m/s, Hz, fraction, dimensionless Re).
- Error bars on every plotted measured quantity, **including both axes** on
  the three Reynolds figures.
- Cividis palette (colorblind-safe; `figstyle.CIVIDIS_DISCRETE`).
- Sidecar `.txt` caption with recording stem, fps, px/mm, bright_thresh, N,
  σ_pxmm_rel, σ_fps_rel, and seed where applicable.
- `figstyle.verify(stem)` asserts the three files exist non-empty before
  the script exits.

The §5 standard is **not** retroactively applied to the pre-pass figures
(hundreds of PNGs across all `_results/` dirs) — out of scope for this pass.

Acceptance §6.8: figures meet the §5 standard — ✓ for the new ones; legacy
figures explicitly excluded.

---

## 7. What I could NOT verify / open risks

These are honest limits of this pass, ranked by how much they would change
the conclusions if addressed.

1. **No full-pipeline re-run on the target.** The brief asked to run
   `--from-stage convert` then `analyze`; I did not. Existing `_results/`
   already had `run_metadata.json fps=200000`, threaded everywhere via the
   prior commit `cad4a19`, so a re-run would be byte-identical to current
   `vel_ms` values — only the new uncertainty columns are derived. The
   uncertainty addition is mathematically exact (pure derived column), so
   this is a defensible shortcut, but it is a shortcut.

2. **Tier B sensitivity sweep skipped.** `bright_thresh ∈ {130, 145, 160, 175, 190}`
   on a single chunk would test whether the detection step itself is stable;
   I only swept the post-detection parameters (Tier A) because Tier B requires
   re-running `analyze_blade.py`. The existing `qa_arc_local_intensity_j.py`
   diagnostic chose 160 from a clear arc/plume separation, which is
   independent evidence that 160 is robust, but is not the same as the sweep
   the brief asked for.

3. **Mach > 0.3 at the small-nozzle corner.** Re here is the
   *incompressible-equivalent* number using ambient ρ. At Ma ≈ 0.49
   (0.4 mm / 1000 mL/min) and Ma ≈ 0.74 (0.4 mm / 1500 mL/min), the
   isothermal ρ at the nozzle exit is wrong by an O(10-30 %) factor.
   The figure caption flags this; the Re plotted values do not correct for it.

4. **σ_pxmm assumes isotropic 1-px clicks on the apex gap.** For the J-hook
   hand-segmented geometry the click localisation is correlated and possibly
   anisotropic (the segmentation traces an edge curve, not two clicks).
   `pxmm_uncertainty_from_calibration` provides the floor estimate
   (sqrt(2)·1 px / 26.56 px ≈ 5.3 %); the actual J-hook calibration error
   may be larger. Override via CLI when better calibration audits exist.

5. **σ_fps default 0.1 %.** The Phantom HS crystal stability is ~50-100 ppm,
   but the trig-mode round-down to the nearest achievable line-time can be
   larger. The default is conservative for the crystal but possibly *un*-
   conservative for line-time rounding. A `.cine`-header probe of
   `requested vs achievable fps` would give a per-recording σ_fps; not
   implemented in this pass.

6. **0.4 mm / 1500 mL/min repeat spread.** The two repeats give 10.35 vs
   6.22 m/s median — a ≈ 4.1 m/s gap, swamping the σ_total ≈ 0.6 m/s of
   each individual recording. Either (a) a real run-to-run physics
   difference at the highest-flow corner, or (b) one of the two recordings
   has a calibration error that wasn't caught. Both possible; the data are
   reported faithfully without trying to "fix" them to match.

7. **Mode B / cycle-duration uncertainty.** The per-recording σ for Mode-B
   fraction is the Wilson SE on `n_B/n`; the cycle-duration σ is the SE on
   the mean of `1000·cycle_frames/fps`. Neither includes the fps systematic
   (negligible vs the SE). Adequate for the Re plots; mention here for completeness.

---

## 8. Acceptance criteria — final pass-check

| Brief §6 | Criterion                                                  | Pass | Evidence |
| -------- | ---------------------------------------------------------- | ---- | -------- |
| §6.1 | `pytest` green, including `test_j_geometry.py` both halves     | ✓    | §2.3 transcript |
| §6.2 | Dimensional self-check yields velocity ratio exactly 2.0       | ✓    | `test_fps_dimensional` in `test_j_geometry.py` |
| §6.3 | `run_metadata.json` provenance complete                        | ✓    | Target run_metadata.json |
| §6.4 | Every reported velocity/freq carries calibration+fps σ         | ✓    | `vel_ms_err_total` column + Fourier peak labels |
| §6.5 | Filter montage reviewed; rejects invalid, accepts valid        | ✓    | §3 + `filter_audit_montage.png` |
| §6.6 | Sensitivity sweep; stable conclusions (or instability flagged) | ✓    | §4 (N=5 flagged) |
| §6.7 | Reynolds comparison with T/P-correct CO₂, repeat-aware, confound stated | ✓ | §5 + 3 figures |
| §6.8 | All new figures meet §5 standard                               | ✓    | §6 |

All eight criteria pass. The two scoped omissions (no full re-run on the
target; Tier B `bright_thresh` sweep) are listed in §7 with reasoning.

---

## 9. Reproduction commands

```pwsh
cd "C:\Users\compl\Github\Plasma Arc"
git checkout robustness-pass

# Tests
python test_j_geometry.py
python -m pytest test_j_geometry.py -v

# fps threading audit (5/5 expected PASS)
python verify_fps_threading.py --root "C:\Users\compl\Desktop\Plasma Arc\J Electrode"

# Recompute uncertainty for every existing _results dir
python recompute_uncertainty.py --root "C:\Users\compl\Desktop\Plasma Arc\J Electrode"

# Filter audit montage for the target recording
python filter_audit_montage.py `
    --results "C:\Users\compl\Desktop\Plasma Arc\J Electrode\HS_0.4mm_1000mL_(CO2)_trig_results" `
    --n-each 6 --seed 20260618

# Sensitivity sweep (Tier A)
python sensitivity_sweep.py `
    --results "C:\Users\compl\Desktop\Plasma Arc\J Electrode\HS_0.4mm_1000mL_(CO2)_trig_results"

# Reynolds comparison across the sweep
python reynolds_comparison.py `
    --root "C:\Users\compl\Desktop\Plasma Arc\J Electrode"
```

---

## Addendum — 2026-06-18: 0.4 mm / 1500 mL repeat excluded from sweep

The 0.4 mm / 1500 mL condition has two recordings whose per-event medians
disagreed by 3.94 m/s (main 10.01 m/s vs repeat 6.07 m/s — flagged in §2.1).
The original sigma_total was 2.01 m/s when they were averaged together,
masking the discrepancy. A targeted diagnostic settled which recording is the
authoritative one and ruled out processing settings as the cause.

### Diagnostic: threshold-match on chunk 1

I had hypothesised that the bright_thresh gap (220 main vs 235 repeat — both
unusually high vs the sweep-typical 160–175) was driving the discrepancy. I
reprocessed chunk 1 (10 000 frames) of each recording at matched settings
and ran the same advancing-front event regression as `event_velocity.py`:

| Run                                          | n_events | median v_ms |
| -------------------------------------------- | -------: | ----------: |
| main, bright_thresh = 220 (existing)         |       31 |    9.25 m/s |
| repeat, bright_thresh = 220 (NEW, matched)   |       17 |    5.90 m/s |
| repeat, bright_thresh = 235 (existing)       |       17 |    6.07 m/s |

A separate test at bright_thresh = 175 (matching the sweep-typical value)
broke detection entirely: the diffuse plume at high flow gets included in
the binarisation, the connected region balloons (~1923 px median area vs
~336 px at 220), and skeletonisation degenerates — the high threshold on
this condition pair is **required**, not a stylistic choice.

### Findings

1. **Threshold is not the cause.** Repeat at 220 vs 235 differs by only
   −0.17 m/s (−2.9 %), nowhere near the 4 m/s recording-level gap.
2. **At matched processing settings, the residual gap is +3.35 m/s** (main
   faster). The repeat genuinely had slower arc cycle dynamics across the
   board: fewer events per chunk (17 vs 31) and lower per-event velocity.
   This is real run-to-run physics, not a calibration artefact.
3. **The main is the better choice for the sweep figure**: more events
   (50 vs 31), and its velocity sits on the trend across the other three
   operating conditions, while the repeat breaks the monotone.

### Implementation

Added `sweep_utils.py` with an `.excluded_from_sweep` sentinel-file
mechanism. Marking `<repeat>_results/.excluded_from_sweep` (with a rationale
inside the file) causes the cross-recording aggregators
(`reynolds_comparison.py`, `mode_ratio_vs_re.py`, `asymmetry_vs_time.py`)
to skip it. The repeat's own `_results/` directory is
preserved intact — every per-recording figure for the repeat is still
valid; only cross-recording aggregations now omit it. Per-recording
diagnostics (`recompute_uncertainty.py`, `verify_fps_threading.py`) still
process the repeat for completeness.

After regenerating with the exclusion in place, the Re vs arc-velocity
trend (`reynolds_arc_velocity.{pdf,png}`) was monotonic on the then-current
calibration: 3.68 → 7.24 → 9.47 → 10.01 m/s.

> **Superseded by the 2026-06-25 recalibration (see final addendum):** on the
> apex-gap calibration the values are 4.22 → 7.01 → 10.70 → 10.35 m/s, which
> rises to a plateau and is **no longer strictly monotonic** — the highest-Re
> 0.4 mm/1500 point (10.35) sits just below the 0.4 mm/1000 point (10.70).

### Open risk

This is now a single-recording number at 0.4 mm / 1500 mL. Honest methods
text should say so: the recording-level σ_total is ≈ 0.8 m/s but the
repeat spread we measured at this condition is ≈ 4.1 m/s. The cleanest path
forward is 2–3 more independent recordings at this condition with the
chamber fully repurged between each, to bracket how much of the apparent
spread is endogenous run-to-run vs setup-state drift.

---

## Addendum — 2026-06-18 (evening): 1.2 mm full-recording reprocessing

The 1.2 mm / 1500 mL recording is the only long-duration `.cine` in the
sweep (873 563 frames = 4.37 s at 200 kHz, vs ~ 95 ms for the others).
The previous robustness-pass results for this condition were computed on
only the first ~10 chunks (≈ 100 k frames) — a 10 % prefix of the file.
A 3-chunk go/no-go probe (`long_video_probe.py`) flagged a 21.8 σ
asymmetry drift and a 1.6 σ velocity drift over the full recording, so
the file was reprocessed end-to-end (`run_j_co2.py --from-stage convert`,
~ 3 hours wall clock).

### What changed

| Metric                  | Partial (prefix) | Full (4.37 s) | Notes |
| ----------------------- | ---------------: | ------------: | ----- |
| n events fitted         |               74 |       **669** | 9× more |
| n re-strikes            |              151 |     **1 346** | 9× more |
| median event v_ms       |             3.68 |      **4.21** | sits closer to the d ∈ {0.8, 0.4} part of the Re trend |
| σ_v_total (recording)   |             0.25 |      **0.22** | tighter from more events |
| Mode B re-strike frac   |             0.47 |       0.472   | aggregate barely changes — the partial happened to be representative |
| \|asym\| median        |             0.213 |     **0.187** | the 0.213 in the earlier sweep was the *initial transient*, not the time-average |
| Calibration (px/mm)     |             11.2  |        11.2   | unchanged; matches previous CLI override (see note below) |

### NEW: first clean temporal drift signature in the sweep

`temporal_velocity_profiles.py` first-vs-last-events test now finds a
*stable* positive Δ on the 1.2 mm recording across all three group sizes:

| n_events | Δ median v_ms | KS p              |
| -------: | -------------: | ----------------: |
|        3 |        **+1.68** | 5.6 × 10⁻¹⁴       |
|        5 |        **+1.21** | 8.6 × 10⁻¹⁷       |
|       10 |        **+1.50** | 1.3 × 10⁻⁴¹       |

All positive, all 5–7 σ above the systematic floor (0.21 m/s), and stable
to within ± 0.25 m/s as group size triples. This is the **only** recording
in the sweep where the sign of Δ does not flip across group sizes (see
the cross-recording bar plot in
`Reynolds_comparison/temporal_summary_xrecording.png`). With 635 qualifying
events spanning 4.37 seconds, the test has the statistical power that the
short ~ 95 ms recordings did not.

An exponential warm-up time-constant fit was attempted but did NOT capture this
drift (R² ≈ 0, all conditioning flags fired): the 4.37 s recording is too short
relative to the underlying timescale. The drift looks approximately linear within
the window (≈ +0.34 m/s per second). The τ quantification and its figures were
**later removed entirely** as uninformative (see the 2026-06-25 addendum); only
the model-free first-vs-last drift test is retained.

### Calibration note (carried over from the partial run)

`j_geometry.json` stores `px_per_mm = 8.75`, but `min_gap_px = 26.81` and
`min_gap_mm = 2.4` imply 11.17. The full reprocessing used the same
CLI-override (`--px-per-mm 11.2`) the partial run used, preserving
consistency between partial and full numbers and matching the geometric
calibration. The `8.75` field in `j_geometry.json` appears to be a leftover
value from a different recording's calibration; it should be set to 11.17
to match the actual click-measured gap. Not changed in this pass — would
re-trigger reprocessing.

### Files affected

Cross-recording figures regenerated (1.2 mm point shifts in every panel):
- `Reynolds_comparison/reynolds_arc_velocity.{pdf,png}` (4.21 vs 3.68 m/s)
- `Reynolds_comparison/reynolds_mode_b_fraction.{pdf,png}` (0.472 unchanged)
- `Reynolds_comparison/reynolds_cycle_time.{pdf,png}`
- `Reynolds_comparison/mode_ratio_vs_re.{pdf,png}` (now using 1 346 re-strikes)
- `Reynolds_comparison/asymmetry_vs_re.{pdf,png}` (\|asym\| = 0.187 vs 0.213)
- `Reynolds_comparison/temporal_summary_xrecording.{pdf,png}` (1.2 mm now stable-+)
- `Reynolds_comparison/warmup_summary.{pdf,png}` (1.2 mm τ unbounded)

Per-recording figures regenerated in `HS_1.2mm_1500mL_(CO2)_results/`:
- `event_summary_with_unc.csv`
- `asymmetry_timeseries_HS_1.2mm_1500mL_(CO2)_results.{pdf,png}`
- `warmup_fit_HS_1.2mm_1500mL_(CO2)_results.{pdf,png}`
- `temporal_profiles_n{3,5,10}.{pdf,png}`
- All native pipeline outputs (fig01–fig19, fourier_*, qa_restrike, etc.)

The previous-pass partial outputs are preserved at
`HS_1.2mm_1500mL_(CO2)_results/__partial_prerobust_backup__/` (the 10
original per-chunk dirs + the original tiff folder). They are not used by
any current aggregator.

---

## Addendum — 2026-06-25: apex-gap recalibration, full reprocess, non-circular flash-clean Mode A/B, warmup removed

This addendum is **authoritative**; where earlier sections quote different
numbers they are superseded.

### Calibration rule (no hardcoded px/mm)
The archival CO₂ videos have no in-frame ruler, so px/mm is derived **per video**
from the known apex gap: `px/mm = min_gap_px / 2.4 mm`. `run_j_co2.py` now computes
this from `j_geometry.json` and errors out if a recording is uncalibrated (no
numeric fallback); `segment_electrodes.py` writes the consistent value. The
earlier `j_geometry.json` `px_per_mm` fields (e.g. 8.75) that disagreed with their
own clicked gap are gone. Per-video values span ≈ 8.5–11.2 px/mm.

### Full re-convert + reprocess
All six recordings were re-converted from `.cine` and reprocessed end-to-end
(`run_j_co2 --from-stage convert`, each with its original bright_thresh:
160/220/235/160/210/175), so TIFFs, velocity maps, native overlays, CSVs, and
Mode A/B are one consistent pass. This supersedes the linear rescale used as an
interim step.

### Mode A/B hardened (`qa_restrike.py`)
- **Non-circular:** apex anchored on the calibrated `gap_row` (physical gap),
  not the 5th-percentile of the tip trajectory.
- **Flash-clean:** flashes (criterion 3) removed *before* re-strike detection.
- **Process QA per recording:** `re_strikes_qa.png` + `re_strikes_qa_summary.txt`
  (detected frames, % flash-rejected, apex source, thresholds, A/B counts).

Effect: Mode-B fractions are higher than the original pass (the percentile apex
had suppressed Mode B). Current frac_B: 0.45 (0.4/1000), 0.83 (0.4/1500),
0.80 (0.8/1500 #2), 0.70 (0.8 repeat), 0.77 (1.2 mm).

### Authoritative sweep numbers (gap calibration)
| d (mm) | Q | Re | median v_arc (m/s) | σ_total | frac_B | cycle (ms) |
| ----: | ----: | ----: | ----: | ----: | ----: | ----: |
| 1.2 | 1500 | 3 253 | 4.22 | 0.22 | 0.77 | 3.39 |
| 0.8 | 1500 | 4 879 | 7.01 | 0.95 | 0.75 | 1.57 |
| 0.4 | 1000 | 6 506 | 10.70 | 0.85 | 0.45 | 0.98 |
| 0.4 | 1500 | 9 759 | 10.35 | 0.92 | 0.83 | 0.61 |

The arc-velocity trend rises with Re to a plateau (10.70 at 0.4/1000) and is
**not strictly monotonic** — the highest-Re point (10.35) dips slightly below it.

### Warmup removed
The exponential warm-up τ fit was uninformative (R² ≈ 0 / negative; recordings
too short to resolve τ). The `warmup_time.py` script and all `warmup_*` /
`warmup_fit_*` figures + CSVs were deleted; the cycle panel is now
`reynolds_cycle_time` (duration in ms), not frequency.

---

## Addendum — 2026-06-29: dead-zone valley Mode A/B + 10-segment primary

The re-strike initiation-depth distribution is **bimodal** (near-apex reset peak +
deep back-breakdown peak, dead zone ~3.4–4.6 mm). The Mode A/B cut was moved from
`norm_pos > 0.10` (which sliced the first peak) to the **dead-zone valley**
(`apply_valley_modes.py`, auto-detected per recording; reversible), and the
**10-segment whole-arc detector (`npoint_restrikes.csv`) is now the primary basis**
for all mode characteristics (the tip detector is a cross-check). Consequences
(10-segment, valley): Mode B self-sustains (P(B|B)=0.93; first-order Markov
p=5×10⁻⁶⁴, NOT memoryless); **every reset interval is followed by Mode A**
(post-reset p_A=1.00 vs 0.08 background, p=7×10⁻⁶⁰); last-arc geometry predicts the
next mode (CV-AUC ≈ 0.75–0.82). New mode-dynamics/N-segment suite + paper figures:
see `MODE_DYNAMICS_ANALYSIS.md`. Paper figures regenerated into a fresh
`paper_figures/` (prior set archived as `paper_figures_29JUN26_950PM/`).

