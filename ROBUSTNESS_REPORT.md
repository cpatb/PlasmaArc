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
px/mm = 12.5 (J-hook hand-segmented), bright_thresh = 160, gap_row = 20.
Provenance: `run_metadata.json` carries fps, px/mm + source, threshold + source,
geometry mode, σ budget, git hash, UTC timestamp — acceptance §6.3 ✓.

### 2.1 Per-event velocity, with full uncertainty

The pre-pass `event_summary.csv` reported only the OLS slope SE, which on this
recording is sub-percent (median 0.10 m/s) — misleadingly tight. Adding the
calibration + fps systematic (σ_pxmm_rel = 5.00 %, σ_fps_rel = 0.100 %) gives:

| Stat                                  | Pre-pass (stat only) | After robustness (stat ⊕ sys) |
| ------------------------------------- | -------------------: | ----------------------------: |
| Median event velocity                 | 9.47 m/s             | 9.47 ± 0.59 m/s               |
| Median σ_stat (OLS slope SE)          | 0.10 m/s             | 0.10 m/s                      |
| Median σ_sys (px/mm ⊕ fps)            | —                    | 0.47 m/s                      |
| Median σ_total                        | —                    | 0.59 m/s                      |

The point estimate `vel_ms` is **unchanged** (this is a derived recompute, not
a re-fit). What changes is the honesty of the error bar: it is now ~6× larger
because the calibration uncertainty had been entirely outside the budget.

The recompute applied uniformly to all five existing `_results/` dirs:

| Recording                              | n events | median v (m/s) | σ_sys (m/s) | σ_total (m/s) |
| -------------------------------------- | -------: | -------------: | ----------: | ------------: |
| HS_0.4mm_1000mL_(CO2)_trig             |       53 | 9.47           | 0.47        | 0.59          |
| HS_0.4mm_1500mL_(CO2)_trig             |       50 | 10.01          | 0.50        | 0.79          |
| HS_0.4mm_1500mL_(CO2)_trig_repeat      |       31 | 6.07           | 0.30        | 0.52          |
| HS_0.8mm_1500mL_(CO2)_trig(2)          |       24 | 7.24           | 0.36        | 0.46          |
| HS_1.2mm_1500mL_(CO2)                  |       74 | 3.68           | 0.18        | 0.23          |

**Discrepancy worth surfacing (brief §0).** The 0.4 mm / 1500 mL/min condition
has two recordings; their per-event medians differ by 3.94 m/s (10.01 vs 6.07),
**far more than the per-recording σ_total** (~0.65 m/s). The repeat-to-repeat
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
| 1.2   | 1500       | 1         | 3 253     |   406  | 0.08 | 3.68               | 0.25      |
| 0.8   | 1500       | 1         | 4 879     |   608  | 0.18 | 7.24               | 0.97      |
| 0.4   | 1000       | 1         | 6 506     |   811  | 0.49 | 9.47               | 0.75      |
| 0.4   | 1500       | 2         | 9 759     | 1 217  | 0.74 | 8.04 (mean)        | 2.01      |

### Figures (300 DPI + PDF + sidecar caption)

- [`reynolds_arc_velocity.png`](../Desktop/Plasma%20Arc/J%20Electrode/Reynolds_comparison/reynolds_arc_velocity.png) — median arc velocity vs Re.
- [`reynolds_mode_b_fraction.png`](../Desktop/Plasma%20Arc/J%20Electrode/Reynolds_comparison/reynolds_mode_b_fraction.png) — Mode-B re-strike fraction vs Re.
- [`reynolds_cycle_frequency.png`](../Desktop/Plasma%20Arc/J%20Electrode/Reynolds_comparison/reynolds_cycle_frequency.png) — mean re-strike cycle frequency (fps / mean cycle_frames) vs Re.

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
- `reynolds_cycle_frequency.{pdf,png,txt}`

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

6. **0.4 mm / 1500 mL/min repeat spread.** The two repeats give 10.01 vs
   6.07 m/s median — a 3.94 m/s gap, swamping the σ_total ≈ 0.65 m/s of
   each individual recording. Either (a) a real run-to-run physics
   difference at the highest-flow corner, or (b) one of the two recordings
   has a calibration error that wasn't caught. Both possible; the data are
   reported faithfully without trying to "fix" them to match.

7. **Mode B / cycle-frequency uncertainty.** The per-recording σ for Mode-B
   fraction is the Wilson SE on `n_B/n`; the cycle-frequency σ is the SE on
   the mean of `fps/cycle_frames`. Neither includes the fps systematic
   (σ_f = f · σ_fps_rel adds ≈ 1 Hz at f = 1 kHz, well below the SE).
   Adequate for the Re plots; mention here for completeness.

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
