# Running the J-electrode (CO₂) analysis from scratch

**Written 2026-08-31 for a collaborator holding this code and unable to run it.**
It covers the CO₂ J-electrode path only. The N₂ blade path is a different set of
recordings with a different calibration and is documented in `PIPELINE_GUIDE.md`;
`METHODS.md` carries the physics for this path and `METHODS_BLADE.md` for that one.

Nothing here is new analysis. It is the four things the code does not tell you: where the
data has to be, which script starts what, which constants live outside the code, and the
places this pipeline has actually gone wrong.

---

## 0. The one-minute version

```powershell
cd "C:\Users\compl\Github\Plasma Arc"
pip install -r requirements.txt

# 1. convert enough frames to have something to click on
python run_j_co2.py --cine "HS_0.4mm_1500mL_(CO2)_trig.cine" --from-stage convert --max-frames 10000

# 2. the ONE interactive step: click the four electrode ends, tip-end first per blade
python calibrate_j_electrodes.py --tiff "<...>_results/tiff/<...>_0001.tiff" --gap-mm 3.2 --arc-direction down

# 3. everything else
python run_j_co2.py --cine "HS_0.4mm_1500mL_(CO2)_trig.cine" --from-stage analyze
```

Step 2 is not optional and it is not automatable: **these archival recordings have no
ruler in frame**, so the only length reference in the whole experiment is the known apex
gap between the electrodes, and somebody has to point at it. Everything downstream — every
velocity in m/s, every arc length in mm — is that click.

---

## 1. What the pipeline is

Six stages, run by `run_j_co2.py`, which is a thin orchestrator: the conversion, merge and
subprocess helpers are **imported from `run_11may26.py`**, not forked, so the CO₂ path is
the same code as the maintained N₂ blade path with a different geometry.

| stage | what runs | writes |
|---|---|---|
| `convert` | `.cine` → 8-bit TIFF chunks (memory-safe, 10 k frames at a time) | `tiff/<stem>_NNNN.tiff` |
| `analyze` | `analyze_blade.py --finite-segments`, per chunk | per-chunk CSVs |
| `merge` | concatenate the chunk CSVs | `measurements.csv` |
| `events` | `event_velocity.py`, then `qa_restrike.py` for Mode A/B | `event_summary.csv`, `measurements_all_events.csv`, `re_strikes_qa.csv` |
| `fourier` | `fourier_analysis.py` | spectra |
| `visualize` | 2-D local-velocity map + figures | `velocity_map_local.*` |

`--from-stage <name>` restarts at a stage and runs everything after it. That is how you
re-run the analysis after changing a threshold without re-converting 300 k frames.

**`qa_restrike.py` is authoritative for Mode A/B.** If two files disagree about a mode
fraction, `re_strikes_qa.csv` is the one that is right.

---

## 2. Where the data has to be

`run_j_co2.py::_resolve_j_dir()` looks, in order, for:

```
~/Desktop/Plasma Arc/J Electrode
~/OneDrive/Desktop/Plasma Arc/J Electrode
C:/Users/compl/Desktop/Plasma Arc/J Electrode
C:/Users/compl/OneDrive/Desktop/Plasma Arc/J Electrode
```

and takes the first that exists. On a different machine, put the recordings in
`~/Desktop/Plasma Arc/J Electrode` or add a path to that list — it is four lines and it is
the only place the tree is hard-coded.

Inside it, one `.cine` per recording, and one `<stem>_results/` directory created beside
it holding everything the pipeline writes. `python run_j_co2.py` with no `--cine` **lists
the recordings it can see** and exits, which is the fastest way to check the path is right.

**This is a sweep, not one experiment.** Nozzle diameter × flow, with repeats, and each
recording is an independent measurement with **its own frame rate** (auto-detected per
file — the `trig` recordings are *not* assumed to be 200 kHz) and **its own electrode
calibration**. There is no default recording on purpose.

---

## 3. The constants that are not derivable from the code

These are experimental facts. If you change data, you must change them, and nothing in the
code will complain.

| constant | value | where | why it is what it is |
|---|---|---|---|
| `J_GAP_MM_DEFAULT` | **3.2 mm** | `run_j_co2.py` | The true minimum apex gap, measured by hand and recorded in `hand_calibration_authoritative_25JUN26`. **It was 2.4 mm until 2026-06-25.** See §5. |
| `J_BRIGHT_THRESH` | 210 DN | `run_j_co2.py` | From the arc-vs-plume local-intensity QA (`qa_arc_local_intensity_j.py`): the arc core saturates near 255 and the plume sits below ~150, so 210 keeps the whole arc and rejects the plume across the sweep. |
| `J_FLASH_EFF_WIDTH` | 25 px | `run_j_co2.py` | The BLADE defaults (20 px eff, 16 px width) are tuned for the thin N₂ arc and **wrongly flag the wide CO₂ arc**, which reaches ~41 px on 0.4 mm/1500. Real-arc `eff_width` p99 is ~13 across the sweep. |
| `J_FLASH_ARC_WIDTH` | 50 px | `run_j_co2.py` | Keeps the genuinely fat 0.4 mm/1500 arcs and only trips on extreme blooms. True electrode-bloom flashes are rejected upstream by the `gap_row` cutoff, not here. |
| `J_ARC_DIRECTION` | `down` | `run_j_co2.py` | Inverted-V electrodes at the top; the CO₂ arc extends downward. |
| per-recording `bright_thresh` | 160 / 220 / 235 / 160 / 210 / 175 | the six `<stem>_results/` | Each recording was converted and reprocessed end to end on 2026-06-25 with **its original** threshold, so every output in the current tree is one consistent pass. |
| `px_per_mm` | **derived, never written** | `j_geometry.json` | `min_gap_px / min_gap_mm`, per video. At 3.2 mm the two clicked recordings give **9.375** (30 px) and **6.5625** (21 px). |

**Imaging, for the record:** 200 kfps, 5 µs interval, 4.53 µs exposure, scale bars 3.2 mm.
Reynolds set ρ = 1.815, µ = 1.48e-5, Re = 3239 / 4859 / 6478 / 9717, Ma 0.08–0.74.

---

## 4. Validation before you trust an output

```powershell
python test_electrode_geometry.py
python test_j_geometry.py
python build_parameter_table.py      # cross-checks px_per_mm == min_gap_px / min_gap_mm
```

`build_parameter_table.py` is the useful one for a new machine: it re-derives the
calibration identity from `j_geometry.json` and **flags any recording where the stored
`px_per_mm` does not equal `min_gap_px / min_gap_mm`**, which is the single failure mode
that silently rescales every velocity in the study.

The golden anchor for `derive_geometry` is `gap_row 29/33`, `px/mm 6.9034/9.2045`,
`min_gap_px 22.091`.

> ⚠ **That anchor is at `gap_mm = 2.4`, and production runs at 3.2.** It is a synthetic
> fixture on made-up clicks, so it pins the *arithmetic* of `derive_geometry` and cannot
> see a change in the production gap. `build_parameter_table.py` is what checks the real
> calibration. Do not read a passing `test_j_geometry.py` as "the calibration is right".

---

## 5. Five things that have actually gone wrong here

Written down because each one produced numbers that looked fine.

1. **The gap was 2.4 mm and is 3.2 mm.** Correcting it on 2026-06-25 rescaled **every arc
   velocity and every mm length by 3.2/2.4 = 1.333×**. If you are comparing against a
   figure or a number from before that date, check which gap it was computed at. The
   `__pre_calib_rescale_backup__/` directories are the stale pre-correction state and are
   superseded by the full reprocess, not by a rescale.

2. **The A/B velocity separation is detector-dependent, and the tip detector is biased.**
   The tip detector gives a clean separation (A fast, B slow) and reproduces the sample
   figure exactly — but **~34 % of cycles, mostly Mode B, have NaN local velocity**,
   because the tip cannot track deep restrikes. Mode-B local velocity from the tip is a
   biased subsample. **The 10-segment (`npoint`) detector is the primary method**: it
   classifies 69–92 % of restrikes as Mode B against 41–66 % for the tip, and fires about
   7× more restrikes, catching mid-arc back-breakdowns the tip misses entirely.

3. **Advance is a signed quantity and was once computed as a magnitude.** The first pass
   used the mean of `|seg_v_normal|` as "advance", which counts Mode B's back-and-forth
   thrashing as forward progress and made B look faster than A. Advance is the per-cycle
   mean **signed** apex `seg_vy_ms`. On that definition A > B at every Re
   (5.44/2.78, 9.51/4.48, 16.12/7.05, 14.49/11.19), and **at Re ≥ 6478 Mode B net-retreats**.
   Mode B does have higher raw local *speed* at low Re (11.03 vs 6.73 at Re 3239) — real,
   but non-directional: A is a directed glide (|advance|/speed ≈ 0.8–0.97), B is chaotic
   thrashing (0.25–0.40). High speed is not advance.

4. **There is no velocity cap and there must not be one.** A 16 m/s cap once stood in for
   flash filtering. It is removed. Flashes are rejected at the *frame* level by
   `arc_filter.py`'s three criteria — detected, complete, not-a-flash — and velocities are
   then reported exactly as measured, with the distribution never truncated. If you find
   yourself wanting a cap, the frame filter is what needs fixing.

5. **There are four modes, not three.** A, B, **C** and D. Mode C is a ~1 ms no-plasma
   interval recurring at ~8.33 ms — a **ground-fault-interrupter trip**, 120 Hz, an
   artefact of the power supply and not of the arc. It is *not* the drop-impact project's
   "≥ 500 frames / 2.5 ms" definition and the two must not be conflated. Mode D is a
   brighter, shape-preserving re-strike (propagation) whose ΔV is near the noise floor.

---

## 6. What this document does not cover

- **The cross-recording sweep.** `reynolds_comparison.py`, `mode_ratio_vs_re.py` and
  `asymmetry_vs_time.py` run across the per-nozzle recordings (d = 0.4 / 0.8 / 1.2 mm) and
  write to `<root>/Reynolds_comparison/`. They assume every recording has already been
  through §0.
- **The paper figures.** `npoint_restrike.py`, `nseg_compare.py`, `fig_reset_to_modeA.py`
  and `velmap_composite_highdpi.py` are the four scripts that write into `paper_figures/`.
  The mapping from those files to numbered figures in the manuscript is **not recorded
  anywhere in this repository** and is the one thing here that could not be reconstructed
  from the code. It needs the manuscript.
- **The N₂ blade path** — `PIPELINE_GUIDE.md`, and note that its headline runs the *other*
  way from this one: blade transitions are mildly persistent (P(A|A) = 0.53,
  P(B|B) = 0.58), where the J-electrode result is alternation.
