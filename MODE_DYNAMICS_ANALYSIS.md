# J-electrode mode-dynamics & N-segment arc analysis

New analyses added on top of the settled hand-authoritative 3.2 mm gap calibration
(px/mm = `min_gap_px / 3.2`, re-segmented; `run_metadata` ≡ `j_geometry` for all 6
recordings). All scripts read per-recording calibration from `run_metadata.json`;
nothing is hardcoded.

## AUTHORITATIVE Mode A/B definition (revised — supersedes everything below)

**Threshold = the dead-zone valley.** The re-strike initiation-depth distribution
(distance of the re-strike below the apex) is **bimodal**: a near-apex *reset* peak
and a deeper *back-breakdown* peak, separated by a sparse dead zone (~3.4–4.6 mm).
Mode A/B is defined by the **valley between the two peaks** (auto-detected per
recording by `valley_reclassify.py` / applied by `apply_valley_modes.py`), NOT the
old `norm_pos > 0.10` cut (~1.5 mm), which sliced through the first peak and
mislabelled the upper half of the near-apex reset population as Mode B. Originals
are preserved (`*__pre_valley_backup.csv`, `mode_normpos010`/`mode_deepest010`
columns); restore with `apply_valley_modes.py --restore`.

**Primary detector = the 10-segment whole-arc detector** (`npoint_restrikes.csv`,
valley cut on the deepest-point depth). It sees the whole arc (incl. one-sided leg
breaks the tip misses) and is the authoritative basis for all characteristics and
analysis. The tip detector (`re_strikes_qa.csv`) is retained as a cross-check
(`--source tip` on every analysis script).

### Headline findings under the revised scheme (10-segment, valley)
- **Mode B strongly self-sustains; the chain is NOT memoryless.** Pooled
  P(B|B)=0.93, P(B|A)=0.74, P(A|A)=0.26 (≫ base rate 0.09) — both modes cluster
  into runs (first-order Markov G-test p=5×10⁻⁶⁴). (`fig_transition_matrix`.)
- **Every reset is followed by Mode A — hypothesis CONFIRMED.** After a full
  extinction the first whole-arc re-strike is Mode A in **53/53** cases
  (post-reset p_A = 1.00 vs 0.08 background; Fisher p=7×10⁻⁶⁰). The earlier
  "not supported" was an artefact of the old loose cut. (`fig_reset_intervals`.)
- **Last-arc geometry predicts the next mode** (multivariate CV-AUC ≈ 0.75–0.82;
  the valley cut is geometrically cleaner than the old cut). (`fig_geometry_predictor`.)
- Mode-B fraction (10-segment, valley): ~0.69–0.92; the whole-arc detector counts
  leg-side back-breakdowns, so Mode B dominates while a *full*-apex reset (Mode A)
  is the stricter, rarer event.

> The tip detector with the valley cut tells a complementary story (balanced
> A/B ~0.48–0.66; A↔B **alternation**, P(A|B)=0.63) — the sparser tip view
> resolves the apex-reset / back-breakdown oscillation, while the whole-arc view
> shows Mode-B persistence. Both are reported (`--source tip` vs `npoint`).

---

Mode A/B is produced for **both** detectors where applicable; the 10-segment
(`npoint_restrikes.csv`) is primary, the tip (`re_strikes_qa.csv`) secondary.

## New modules

| Script | Deliverable | Output location |
|---|---|---|
| `arc_resets.py` | shared reset-interval detection | (library) |
| `predict_stats.py` | scipy/numpy stats (AUC, Cliff's δ, logistic, CV-AUC, BH-FDR) | (library) |
| `npoint_geometry.py` | full **row+col** N-point reconstruction → `npoint_xy_cache.npz` | per `_results/` |
| `mode_transition_analysis.py` | **conditional probabilities / Markov** | `analysis/mode_transitions/` |
| `restrike_location_maps.py` | **Mode A/B initiation heatmaps** (velocity-map style) | per `_results/` + `Reynolds_comparison/` |
| `segment_velocity.py` | **per-segment local velocity over time** | per `_results/segment_velocities.csv` |
| `segment_velocity_map.py` | **segment-resolved 2-D velocity map** | per `_results/velocity_map_segment.*` |
| `npoint_arc_stats.py` | **tortuosity + arc-characteristic stats** by mode, vs Re | `analysis/arc_stats/` |
| `restrike_geometry_predictor.py` | **last-arc geometry → next mode** | `analysis/geometry_predictor/` |
| `reset_interval_analysis.py` | **reset intervals** (1.2 mm) | per `_results/` + `Reynolds_comparison/` |

The heavy `npoint_xy_cache.npz` (row+col per N-point per detected frame) was built
for all 6 recordings incl. the 873k-frame 1.2 mm (154,188 reconstructed frames);
validated: deepest-point row offset −3 px (mad ~1), col |Δ| ~3–4 px.

## Key findings

### Conditional probabilities (how Mode B sustains itself)
Pooled CO₂: P(A|A)=0.27, P(B|A)=0.73, P(A|B)=0.26, **P(B|B)=0.745** (marginal
p_B=0.738). Excess persistence P(B|B)−p_B = **+0.007**; first-order Markov G-test
p=0.55 → CO₂ re-strikes are **memoryless**: the next mode is ~a biased coin set by
the marginal Mode-B probability, not by the current mode. Mode B "loops onto
itself" for a **mean run of 2–13 cycles** per recording, but that run length is
explained by the high p_B alone (obs ≈ i.i.d.). Only the N₂ blade shows genuine
first-order structure. Transitions are broken across reset intervals (53 on 1.2 mm).
→ `analysis/mode_transitions/mode_transition_report.md`

### Mode A vs B initiation locations
Mode A ignites in a tight spot **at the apex gap**; Mode B ignites both at the apex
**and from a second hotspot further out and off to one leg** (B−A difference map).
The whole-arc detector exposes far more Mode-B leg breaks (8994 vs 991 on 1.2 mm)
that the tip detector cannot localise laterally.
→ `_results/restrike_location_map_{tip,npoint}.png`,
`Reynolds_comparison/restrike_initiation_depth_xrecording.*`

### Per-segment velocity & segment-resolved map
Per-segment axial velocity (sliding-window OLS, same method as the tip's
`local_vel_ms`, window N=10) peaks mid-arc/tip (~12 m/s) and is slowest at the
anchored feet (~5 m/s). The segment-resolved velocity map paints each segment with
its own velocity, revealing internal structure the uniform tip map washes out (the
lower spreading extremities move fastest). Also reports normal (lateral-sweep) and
tangential components per segment. **The 2-D map paints per-segment SPEED** (`--metric
seg_speed_ms`, default): the axial component goes negative on laterally-sweeping body
segments (~19% of samples), which a `>0` display would hide, so speed (always ≥0) is
the right always-positive analogue of the tip map and fills the whole arc.
→ `_results/segment_velocity_profile.*`, `velocity_map_segment_vs_tip.png`

### Arc-characteristic statistics
N=10 resampled tortuosity reconciles with the full-skeleton pipeline tortuosity at
Pearson r=0.82–0.91 (N=10 ≈ 94% of full, as expected for coarse-graining). Mode-A
cycles have slightly **higher** tortuosity / longer / wider arcs than Mode-B. The
two 0.4 mm/1500 recordings show a ~5 mm median "width" — the diffuse high-flow CO₂
plume, not a thin channel.
→ `analysis/arc_stats/arc_stats_report.md`, `arc_stats_vs_re.*`,
`tortuosity_reconciliation.*`

### Geometry → next mode (is there a critical tortuosity / arc length?)
Last-visible-arc (peak extension before collapse) geometry predicts the next mode
**moderately within a recording** (multivariate CV-AUC 0.58–0.72; 1.2 mm 0.69).
**Arc width** is the strongest single predictor (wider → Mode B, pooled AUC 0.59);
**tortuosity does NOT separate** (AUC 0.49, n.s.). Critically, **every single
feature is INCONSISTENT across recordings** → the pooled univariate separation is a
Reynolds confound, so there is **no universal critical tortuosity/length cut**; only
the multivariate combination predicts within a fixed operating point.
→ `analysis/geometry_predictor/geometry_predictor_report_{tip,npoint}.md`

### Reset intervals (1.2 mm) — hypothesis "reset ⇒ Mode A"
**There are essentially no long extinctions**: the longest no-arc stretch is only
**3.3 ms (663 frames)** — sweeping the dead-interval threshold 500→1000 by 100 gives
53 resets at ≥500 fr, 4 at ≥600 fr, and **0 at ≥700 fr** (none beyond 700, let alone
950). Post-reset Mode-A fraction is weakly A-enriched at the 500-fr threshold (p_A
0.38 vs 0.28 background) and is **gone by 600 fr** (0.25); **Mode B still dominates**
after a reset, so the hypothesis is not supported. (Caveat: "first re-strike after
reset" = first collapse after re-ignition, not the re-ignition itself.)

### Does a tighter Mode-A/B threshold change the picture?
No — it makes it worse. Mode A clusters tight against the apex, Mode B is bimodal (a
near-apex component that **overlaps** A plus a long tail to 7–15 mm), so the two
genuinely share near-apex ground. A 3-px allowance (≈0.4 mm) sits *below* essentially
every re-strike (0% of either mode is within 3 px of the apex), so it would relabel
~100% of re-strikes as Mode B — erasing Mode A rather than separating it. The current
0.10·L_eff cut (~1.5 mm) sits between the A cluster and the B tail and is already near
the tight end. The A/B overlap is an intrinsic continuum of initiation depth, not a
loose-threshold artefact (`fig_initiation_depth`).
→ `_results/reset_interval_report.md`, `reset_interval_analysis.*`,
`Reynolds_comparison/reset_interval_summary.csv`

## Paper figures
All analyses were run on **all six** J-electrode recordings. The paper-figure set
(`…/J Electrode/paper_figures/`) was updated:
- Existing figures **refreshed** on the current data (`build_figure_data.py` →
  `hs_figures.py`).
- New companion `hs_figures_new.py` (reuses `hs_figures` style/helpers) adds, in the
  same house style (serif, 300 dpi PNG):
  `fig_segment_velocity_map_montage_{electrodes,noelectrodes}` (per-segment 2-D maps
  vs the tip-uniform montage), `fig_segment_velocity_profile`,
  `fig_restrike_initiation_map` (A | B | B−A), `fig_transition_matrix` (pooled CO₂,
  memoryless), `fig_geometry_predictor` (per-feature AUC + CV-AUC),
  `fig_reset_intervals` (1.2 mm). Regenerate with `python hs_figures_new.py`.
- `_render_velmap_montage` gained a `cbar_label` arg so the segment montage is
  labelled correctly. Reset intervals exist only on the 1.2 mm recording (the short
  recordings have **zero** no-arc gaps ≥500 frames — the rapid-cycling finding
  generalises).

## Reproduce
```pwsh
cd "C:\Users\compl\Github\Plasma Arc"
$root = "C:\Users\compl\Desktop\Plasma Arc\J Electrode"
python npoint_geometry.py build --all          # heavy row+col cache (1.2mm ~45min)
python npoint_geometry.py validate --all
python mode_transition_analysis.py --data-root "C:\Users\compl\Desktop\Plasma Arc"
python restrike_location_maps.py --all --source both
python segment_velocity.py --all
python segment_velocity_map.py --all
python npoint_arc_stats.py --all
python restrike_geometry_predictor.py --all --source both
python reset_interval_analysis.py --recording "HS_1.2mm_1500mL_(CO2)"
```
