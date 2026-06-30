# N-segmentation sensitivity (N = 10, 20, 50)

Exploratory check: does the arc segmentation resolution change the whole-arc
mode-dynamics characterizations? Re-ran the entire npoint pipeline (detector →
dead-zone valley Mode A/B → transitions / reset / geometry predictor / segment
velocity) at **N = 20** and **N = 50** and compared to the headline **N = 10**.

## How it was generated
- `npoint_geometry.py build --all --n 50 --cache-suffix _N50` — fresh row+col
  reconstruction at N=50 into a separate `npoint_xy_cache_N50.npz` (N=10/15/20 are
  already in the committed `npoint_xy_cache.npz`; the 1.2 mm 873k-frame pass is the
  long pole, ~45 min).
- `nseg_pipeline.py --n {20,50} [--maps]` — N-tagged detector/valley/segment outputs
  per recording (`npoint_restrikes_N{n}.csv`, `segment_velocities_N{n}.csv`,
  `velocity_map_segment_N{n}.npy`, initiation coords) + cross-recording
  `mode_transition_summary_npoint_N{n}.csv` and `geometry_predictor_*_npoint_N{n}.csv`.
- Paper folders `paper_figures/20_segment/` and `paper_figures/50_segment/` (each a
  self-contained `hs_figures.py` master, `ARC_N` set) sit next to `10_segment/`.
- `nseg_compare.py` → `paper_figures/fig_Nsegment_comparison.{png,pdf}` +
  `nseg_comparison_summary.csv`.

## Result — the characterizations are N-INVARIANT

| metric (pooled, valley cut) | N=10 | N=20 | N=50 |
|---|--:|--:|--:|
| Mode-B self-loop P(B\|B)        | 0.929 | 0.931 | 0.932 |
| overall Mode-B rate p_B         | 0.907 | 0.909 | 0.911 |
| first-order Markov G-test p     | 5e-64 | 4e-66 | 4e-69 |
| geometry→mode CV-AUC            | 0.705 | 0.707 | 0.705 |
| reset → Mode A (1.2 mm, n=53)  | 1.00  | 1.00  | 1.00  |

Per-recording Mode-B fraction and the segment speed profile are likewise
indistinguishable across N (see `fig_Nsegment_comparison`). The only thing that
moves with N is the **raw detector event count**, which creeps up slightly
(1.2 mm: 9 553 → 9 795 → 10 029 for N=10→20→50) because more points can each
catch a finer leg break — but those extra events are the same population
(deep back-breakdowns), so every *fraction*, transition probability, predictive
AUC, and the reset→A result are unchanged.

**Conclusion.** N=10 is sufficient; the whole-arc conclusions are not an artefact
of the segmentation count. Finer segmentation (N=20, 50) only sharpens the segment
velocity profile/map spatially and adds a handful more leg-break detections, with
no effect on the physics readouts. The N=20 and N=50 paper folders are provided
for side-by-side inspection alongside 10_segment.

## Files
- Comparison: `paper_figures/fig_Nsegment_comparison.png`, `nseg_comparison_summary.csv`
- Folders: `paper_figures/{10,20,50}_segment/` (run each folder's `hs_figures.py`)
- Drivers: `nseg_pipeline.py` (N-tagged data), `nseg_compare.py` (cross-N figure),
  `npoint_geometry.py --n/--cache-suffix` (high-N reconstruction).
