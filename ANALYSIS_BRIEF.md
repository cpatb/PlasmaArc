# Analysis Brief — J-electrode pipeline + Reynolds comparison

**How to use this file:** In Claude Code, from the repo root, say:
> Read ANALYSIS_BRIEF.md and execute it end to end. Work on a branch. Do not report
> success until every item in "Acceptance criteria" passes. Then write the report
> described in "Reporting".

Claude Code will loop: edit → run → inspect outputs (including the PNGs) → check the
criteria → iterate. It only stops when the criteria are met or it hits a blocker it
must escalate.

---

## 0. Operating rules (read first, do not violate)

- Work on a new git branch (`robustness-pass`). Commit in logical steps. Never force-push.
- **Verify edits via host Read, not just the mount** — this repo has a known sandbox
  truncation issue where freshly-edited files read back truncated through the Linux
  mount. After each edit, confirm the file is intact before running.
- **Preserve BLADE parity.** Do not change `analyze_blade.py` defaults. J-specific
  values are passed by `run_j_co2.py`. `test_j_geometry.py` must still pass both halves.
- If a result contradicts a prior reported number, surface it loudly — do not quietly
  "fix" data to match expectations.
- If an acceptance criterion can't be met, stop and report why. Do not relax the
  criterion to declare success.

## 1. Invariants (these are correctness, not style)

- **fps is per-recording** (auto-detected by `probe_cine`). It MUST thread into
  `event_velocity.py`, `qa_restrike.py`, and `fourier_analysis.py`. A regression here
  previously mis-scaled every velocity and frequency by `true_fps/200000`.
- **px/mm** is set by the ~2.4 mm apex gap from `calibrate_j_electrodes.py` → `j_geometry.json`.
- **Intensity** is measured on raw DN (0–255), never on the per-frame normalized image.
- **Velocities** are reported in m/s; **frequencies** in Hz/kHz. Every reported value
  carries an uncertainty (see §3).
- Electrodes are finite hooked J-segments; geometry comes from calibration, enforced
  with `--finite-segments`.

## 2. Optimize + verify the measurement code

Target recording for the verification loop: "C:\Users\compl\Desktop\Plasma Arc\J Electrode\HS_0.4mm_1000mL_(CO2)_trig.cine"

1. Run the full pipeline on the target recording (`run_j_co2.py --from-stage convert`,
   then `analyze`). Capture stdout to a log.
2. Confirm the threaded fps appears in `run_metadata.json` and is used by stages 4/5.
3. Re-measure a handful of frames and confirm the threshold used matches the run's
   resolved `bright_thresh` (CLI > j_geometry.json > threshold_suggested.txt > 210).
4. Profile the slow stages; optimize only where it does not change numerical output
   (assert outputs identical before/after any optimization).

## 3. Uncertainty + filtering correctness

- Every velocity must carry error that includes **calibration (px/mm) and fps**
  uncertainty, not just the OLS slope SE currently in `vel_ms_err`. Add propagation.
- **Filtering audit:** document and verify the arc-validity criteria (`arc_filter.py`),
  the gap-row flash cutoff, and `--max-arc-px`. For the target recording, produce a
  before/after-filter frame montage and confirm rejected frames are genuinely invalid
  (saturation, flash, no arc) and accepted ones are genuinely arcs.
- **Sensitivity sweep:** rerun the headline metrics across a reasonable range of
  threshold, ROI, and local-velocity window N. Any conclusion that flips under
  reasonable choices is NOT a result — flag it.

## 4. Reynolds comparison (currently absent from the code — build it correctly)

- Define Re explicitly: characteristic length = nozzle diameter,
  characteristic velocity = bulk gas velocity from flow rate + nozzle area (NOT arc
  velocity), fluid = CO₂ at ambient room temperature conditions. Cite the property source.
- Propagate flow-rate and geometry uncertainty into Re → error bars on the Re axis.
- The sweep has repeats per (nozzle, flow) condition. Use the repeats for error bars /
  a repeat-aware fit, not single-run point comparisons.
- **Confound note:** changing nozzle diameter changes Re AND geometry. State this in the
  comparison; do not claim Re alone explains a trend without addressing it.
- Output plot(s): «SET THIS — e.g. arc velocity vs Re; cycle frequency vs Re; mode
  fraction vs Re», with error bars on both axes.

## 5. Paper-grade figure standard (every output figure must meet ALL of these)

- ≥ 300 DPI, vector (PDF/SVG) where possible plus a PNG preview.
- Axis labels with **units**; readable font sizes; no clipped labels.
- Error bars or uncertainty bands present wherever a measured quantity is plotted.
- Colorblind-safe palette; legend; caption text in a sidecar `.txt` stating recording
  stem, fps, px/mm, threshold, and N.
- Claude Code must **open and visually inspect each generated PNG** and confirm the
  above before declaring the figure done.

## 6. Acceptance criteria (all must pass before reporting success)

- [ ] `pytest` green, including `test_j_geometry.py` (both halves).
- [ ] Dimensional self-check: a synthetic 200 kHz vs 100 kHz case yields a velocity
      ratio of exactly 2.0 (guards the fps bug).
- [ ] `run_metadata.json` provenance complete (fps, px/mm, threshold+source, geometry,
      git hash, timestamp) for the verification run.
- [ ] Every reported velocity/frequency has an uncertainty including calibration + fps.
- [ ] Filtering montage reviewed; no genuine arc frames rejected, no junk accepted.
- [ ] Sensitivity sweep run; headline conclusions stable (or instability flagged).
- [ ] Reynolds comparison built with T/P-correct CO₂ properties, repeat-aware stats,
      and the confound stated.
- [ ] All figures meet the §5 standard (verified by visual inspection).

## 7. Reporting

Write `ROBUSTNESS_REPORT.md` containing:
1. What changed and why (per file), with commit hashes.
2. The verification-run numbers WITH uncertainties, vs any prior reported numbers —
   call out discrepancies explicitly.
3. The sensitivity-sweep table.
4. The Reynolds comparison: method, property sources, the plot(s), and the confound caveat.
5. A "what I could NOT verify / open risks" section — be honest about limits.
6. Links to every figure generated.
