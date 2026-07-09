#!/usr/bin/env python3
"""
restrike_sensitivity.py  —  sensitivity bands for the tip-based re-strike metric.

The re-strike COUNT and the Mode-B FRACTION depend on two free parameters that
have no first-principles value:
  * ext_prominence_frac (0.10) — sets how many arc extensions are detected, hence
    the re-strike count.  Sweeping it moves the count by ~20-48% (QA report E1).
  * the Mode A/B cut norm_pos > 0.10 — for recordings whose norm_pos distribution
    piles up near 0.10 (notably 1.2mm: ~46% of events within +/-0.05 of the cut),
    a 0.05 shift swings frac_B from ~0.90 to ~0.53 (QA report E2).

This script does NOT change the authoritative qa_restrike.py outputs.  It re-runs
the SAME detector (qa_restrike.detect_restrikes/classify) over a grid of the two
parameters and reports, per recording:
  * re-strike count at prominence in {0.05, 0.10, 0.15, 0.20}  -> a count BAND
  * frac_B at threshold in {0.05, 0.10, 0.15} (prominence fixed 0.10) -> a B BAND
  * the fraction of events within +/-0.05 of the chosen 0.10 cut (cut fragility)
so every headline re-strike number can be quoted with an explicit sensitivity
band instead of a false point estimate.

Outputs (Reynolds_comparison/):
  restrike_sensitivity.csv          one row per (recording, prominence, threshold)
  restrike_sensitivity_summary.txt  the per-recording bands, human-readable

Usage:
    python restrike_sensitivity.py
    python restrike_sensitivity.py --data-root "C:\\...\\J Electrode"
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import qa_restrike
from npoint_restrike import (RECORDINGS, DATA_ROOT_DEFAULT, load_metadata,
                             detector_params, detected_noflash_frames,
                             MIN_CYCLE, SMOOTH)

PROM_GRID = [0.05, 0.10, 0.15, 0.20]
THR_GRID = [0.05, 0.10, 0.15]
PROM_REF, THR_REF = 0.10, 0.10


def _classify_counts(tip, erow, l_eff, prom, thr):
    rk, _, _ = qa_restrike.detect_restrikes(tip, erow, l_eff, prom, MIN_CYCLE, SMOOTH)
    n_b = 0
    npos_all = []
    for r in rk:
        k = r["idx"]
        npos = float(np.clip((tip[k] - erow) / l_eff, 0, 1))
        denom = max(r["prev_ext"] - erow, 1e-6)
        rfrac = float(np.clip((r["prev_ext"] - tip[k]) / denom, 0, 1))
        npos_all.append(npos)
        if qa_restrike.classify(npos, rfrac, "abs", thr, 0.70) == "B":
            n_b += 1
    return len(rk), n_b, np.array(npos_all)


def run(data_root: Path):
    rows = []
    summary = ["Re-strike sensitivity bands (tip detector)",
               "=" * 60,
               "count band = re-strikes over prominence {0.05..0.20}",
               "B band     = frac_B over threshold {0.05,0.10,0.15} at prominence 0.10",
               "cut-frag   = fraction of events within +/-0.05 of the 0.10 cut",
               ""]
    for stem, info in RECORDINGS.items():
        rd = data_root / f"{stem}_results"
        if not (rd / "measurements_all_events.csv").exists():
            continue
        meta = load_metadata(rd)
        p = detector_params(meta)
        frames, tip, _ = detected_noflash_frames(rd)
        apex = float(p["gap_row"]) if p["arc_direction"] == "down" \
            else float(meta["height"] - 1 - p["gap_row"])
        erow, l_eff, _ = qa_restrike.apex_and_range(tip, None, apex_override=apex)

        counts = {}
        fracB_by_thr = {}
        npos_ref = None
        for prom in PROM_GRID:
            for thr in THR_GRID:
                n, n_b, npos = _classify_counts(tip, erow, l_eff, prom, thr)
                rows.append(dict(file=stem, Re=info["re"], is_repeat=int(info["is_repeat"]),
                                 prominence=prom, threshold=thr, n_restrikes=n,
                                 n_B=n_b, frac_B=round(n_b / n, 4) if n else np.nan))
                if thr == THR_REF:
                    counts[prom] = n
                if prom == PROM_REF:
                    fracB_by_thr[thr] = (n_b / n) if n else np.nan
                    if npos_ref is None:
                        npos_ref = npos
        cband = (min(counts.values()), max(counts.values()))
        bband = (min(fracB_by_thr.values()), max(fracB_by_thr.values()))
        cut_frag = float(np.mean(np.abs(npos_ref - THR_REF) <= 0.05)) if npos_ref is not None and len(npos_ref) else np.nan
        ref_n = counts[PROM_REF]
        summary.append(
            f"{stem:<44} Re{info['re']:>5}{'(rep)' if info['is_repeat'] else '':5} "
            f"count {ref_n} [{cband[0]}-{cband[1]}]  "
            f"frac_B {fracB_by_thr[THR_REF]:.2f} [{bband[0]:.2f}-{bband[1]:.2f}]  "
            f"cut-frag {cut_frag:.0%}")
    outdir = data_root / "Reynolds_comparison"
    outdir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / "restrike_sensitivity.csv", index=False)
    (outdir / "restrike_sensitivity_summary.txt").write_text("\n".join(summary) + "\n",
                                                             encoding="utf-8")
    print("\n".join(summary))
    print(f"\nWrote {outdir/'restrike_sensitivity.csv'}")
    print(f"Wrote {outdir/'restrike_sensitivity_summary.txt'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    args = ap.parse_args()
    run(Path(args.data_root))


if __name__ == "__main__":
    main()
