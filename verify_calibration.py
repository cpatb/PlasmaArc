#!/usr/bin/env python3
"""
verify_calibration.py  —  guard against the calibration-provenance class of bug.

The J pipeline derives px/mm per recording as ``min_gap_px / min_gap_mm`` from
``j_geometry.json`` (``run_j_co2.resolve_px_per_mm``), but the published results
record the px/mm actually used in ``run_metadata.json``.  When a re-segmentation
overwrites ``min_gap_px`` without re-running analyze, the two silently diverge and
a routine re-run changes every velocity (this happened to 0.8mm(2) and 1.2mm —
see the QA report).  This script asserts, for every ``<stem>_results/`` dir, that

    | min_gap_px / min_gap_mm  −  run_metadata.px_per_mm |  < tol

so the geometry on disk reproduces the published scale.  Run it after any
recalibration / resegmentation, and in CI.

Usage:
    python verify_calibration.py
    python verify_calibration.py --data-root "C:\\...\\J Electrode" --tol 1e-3
Exit code 0 = all reproduce; 1 = at least one mismatch.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

DATA_ROOT_DEFAULT = Path(r"C:\Users\compl\Desktop\Plasma Arc\J Electrode")


def _load_json_robust(p: Path) -> dict:
    t = p.read_text(encoding="utf-8", errors="replace")
    return json.loads(t[:t.rfind("}") + 1])


def check(data_root: Path, tol: float = 1e-3) -> int:
    results = sorted(data_root.glob("*_results"))
    n_ok = n_bad = n_skip = 0
    print(f"{'recording':<46} {'geom px/mm':>11} {'meta px/mm':>11}  verdict")
    for rd in results:
        gj, mj = rd / "j_geometry.json", rd / "run_metadata.json"
        if not gj.exists() or not mj.exists():
            continue
        try:
            g = _load_json_robust(gj)
            m = _load_json_robust(mj)
        except Exception as e:  # noqa: BLE001
            print(f"{rd.name:<46} {'?':>11} {'?':>11}  SKIP ({e})")
            n_skip += 1
            continue
        gpx, gmm = g.get("min_gap_px"), g.get("min_gap_mm")
        meta_px = m.get("px_per_mm")
        if not gpx or not gmm or not meta_px:
            print(f"{rd.name:<46} {'n/a':>11} {meta_px!s:>11}  SKIP (missing field)")
            n_skip += 1
            continue
        derived = gpx / gmm
        ok = abs(derived - meta_px) < tol
        print(f"{rd.name:<46} {derived:>11.4f} {meta_px:>11.4f}  "
              f"{'OK' if ok else 'MISMATCH'}")
        n_ok += ok
        n_bad += (not ok)
    print(f"\n{n_ok} OK, {n_bad} MISMATCH, {n_skip} skipped")
    return 1 if n_bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    sys.exit(check(Path(args.data_root), args.tol))


if __name__ == "__main__":
    main()
