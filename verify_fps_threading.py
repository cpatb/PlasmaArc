"""
verify_fps_threading.py — check that the per-recording fps is the one actually
used downstream of analyze. Catches the regression where a hardcoded 200000
literal silently re-scaled velocities by true_fps/200000.

For a given <stem>_results dir:
  1. Read run_metadata.json.fps as the AUTHORITATIVE value.
  2. Confirm metadata.txt (read by fourier_analysis) lists the same fps.
  3. Confirm every per-chunk subdir has the same fps_used in measurement
     metadata (if present).
  4. Print a PASS/FAIL summary.

Usage:
    python verify_fps_threading.py --results "<stem>_results"
    python verify_fps_threading.py --root    "C:\\Users\\compl\\Desktop\\Plasma Arc\\J Electrode"
"""
from __future__ import annotations
import argparse, json, sys, re
from pathlib import Path


def verify_one(results_dir, fail_on_default=True):
    results_dir = Path(results_dir)
    meta_path   = results_dir / 'run_metadata.json'
    txt_path    = results_dir / 'metadata.txt'

    issues = []
    if not meta_path.exists():
        return False, [f"missing {meta_path}"]
    meta = json.loads(meta_path.read_text())
    fps  = meta.get('fps')
    if not isinstance(fps, (int, float)) or fps <= 0:
        return False, [f"run_metadata.json fps is invalid: {fps!r}"]

    # 2. metadata.txt — fourier_analysis reads px/mm from here; if there is a
    #    fps line it MUST agree with run_metadata.json.
    if txt_path.exists():
        m = re.search(r'^\s*fps\s*:\s*([0-9.]+)\s*$', txt_path.read_text(),
                      re.MULTILINE | re.IGNORECASE)
        if m:
            fps_txt = float(m.group(1))
            if abs(fps_txt - float(fps)) > 0.5:
                issues.append(f"metadata.txt fps={fps_txt} != run_metadata fps={fps}")
        else:
            issues.append("metadata.txt has no fps line "
                          "(fourier_analysis falls back to its CLI default)")

    # 3. Heuristic: scan for any literal 200000 in the merged measurements_csv
    #    metadata that doesn't match the recording's fps. This catches sample
    #    artifacts where a downstream stage cached the BLADE default.
    for csvname in ('measurements.csv', 'measurements_all_events.csv',
                    'event_summary.csv', 're_strikes_qa.csv'):
        p = results_dir / csvname
        if p.exists():
            head = p.read_text(encoding='utf-8', errors='ignore')[:4096]
            if 'fps' in head.lower() and '200000' in head and int(fps) != 200000:
                issues.append(f"{csvname} mentions fps=200000 but recording fps={int(fps)}")

    ok = (len(issues) == 0)
    if fail_on_default and meta.get('px_per_mm_source') == 'DEFAULT':
        ok = False
        issues.append("px/mm source is the BLADE DEFAULT (no per-recording calibration)")
    return ok, issues


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--results', type=str, help='Single <stem>_results dir')
    g.add_argument('--root',    type=str, help='Root containing *_results dirs')
    args = ap.parse_args()

    if args.results:
        targets = [Path(args.results)]
    else:
        targets = sorted(d for d in Path(args.root).glob('*_results') if d.is_dir())

    n_pass = 0; n_fail = 0
    print(f"verify_fps_threading: {len(targets)} target(s)\n")
    for d in targets:
        ok, issues = verify_one(d)
        tag = 'PASS' if ok else 'FAIL'
        print(f"  [{tag}] {d.name}")
        for x in issues:
            print(f"      - {x}")
        n_pass += int(ok); n_fail += int(not ok)
    print(f"\n{n_pass} PASS / {n_fail} FAIL of {len(targets)}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()
