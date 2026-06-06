#!/usr/bin/env python3
"""
reset_derived.py  -  Safely clear DERIVED pipeline outputs so a recording can be
re-analysed from scratch with the current code, without touching raw data or
calibration.

What it PRESERVES (never deleted):
  * <recording>.cine                 raw camera recordings (the ground truth)
  * *.mp4                            preview clips
  * *.md                            notes / assessments
  * Calibrate/                      calibration inputs
  * <...>_results/blade_lines.npy   per-recording electrode geometry
                                    (unless --regen-blade-lines)
  * <...>_results/tiff/             decoded TIFF chunks
                                    (unless --include-tiffs)

What it DELETES (derived, reproducible):
  * everything else inside each <recording>_results/ dir
    (per-chunk *_results/ subdirs, measurements*.csv, qa_frames/, events/,
     velocity-map .npy, all figures, ...)
  * top-level cross-recording outputs: mode_compare/, mode_sample_frames/,
    mode_threshold_sweep/, mode_*.png/.csv, restrike_*.png, threshold_qa_*.png

DRY-RUN BY DEFAULT.  Pass --apply to actually delete.

Usage:
    python reset_derived.py                      # dry run (default base path)
    python reset_derived.py --apply              # delete (keep tiffs + blade_lines)
    python reset_derived.py --apply --include-tiffs        # also re-decode from .cine
    python reset_derived.py --apply --regen-blade-lines    # also re-detect blade lines
    python reset_derived.py --base "D:/path/to/BLADE 11MAY26"
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

DEFAULT_BASE = Path.home() / 'OneDrive' / 'Desktop' / 'BLADE' / 'BLADE 11MAY26'

# Top-level entries that are RAW / calibration / notes -> always preserved.
KEEP_TOP_SUFFIXES = {'.cine', '.mp4', '.md'}
KEEP_TOP_NAMES = {'Calibrate'}


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def path_size(path):
    if path.is_dir():
        return dir_size(path)
    try:
        return path.getsize() if hasattr(path, 'getsize') else os.path.getsize(path)
    except OSError:
        return 0


def human(n):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if n < 1024:
            return f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} PB'


def collect_targets(base, include_tiffs, regen_blade_lines):
    """Return list of (path, size_bytes) to delete."""
    targets = []
    for entry in sorted(base.iterdir()):
        name = entry.name
        # raw / calibration / notes -> preserve
        if entry.suffix.lower() in KEEP_TOP_SUFFIXES or name in KEEP_TOP_NAMES:
            continue
        if entry.is_dir() and name.endswith('_results'):
            # recording results container: clean INSIDE, preserving tiff/ + blade_lines
            for sub in sorted(entry.iterdir()):
                if sub.name == 'tiff' and not include_tiffs:
                    continue
                if sub.name == 'blade_lines.npy' and not regen_blade_lines:
                    continue
                targets.append(sub)
        else:
            # top-level derived file or cross-recording output dir
            targets.append(entry)
    return targets


def main():
    ap = argparse.ArgumentParser(description='Clear derived pipeline outputs (dry-run by default).')
    ap.add_argument('--base', type=Path, default=DEFAULT_BASE,
                    help='Dataset root (default: %(default)s)')
    ap.add_argument('--apply', action='store_true',
                    help='Actually delete. Without this, only prints what WOULD be deleted.')
    ap.add_argument('--include-tiffs', action='store_true',
                    help='Also delete the decoded tiff/ chunks (forces re-decode from .cine).')
    ap.add_argument('--regen-blade-lines', action='store_true',
                    help='Also delete blade_lines.npy (forces electrode re-detection).')
    args = ap.parse_args()

    base = args.base
    if not base.is_dir():
        sys.exit(f'ERROR: base path not found: {base}')

    targets = collect_targets(base, args.include_tiffs, args.regen_blade_lines)

    # Hard safety net: never delete a .cine under any circumstance.
    safe = []
    for t in targets:
        if t.suffix.lower() == '.cine' or (t.is_dir() and any(p.suffix.lower() == '.cine' for p in t.rglob('*'))):
            print(f'  [SKIP - contains/.cine] {t}')
            continue
        safe.append(t)
    targets = safe

    total = 0
    print(f'{"APPLY" if args.apply else "DRY RUN"} - base: {base}')
    print(f'  preserve tiffs       : {not args.include_tiffs}')
    print(f'  preserve blade_lines : {not args.regen_blade_lines}\n')
    print(f'{"WOULD DELETE" if not args.apply else "DELETING"} ({len(targets)} items):')
    for t in targets:
        sz = dir_size(t) if t.is_dir() else (os.path.getsize(t) if t.exists() else 0)
        total += sz
        kind = 'dir ' if t.is_dir() else 'file'
        rel = t.relative_to(base)
        print(f'  {kind}  {human(sz):>10}   {rel}')
    print(f'\nTotal: {human(total)} across {len(targets)} items.')

    if not args.apply:
        print('\nDry run only. Re-run with --apply to delete.')
        return

    print('\nDeleting...')
    for t in targets:
        try:
            if t.is_dir():
                shutil.rmtree(t)
            else:
                t.unlink()
        except OSError as e:
            print(f'  [ERROR] {t}: {e}')
    print('Done. Raw .cine, Calibrate/, notes'
          + ('' if args.include_tiffs else ', tiff/')
          + ('' if args.regen_blade_lines else ', blade_lines.npy')
          + ' preserved.')


if __name__ == '__main__':
    main()
