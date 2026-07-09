"""
apply_hand_calibration.py — make the hand-measured apex gaps the authoritative
per-recording calibration carried down the J-electrode pipeline.

Reads a free-text hand-calibration file (one line per recording, e.g.
"0.4 mm 1500 mL repeat calibration: 21/2.4 px/mm = 8.75 px/mm"), matches each
line to its <stem>_results directory by (diameter, flow, repeat/weird), and:

  * backs up j_geometry.json + run_metadata.json to __hand_calib_backup__/,
  * sets j_geometry.json  min_gap_px / min_gap_mm / px_per_mm  to the hand value
    (px/mm is derived downstream as min_gap_px/min_gap_mm, so this IS what the
    pipeline will use), with a provenance note,
  * sets run_metadata.json px_per_mm + px_per_mm_source (+ keeps the prior value
    as px_per_mm_before_hand), because run_metadata is only rewritten by the
    expensive analyze stage and we are restarting from `merge`.

This does NOT rescale any CSV or figure — that happens when you re-run the
pipeline from the merge stage (see regen_hand_calib.ps1). px/mm enters only as a
final scale, so a merge-stage restart reproduces a full re-detect exactly.

    python apply_hand_calibration.py --root "<J Electrode>" --calib "<...>.txt" --dry-run
    python apply_hand_calibration.py --root "<J Electrode>" --calib "<...>.txt" --apply
"""
from __future__ import annotations
import argparse, json, re, shutil
from pathlib import Path

BACKUP = '__hand_calib_backup__'


def parse_calib(text):
    """Return list of dicts: {d, q, repeat, weird, gap_px, px_per_mm}."""
    out = []
    for line in text.splitlines():
        s = line.strip().lower()
        if 'px/mm' not in s or 'mm' not in s:
            continue
        dm = re.search(r'([\d.]+)\s*mm', s)
        qm = re.search(r'(\d+)\s*ml', s)
        gap = re.search(r'(\d+(?:\.\d+)?)\s*/\s*3\.2', s)  # gap ref updated 2.4 -> 3.2 mm
        pm = re.findall(r'=\s*([\d.]+)\s*px/mm', s)
        if not (dm and qm and gap and pm):
            continue
        out.append(dict(d=float(dm.group(1)), q=int(qm.group(1)),
                        repeat=('repeat' in s), weird=('weird' in s),
                        gap_px=float(gap.group(1)), px_per_mm=float(pm[-1])))
    return out


def dir_key(name):
    n = name.lower()
    d = re.search(r'([\d.]+)mm', n)
    q = re.search(r'(\d+)ml', n)
    return dict(d=float(d.group(1)) if d else None,
                q=int(q.group(1)) if q else None,
                repeat=('repeat' in n), weird=('weird' in n))


def match(dirkey, calib):
    for c in calib:
        if (c['d'] == dirkey['d'] and c['q'] == dirkey['q']
                and c['repeat'] == dirkey['repeat']
                and c['weird'] == dirkey['weird']):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--apply', action='store_true', help='write changes (default: dry-run)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    dry = not args.apply

    root = Path(args.root)
    calib = parse_calib(Path(args.calib).read_text(encoding='utf-8'))
    print(f'parsed {len(calib)} hand-calibration lines\n')
    print(f"{'recording':46s} {'old px/mm':>9} {'new px/mm':>9} {'gap_px':>7} {'v factor':>9}")
    print('-' * 84)

    for d in sorted(root.glob('*_results')):
        if not d.is_dir():
            continue
        gp, mp = d / 'j_geometry.json', d / 'run_metadata.json'
        if not gp.exists():
            print(f'{d.name[:46]:46s}  [skip] no j_geometry.json'); continue
        c = match(dir_key(d.name), calib)
        if not c:
            print(f'{d.name[:46]:46s}  [skip] no matching calib line'); continue

        geom = json.loads(gp.read_text())
        meta = json.loads(mp.read_text()) if mp.exists() else {}
        old = meta.get('px_per_mm') or geom.get('px_per_mm')
        new = c['px_per_mm']
        factor = (old / new) if old else float('nan')   # velocity *= old/new
        print(f'{d.name[:46]:46s} {old:9.3f} {new:9.3f} {c["gap_px"]:7.1f} {factor:9.4f}')

        if dry:
            continue
        bak = d / BACKUP
        bak.mkdir(exist_ok=True)
        for f in (gp, mp):
            if f.exists() and not (bak / f.name).exists():
                shutil.copy2(f, bak / f.name)

        geom['min_gap_px'] = round(c['gap_px'], 3)
        geom['min_gap_mm'] = 3.2   # confirmed true minimum electrode gap (was 2.4)
        geom['px_per_mm'] = round(new, 4)
        geom['px_per_mm_source'] = 'hand-authoritative 25JUN26'
        geom['min_gap_px_note'] = (
            'hand-measured tip-to-tip gap (supersedes auto tip-detection); '
            'electrode_edges.npz tips left as-is for attachment geometry only')
        gp.write_text(json.dumps(geom, indent=2))

        if mp.exists():
            if old:
                meta['px_per_mm_before_hand'] = old
            meta['px_per_mm'] = round(new, 5)
            meta['px_per_mm_source'] = 'hand-authoritative 25JUN26 (min_gap_px/2.4)'
            mp.write_text(json.dumps(meta, indent=2))

    print('\n' + ('DRY RUN — nothing written. Re-run with --apply to commit.'
                  if dry else 'APPLIED. Backups in each <results>/__hand_calib_backup__/.'))
    print('v factor = old/new px/mm = how each velocity rescales after merge re-run.')


if __name__ == '__main__':
    main()
