"""Pack each run's per-event CSVs (events/event_NNNN_*.csv) into ONE zip in that events/ folder.

Why: E: is exFAT with 512 KB clusters, so each ~1 KB per-event CSV occupies 512 KB on disk; ~580 000 of them
held 2.2 GB of data in ~283 GB and filled the drive (2026-09-29).

Safety, per events/ folder:
  1. build the zip on the LOCAL disk (--staging), ZIP_DEFLATED, one member per file (same name);
  2. verify: testzip() clean, member set == file set, and every member's bytes == the file's bytes read again;
  3. only then delete the loose files that were verified;
  4. move the zip into events/ as per_event_files.zip (or _2, _3 ... if one exists) and testzip() it there.
events_index.csv is never touched. A manifest row per folder is written to --manifest.
Stops at the first failed verification.

usage: python pack_event_files.py <root> [<root> ...] --staging DIR --manifest FILE.csv [--dry-run]
"""
import argparse
import csv
import hashlib
import shutil
import sys
import time
import zipfile
from pathlib import Path


def per_event_files(ev_dir: Path):
    return sorted(f for f in ev_dir.iterdir()
                  if f.is_file() and f.name.startswith("event_") and f.suffix == ".csv")


def pack_one(ev_dir: Path, staging: Path, dry: bool) -> dict:
    files = per_event_files(ev_dir)
    row = dict(events_dir=str(ev_dir), n_files=len(files), bytes=sum(f.stat().st_size for f in files),
               zip="", zip_bytes=0, status="")
    if not files:
        row["status"] = "nothing to pack"
        return row
    if dry:
        row["status"] = "dry-run"
        return row
    staging.mkdir(parents=True, exist_ok=True)
    tmp = staging / (hashlib.md5(str(ev_dir).encode()).hexdigest() + ".zip")
    tmp.unlink(missing_ok=True)
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(f, arcname=f.name)
    with zipfile.ZipFile(tmp) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"corrupt member {bad} in staging zip")
        if set(z.namelist()) != {f.name for f in files}:
            raise RuntimeError("zip members differ from the files")
        for f in files:
            if z.read(f.name) != f.read_bytes():
                raise RuntimeError(f"{f.name} differs between zip and disk")
    for f in files:                                  # every one verified above
        f.unlink()
    k, dst = 1, ev_dir / "per_event_files.zip"
    while dst.exists():
        k += 1
        dst = ev_dir / f"per_event_files_{k}.zip"
    shutil.move(str(tmp), str(dst))
    with zipfile.ZipFile(dst) as z:
        if z.testzip() is not None or len(z.namelist()) != len(files):
            raise RuntimeError(f"{dst} failed its check after the move (loose files already deleted)")
    row.update(zip=str(dst), zip_bytes=dst.stat().st_size, status="packed")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--staging", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rows, t0 = [], time.time()
    dirs = [d for r in a.roots for d in sorted(Path(r).rglob("events")) if d.is_dir()]
    print(f"{len(dirs)} events/ folders", flush=True)
    for i, d in enumerate(dirs, 1):
        try:
            rows.append(pack_one(d, Path(a.staging), a.dry_run))
        except Exception as e:  # noqa: BLE001 -- record and stop: never continue past a failed verification
            rows.append(dict(events_dir=str(d), status=f"ERROR {e}"))
            print(f"ERROR {d}: {e}", flush=True)
            break
        if i % 20 == 0 or i == len(dirs):
            n = sum(r.get("n_files", 0) for r in rows if r.get("status") == "packed")
            print(f"  {i}/{len(dirs)} folders, {n} files packed, {time.time() - t0:.0f} s", flush=True)
    Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(a.manifest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["events_dir", "n_files", "bytes", "zip", "zip_bytes", "status"])
        w.writeheader()
        w.writerows(rows)
    errs = [r for r in rows if str(r.get("status", "")).startswith("ERROR")]
    print(f"done: {sum(r.get('status') == 'packed' for r in rows)} packed, {len(errs)} errors; manifest {a.manifest}")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
