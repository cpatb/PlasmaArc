#!/usr/bin/env python3
"""Assemble a shareable master dataset of LEADING-ARC-TIP J-electrode data.

Scope
-----
Concatenates the *tip-level* processed outputs across all J-electrode CO2
recordings into a small bundle a collaborator can open directly.  Only the
leading-arc *tip* products are included; the N-point / segment-resolved
products (`npoint_restrikes*.csv`, `segment_velocities*.csv`,
`*_npoint.csv`, N20/N50 variants) are deliberately excluded.

Three granularities are emitted, each concatenated over all recordings with
per-recording condition + calibration columns prepended:

  master_per_frame.csv     one row per frame  (measurements.csv)
  master_per_restrike.csv  one row per leading-arc re-strike, authoritative
                           Mode A/B            (re_strikes_qa.csv)
  master_per_event.csv     one row per front-advance event, fitted velocity
                           + uncertainty       (events/events_index.csv)

plus:

  recordings_index.csv     one row per recording (calibration, counts)
  DATA_DICTIONARY.md       column definitions + units for every table

The join key across all tables is `recording_id` (== the source `file`
column, verified unique per recording).

Run from anywhere; paths are resolved relative to the J-Electrode data dir.
"""
from __future__ import annotations

import csv
import glob
import json
import re
import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Locate the J-Electrode data directory (Windows OneDrive Known-Folder-Move
# can put Desktop at ~/Desktop or ~/OneDrive/Desktop; try both).
# --------------------------------------------------------------------------
def find_data_dir() -> Path:
    home = Path.home()
    candidates = [
        home / "Desktop" / "Plasma Arc" / "J Electrode",
        home / "OneDrive" / "Desktop" / "Plasma Arc" / "J Electrode",
        Path("C:/Users/compl/Desktop/Plasma Arc/J Electrode"),
    ]
    for c in candidates:
        if c.is_dir():
            return c
    sys.exit("Could not locate the 'J Electrode' data directory.")


DATA_DIR = find_data_dir()
OUT_DIR = DATA_DIR / "j_master_tip"

# Condition/calibration columns prepended to every master table, in order.
CONTEXT_COLS = [
    "recording_id", "nozzle_mm", "flow_mL_min", "gas",
    "is_repeat", "is_flagged_weird", "px_per_mm", "fps_hz",
]


def parse_condition(file_stem: str) -> dict:
    """Extract experimental condition from the recording stem, e.g.
    'HS_0.8mm_1500mL_(CO2)_trig_repeat_weird(2)'."""
    nozzle = re.search(r"([0-9]+\.?[0-9]*)\s*mm", file_stem)
    flow = re.search(r"([0-9]+)\s*mL", file_stem)
    gas = re.search(r"\(([^)]+)\)", file_stem)
    return {
        "recording_id": file_stem,
        "nozzle_mm": float(nozzle.group(1)) if nozzle else "",
        "flow_mL_min": int(flow.group(1)) if flow else "",
        "gas": gas.group(1) if gas else "",
        "is_repeat": int("repeat" in file_stem.lower()),
        "is_flagged_weird": int("weird" in file_stem.lower()),
    }


def load_meta(result_dir: Path) -> dict:
    p = result_dir / "run_metadata.json"
    return json.loads(p.read_text()) if p.exists() else {}


def recording_context(result_dir: Path, file_stem: str) -> dict:
    """Full context dict (CONTEXT_COLS) for one recording."""
    ctx = parse_condition(file_stem)
    meta = load_meta(result_dir)
    ctx["px_per_mm"] = meta.get("px_per_mm", "")
    ctx["fps_hz"] = meta.get("fps", "")
    return ctx


def file_stem_of(result_dir: Path) -> str:
    """Authoritative recording_id = the `file` value inside re_strikes_qa.csv."""
    p = result_dir / "re_strikes_qa.csv"
    with p.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        row = next(r)
        return row["file"]


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------
def main() -> None:
    result_dirs = sorted(DATA_DIR.glob("*_results"))
    result_dirs = [d for d in result_dirs if (d / "re_strikes_qa.csv").exists()]
    if not result_dirs:
        sys.exit("No *_results directories with re_strikes_qa.csv found.")

    OUT_DIR.mkdir(exist_ok=True)
    print(f"Data dir : {DATA_DIR}")
    print(f"Out dir  : {OUT_DIR}")
    print(f"Recordings ({len(result_dirs)}):")

    contexts = {}   # result_dir -> context dict
    stems = {}
    for d in result_dirs:
        stem = file_stem_of(d)
        stems[d] = stem
        contexts[d] = recording_context(d, stem)
        print(f"  - {stem}")

    index_rows = []

    # ---- per-frame (streamed; the 1.2 mm run alone is ~873k rows) ----------
    per_frame_path = OUT_DIR / "master_per_frame.csv"
    total_frames = 0
    with per_frame_path.open("w", newline="", encoding="utf-8") as fout:
        writer = None
        for d in result_dirs:
            src = d / "measurements.csv"
            ctx = contexts[d]
            with src.open(encoding="utf-8") as fin:
                reader = csv.reader(fin)
                header = next(reader)
                if writer is None:
                    writer = csv.writer(fout)
                    writer.writerow(CONTEXT_COLS + header)
                ctx_vals = [ctx[c] for c in CONTEXT_COLS]
                n = 0
                for row in reader:
                    writer.writerow(ctx_vals + row)
                    n += 1
            total_frames += n
            print(f"  [per-frame] {stems[d]}: {n} frames")
    print(f"Wrote {per_frame_path.name}: {total_frames} rows")

    # ---- per-restrike & per-event (small; pandas) --------------------------
    def concat_table(rel: str, out_name: str) -> pd.DataFrame:
        parts = []
        for d in result_dirs:
            src = d / rel
            df = pd.read_csv(src)
            ctx = contexts[d]
            for c in reversed(CONTEXT_COLS):
                df.insert(0, c, ctx[c])
            parts.append(df)
        out = pd.concat(parts, ignore_index=True)
        out.to_csv(OUT_DIR / out_name, index=False)
        print(f"Wrote {out_name}: {len(out)} rows")
        return out

    restrikes = concat_table("re_strikes_qa.csv", "master_per_restrike.csv")
    events = concat_table("events/events_index.csv", "master_per_event.csv")

    # ---- recordings index --------------------------------------------------
    for d in result_dirs:
        stem = stems[d]
        ctx = contexts[d]
        meta = load_meta(d)
        rsub = restrikes[restrikes["recording_id"] == stem]
        esub = events[events["recording_id"] == stem]
        # arc-detected frame count (cheap re-scan of measurements 'arc_detected')
        n_detected = 0
        with (d / "measurements.csv").open(encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if row.get("arc_detected") == "1":
                    n_detected += 1
        valley_cut = ""
        if "valley_cut_mm" in rsub.columns and len(rsub):
            valley_cut = rsub["valley_cut_mm"].dropna().iloc[0] if rsub["valley_cut_mm"].notna().any() else ""
        index_rows.append({
            **{c: ctx[c] for c in CONTEXT_COLS},
            "recording_dir": d.name,
            "source_cine": meta.get("recording", ""),
            "n_frames": meta.get("n_frames", ""),
            "n_frames_arc_detected": n_detected,
            "n_restrikes": len(rsub),
            "n_mode_A": int((rsub["mode"] == "A").sum()) if len(rsub) else 0,
            "n_mode_B": int((rsub["mode"] == "B").sum()) if len(rsub) else 0,
            "n_events": len(esub),
            "px_per_mm_source": meta.get("px_per_mm_source", ""),
            "bright_thresh_DN": meta.get("bright_thresh", ""),
            "gap_row": meta.get("gap_row", ""),
            "arc_direction": meta.get("arc_direction", ""),
            "valley_cut_mm": valley_cut,
            "code_version": meta.get("code_version", ""),
        })
    idx = pd.DataFrame(index_rows)
    idx.to_csv(OUT_DIR / "recordings_index.csv", index=False)
    print(f"Wrote recordings_index.csv: {len(idx)} rows")

    # ---- validation --------------------------------------------------------
    assert total_frames == int(idx["n_frames"].astype(int).sum()), \
        "per-frame row total != sum of n_frames"
    assert len(restrikes) == int(idx["n_restrikes"].sum())
    assert len(events) == int(idx["n_events"].sum())
    assert restrikes["recording_id"].nunique() == len(result_dirs)
    print("\nValidation OK.")
    print(idx[["recording_id", "nozzle_mm", "flow_mL_min", "px_per_mm",
               "n_frames", "n_restrikes", "n_mode_A", "n_mode_B",
               "n_events"]].to_string(index=False))


if __name__ == "__main__":
    main()
