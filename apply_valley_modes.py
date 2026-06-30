#!/usr/bin/env python3
"""
apply_valley_modes.py  —  adopt the DEAD-ZONE (valley) cut as the authoritative
Mode A/B definition in re_strikes_qa.csv.

The re-strike initiation-depth distribution (distance of the re-strike tip below
the apex) is BIMODAL: a near-apex "reset" peak (Mode A) and a deeper
"back-breakdown" peak (Mode B), separated by a sparse dead zone (~4 mm). The
original norm_pos>0.10 cut (~1.5 mm) slices through the FIRST peak, mislabelling
the upper half of the near-apex reset population as Mode B. This script moves the
cut to the valley between the two peaks so the whole first peak is Mode A and only
the deep peak is Mode B (see valley_reclassify.py for the before/after analysis).

For each recording it (idempotently, reversibly):
  * backs up re_strikes_qa.csv -> re_strikes_qa__pre_valley_backup.csv (once),
  * preserves the original modes in a `mode_normpos010` column,
  * computes depth_below_apex_mm = norm_pos * l_eff_px / px_per_mm,
  * sets the valley cut (auto-detected between the <=3 mm peak and the deeper peak,
    or a per-recording / global override) and writes it to `valley_cut_mm`,
  * rewrites `mode` = 'A' if depth < valley else 'B',
  * writes provenance to re_strikes_qa_modedef.json.

Every downstream consumer that reads the `mode` column (mode_transition_analysis,
restrike_location_maps, restrike_geometry_predictor, npoint_arc_stats cycle mode,
reset_interval_analysis, the paper figures' _mode_dynamics_data) then uses the
dead-zone definition automatically.

    python apply_valley_modes.py             # all recordings, auto valley
    python apply_valley_modes.py --cut-mm 4.0
    python apply_valley_modes.py --restore   # undo: restore from the backup
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

import numpy as np
import pandas as pd

from npoint_restrike import RECORDINGS
from valley_reclassify import find_valley

DATA_ROOT_DEFAULT = Path(r"C:\Users\compl\Desktop\Plasma Arc\J Electrode")
BACKUP = "re_strikes_qa__pre_valley_backup.csv"
# optional per-recording manual valley overrides (mm below apex); else auto
VALLEY_OVERRIDE: dict = {}


def _meta_pxmm(rd: Path) -> float:
    txt = (rd / "run_metadata.json").read_text(encoding="utf-8", errors="replace")
    return float(json.loads(txt.rsplit("}", 1)[0] + "}")["px_per_mm"])


def apply_one(stem: str, data_root: Path, cut_mm=None) -> dict:
    rd = data_root / f"{stem}_results"
    csv = rd / "re_strikes_qa.csv"
    bak = rd / BACKUP
    if not bak.exists():
        shutil.copy2(csv, bak)
    # always work from the ORIGINAL (backup) so re-runs are idempotent
    df = pd.read_csv(bak)
    orig_mode = df["mode"].copy()
    df = df[df["mode"].isin(["A", "B"]) | df["mode"].notna()].copy()
    ppm = _meta_pxmm(rd)
    npv = pd.to_numeric(df["norm_pos"], errors="coerce")
    leff = pd.to_numeric(df["l_eff_px"], errors="coerce")
    depth = (npv * leff / ppm).to_numpy()

    if cut_mm is not None:
        valley = float(cut_mm); src = f"global override {cut_mm} mm"
    elif stem in VALLEY_OVERRIDE:
        valley = float(VALLEY_OVERRIDE[stem]); src = "per-recording override"
    else:
        v = find_valley(depth)
        valley = float(v["valley"]) if v else float("nan"); src = "auto dead-zone valley"

    ab = pd.Series(df["mode"].values, index=df.index)
    valid = ab.isin(["A", "B"])
    new_mode = np.where(depth < valley, "A", "B")
    df["mode_normpos010"] = orig_mode.values
    df["depth_below_apex_mm"] = np.round(depth, 3)
    df["valley_cut_mm"] = round(valley, 3)
    df.loc[valid, "mode"] = new_mode[valid.values]
    df.to_csv(csv, index=False)

    old_fracB = float((orig_mode[valid.values] == "B").mean())
    new_fracB = float((pd.Series(new_mode)[valid.values] == "B").mean())
    prov = dict(definition="dead-zone valley cut on tip initiation depth",
                valley_cut_mm=round(valley, 3), valley_cut_px=round(valley * ppm, 1),
                source=src, supersedes="norm_pos>0.10 (qa_restrike default)",
                old_frac_B=round(old_fracB, 4), new_frac_B=round(new_fracB, 4),
                n=int(valid.sum()), px_per_mm=ppm)
    (rd / "re_strikes_qa_modedef.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return dict(stem=stem, **prov)


def apply_one_npoint(stem: str, data_root: Path, cut_mm=None) -> dict:
    """Same dead-zone valley reclassification for the 10-segment (N-point) detector,
    using its whole-arc DEEPEST-point depth (deepest_depth_mm): Mode A only when the
    whole arc has reset near the apex (first peak), Mode B otherwise (deep peak)."""
    rd = data_root / f"{stem}_results"
    csv = rd / "npoint_restrikes.csv"
    if not csv.exists():
        return dict(stem=stem, skipped=True)
    bak = rd / "npoint_restrikes__pre_valley_backup.csv"
    if not bak.exists():
        shutil.copy2(csv, bak)
    df = pd.read_csv(bak)
    orig = df["mode"].copy()
    depth = pd.to_numeric(df["deepest_depth_mm"], errors="coerce").to_numpy()
    if cut_mm is not None:
        valley = float(cut_mm); src = f"global override {cut_mm} mm"
    elif stem in VALLEY_OVERRIDE:
        valley = float(VALLEY_OVERRIDE[stem]); src = "per-recording override"
    else:
        v = find_valley(depth)
        valley = float(v["valley"]) if v else float("nan"); src = "auto dead-zone valley"
    valid = df["mode"].isin(["A", "B"])
    new_mode = np.where(depth < valley, "A", "B")
    df["mode_deepest010"] = orig.values
    df["valley_cut_mm"] = round(valley, 3)
    df.loc[valid, "mode"] = new_mode[valid.values]
    df.to_csv(csv, index=False)
    old_fb = float((orig[valid.values] == "B").mean())
    new_fb = float((pd.Series(new_mode)[valid.values] == "B").mean())
    prov = dict(definition="dead-zone valley cut on N-point deepest-arc depth",
                valley_cut_mm=round(valley, 3), source=src,
                supersedes="deepest_depth>0.10*L_eff (npoint_restrike default)",
                old_frac_B=round(old_fb, 4), new_frac_B=round(new_fb, 4), n=int(valid.sum()))
    (rd / "npoint_restrikes_modedef.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    return dict(stem=stem, **prov)


def restore_one(stem: str, data_root: Path) -> bool:
    rd = data_root / f"{stem}_results"
    ok = False
    for bk, tgt in [(BACKUP, "re_strikes_qa.csv"),
                    ("npoint_restrikes__pre_valley_backup.csv", "npoint_restrikes.csv")]:
        if (rd / bk).exists():
            shutil.copy2(rd / bk, rd / tgt)
            ok = True
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    ap.add_argument("--cut-mm", type=float, default=None,
                    help="global valley cut (mm); default = auto per recording")
    ap.add_argument("--restore", action="store_true",
                    help="restore re_strikes_qa.csv from the pre-valley backup")
    args = ap.parse_args()
    data_root = Path(args.data_root)

    if args.restore:
        for stem in RECORDINGS:
            ok = restore_one(stem, data_root)
            print(f"  {'restored' if ok else 'no backup'}: {stem}")
        return

    print(f"{'recording':44s} {'det':>6} {'valley_mm':>9} {'fracB old->new':>16} {'n':>6}")
    for stem in RECORDINGS:
        r = apply_one(stem, data_root, cut_mm=args.cut_mm)
        print(f"{stem:44s} {'tip':>6} {r['valley_cut_mm']:9.2f} "
              f"{r['old_frac_B']:.3f}->{r['new_frac_B']:.3f}   {r['n']:6d}")
        rn = apply_one_npoint(stem, data_root, cut_mm=args.cut_mm)
        if not rn.get("skipped"):
            print(f"{'':44s} {'npoint':>6} {rn['valley_cut_mm']:9.2f} "
                  f"{rn['old_frac_B']:.3f}->{rn['new_frac_B']:.3f}   {rn['n']:6d}")
    print("\nAuthoritative Mode A/B (tip re_strikes_qa.csv AND 10-segment "
          "npoint_restrikes.csv) is now the dead-zone valley cut.")
    print("Originals preserved in *__pre_valley_backup.csv + mode_*010 columns.")


if __name__ == "__main__":
    main()
