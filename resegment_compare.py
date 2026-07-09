"""
resegment_compare.py — snapshot + before/after comparison for re-segmenting the
J-electrode geometry.

WHY: the electrode trace (electrode_edges.npz) gates arc recognition. In
analyze_blade.py a frame is only recorded as arc_detected=True if its feet attach
to BOTH traced electrode edges within elec_tol_px; otherwise it is recorded as
"no arc". So a too-short trace silently drops arcs that attach to the un-traced
electrode. Re-tracing the full electrode can therefore change recognition, and the
only way to measure it is to snapshot the current outputs, re-segment + re-run
analyze, then diff.

USAGE
-----
1) BEFORE re-segmenting, snapshot the current outputs:
       python resegment_compare.py snapshot --root "<J Electrode>"
   -> copies geometry + key CSVs into each <results>/__pre_resegment_snapshot__/

2) Re-segment each recording (needs a display), keeping the facing tips put and
   tracing the FULL electrode edge; pass the confirmed gap so the calibration
   basis is consistent:
       python segment_electrodes.py --results "<...>_results" --gap-mm 3.2 --backend matplotlib
   Then RESTORE the 3.2 mm hand calibration (segment_electrodes rewrites px/mm
   from the new clicks):
       python apply_hand_calibration.py --root "<J Electrode>" --calib hand_calibration_authoritative_25JUN26.txt --apply

3) Re-run detection with the new geometry (per recording):
       python run_j_co2.py --cine "<name>.cine" --from-stage analyze --bright-thresh <N>

4) Diff against the snapshot:
       python resegment_compare.py compare --root "<J Electrode>"
"""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
import numpy as np
import pandas as pd

SNAP = "__pre_resegment_snapshot__"
KEY_FILES = ["electrode_edges.npz", "j_geometry.json", "run_metadata.json",
             "measurements_all_events.csv", "event_summary_with_unc.csv",
             "event_summary.csv", "re_strikes_qa.csv"]
FPS = 200000


def _metrics(results_dir):
    d = Path(results_dir)
    out = {}
    mp = d / "measurements_all_events.csv"
    if mp.exists():
        m = pd.read_csv(mp)
        det = m["arc_detected"] == True
        out["n_arc_frames"] = int(det.sum())
        out["arc_length_mm_med"] = float(m.loc[det, "arc_length_mm"].median()) if det.any() else np.nan
    ep = d / "event_summary_with_unc.csv"
    ep = ep if ep.exists() else d / "event_summary.csv"
    if ep.exists():
        e = pd.read_csv(ep)
        out["n_events"] = int(len(e))
        if "local_vel_median" in e:
            out["local_vel_median"] = float(e["local_vel_median"].median())
        out["vel_ms_median"] = float(e["vel_ms"].median()) if "vel_ms" in e else np.nan
    qp = d / "re_strikes_qa.csv"
    if qp.exists():
        q = pd.read_csv(qp); qc = q[q["cycle_frames"] > 0]
        nB = int((qc["mode"] == "B").sum()); nA = int((qc["mode"] == "A").sum())
        out["n_restrikes"] = len(qc)
        out["frac_B"] = nB / (nA + nB) if (nA + nB) else np.nan
    gp = d / "j_geometry.json"
    if gp.exists():
        g = json.loads(gp.read_text())
        out["px_per_mm"] = g.get("px_per_mm")
        out["edge_pts_A"] = out["edge_pts_B"] = None
    epz = d / "electrode_edges.npz"
    if epz.exists():
        z = np.load(epz, allow_pickle=True)
        out["edge_pts_A"] = int(len(z["A"])); out["edge_pts_B"] = int(len(z["B"]))
    return out


def do_snapshot(root):
    for d in sorted(Path(root).glob("*_results")):
        if not d.is_dir():
            continue
        snap = d / SNAP; snap.mkdir(exist_ok=True)
        copied = []
        for f in KEY_FILES:
            src = d / f
            if src.exists() and not (snap / f).exists():
                shutil.copy2(src, snap / f); copied.append(f)
        (snap / "metrics.json").write_text(json.dumps(_metrics(d), indent=2))
        print(f"[snap] {d.name}: {len(copied)} files + metrics.json")
    print("\nSnapshot done. Re-segment + re-run analyze, then: compare")


def do_compare(root):
    rows = []
    for d in sorted(Path(root).glob("*_results")):
        snap = d / SNAP
        if not (snap / "metrics.json").exists():
            continue
        before = json.loads((snap / "metrics.json").read_text())
        after = _metrics(d)
        for k in sorted(set(before) | set(after)):
            b, a = before.get(k), after.get(k)
            if isinstance(b, (int, float)) and isinstance(a, (int, float)) and b not in (None, 0):
                pct = 100.0 * (a - b) / b
            else:
                pct = np.nan
            rows.append(dict(recording=d.name, metric=k, before=b, after=a, pct_change=pct))
    df = pd.DataFrame(rows)
    if df.empty:
        print("No snapshots found — run `snapshot` first, then re-segment + re-run.")
        return
    pd.set_option("display.width", 200)
    for rec, g in df.groupby("recording"):
        print(f"\n=== {rec} ===")
        print(g[["metric", "before", "after", "pct_change"]].to_string(index=False))
    df.to_csv(Path(root) / "resegment_comparison.csv", index=False)
    print(f"\nwrote {Path(root) / 'resegment_comparison.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["snapshot", "compare"])
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    (do_snapshot if a.cmd == "snapshot" else do_compare)(a.root)


if __name__ == "__main__":
    main()
