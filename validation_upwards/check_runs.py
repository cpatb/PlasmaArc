"""Completeness + consistency check of every BLADE-upwards annotated run (read-only).

Per recording (all .cine under the data root):
  status        DONE / NO_GLIDING / other / MISSING (no run directory or no metadata)
  files         every output a DONE run must have
  npoint        points on every detected frame (npoint_frames.with_points == detected) and method/version tags
  auto_vs_hand  the video's own automatic arc-attachment gap ends vs the hand cutoffs of its setup (px);
                > 4 px means the video may be in the wrong setup or the image moved
  flags         the run's recorded calibration flags
Prints a table of problems only, then totals; writes the full table to <out_csv>.

usage: python check_runs.py <data_root> <runs_root> <out_csv>
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_blade_upwards as rbu  # noqa: E402

REQUIRED = ["measurements.csv", "measurements_all_events.csv", "event_summary.csv", "re_strikes_qa.csv",
            "npoint_xy.npz", "npoint_restrikes.csv", "segment_velocities.csv", "npoint_summary.json",
            "run_metadata.json", "DONE.json"]
AUTO_HAND_MAX_PX = 4.0

data_root, runs_root, out_csv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = []
for cine in sorted(data_root.rglob("*.cine")):
    od = runs_root / cine.relative_to(data_root).parent / cine.stem
    r = dict(recording=cine.stem, condition=" / ".join(rbu.condition_of(cine) or ("?",)))
    mp = od / "run_metadata.json"
    if not mp.exists():
        r["status"] = "MISSING"
        rows.append(r)
        continue
    m = json.loads(mp.read_text())
    r["status"] = m.get("status", "?")
    if r["status"] == "DONE":
        miss = [f for f in REQUIRED if not (od / f).exists()]
        ev = od / "events"
        if not ((ev / "events_frames.csv").exists() or list(ev.glob("per_event_files*.zip"))
                or list(ev.glob("event_*_frames.csv"))):
            miss.append("events/(combined or zipped per-event files)")
        r["missing_files"] = ";".join(miss)
        npf = m.get("npoint_frames", {})
        r["npoint_ok"] = (npf.get("with_points") == npf.get("detected")
                          and m.get("npoint_method") == rbu.NPOINT_METHOD and m.get("npoint_n") == rbu.NPOINT_N)
        cal = m.get("calibration", {})
        ah = cal.get("auto_vs_hand_end_px")
        # the cross-check is only informative where the automatic calibration passed its own checks: on
        # spot-dominated videos it fails and its gap ends are spot feet, not the gap (e.g. Ar-CO2 Re3156 t3)
        auto_ok = m.get("calibration_auto", {}).get("status") == "OK"
        r["auto_cal_status"] = m.get("calibration_auto", {}).get("status")
        r["auto_vs_hand_max_px"] = max(ah) if (ah and auto_ok) else None
        r["flags"] = " | ".join(cal.get("flags", []))
        r["px_per_mm"] = cal.get("px_per_mm")
        r["complete_frames"] = m.get("frames", {}).get("arc_complete")
    rows.append(r)

T = pd.DataFrame(rows)
T.to_csv(out_csv, index=False)
done = T["status"] == "DONE"
prob = T[~T["status"].isin(["DONE", "NO_GLIDING"])
         | (done & (T.get("missing_files", "").fillna("") != ""))
         | (done & (T.get("npoint_ok") == False))  # noqa: E712
         | (done & (T.get("auto_vs_hand_max_px").fillna(0) > AUTO_HAND_MAX_PX))]
print(f"recordings {len(T)}: " + ", ".join(f"{k} {v}" for k, v in T["status"].value_counts().items()))
if len(prob):
    cols = [c for c in ["recording", "condition", "status", "missing_files", "npoint_ok", "auto_vs_hand_max_px"]
            if c in prob.columns]
    with pd.option_context("display.width", 250, "display.max_colwidth", 60, "display.max_rows", 200):
        print(prob[cols].to_string(index=False))
else:
    print("no problems")
if done.any():
    a = T.loc[done, "auto_vs_hand_max_px"].dropna()
    print(f"auto-vs-hand end miss over DONE runs: median {a.median():.2f} px, max {a.max():.2f} px, "
          f"> {AUTO_HAND_MAX_PX} px on {(a > AUTO_HAND_MAX_PX).sum()} runs")
print("wrote", out_csv)
