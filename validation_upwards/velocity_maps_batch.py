"""Build the per-recording velocity maps (build_velocity_map.py) for annotated BLADE-upwards runs, with exactly the
arguments run_blade_upwards.py's --figures stage passes (detector thresholds from each run's run_metadata.json).

The map: for every advancing-front frame with a local tip velocity, the frame is re-binarised from the .cine and that
velocity is added to every arc pixel; map = MEAN over frames per pixel (mm/s in the .npy). The mean, not a median,
is what the quantised 5-frame velocity needs (fig_tip_kinematics / code review 2026-09-30).

Runs one recording at a time per worker, reading the .cine straight from the USB drive; skips runs whose
velocity_map_local.npy already exists (so an interrupted batch resumes).

usage: python velocity_maps_batch.py <runs_root> [--match SUBSTR] [--workers N]
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
MIN_ARC_PX = 22          # run_blade_upwards.MIN_ARC_PX (speckle floor)


def one(rd: Path):
    out = rd / "velocity_map_local.npy"
    if out.exists() and (rd / "velocity_map_count.npy").exists():
        return rd.name, "skip (exists)", 0.0
    m = json.loads((rd / "run_metadata.json").read_text(encoding="utf-8"))
    det, pxmm = m["detector"], float(m["calibration"]["px_per_mm"])
    cmd = [sys.executable, str(HERE / "build_velocity_map.py"), "--csv", str(rd / "measurements_all_events.csv"),
           "--cine", m["cine"], "--background", str(rd / "background_b10.npy"), "--out-dir", str(rd),
           "--px-per-mm", f"{pxmm:.5f}", "--fps", f"{float(m['fps']):.0f}",
           "--thresh", f"{det['T_high']:.3f}", "--low-thresh", f"{det['T_low']:.3f}", "--low-frac", str(m["low_frac"]),
           "--dark-guard", f"{det['T_high']:.3f}", "--min-arc-px", str(MIN_ARC_PX), "--max-arc-px", str(10 ** 9)]
    t0 = time.time()
    with open(rd / "velocity_map.log", "w", encoding="utf-8") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    return rd.name, f"rc={rc}" + ("" if out.exists() else " NO MAP"), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_root")
    ap.add_argument("--match", default="")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    runs = sorted(p.parent for p in Path(a.runs_root).rglob("run_metadata.json")
                  if a.match in p.parent.name and (p.parent / "measurements_all_events.csv").exists())
    print(f"{len(runs)} runs, {a.workers} workers", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for name, status, dt in ex.map(one, runs):
            print(f"{name}: {status} ({dt / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
