"""Post-stage every finished nominal run that has no POST.json yet (pooled scale, no flash cut).

usage: post_all.py [--parallel 2]
Pooled scale 9.583 px/mm = 23.00 px / 2.4 mm (median of 26 calibrations, OVERNIGHT_LOG 'POOLED SCALE').
"""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
RUNS = Path(r"E:\BladeReactor_Upwards_analysis\runs")
POOLED_PX_PER_MM = 23.00 / 2.4


def todo():
    for d in sorted({p.parent for p in RUNS.rglob("DONE.json")}):
        m = json.loads((d / "DONE.json").read_text())
        if m.get("status") == "DONE" and not (d / "POST.json").exists():
            yield d


def one(d):
    r = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "validation_upwards" / "post_stage.py"), str(d),
                        "--eff-width", "0", "--px-per-mm", f"{POOLED_PX_PER_MM:.6f}"], capture_output=True, text=True)
    print(("[post ok] " if r.returncode == 0 else f"[post FAIL rc={r.returncode}] ") + d.name, flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--parallel", type=int, default=2)
a = ap.parse_args()
dirs = list(todo())
print(f"{len(dirs)} runs to post-stage", flush=True)
with ThreadPoolExecutor(a.parallel) as ex:
    list(ex.map(one, dirs))
print("post_all finished", flush=True)
