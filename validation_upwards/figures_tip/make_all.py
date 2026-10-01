"""Regenerate the whole BLADE-upwards leading-tip figure suite in order (one command).

usage: python make_all.py [--no-cache]   (--no-cache skips rebuilding frames_cache.parquet, ~4 min)
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = ["build_frames_cache.py", "fig_tip_kinematics.py", "fig_velocity_distributions.py",
           "fig_velocity_vs_height.py", "fig_flow_scaling.py", "fig_regime.py", "fig_restrike_height.py", "fig_restrike_height_by_re.py",
           "fig_restrike_height_vs_re.py", "fig_stationarity.py", "fig_discharge_time_bound.py", "fig_velocity_maps.py", "fig_validation.py"]

if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    for s in SCRIPTS:
        if s == "build_frames_cache.py" and "--no-cache" in sys.argv:
            continue
        print(f"== {s}", flush=True)
        runpy.run_path(str(HERE / s), run_name="__main__")
