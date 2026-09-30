"""Which observable separates re-strikes from gliding? Per-frame (dt = 5 us) changes of
  L   : skeleton length (arc_length_px, measurements.csv)
  zc  : centroid height = mean height above the gap of the 10 equally spaced arc points
        (the arc's centre of mass per unit length)
  zt  : apex height (arc_tip_row; 'up' rig: already H-1-row)
on frame pairs where BOTH frames are complete arcs, plus the 10-segment detector's dip sizes (strength_px).
Gliding at ~0.6 m/s is ~0.03 px/frame at 200 kfps and ~9 px/mm, so a breakdown should appear as a
single-frame jump far outside the gliding/jitter core.

usage: python restrike_jumps.py OUT.png <run_dir> [<run_dir> ...]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

out = Path(sys.argv[1])
runs = [Path(p) for p in sys.argv[2:]]
fig, axs = plt.subplots(len(runs), 4, figsize=(13, 2.8 * len(runs)), squeeze=False)
for row, rd in enumerate(runs):
    z = np.load(rd / "npoint_xy.npz")
    H = int(z["H"])
    f = z["frames"]
    zc = ((H - 1) - z["pts"][:, :, 0].astype(float)).mean(1)
    m = pd.read_csv(rd / "measurements.csv", usecols=["frame", "arc_length_px", "arc_tip_row"]).set_index("frame")
    L = m["arc_length_px"].reindex(f).to_numpy(float)
    zt = m["arc_tip_row"].reindex(f).to_numpy(float)
    pair = np.diff(f) == 1
    d = {"arc length L": np.diff(L)[pair], "centroid height": np.diff(zc)[pair], "apex height": np.diff(zt)[pair]}
    tip = pd.read_csv(rd / "re_strikes_qa.csv")
    npr = pd.read_csv(rd / "npoint_restrikes.csv")
    print(f"\n{rd.name}: {len(f)} complete frames, {int(pair.sum())} consecutive pairs; "
          f"tip re-strikes {len(tip)}, 10-segment events {len(npr)}")
    for ax, (k, x) in zip(axs[row], d.items()):
        q = np.percentile(x, [0.1, 1, 25, 50, 75, 99, 99.9])
        rsd = 1.4826 * np.median(np.abs(x - np.median(x)))
        print(f"  d({k:15s}) px: pctl 0.1/1/25/50/75/99/99.9 = {np.round(q, 1)}  robust sd {rsd:.2f}  "
              f"drops < -5: {(x < -5).sum():5d}  < -10: {(x < -10).sum():5d}  < -20: {(x < -20).sum():4d}")
        ax.hist(x, bins=np.arange(-80.5, 81, 1), log=True, color="0.3")
        ax.set_xlabel(f"per-frame change, {k} (px)")
    axs[row, 3].hist(npr["strength_px"], bins=np.arange(0, 41, 0.5), log=True, color="0.3")
    axs[row, 3].set_xlabel("10-segment dip size (px)")
    axs[row, 0].set_ylabel(rd.name.replace("BLADE_ADCB_", "").replace("_06mmNozzle", ""), fontsize=8)
fig.tight_layout()
fig.savefig(out, dpi=110)
print("\nwrote", out)
