"""Count of centroid-drop re-strike candidates vs the drop threshold, for every finished annotated run.

Centroid height = mean height above the gap of the 10 equally spaced arc points (npoint_xy.npz), in mm with the
run's own px/mm. A candidate = the drop of that height between SUCCESSIVE complete arcs exceeding the threshold.
One panel per gas, one line per recording (colour = Re). Writes the figure, the curve table (CSV) and the
per-drop table (every drop > 0.1 mm, so the threshold can be moved later without recomputing).

usage: python centroid_threshold_curve.py <runs_root> <out_prefix>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

root, prefix = Path(sys.argv[1]), sys.argv[2]
THR_MM = np.round(np.arange(0.1, 5.01, 0.05), 3)
curves, drops = [], []
for npz in sorted(root.rglob("npoint_xy.npz")):
    rd = npz.parent
    if not (rd / "run_metadata.json").exists():          # the per-window copies under chunks/
        continue
    m = json.loads((rd / "run_metadata.json").read_text())
    if m.get("status") != "DONE":
        continue
    z = np.load(npz)
    H = int(z["H"])
    f = z["frames"]
    pxmm = float(m["calibration"]["px_per_mm"])
    zc = ((H - 1) - z["pts"][:, :, 0].astype(float)).mean(1) / pxmm         # height, mm (above frame bottom)
    dz = np.diff(zc)
    gap = np.diff(f)
    for k in np.flatnonzero(dz < -0.1):
        drops.append(dict(run=rd.name, gas=m["gas"], Re=m["Re"], trial=m["trial"], frame_before=int(f[k]),
                          frame_after=int(f[k + 1]), gap_frames=int(gap[k]), drop_mm=float(-dz[k])))
    dur_s = m["n_frames"] / m["fps"]
    for t in THR_MM:
        curves.append(dict(run=rd.name, gas=m["gas"], Re=m["Re"], trial=m["trial"], thr_mm=t,
                           n=int((dz < -t).sum()), per_s=float((dz < -t).sum() / dur_s)))
C, D = pd.DataFrame(curves), pd.DataFrame(drops)
C.to_csv(prefix + "_curves.csv", index=False)
D.to_csv(prefix + "_drops.csv", index=False)
print(f"runs: {C['run'].nunique()}  (gas: {C.groupby('gas')['run'].nunique().to_dict()})")

gases = sorted(C["gas"].unique())
fig, axs = plt.subplots(1, len(gases), figsize=(3.2 * len(gases), 3.0), sharey=True, squeeze=False)
res = sorted(C["Re"].unique())
cmap = plt.get_cmap("viridis")
for ax, g in zip(axs[0], gases):
    for run, grp in C[C["gas"] == g].groupby("run"):
        re = grp["Re"].iloc[0]
        ax.plot(grp["thr_mm"], grp["n"].clip(lower=0.8), color=cmap(res.index(re) / max(len(res) - 1, 1)), lw=1)
    ax.set_yscale("log")
    ax.set_xlim(0, 5)
    ax.set_title(g, fontsize=9)
    ax.set_xlabel("drop threshold (mm)")
axs[0, 0].set_ylabel("centroid drops per recording")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min(res), max(res)))
fig.colorbar(sm, ax=axs[0].tolist(), label="Re", fraction=0.02, pad=0.01)
fig.savefig(prefix + ".png", dpi=130, bbox_inches="tight")
print("wrote", prefix + ".png")
