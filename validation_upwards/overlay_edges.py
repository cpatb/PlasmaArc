"""Overlay each run's hand-annotated blade edges (the averaged, extended lines the analysis used) on that video's
temporal-std image (2000 frames spread over the record), to check the geometry against the video itself.

usage: python overlay_edges.py OUT.png <run_dir> [<run_dir> ...]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cine_frames as cf  # noqa: E402

N_FRAMES = 2000
out = Path(sys.argv[1])
runs = [Path(p) for p in sys.argv[2:]]
fig, axs = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.2), squeeze=False)
for ax, rd in zip(axs[0], runs):
    m = json.loads((rd / "run_metadata.json").read_text())
    src = cf.open_cine(m["cine"])
    idx = np.linspace(0, src.n - 1, N_FRAMES).astype(int)
    std = np.stack([src.frame(int(i)) for i in idx]).astype(np.float32).std(0)
    ax.imshow(np.clip(std / np.percentile(std, 99.5), 0, 1), cmap="gray")
    z = np.load(next((rd / "calibration").glob("*_edges_annotated.npz")))
    for k in ("A", "B"):
        ax.plot(z[k][:, 1], z[k][:, 0], color="cyan", lw=0.8)
    a = m.get("calibration_auto", {}).get("gap_ends")
    if a:
        ax.plot([p[1] for p in a], [p[0] for p in a], "r+", ms=10)
    ax.set_xlim(60, 170)
    ax.set_ylim(250, 150)
    ax.set_title(rd.name.replace("BLADE_ADCB_", "").replace("_06mmNozzle", ""), fontsize=8)
fig.tight_layout()
fig.savefig(out, dpi=110)
print("wrote", out)
