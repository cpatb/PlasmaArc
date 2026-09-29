"""Visual validation (V-T3): random frames of a cine with the binarize mask contour overlaid.

usage: mask_montage.py <cine> <T> <out.png> [--n 24] [--seed 1] [--select detected|all|flood|wide]
       [--frames a,b,...]  [--gamma 0.5]
Frames are drawn with a fixed display gamma on the 0..4095 scale (not per-frame stretched) so glow
vs filament brightness is comparable across panels. Title: frame, frame max, mask area, width.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from skimage.morphology import skeletonize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cine_frames as cf  # noqa: E402
from analyze_blade import binarize  # noqa: E402

CLOSE_R, MIN_PX = 3, 22


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cine")
    ap.add_argument("T", type=float)
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--select", default="detected", choices=["detected", "all", "flood", "wide"])
    ap.add_argument("--max-arc-px", type=float, default=1e9)
    ap.add_argument("--wide-px", type=float, default=10.0)
    ap.add_argument("--frames", default=None)
    ap.add_argument("--low", type=float, default=None, help="hysteresis low threshold (T is then the high)")
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--background", default=None)
    ap.add_argument("--low-frac", type=float, default=None)
    ap.add_argument("--pool", type=int, default=4000, help="random candidates examined")
    ap.add_argument("--crop", default="120,256,30,220", help="r0,r1,c0,c1")
    a = ap.parse_args()
    src = cf.open_cine(a.cine, background=a.background)
    rng = np.random.default_rng(a.seed)
    if a.frames:
        picks = [int(x) for x in a.frames.split(",")]
    else:
        cand = np.sort(rng.choice(src.n, size=min(a.pool, src.n), replace=False))
        picks = []
        for i in cand:
            f = src.frame(int(i))
            b = binarize(f, CLOSE_R, dark_guard=a.T, bright_thresh=a.T, min_arc_px=MIN_PX, max_arc_px=1e9, low_thresh=a.low, low_frac=a.low_frac)
            area = int(b.sum())
            if a.select == "detected" and not (0 < area <= a.max_arc_px):
                continue
            if a.select == "flood" and not area > a.max_arc_px:
                continue
            if a.select == "wide":
                if not area:
                    continue
                if area / max(skeletonize(b).sum(), 1) <= a.wide_px:
                    continue
            picks.append(int(i))
            if len(picks) >= a.n:
                break
    r0, r1, c0, c1 = map(int, a.crop.split(","))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = 6
    nrow = max(1, int(np.ceil(len(picks) / ncol)))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3 * nrow), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, i in zip(axs.ravel(), picks):
        f = src.frame(i).astype(float)
        b = binarize(f, CLOSE_R, dark_guard=a.T, bright_thresh=a.T, min_arc_px=MIN_PX, max_arc_px=1e9, low_thresh=a.low, low_frac=a.low_frac)
        ax.imshow((np.clip(f, 0, 4095) / 4095) ** a.gamma, cmap="gray", vmin=0, vmax=1)
        if b.any():
            ax.contour(b.astype(float), levels=[0.5], colors="r", linewidths=0.6)
        sk = skeletonize(b).sum() if b.any() else 0
        ax.set_xlim(c0, c1)
        ax.set_ylim(r1, r0)
        ax.set_title(f"{i} max {int(f.max())} A {int(b.sum())} w {b.sum() / max(sk, 1):.1f}", fontsize=7)
    fig.suptitle(f"{Path(a.cine).stem}  T={a.T:.0f} low={a.low} frac={a.low_frac}  select={a.select}", fontsize=9)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=80)
    print("wrote", a.out, "frames", picks)


if __name__ == "__main__":
    main()
