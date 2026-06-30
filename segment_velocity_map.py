#!/usr/bin/env python3
"""
segment_velocity_map.py  —  2-D velocity map painted with the PER-SEGMENT local
arc velocity, not the single tip velocity.

``build_velocity_map.py`` paints ONE scalar (the tip ``local_vel_ms``) onto every
pixel of an arc in a frame, so the whole arc is a single colour.  This script
instead paints EACH part of the arc with ITS OWN local velocity: every arc pixel
is assigned to the nearest of the N reconstructed arc points (cKDTree) and takes
that segment's local velocity (from ``segment_velocity.py``).  The result resolves
how the propagation speed varies ALONG the arc body and in space, instead of a
flat per-frame colour.

INPUTS (all already produced)
-----------------------------
* ``segment_velocities.csv``   (segment_velocity.py): per frame, per segment vel.
* ``npoint_xy_cache.npz``      (npoint_geometry.py):  per frame N-point (row,col).
* the recording's TIFF chunks: re-binarised with the SAME detector params as the
  analyze stage (bright_thresh / max_arc_px / min_arc_px from run_metadata.json),
  exactly like build_velocity_map, to recover each frame's arc pixel mask.

OUTPUTS (per recording, velocity-map house style: plasma, mm extent, origin upper)
---------------------------------------------------------------------------------
``velocity_map_segment.npy`` (mm/s) + ``.png``
``velocity_map_segment_vs_tip.png``  — segment map vs the existing tip-uniform
                                       ``velocity_map_local.npy`` on a shared scale.

By default the AXIAL component ``seg_vy_ms`` is painted (matches the axial
``local_vel_ms`` of the tip map); ``--metric speed`` paints ``seg_speed_ms``.

USAGE
-----
    python segment_velocity_map.py --recording "HS_0.4mm_1000mL_(CO2)_trig"
    python segment_velocity_map.py --all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize  # noqa: F401  (parity w/ reconstruction)
import tifffile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from npoint_restrike import (RECORDINGS, N_HEADLINE, CLOSE_RADIUS, DARK_GUARD,
                             load_metadata, detector_params, tiff_chunks)
from npoint_geometry import points_for
from analyze_blade import binarize as ab_binarize

DATA_ROOT_DEFAULT = Path(r"C:\Users\compl\Desktop\Plasma Arc\J Electrode")


def build_map(stem: str, data_root: Path, n: int = N_HEADLINE,
              metric: str = "seg_vy_ms") -> dict:
    rd = data_root / f"{stem}_results"
    meta = load_metadata(rd)
    p = detector_params(meta)
    pxmm = p["px_per_mm"]

    sv = pd.read_csv(rd / "segment_velocities.csv")
    sv = sv[["frame", "seg_idx", metric]].copy()
    # per frame -> array of N segment velocities (index by seg_idx)
    vel_by_frame = {}
    for f, g in sv.groupby("frame"):
        arr = np.full(n, np.nan)
        arr[g["seg_idx"].to_numpy()] = g[metric].to_numpy()
        vel_by_frame[int(f)] = arr

    frames_xy, pts_xy = points_for(rd, N=n)         # (M,), (M,n,2)
    pt_index = {int(f): i for i, f in enumerate(frames_xy)}

    tiffs, fpf = tiff_chunks(rd)
    H = W = None
    vel_sum = vel_cnt = None
    painted = 0
    target_frames = sorted(set(vel_by_frame) & set(pt_index))
    by_chunk = {}
    for f in target_frames:
        by_chunk.setdefault(f // fpf, []).append(f)

    for c, fl in sorted(by_chunk.items()):
        if c >= len(tiffs):
            continue
        with tifffile.TiffFile(str(tiffs[c])) as tf:
            npages = len(tf.pages)
            for f in fl:
                page = f % fpf
                if page >= npages:
                    continue
                raw = tf.pages[page].asarray()
                if raw.ndim == 3:
                    raw = raw.mean(axis=2).astype(np.uint8)
                mask = ab_binarize(raw, CLOSE_RADIUS, dark_guard=DARK_GUARD,
                                   bright_thresh=p["bright_thresh"],
                                   min_arc_px=p["min_arc_px"], max_arc_px=p["max_arc_px"])
                if mask is None or not mask.any():
                    continue
                if vel_sum is None:
                    H, W = mask.shape
                    vel_sum = np.zeros((H, W)); vel_cnt = np.zeros((H, W))
                pix = np.argwhere(mask)                       # (P,2) row,col
                pts = pts_xy[pt_index[f]]                      # (n,2) row,col
                segvel = vel_by_frame[f]                       # (n,)
                ok = np.isfinite(pts[:, 0]) & np.isfinite(segvel)
                if ok.sum() < 2:
                    continue
                tree = cKDTree(pts[ok])
                _, nn = tree.query(pix)                        # nearest kept point
                vsel = segvel[ok][nn]
                good = np.isfinite(vsel)
                rr = pix[good, 0]; cc = pix[good, 1]
                vel_sum[rr, cc] += vsel[good]
                vel_cnt[rr, cc] += 1
                painted += 1

    if vel_sum is None:
        return dict(stem=stem, painted=0)
    vmap = np.full((H, W), np.nan)
    m = vel_cnt > 0
    vmap[m] = vel_sum[m] / vel_cnt[m]
    np.save(rd / "velocity_map_segment.npy", vmap * 1000.0)    # mm/s, pipeline convention
    return dict(stem=stem, painted=painted, vmap=vmap, px_per_mm=pxmm,
                n_pixels=int(m.sum()), metric=metric)


def make_figures(res: dict, data_root: Path):
    stem = res["stem"]
    rd = data_root / f"{stem}_results"
    vmap = res["vmap"]
    pxmm = res["px_per_mm"]
    H, W = vmap.shape
    ext = [0, W / pxmm, H / pxmm, 0]
    finite = vmap[np.isfinite(vmap)]
    vmax = float(np.percentile(finite, 97)) if finite.size else 1.0
    vmax = max(vmax, 0.01)

    # 1) segment-resolved map (house style)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(vmap, cmap="plasma", vmin=0, vmax=vmax, origin="upper",
                   extent=ext, aspect="equal")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"per-segment local {('axial' if res['metric']=='seg_vy_ms' else 'speed')} "
                 f"velocity (m/s)")
    fig.tight_layout()
    fig.savefig(rd / "velocity_map_segment.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) side-by-side vs the tip-uniform map (shared scale)
    tip_path = rd / "velocity_map_local.npy"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    tip = None
    if tip_path.exists():
        tip = np.load(tip_path).astype(float)
        if np.isfinite(tip).any() and np.nanmax(tip) > 100:   # stored mm/s
            tip = tip / 1000.0
    pooled = [finite]
    if tip is not None and np.isfinite(tip).any():
        pooled.append(tip[np.isfinite(tip)])
    shared_vmax = max(float(np.percentile(np.concatenate(pooled), 97)), 0.01)
    for ax, dat, title in [(axes[0], tip, "tip (uniform per frame)"),
                           (axes[1], vmap, "per-segment (this work)")]:
        if dat is None:
            ax.text(0.5, 0.5, "tip map not found", ha="center", va="center")
            ax.axis("off"); continue
        im = ax.imshow(dat, cmap="plasma", vmin=0, vmax=shared_vmax, origin="upper",
                       extent=ext, aspect="equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("local velocity (m/s)")
    fig.tight_layout()
    fig.savefig(rd / "velocity_map_segment_vs_tip.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    ap.add_argument("--n", type=int, default=N_HEADLINE)
    ap.add_argument("--metric", choices=["seg_vy_ms", "seg_speed_ms"], default="seg_vy_ms")
    args = ap.parse_args()
    data_root = Path(args.data_root)
    stems = list(RECORDINGS.keys()) if (args.all or not args.recording) else [args.recording]

    for stem in stems:
        print(f">>> {stem}")
        res = build_map(stem, data_root, n=args.n, metric=args.metric)
        if res["painted"] == 0:
            print("    no frames painted")
            continue
        make_figures(res, data_root)
        print(f"    painted {res['painted']:,} frames, {res['n_pixels']:,} pixels  "
              f"-> velocity_map_segment.npy/.png + _vs_tip.png")


if __name__ == "__main__":
    main()
