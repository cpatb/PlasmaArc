#!/usr/bin/env python3
"""
npoint_geometry.py  —  whole-arc N-point reconstruction that keeps the FULL
(row, col) coordinate of every point, not just its depth.

WHY THIS EXISTS
---------------
``npoint_restrike.reconstruct_depths`` reconstructs the N-point arc for every
detected & ~flash frame but checkpoints only the DEPTH of each point
(``row - gap_row``) to ``npoint_depth_cache.npz``.  The depth (axial row) is all
the tip/leg re-strike detector needs.  But three downstream analyses need the
arc's LATERAL (column) coordinate too:

  * per-segment LATERAL velocity over time (``segment_velocity.py``),
  * the segment-resolved 2-D velocity map (``segment_velocity_map.py``),
  * N-point initiation-location heatmaps and N=10 geometry stats.

The column is not recoverable from the depth cache or from any CSV, so this
module re-runs the SAME reconstruction (identical detector / skeleton / resample
as ``reconstruct_depths``) and checkpoints the full ``(row, col)`` per point to a
SEPARATE cache ``npoint_xy_cache.npz`` — the existing depth cache is left intact
so ``npoint_restrike`` keeps working unchanged.

Everything heavy (TIFF I/O + binarise + skeletonise + resample) is shared with
``npoint_restrike`` by import; the ONLY difference is that we append
``orient_arc(pts, gap_row)`` whole (both columns) instead of ``pts[:,0]-gap_row``.
Depths remain derivable as ``xy[N][:, :, 0] - gap_row`` whenever needed.

USAGE
-----
    python npoint_geometry.py build --recording "HS_0.4mm_1000mL_(CO2)_trig"
    python npoint_geometry.py build --all            # all six recordings (resumable)
    python npoint_geometry.py validate               # deepest-row/col vs arc_tip_row/col
    python npoint_geometry.py validate --all

All per-recording calibration is read from each ``run_metadata.json`` via
``npoint_restrike.detector_params`` — nothing hardcoded.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
import tifffile

# Reuse the EXACT machinery behind the depth cache so the two reconstructions are
# byte-for-byte the same detector, just storing more per point.
import npoint_restrike as npr
from npoint_restrike import (
    RECORDINGS, N_LIST, CLOSE_RADIUS, DARK_GUARD, N_HEADLINE,
    DATA_ROOT_DEFAULT, load_metadata, detector_params, tiff_chunks,
    detected_noflash_frames,
)
from analyze_blade import binarize as ab_binarize
from arc_shape import ordered_path, resample_arc, orient_arc


def _cache_path(results_dir: Path) -> Path:
    return results_dir / "npoint_xy_cache.npz"


def reconstruct_points(results_dir: Path, meta: dict, frame_limit: int | None = None,
                       resume: bool = True, progress: bool = True) -> dict:
    """Reconstruct per-point arc (row, col) for every detected & ~flash frame.

    Mirrors ``npoint_restrike.reconstruct_depths`` exactly (same frame set, same
    binarise/skeletonise/order/resample/orient), but stores the full oriented
    ``(row, col)`` per point.  Checkpointed per chunk to ``npoint_xy_cache.npz``.

    Returns dict(frames, pts={N:(M,N,2)}, n_recon, n_attempted).
    """
    p = detector_params(meta)
    gap_row = p["gap_row"]
    tiffs, fpf = tiff_chunks(results_dir)
    frames_all, _, _ = detected_noflash_frames(results_dir)
    if frame_limit is not None:
        frames_all = frames_all[frames_all < frame_limit]

    chunk_of = frames_all // fpf
    chunks_sorted = sorted(np.unique(chunk_of).tolist())
    sig = f"{len(frames_all)}|{int(frames_all[-1]) if len(frames_all) else -1}|{frame_limit}"

    done_chunks: set = set()
    frames_acc: list = []
    xy_acc = {N: [] for N in N_LIST}
    cache = _cache_path(results_dir)
    if resume and cache.exists():
        try:
            z = np.load(cache, allow_pickle=True)
            if str(z["sig"]) == sig:
                done_chunks = set(int(c) for c in z["done_chunks"])
                frames_acc = list(z["frames"].astype(np.int64))
                for N in N_LIST:
                    arr = z[f"xy{N}"].astype(np.float32)
                    xy_acc[N] = [arr[i] for i in range(arr.shape[0])]
                if progress and done_chunks:
                    print(f"    [resume] {len(done_chunks)}/{len(chunks_sorted)} chunks "
                          f"already reconstructed ({len(frames_acc)} frames cached)")
        except Exception as e:  # noqa: BLE001
            print(f"    [resume] ignoring unusable cache ({e})")

    def _save_ckpt():
        np.savez(cache, sig=sig,
                 done_chunks=np.array(sorted(done_chunks), dtype=np.int64),
                 frames=np.array(frames_acc, dtype=np.int64),
                 **{f"xy{N}": (np.array(xy_acc[N], dtype=np.float32)
                              if xy_acc[N] else np.zeros((0, N, 2), np.float32))
                    for N in N_LIST})

    t0 = time.time()
    n_attempted = 0
    for ci, c in enumerate(chunks_sorted):
        if c in done_chunks:
            continue
        fr_chunk = frames_all[chunk_of == c]
        tpath = tiffs[c] if c < len(tiffs) else None
        if tpath is None:
            continue
        with tifffile.TiffFile(str(tpath)) as tf:
            npages = len(tf.pages)
            for f in fr_chunk:
                n_attempted += 1
                page = int(f) % fpf
                if page >= npages:
                    continue
                raw = tf.pages[page].asarray()
                if raw.ndim == 3:
                    raw = raw.mean(axis=2).astype(np.uint8)
                mask = ab_binarize(raw, CLOSE_RADIUS, dark_guard=DARK_GUARD,
                                   bright_thresh=p["bright_thresh"],
                                   min_arc_px=p["min_arc_px"], max_arc_px=p["max_arc_px"])
                if not mask.any():
                    continue
                skel = skeletonize(mask)
                if not skel.any():
                    continue
                path = ordered_path(skel)
                if path is None:
                    continue
                rec = {}
                ok = True
                for N in N_LIST:
                    pts = resample_arc(path, N)
                    if pts is None:
                        ok = False
                        break
                    rec[N] = orient_arc(pts, gap_row).astype(np.float32)   # (N,2) row,col
                if not ok:
                    continue
                frames_acc.append(int(f))
                for N in N_LIST:
                    xy_acc[N].append(rec[N])
        done_chunks.add(int(c))
        _save_ckpt()
        if progress:
            rate = n_attempted / max(time.time() - t0, 1e-6)
            print(f"    chunk {ci+1}/{len(chunks_sorted)} (idx {c})  "
                  f"recon={len(frames_acc)}  {rate:.0f} fr/s", flush=True)

    frames = np.array(frames_acc, dtype=np.int64)
    order = np.argsort(frames, kind="stable")
    frames = frames[order]
    pts = {N: (np.array(xy_acc[N], dtype=np.float32)[order]
               if len(xy_acc[N]) else np.zeros((0, N, 2), np.float32))
           for N in N_LIST}
    return dict(frames=frames, pts=pts, n_recon=len(frames), n_attempted=n_attempted)


def points_for(results_dir: Path, N: int = N_HEADLINE, meta: dict | None = None,
               resume: bool = True, progress: bool = False):
    """Load-or-build the xy cache and return (frames, pts(M,N,2)) for one N.

    The deepest-point DEPTH (axial) for that N is ``pts[:, :, 0] - gap_row``.
    """
    results_dir = Path(results_dir)
    if meta is None:
        meta = load_metadata(results_dir)
    recon = reconstruct_points(results_dir, meta, resume=resume, progress=progress)
    return recon["frames"], recon["pts"][N]


# ─────────────────────────────────────────────────────────────────────────────
# validation — deepest-point row vs arc_tip_row, deepest col vs arc_tip_col
# ─────────────────────────────────────────────────────────────────────────────
def validate_recording(stem: str, data_root: Path, n_sample: int = 800, seed: int = 0) -> dict:
    """The deepest N-point (max row) should track the measured tip.

    ROW: the N-point CENTERLINE deepest row sits ~half-a-tip-width above the
    measured arc-front ``arc_tip_row`` (a small, consistent physical offset; see
    npoint_restrike V2).  We therefore report the MEDIAN offset and its spread
    (the offset should be small and tight) rather than requiring < 1 px.

    COL: the deepest-point column should match the measured ``arc_tip_col``
    closely (both are lateral positions of the same end of the arc).  Report the
    median |Δcol|.
    """
    rd = data_root / f"{stem}_results"
    meta = load_metadata(rd)
    p = detector_params(meta)
    frames, tip, sel = detected_noflash_frames(rd)
    tiprow = dict(zip(frames.tolist(), tip.tolist()))
    tipcol = {}
    if "arc_tip_col" in sel.columns:
        tipcol = dict(zip(frames.tolist(),
                          pd.to_numeric(sel["arc_tip_col"], errors="coerce").tolist()))

    recon = reconstruct_points(rd, meta, resume=True, progress=False)
    fr = recon["frames"]
    pts = recon["pts"][N_HEADLINE]                       # (M,N,2)
    if len(fr) == 0:
        return dict(stem=stem, ok=False, n=0, note="no reconstructed frames")

    deep_idx = np.argmax(pts[:, :, 0], axis=1)           # deepest point per frame
    deep_row = pts[np.arange(len(fr)), deep_idx, 0]
    deep_col = pts[np.arange(len(fr)), deep_idx, 1]

    tref = np.array([tiprow.get(int(f), np.nan) for f in fr])
    good = np.isfinite(tref)
    row_off = float(np.median((deep_row[good] - tref[good]))) if good.any() else float("nan")
    row_mad = float(np.median(np.abs((deep_row[good] - tref[good]) - row_off))) if good.any() else float("nan")

    col_mad = float("nan")
    if tipcol:
        cref = np.array([tipcol.get(int(f), np.nan) for f in fr])
        cg = np.isfinite(cref) & good
        if cg.any():
            col_mad = float(np.median(np.abs(deep_col[cg] - cref[cg])))

    ok = (abs(row_off) < 6.0) and (row_mad < 3.0) and (np.isnan(col_mad) or col_mad < 6.0)
    return dict(stem=stem, ok=ok, n=int(good.sum()),
                row_offset_px=round(row_off, 2), row_mad_px=round(row_mad, 2),
                col_mad_px=round(col_mad, 2), n_recon=recon["n_recon"])


def _resolve_stems(args) -> list:
    if getattr(args, "all", False) or getattr(args, "recording", None) in (None, "all"):
        return list(RECORDINGS.keys())
    return [args.recording]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "validate"])
    ap.add_argument("--recording", default=None, help="recording stem (default: all)")
    ap.add_argument("--all", action="store_true", help="all recordings")
    ap.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore the xy checkpoint and start fresh")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    stems = _resolve_stems(args)
    resume = not args.no_resume

    if args.command == "build":
        print("=" * 78)
        print(f"BUILD npoint_xy_cache  on {len(stems)} recording(s)")
        print("=" * 78)
        for stem in stems:
            rd = data_root / f"{stem}_results"
            meta = load_metadata(rd)
            p = detector_params(meta)
            print(f"\n>>> {stem}  bright={p['bright_thresh']} gap_row={p['gap_row']} "
                  f"px/mm={p['px_per_mm']:.3f}")
            recon = reconstruct_points(rd, meta, resume=resume)
            print(f"    done: {recon['n_recon']} frames reconstructed -> "
                  f"{_cache_path(rd).name}")

    elif args.command == "validate":
        print("=" * 78)
        print("VALIDATE xy reconstruction (deepest-row offset, deepest-col |d|)")
        print("=" * 78)
        for stem in stems:
            r = validate_recording(stem, data_root)
            tag = "PASS" if r.get("ok") else "WARN"
            print(f"  [{tag}] {stem}: row_offset={r.get('row_offset_px')} px "
                  f"(mad {r.get('row_mad_px')}), col |d|={r.get('col_mad_px')} px "
                  f"(n={r.get('n')})")


if __name__ == "__main__":
    main()
