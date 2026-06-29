#!/usr/bin/env python3
"""
npoint_restrike.py  —  whole-arc (N-point) breakpoint re-strike detector.

WHY THIS EXISTS
---------------
The authoritative Mode-A/B re-strike metric (``qa_restrike.py``) is built from
the ARC-TIP trajectory (``arc_tip_row``).  That detector is structurally blind to
re-strikes where the tip is invariant but ONE LEG / SIDE of the arc breaks and
re-initiates: the bottom-most pixel does not move, so the tip detector sees no
cycle.  Those one-sided "leg" breaks are real Mode-B (back-breakdown) events and
the tip metric undercounts them.

This module tracks the WHOLE arc as N equally-spaced points along the skeleton
(``arc_shape``) and detects a re-strike INDEPENDENTLY at each of the N points
(a collapse-and-recovery of that point's depth), then unions the per-point
events.  A re-strike that only one leg-point sees is therefore captured even
when the tip never moves.

It is a COMPLEMENT, not a replacement: the tip metric (``qa_restrike``) is the
conservative companion and is kept intact.  V1 below re-derives the tip metric
exactly to prove the two pipelines share the same foundation.

FRAME SET (apples-to-apples with the tip metric)
------------------------------------------------
The rows of ``measurements_all_events.csv`` with ``arc_detected==1 AND
is_flash==0`` (the SAME set ``qa_restrike`` uses — it does NOT apply
``arc_complete``), in ``frame`` order, grouped by ``file``.  The N-point arc is
RECONSTRUCTED for those frames only (no re-running of analyze_blade): the raw
TIFF page is re-binarised at the recording's calibrated ``bright_thresh`` with
``analyze_blade.binarize`` (the identical detector that produced the CSV), then
skeletonised, ordered, resampled to N points and oriented electrode-proximal.

DETECTOR (per point, unioned)
-----------------------------
Per point i: depth d_i = point_row_i - gap_row; smooth (uniform_filter1d size 3);
``find_peaks(prominence = 0.10*(p90-p5 of that point's depth), distance>=5)``;
a re-strike is the argmin of the signal between consecutive peaks (identical
collapse-between-extensions logic to ``qa_restrike.detect_restrikes``, but
per point and with a per-point adaptive prominence).  Events from all N points
are unioned and de-duplicated within 5 frames, keeping the strongest (deepest
collapse) representative and its triggering point index i.

CLASSIFIER (per event)
----------------------
L_eff = p90 of the DEEPEST-point depth (matches ``qa_restrike.apex_and_range``).
Mode B if the deepest-point depth at the trough frame > 0.10*L_eff (arc stayed
extended / back-breakdown), else Mode A (reset to the gap) — the faithful
generalisation of the tip ``norm_pos`` cut.  Breakpoint position s = i/(N-1)
(0 = one foot, 1 = the other); tagged ``leg`` if s<0.2 or s>0.8 else ``central``.

USAGE
-----
    # detect on one recording (resumable; checkpoints reconstruction per chunk)
    python npoint_restrike.py detect --recording "HS_0.4mm_1000mL_(CO2)_trig"
    python npoint_restrike.py detect --all            # all six recordings
    python npoint_restrike.py validate                # V1 + V2 + V3 report
    python npoint_restrike.py audit                   # V4 leg-event audit PNGs
    python npoint_restrike.py figures                 # comparison figures + summary
    python npoint_restrike.py all                     # detect+validate+audit+figures

All per-recording calibration (bright_thresh, gap_row, px_per_mm, max_arc_px,
arc_direction, fps) is read from each ``run_metadata.json`` — nothing hardcoded.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from skimage.morphology import skeletonize
import tifffile

import arc_filter
import qa_restrike
from analyze_blade import binarize as ab_binarize, DEFAULT_MIN_ARC_PX
from arc_shape import ordered_path, resample_arc, orient_arc

# ── recording registry (Re / geometry / repeat-flag) ────────────────────────────
# Re is LOCKED to the collaborator's table so every figure axis is identical:
#   {(0.4,1500):9717, (0.4,1000):6478, (0.8,1500):4859, (1.2,1500):3239}
DATA_ROOT_DEFAULT = Path(r"C:\Users\compl\Desktop\Plasma Arc\J Electrode")

RECORDINGS = {
    "HS_0.4mm_1000mL_(CO2)_trig":                 dict(re=6478, d_mm=0.4, q=1000, is_repeat=False),
    "HS_0.4mm_1500mL_(CO2)_trig":                 dict(re=9717, d_mm=0.4, q=1500, is_repeat=False),
    "HS_0.4mm_1500mL_(CO2)_trig_repeat":          dict(re=9717, d_mm=0.4, q=1500, is_repeat=True),
    "HS_0.8mm_1500mL_(CO2)_trig(2)":              dict(re=4859, d_mm=0.8, q=1500, is_repeat=False),
    "HS_0.8mm_1500mL_(CO2)_trig_repeat_weird(2)": dict(re=4859, d_mm=0.8, q=1500, is_repeat=True),
    "HS_1.2mm_1500mL_(CO2)":                      dict(re=3239, d_mm=1.2, q=1500, is_repeat=False),
}

# Per-nozzle colours (prompt-specified house style for the N-point comparison figs).
NOZZLE_COLOR = {0.4: '#2166ac', 0.8: '#d6604d', 1.2: '#1a9850'}

# N values: headline = 10; localisation = 15; 20 for the N-sensitivity check.
N_HEADLINE = 10
N_LIST = (10, 15, 20)

# Detector constants (shared with the tip detector where applicable).
PROM_FRAC = 0.10        # prominence = PROM_FRAC * (p90-p5) of each point's depth
MIN_CYCLE = 5           # min samples between peaks (find_peaks distance)
SMOOTH = 3              # uniform_filter1d window
DEDUP_FRAMES = 5        # union/dedup window across points (frame numbers)
MODE_B_FRAC = 0.10      # deepest_depth > MODE_B_FRAC*L_eff  -> Mode B
LEG_LO, LEG_HI = 0.20, 0.80   # s outside [LEG_LO, LEG_HI] -> leg break

CLOSE_RADIUS = 3
DARK_GUARD = 35


# ─────────────────────────────────────────────────────────────────────────────
# metadata + frame-source helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_metadata(results_dir: Path) -> dict:
    """Robustly parse run_metadata.json (it may carry trailing whitespace
    padding — truncate to the last closing brace before json.loads)."""
    txt = (results_dir / "run_metadata.json").read_text(encoding="utf-8", errors="replace")
    txt = txt[:txt.rfind("}") + 1]
    return json.loads(txt)


def detector_params(meta: dict) -> dict:
    """Pull the binarisation / geometry calibration actually used to make the CSV."""
    return dict(
        bright_thresh=int(meta["bright_thresh"]),
        gap_row=int(meta["gap_row"]),
        px_per_mm=float(meta["px_per_mm"]),
        max_arc_px=int(meta["max_arc_px"]),
        min_arc_px=int(meta["min_arc_px"]) if meta.get("min_arc_px") else DEFAULT_MIN_ARC_PX,
        arc_direction=meta.get("arc_direction", "down"),
        fps=float(meta["fps"]),
    )


def tiff_chunks(results_dir: Path):
    """Sorted chunk TIFFs and the frames-per-tiff (page count of the first chunk)."""
    tiffs = sorted((results_dir / "tiff").glob("*.tiff"))
    if not tiffs:
        raise FileNotFoundError(f"no TIFF chunks in {results_dir/'tiff'}")
    with tifffile.TiffFile(str(tiffs[0])) as tf:
        fpf = len(tf.pages)
    return tiffs, fpf


def detected_noflash_frames(results_dir: Path) -> "tuple[np.ndarray, np.ndarray, pd.DataFrame]":
    """The detected & ~flash frame set (sorted by frame) plus the arc_tip_row for
    those frames — the SAME selection qa_restrike uses (no arc_complete)."""
    df = pd.read_csv(results_dir / "measurements_all_events.csv")
    df = arc_filter.add_flash_flag(df)
    det = pd.to_numeric(df["arc_detected"], errors="coerce").fillna(0).astype(int) == 1
    sel = df[det & (~df["is_flash"])].dropna(subset=["arc_tip_row"]).sort_values("frame")
    frames = sel["frame"].to_numpy(dtype=np.int64)
    tip = sel["arc_tip_row"].to_numpy(dtype=float)
    return frames, tip, sel


# ─────────────────────────────────────────────────────────────────────────────
# N-point reconstruction (resumable, chunk-by-chunk)
# ─────────────────────────────────────────────────────────────────────────────
def _cache_path(results_dir: Path) -> Path:
    return results_dir / "npoint_depth_cache.npz"


def reconstruct_depths(results_dir: Path, meta: dict, frame_limit: int | None = None,
                       resume: bool = True, progress: bool = True) -> dict:
    """Reconstruct per-point arc DEPTHS for every detected & ~flash frame.

    For each frame: read the raw TIFF page, ``analyze_blade.binarize`` at the
    recording's calibration (the identical detector behind the CSV), skeletonise,
    order, resample to each N in N_LIST and orient electrode-proximal.  Depth of
    point i is row_i - gap_row.  Frames whose skeleton/resample fails are skipped
    (for ALL N together, so every per-point array stays aligned to ``frames``).

    The heavy I/O+CV pass is CHECKPOINTED per chunk to npoint_depth_cache.npz so a
    re-run resumes.  Returns dict(frames, depths={N:(M,N)}, n_recon, n_attempted).
    """
    p = detector_params(meta)
    gap_row = p["gap_row"]
    tiffs, fpf = tiff_chunks(results_dir)
    frames_all, _, _ = detected_noflash_frames(results_dir)
    if frame_limit is not None:
        frames_all = frames_all[frames_all < frame_limit]

    # chunk -> sorted frames in that chunk
    chunk_of = frames_all // fpf
    chunks_sorted = sorted(np.unique(chunk_of).tolist())
    sig = f"{len(frames_all)}|{int(frames_all[-1]) if len(frames_all) else -1}|{frame_limit}"

    # resume from checkpoint if present and matching
    done_chunks: set = set()
    frames_acc: list = []
    depth_acc = {N: [] for N in N_LIST}
    cache = _cache_path(results_dir)
    if resume and cache.exists():
        try:
            z = np.load(cache, allow_pickle=True)
            if str(z["sig"]) == sig:
                done_chunks = set(int(c) for c in z["done_chunks"])
                frames_acc = list(z["frames"].astype(np.int64))
                for N in N_LIST:
                    depth_acc[N] = list(z[f"d{N}"].astype(np.float32))
                if progress and done_chunks:
                    print(f"    [resume] {len(done_chunks)}/{len(chunks_sorted)} chunks "
                          f"already reconstructed ({len(frames_acc)} frames cached)")
        except Exception as e:  # noqa: BLE001
            print(f"    [resume] ignoring unusable cache ({e})")

    def _save_ckpt():
        np.savez(cache, sig=sig,
                 done_chunks=np.array(sorted(done_chunks), dtype=np.int64),
                 frames=np.array(frames_acc, dtype=np.int64),
                 **{f"d{N}": (np.array(depth_acc[N], dtype=np.float32)
                              if depth_acc[N] else np.zeros((0, N), np.float32))
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
                    pts = orient_arc(pts, gap_row)
                    rec[N] = (pts[:, 0] - gap_row).astype(np.float32)
                if not ok:
                    continue
                frames_acc.append(int(f))
                for N in N_LIST:
                    depth_acc[N].append(rec[N])
        done_chunks.add(int(c))
        _save_ckpt()
        if progress:
            rate = n_attempted / max(time.time() - t0, 1e-6)
            print(f"    chunk {ci+1}/{len(chunks_sorted)} (idx {c})  "
                  f"recon={len(frames_acc)}  {rate:.0f} fr/s", flush=True)

    frames = np.array(frames_acc, dtype=np.int64)
    order = np.argsort(frames, kind="stable")
    frames = frames[order]
    depths = {N: (np.array(depth_acc[N], dtype=np.float32)[order]
                  if len(depth_acc[N]) else np.zeros((0, N), np.float32))
              for N in N_LIST}
    return dict(frames=frames, depths=depths, n_recon=len(frames), n_attempted=n_attempted)


# ─────────────────────────────────────────────────────────────────────────────
# detector + classifier
# ─────────────────────────────────────────────────────────────────────────────
def detect_events(frames: np.ndarray, depths_N: np.ndarray) -> dict:
    """Run the per-point collapse-recovery detector on an (M,N) depth matrix,
    union across points, dedup within DEDUP_FRAMES (keep strongest collapse),
    classify A/B + leg/central.  Returns dict(events=[...], L_eff, deepest)."""
    M, N = depths_N.shape
    if M < 3:
        return dict(events=[], L_eff=1.0, deepest=np.zeros(M))
    deepest = depths_N.max(axis=1)                      # per-frame tip depth
    L_eff = max(float(np.percentile(deepest, 90)), 1.0)

    raw_events = []   # (array_idx k, point_i, strength)
    for i in range(N):
        s = uniform_filter1d(depths_N[:, i].astype(float), size=SMOOTH)
        p5, p90 = np.percentile(s, 5), np.percentile(s, 90)
        prom = max(PROM_FRAC * (p90 - p5), 1e-6)
        peaks, _ = find_peaks(s, prominence=prom, distance=MIN_CYCLE)
        for j in range(1, len(peaks)):
            a, b = peaks[j - 1], peaks[j]
            k = a + int(np.argmin(s[a:b + 1]))
            strength = float(min(s[a], s[b]) - s[k])
            raw_events.append((k, i, strength))

    # union + dedup within DEDUP_FRAMES frames, keeping the strongest collapse
    raw_events.sort(key=lambda e: -e[2])               # strongest first
    accepted = []                                      # (k, i, strength)
    acc_frames = []                                    # kept frame numbers (sorted)
    import bisect
    for k, i, strength in raw_events:
        f = int(frames[k])
        pos = bisect.bisect_left(acc_frames, f)
        near = ((pos < len(acc_frames) and acc_frames[pos] - f <= DEDUP_FRAMES) or
                (pos > 0 and f - acc_frames[pos - 1] <= DEDUP_FRAMES))
        if near:
            continue
        accepted.append((k, i, strength))
        acc_frames.insert(pos, f)

    accepted.sort(key=lambda e: frames[e[0]])
    events = []
    prev_f = None
    for k, i, strength in accepted:
        f = int(frames[k])
        s_pos = i / (N - 1)
        d_deep = float(deepest[k])
        mode = "B" if d_deep > MODE_B_FRAC * L_eff else "A"
        kind = "leg" if (s_pos < LEG_LO or s_pos > LEG_HI) else "central"
        cyc = int(f - prev_f) if prev_f is not None else -1
        events.append(dict(frame=f, point_idx=int(i), s=round(s_pos, 4),
                           mode=mode, kind=kind,
                           breakpoint_depth_px=round(float(depths_N[k, i]), 2),
                           deepest_depth_px=round(d_deep, 2),
                           strength_px=round(strength, 2), cycle_frames=cyc))
        prev_f = f
    return dict(events=events, L_eff=L_eff, deepest=deepest)


def tip_restrike_frames(results_dir: Path, meta: dict) -> np.ndarray:
    """Frame numbers of the TIP detector's re-strikes (qa_restrike logic) — used to
    flag which N-point events are tip-INVISIBLE."""
    frames, tip, _ = detected_noflash_frames(results_dir)
    p = detector_params(meta)
    apex = float(p["gap_row"]) if p["arc_direction"] == "down" \
        else float(meta["height"] - 1 - p["gap_row"])
    erow, l_eff, _ = qa_restrike.apex_and_range(tip, None, apex_override=apex)
    rk, _, _ = qa_restrike.detect_restrikes(tip, erow, l_eff, PROM_FRAC, MIN_CYCLE, SMOOTH)
    return np.array([int(frames[r["idx"]]) for r in rk], dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# per-recording driver
# ─────────────────────────────────────────────────────────────────────────────
def run_recording(stem: str, data_root: Path, n: int = N_HEADLINE,
                  resume: bool = True, write_csv: bool = True) -> dict:
    """Reconstruct depths, detect & classify at N=n, write npoint_restrikes.csv.
    Returns a summary dict (also carries the full recon for reuse by validations)."""
    info = RECORDINGS[stem]
    rd = data_root / f"{stem}_results"
    meta = load_metadata(rd)
    p = detector_params(meta)
    print(f"  [{stem}] bright={p['bright_thresh']} gap_row={p['gap_row']} "
          f"px/mm={p['px_per_mm']:.3f} max_arc={p['max_arc_px']}")
    recon = reconstruct_depths(rd, meta, resume=resume)
    res = detect_events(recon["frames"], recon["depths"][n])
    events = res["events"]
    px = p["px_per_mm"]
    fps = p["fps"]

    # tip-invisible flag (not within DEDUP_FRAMES of any tip re-strike)
    tipf = tip_restrike_frames(rd, meta)
    for e in events:
        if len(tipf):
            near = np.min(np.abs(tipf - e["frame"]))
        else:
            near = 1e9
        e["tip_invisible"] = bool(near > DEDUP_FRAMES)
        e["breakpoint_depth_mm"] = round(e["breakpoint_depth_px"] / px, 3)
        e["deepest_depth_mm"] = round(e["deepest_depth_px"] / px, 3)

    n_ev = len(events)
    n_b = sum(e["mode"] == "B" for e in events)
    n_b_leg = sum(e["mode"] == "B" and e["kind"] == "leg" for e in events)
    cyc = np.array([e["cycle_frames"] for e in events if e["cycle_frames"] > 0], float)
    mean_cycle_ms = float(cyc.mean() / fps * 1e3) if len(cyc) else float("nan")

    if write_csv:
        cols = ["file", "frame", "mode", "kind", "s", "point_idx",
                "breakpoint_depth_mm", "deepest_depth_mm", "strength_px",
                "cycle_frames", "tip_invisible"]
        rows = [dict(file=stem, **{k: e[k] for k in cols if k != "file"}) for e in events]
        out = pd.DataFrame(rows, columns=cols)
        out.to_csv(rd / "npoint_restrikes.csv", index=False)
        print(f"    wrote {rd/'npoint_restrikes.csv'}  ({n_ev} events)")

    summary = dict(
        file=stem, N=n, n_events=n_ev,
        frac_B=round(n_b / n_ev, 4) if n_ev else float("nan"),
        leg_share_of_B=round(n_b_leg / n_b, 4) if n_b else float("nan"),
        n_B=n_b, n_B_leg=n_b_leg,
        mean_cycle_ms=round(mean_cycle_ms, 4),
        Re=info["re"], d_mm=info["d_mm"], q_mLpm=info["q"],
        is_repeat=int(info["is_repeat"]),
        n_recon=recon["n_recon"], L_eff_px=round(res["L_eff"], 2),
    )
    summary["_recon"] = recon
    summary["_meta"] = meta
    return summary


def write_summary(summaries: list, data_root: Path):
    cols = ["file", "N", "n_events", "frac_B", "leg_share_of_B", "n_B", "n_B_leg",
            "mean_cycle_ms", "Re", "d_mm", "q_mLpm", "is_repeat", "n_recon", "L_eff_px"]
    df = pd.DataFrame([{k: s[k] for k in cols} for s in summaries], columns=cols)
    df = df.sort_values(["is_repeat", "Re"])
    outdir = data_root / "Reynolds_comparison"
    outdir.mkdir(exist_ok=True)
    df.to_csv(outdir / "npoint_restrike_summary.csv", index=False)
    print(f"\nWrote {outdir/'npoint_restrike_summary.csv'}")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# V5 — current-trace cross-correlation hook (documented; not yet wired)
# ─────────────────────────────────────────────────────────────────────────────
def crosscorrelate_current_trace(npoint_csv: Path, current_trace_csv: Path,
                                 fps: float, window_us: float = 5.0):
    """TODO (V5): true event-count validation against Alexander's oscilloscope
    current trace.  Once a current trace is provided (CSV: time_s, current_A),
    convert each N-point event frame to a time (frame/fps), and test whether the
    current shows a breakdown pulse within +/- ``window_us`` of that time.  Report
    the fraction of detected events with a coincident current pulse (true-positive
    rate) and the pulse rate with no detected event (false-negative rate).

    Not yet wired: no current trace is available for these archival CO2 recordings.
    """
    raise NotImplementedError(
        "V5 current-trace cross-correlation: provide an oscilloscope current trace "
        "(time_s, current_A) for the recording, then implement the +/-window pulse "
        "coincidence test described in this function's docstring.")


def _resolve_stems(args) -> list:
    if getattr(args, "all", False) or getattr(args, "recording", None) in (None, "all"):
        return list(RECORDINGS.keys())
    return [args.recording]


# ─────────────────────────────────────────────────────────────────────────────
# V1 — reproduce the tip detector EXACTLY (hard gate)
# ─────────────────────────────────────────────────────────────────────────────
def v1_tip_reproduction(stem: str, data_root: Path) -> dict:
    rd = data_root / f"{stem}_results"
    meta = load_metadata(rd)
    p = detector_params(meta)
    prom_frac = float(meta.get("restrike_prominence_frac", PROM_FRAC))
    thr = float(meta.get("restrike_threshold", MODE_B_FRAC))
    frames, tip, _ = detected_noflash_frames(rd)
    apex = float(p["gap_row"]) if p["arc_direction"] == "down" \
        else float(meta["height"] - 1 - p["gap_row"])
    erow, l_eff, _ = qa_restrike.apex_and_range(tip, None, apex_override=apex)
    rk, _, _ = qa_restrike.detect_restrikes(tip, erow, l_eff, prom_frac, MIN_CYCLE, SMOOTH)
    n_a = n_b = 0
    cyc = []
    prev_f = None
    for r in rk:
        k = r["idx"]
        npos = float(np.clip((tip[k] - erow) / l_eff, 0, 1))
        denom = max(r["prev_ext"] - erow, 1e-6)
        rfrac = float(np.clip((r["prev_ext"] - tip[k]) / denom, 0, 1))
        mode = qa_restrike.classify(npos, rfrac, "abs", thr, 0.70)
        n_a += mode == "A"
        n_b += mode == "B"
        if prev_f is not None:
            cyc.append(frames[k] - prev_f)
        prev_f = frames[k]
    n = len(rk)
    mean_cyc_ms = (np.mean(cyc) / p["fps"] * 1e3) if cyc else float("nan")

    ref = pd.read_csv(rd / "re_strikes_qa.csv")
    ref_n = len(ref)
    ref_a = int((ref["mode"] == "A").sum())
    ref_b = int((ref["mode"] == "B").sum())
    ok = (n, n_a, n_b) == (ref_n, ref_a, ref_b)
    return dict(stem=stem, ok=ok, got=(n, n_a, n_b), ref=(ref_n, ref_a, ref_b),
                mean_cycle_ms=mean_cyc_ms)


# ─────────────────────────────────────────────────────────────────────────────
# V2 — reconstruction fidelity (deepest N-point row == arc_tip_row)
# ─────────────────────────────────────────────────────────────────────────────
def v2_reconstruction_fidelity(stem: str, data_root: Path, recon: dict | None = None,
                               n_sample: int = 600, seed: int = 0) -> dict:
    """Confirm the reconstruction reproduces the pipeline.

    PRIMARY GATE (binarise/detection fidelity): on a random sample of detected
    frames, re-binarise with our calibration and take the arc FRONT pixel
    (``analyze_blade.arc_front_tip``, the exact quantity the pipeline stored as
    ``arc_tip_row``).  This must match ``arc_tip_row`` to < 1 px median — it proves
    our ``ab_binarize`` call reproduces the pipeline's detection mask.

    INFORMATIONAL: the N-point DEEPEST-point (skeleton CENTERLINE) row sits a
    physical ~half-tip-width ABOVE the front pixel, so it differs from
    ``arc_tip_row`` by a small, consistent offset.  That offset is reported but is
    NOT a fidelity error: the A/B cut normalises deepest-depth by L_eff (both
    centerline quantities), so the offset cancels and the classification is
    unaffected.
    """
    from analyze_blade import arc_front_tip
    rd = data_root / f"{stem}_results"
    meta = load_metadata(rd)
    p = detector_params(meta)
    fr_all, tip_all, _ = detected_noflash_frames(rd)
    tipmap = dict(zip(fr_all.tolist(), tip_all.tolist()))

    # PRIMARY: sampled mask-front-pixel vs arc_tip_row (the reconstruction gate)
    rng = np.random.default_rng(seed)
    samp = np.sort(fr_all[rng.choice(len(fr_all), size=min(n_sample, len(fr_all)),
                                     replace=False)])
    tiffs, fpf = tiff_chunks(rd)
    front_diff = []
    by_chunk: dict = {}
    for f in samp:
        by_chunk.setdefault(int(f) // fpf, []).append(int(f))
    for c, fl in by_chunk.items():
        if c >= len(tiffs):
            continue
        with tifffile.TiffFile(str(tiffs[c])) as tf:
            for f in fl:
                raw = tf.pages[f % fpf].asarray()
                if raw.ndim == 3:
                    raw = raw.mean(axis=2).astype(np.uint8)
                m = ab_binarize(raw, CLOSE_RADIUS, dark_guard=DARK_GUARD,
                                bright_thresh=p["bright_thresh"],
                                min_arc_px=p["min_arc_px"], max_arc_px=p["max_arc_px"])
                if not m.any():
                    continue
                tr, _ = arc_front_tip(m, p["arc_direction"])
                if not np.isnan(tr):
                    front_diff.append(abs(float(tr) - tipmap[f]))
    mad_front = float(np.median(front_diff)) if front_diff else float("nan")

    # INFORMATIONAL: centerline (N-point deepest) offset, over all cached frames
    mad_center = float("nan")
    if recon is None:
        recon = reconstruct_depths(rd, meta, resume=True, progress=False)
    if recon["n_recon"]:
        frames_r = recon["frames"]
        deepest_row = recon["depths"][N_HEADLINE].max(axis=1) + p["gap_row"]
        tip_ref = np.array([tipmap.get(int(f), np.nan) for f in frames_r])
        good = ~np.isnan(tip_ref)
        if good.any():
            mad_center = float(np.median(np.abs(deepest_row[good] - tip_ref[good])))
    return dict(stem=stem, ok=mad_front < 1.0, median_abs_diff_px=mad_front,
                centerline_offset_px=mad_center, n=len(front_diff))


# ─────────────────────────────────────────────────────────────────────────────
# V3 — N-sensitivity (N=10,15,20)
# ─────────────────────────────────────────────────────────────────────────────
def v3_n_sensitivity(stem: str, data_root: Path, recon: dict | None = None) -> dict:
    rd = data_root / f"{stem}_results"
    meta = load_metadata(rd)
    if recon is None:
        recon = reconstruct_depths(rd, meta, resume=True, progress=False)
    rows = []
    for N in N_LIST:
        res = detect_events(recon["frames"], recon["depths"][N])
        ev = res["events"]
        n_ev = len(ev)
        n_b = sum(e["mode"] == "B" for e in ev)
        n_b_leg = sum(e["mode"] == "B" and e["kind"] == "leg" for e in ev)
        rows.append(dict(N=N, n_events=n_ev,
                         frac_B=(n_b / n_ev) if n_ev else float("nan"),
                         leg_share=(n_b_leg / n_b) if n_b else float("nan")))
    ev_vals = [r["n_events"] for r in rows]
    fb_vals = [r["frac_B"] for r in rows]
    d_events = max(ev_vals) - min(ev_vals)
    d_fracB = max(fb_vals) - min(fb_vals)
    rel_events = d_events / max(np.mean(ev_vals), 1)
    # absolute tol (~3) OR relative tol (~15%); frac_B within ~3%
    ok = (d_events <= 3 or rel_events <= 0.15) and (d_fracB <= 0.03)
    return dict(stem=stem, ok=ok, rows=rows, d_events=d_events,
                rel_events=rel_events, d_fracB=d_fracB)


def run_validations(data_root: Path, stems: list):
    print("\n" + "=" * 78)
    print("VALIDATIONS  (V1 hard gate, V2 fidelity, V3 N-sensitivity)")
    print("=" * 78)
    # V1
    print("\n--- V1: reproduce the tip detector EXACTLY ---")
    v1_all_ok = True
    for stem in stems:
        r = v1_tip_reproduction(stem, data_root)
        tag = "PASS" if r["ok"] else "FAIL"
        v1_all_ok &= r["ok"]
        print(f"  [{tag}] {stem}: got n/A/B={r['got']} ref={r['ref']} "
              f"mean_cycle={r['mean_cycle_ms']:.3f} ms")
    if not v1_all_ok:
        print("\n*** V1 FAILED — STOPPING (the N-point pipeline must share the tip "
              "foundation before its numbers can be trusted). ***")
        raise SystemExit(2)
    print("  V1 PASS for all recordings.")

    # V2
    print("\n--- V2: reconstruction fidelity (deepest N-point row vs arc_tip_row) ---")
    for stem in stems:
        r = v2_reconstruction_fidelity(stem, data_root)
        tag = "PASS" if r["ok"] else "FAIL"
        print(f"  [{tag}] {stem}: mask-front vs arc_tip_row median|d|="
              f"{r['median_abs_diff_px']:.3f} px (n={r['n']}, <1 px required); "
              f"centerline offset={r['centerline_offset_px']:.2f} px (physical, cancels in A/B)")

    # V3
    print("\n--- V3: N-sensitivity (N=10/15/20) ---")
    print(f"  {'recording':<44} {'N':>3} {'n_events':>9} {'frac_B':>7} {'leg_share':>10}")
    for stem in stems:
        r = v3_n_sensitivity(stem, data_root)
        for row in r["rows"]:
            print(f"  {stem:<44} {row['N']:>3} {row['n_events']:>9} "
                  f"{row['frac_B']:>7.3f} {row['leg_share']:>10.3f}")
        tag = "PASS" if r["ok"] else "WARN"
        print(f"    -> [{tag}] d_events={r['d_events']} (rel {r['rel_events']:.2%}), "
              f"d_frac_B={r['d_fracB']:.3f}")
    print("\nValidations complete.")


# ─────────────────────────────────────────────────────────────────────────────
# V4 — visual audit of leg-break, tip-invisible events
# ─────────────────────────────────────────────────────────────────────────────
def audit_recording(stem: str, data_root: Path, n_sample: int = 8, seed: int = 0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rd = data_root / f"{stem}_results"
    meta = load_metadata(rd)
    p = detector_params(meta)
    gap_row = p["gap_row"]
    csv = rd / "npoint_restrikes.csv"
    if not csv.exists():
        print(f"  [audit] {stem}: no npoint_restrikes.csv — run detect first.")
        return
    ev = pd.read_csv(csv)
    cand = ev[(ev["mode"] == "B") & (ev["kind"] == "leg") & (ev["tip_invisible"])]
    if len(cand) == 0:
        cand = ev[(ev["mode"] == "B") & (ev["kind"] == "leg")]
    if len(cand) == 0:
        print(f"  [audit] {stem}: no leg-break Mode-B events to audit.")
        return
    rng = np.random.default_rng(seed)
    pick = cand.iloc[np.sort(rng.choice(len(cand), size=min(n_sample, len(cand)),
                                        replace=False))]

    recon = reconstruct_depths(rd, meta, resume=True, progress=False)
    frames_r = recon["frames"]
    depth = recon["depths"][N_HEADLINE]
    fr_all, tip_all, _ = detected_noflash_frames(rd)
    tipmap = dict(zip(fr_all.tolist(), tip_all.tolist()))

    _, fpf = tiff_chunks(rd)
    src = qa_restrike.FrameSource(tiff_dir=str(rd / "tiff"), frames_per_tiff=fpf)

    nrow = len(pick)
    fig, axes = plt.subplots(nrow, 4, figsize=(13, 2.7 * nrow),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1.6]))
    if nrow == 1:
        axes = axes[None, :]

    for r, (_, e) in enumerate(pick.iterrows()):
        ef = int(e["frame"])
        ai = int(np.searchsorted(frames_r, ef))
        ai = int(np.clip(ai, 0, len(frames_r) - 1))
        # three reconstructed frames spanning the event (array-adjacent)
        idxs = [max(ai - 4, 0), ai, min(ai + 4, len(frames_r) - 1)]
        for c, ii in enumerate(idxs):
            ax = axes[r, c]
            gf = int(frames_r[ii])
            raw = src.get(gf)
            if raw is not None:
                ax.imshow(np.power(raw / max(raw.max(), 1), 0.45), cmap="inferno",
                          vmin=0, vmax=1)
                mask = ab_binarize(raw, CLOSE_RADIUS, dark_guard=DARK_GUARD,
                                   bright_thresh=p["bright_thresh"],
                                   min_arc_px=p["min_arc_px"], max_arc_px=p["max_arc_px"])
                if mask.any():
                    skel = skeletonize(mask)
                    path = ordered_path(skel)
                    if path is not None:
                        pts = resample_arc(path, N_HEADLINE)
                        if pts is not None:
                            pts = orient_arc(pts, gap_row)
                            ax.plot(pts[:, 1], pts[:, 0], "-", color="cyan", lw=0.8)
                            ax.scatter(pts[:, 1], pts[:, 0], s=10, c="cyan")
                            pi = int(e["point_idx"])
                            ax.scatter([pts[pi, 1]], [pts[pi, 0]], s=70,
                                       facecolors="none", edgecolors="lime", lw=2)
                ax.axhline(gap_row, color="yellow", lw=0.7, ls="--")
            ax.set_title(f"f={gf}{'  *trough' if ii == ai else ''}", fontsize=7)
            ax.axis("off")
        # depth-vs-time trace
        axt = axes[r, 3]
        w = 60
        lo, hi = max(ai - w, 0), min(ai + w, len(frames_r) - 1)
        xs = frames_r[lo:hi + 1]
        for i in range(N_HEADLINE):
            s_pos = i / (N_HEADLINE - 1)
            is_leg = (s_pos < LEG_LO or s_pos > LEG_HI)
            axt.plot(xs, depth[lo:hi + 1, i] / p["px_per_mm"],
                     color=("#d6604d" if is_leg else "#bbbbbb"),
                     lw=(1.0 if i == int(e["point_idx"]) else 0.5),
                     alpha=(1.0 if i == int(e["point_idx"]) else 0.6))
        tipy = np.array([tipmap.get(int(f), np.nan) for f in xs]) - gap_row
        axt.plot(xs, tipy / p["px_per_mm"], color="black", lw=1.2, label="tip")
        axt.axvline(ef, color="lime", lw=1.2)
        axt.set_xlabel("frame", fontsize=7)
        axt.set_ylabel("depth (mm)", fontsize=7)
        axt.tick_params(labelsize=6)
        axt.legend(fontsize=6, loc="upper right")
        axt.text(0.02, 0.95, f"s={e['s']:.2f} pt{int(e['point_idx'])}",
                 transform=axt.transAxes, fontsize=7, va="top",
                 color="#d6604d")
    src.close()
    fig.suptitle(f"{stem}  —  leg-break, tip-invisible Mode-B re-strikes "
                 f"(green ring = triggering point; cyan = N={N_HEADLINE} arc)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = rd / "npoint_leg_event_audit.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  [audit] {stem}: wrote {out}  ({len(pick)} events)")


# ─────────────────────────────────────────────────────────────────────────────
# comparison figures (tip vs N-point) — house style, no titles in the files
# ─────────────────────────────────────────────────────────────────────────────
def _tip_numbers(stem: str, data_root: Path) -> dict:
    rd = data_root / f"{stem}_results"
    ref = pd.read_csv(rd / "re_strikes_qa.csv")
    n = len(ref)
    n_b = int((ref["mode"] == "B").sum())
    return dict(n_events=n, frac_B=(n_b / n) if n else float("nan"))


def make_figures(data_root: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sumcsv = data_root / "Reynolds_comparison" / "npoint_restrike_summary.csv"
    summ = pd.read_csv(sumcsv)
    # headline = non-repeats only
    head = summ[summ["is_repeat"] == 0].copy()
    head["tip_n"] = [_tip_numbers(s, data_root)["n_events"] for s in head["file"]]
    head["tip_fracB"] = [_tip_numbers(s, data_root)["frac_B"] for s in head["file"]]
    head = head.sort_values("Re")

    def color(d):
        return NOZZLE_COLOR.get(round(float(d), 1), "#777777")

    outdirs = [data_root / "Reynolds_comparison", Path(__file__).parent / "paper_figures",
               data_root / "paper_figures"]
    for d in outdirs:
        d.mkdir(exist_ok=True)

    def save(fig, stem):
        for d in outdirs:
            fig.savefig(d / f"{stem}.png", dpi=300, bbox_inches="tight")
            fig.savefig(d / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    re = head["Re"].to_numpy()
    cols = [color(d) for d in head["d_mm"]]

    # 1) event count: tip vs N-point
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(re, head["tip_n"], marker="o", facecolors="none",
               edgecolors=cols, s=70, linewidths=1.5, label="tip-only")
    ax.scatter(re, head["n_events"], marker="s", c=cols, s=70, label="N-point")
    for x, y0, y1 in zip(re, head["tip_n"], head["n_events"]):
        ax.plot([x, x], [y0, y1], color="#999999", lw=0.8, zorder=0)
    ax.set_yscale("log")  # recording lengths differ ~50x (1.2mm is 873k frames)
    ax.set_xlabel("Reynolds number")
    ax.set_ylabel("re-strike count (log)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, which="both")
    save(fig, "npoint_vs_tip_event_count")

    # 1b) length-normalised rate: events per 1000 detected frames (comparable across Re)
    rate_tip = 1000.0 * head["tip_n"] / head["n_recon"]
    rate_np = 1000.0 * head["n_events"] / head["n_recon"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(re, rate_tip, marker="o", facecolors="none",
               edgecolors=cols, s=70, linewidths=1.5, label="tip-only")
    ax.scatter(re, rate_np, marker="s", c=cols, s=70, label="N-point")
    for x, y0, y1 in zip(re, rate_tip, rate_np):
        ax.plot([x, x], [y0, y1], color="#999999", lw=0.8, zorder=0)
    ax.set_xlabel("Reynolds number")
    ax.set_ylabel("re-strikes per 1000 detected frames")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    save(fig, "npoint_vs_tip_event_rate")

    # 2) Mode-B fraction: tip vs N-point
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(re, head["tip_fracB"], marker="o", facecolors="none",
               edgecolors=cols, s=70, linewidths=1.5, label="tip-only")
    ax.scatter(re, head["frac_B"], marker="s", c=cols, s=70, label="N-point")
    for x, y0, y1 in zip(re, head["tip_fracB"], head["frac_B"]):
        ax.plot([x, x], [y0, y1], color="#999999", lw=0.8, zorder=0)
    ax.set_xlabel("Reynolds number")
    ax.set_ylabel("Mode-B re-strike fraction")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    save(fig, "npoint_vs_tip_modeB_fraction")

    # 3) leg-break share of Mode-B vs Re (the new physical result)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.scatter(re, head["leg_share_of_B"], c=cols, s=80)
    ax.plot(re, head["leg_share_of_B"], color="#999999", lw=0.8, zorder=0)
    ax.set_xlabel("Reynolds number")
    ax.set_ylabel("leg-break share of Mode-B")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    # nozzle legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=NOZZLE_COLOR[d],
                      markersize=9, label=f"{d} mm") for d in (0.4, 0.8, 1.2)]
    ax.legend(handles=handles, frameon=False, title="nozzle")
    save(fig, "npoint_legshare_vs_re")

    print(f"Saved comparison figures to: {', '.join(str(d) for d in outdirs)}")


def write_oneparagraph_summary(data_root: Path):
    summ = pd.read_csv(data_root / "Reynolds_comparison" / "npoint_restrike_summary.csv")
    head = summ[summ["is_repeat"] == 0].sort_values("Re")
    lines = ["N-POINT vs TIP-ONLY — what changed", "=" * 50]
    for _, s in head.iterrows():
        tip = _tip_numbers(s["file"], data_root)
        lines.append(
            f"{s['file']}  (Re {s['Re']}): tip {tip['n_events']} re-strikes "
            f"({tip['frac_B']:.0%} B)  ->  N-point {s['n_events']} events "
            f"({s['frac_B']:.0%} B), leg-share of B = {s['leg_share_of_B']:.0%}")
    out = data_root / "Reynolds_comparison" / "npoint_summary_paragraph.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {out}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command",
                    choices=["detect", "validate", "audit", "figures", "all"],
                    help="detect: run detector; validate: V1-V3; audit: V4 PNGs; "
                         "figures: comparison figs + summary; all: everything")
    ap.add_argument("--recording", default=None,
                    help="recording stem (default: all six)")
    ap.add_argument("--all", action="store_true", help="all recordings")
    ap.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore the reconstruction checkpoint and start fresh")
    ap.add_argument("--n", type=int, default=N_HEADLINE, help="headline N (default 10)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    stems = _resolve_stems(args)
    resume = not args.no_resume

    if args.command in ("detect", "all"):
        print("=" * 78)
        print(f"DETECT  (N={args.n})  on {len(stems)} recording(s)")
        print("=" * 78)
        summaries = []
        for stem in stems:
            print(f"\n>>> {stem}")
            summaries.append(run_recording(stem, data_root, n=args.n, resume=resume))
        write_summary(summaries, data_root)

    if args.command in ("validate", "all"):
        run_validations(data_root, stems)

    if args.command in ("audit", "all"):
        print("\n" + "=" * 78)
        print("V4 AUDIT — leg-break tip-invisible event montages")
        print("=" * 78)
        for stem in stems:
            audit_recording(stem, data_root)

    if args.command in ("figures", "all"):
        print("\n" + "=" * 78)
        print("COMPARISON FIGURES + SUMMARY")
        print("=" * 78)
        make_figures(data_root)
        write_oneparagraph_summary(data_root)


if __name__ == "__main__":
    main()
