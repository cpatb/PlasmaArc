#!/usr/bin/env python3
"""
nseg_pipeline.py  —  run the whole-arc (N-segment) mode-dynamics pipeline at an
ARBITRARY N (exploration of N=20 and N=50 vs the headline N=10), writing N-tagged
outputs so the existing per-N folders/figures stay intact.

For each recording and a given N it produces, in the recording's _results dir:
  npoint_restrikes_N{n}.csv               detector (npoint_restrike.detect_events on
                                          the N-point depths) re-classified by the
                                          DEAD-ZONE VALLEY of the deepest-point depth
  restrike_initiation_coords_npoint_N{n}.csv   breakpoint (row,col,mode) for the maps
  segment_velocities_N{n}.csv             per-segment local velocity (segment_velocity)
  velocity_map_segment_N{n}.npy           per-segment 2-D velocity map (heavy; --maps)
and cross-recording, under <data-root>/analysis/:
  mode_transitions/mode_transition_summary_npoint_N{n}.csv
  geometry_predictor/geometry_predictor_stats_npoint_N{n}.csv (+ _multivar_)

N=10/15/20 read the committed npoint_xy_cache.npz; other N (e.g. 50) read
npoint_xy_cache_N{n}.npz (build it first with
  `python npoint_geometry.py build --all --n {n} --cache-suffix _N{n}`).

USAGE
  python nseg_pipeline.py --n 20            # data only (fast for N<=20: cache exists)
  python nseg_pipeline.py --n 50 --maps     # data + heavy segment-map paint pass
  python nseg_pipeline.py --n 20 --maps-only
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import arc_resets
from valley_reclassify import find_valley
from npoint_restrike import detect_events, RECORDINGS, load_metadata, detector_params
from npoint_geometry import points_for
import segment_velocity
import segment_velocity_map
from mode_transition_analysis import transition_counts, chain_stats, run_lengths, lag1_autocorr
import restrike_geometry_predictor as RGP

DATA_ROOT_DEFAULT = Path(r"C:\Users\compl\Desktop\Plasma Arc\J Electrode")
MAIN_CACHE_N = (10, 15, 20)          # already in npoint_xy_cache.npz
GROUP = "CO2 (J-electrode)"


def cache_for(rd: Path, N: int):
    return None if N in MAIN_CACHE_N else rd / f"npoint_xy_cache_N{N}.npz"


# ───────────────────────── per-recording: detector + valley + coords ────────────
def build_restrikes(stem: str, data_root: Path, N: int) -> pd.DataFrame:
    rd = data_root / f"{stem}_results"
    meta = load_metadata(rd)
    p = detector_params(meta)
    pxmm, gap_row = p["px_per_mm"], p["gap_row"]
    frames, pts = points_for(rd, N=N, cache_path=cache_for(rd, N))   # (M,), (M,N,2)
    depths = pts[:, :, 0] - gap_row
    res = detect_events(frames, depths)
    ev = res["events"]
    if not ev:
        return pd.DataFrame()
    fr_sorted = frames
    rows = []
    deep_mm = []
    for e in ev:
        f = int(e["frame"]); pi = int(e["point_idx"])
        idx = int(np.searchsorted(fr_sorted, f))
        idx = min(max(idx, 0), len(fr_sorted) - 1)
        pt = pts[idx, pi]
        ddm = e["deepest_depth_px"] / pxmm
        deep_mm.append(ddm)
        rows.append(dict(file=stem, frame=f, kind=e["kind"], s=e["s"],
                         point_idx=pi, is_endpoint=e["is_endpoint"],
                         deepest_depth_mm=round(ddm, 3),
                         breakpoint_depth_mm=round(e["breakpoint_depth_px"] / pxmm, 3),
                         cycle_frames=e["cycle_frames"],
                         bp_row=float(pt[0]), bp_col=float(pt[1])))
    df = pd.DataFrame(rows)
    valley = find_valley(np.asarray(deep_mm, float))
    valley = float(valley["valley"]) if valley else float("nan")
    df["mode"] = np.where(df["deepest_depth_mm"].to_numpy() < valley, "A", "B")
    df["valley_cut_mm"] = round(valley, 3)
    return df


def run_recording_data(stem: str, data_root: Path, N: int) -> dict:
    rd = data_root / f"{stem}_results"
    df = build_restrikes(stem, data_root, N)
    if df.empty:
        return dict(stem=stem, n=0)
    # 1) npoint_restrikes_N{n}.csv
    cols = ["file", "frame", "mode", "kind", "s", "point_idx", "is_endpoint",
            "deepest_depth_mm", "breakpoint_depth_mm", "cycle_frames", "valley_cut_mm"]
    df[cols].to_csv(rd / f"npoint_restrikes_N{N}.csv", index=False)
    # 2) initiation coords (breakpoint row/col + mode)
    df[["bp_row", "bp_col", "mode"]].rename(columns={"bp_row": "row", "bp_col": "col"}) \
        .to_csv(rd / f"restrike_initiation_coords_npoint_N{N}.csv", index=False)
    # 3) segment velocities at N
    sv, info = segment_velocity.compute(stem, data_root, n=N, cache_path=cache_for(rd, N))
    if not sv.empty:
        sv.to_csv(rd / f"segment_velocities_N{N}.csv", index=False)
    fb = float((df["mode"] == "B").mean())
    return dict(stem=stem, n=len(df), frac_B=round(fb, 4),
                valley_mm=df["valley_cut_mm"].iloc[0],
                n_seg_rows=int(len(sv)))


def run_recording_maps(stem: str, data_root: Path, N: int):
    rd = data_root / f"{stem}_results"
    res = segment_velocity_map.build_map(
        stem, data_root, n=N, metric="seg_speed_ms", cache_path=cache_for(rd, N),
        sv_csv=f"segment_velocities_N{N}.csv", out_npy=f"velocity_map_segment_N{N}.npy")
    return res.get("painted", 0)


# ───────────────────────── cross-recording summaries ────────────────────────────
def transition_summary(stems, data_root, N):
    rows, Cs, modes_all = [], [], []
    for stem in stems:
        rd = data_root / f"{stem}_results"
        f = rd / f"npoint_restrikes_N{N}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f).sort_values("frame")
        d = d[d["mode"].isin(["A", "B"])]
        modes = d["mode"].tolist()
        frames = pd.to_numeric(d["frame"], errors="coerce").to_numpy()
        # reset-aware contiguity (same as mode_transition_analysis)
        resets = None
        mp = rd / "measurements_all_events.csv"
        if mp.exists():
            meas = pd.read_csv(mp, usecols=["frame", "arc_detected", "is_flash"])
            resets = arc_resets.reset_intervals(meas, reset_gap=arc_resets.DEFAULT_RESET_GAP)
        cont = [False] * len(d)
        for k in range(1, len(d)):
            c = bool(frames[k] > frames[k - 1])
            if c and resets is not None and arc_resets.crosses_reset(
                    int(frames[k - 1]), int(frames[k]), resets):
                c = False
            cont[k] = c
        C = transition_counts(modes, cont)
        s = chain_stats(C, modes)
        obs, exp, _ = run_lengths(modes, cont, "B")
        info = RECORDINGS.get(stem, {})
        s.update(label=stem.replace("HS_", ""), group=GROUP, Re=info.get("re"),
                 meanrun_B_obs=obs, meanrun_B_iid=exp, lag1_autocorr=lag1_autocorr(modes, cont))
        rows.append(s)
        Cs.append(C); modes_all.extend(modes)
    if Cs:
        Csum = np.sum(Cs, axis=0)
        sp = chain_stats(Csum, modes_all)
        sp.update(label=f"POOLED {GROUP}", group=GROUP, Re=np.nan,
                  meanrun_B_obs=np.nan, meanrun_B_iid=np.nan, lag1_autocorr=np.nan)
        rows.append(sp)
    # transition summaries live under the PARENT analysis dir (same place
    # mode_transition_analysis writes and the figures read: dirname(ROOT)/analysis)
    out = data_root.parent / "analysis" / "mode_transitions"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"mode_transition_summary_npoint_N{N}.csv", index=False)
    return df


def geometry_summary(stems, data_root, N):
    per, pooled_frames, mv_rows = [], [], []
    for stem in stems:
        rd = data_root / f"{stem}_results"
        if not (rd / f"npoint_restrikes_N{N}.csv").exists():
            continue
        feats = RGP.build_features(stem, data_root, "npoint",
                                   rs_fname=f"npoint_restrikes_N{N}.csv")
        if feats.empty:
            continue
        pooled_frames.append(feats)
        mv = RGP.multivariate(feats)
        info = RECORDINGS.get(stem, {})
        mv_rows.append(dict(file=stem, Re=info.get("re"), n=mv["n"],
                            cv_auc=mv.get("cv_auc"), in_sample_auc=mv.get("in_sample_auc")))
    out = data_root / "analysis" / "geometry_predictor"
    out.mkdir(parents=True, exist_ok=True)
    if pooled_frames:
        pooled = pd.concat(pooled_frames, ignore_index=True)
        pstats = RGP.per_feature_stats(pooled)
        pstats.insert(0, "file", "POOLED")
        pstats.to_csv(out / f"geometry_predictor_stats_npoint_N{N}.csv", index=False)
        mvp = RGP.multivariate(pooled)
        mv_rows.append(dict(file="POOLED", Re=np.nan, n=mvp["n"],
                            cv_auc=mvp.get("cv_auc"), in_sample_auc=mvp.get("in_sample_auc")))
    pd.DataFrame(mv_rows).to_csv(out / f"geometry_predictor_multivar_npoint_N{N}.csv", index=False)
    return pd.DataFrame(mv_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    ap.add_argument("--maps", action="store_true", help="also build the (heavy) segment maps")
    ap.add_argument("--maps-only", action="store_true")
    args = ap.parse_args()
    data_root = Path(args.data_root)
    N = args.n
    stems = list(RECORDINGS.keys())

    if not args.maps_only:
        print(f"=== N={N}: per-recording data ===")
        for stem in stems:
            r = run_recording_data(stem, data_root, N)
            print(f"  {stem:44s} n={r.get('n',0):5d} frac_B={r.get('frac_B','-')} "
                  f"valley={r.get('valley_mm','-')}mm seg_rows={r.get('n_seg_rows',0)}")
        print(f"=== N={N}: cross-recording summaries ===")
        ts = transition_summary(stems, data_root, N)
        gm = geometry_summary(stems, data_root, N)
        pj = ts[ts["label"].astype(str).str.startswith("POOLED")]
        if len(pj):
            r = pj.iloc[0]
            print(f"  transition POOLED: p_B={r['p_B']:.2f} P(B|B)={r['P_B_given_B']:.2f} "
                  f"P(A|B)={r['P_A_given_B']:.2f} G-p={r['g_p']:.1e}")
        pg = gm[gm["file"] == "POOLED"]
        if len(pg):
            print(f"  geometry POOLED CV-AUC={float(pg['cv_auc'].iloc[0]):.3f}")

    if args.maps or args.maps_only:
        print(f"=== N={N}: segment velocity maps (heavy) ===")
        for stem in stems:
            painted = run_recording_maps(stem, data_root, N)
            print(f"  {stem:44s} painted {painted:,} frames -> velocity_map_segment_N{N}.npy")


if __name__ == "__main__":
    main()
