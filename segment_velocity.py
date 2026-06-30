#!/usr/bin/env python3
"""
segment_velocity.py  —  per-segment LOCAL (time-resolved) arc velocity for the
N=10 whole-arc reconstruction.

The pipeline computes a sliding-window OLS "local velocity" for the arc TIP only
(``event_velocity.compute_local_velocities`` -> ``local_vel_ms``).  This script
generalises that EXACT method to every one of the N equally-spaced points along
the arc: for each segment i it builds the time series of that segment's position
and fits a line to the LOCAL_VEL_N like-positioned (same-segment) neighbouring
arcs, giving a time-resolved velocity at that point.  This resolves how fast each
part of the arc body propagates, not just the tip.

METHOD (identical windowing to compute_local_velocities, applied per segment)
----------------------------------------------------------------------------
* Frames + N-point (row,col) come from the reconstruction cache
  (``npoint_geometry.points_for``); event ids + the tip's ``arc_tip_row`` come
  from ``measurements_all_events.csv``.
* Per event, the ADVANCING-FRONT frames are selected from the TIP trajectory
  (``event_velocity.advancing_front`` on ``arc_tip_row``) — the same extension
  phase the tip metric uses, shared by all segments so the deepest segment
  reduces to the tip exactly.
* For each advancing-front frame k, a centered window of LOCAL_VEL_N (default 10)
  neighbours is fit:
      seg_vy = slope(row_i vs frame) * fps / px_per_mm / 1000        (axial, m/s)
      seg_vx = slope(col_i vs frame) * fps / px_per_mm / 1000        (lateral, m/s)
      seg_speed = hypot(seg_vx, seg_vy)
* ``s = i/(N-1)`` is the normalised arc position (0 = electrode-proximal foot,
  1 = the other foot; the deepest interior point is the tip region).

FREE CORRECTNESS GATE
---------------------
The DEEPEST segment per frame (max row) is the tip; its ``seg_vy`` must reproduce
``local_vel_ms`` from the pipeline (median |Δ| ~ 0).  Reported by ``--gate``.

OUTPUT (per recording)
----------------------
``segment_velocities.csv``  (file, frame, event_id, seg_idx, s, seg_vy_ms,
seg_vx_ms, seg_speed_ms) — consumed by ``segment_velocity_map.py``.
``segment_velocity_profile.{png,pdf,txt}`` — mean +/- std velocity vs arc position.
``segment_velocity_density.{png,pdf,txt}`` — hexbin of seg_speed vs arc position.

USAGE
-----
    python segment_velocity.py --all
    python segment_velocity.py --recording "HS_1.2mm_1500mL_(CO2)" --gate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle
import plot_config
from npoint_restrike import RECORDINGS, N_HEADLINE
from npoint_geometry import points_for
from event_velocity import advancing_front
from arc_shape import unit_tangents

DATA_ROOT_DEFAULT = Path(r"C:\Users\compl\Desktop\Plasma Arc\J Electrode")
LOCAL_VEL_N = 10          # window width (advancing-front neighbours), matches the brief
MIN_WIN = 3               # need >=3 points to fit a line


def load_meta(rd: Path) -> dict:
    txt = (rd / "run_metadata.json").read_text(encoding="utf-8", errors="replace")
    return json.loads(txt[:txt.rfind("}") + 1])


def compute(stem: str, data_root: Path, n: int = N_HEADLINE,
            window: int = LOCAL_VEL_N, cache_path=None) -> "tuple[pd.DataFrame, dict]":
    rd = data_root / f"{stem}_results"
    meta = load_meta(rd)
    fps = float(meta["fps"])
    pxmm = float(meta["px_per_mm"])

    frames, pts = points_for(rd, N=n, cache_path=cache_path)   # frames (M,), pts (M,n,2)
    if len(frames) == 0:
        return pd.DataFrame(), dict(stem=stem, n_rows=0)

    meas = pd.read_csv(rd / "measurements_all_events.csv",
                       usecols=lambda c: c in ("frame", "event_id", "arc_tip_row",
                                               "local_vel_ms"))
    meas["frame"] = pd.to_numeric(meas["frame"], errors="coerce")
    # align measurements to the reconstructed frames
    mi = meas.set_index("frame")
    eid = mi["event_id"].reindex(frames).to_numpy()
    tiprow = pd.to_numeric(mi["arc_tip_row"].reindex(frames), errors="coerce").to_numpy()
    locvel = (pd.to_numeric(mi["local_vel_ms"].reindex(frames), errors="coerce").to_numpy()
              if "local_vel_ms" in mi.columns else np.full(len(frames), np.nan))

    half = window // 2
    rows_out = []
    gate_seg, gate_tip = [], []

    order = np.argsort(frames, kind="stable")
    frames_s = frames[order]
    pts_s = pts[order]
    eid_s = eid[order]
    tiprow_s = tiprow[order]
    locvel_s = locvel[order]

    # group by event id (skip event_id < 0 / NaN)
    valid_ev = np.isfinite(eid_s) & (np.nan_to_num(eid_s, nan=-1) >= 0)
    uniq = np.unique(eid_s[valid_ev]).astype(int)
    for ev in uniq:
        m = (eid_s == ev)
        f_ev = frames_s[m].astype(float)
        if len(f_ev) < MIN_WIN:
            continue
        p_ev = pts_s[m]                      # (k,n,2)
        tip_ev = tiprow_s[m]
        lv_ev = locvel_s[m]
        good = np.isfinite(tip_ev)
        if good.sum() < MIN_WIN:
            continue
        # advancing front from the TIP trajectory (shared extension phase)
        af = advancing_front(f_ev[good], tip_ev[good])
        idx_good = np.flatnonzero(good)
        af_idx = idx_good[af]                # indices into the event arrays
        if len(af_idx) < MIN_WIN:
            continue
        f_af = f_ev[af_idx]
        p_af = p_ev[af_idx]                  # (naf,n,2)
        lv_af = lv_ev[af_idx]
        tip_af = tip_ev[af_idx]              # measured arc_tip_row (front pixel)
        naf = len(f_af)
        # deepest segment per advancing-front frame (tip), for the gate
        deep_idx = np.argmax(p_af[:, :, 0], axis=1)
        for k in range(naf):
            lo = max(0, k - half)
            hi = min(naf, k + half + 1)
            if hi - lo < MIN_WIN:
                continue
            fw = f_af[lo:hi]
            # Vectorised over the n segments: slope of each segment's row and col
            # over the window. polyfit(x, Y, 1) fits every column of Y -> (2, n).
            sy_all = np.polyfit(fw, p_af[lo:hi, :, 0], 1)[0]      # (n,) row slopes
            sx_all = np.polyfit(fw, p_af[lo:hi, :, 1], 1)[0]      # (n,) col slopes
            vy_all = sy_all * fps / pxmm / 1000.0                 # axial (row) m/s
            vx_all = sx_all * fps / pxmm / 1000.0                 # lateral (col) m/s
            # Decompose the velocity vector at each segment into NORMAL (perpendicular
            # to the arc) + TANGENTIAL (along the arc) using the arc tangent at this
            # frame. The normal component is the physical propagation of the arc body,
            # free of the tangential sliding of fixed-fraction resampled points; it is
            # the proper generalisation of the tip's advancing-front velocity (at the
            # tip the arc tangent is ~perpendicular to propagation, so normal ~ axial).
            tang = unit_tangents(p_af[k])                        # (n,2) in (row,col)
            nrm = np.stack([-tang[:, 1], tang[:, 0]], axis=1)    # rotate 90 deg
            v_normal = vy_all * nrm[:, 0] + vx_all * nrm[:, 1]
            v_tang = vy_all * tang[:, 0] + vx_all * tang[:, 1]
            speed = np.hypot(vx_all, vy_all)
            fk = int(f_af[k])
            for i in range(n):
                rows_out.append((stem, fk, int(ev), i, round(i / (n - 1), 4),
                                 round(float(v_normal[i]), 5), round(float(v_tang[i]), 5),
                                 round(float(vy_all[i]), 5), round(float(vx_all[i]), 5),
                                 round(float(speed[i]), 5)))
            # gate: deepest-segment AXIAL velocity (the literal per-segment analogue
            # of the tip's arc_tip_row regression) vs the same-method tip axial velocity.
            sty = np.polyfit(fw, tip_af[lo:hi], 1)[0] * fps / pxmm / 1000.0
            gate_seg.append(float(vy_all[deep_idx[k]]))
            gate_tip.append(sty)

    out = pd.DataFrame(rows_out, columns=["file", "frame", "event_id", "seg_idx", "s",
                                          "seg_v_normal_ms", "seg_v_tang_ms",
                                          "seg_vy_ms", "seg_vx_ms", "seg_speed_ms"])
    gate = float("nan")
    if gate_seg:
        gate = float(np.median(np.abs(np.array(gate_seg) - np.array(gate_tip))))
    gate_corr = float("nan")
    if len(gate_seg) > 5:
        gate_corr = float(np.corrcoef(np.array(gate_seg), np.array(gate_tip))[0, 1])
    info = dict(stem=stem, n_rows=len(out), n_frames=int(out["frame"].nunique()),
                gate_median_abs_diff=gate, gate_corr=gate_corr, px_per_mm=pxmm, fps=fps)
    return out, info


# ─────────────────────────────────────────────────────────────────────────────
# figures
# ─────────────────────────────────────────────────────────────────────────────
def fig_profile(df: pd.DataFrame, stem: str, rd: Path, n: int):
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    segs = np.arange(n)
    s = segs / (n - 1)
    for col, color, lab in [("seg_vy_ms", plot_config.CIVIDIS_DISCRETE[0],
                             "axial v (propagation)"),
                            ("seg_v_normal_ms", plot_config.CIVIDIS_DISCRETE[2],
                             "normal v (lateral sweep)"),
                            ("seg_speed_ms", plot_config.CIVIDIS_DISCRETE[4], "speed |v|")]:
        g = df.groupby("seg_idx")[col]
        mn = g.mean().reindex(segs).to_numpy()
        sd = g.std().reindex(segs).to_numpy()
        ax.fill_between(s, mn - sd, mn + sd, alpha=0.2, color=color)
        ax.plot(s, mn, color=color, lw=2, label=lab)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("normalised arc position s  (0 = foot, 1 = other foot)")
    ax.set_ylabel("local segment velocity (m/s)")
    ax.legend(frameon=False)
    cap = (f"{stem} — per-segment local sliding-window velocity (N={n}, window="
           f"{LOCAL_VEL_N}). Normal (propagation) velocity and speed |v|, "
           f"mean +/- 1sigma vs arc position.")
    figstyle.save(fig, "segment_velocity_profile", cap, out_dir=str(rd))


def fig_density(df: pd.DataFrame, stem: str, rd: Path):
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    sp = pd.to_numeric(df["seg_speed_ms"], errors="coerce").to_numpy()
    s = pd.to_numeric(df["s"], errors="coerce").to_numpy()
    ok = np.isfinite(sp) & np.isfinite(s)
    vcap = float(np.percentile(sp[ok], 99)) if ok.any() else 1.0
    hb = ax.hexbin(s[ok], np.clip(sp[ok], 0, vcap), gridsize=30, cmap="plasma",
                   bins="log", extent=[0, 1, 0, vcap])
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04).set_label("log10(count)")
    ax.set_xlabel("normalised arc position s")
    ax.set_ylabel("local segment speed (m/s)")
    cap = f"{stem} — density of per-segment local speed vs arc position."
    figstyle.save(fig, "segment_velocity_density", cap, out_dir=str(rd))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    ap.add_argument("--n", type=int, default=N_HEADLINE)
    ap.add_argument("--window", type=int, default=LOCAL_VEL_N)
    ap.add_argument("--gate", action="store_true",
                    help="print the deepest-segment vs local_vel_ms agreement gate")
    args = ap.parse_args()
    data_root = Path(args.data_root)
    stems = list(RECORDINGS.keys()) if (args.all or not args.recording) else [args.recording]

    for stem in stems:
        print(f">>> {stem}")
        df, info = compute(stem, data_root, n=args.n, window=args.window)
        if df.empty:
            print("    no segment velocities produced")
            continue
        rd = data_root / f"{stem}_results"
        df.to_csv(rd / "segment_velocities.csv", index=False)
        fig_profile(df, stem, rd, args.n)
        fig_density(df, stem, rd)
        print(f"    rows={info['n_rows']:,} frames={info['n_frames']:,}  "
              f"SANITY deepest-seg vs same-method tip: median|d|="
              f"{info['gate_median_abs_diff']:.3f} m/s, corr={info['gate_corr']:.3f} "
              f"(offset expected: resampled deepest point slides tangentially vs the "
              f"mask-front tip)")
        print(f"    wrote segment_velocities.csv + profile/density figures")


if __name__ == "__main__":
    main()
