#!/usr/bin/env python3
"""
fig_reset_to_modeA.py  —  SHOW (not just state) that every reset interval is
followed by a Mode-A (apex) re-strike, and incidentally why the tip and 10-segment
detectors disagree.

Two panels (1.2 mm recording, the only one with long dark gaps):

  LEFT  — a representative time window of the arc-tip DEPTH (distance below the
          apex). Reset intervals (no usable arc) are shaded grey. Re-strikes are
          marked: big circles = the TIP detector, small triangles = the 10-SEGMENT
          detector, coloured by mode (A = blue, B = red). You can see (a) the first
          re-strike after every grey gap sits at the apex and is blue (Mode A), and
          (b) the 10-segment detector fires many MORE times between tip cycles
          (the leg-side back-breakdowns the tip never sees) — which is why the two
          methods give different Mode-B fractions and transition structure.

  RIGHT — a peri-reset raster: one row per reset interval (n=53), columns = the
          1st, 2nd, 3rd ... re-strike AFTER that reset. Cell colour = mode. The
          first column is entirely blue: 53/53 resets are followed by Mode A.

    python fig_reset_to_modeA.py            # writes into the 1.2mm results dir
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

DATA_ROOT = Path(r"C:\Users\compl\Desktop\Plasma Arc\J Electrode")
STEM = "HS_1.2mm_1500mL_(CO2)"
RESET_GAP = 500
K = 8                       # re-strikes shown after each reset in the raster
CA, CB = "#3a6fb0", "#d6604d"   # Mode A / Mode B


def reset_intervals(meas, reset_gap):
    """Maximal runs of no-usable-arc (arc_detected!=1 OR is_flash) >= reset_gap frames.
    Self-contained (mirrors arc_resets.reset_intervals) so this script needs no repo
    imports when run from the data-dir paper_figures."""
    d = meas.sort_values("frame")
    fr = pd.to_numeric(d["frame"], errors="coerce").to_numpy()
    det = pd.to_numeric(d["arc_detected"], errors="coerce").fillna(0).to_numpy()
    fl = pd.to_numeric(d.get("is_flash", 0), errors="coerce").fillna(0).to_numpy()
    inv = (det != 1) | (fl == 1)
    fmin, fmax = int(fr.min()), int(fr.max())
    full = np.ones(fmax - fmin + 1, dtype=bool)          # missing frames -> invalid
    full[fr.astype(int) - fmin] = inv
    edges = np.flatnonzero(np.diff(np.concatenate(([0], full.astype(np.int8), [0]))))
    s, e = edges[0::2], edges[1::2] - 1
    rng = np.arange(fmin, fmax + 1)
    sf, ef = rng[s], rng[e]
    gap = ef - sf + 1
    keep = gap >= reset_gap
    return pd.DataFrame(dict(start_frame=sf[keep], end_frame=ef[keep], gap_frames=gap[keep]))


def main():
    rd = DATA_ROOT / f"{STEM}_results"
    meta = json.loads((rd / "run_metadata.json").read_text(encoding="utf-8",
                                                           errors="replace").rsplit("}", 1)[0] + "}")
    ppm = float(meta["px_per_mm"]); gap_row = float(meta["gap_row"]); fps = float(meta["fps"])

    meas = pd.read_csv(rd / "measurements_all_events.csv",
                       usecols=["frame", "arc_detected", "is_flash", "arc_tip_row"]).sort_values("frame")
    resets = reset_intervals(meas, reset_gap=RESET_GAP)

    npt = pd.read_csv(rd / "npoint_restrikes.csv")
    npt = npt[npt["mode"].isin(["A", "B"])].sort_values("frame")
    tip = pd.read_csv(rd / "re_strikes_qa.csv")
    tip = tip[tip["mode"].isin(["A", "B"])].sort_values("frame")
    npf, npm = npt["frame"].to_numpy(), npt["mode"].to_numpy()
    tpf, tpm = tip["frame"].to_numpy(), tip["mode"].to_numpy()

    # ---- peri-reset raster: first K re-strikes (10-segment) after each reset end ----
    ends = resets["end_frame"].to_numpy()
    raster = np.full((len(ends), K), np.nan)     # 0=A, 1=B
    for i, e in enumerate(ends):
        pos = np.searchsorted(npf, e, side="right")
        seq = npm[pos:pos + K]
        for j, mm in enumerate(seq):
            raster[i, j] = 0.0 if mm == "A" else 1.0
    # order rows by reset time (already sorted)
    fracA_col = np.nanmean(raster == 0, axis=0)

    # ---- choose a representative window: two consecutive resets that are close ----
    rs_start = resets["start_frame"].to_numpy(); rs_end = resets["end_frame"].to_numpy()
    gaps_between = rs_start[1:] - rs_end[:-1]
    cand = np.where(gaps_between > 300)[0]       # need some cycling between the two
    i0 = cand[np.argmin(gaps_between[cand])] if len(cand) else 0
    w0 = int(rs_start[i0] - 300); w1 = int(rs_end[i0 + 1] + 900)

    mw = meas[(meas["frame"] >= w0) & (meas["frame"] <= w1)].copy()
    valid = (pd.to_numeric(mw["arc_detected"], errors="coerce").fillna(0).astype(int) == 1) & \
            (pd.to_numeric(mw["is_flash"], errors="coerce").fillna(0).astype(int) == 0)
    mv = mw[valid]
    fr_v = mv["frame"].to_numpy()
    t_ms = (fr_v - w0) / fps * 1e3
    depth_mm = (pd.to_numeric(mv["arc_tip_row"], errors="coerce").to_numpy() - gap_row) / ppm
    # break the trace where the arc is absent (don't draw a line through dark gaps)
    brk = np.where(np.diff(fr_v) > 100)[0] + 1
    t_line = np.insert(t_ms, brk, np.nan)
    depth_line = np.insert(depth_mm, brk, np.nan)

    fig = plt.figure(figsize=(13.5, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.25)

    # ===================== LEFT: representative trace =====================
    axL = fig.add_subplot(gs[0, 0])
    axL.plot(t_line, depth_line, "-", color="0.6", lw=0.8, zorder=1)
    # shade reset intervals in the window
    for s, e in zip(rs_start, rs_end):
        if e >= w0 and s <= w1:
            axL.axvspan((s - w0) / fps * 1e3, (e - w0) / fps * 1e3,
                        color="0.85", zorder=0)
    # tip re-strikes (big circles) at their depth
    depth_at = dict(zip(mv["frame"].to_numpy(), depth_mm))
    for f, m in zip(tpf, tpm):
        if w0 <= f <= w1:
            y = depth_at.get(int(f), np.nan)
            axL.scatter([(f - w0) / fps * 1e3], [y if np.isfinite(y) else 0.2],
                        s=90, marker="o", facecolor=(CA if m == "A" else CB),
                        edgecolor="black", linewidths=0.8, zorder=4)
    # 10-segment re-strikes (small triangles), drawn near the apex line so the
    # extra leg-break events are visible as a dense row
    for f, m in zip(npf, npm):
        if w0 <= f <= w1:
            axL.scatter([(f - w0) / fps * 1e3], [-0.6], s=26, marker="^",
                        facecolor=(CA if m == "A" else CB), edgecolor="none",
                        zorder=3, alpha=0.9)
    axL.axhline(0, color="k", lw=0.8, ls="--")
    axL.text(0.01, 0.97, "grey = reset (arc dark)", transform=axL.transAxes,
             va="top", fontsize=9, color="0.4")
    axL.set_xlabel("time (ms)"); axL.set_ylabel("arc-tip depth below apex (mm)")
    axL.set_title("After every dark gap the arc re-strikes at the apex (Mode A);\n"
                  "the 10-segment detector (▲) also fires on leg breaks the tip (●) misses",
                  fontsize=11)
    axL.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.5",
               markeredgecolor="k", markersize=10, label="tip re-strike"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="0.5",
               markersize=9, label="10-segment re-strike"),
        Patch(facecolor=CA, label="Mode A"), Patch(facecolor=CB, label="Mode B")],
        loc="upper right", fontsize=8, ncol=2)

    # ===================== RIGHT: peri-reset raster =====================
    axR = fig.add_subplot(gs[0, 1])
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([CA, CB]); cmap.set_bad("white")
    axR.imshow(raster, aspect="auto", cmap=cmap, vmin=0, vmax=1,
               interpolation="nearest")
    axR.set_xticks(range(K))
    axR.set_xticklabels([f"{i+1}\n{100*fracA_col[i]:.0f}%A" for i in range(K)], fontsize=8)
    axR.set_xlabel("n-th re-strike after the reset  (and % Mode A in that column)")
    axR.set_ylabel("reset interval (sorted by time)")
    axR.set_title(f"Every reset is followed by Mode A\n"
                  f"(column 1 = {int(np.nansum(raster[:,0]==0))}/"
                  f"{int(np.isfinite(raster[:,0]).sum())} Mode A)", fontsize=11, pad=10)
    axR.legend(handles=[Patch(facecolor=CA, label="Mode A"),
                        Patch(facecolor=CB, label="Mode B")],
               loc="lower right", fontsize=8, framealpha=0.9)

    # write into ARC_OUT (the per-method paper-figure folder) when set, else the
    # recording's results dir
    out_dir = Path(os.environ.get("ARC_OUT") or rd)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "fig_reset_to_modeA"
    fig.savefig(str(out) + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.png/.pdf")
    print(f"  window frames [{w0},{w1}]  ({(w1-w0)/fps*1e3:.1f} ms); "
          f"raster {raster.shape[0]} resets; col-1 Mode-A = {100*fracA_col[0]:.0f}%")


if __name__ == "__main__":
    main()
