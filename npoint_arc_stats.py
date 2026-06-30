#!/usr/bin/env python3
"""
npoint_arc_stats.py  —  advanced tortuosity + arc-characteristic statistics,
split by re-strike mode (A vs B) and as a function of Reynolds number.

The per-frame arc geometry is already measured by the pipeline and stored in
``measurements_all_events.csv`` (``tortuosity``, ``arc_length_mm``,
``arc_width_mm``, ``arclen_asym``, ``tort_asym``, ``total_curvature_rad``,
``curvature_mean``, ``area_px``).  This script:

1. assigns every valid arc frame the MODE of the cycle it belongs to (the next
   re-strike's mode, via ``merge_asof`` forward on ``re_strikes_qa.csv``), so the
   arc-shape distributions can be compared between Mode-A and Mode-B cycles;
2. summarises each characteristic (median / IQR / p90) per recording and per mode,
   and plots each vs Reynolds number (nozzle-coloured, A vs B);
3. RECONCILES the pipeline's full-skeleton ``tortuosity`` against a tortuosity
   recomputed from the robust N=10 resampled arc (``npoint_geometry``):
   ``tort10 = (sum of the 10-point chord lengths) / (foot-to-foot distance)`` and
   the N=10 total turning angle — confirming the coarse-grained 10-segment measure
   tracks the full-skeleton one (Pearson r + median ratio).

Per the project decision, the *existing* per-frame columns are the authoritative
geometry source; the N=10 quantities are provided as the reconciliation /
robust-smoothed companion requested ("use the 10-segmented arc approach").

OUTPUTS (under <data-root>/analysis/arc_stats/)
  arc_stats_summary.csv          per recording x mode: median/IQR/p90 of each feature
  arc_stats_vs_re.{png,pdf,txt}  each characteristic vs Re, Mode A vs Mode B
  tortuosity_reconciliation.{png,pdf,txt}  N=10 tort vs full-skeleton tort
  arc_stats_report.md

USAGE
  python npoint_arc_stats.py --all
"""
from __future__ import annotations

import json
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import figstyle
import plot_config
from npoint_restrike import RECORDINGS, N_HEADLINE, load_metadata, detector_params
from npoint_geometry import points_for

DATA_ROOT_DEFAULT = Path(r"C:\Users\compl\Desktop\Plasma Arc\J Electrode")

FEATURES = ["tortuosity", "arc_length_mm", "arc_width_mm", "arclen_asym",
            "tort_asym", "total_curvature_rad", "curvature_mean", "area_mm2"]
FEATURE_LABEL = {
    "tortuosity": "tortuosity", "arc_length_mm": "arc length (mm)",
    "arc_width_mm": "arc width (mm)", "arclen_asym": "leg-length asym",
    "tort_asym": "tortuosity asym", "total_curvature_rad": "total curvature (rad)",
    "curvature_mean": "mean curvature", "area_mm2": "area (mm^2)"}


def valid_frames_with_mode(rd: Path, pxmm: float) -> pd.DataFrame:
    """Valid (detected & ~flash) frames with geometry + the mode of the cycle they
    belong to (next re-strike's mode)."""
    use = ["frame", "arc_detected", "is_flash", "tortuosity", "arc_length_mm",
           "arc_width_mm", "arclen_asym", "tort_asym", "total_curvature_rad",
           "curvature_mean", "area_px"]
    df = pd.read_csv(rd / "measurements_all_events.csv",
                     usecols=lambda c: c in use)
    det = pd.to_numeric(df["arc_detected"], errors="coerce").fillna(0).astype(int)
    flash = pd.to_numeric(df.get("is_flash", 0), errors="coerce").fillna(0).astype(int)
    df = df[(det == 1) & (flash == 0)].copy()
    for c in FEATURES[:-1] + ["area_px"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["area_mm2"] = df["area_px"] / (pxmm ** 2)
    df = df.sort_values("frame").reset_index(drop=True)

    rs = pd.read_csv(rd / "re_strikes_qa.csv")
    rs = rs[rs["mode"].isin(["A", "B"])][["frame", "mode"]].copy()
    rs["frame"] = pd.to_numeric(rs["frame"], errors="coerce")
    rs = rs.dropna(subset=["frame"]).sort_values("frame")
    # each frame belongs to the cycle ending at the NEXT re-strike
    merged = pd.merge_asof(df, rs.rename(columns={"frame": "rs_frame", "mode": "cycle_mode"}),
                           left_on="frame", right_on="rs_frame", direction="forward")
    return merged


def n10_tortuosity(rd: Path, n: int = N_HEADLINE) -> pd.DataFrame:
    """tort10 (path/chord) and turn10 (total turning angle) per frame from the
    robust N=10 resampled arc."""
    frames, pts = points_for(rd, N=n)
    if len(frames) == 0:
        return pd.DataFrame(columns=["frame", "tort10", "turn10"])
    seg = np.diff(pts, axis=1)                                  # (M,n-1,2)
    seglen = np.hypot(seg[:, :, 0], seg[:, :, 1])              # (M,n-1)
    path = seglen.sum(axis=1)
    chord = np.hypot(pts[:, -1, 0] - pts[:, 0, 0], pts[:, -1, 1] - pts[:, 0, 1])
    tort10 = path / np.maximum(chord, 1e-6)
    # turning angle at interior points
    v1 = seg[:, :-1, :]; v2 = seg[:, 1:, :]
    dot = (v1 * v2).sum(axis=2)
    crs = v1[:, :, 0] * v2[:, :, 1] - v1[:, :, 1] * v2[:, :, 0]
    ang = np.abs(np.arctan2(crs, dot))
    turn10 = ang.sum(axis=1)
    return pd.DataFrame(dict(frame=frames.astype(int), tort10=tort10, turn10=turn10))


def summarise(stem: str, data_root: Path):
    rd = data_root / f"{stem}_results"
    meta = load_metadata(rd)
    pxmm = detector_params(meta)["px_per_mm"]
    info = RECORDINGS.get(stem, {})
    vf = valid_frames_with_mode(rd, pxmm)
    t10 = n10_tortuosity(rd)
    vf = vf.merge(t10, on="frame", how="left")

    rows = []
    for mode in ["A", "B", "all"]:
        sub = vf if mode == "all" else vf[vf["cycle_mode"] == mode]
        if len(sub) == 0:
            continue
        rec = dict(file=stem, Re=info.get("re"), d_mm=info.get("d_mm"),
                   q_mLpm=info.get("q"), mode=mode, n_frames=len(sub))
        for f in FEATURES:
            v = pd.to_numeric(sub[f], errors="coerce").dropna()
            if len(v):
                rec[f"{f}_median"] = float(v.median())
                rec[f"{f}_iqr"] = float(v.quantile(0.75) - v.quantile(0.25))
                rec[f"{f}_p90"] = float(v.quantile(0.90))
        rows.append(rec)
    # reconciliation (all valid frames with both measures)
    both = vf.dropna(subset=["tortuosity", "tort10"])
    recon = dict(file=stem, Re=info.get("re"), n=len(both))
    if len(both) > 10:
        recon["pearson_r"] = float(np.corrcoef(both["tortuosity"], both["tort10"])[0, 1])
        recon["median_ratio_t10_over_full"] = float(
            np.median(both["tort10"] / np.maximum(both["tortuosity"], 1e-6)))
    return rows, recon, vf


# ─────────────────────────────────────────────────────────────────────────────
# figures
# ─────────────────────────────────────────────────────────────────────────────
def fig_vs_re(summary: pd.DataFrame, out_dir: Path):
    figstyle.apply()
    feats = FEATURES
    ncol = 4
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow), squeeze=False)
    for i, f in enumerate(feats):
        ax = axes[i // ncol][i % ncol]
        for mode, mk in [("A", "o"), ("B", "s")]:
            sub = summary[summary["mode"] == mode].dropna(subset=[f"{f}_median", "Re"])
            cols = [plot_config.nozzle_color(d) for d in sub["d_mm"]]
            ax.scatter(sub["Re"], sub[f"{f}_median"], marker=mk, c=cols, s=55,
                       edgecolors="k", linewidths=0.4,
                       label=f"Mode {mode}" if i == 0 else None)
        ax.set_title(FEATURE_LABEL[f], fontsize=9)
        ax.set_xlabel("Re", fontsize=8); ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for j in range(len(feats), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    # legend (mode markers + nozzle colours)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
                      markeredgecolor="k", markersize=8, label="Mode A (o)"),
               Line2D([0], [0], marker="s", color="w", markerfacecolor="#555",
                      markeredgecolor="k", markersize=8, label="Mode B (s)")]
    handles += [Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=plot_config.nozzle_color(d), markersize=8,
                       label=f"{d} mm") for d in (0.4, 0.8, 1.2)]
    axes[0][0].legend(handles=handles, fontsize=6, frameon=False, ncol=1)
    cap = ("Arc-characteristic medians vs Reynolds number, by cycle mode (A=o, B=s; "
           "nozzle-coloured). Median over all valid arc frames per recording x mode.")
    figstyle.save(fig, "arc_stats_vs_re", cap, out_dir=str(out_dir))


def fig_reconcile(vf_all: pd.DataFrame, out_dir: Path):
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    d = vf_all.dropna(subset=["tortuosity", "tort10"])
    if len(d) > 50000:
        d = d.sample(50000, random_state=0)
    ax.hexbin(d["tortuosity"], d["tort10"], gridsize=50, cmap="cividis", bins="log")
    lim = [1, float(np.nanpercentile(d[["tortuosity", "tort10"]].to_numpy(), 99))]
    ax.plot(lim, lim, "r--", lw=1, label="y = x")
    r = float(np.corrcoef(d["tortuosity"], d["tort10"])[0, 1])
    ax.set_xlabel("full-skeleton tortuosity (pipeline)")
    ax.set_ylabel("N=10 resampled tortuosity")
    ax.set_title(f"tortuosity reconciliation  (Pearson r = {r:.3f})", fontsize=10)
    ax.legend(frameon=False)
    cap = ("Reconciliation of the robust N=10 resampled tortuosity against the "
           "pipeline's full-skeleton tortuosity over all valid frames (pooled). "
           "The N=10 measure is a coarse-grained lower bound; r quantifies agreement.")
    figstyle.save(fig, "tortuosity_reconciliation", cap, out_dir=str(out_dir))


def write_report(summary: pd.DataFrame, recon: pd.DataFrame, out_dir: Path):
    L = ["# Arc-characteristic statistics (by mode, vs Re)\n",
         "Per-frame arc geometry from the pipeline (full-skeleton), summarised over "
         "all valid frames, split by the mode of the cycle each frame belongs to, "
         "with the robust N=10 tortuosity as a reconciliation companion.\n",
         "## Median arc characteristics per recording (mode = all)\n",
         "| recording | Re | tortuosity | arc len (mm) | width (mm) | leg asym | "
         "tort asym | tot curv (rad) |",
         "|---|--:|--:|--:|--:|--:|--:|--:|"]
    allm = summary[summary["mode"] == "all"].sort_values("Re")
    for _, r in allm.iterrows():
        L.append(f"| {r['file'].replace('HS_','')} | {int(r['Re'])} | "
                 f"{r.get('tortuosity_median', float('nan')):.2f} | "
                 f"{r.get('arc_length_mm_median', float('nan')):.1f} | "
                 f"{r.get('arc_width_mm_median', float('nan')):.2f} | "
                 f"{r.get('arclen_asym_median', float('nan')):+.3f} | "
                 f"{r.get('tort_asym_median', float('nan')):+.3f} | "
                 f"{r.get('total_curvature_rad_median', float('nan')):.1f} |")
    L.append("\n## Mode A vs Mode B (median, pooled direction of difference)\n")
    for f in FEATURES:
        a = summary[summary["mode"] == "A"][f"{f}_median"].dropna()
        b = summary[summary["mode"] == "B"][f"{f}_median"].dropna()
        if len(a) and len(b):
            L.append(f"- {FEATURE_LABEL[f]}: median(A)={a.median():.3g}, "
                     f"median(B)={b.median():.3g} "
                     f"(B {'>' if b.median() > a.median() else '<'} A across recordings)")
    L.append("\n## N=10 vs full-skeleton tortuosity reconciliation\n")
    L.append("| recording | n | Pearson r | median(tort10/full) |")
    L.append("|---|--:|--:|--:|")
    for _, r in recon.sort_values("Re").iterrows():
        L.append(f"| {r['file'].replace('HS_','')} | {int(r['n'])} | "
                 f"{r.get('pearson_r', float('nan')):.3f} | "
                 f"{r.get('median_ratio_t10_over_full', float('nan')):.3f} |")
    (out_dir / "arc_stats_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    args = ap.parse_args()
    data_root = Path(args.data_root)
    stems = list(RECORDINGS.keys()) if (args.all or not args.recording) else [args.recording]

    out_dir = data_root / "analysis" / "arc_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, recon_rows, vf_pool = [], [], []
    for stem in stems:
        rd = data_root / f"{stem}_results"
        if not (rd / "re_strikes_qa.csv").exists():
            continue
        rows, recon, vf = summarise(stem, data_root)
        all_rows.extend(rows)
        recon_rows.append(recon)
        vf_pool.append(vf[["tortuosity", "tort10"]])
        print(f"  [{stem}] {len(rows)} mode-groups, recon r="
              f"{recon.get('pearson_r', float('nan')):.3f}")
    summary = pd.DataFrame(all_rows)
    summary.to_csv(out_dir / "arc_stats_summary.csv", index=False)
    recon = pd.DataFrame(recon_rows)
    fig_vs_re(summary, out_dir)
    fig_reconcile(pd.concat(vf_pool, ignore_index=True), out_dir)
    write_report(summary, recon, out_dir)
    print(f"Wrote arc_stats_summary.csv + figures + report to {out_dir}")


if __name__ == "__main__":
    main()
