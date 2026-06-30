#!/usr/bin/env python3
"""
restrike_geometry_predictor.py  —  does the geometry of the LAST VISIBLE ARC
before a collapse predict whether the re-strike that follows is Mode A or Mode B?

For every re-strike (a collapse/trough in the arc-tip trajectory) we look back at
the FULLY DEVELOPED ARC immediately before it collapsed — the frame of maximum
extension in the cycle window leading up to the re-strike — and read its shape:
tortuosity, arc length, leg-length asymmetry, tortuosity asymmetry, width, area,
total curvature.  We then ask whether any of these separates the next mode (A vs
B), i.e. whether there is a CRITICAL TORTUOSITY / ARC LENGTH beyond which the arc
collapses one way rather than the other.

"Last visible arc before extinction" for re-strike i:
  * valid frames = ``arc_detected==1 AND ~is_flash`` (the set qa_restrike uses);
  * cycle window = (frame of re-strike i-1, frame of re-strike i]  (per file);
  * the last visible arc = the valid frame of MAX ``arc_tip_row`` (peak extension)
    in that window.  ``--anchor pretrough`` instead averages the last W=3 valid
    frames up to the re-strike (robustness variant).
Features are read straight from the existing per-frame geometry columns in
``measurements_all_events.csv`` (no reconstruction needed); ``area_mm2`` is
``area_px / px_per_mm**2`` from each recording's calibration.

Mode source: ``--source tip`` (re_strikes_qa.csv, authoritative) or
``--source npoint`` (npoint_restrikes.csv, whole-arc); ``--source both`` runs each.

Tests (scipy/numpy only; see predict_stats.py): per feature Mann-Whitney U +
Cliff's delta + ROC AUC with bootstrap CI + Youden cut (the "critical" value in
physical units), Benjamini-Hochberg FDR across features; multivariate logistic
with stratified 5-fold CV-AUC.  Run PER RECORDING and POOLED — a separating cut
is only claimed if it holds per-recording, not merely pooled (the recordings span
different Re, a confound).

USAGE
-----
    python restrike_geometry_predictor.py --all --source both
    python restrike_geometry_predictor.py --recording "HS_1.2mm_1500mL_(CO2)"
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
import predict_stats as ps
from npoint_restrike import RECORDINGS

DATA_ROOT_DEFAULT = Path(r"C:\Users\compl\Desktop\Plasma Arc\J Electrode")

# Geometry features (all in mm / dimensionless / radians; area converted to mm^2)
# so pooling across recordings of different px/mm is not confounded by pixel scale.
FEATURES = ["tortuosity", "arc_length_mm", "arc_width_mm", "arclen_asym",
            "tort_asym", "total_curvature_rad", "area_mm2"]
FEATURE_LABEL = {
    "tortuosity": "tortuosity (path/chord)",
    "arc_length_mm": "arc length (mm)",
    "arc_width_mm": "arc width (mm)",
    "arclen_asym": "leg-length asymmetry",
    "tort_asym": "tortuosity asymmetry",
    "total_curvature_rad": "total curvature (rad)",
    "area_mm2": "arc area (mm^2)",
}
W_PRETROUGH = 3


def load_meta(rd: Path) -> dict:
    txt = (rd / "run_metadata.json").read_text(encoding="utf-8", errors="replace")
    return json.loads(txt[:txt.rfind("}") + 1])


def valid_geometry(rd: Path, px_per_mm: float) -> pd.DataFrame:
    """Valid (detected & ~flash) frames with the geometry features, sorted by frame."""
    cols = ["frame", "arc_detected", "is_flash", "arc_tip_row",
            "tortuosity", "arc_length_mm", "arc_width_mm", "arclen_asym",
            "tort_asym", "total_curvature_rad", "area_px"]
    df = pd.read_csv(rd / "measurements_all_events.csv",
                     usecols=lambda c: c in cols)
    det = pd.to_numeric(df["arc_detected"], errors="coerce").fillna(0).astype(int)
    flash = pd.to_numeric(df.get("is_flash", 0), errors="coerce").fillna(0).astype(int)
    df = df[(det == 1) & (flash == 0)].copy()
    for c in ["arc_tip_row", "tortuosity", "arc_length_mm", "arc_width_mm",
              "arclen_asym", "tort_asym", "total_curvature_rad", "area_px"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["area_mm2"] = df["area_px"] / (px_per_mm ** 2)
    return df.dropna(subset=["arc_tip_row"]).sort_values("frame").reset_index(drop=True)


def restrike_frames(rd: Path, source: str, fname: str | None = None) -> pd.DataFrame:
    """(frame, mode) per re-strike, sorted by frame. fname overrides the detector
    file (e.g. an N-tagged npoint_restrikes_N50.csv for the N-segmentation sweep)."""
    fn = fname or ("re_strikes_qa.csv" if source == "tip" else "npoint_restrikes.csv")
    df = pd.read_csv(rd / fn)
    df = df[df["mode"].isin(["A", "B"])].copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="coerce")
    # npoint can have several breakpoint events at the same frame; keep one per frame
    df = df.dropna(subset=["frame"]).sort_values("frame")
    df = df.drop_duplicates(subset=["frame"], keep="first").reset_index(drop=True)
    return df[["frame", "mode"]]


def build_features(stem: str, data_root: Path, source: str, anchor: str = "peak",
                   rs_fname: str | None = None) -> pd.DataFrame:
    """One row per re-strike: next_mode + last-visible-arc geometry features."""
    rd = data_root / f"{stem}_results"
    meta = load_meta(rd)
    pxmm = float(meta["px_per_mm"])
    vg = valid_geometry(rd, pxmm)
    rs = restrike_frames(rd, source, fname=rs_fname)
    vf = vg["frame"].to_numpy()
    rows = []
    prev = -1
    for _, r in rs.iterrows():
        f_i = int(r["frame"])
        lo = int(np.searchsorted(vf, prev, side="right"))
        hi = int(np.searchsorted(vf, f_i, side="right"))   # include f_i
        prev = f_i
        if hi - lo < 1:
            continue
        win = vg.iloc[lo:hi]
        peak_pos = int(win["arc_tip_row"].values.argmax())
        peak_frame = int(win["frame"].values[peak_pos])
        if anchor == "pretrough":
            seg = win.tail(W_PRETROUGH)
            feat = {c: float(seg[c].mean()) for c in FEATURES}
        else:  # peak
            prow = win.iloc[peak_pos]
            feat = {c: float(prow[c]) for c in FEATURES}
        rows.append(dict(file=stem, restrike_frame=f_i, next_mode=r["mode"],
                         peak_frame=peak_frame, n_window=int(hi - lo), **feat))
    out = pd.DataFrame(rows)
    out.attrs["px_per_mm"] = pxmm
    return out


def per_feature_stats(df: pd.DataFrame, label_col: str = "next_mode") -> pd.DataFrame:
    """For each feature: median A/B, MWU p (+BH q), AUC(+CI), Cliff's delta, Youden
    cut. Positive class = Mode B."""
    y = (df[label_col].to_numpy() == "B")
    recs = []
    for f in FEATURES:
        s = pd.to_numeric(df[f], errors="coerce").to_numpy()
        ok = np.isfinite(s)
        s2, y2 = s[ok], y[ok]
        if len(s2) < 6 or y2.sum() == 0 or (~y2).sum() == 0:
            recs.append(dict(feature=f, n=int(ok.sum()), med_A=np.nan, med_B=np.nan,
                             mwu_p=np.nan, auc=np.nan, auc_lo=np.nan, auc_hi=np.nan,
                             cliffs=np.nan, cut=np.nan, cut_dir="", sens=np.nan,
                             spec=np.nan))
            continue
        auc = ps.auc_mannwhitney(s2, y2)
        lo, hi = ps.bootstrap_auc_ci(s2, y2)
        yc = ps.youden_cut(s2, y2)
        recs.append(dict(
            feature=f, n=int(ok.sum()),
            med_A=float(np.median(s2[~y2])), med_B=float(np.median(s2[y2])),
            mwu_p=ps.mannwhitney_p(s2, y2), auc=auc, auc_lo=lo, auc_hi=hi,
            cliffs=ps.cliffs_delta(s2, y2),
            cut=yc.get("threshold"), cut_dir=yc.get("direction"),
            sens=yc.get("sensitivity"), spec=yc.get("specificity")))
    out = pd.DataFrame(recs)
    out["mwu_q_bh"] = ps.bh_fdr(out["mwu_p"].to_numpy())
    return out


def multivariate(df: pd.DataFrame) -> dict:
    y = (df["next_mode"].to_numpy() == "B")
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy()
    ok = np.isfinite(X).all(axis=1)
    X, y = X[ok], y[ok]
    if len(y) < 20 or y.sum() < 5 or (~y).sum() < 5:
        return dict(cv_auc=float("nan"), n=int(len(y)))
    mu, sd = X.mean(0), X.std(0) + 1e-9
    w = ps.logistic_fit((X - mu) / sd, y.astype(float))
    coefs = {f: float(c) for f, c in zip(FEATURES, w[1:])}
    return dict(cv_auc=ps.cv_auc(X, y), n=int(len(y)),
                in_sample_auc=ps.auc_mannwhitney(ps.logistic_proba(w, (X - mu) / sd), y),
                coefs=coefs)


# ─────────────────────────────────────────────────────────────────────────────
# figures
# ─────────────────────────────────────────────────────────────────────────────
def fig_feature_distributions(df: pd.DataFrame, stem: str, source: str, out_dir: Path):
    figstyle.apply()
    n = len(FEATURES)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.0 * nrow), squeeze=False)
    cA, cB = plot_config.CIVIDIS_DISCRETE[0], plot_config.CIVIDIS_DISCRETE[4]
    y = df["next_mode"].to_numpy()
    for i, f in enumerate(FEATURES):
        ax = axes[i // ncol][i % ncol]
        s = pd.to_numeric(df[f], errors="coerce").to_numpy()
        a = s[(y == "A") & np.isfinite(s)]
        b = s[(y == "B") & np.isfinite(s)]
        if len(a) and len(b):
            lo = np.nanpercentile(np.concatenate([a, b]), 1)
            hi = np.nanpercentile(np.concatenate([a, b]), 99)
            bins = np.linspace(lo, hi, 30)
            ax.hist(a, bins=bins, density=True, alpha=0.55, color=cA, label="next A")
            ax.hist(b, bins=bins, density=True, alpha=0.55, color=cB, label="next B")
            ax.axvline(np.median(a), color=cA, ls="--", lw=1.2)
            ax.axvline(np.median(b), color=cB, ls="--", lw=1.2)
        ax.set_title(FEATURE_LABEL[f], fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, frameon=False)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    cap = (f"{stem} [{source}] — last-visible-arc geometry by next re-strike mode. "
           f"Distributions of A (next Mode A) vs B (next Mode B); dashed = medians.")
    figstyle.save(fig, f"restrike_geometry_dist_{source}", cap, out_dir=str(out_dir))


def fig_auc_summary(stats_df: pd.DataFrame, stem: str, source: str, out_dir: Path):
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    d = stats_df.dropna(subset=["auc"]).copy()
    d["sep"] = (d["auc"] - 0.5).abs()
    d = d.sort_values("sep")
    yy = np.arange(len(d))
    auc = d["auc"].to_numpy()
    ax.errorbar(auc, yy, xerr=[auc - d["auc_lo"], d["auc_hi"] - auc],
                fmt="o", color=plot_config.CIVIDIS_DISCRETE[1], capsize=3)
    ax.axvline(0.5, color="grey", ls=":", lw=1)
    ax.set_yticks(yy)
    ax.set_yticklabels([FEATURE_LABEL[f] for f in d["feature"]], fontsize=8)
    ax.set_xlabel("ROC AUC (last-arc feature -> next mode = B), 95% bootstrap CI")
    ax.set_xlim(0, 1)
    cap = (f"{stem} [{source}] — discriminating power of each last-visible-arc "
           f"feature for the next re-strike mode (0.5 = no separation).")
    figstyle.save(fig, f"restrike_geometry_auc_{source}", cap, out_dir=str(out_dir))


# ─────────────────────────────────────────────────────────────────────────────
# driver
# ─────────────────────────────────────────────────────────────────────────────
def run_source(stems: list, data_root: Path, source: str, anchor: str):
    out_root = data_root / "analysis" / "geometry_predictor"
    out_root.mkdir(parents=True, exist_ok=True)
    per_recording_stats = []
    pooled_frames = []
    mv_rows = []
    for stem in stems:
        rd = data_root / f"{stem}_results"
        if not (rd / "re_strikes_qa.csv").exists():
            continue
        feats = build_features(stem, data_root, source, anchor=anchor)
        if feats.empty:
            print(f"  [{stem}] no features built ({source})")
            continue
        feats.to_csv(rd / f"restrike_geometry_features_{source}.csv", index=False)
        pooled_frames.append(feats)
        sdf = per_feature_stats(feats)
        sdf.insert(0, "file", stem)
        info = RECORDINGS.get(stem, {})
        sdf.insert(1, "Re", info.get("re"))
        nA = int((feats["next_mode"] == "A").sum())
        nB = int((feats["next_mode"] == "B").sum())
        sdf.insert(2, "n_A", nA); sdf.insert(3, "n_B", nB)
        per_recording_stats.append(sdf)
        mv = multivariate(feats)
        mv_rows.append(dict(file=stem, Re=info.get("re"), n=mv["n"],
                            cv_auc=mv.get("cv_auc"), in_sample_auc=mv.get("in_sample_auc")))
        fig_feature_distributions(feats, stem, source, rd)
        fig_auc_summary(sdf, stem, source, rd)
        print(f"  [{stem}] {source}: n={len(feats)} (A={nA},B={nB})  "
              f"multivar CV-AUC={mv.get('cv_auc'):.3f}")

    if not pooled_frames:
        print(f"  no data for source={source}")
        return
    pooled = pd.concat(pooled_frames, ignore_index=True)
    pstats = per_feature_stats(pooled)
    pstats.insert(0, "file", "POOLED")
    pstats.insert(1, "Re", np.nan)
    pstats.insert(2, "n_A", int((pooled["next_mode"] == "A").sum()))
    pstats.insert(3, "n_B", int((pooled["next_mode"] == "B").sum()))
    per_recording_stats.append(pstats)
    mvp = multivariate(pooled)
    mv_rows.append(dict(file="POOLED", Re=np.nan, n=mvp["n"],
                        cv_auc=mvp.get("cv_auc"), in_sample_auc=mvp.get("in_sample_auc")))

    allstats = pd.concat(per_recording_stats, ignore_index=True)
    allstats.to_csv(out_root / f"geometry_predictor_stats_{source}.csv", index=False)
    pd.DataFrame(mv_rows).to_csv(out_root / f"geometry_predictor_multivar_{source}.csv",
                                 index=False)
    fig_feature_distributions(pooled, "POOLED", source, out_root)
    fig_auc_summary(pstats, "POOLED", source, out_root)
    write_report(allstats, pd.DataFrame(mv_rows), source, out_root)
    print(f"  wrote geometry_predictor_stats_{source}.csv + report to {out_root}")


def write_report(allstats: pd.DataFrame, mv: pd.DataFrame, source: str, out_root: Path):
    L = [f"# Last-visible-arc geometry -> next re-strike mode  [{source}]\n",
         "Does the shape of the fully-developed arc (frame of peak extension before "
         "collapse) predict whether the following re-strike is Mode A or Mode B? "
         "Positive class = Mode B; AUC>0.5 means a larger feature favours Mode B.\n"]
    pooled = allstats[allstats["file"] == "POOLED"]
    L.append("## Pooled discriminating power (per feature)\n")
    L.append("| feature | median A | median B | AUC [95% CI] | Cliff's d | MWU q (BH) | "
             "Youden cut | sens/spec |")
    L.append("|---|--:|--:|--|--:|--:|--|--|")
    for _, r in pooled.sort_values("auc", key=lambda s: (s - 0.5).abs(),
                                   ascending=False).iterrows():
        if not np.isfinite(r["auc"]):
            continue
        L.append(f"| {FEATURE_LABEL[r['feature']]} | {r['med_A']:.3g} | {r['med_B']:.3g} | "
                 f"{r['auc']:.2f} [{r['auc_lo']:.2f},{r['auc_hi']:.2f}] | {r['cliffs']:+.2f} | "
                 f"{r['mwu_q_bh']:.2g} | {r['cut_dir']} {r['cut']:.3g} | "
                 f"{r['sens']:.2f}/{r['spec']:.2f} |")
    # consistency across recordings: a cut is only credible if AUC>0.5 (same side)
    # holds per recording
    L.append("\n## Cross-recording consistency (is any cut real, not a Re confound?)\n")
    for f in FEATURES:
        sub = allstats[(allstats["feature"] == f) & (allstats["file"] != "POOLED")]
        aucs = sub["auc"].dropna()
        if len(aucs) == 0:
            continue
        above = (aucs > 0.55).sum(); below = (aucs < 0.45).sum()
        consistent = "B-side" if above >= max(1, len(aucs) - 1) else (
            "A-side" if below >= max(1, len(aucs) - 1) else "INCONSISTENT")
        L.append(f"- {FEATURE_LABEL[f]}: per-recording AUC range "
                 f"[{aucs.min():.2f},{aucs.max():.2f}] median {aucs.median():.2f} -> "
                 f"**{consistent}**")
    L.append("\n## Multivariate logistic (all features), cross-validated AUC\n")
    L.append("| recording | n | CV-AUC | in-sample AUC |")
    L.append("|---|--:|--:|--:|")
    for _, r in mv.iterrows():
        cv = r["cv_auc"]; ins = r["in_sample_auc"]
        L.append(f"| {r['file']} | {int(r['n'])} | "
                 f"{cv:.3f} | {ins:.3f} |" if np.isfinite(cv) else
                 f"| {r['file']} | {int(r['n'])} | n/a | n/a |")
    pooled_mv = mv[mv["file"] == "POOLED"]
    cvp = float(pooled_mv["cv_auc"].iloc[0]) if len(pooled_mv) else float("nan")
    L.append("\n## Interpretation\n")
    if np.isfinite(cvp):
        verdict = ("strong" if cvp >= 0.75 else "moderate" if cvp >= 0.65 else
                   "weak" if cvp >= 0.58 else "no useful")
        L.append(f"- Pooled multivariate CV-AUC = **{cvp:.3f}** -> {verdict} prediction "
                 f"of the next mode from last-arc geometry.")
    L.append("- A 'critical tortuosity / arc length' is credible only for features "
             "flagged **B-side** or **A-side** above (consistent across recordings); "
             "features marked INCONSISTENT separate the pooled data only because the "
             "recordings sit at different Re, not because of a within-recording "
             "geometric threshold.")
    (out_root / f"geometry_predictor_report_{source}.md").write_text(
        "\n".join(L) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    ap.add_argument("--source", choices=["tip", "npoint", "both"], default="both")
    ap.add_argument("--anchor", choices=["peak", "pretrough"], default="peak")
    args = ap.parse_args()
    data_root = Path(args.data_root)
    stems = list(RECORDINGS.keys()) if (args.all or not args.recording) else [args.recording]
    sources = ["tip", "npoint"] if args.source == "both" else [args.source]
    for src in sources:
        print(f"=== source={src} anchor={args.anchor} ===")
        run_source(stems, data_root, src, args.anchor)


if __name__ == "__main__":
    main()
