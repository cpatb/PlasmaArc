#!/usr/bin/env python3
"""
nseg_compare.py  —  cross-N comparison of the whole-arc characterizations at
N = 10, 20, 50 segments (the N-sensitivity exploration). Answers: does finer arc
segmentation change the conclusions?

Reads the N-tagged pipeline outputs (npoint_restrikes{sfx}.csv,
mode_transition_summary_npoint{sfx}.csv, geometry_predictor_multivar_npoint{sfx}.csv,
segment_velocities{sfx}.csv; sfx = "" for N=10 else _N{n}) and writes, into
paper_figures/:
  fig_Nsegment_comparison.{png,pdf}   5-panel comparison
  nseg_comparison_summary.csv         the numbers behind it

N's whose data is not present yet are skipped (so this can run before N=50 finishes).
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import arc_resets
from npoint_restrike import RECORDINGS

DATA_ROOT = Path(r"C:\Users\compl\Desktop\Plasma Arc\J Electrode")
OUT = DATA_ROOT / "paper_figures"
NS = [10, 20, 50]
NCOL = {10: "#1a9850", 20: "#2166ac", 50: "#d6604d"}
BIG = "HS_1.2mm_1500mL_(CO2)"
RESET_GAP = 500


def sfx(n):
    return "" if n == 10 else f"_N{n}"


def restrikes_path(stem, n):
    return DATA_ROOT / f"{stem}_results" / f"npoint_restrikes{sfx(n)}.csv"


def reset_pA(stem, n):
    rd = DATA_ROOT / f"{stem}_results"
    rp = restrikes_path(stem, n)
    if not rp.exists():
        return np.nan, 0
    meas = pd.read_csv(rd / "measurements_all_events.csv",
                       usecols=["frame", "arc_detected", "is_flash"])
    resets = arc_resets.reset_intervals(meas, reset_gap=RESET_GAP)
    rs = pd.read_csv(rp)
    rs = rs[rs["mode"].isin(["A", "B"])].sort_values("frame")
    res = arc_resets.first_restrike_after(resets, rs)
    v = res.dropna(subset=["first_restrike_mode"])
    if not len(v):
        return np.nan, 0
    return float((v["first_restrike_mode"].astype(str) == "A").mean()), len(v)


def collect():
    rows = []
    for n in NS:
        # POOLED transition + geometry
        mt = DATA_ROOT.parent / "analysis" / "mode_transitions" / f"mode_transition_summary_npoint{sfx(n)}.csv"
        gm = DATA_ROOT / "analysis" / "geometry_predictor" / f"geometry_predictor_multivar_npoint{sfx(n)}.csv"
        pbb = pb = gp = cvauc = np.nan
        if mt.exists():
            s = pd.read_csv(mt)
            pj = s[s["label"].astype(str).str.startswith("POOLED")]
            if len(pj):
                r = pj.iloc[0]; pbb, pb, gp = r["P_B_given_B"], r["p_B"], r["g_p"]
        if gm.exists():
            g = pd.read_csv(gm); pg = g[g["file"] == "POOLED"]
            if len(pg):
                cvauc = float(pg["cv_auc"].iloc[0])
        pA, npost = reset_pA(BIG, n)
        # per-recording frac_B + count
        per = {}
        for stem in RECORDINGS:
            rp = restrikes_path(stem, n)
            if rp.exists():
                d = pd.read_csv(rp); d = d[d["mode"].isin(["A", "B"])]
                per[stem] = (len(d), float((d["mode"] == "B").mean()))
        rows.append(dict(N=n, P_BB=pbb, p_B=pb, g_p=gp, cv_auc=cvauc,
                         reset_pA=pA, reset_n=npost, per=per,
                         present=(mt.exists() or any(restrikes_path(s, n).exists() for s in RECORDINGS))))
    return [r for r in rows if r["present"]]


def main():
    rows = collect()
    present_N = [r["N"] for r in rows]
    print("N present:", present_N)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8.4))

    # 1) frac_B vs Re, one line per N
    a = ax[0, 0]
    for r in rows:
        re = [RECORDINGS[s]["re"] for s in r["per"]]
        fb = [r["per"][s][1] for s in r["per"]]
        order = np.argsort(re)
        a.plot(np.array(re)[order], np.array(fb)[order], "o-", color=NCOL[r["N"]],
               label=f"N={r['N']}", ms=7, mec="white")
    a.set_xlabel("Reynolds number"); a.set_ylabel("Mode-B fraction (valley cut)")
    a.set_ylim(0, 1); a.set_title("Mode-B fraction vs Re"); a.legend(frameon=False)

    # 2) re-strike count vs Re, per N (log)
    a = ax[0, 1]
    for r in rows:
        re = [RECORDINGS[s]["re"] for s in r["per"]]
        nn = [r["per"][s][0] for s in r["per"]]
        order = np.argsort(re)
        a.plot(np.array(re)[order], np.array(nn)[order], "s-", color=NCOL[r["N"]],
               label=f"N={r['N']}", ms=7, mec="white")
    a.set_yscale("log"); a.set_xlabel("Reynolds number")
    a.set_ylabel("re-strike count (log)"); a.set_title("Detector event count vs Re")
    a.legend(frameon=False)

    # 3) pooled transition / persistence vs N
    a = ax[0, 2]
    xs = [r["N"] for r in rows]
    a.plot(xs, [r["P_BB"] for r in rows], "o-", color="#333", label="P(B|B)")
    a.plot(xs, [r["p_B"] for r in rows], "o--", color="#999", label="overall p_B")
    a.set_xlabel("N (segments)"); a.set_ylabel("probability"); a.set_ylim(0, 1)
    a.set_title("Mode-B persistence vs N (pooled)"); a.set_xticks(xs); a.legend(frameon=False)

    # 4) geometry CV-AUC vs N
    a = ax[1, 0]
    a.plot(xs, [r["cv_auc"] for r in rows], "o-", color="#7a2fb0", ms=8)
    a.axhline(0.5, color="grey", ls=":")
    a.set_xlabel("N (segments)"); a.set_ylabel("geometry->mode CV-AUC (pooled)")
    a.set_ylim(0.4, 1.0); a.set_title("Geometry predictive power vs N"); a.set_xticks(xs)

    # 5) reset -> Mode A vs N (1.2mm)
    a = ax[1, 1]
    a.plot(xs, [r["reset_pA"] for r in rows], "o-", color="#d6604d", ms=8)
    a.axhline(1.0, color="grey", ls=":")
    for r in rows:
        a.annotate(f"n={r['reset_n']}", (r["N"], r["reset_pA"]),
                   textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
    a.set_xlabel("N (segments)"); a.set_ylabel("post-reset Mode-A fraction (1.2mm)")
    a.set_ylim(0, 1.05); a.set_title("Reset -> Mode A vs N"); a.set_xticks(xs)

    # 6) segment speed profile (1.2mm) overlaid for each N
    a = ax[1, 2]
    for r in rows:
        p = DATA_ROOT / f"{BIG}_results" / f"segment_velocities{sfx(r['N'])}.csv"
        if not p.exists():
            continue
        s = pd.read_csv(p, usecols=["s", "seg_speed_ms"])
        g = s.groupby("s")["seg_speed_ms"].mean()
        a.plot(g.index.values, g.values, "-", color=NCOL[r["N"]], lw=2, label=f"N={r['N']}")
    a.set_xlabel("position along arc s (0,1=feet; 0.5=tip)")
    a.set_ylabel("mean local speed (m/s)")
    a.set_title("Segment speed profile vs N (1.2mm)"); a.set_xlim(0, 1); a.legend(frameon=False)

    fig.suptitle("N-segmentation sensitivity: whole-arc characterizations at N = "
                 + ", ".join(str(n) for n in present_N), fontsize=14, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "fig_Nsegment_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig_Nsegment_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    summ = pd.DataFrame([{k: v for k, v in r.items() if k != "per"} for r in rows])
    summ.to_csv(OUT / "nseg_comparison_summary.csv", index=False)
    print(summ.to_string(index=False))
    print(f"\nwrote {OUT/'fig_Nsegment_comparison.png'}")


if __name__ == "__main__":
    main()
