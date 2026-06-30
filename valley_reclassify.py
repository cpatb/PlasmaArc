#!/usr/bin/env python3
"""
valley_reclassify.py  —  re-classify Mode A/B at the DEAD-ZONE (valley) between the
two peaks of the re-strike initiation-depth distribution, instead of the current
norm_pos>0.10 cut.

Physical rationale (from the initiation-depth histograms): the distance-below-apex
distribution is BIMODAL — a near-apex first peak (a full reset from the apex gap)
and a deeper second peak (genuine back-breakdown), separated by a sparse "dead
zone" around ~4 mm. The current 0.10*L_eff cut (~1.5 mm) slices THROUGH the first
peak, so the upper half of that near-apex reset population is mislabelled Mode B.
This script puts the cut at the valley: everything in the first peak -> Mode A, the
rest -> Mode B, and reports how the headline numbers change.

Per recording it:
  * computes depth_mm = norm_pos * l_eff_px / px_per_mm  (distance below apex),
  * auto-detects the valley between the first (<=3 mm) and second peaks
    (or uses --cut-mm), with a tight +/- allowance band for sensitivity,
  * re-labels A = depth < valley, B = depth >= valley,
  * recomputes (reset-aware contiguity, reusing arc_resets + mode_transition_analysis):
      frac_B, P(A|A)/P(B|A)/P(A|B)/P(B|B), excess persistence, memoryless G-test,
      mean Mode-B run length (obs vs i.i.d.),
  * prints CURRENT vs VALLEY side by side and writes a CSV + markdown report.

Does NOT modify re_strikes_qa.csv or any authoritative output.

    python valley_reclassify.py            # all recordings, auto valley
    python valley_reclassify.py --cut-mm 4.0
"""
from __future__ import annotations
import argparse, os, json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import arc_resets
from mode_transition_analysis import (transition_counts, chain_stats,
                                      run_lengths, lag1_autocorr)
from npoint_restrike import RECORDINGS

DATA_ROOT_DEFAULT = Path(r"C:\Users\compl\Desktop\Plasma Arc\J Electrode")


def load_depth(rd: Path):
    q = pd.read_csv(rd / "re_strikes_qa.csv")
    q = q[q["mode"].isin(["A", "B"])].sort_values("frame").reset_index(drop=True)
    meta = json.loads((rd / "run_metadata.json").read_text(encoding="utf-8", errors="replace")
                      .rsplit("}", 1)[0] + "}")
    ppm = float(meta["px_per_mm"])
    depth = (pd.to_numeric(q["norm_pos"], errors="coerce")
             * pd.to_numeric(q["l_eff_px"], errors="coerce") / ppm).to_numpy()
    return q, depth, ppm


def find_valley(depth, binw=0.4, first_max_mm=3.0, sep_mm=1.5):
    """Valley (mm) between the near-apex first peak and the deeper second peak."""
    d = depth[np.isfinite(depth)]
    if len(d) < 20:
        return None
    bins = np.arange(0, max(d.max(), 8.0) + binw, binw)
    h, _ = np.histogram(d, bins=bins)
    ctr = 0.5 * (bins[:-1] + bins[1:])
    hs = gaussian_filter1d(h.astype(float), 1.5)
    m1 = ctr <= first_max_mm
    if not m1.any() or hs[m1].max() == 0:
        return None
    p1 = ctr[m1][np.argmax(hs[m1])]
    m2 = ctr > p1 + sep_mm
    if not m2.any() or hs[m2].max() == 0:
        return None
    p2 = ctr[m2][np.argmax(hs[m2])]
    mv = (ctr >= p1) & (ctr <= p2)
    if mv.sum() < 2:
        return None
    valley = float(ctr[mv][np.argmin(hs[mv])])
    return dict(valley=valley, peak1=float(p1), peak2=float(p2))


def contiguity(rd: Path, q: pd.DataFrame, reset_gap=arc_resets.DEFAULT_RESET_GAP):
    """Reset-aware contiguity flags for the re-strike sequence (matches
    mode_transition_analysis): pair (k-1,k) valid iff ordered AND no reset between."""
    frames = pd.to_numeric(q["frame"], errors="coerce").to_numpy()
    resets = None
    mp = rd / "measurements_all_events.csv"
    if mp.exists():
        try:
            meas = pd.read_csv(mp, usecols=["frame", "arc_detected", "is_flash"])
            resets = arc_resets.reset_intervals(meas, reset_gap=reset_gap)
        except Exception:
            resets = None
    cont = [False] * len(q)
    for k in range(1, len(q)):
        c = bool(frames[k] > frames[k - 1])
        if c and resets is not None and arc_resets.crosses_reset(
                int(frames[k - 1]), int(frames[k]), resets):
            c = False
        cont[k] = c
    return cont


def stats_for(modes, cont):
    C = transition_counts(modes, cont)
    s = chain_stats(C, modes)
    obs, exp, nruns = run_lengths(modes, cont, "B")
    s["meanrun_B_obs"] = obs
    s["meanrun_B_iid"] = exp
    s["ac1"] = lag1_autocorr(modes, cont)
    return s


def analyse(stem, data_root, cut_mm=None):
    rd = data_root / f"{stem}_results"
    q, depth, ppm = load_depth(rd)
    cont = contiguity(rd, q)
    cur = stats_for(q["mode"].tolist(), cont)

    v = find_valley(depth)
    if cut_mm is not None:
        valley = cut_mm
        vinfo = dict(valley=cut_mm, peak1=np.nan, peak2=np.nan)
    elif v is not None:
        valley = v["valley"]; vinfo = v
    else:
        valley = float("nan"); vinfo = dict(valley=np.nan, peak1=np.nan, peak2=np.nan)

    new_modes = np.where(depth < valley, "A", "B")
    new = stats_for(list(new_modes), cont)
    # how many re-strikes were re-labelled B->A by moving the cut out to the valley
    old = q["mode"].to_numpy()
    flipped_BtoA = int(((old == "B") & (new_modes == "A")).sum())
    info = RECORDINGS.get(stem, {})
    return dict(
        stem=stem, Re=info.get("re"), d_mm=info.get("d_mm"), is_repeat=int(info.get("is_repeat", 0)),
        n=len(q), ppm=ppm, peak1_mm=round(vinfo["peak1"], 2), peak2_mm=round(vinfo["peak2"], 2),
        valley_mm=round(valley, 2), valley_px=round(valley * ppm, 1),
        cur_fracB=round(cur["p_B"], 3), new_fracB=round(new["p_B"], 3),
        cur_PBB=round(cur["P_B_given_B"], 3), new_PBB=round(new["P_B_given_B"], 3),
        cur_PBA=round(cur["P_B_given_A"], 3), new_PBA=round(new["P_B_given_A"], 3),
        cur_excessB=round(cur["P_B_given_B"] - cur["p_B"], 3),
        new_excessB=round(new["P_B_given_B"] - new["p_B"], 3),
        cur_g_p=cur["g_p"], new_g_p=new["g_p"],
        cur_runB_obs=round(cur["meanrun_B_obs"], 2), new_runB_obs=round(new["meanrun_B_obs"], 2),
        new_runB_iid=round(new["meanrun_B_iid"], 2),
        flipped_B_to_A=flipped_BtoA, flipped_frac=round(flipped_BtoA / len(q), 3),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    ap.add_argument("--cut-mm", type=float, default=None,
                    help="fixed cut (mm below apex); default = auto-detected valley")
    args = ap.parse_args()
    data_root = Path(args.data_root)
    rows = [analyse(s, data_root, cut_mm=args.cut_mm) for s in RECORDINGS]
    df = pd.DataFrame(rows).sort_values(["is_repeat", "Re"])

    pd.set_option("display.width", 200)
    show = ["stem", "Re", "n", "peak1_mm", "peak2_mm", "valley_mm",
            "cur_fracB", "new_fracB", "flipped_B_to_A", "flipped_frac",
            "cur_PBB", "new_PBB", "cur_runB_obs", "new_runB_obs", "new_runB_iid",
            "cur_g_p", "new_g_p"]
    print(df[show].to_string(index=False))

    out = data_root / "analysis" / "geometry_predictor"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "valley_reclassify.csv", index=False)

    # markdown report
    L = ["# Valley-based Mode A/B reclassification\n",
         "Cut placed at the DEAD ZONE (valley) between the two peaks of the "
         "initiation-depth distribution: first (near-apex) peak -> Mode A, "
         "second (deep) peak -> Mode B. Compared to the current norm_pos>0.10 cut. "
         "Reset-aware contiguity throughout.\n",
         "| recording | Re | n | peak1/valley/peak2 (mm) | fracB cur->new | "
         "B->A re-labelled | P(B\\|B) cur->new | mean B-run cur->new (iid) | "
         "memoryless G-test p cur->new |",
         "|---|--:|--:|--|--|--:|--|--|--|"]
    for _, r in df.iterrows():
        L.append(f"| {r['stem'].replace('HS_','')} | {int(r['Re'])} | {int(r['n'])} | "
                 f"{r['peak1_mm']}/{r['valley_mm']}/{r['peak2_mm']} | "
                 f"{r['cur_fracB']}->{r['new_fracB']} | {int(r['flipped_B_to_A'])} "
                 f"({100*r['flipped_frac']:.0f}%) | {r['cur_PBB']}->{r['new_PBB']} | "
                 f"{r['cur_runB_obs']}->{r['new_runB_obs']} ({r['new_runB_iid']}) | "
                 f"{r['cur_g_p']:.2g}->{r['new_g_p']:.2g} |")
    nonrep = df[df["is_repeat"] == 0]
    L.append("\n## What changes (non-repeats)\n")
    L.append(f"- Mode-B fraction drops from a mean of {nonrep['cur_fracB'].mean():.2f} to "
             f"**{nonrep['new_fracB'].mean():.2f}** once the near-apex first peak is all "
             f"assigned to Mode A ({nonrep['flipped_B_to_A'].sum()} re-strikes re-labelled "
             f"B->A across non-repeats).")
    L.append(f"- Valley sits at {nonrep['valley_mm'].min():.1f}-{nonrep['valley_mm'].max():.1f} mm "
             f"below the apex ({nonrep['valley_px'].min():.0f}-{nonrep['valley_px'].max():.0f} px), "
             f"i.e. well past the current ~1.5 mm cut.")
    memo_cur = (pd.to_numeric(nonrep['cur_g_p'], errors='coerce') > 0.05).mean()
    memo_new = (pd.to_numeric(nonrep['new_g_p'], errors='coerce') > 0.05).mean()
    L.append(f"- Memoryless (G-test p>0.05) held for {memo_cur:.0%} of non-repeats before, "
             f"{memo_new:.0%} after.")
    L.append(f"- P(B|B) cur->new: see table; excess persistence of B becomes "
             f"{nonrep['new_excessB'].mean():+.3f} on average.")
    (out / "valley_reclassify_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nWrote {out/'valley_reclassify.csv'} and valley_reclassify_report.md")
    return df


if __name__ == "__main__":
    main()
