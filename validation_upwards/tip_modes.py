"""Mode A / Mode B for the KEPT leading-tip re-strikes (fall >= T_TIP), split at the VALLEY of the re-strike height.

Observable. Re-strike height h = tip height above the gap at the re-strike (tip_after_mm, i.e. qa_restrike's marked
minimum), in mm with the run's hand scale. Kept events only (tip_restrike_threshold.py).

Why not norm_pos > 0.10. norm_pos divides h by L_eff = p90(tip) - gap, which grows ~2.5x with Re. At low Re that
puts the 0.10 cut at ~0.4-0.5 mm, INSIDE the near-gap cluster (h = 0.2-0.5 mm), so near-gap re-strikes are counted
as Mode B: CO2 has no kept re-strike above 1 mm at any Re, yet norm_pos gave it B = 0.39-0.76 at low Re.

Rule. The repo's documented Mode A/B boundary is the valley between the two peaks of the re-strike depth
distribution (qa_restrike.py docstring; valley_reclassify.find_valley, used unchanged, its own defaults). It is
computed on the POOLED kept events of all gases (one boundary for the campaign) and printed per gas beside it.
Robustness: B fractions are also given at cuts 1.0 and 2.0 mm, the edges of the near-empty band.

usage: python tip_modes.py <analysis_root>
Writes <analysis_root>/tip_restrike_check/threshold/tip_modes_by_run.csv, tip_modes_by_condition.csv, tip_modes.txt
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from valley_reclassify import find_valley  # noqa: E402

CUTS_MM = (1.0, 2.0)


def main():
    root = Path(sys.argv[1])
    thr = root / "tip_restrike_check" / "threshold"
    E = pd.read_csv(thr / "tip_restrike_events_kept.csv")
    F = pd.read_csv(root / "aggregate_annotated" / "per_recording.csv").set_index("stem")
    n0 = len(E)
    E = E[E["kept"]].copy()
    E["gas"], E["Re"], E["trial"] = E["run"].map(F["gas"]), E["run"].map(F["Re"]), E["run"].map(F["trial"])
    if E[["gas", "Re"]].isna().any().any():
        raise SystemExit("events whose run is missing from per_recording.csv")
    out = [f"events: {n0} tip re-strikes, {len(E)} kept (fall >= T_TIP)"]
    v = find_valley(E["tip_after_mm"].to_numpy(float))
    if v is None:
        raise SystemExit("no valley in the pooled re-strike height distribution")
    VAL = v["valley"]
    out.append(f"pooled valley {VAL:.2f} mm (peaks {v['peak1']:.2f} / {v['peak2']:.2f} mm)")
    for g, d in E.groupby("gas"):
        vg = find_valley(d["tip_after_mm"].to_numpy(float))
        out.append(f"  {g}: " + (f"valley {vg['valley']:.2f} mm (peaks {vg['peak1']:.2f} / {vg['peak2']:.2f})"
                                  if vg else "no second peak (unimodal)")
                   + f"; events above {VAL:.2f} mm: {int((d['tip_after_mm'] > VAL).sum())} of {len(d)}")
    band = E["tip_after_mm"].between(*CUTS_MM)
    out.append(f"events in the {CUTS_MM[0]}-{CUTS_MM[1]} mm band: {int(band.sum())} of {len(E)} "
               f"({100 * band.mean():.2f} %): the most any cut inside the band can reclassify")
    E["B"] = E["tip_after_mm"] > VAL
    per = []
    for run, d in E.groupby("run"):
        r = dict(run=run, gas=d["gas"].iloc[0], Re=int(d["Re"].iloc[0]), trial=int(d["trial"].iloc[0]),
                 n_kept=len(d), fracB=d["B"].mean(), h_A_med_mm=d.loc[~d["B"], "tip_after_mm"].median(),
                 h_B_med_mm=d.loc[d["B"], "tip_after_mm"].median())
        for c in CUTS_MM:
            r[f"fracB_cut{c:g}"] = (d["tip_after_mm"] > c).mean()
        per.append(r)
    R = pd.DataFrame(per)
    R.to_csv(thr / "tip_modes_by_run.csv", index=False)
    cols = ["fracB", "fracB_cut1", "fracB_cut2", "h_A_med_mm", "h_B_med_mm"]
    K = R.groupby(["gas", "Re"])[cols].agg(["mean", "std"])
    K.columns = [f"{a}_{b}" for a, b in K.columns]
    K = K.reset_index()
    K.to_csv(thr / "tip_modes_by_condition.csv", index=False)
    dmax = (R["fracB_cut2"] - R["fracB_cut1"]).abs().max()
    out.append(f"largest per-run change in B fraction between the {CUTS_MM[0]} and {CUTS_MM[1]} mm cuts: {dmax:.3f}")
    out.append(f"valley value used: {VAL:.2f} mm")
    txt = "\n".join(out)
    print(txt)
    print("\n" + K[["gas", "Re", "fracB_mean", "fracB_std", "fracB_cut1_mean", "fracB_cut2_mean",
                    "h_B_med_mm_mean"]].round(3).to_string(index=False))
    (thr / "tip_modes.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
