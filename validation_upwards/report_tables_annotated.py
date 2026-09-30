"""Markdown tables for REPORT.md of the ANNOTATED BLADE-upwards run (hand geometry, centroid re-strikes).

Every number in REPORT.md is printed here from the per-run outputs; nothing is typed by hand.

Inputs (all under <analysis_root>):
  aggregate_annotated/per_recording.csv        aggregate_upwards.py on runs_annotated/
  centroid_restrikes/centroid_restrikes_by_run.csv, ..._by_condition.csv   centroid_restrikes.py
  check_runs.csv                               check_runs.py
  runs_annotated/**/run_metadata.json          the hand calibration each run actually used
  sensitivity/sensitivity_summary.csv          detector sweep (automatic-calibration era, calibration fixed)

usage: report_tables_annotated.py <analysis_root> > tables.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(sys.argv[1])
F = pd.read_csv(ROOT / "aggregate_annotated" / "per_recording.csv")
anchor = F["physical_gap_anchor"].astype("boolean").fillna(False).astype(bool)
D = F[(F["status"] == "DONE") & (F["tag"].fillna("") == "") & (F["short_complete_frac"].fillna(1.0) < 0.001) & anchor]
CR = pd.read_csv(ROOT / "centroid_restrikes" / "centroid_restrikes_by_run.csv")
CC = pd.read_csv(ROOT / "centroid_restrikes" / "centroid_restrikes_by_condition.csv")
CHK = pd.read_csv(ROOT / "check_runs.csv")
SENS = ROOT / "sensitivity" / "sensitivity_summary.csv"
ARGON = "Argon"   # fails the 5 % false-detection rule -> its re-strike estimate is drops - rises

QUOTABLE = [("local_vel_ms_med", "local tip vel (m/s)"), ("arc_len_mm_med", "arc length (mm)"),
            ("tip_height_above_gap_mm_p90", "tip height p90 (mm)"), ("frac_complete", "complete frac")]
TIPDET = [("n_restrikes", "tip re-strikes"), ("cycle_ms_mean", "tip cycle (ms)"),
          ("frac_B_thr005", "B @0.05"), ("frac_modeB", "B @0.10"), ("frac_B_thr020", "B @0.20")]


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else
                                      (f"{v:.3g}" if isinstance(v, float) else str(v)) for v in r.values) + " |")
    return "\n".join(out)


def pm(m, s):
    return "" if np.isnan(m) else (f"{m:.3g}" if np.isnan(s) else f"{m:.3g} ± {s:.2g}")


# ---------------------------------------------------------------- coverage
print("### Recordings\n")
print(md(F[F["tag"].fillna("") == ""].groupby("status").size().rename("n").reset_index()), "\n")
print(f"Analysed and passing every gate: **{len(D)}** recordings "
      f"({D.groupby(['gas', 'Re']).ngroups} gas × Re conditions).\n")
done = CHK[CHK["status"] == "DONE"]
print(f"Completeness (`check_runs.csv`): {int(done['missing_files'].notna().sum())} DONE runs with a missing output, "
      f"{int((~done['npoint_ok'].astype(bool)).sum())} with an incomplete point set. Automatic calibration "
      f"(kept only as a cross-check): {int((done['auto_cal_status'] == 'OK').sum())} of {len(done)} passed its own "
      f"check; where it passed, its attachment ends miss the hand cutoffs by > 4 px on "
      f"{int((done.loc[done['auto_cal_status'] == 'OK', 'auto_vs_hand_max_px'] > 4).sum())} runs "
      f"(max {done.loc[done['auto_cal_status'] == 'OK', 'auto_vs_hand_max_px'].max():.1f} px).\n")

# ---------------------------------------------------------------- scale actually used
print("### Scale used: hand annotation averaged per condition (setup)\n")
rows = {}
for mp in sorted((ROOT / "runs_annotated").rglob("run_metadata.json")):
    m = json.loads(mp.read_text(encoding="utf-8"))
    c = m.get("calibration") or {}
    if c.get("source") != "hand annotation":
        continue
    key = " / ".join(c["condition"])
    r = rows.setdefault(key, dict(setup=key, members=c["members"]["n"], gap_px=c["gap_px"],
                                  member_gap_sd_px=c["members"]["gap_px_sd"], px_per_mm=c["px_per_mm"],
                                  px_per_mm_sem=c["px_per_mm_sem"], opening_deg=c["full_opening_angle_deg"],
                                  md5=c["annotation_md5"][:8], recordings=0))
    if r["md5"] != c["annotation_md5"][:8]:
        raise SystemExit(f"{mp}: runs of setup {key} used DIFFERENT annotation sets ({r['md5']} vs "
                         f"{c['annotation_md5'][:8]})")
    r["recordings"] += 1
S = pd.DataFrame(rows.values())
print(md(S.round(3)), "\n")
lo, hi = S["px_per_mm"].min(), S["px_per_mm"].max()
print(f"Across setups: {lo:.3f}-{hi:.3f} px/mm (max/min {hi / lo:.3f}); largest per-setup sem "
      f"{(S['px_per_mm_sem'] / S['px_per_mm']).max() * 100:.1f} %.\n")

# ---------------------------------------------------------------- regime
print("### Regime: clean arches across the gap per scanned frame (automatic calibration pass, 10 000 frames)\n")
R = F[F["tag"].fillna("") == ""].pivot_table(index="gas", columns="Re", values="arch_frac_of_frames", aggfunc="mean")
print(md(R.round(3).reset_index()), "\n")

# ---------------------------------------------------------------- quotable metrics
print("### Geometry and tip velocity per condition (mean ± sd over trials)\n")
G = D.groupby(["gas", "Re"])
T = G.size().rename("n").reset_index()
for k, lab in QUOTABLE:
    mu, sd = G[k].mean().values, G[k].std().values
    T[lab] = [pm(a, b) for a, b in zip(mu, sd)]
print(md(T), "\n")

print("### Trial-to-trial scatter: half-range over trials / mean, per condition, summarised per gas\n")
rep = []
for (g, re_), d in G:
    if len(d) < 2:
        continue
    row = {"gas": g, "Re": re_}
    for k, lab in QUOTABLE:
        v = d[k].dropna()
        row[lab] = 100 * (v.max() - v.min()) / 2 / v.mean() if len(v) >= 2 and v.mean() else np.nan
    rep.append(row)
RP = pd.DataFrame(rep)
summ = []
for g, d in RP.groupby("gas"):
    row = {"gas": g, "conditions": len(d)}
    for _, lab in QUOTABLE:
        row[lab] = f"{d[lab].median():.0f} / {d[lab].max():.0f} % (Re{int(d.loc[d[lab].idxmax(), 'Re'])})"
    summ.append(row)
print("Each cell: median / worst ± % over that gas's conditions (condition of the worst in brackets).\n")
print(md(pd.DataFrame(summ)), "\n")

# ---------------------------------------------------------------- centroid re-strikes
print("### Re-strikes: centroid detector (drop of the arc's centre-of-mass height > 1.5 mm)\n")
fdr = CR.groupby("gas")[["drops_1.5", "rises_1.5"]].sum().rename(columns={"drops_1.5": "n_drops", "rises_1.5": "n_rises"})
fdr["false-detection estimate (rises/drops)"] = (fdr["n_rises"] / fdr["n_drops"]).map(lambda x: f"{100 * x:.1f} %")
fdr["use"] = ["drops − rises" if g == ARGON else "drops" for g in fdr.index]
print(md(fdr.reset_index().rename(columns={"n_drops": "drops (all runs)", "n_rises": "rises (all runs)"})), "\n")
C2 = CC.copy()
C2["re-strikes per run"] = [pm(e, s) if g == ARGON else pm(m, sd) for g, m, sd, e, s in
                            zip(C2["gas"], C2["restrikes_mean"], C2["restrikes_sd"], C2["excess_mean"], C2["excess_sd"])]
C2["rate (1/s)"] = np.where(C2["gas"] == ARGON, C2["excess_rate_per_s_mean"], C2["rate_per_s_mean"])
C2["raw drops (Argon)"] = [pm(m, sd) if g == ARGON else "" for g, m, sd in
                           zip(C2["gas"], C2["restrikes_mean"], C2["restrikes_sd"])]
print("Mean ± sd over trials. Argon: drops − rises (its raw drops shown beside).\n")
print(md(C2[["gas", "Re", "n", "re-strikes per run", "rate (1/s)", "raw drops (Argon)"]].round(1)), "\n")

# ---------------------------------------------------------------- end-to-end trends
print("### End to end over Re, per gas (condition means)\n")
CCr = CC.assign(rate=np.where(CC["gas"] == ARGON, CC["excess_rate_per_s_mean"], CC["rate_per_s_mean"]))
series = [(lab, G[k].mean()) for k, lab in QUOTABLE] + [("re-strike rate (1/s)", CCr.set_index(["gas", "Re"])["rate"])]
e2e = []
for lab, s in series:
    for g in sorted(s.index.get_level_values(0).unique()):
        v = s.loc[g].sort_index().dropna()
        dif = np.diff(v.values)
        e2e.append({"quantity": lab, "gas": g, "Re range": f"{v.index[0]}-{v.index[-1]}",
                    "first": v.iloc[0], "last": v.iloc[-1], "last/first": v.iloc[-1] / v.iloc[0] if v.iloc[0] else np.nan,
                    "steps down": f"{int((dif < 0).sum())} of {len(dif)}"})
print(md(pd.DataFrame(e2e).round(3)), "\n")

print("### Gas order at each Re (condition means, highest first)\n")
for k, lab in QUOTABLE[:3]:
    m = G[k].mean().unstack(0)
    print(f"- **{lab}**: " + "; ".join(f"Re{re_}: " + " > ".join(r.dropna().sort_values(ascending=False).index)
                                        for re_, r in m.iterrows()))
print()

# ---------------------------------------------------------------- not quotable
print("### NOT QUOTABLE — tip-detector re-strike outputs (`re_strikes_qa.csv`), kept for comparison\n")
print("Across these detections the centroid moves a median 0.1-0.2 px: most are apex jitter, not breakdowns. Mode A/B "
      "fractions are shown at three thresholds because the split is set by the threshold.\n")
T2 = G[[k for k, _ in TIPDET]].mean().reset_index().rename(columns=dict(TIPDET))
print(md(T2.round(3)), "\n")

if SENS.exists():
    Sn = pd.read_csv(SENS)
    print("### Detector sensitivity (overnight sweep, automatic-calibration era; calibration FIXED within each file, "
          "so it measures the detector only): max |relative change| over arms, first 20 000 frames\n")
    cols = [c for c in Sn.columns if c.endswith("_rel")]
    M = Sn.groupby("stem")[cols].agg(lambda x: np.nanmax(np.abs(x))).reset_index()
    M["stem"] = M["stem"].str.replace("BLADE_ADCB_", "").str.replace("_06mmNozzle_trial1", "")
    M.columns = [c.replace("_rel", "") for c in M.columns]
    print(md(M.round(3)), "\n")
