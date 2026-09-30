"""Minimum-fall rule for the leading-tip re-strike detector, derived from the tip data (author's choice, 2026-09-30:
the same 5 % false-detection rule as the centroid detector).

Observable. The tip series exactly as qa_restrike.py sees it (tip_restrike_check.load_tip): one entry per complete,
non-flash frame; a STEP is the change between successive entries (it may span a break with no complete arc).

Rule. A breakdown makes the tip FALL in one step; nothing makes it RISE by millimetres in one step (gliding is
~0.03 px per frame). So single-step RISES >= T estimate the false single-step FALLS >= T, and the false-detection
estimate at T is rises(>= T) / falls(>= T), totals pooled per gas over all its runs. For each gas, T_gas = the
smallest T on the grid at which the estimate is <= FDR_MAX there AND at every larger grid T (a single dip below
the bar does not count). T_TIP = the largest T_gas over the gases that pass, i.e. the smallest common threshold.
A gas with no passing T is reported as failing; its count is then kept events - rises(>= T_TIP) per run, as
for the centroid rule. Assumption, UNVERIFIED: errors move the tip up as often as down. Rises across a break can
also be real (the arc kept growing while incomplete), which makes the estimate conservative (T higher). The
consecutive-frame-only estimate is printed beside it.

Applied. A tip re-strike (re_strikes_qa.csv) is KEPT if its largest single-step fall from the previous extension
peak (tip_restrike_check.py, max_step_mm) is >= T_TIP. Mode B fractions (norm_pos > 0.05 / 0.10 / 0.20) and cycle
times (between successive KEPT re-strikes) are recomputed on the kept events.

usage: python tip_restrike_threshold.py <runs_root> <tip_restrike_check_dir> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tip_restrike_check import load_tip  # noqa: E402

FDR_MAX = 0.05
GRID_MM = np.round(np.arange(0.1, 5.01, 0.1), 2)
MODE_THR = (0.05, 0.10, 0.20)


def fdr(rises, falls):
    """rises/falls; 0 when there is nothing at all (no fall and no rise), inf when rises exist but no falls."""
    r, f = np.asarray(rises, float), np.asarray(falls, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(f > 0, r / np.where(f > 0, f, 1), np.where(r > 0, np.inf, 0.0))


def smallest_T(d, col):
    """Smallest grid T with d[col] <= FDR_MAX there AND at every larger T; NaN if none."""
    d = d.sort_values("T_mm")
    ok = (d[col] <= FDR_MAX).to_numpy()
    tail_ok = np.flip(np.cumprod(np.flip(ok))).astype(bool)
    return float(d["T_mm"].to_numpy()[tail_ok][0]) if tail_ok.any() else np.nan


def main():
    root, chk, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    rows, runs = [], {}
    for rd in sorted(p.parent for p in root.rglob("re_strikes_qa.csv")):
        meta, pxmm, fr, tip, _, _ = load_tip(rd)
        st = np.diff(tip) / pxmm
        cons = np.diff(fr) == 1
        runs[rd.name] = dict(gas=meta["gas"], Re=meta["Re"], trial=meta["trial"], fps=float(meta["fps"]),
                             duration_s=int(meta["n_frames"]) / float(meta["fps"]))   # whole record, as centroid
        for T in GRID_MM:
            rows.append(dict(run=rd.name, gas=meta["gas"], T_mm=T,
                             falls=int((st <= -T).sum()), rises=int((st >= T).sum()),
                             falls_cons=int((st[cons] <= -T).sum()), rises_cons=int((st[cons] >= T).sum())))
    C = pd.DataFrame(rows)
    G = C.groupby(["gas", "T_mm"])[["falls", "rises", "falls_cons", "rises_cons"]].sum().reset_index()
    G["falls_brk"], G["rises_brk"] = G["falls"] - G["falls_cons"], G["rises"] - G["rises_cons"]
    for sfx in ("", "_cons", "_brk"):
        G["fdr" + sfx] = fdr(G["rises" + sfx], G["falls" + sfx])
    G.to_csv(out / "tip_threshold_curve.csv", index=False)

    tg = {g: smallest_T(d, "fdr") for g, d in G.groupby("gas")}
    passing = {g: t for g, t in tg.items() if np.isfinite(t)}
    if not passing:
        raise SystemExit("no gas meets the false-detection rule at any grid threshold")
    T_TIP = max(passing.values())
    failing = sorted(g for g, t in tg.items() if not np.isfinite(t))

    print("### False-detection estimate rises/falls (pooled per gas) at selected thresholds\n")
    show = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, T_TIP]
    P = G[G["T_mm"].isin(sorted(set(show)))].pivot(index="gas", columns="T_mm", values="fdr")
    print((100 * P).round(1).astype(str).add(" %").to_string(), "\n")
    print("Smallest threshold meeting <= 5 % there and at every larger T, per gas: "
          + ", ".join(f"{g} {t:.1f} mm" if np.isfinite(t) else f"{g} NONE" for g, t in tg.items()))
    print(f"**T_TIP = {T_TIP:.1f} mm** (smallest common threshold over {', '.join(sorted(passing))}); "
          f"failing: {', '.join(failing) if failing else 'none'}")
    at = G[np.isclose(G["T_mm"], T_TIP)].set_index("gas")
    print("\nAt T_TIP: " + "; ".join(f"{g} {100 * r.fdr:.1f} % (consecutive frames only {100 * r.fdr_cons:.1f} %, "
                                      f"{int(r.rises_cons)}/{int(r.falls_cons)}; across a break {100 * r.fdr_brk:.1f} %)"
                                      for g, r in at.iterrows()))
    sub = {g: (smallest_T(d, "fdr_cons"), smallest_T(d, "fdr_brk")) for g, d in G.groupby("gas")}
    bad = [g for g, r in at.iterrows() if not (r.fdr_cons <= FDR_MAX and r.fdr_brk <= FDR_MAX)]
    print("SUB-CHECK (reported, not applied; the registered rule is the pooled one): the 5 % bar applied to each "
          f"step type separately {'FAILS for ' + ', '.join(bad) if bad else 'passes for every gas'} at T_TIP. "
          "Smallest T per type (consecutive / across a break): "
          + ", ".join(f"{g} {c:.1f} / {b:.1f} mm" for g, (c, b) in sub.items()) + "\n")

    # ---- apply to the tip re-strikes
    E = pd.read_csv(chk / "tip_restrike_events.csv")
    Q = pd.concat(pd.read_csv(p, usecols=["file", "frame", "norm_pos"]).rename(columns={"file": "run"})
                  for p in root.rglob("re_strikes_qa.csv"))
    E = E.merge(Q, on=["run", "frame"], how="left", validate="one_to_one")
    if E["norm_pos"].isna().any():
        raise SystemExit(f"{int(E['norm_pos'].isna().sum())} events without a norm_pos")
    # kept: a single-step fall >= T_TIP, a net fall from the peak >= T_TIP, and no single-step rise >= T_TIP after
    # the fall (a one-step rise is what the rule counts as a false detection, so a fall undone by one is a spike)
    E["kept"] = (E["max_step_mm"] >= T_TIP) & (E["fall_mm"] >= T_TIP) & (E["rise_after_mm"] < T_TIP)
    print(f"events: {len(E)}; max step >= T_TIP: {int((E['max_step_mm'] >= T_TIP).sum())}; also net fall >= T_TIP: "
          f"{int(((E['max_step_mm'] >= T_TIP) & (E['fall_mm'] >= T_TIP)).sum())}; also no rebound: {int(E['kept'].sum())}")
    E.to_csv(out / "tip_restrike_events_kept.csv", index=False)
    rises_T = C[np.isclose(C["T_mm"], T_TIP)].set_index("run")["rises"]
    per = []
    for run, m in runs.items():                    # every run, including one with no event (rate 0, not NaN)
        d = E[E["run"] == run]
        k = d[d["kept"]].sort_values("frame_min")
        cyc = np.diff(k["frame_min"].to_numpy()) / m["fps"] * 1e3
        n = len(k) - (int(rises_T[run]) if m["gas"] in failing else 0)
        r = dict(run=run, gas=m["gas"], Re=m["Re"], trial=m["trial"], n_all=len(d), n_kept=len(k), n_count=n,
                 rate_per_s=n / m["duration_s"], cycle_ms_mean=cyc.mean() if len(cyc) else np.nan,
                 cycle_ms_med=np.median(cyc) if len(cyc) else np.nan)
        for t in MODE_THR:
            r[f"B@{t:.2f}_all"] = (d["norm_pos"] > t).mean()
            r[f"B@{t:.2f}_kept"] = (k["norm_pos"] > t).mean() if len(k) else np.nan
        per.append(r)
    R = pd.DataFrame(per)
    R.to_csv(out / "tip_restrikes_kept_by_run.csv", index=False)
    agg = {c: (c, "mean") for c in R.columns if c not in ("run", "gas", "Re", "trial")}
    K = R.groupby(["gas", "Re"]).agg(n=("run", "size"), n_count_sd=("n_count", "std"), **agg).reset_index()
    K.to_csv(out / "tip_restrikes_kept_by_condition.csv", index=False)
    print(f"Kept {int(E['kept'].sum())} of {len(E)} tip re-strikes ({100 * E['kept'].mean():.1f} %).")
    by = E.groupby(E["run"].map(lambda r: runs[r]["gas"]))
    print("Kept fraction per gas: " + ", ".join(f"{g} {100 * d['kept'].mean():.1f} %" for g, d in by) + "\n")
    cols = ["gas", "Re", "n", "n_all", "n_kept", "n_count", "n_count_sd", "rate_per_s", "cycle_ms_mean",
            "B@0.10_all", "B@0.05_kept", "B@0.10_kept", "B@0.20_kept"]
    print("### Per condition (mean over trials)\n")
    print(K[cols].round(3).to_string(index=False))
    Path(out / "T_TIP.txt").write_text(f"{T_TIP:.1f}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
