"""Direct leading-tip re-strike detector: every single-step tip fall that pixel/mask noise cannot explain at a stated
false-detection bar (author's choice, 2026-10-06: option 1, per gas, 20 %).

Why a new detector. tip_restrike_threshold.py only FILTERS the candidates qa_restrike.py proposes, and qa_restrike
proposes a re-strike only between two extension peaks of prominence >= max(0.10 * l_eff, 1 px) on a 3-frame
smoothed tip (l_eff = p90 tip height above the gap). That floor is ~0.1 x the tip range (up to ~1.4 mm in high-Re
Argon), so lowering the filter threshold cannot reach drops smaller than it. Here the candidates are the raw steps.

Observable. The tip series exactly as qa_restrike.py sees it (tip_restrike_check.load_tip): one entry per complete,
non-flash frame; a STEP is the change between successive entries (it may span a break with no complete arc).

Threshold. Per gas, T_gas = the smallest grid T at which the false-detection estimate rises(>= T)/falls(>= T)
(tip_threshold_curve.csv, the same steps) is <= BAR there AND at every larger grid T. Assumption, UNVERIFIED as
before: noise moves the tip up as often as down.

Event. A step i with fall >= T_gas is a re-strike at frame fr[i+1], unless one of the next SPIKE_N steps is a RISE
>= T_gas (the fall was undone: a spike). Successive qualifying falls each count (each is a new, lower arc).

Grading (both sets labelled by the author by eye, blind to this detector):
  windows  restrike_windows clicks (detector-independent streak images): fraction of clicks matched one-to-one within
           MATCH_FRAMES by a kept direct event, beside the same number for the current 1.8 mm kept set;
  audit    restrike_audit strips (events of the OLD detector, judged real/not): does a kept direct event fall in the
           strip's fall window (peak_frame, frame_min]?

usage: python tip_restrike_direct.py <analysis_root> <out_dir> [bar]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tip_restrike_check import load_tip  # noqa: E402
from restrike_windows import match, MATCH_FRAMES  # noqa: E402

SPIKE_N = 3
MODE_SPLIT_MM = 2.2   # the valley of the re-strike height (tip_modes.py); below = Mode A (at the gap)


def gas_thresholds(curve: pd.DataFrame, bar: float) -> dict:
    out = {}
    for g, d in curve.groupby("gas"):
        d = d.sort_values("T_mm")
        ok = (d["fdr"] <= bar).to_numpy()
        idx = next((i for i in range(len(ok)) if ok[i:].all()), None)
        out[g] = float(d["T_mm"].iloc[idx]) if idx is not None else np.nan
    return out


def one_run(rd: Path, Tg: dict):
    meta, pxmm, fr, tip, erow, _ = load_tip(rd)
    T_mm = Tg[meta["gas"]]
    step = np.diff(tip)
    T = T_mm * pxmm
    rows = []
    for i in np.flatnonzero(-step >= T - 1e-9):
        nxt = step[i + 1:i + 1 + SPIKE_N]
        spike = bool((nxt >= T - 1e-9).any())
        rows.append(dict(run=rd.name, gas=meta["gas"], Re=meta["Re"], trial=meta["trial"],
                         frame_before=int(fr[i]), frame=int(fr[i + 1]), step_gap_frames=int(fr[i + 1] - fr[i]),
                         fall_mm=-step[i] / pxmm, tip_before_mm=(tip[i] - erow) / pxmm,
                         tip_after_mm=(tip[i + 1] - erow) / pxmm, spike=spike, kept=not spike))
    ev = pd.DataFrame(rows)
    n_frames = int(meta.get("n_frames", 0) or 0)
    return meta, ev, n_frames


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    bar = float(sys.argv[3]) if len(sys.argv) > 3 else 0.20
    out.mkdir(parents=True, exist_ok=True)
    thr_dir = root / "tip_restrike_check" / "threshold"
    Tg = gas_thresholds(pd.read_csv(thr_dir / "tip_threshold_curve.csv"), bar)
    print(f"bar {bar:.0%}: T_gas = " + ", ".join(f"{g} {t:.1f} mm" for g, t in sorted(Tg.items())))
    evs, runs = [], []
    for rd in sorted(p.parent for p in (root / "runs_annotated").rglob("re_strikes_qa.csv")):
        meta, ev, _ = one_run(rd, Tg)
        evs.append(ev)
        k = ev[ev["kept"]] if len(ev) else ev
        runs.append(dict(run=rd.name, gas=meta["gas"], Re=meta["Re"], trial=meta["trial"], T_mm=Tg[meta["gas"]],
                         n_candidates=len(ev), n_spikes=int(ev["spike"].sum()) if len(ev) else 0, n_kept=len(k),
                         n_modeA=int((k["tip_after_mm"] < MODE_SPLIT_MM).sum()) if len(k) else 0,
                         mean_interval_ms=float(np.diff(np.sort(k["frame"].to_numpy())).mean() / 200.0)
                         if len(k) >= 2 else np.nan))
    E = pd.concat(evs, ignore_index=True)
    R = pd.DataFrame(runs)
    E.to_csv(out / "tip_restrike_direct_events.csv", index=False)
    R.to_csv(out / "tip_restrike_direct_by_run.csv", index=False)
    n_old = int(pd.read_csv(thr_dir / "tip_restrike_events_kept.csv")["kept"].sum())
    print(f"runs {len(R)}; candidates {len(E)}, spikes {int(E['spike'].sum())}, kept {int(E['kept'].sum())} "
          f"(current 1.8 mm rule kept: {n_old})")
    print(R.groupby("gas")[["n_candidates", "n_spikes", "n_kept", "n_modeA"]].sum().to_string())

    # --- grading 1: completeness windows (detector-independent clicks)
    W = pd.read_csv(root / "restrike_windows" / "windows_key.csv")
    Cl = pd.read_csv(root / "restrike_windows" / "restrike_window_clicks.csv")
    oldE = pd.read_csv(thr_dir / "tip_restrike_events_kept.csv")
    gr = []
    for _, w in W.iterrows():
        c = Cl.loc[Cl["code"] == w["code"], "x_px"].to_numpy(int)
        if len(c) < 5:
            continue
        cfr = w["i0"] + c * w["bin"] + w["bin"] // 2
        inw = lambda f: (f >= w["i0"]) & (f < w["i0"] + w["n"])  # noqa: E731
        dnew = E[(E["run"] == w["run"]) & E["kept"] & inw(E["frame"])]["frame"].to_numpy()
        dold = oldE[(oldE["run"] == w["run"]) & oldE["kept"] & inw(oldE["frame_min"])]["frame_min"].to_numpy()
        mn, un_new = match(cfr, dnew)
        mo, un_old = match(cfr, dold)
        gr.append(dict(code=w["code"], gas=w["gas"], Re=w["Re"], clicks=len(c),
                       found_new=int(mn.sum()), found_old=int(mo.sum()),
                       detected_new=len(dnew), unmatched_new=un_new, detected_old=len(dold), unmatched_old=un_old))
    G = pd.DataFrame(gr)
    G.to_csv(out / "grade_windows.csv", index=False)
    print(f"\nCompleteness windows (clicks matched within {MATCH_FRAMES} frames):\n" + G.to_string(index=False))
    print(f"pooled: new {G['found_new'].sum()}/{G['clicks'].sum()}, old {G['found_old'].sum()}/{G['clicks'].sum()}; "
          f"detections not matched to a click: new {G['unmatched_new'].sum()}/{G['detected_new'].sum()}, "
          f"old {G['unmatched_old'].sum()}/{G['detected_old'].sum()}")

    # --- grading 2: blind audit strips
    A = pd.read_csv(root / "restrike_audit" / "graded.csv")
    hit = []
    for _, a in A.iterrows():
        d = E[(E["run"] == a["run"]) & E["kept"]]
        hit.append(bool(((d["frame"] > a["peak_frame"]) & (d["frame"] <= a["frame_min"])).any()))
    A["direct_hit"] = hit
    A.to_csv(out / "grade_audit.csv", index=False)
    print("\nBlind audit strips (q1 = author's verdict; cls = old detector class):")
    print(A.groupby(["cls", "q1"])["direct_hit"].agg(["size", "sum"]).rename(
        columns={"size": "strips", "sum": "direct_kept_in_window"}).to_string())


if __name__ == "__main__":
    main()
