"""Mode A/B split of BLADE-upwards re-strikes by two rules, with position, glide velocity and discharge time
(author's request, 2026-11-23).

Post-processing only. Every input already exists and is documented in data_package/README.txt:
  data_package/restrikes.csv      re-strikes of the leading arc tip, 1.8 mm rule (blind audit: 79/80 real)
  data_package/recordings.csv     per recording px/mm, fps, and the published mean discharge time (anchor)
  data_package/frames/*.csv       per-frame tip height (used only for the Mode B observability share)
  tip_cycles/tip_cycles.csv       rule 'old1.8': one cycle per pair of successive counted re-strikes

Event. One counted re-strike. Its POSITION is the new arc's tip height above the minimum gap at the frame where the
tip is lowest (restrike_height_mm). The arc it creates glides until the next counted re-strike: that cycle gives
  glide_velocity_m_s   Theil-Sen slope of tip height vs time from the cycle's first frame to its highest tip
                       (vertical, along the blade bisector; blank if fewer than 5 rise frames; tip_cycles.py)
  discharge_time_ms    interval to the next counted re-strike (blank for the last event of a recording)

Rules, fixed before the script was run:
  FIXED  Mode B if restrike_height_mm >= 2.4 mm (one minimum-gap width above the gap; not fitted to the data).
  VALLEY Per gas x Re, trial-pooled heights. Each height is smeared uniformly over its pixel, x = log10(h + 1 px),
         Gaussian KDE with Silverman's factor for the number of EVENTS. The valley is the density minimum between
         the two highest local maxima. Accepted only if (a) valley/smaller-peak density <= 0.5, (b) >= 20 events on
         each side, and (c) (a)+(b) hold in >= 90 % of the exact trial bootstrap (all n^n ordered trial resamples).
         Amended after code review, before any result was used: in (c), rule (b) counts DISTINCT events of the
         resampled trials (a resample repeating one trial does not multiply its events); the h = 0 clip that made
         a single zero-height event into a KDE peak was removed.
         Otherwise the cell is 'no valley': no mode labels, no Mode B fraction, and NO substitute cut.
The same estimator run on gas-pooled heights is reported as a cross-check only.

What the numbers are (carried into every output):
  - The detector misses real re-strikes (0.47-0.90 of those seen by eye where the arc glides; 1/32 in the low-Re
    Ar-CO2 spot regime). Discharge times are UPPER BOUNDS; counts are LOWER BOUNDS; Mode B fraction is the share of
    DETECTED re-strikes, not of all re-strikes.
  - Mode B is only detectable where the old arc's tip reached >= cut + 1.8 mm; modeB_observable_share gives the
    share of tip frames where that holds. Where it is small, 'few Mode B' is a property of the detector.
  - Velocity is vertical only (the tip column is biased; analyze_blade.py:347-356).
  - Air Re are as labelled on the files (the x1.225 correction is not applied).

usage: python mode_split.py <analysis_root> <out_dir>
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import gaussian_kde

FIXED_CUT_MM = 2.4          # one minimum-gap width (2.4 mm gap)
DETECT_FALL_MM = 1.8        # the 1.8 mm re-strike rule's minimum fall (data_package/README.txt)
SWEEP_MM = [1.0, 1.5, 2.0, 2.4, 3.0]
RATIO_MAX = 0.5
MIN_SIDE = 20
BOOT_PASS = 0.9
N_SMEAR = 10                # sub-samples per event across its pixel
GRID_N = 2000

KEY = ["gas", "Re"]


def valley(h, px_mm, h_distinct=None):
    """Return dict(valley_mm, ratio, n_low, n_high, ok) for heights h (mm); px_mm = mm per pixel.
    h_distinct: the distinct events behind h (bootstrap resamples repeat trials); the >= MIN_SIDE rule counts these."""
    h = np.asarray(h, float)
    hd = h if h_distinct is None else np.asarray(h_distinct, float)
    n = len(h)
    out = dict(valley_mm=np.nan, ratio=np.nan, n_low=0, n_high=0, n_peaks=0, ok=False)
    if len(hd) < 2 * MIN_SIDE:
        return out
    # heights on the float origin can sit up to half a pixel below 0 (|round(gap_row) - gap_row| <= 0.5 px)
    assert h.min() > -0.5 * px_mm - 1e-9
    off = (np.arange(N_SMEAR) + 0.5) / N_SMEAR - 0.5      # within +/-0.45 px, so hs + px_mm > 0.05 px: no clip
    hs = (h[:, None] + off[None, :] * px_mm).ravel()
    x = np.log10(hs + px_mm)
    xe = np.log10(h + px_mm)
    # Silverman's rule for n EVENTS (not n * N_SMEAR samples): kernel sd = factor * sd(event x).
    # gaussian_kde's kernel sd is bw_method * sd(its data), so rescale by sd(event x) / sd(smeared x).
    factor = (n * 3.0 / 4.0) ** (-1.0 / 5.0)
    k = gaussian_kde(x, bw_method=factor * np.std(xe, ddof=1) / np.std(x, ddof=1))
    g = np.linspace(x.min() - 0.2, x.max() + 0.2, GRID_N)
    d = k(g)
    pk = argrelextrema(d, np.greater)[0]
    out["n_peaks"] = len(pk)
    if len(pk) < 2:
        return out
    top = np.sort(pk[np.argsort(d[pk])[-2:]])
    j = top[0] + int(np.argmin(d[top[0]:top[1] + 1]))
    v_mm = 10 ** g[j] - px_mm
    out.update(valley_mm=v_mm, ratio=d[j] / min(d[top[0]], d[top[1]]),
               n_low=int((hd < v_mm).sum()), n_high=int((hd >= v_mm).sum()))
    out["ok"] = bool(out["ratio"] <= RATIO_MAX and out["n_low"] >= MIN_SIDE and out["n_high"] >= MIN_SIDE)
    return out


def valley_with_bootstrap(cell, px_mm):
    trials = sorted(cell["trial"].unique())
    pt = valley(cell["restrike_height_mm"], px_mm)
    vs, passes = [], 0
    combos = list(itertools.product(trials, repeat=len(trials)))
    for c in combos:
        hb = np.concatenate([cell.loc[cell.trial == t, "restrike_height_mm"].to_numpy() for t in c])
        hd = cell.loc[cell.trial.isin(set(c)), "restrike_height_mm"].to_numpy()
        r = valley(hb, px_mm, h_distinct=hd)
        passes += r["ok"]
        if r["ok"]:
            vs.append(r["valley_mm"])
    pt.update(n_events=len(cell), n_trials=len(trials), boot_n=len(combos), boot_pass_share=passes / len(combos),
              boot_valley_min_mm=min(vs) if vs else np.nan, boot_valley_max_mm=max(vs) if vs else np.nan)
    pt["accepted"] = bool(pt["ok"] and pt["boot_pass_share"] >= BOOT_PASS)
    return pt


def run_table(E, obs):
    """Per recording summaries from labelled events E (mode in {'A','B'} or NaN)."""
    rows = []
    for (rec, gas, Re, trial), e in E.groupby(["recording", "gas", "Re", "trial"]):
        r = dict(recording=rec, gas=gas, Re=Re, trial=trial, cut_mm=e["cut_mm"].iloc[0],
                 n_restrikes_detected=len(e),
                 mean_discharge_time_ms_UPPER_BOUND=e["discharge_time_ms"].mean(),
                 median_glide_velocity_m_s=e["glide_velocity_m_s"].median(),
                 median_restrike_height_mm=e["restrike_height_mm"].median(),
                 modeB_observable_share=obs.get((rec, e["cut_mm"].iloc[0]), np.nan))
        lab = e["mode"].notna().all()
        for m in "AB":
            s = e[e["mode"] == m]
            r[f"n_mode{m}"] = len(s) if lab else np.nan
            r[f"mode{m}_mean_discharge_time_ms_UPPER_BOUND"] = s["discharge_time_ms"].mean() if lab else np.nan
            r[f"mode{m}_n_discharge_intervals"] = int(s["discharge_time_ms"].notna().sum()) if lab else np.nan
            r[f"mode{m}_median_glide_velocity_m_s"] = s["glide_velocity_m_s"].median() if lab else np.nan
            r[f"mode{m}_n_velocities"] = int(s["glide_velocity_m_s"].notna().sum()) if lab else np.nan
            r[f"mode{m}_median_height_mm"] = s["restrike_height_mm"].median() if lab else np.nan
        r["fracB_of_detected"] = r["n_modeB"] / len(e) if lab else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# per-condition output: (output name, per-recording column); each gets _mean and _sd over trials
COND_STATS = [
    ("fracB_of_detected", "fracB_of_detected"),
    ("discharge_time_all_ms_UPPER_BOUND", "mean_discharge_time_ms_UPPER_BOUND"),
    ("discharge_time_A_ms_UPPER_BOUND", "modeA_mean_discharge_time_ms_UPPER_BOUND"),
    ("discharge_time_B_ms_UPPER_BOUND", "modeB_mean_discharge_time_ms_UPPER_BOUND"),
    ("glide_velocity_all_m_s", "median_glide_velocity_m_s"),
    ("glide_velocity_A_m_s", "modeA_median_glide_velocity_m_s"),
    ("glide_velocity_B_m_s", "modeB_median_glide_velocity_m_s"),
    ("restrike_height_A_mm", "modeA_median_height_mm"),
    ("restrike_height_B_mm", "modeB_median_height_mm"),
]


def condition_table(R):
    """Mean and sd over trials of the per-recording values (a trial with no event of a mode is left out of that
    mode's mean; n_trials_with_B says how many trials carry Mode B values)."""
    rows = []
    for (gas, Re), r in R.groupby(KEY):
        d = dict(gas=gas, Re=Re, cut_mm=r["cut_mm"].iloc[0], n_trials=len(r),
                 n_restrikes=int(r["n_restrikes_detected"].sum()),
                 n_modeA=r["n_modeA"].sum(min_count=1), n_modeB=r["n_modeB"].sum(min_count=1),
                 n_trials_with_B=int((r["n_modeB"] > 0).sum()) if r["n_modeB"].notna().any() else np.nan,
                 modeB_observable_share=r["modeB_observable_share"].mean())
        for name, col in COND_STATS:
            v = r[col].dropna()
            d[f"{name}_mean"] = v.mean() if len(v) else np.nan
            d[f"{name}_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
        rows.append(d)
    return pd.DataFrame(rows)


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    log = []

    def say(s=""):
        print(s)
        log.append(s)

    R0 = pd.read_csv(root / "data_package" / "restrikes.csv")
    REC = pd.read_csv(root / "data_package" / "recordings.csv")
    C0 = pd.read_csv(root / "tip_cycles" / "tip_cycles.csv")
    C0 = C0[C0["rule"] == "old1.8"]
    say(f"re-strikes (1.8 mm rule) {len(R0)} in {R0.recording.nunique()} recordings; cycles {len(C0)}")
    say(f"height range {R0.restrike_height_above_gap_mm.min():.4f} .. {R0.restrike_height_above_gap_mm.max():.3f} mm")

    E = R0.rename(columns={"restrike_height_above_gap_mm": "restrike_height_mm",
                           "tip_height_before_fall_mm": "old_arc_tip_height_mm", "tip_fall_mm": "fall_mm"})
    E = E[["recording", "gas", "Re", "trial", "frame", "time_ms", "restrike_height_mm", "old_arc_tip_height_mm",
           "fall_mm"]]
    cyc = C0.rename(columns={"run": "recording", "frame_start": "frame", "cycle_ms": "discharge_time_ms",
                             "event_velocity_m_s": "glide_velocity_m_s", "tip_max_mm": "glide_max_tip_mm",
                             "arc_length_max_mm": "glide_max_arc_length_mm"})
    cyc = cyc[["recording", "frame", "discharge_time_ms", "glide_velocity_m_s", "glide_max_tip_mm",
               "glide_max_arc_length_mm"]]
    n0 = len(E)
    E = E.merge(cyc, on=["recording", "frame"], how="left", validate="one_to_one")
    assert len(E) == n0
    E = E.sort_values(["recording", "frame"]).reset_index(drop=True)
    last = E.groupby("recording")["frame"].transform("max") == E["frame"]
    assert E.loc[~last, "discharge_time_ms"].notna().all(), "a non-final event has no following cycle"
    assert E.loc[last, "discharge_time_ms"].isna().all()
    say(f"events with a following interval {E.discharge_time_ms.notna().sum()}; with a glide velocity "
        f"{E.glide_velocity_m_s.notna().sum()} (blank = < 5 rise frames); negative velocities "
        f"{(E.glide_velocity_m_s < 0).sum()}")

    # anchor: overall per-recording mean interval must reproduce the published package value
    chk = E.groupby("recording")["discharge_time_ms"].mean().rename("mine").to_frame().join(
        REC.set_index("recording")["mean_discharge_time_UPPER_BOUND_ms"])
    rel = ((chk.mine - chk.mean_discharge_time_UPPER_BOUND_ms) / chk.mean_discharge_time_UPPER_BOUND_ms).abs()
    say(f"ANCHOR mean discharge time vs data_package/recordings.csv: {len(chk)} recordings, max rel diff {rel.max():.2e}")
    assert rel.max() < 1e-4, "does not reproduce the published discharge times"

    pxmm = REC.set_index("recording")["px_per_mm"]
    E["mm_per_px"] = 1.0 / E["recording"].map(pxmm)
    assert E["mm_per_px"].notna().all()

    # Origin. restrikes.csv and tip_cycles measure heights from round(gap_row) (tip_restrike_check.load_tip); the
    # per-frame files (data_package/frames, the Alex package) use the float gap_row. Re-reference every height to the
    # float origin, offset = (round(gap_row) - gap_row) / px_per_mm, as export_for_alex.py does.
    meta = {p.parent.name: json.loads(p.read_text(encoding="utf-8"))["calibration"]
            for p in (root / "runs_annotated").rglob("run_metadata.json")}
    off = {}
    for rec in E.recording.unique():
        cal = meta[rec]
        assert np.isclose(float(cal["px_per_mm"]), pxmm[rec], rtol=0, atol=1e-4), rec
        off[rec] = (round(float(cal["gap_row"])) - float(cal["gap_row"])) / float(cal["px_per_mm"])
    o = E["recording"].map(off)
    for c in ("restrike_height_mm", "old_arc_tip_height_mm", "glide_max_tip_mm"):
        E[c] = E[c] - o
    say(f"origin offset per recording {min(off.values()):+.4f} .. {max(off.values()):+.4f} mm, applied to heights")

    # per-frame tip heights for the observability share (cut + 1.8 mm reached by the old arc's tip), and the
    # anchor: each re-strike height must equal the per-frame tip height at its frame (5 significant figures)
    tips, dmax = {}, 0.0
    for rec, grp in E.groupby("recording"):
        f = pd.read_csv(root / "data_package" / "frames" / f"{rec}.csv", usecols=["frame", "tip_height_above_gap_mm"])
        y = f.set_index("frame").loc[grp["frame"], "tip_height_above_gap_mm"].to_numpy(float)
        dmax = max(dmax, float(np.nanmax(np.abs(grp["restrike_height_mm"].to_numpy() - y))))
        tips[rec] = f["tip_height_above_gap_mm"].dropna().to_numpy()
    say(f"ANCHOR re-strike height vs per-frame tip height at the same frame: max |diff| {dmax:.2e} mm")
    assert dmax < 1e-3, "re-strike heights are not on the per-frame origin"
    assert E["restrike_height_mm"].min() > -0.1
    say(f"per-frame tip series read: {len(tips)}; tip frames {sum(len(v) for v in tips.values())}")

    # ---------------- valleys ----------------
    V = []
    for (gas, Re), cell in E.groupby(KEY):
        r = valley_with_bootstrap(cell, cell["mm_per_px"].median())
        V.append(dict(level="gas x Re", gas=gas, Re=Re, **r))
    for gas, cell in E.groupby("gas"):
        r = valley(cell["restrike_height_mm"], cell["mm_per_px"].median())
        V.append(dict(level="gas (cross-check)", gas=gas, Re=np.nan, n_events=len(cell), **r))
    V = pd.DataFrame(V)
    cols = ["level", "gas", "Re", "n_events", "n_trials", "n_peaks", "valley_mm", "ratio", "n_low", "n_high", "ok",
            "boot_n", "boot_pass_share", "boot_valley_min_mm", "boot_valley_max_mm", "accepted"]
    V = V[cols]
    V.to_csv(out / "valleys.csv", index=False)
    say("\nVALLEYS (accepted = ratio <= 0.5, >= 20 events each side, and >= 90 % of trial resamples agree)")
    say(V.round(3).to_string(index=False))
    acc = V[(V.level == "gas x Re") & V.accepted].set_index(KEY)["valley_mm"]
    say(f"accepted valleys: {len(acc)} of {(V.level == 'gas x Re').sum()} gas x Re conditions")

    # ---------------- label, per method ----------------
    def label(cuts):
        L = E.copy()
        L["cut_mm"] = cuts
        L["mode"] = np.where(L["cut_mm"].isna(), None, np.where(L["restrike_height_mm"] >= L["cut_mm"], "B", "A"))
        L["mode"] = L["mode"].where(L["cut_mm"].notna())
        return L

    fixed = label(np.full(len(E), FIXED_CUT_MM))
    vcut = E.set_index(KEY).index.map(lambda k: acc.get(k, np.nan)).to_numpy(float)
    vall = label(vcut)

    obs = {}
    for L in (fixed, vall):
        for rec, c in L.groupby("recording")["cut_mm"].first().items():
            if np.isfinite(c):
                obs[(rec, c)] = float((tips[rec] >= c + DETECT_FALL_MM).mean())

    keep = ["recording", "gas", "Re", "trial", "frame", "time_ms", "restrike_height_mm", "cut_mm", "mode",
            "old_arc_tip_height_mm", "fall_mm", "discharge_time_ms", "glide_velocity_m_s", "glide_max_tip_mm",
            "glide_max_arc_length_mm"]
    for name, L in (("fixed_cut_2p4mm", fixed), ("valley_cut", vall)):
        d = out / name
        d.mkdir(exist_ok=True)
        L[keep].to_csv(d / "events.csv", index=False)
        RT = run_table(L, obs)
        RT.to_csv(d / "runs.csv", index=False)
        CT = condition_table(RT)
        CT.to_csv(d / "conditions.csv", index=False)
        lab = L["mode"].notna()
        say(f"\n{name}: events {len(L)}, labelled {lab.sum()}, Mode A {(L['mode'] == 'A').sum()}, "
            f"Mode B {(L['mode'] == 'B').sum()}; recordings {len(RT)}; conditions {len(CT)}")

    # ---------------- fixed-cut sweep ----------------
    S = []
    for (gas, Re), cell in E.groupby(KEY):
        row = dict(gas=gas, Re=Re, n_events=len(cell))
        for c in SWEEP_MM:
            row[f"n_B_at_{c:g}mm"] = int((cell["restrike_height_mm"] >= c).sum())
        row["events_between_1_and_3mm"] = int(((cell["restrike_height_mm"] >= 1.0) &
                                               (cell["restrike_height_mm"] < 3.0)).sum())
        S.append(row)
    S = pd.DataFrame(S)
    S.to_csv(out / "fixed_cut_sweep.csv", index=False)
    say("\nFIXED-CUT SWEEP (number of Mode B events per condition at each cut)")
    say(S.to_string(index=False))
    (out / "run_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
