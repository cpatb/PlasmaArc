"""Export the BLADE-upwards LEADING-TIP results as plain, readable CSVs for a collaborator (no figures).

Writes E:\\BladeReactor_Upwards_analysis\\data_package\\:
  recordings.csv   one row per analysed recording (113)
  conditions.csv   one row per gas x Re: mean, sd over trials; tip-velocity estimator range and a resolved flag
  restrikes.csv    one row per COUNTED leading-tip re-strike (kept events only)
  README.txt       column definitions, units, and what is deliberately left out

Only quantities the analysis treats as trustworthy are exported (LEADING_TIP_FINDINGS.md, figure_suite/README.md).
Re-strike RATES and MODE B FRACTIONS are NOT exported: the pipeline misses a Re- and gas-dependent share of real
re-strikes (restrike_windows), so they are not measurements. The mean discharge time IS exported, named as the upper
bound it is. Every value is copied or computed from the same inputs the figure suite uses; the script checks that the
per-condition table reproduces the figure suite's own numbers.

usage: python export_data_package.py
"""
import numpy as np
import pandas as pd

import _common as C

OUT = C.ROOT / "data_package"
SIG = 5   # significant figures written: well above any quantity's own precision, readable in a spreadsheet


def per_recording(D):
    """One row per recording, readable names, units in the column names."""
    V = pd.read_parquet(C.CACHE, columns=["run", "v_ms"])
    V = V[V["v_ms"].notna()]
    nz = V["v_ms"].abs() > C.ZERO_EPS
    g = V.groupby("run")["v_ms"]
    E = pd.DataFrame({"tip_velocity_median_all_frames_m_s": g.median(),
                      "tip_velocity_median_moving_frames_m_s": V[nz].groupby("run")["v_ms"].median(),
                      "tip_velocity_mean_all_frames_m_s": g.mean(),
                      "stalled_frame_share": (~nz).groupby(V["run"]).mean(),
                      "n_velocity_frames": g.size()})
    R = D.set_index("stem")
    out = pd.DataFrame({
        "recording": R.index, "gas": R["gas"].values, "Re": R["Re"].values, "trial": R["trial"].values,
        "fps": R["fps"].round().astype(int).values, "n_frames": R["n_frames"].values, "px_per_mm": R["px_per_mm"].values,
        "tip_velocity_median_of_event_medians_m_s": R["local_vel_ms_med"].values,
        "arc_length_median_mm": R["arc_len_mm_med"].values,
        "tip_height_above_gap_median_mm": R["tip_height_above_gap_mm_med"].values,
        "tip_height_above_gap_p90_mm": R["tip_height_above_gap_mm_p90"].values,
        "complete_arc_fraction_of_frames": R["frac_complete"].values,
        "n_restrikes_counted": np.nan,
        "mean_discharge_time_UPPER_BOUND_ms": R["cycle_kept_ms"].values,
    }).set_index("recording")
    out = out.join(E, how="left")
    if out["n_velocity_frames"].isna().any():
        raise SystemExit("recordings without velocity frames in the cache")
    cols = ["gas", "Re", "trial", "fps", "n_frames", "px_per_mm", "tip_velocity_median_all_frames_m_s",
            "tip_velocity_median_moving_frames_m_s", "tip_velocity_mean_all_frames_m_s",
            "tip_velocity_median_of_event_medians_m_s", "stalled_frame_share", "n_velocity_frames",
            "arc_length_median_mm", "tip_height_above_gap_median_mm", "tip_height_above_gap_p90_mm",
            "complete_arc_fraction_of_frames", "n_restrikes_counted", "mean_discharge_time_UPPER_BOUND_ms"]
    return out[cols]


def restrikes(D):
    E = pd.read_csv(C.THR / "tip_restrike_events_kept.csv")
    n0 = len(E)
    R = D.set_index("stem")
    E = E[E["kept"] & E["run"].isin(R.index)].copy()
    print(f"re-strike events {n0}; counted (kept) in analysed recordings {len(E)}")
    fps = E["run"].map(R["fps"])
    return pd.DataFrame({
        "recording": E["run"], "gas": E["run"].map(R["gas"]), "Re": E["run"].map(R["Re"]),
        "trial": E["run"].map(R["trial"]), "frame": E["frame_min"], "time_ms": E["frame_min"] / fps * 1e3,
        "restrike_height_above_gap_mm": E["tip_after_mm"],
        "tip_height_before_fall_mm": E["tip_after_mm"] + E["fall_mm"], "tip_fall_mm": E["fall_mm"],
        "largest_single_frame_fall_mm": E["max_step_mm"],
    }).sort_values(["gas", "Re", "trial", "frame"]).reset_index(drop=True)


def per_condition(rec, ev):
    q = ["arc_length_median_mm", "tip_height_above_gap_p90_mm", "complete_arc_fraction_of_frames",
         "mean_discharge_time_UPPER_BOUND_ms"]
    g = rec.groupby(["gas", "Re"])
    T = g.size().rename("n_trials").to_frame()
    for c in q:
        T[c + "_mean"], T[c + "_sd"] = g[c].mean(), g[c].std()
    V = C.velocity_estimators().set_index(["gas", "Re"])
    T["tip_velocity_estimator_min_m_s"] = V["lo"]
    T["tip_velocity_estimator_max_m_s"] = V["hi"]
    T["tip_velocity_estimator_mean_m_s"] = V[["med", "med_nz", "mean", "agg"]].mean(axis=1)
    T["tip_velocity_estimator_spread"] = V["spread"]
    T["tip_velocity_resolved"] = V["spread"] <= C.EST_WIDE
    T["stalled_frame_share"] = V["zero"]
    h = ev.groupby(["gas", "Re"])["restrike_height_above_gap_mm"]
    T["n_restrikes_counted"] = h.size()
    T["restrike_height_median_mm"] = h.median()
    return T.reset_index()


def check_against_suite(T):
    """The per-condition table must reproduce the figure suite's plotted means (an anchor outside this script)."""
    K = pd.read_csv(C.OUT / "data" / "fig_tip_kinematics.csv", comment="#")
    for col, name in (("arc_len_mm_med", "arc_length_median_mm_mean"),
                      ("tip_height_above_gap_mm_p90", "tip_height_above_gap_p90_mm_mean")):
        k = K[K["quantity"] == col].set_index(["gas", "Re"])["mean"]
        t = T.set_index(["gas", "Re"])[name].reindex(k.index)
        np.testing.assert_allclose(t.to_numpy(), k.to_numpy(), rtol=1e-5)   # suite CSV is written at 6 sig. figs
    print(f"conditions.csv reproduces fig_tip_kinematics.csv on {len(k)} conditions x 2 quantities")


def write(df, name, index=False):
    with open(OUT / name, "w", encoding="utf-8", newline="\n") as f:
        df.to_csv(f, index=index, float_format=f"%.{SIG}g", lineterminator="\n")
    print(f"wrote {OUT / name}: {len(df)} rows x {df.shape[1] + int(index)} columns")


README = """BLADE reactor, upwards 45 deg, 2.4 mm gap: leading-arc-tip data (plain CSV, UTF-8)
Generated by validation_upwards/figures_tip/export_data_package.py (git {git}). Every number comes from that script.

DATA SET
  {n_rec} high-speed recordings (200 000 fps, {n_frames_note}), 5 gases (Ar, Air, Ar-Air 1:1, Ar-CO2 1:1, CO2),
  nozzle Reynolds number Re 2250-8594 (3156-8594 for the CO2 gases), 0.6 mm nozzle, 3 trials per condition
  (Ar-Air Re2250: 2 analysed; its trial 1 does not glide). Lengths use each set-up's hand-annotated px/mm scale.
  Heights are measured upward from the minimum gap between the blades.
  Air Re values are as labelled on the files; a possible x1.225 correction noted in the Air folder is NOT applied.

FILES
  recordings.csv  one row per recording
  conditions.csv  one row per gas and Re: mean and sd over trials, plus the tip-velocity estimator range
  restrikes.csv   one row per counted re-strike of the leading arc tip

DEFINITIONS
  Leading tip: the highest point of the arc (furthest from the gap along the blades).
  Tip velocity: 5-frame local velocity of the leading tip along the blades, on frames where the tip advances.
    It is QUANTISED by whole-pixel tip positions (~0.1 mm), so a slow tip reads exactly 0 on many frames
    (stalled_frame_share) and the answer depends on the estimator. Four estimators are given:
      median over all velocity frames; median over moving frames; mean over all frames; median of event medians.
    conditions.csv gives their min, max and mean; tip_velocity_resolved = False where they differ by more than
    20 % of their mean. Do not quote a single velocity there; quote the range.
  Arc length: median over complete arcs (attached to both blades) of the arc's length, mm.
  Complete-arc fraction: share of all frames holding a complete arc.
  Re-strike: a fall of the leading tip of at least 1.8 mm, with a single-frame fall of at least 1.8 mm and no
    rebound (threshold set by a 5 % false-detection rule). A fall across a break with no arc counts. The time and
    frame are those at which the new tip is lowest; restrike_height_above_gap_mm is the new tip's height there.
  mean_discharge_time_UPPER_BOUND_ms: mean interval between successive counted re-strikes in a recording; blank
    where a recording has fewer than two counted re-strikes (no interval exists).

CAVEATS (read before use)
  - Counted re-strikes are real (blind visual audit: 79 of 80), but the analysis MISSES real ones: it counted
    0.47-0.90 of those seen by eye where the arc glides, and 1 of 32 in the low-Re Ar-CO2 spot regime. Hence:
      * the mean discharge time is an UPPER BOUND (each miss merges two intervals); the true value is shorter;
      * n_restrikes_counted is a LOWER BOUND and must not be turned into a rate or compared across Re or gas;
      * restrike heights show WHERE re-strikes happen; their numbers per height are not comparable across Re.
  - Re-strike rates and the fraction of re-strikes partway up the blades are deliberately NOT included: they are
    not measurements with this detector.
  - Tip velocities are not resolved where tip_velocity_resolved is False (low Re, especially the CO2 gases).
"""


def main():
    D = C.per_run()
    OUT.mkdir(exist_ok=True)
    rec = per_recording(D)
    ev = restrikes(D)
    rec["n_restrikes_counted"] = ev.groupby("recording").size().reindex(rec.index).fillna(0).astype(int)
    T = per_condition(rec, ev)
    check_against_suite(T)
    if int(rec["n_restrikes_counted"].sum()) != len(ev):
        raise SystemExit("re-strike counts do not add up")
    write(rec, "recordings.csv", index=True)
    write(T, "conditions.csv")
    write(ev, "restrikes.csv")
    nf = rec["n_frames"]
    (OUT / "README.txt").write_text(README.format(git=C.git_hash(), n_rec=len(rec),
                                                  n_frames_note=f"{int(nf.min())}-{int(nf.max())} frames each"),
                                    encoding="utf-8", newline="\n")
    print(f"wrote {OUT / 'README.txt'}")


if __name__ == "__main__":
    main()
