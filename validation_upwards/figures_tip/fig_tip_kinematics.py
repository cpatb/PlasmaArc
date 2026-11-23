"""Leading-tip kinematics vs Re, per gas, per re-strike cycle (author, 2026-10-28), top to bottom: mean local event
velocity, mean maximum arc length, mean maximum penetration height. No panel letters (author, 2026-10-28). The
Theil-Sen event velocity (event_velocity_m_s) was shown and then dropped at the author's request; only its count of
blank cycles is still printed.

Cycles come from tip_cycles/tip_cycles.csv (validation_upwards/tip_cycles.py), rule direct20 (the author's chosen
re-strike rule). A cycle runs from one kept re-strike to the next, frames [frame_start, frame_end); per cycle:
  local_vel_mean_m_s  mean of the per-frame local velocity v_ms (frames_cache.parquet: OLS over the frame and +/-5
                      advancing-front points) over the cycle's frames that carry one (advancing-front frames; a stalled
                      tip contributes its ~0)
  event_velocity_m_s  Theil-Sen slope of tip height from the cycle's first frame to its highest tip (blank if < 5 points)
  arc_length_max_mm   largest arc length in the cycle
  tip_max_mm          highest tip above the gap in the cycle (the penetration height)
Per trial: the mean over its cycles. Plotted: grey open markers single trials, coloured mean +/- sd over trials.
The mean event velocity is outlier-sensitive at low Re (Air Re 2250 trial 3: a few 5-7 point cycles starting 5.8 mm up
fit 7-20 m/s, trial mean 2.6 vs median 0.51 m/s); kept as a mean at the author's request.
Replaces the earlier panels (range of four local-velocity estimators, median arc length, p90 tip height).
Caveat carried from tip_cycles.py: a missed re-strike merges two cycles (maxima biased high), a false one splits a
cycle (biased low); the direct rule misses ~39 % of eye-seen re-strikes in the graded windows.

usage: python fig_tip_kinematics.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as C

RULE = "direct20"
PANELS = [("local_vel_mean_m_s", "Mean Local Event Velocity (m/s)"),
          ("arc_length_max_mm", "Mean Max. Arc Length (mm)"),
          ("tip_max_mm", "Mean Max. Penetration Height (mm)")]


def local_cycle_means(T):
    """Mean of the per-frame local velocity over each cycle's frames [frame_start, frame_end) that carry one."""
    F = pd.read_parquet(C.CACHE, columns=["run", "frame", "v_ms"])
    F = F[F["v_ms"].notna()]
    out = pd.Series(np.nan, index=T.index)
    for run, t in T.groupby("run"):
        f = F[F["run"] == run].sort_values("frame")
        fr, v = f["frame"].to_numpy(), f["v_ms"].to_numpy(float)
        cs = np.concatenate([[0.0], np.cumsum(v)])
        a = np.searchsorted(fr, t["frame_start"].to_numpy(), "left")
        b = np.searchsorted(fr, t["frame_end"].to_numpy(), "left")
        n = b - a
        out.loc[t.index] = np.where(n > 0, (cs[b] - cs[a]) / np.maximum(n, 1), np.nan)
    return out


def main():
    C.style()
    D = C.per_run()
    T = pd.read_csv(C.ROOT / "tip_cycles" / "tip_cycles.csv")
    n0 = len(T)
    T = T[T["rule"] == RULE].copy()
    T["local_vel_mean_m_s"] = local_cycle_means(T)
    print(f"tip_cycles: {n0} rows, {len(T)} with rule {RULE}; "
          f"event velocity blank on {T['event_velocity_m_s'].isna().sum()} cycles, "
          f"local velocity blank on {T['local_vel_mean_m_s'].isna().sum()} cycles (no local-velocity frame)")
    R = T.groupby("run")[[c for c, _ in PANELS]].mean()
    n_cyc = T.groupby("run").size().rename("n_cycles")
    D = D.drop(columns=[c for c, _ in PANELS], errors="ignore").merge(
        R.join(n_cyc), left_on="stem", right_index=True, how="left", validate="one_to_one")
    for c, _ in PANELS:
        print(f"  {c}: {int(D[c].notna().sum())} of {len(D)} runs have a value")
    fig, axs = plt.subplots(len(PANELS), 1, figsize=(C.SINGLE_W, 7.4), sharex=True)
    data = [C.vs_re_panel(ax, D, col, lab) for ax, (col, lab) in zip(axs, PANELS)]
    for ax in axs[:-1]:
        ax.set_xlabel("")
    C.legend(axs[0], handles=C.gas_handles(), title="Gas", loc="upper left", bbox_to_anchor=(0.1, 1.0))
    fig.tight_layout()
    C.save(fig, "fig_tip_kinematics", pd.concat(data, ignore_index=True),
           f"validation_upwards/figures_tip/fig_tip_kinematics.py; per-cycle means, rule {RULE}; "
           "input tip_cycles/tip_cycles.csv, frames_cache.parquet, aggregate_annotated/per_recording.csv",
           f"""# Leading-tip kinematics vs Re

Per re-strike cycle (one kept re-strike to the next, the author's direct 20 % rule), averaged over the cycles of each
trial. Top: mean local event velocity, the mean over the cycle's advancing-front frames of the local tip velocity
(least-squares fit over the frame and ±5 advancing-front points; a stalled tip counts as ~0; cycles with no such frame
have none). Middle: mean maximum arc length in a cycle. Bottom: mean maximum penetration height, the highest tip above
the minimum gap in a cycle. Small grey open markers are single trials, coloured markers
the mean ± sd over trials (3 per condition, 2 for Ar–Air at Re 2250). {len(D)} recordings; lengths use each set-up's
hand-annotated scale. A missed re-strike merges two cycles and biases (b, c) high; the rule misses ~39 % of eye-seen
re-strikes in the graded windows. Air Re values as labelled (the possible ×1.225 correction is not applied).""")

if __name__ == "__main__":
    main()
