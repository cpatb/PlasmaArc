"""Optical mean discharge time (mean interval between successive re-strikes, the prior suites' definition) as an
UPPER BOUND, per gas against Re.

Why a bound and not a value: the counted leading-tip re-strikes are real (blind audit 79/80), but the pipeline misses
real ones (completeness 0.47-0.90 where the arc glides, 1/32 in the spot regime; restrike_windows). A missed
re-strike merges two intervals into one, so the counted mean interval can only be LONGER than the true one. The ~2.5 %
of counted events that are not real (1/40 Mode A in the audit) would shorten it slightly; the bound is approximate
to that level.

Counted (filled): per recording, mean interval between successive KEPT re-strikes (tip_restrikes_kept_by_run.csv,
cycle_ms_mean; intervals across no-arc breaks included, author's call 8); mean +/- sd over trials.
By eye (open stars, 6 conditions): the author's clicks in the completeness windows, mean interval between successive
clicks. Also an upper bound (partway re-strikes are hard to see in that format), tighter where it exists. One window,
one trial each.

usage: python fig_discharge_time_bound.py
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import _common as C


def eye_intervals():
    W = pd.read_csv(C.ROOT / "restrike_windows" / "windows_key.csv")
    Cl = pd.read_csv(C.ROOT / "restrike_windows" / "restrike_window_clicks.csv")
    rows = []
    for _, w in W.iterrows():
        x = np.sort(Cl.loc[Cl["code"] == w["code"], "x_px"].to_numpy())
        if len(x) < 5:
            continue
        frames = w["i0"] + x * w["bin"] + w["bin"] // 2
        dt = np.diff(frames) / 200000.0 * 1e3
        rows.append(dict(gas=w["gas"], Re=w["Re"], eye_interval_ms=float(dt.mean()), n_clicks=len(x),
                         window_ms=w["n"] / 200.0))
    return pd.DataFrame(rows)


def main():
    C.style()
    D = C.per_run()
    E = eye_intervals()
    print("by-eye windows used:\n" + E.round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(C.SINGLE_W, 3.0))
    data = C.vs_re_panel(ax, D, "cycle_kept_ms", "Mean Discharge Time, Upper Bound (ms)")
    for gas in C.GASES:
        e = E[E["gas"] == gas]
        if len(e):
            ax.plot(e["Re"], e["eye_interval_ms"], "*", ms=11, mfc=C.COLOR[gas], mec=C.EDGE, mew=0.8, zorder=4)
        ax.set_yscale("log")
    ax.set_ylim(0.5, 100)
    h = C.gas_handles() + [Line2D([0], [0], marker="*", ls="none", ms=11, mfc="white", mec=C.EDGE, mew=0.8,
                                    label="by eye")]
    C.legend(ax, handles=h, title="Gas", loc="lower left", ncol=2, columnspacing=0.8)
    fig.tight_layout()
    C.save(fig, "fig_discharge_time_bound", pd.concat([data, E.assign(quantity="eye_interval_ms")], ignore_index=True),
           "validation_upwards/figures_tip/fig_discharge_time_bound.py; counted = kept tip re-strikes (T_TIP); eye = "
           "restrike_windows clicks; inputs tip_restrikes_kept_by_run.csv, restrike_windows",
           """# Mean discharge time: an upper bound

Optical mean discharge time, defined as in the earlier suites (mean interval between successive re-strikes), shown as
an UPPER BOUND. Gas markers: from the counted leading-tip re-strikes, per recording (small grey open markers) and mean ± sd over
trials. Counted re-strikes are real (blind audit 79/80), but real ones are missed (the pipeline found 0.47–0.90 of the
re-strikes seen by eye where the arc glides, 1 of 32 in the spot regime), and each miss merges two intervals, so the
true mean is shorter. Stars: mean interval between the author's by-eye marks in single windows. Also an upper bound
(partway re-strikes are hard to see in that format), but tighter. Log axis, because the counted bound at the lowest Re
of the CO₂ gases is dominated by the missed re-strikes and runs to tens of ms. Intervals across no-arc breaks are
included (a re-strike after the arc fails is a re-strike). The ~2.5 % of counted events that are not real would
shorten the counted interval slightly, so the bound is approximate at that level.""")


if __name__ == "__main__":
    main()
