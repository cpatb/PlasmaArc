"""Optical mean discharge time (mean interval between successive re-strikes, the prior suites' definition), per gas
against Re. It is an UPPER BOUND; the axis label omits the words at the author's request (2026-10-01), the caption
keeps them.

Why a bound and not a value: the counted leading-tip re-strikes are real (blind audit 79/80), but the pipeline misses
real ones (completeness 0.47-0.90 where the arc glides, 1/32 in the spot regime; restrike_windows). A missed
re-strike merges two intervals into one, so the counted mean interval can only be LONGER than the true one. The ~2.5 %
of counted events that are not real (1/40 Mode A in the audit) would shorten it slightly; the bound is approximate
to that level.

Per recording, mean interval between successive KEPT re-strikes (tip_restrikes_kept_by_run.csv, cycle_ms_mean;
intervals across no-arc breaks included, author's call 8); mean +/- sd over trials. The by-eye window intervals
drawn in earlier versions were removed at the author's request (2026-10-01).

usage: python fig_discharge_time_bound.py
"""
import matplotlib.pyplot as plt

import _common as C


def main():
    C.style()
    D = C.per_run()
    fig, ax = plt.subplots(figsize=(C.SINGLE_W, 3.0))
    data = C.vs_re_panel(ax, D, "cycle_kept_ms", "Mean Discharge Time (ms)")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 100)
    C.legend(ax, handles=C.gas_handles(), title="Gas", loc="lower left", ncol=2, columnspacing=0.8)
    fig.tight_layout()
    C.save(fig, "fig_discharge_time_bound", data,
           "validation_upwards/figures_tip/fig_discharge_time_bound.py; counted = kept tip re-strikes (T_TIP); "
           "input tip_restrikes_kept_by_run.csv",
           """# Mean discharge time: an upper bound

Optical mean discharge time, defined as in the earlier suites (mean interval between successive re-strikes), from the
counted leading-tip re-strikes: per recording (small grey open markers) and mean ± sd over trials (gas markers).
**These values are UPPER BOUNDS.** Counted re-strikes are real (blind audit 79/80), but real ones are missed (the
pipeline found 0.47–0.90 of the re-strikes seen by eye where the arc glides, 1 of 32 in the spot regime), and each
miss merges two intervals, so the true mean is shorter. Log axis, because the counted bound at the lowest Re of the
CO₂ gases is dominated by the missed re-strikes and runs to tens of ms. Intervals across no-arc breaks are included
(a re-strike after the arc fails is a re-strike). The ~2.5 % of counted events that are not real would shorten the
counted interval slightly, so the bound is approximate at that level.""")


if __name__ == "__main__":
    main()
