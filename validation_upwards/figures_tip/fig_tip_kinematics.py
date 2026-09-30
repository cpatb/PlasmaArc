"""Leading-tip kinematics vs Re, per gas: (a) tip velocity, (b) arc length, (c) p90 tip height above the gap.

(a) Tip velocity is drawn as the RANGE over four estimators per condition (C.velocity_estimators: frame median,
median of moving frames, frame mean, the per-event-median headline), because the 5-frame local velocity is quantised
by the integer tip row and at low Re the estimators disagree by up to x3 (code review 2026-09-30). Band = range;
line and markers = the mean of the four; a marker is hollow where the range exceeds EST_WIDE of the mean.
(b, c) Per-run values from aggregate_annotated/per_recording.csv; open markers single trials, filled mean +/- sd.

usage: python fig_tip_kinematics.py
"""
import matplotlib.pyplot as plt
import pandas as pd

import _common as C

PANELS = [("arc_len_mm_med", "arc length (mm)"), ("tip_height_above_gap_mm_p90", "tip height, p90 (mm)")]


def main():
    C.style()
    D = C.per_run()
    V = C.velocity_estimators()
    V["mid"] = V[["med", "med_nz", "mean", "agg"]].mean(axis=1)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
    a = axs[0]
    for gas in C.GASES:
        v = V[V["gas"] == gas].sort_values("Re")
        a.fill_between(v["Re"], v["lo"], v["hi"], color=C.COLOR[gas], alpha=0.25, lw=0)
        a.plot(v["Re"], v["mid"], "-", color=C.COLOR[gas], lw=1.1)
        wide = v["spread"] > C.EST_WIDE
        a.plot(v.loc[~wide, "Re"], v.loc[~wide, "mid"], C.MARK[gas], color=C.COLOR[gas], ms=4.5, ls="none",
               label=C.LABEL[gas])
        a.plot(v.loc[wide, "Re"], v.loc[wide, "mid"], C.MARK[gas], mfc="white", mec=C.COLOR[gas], ms=4.5, ls="none")
    a.set_xlabel(C.RE_LABEL), a.set_ylabel("tip velocity (m/s)")
    a.set_xlim(0, 9000), a.set_ylim(bottom=0)
    data = [V.assign(quantity="tip_velocity_estimators")]
    data += [C.vs_re_panel(ax, D, col, lab) for ax, (col, lab) in zip(axs[1:], PANELS)]
    for ax, t in zip(axs, "abc"):
        ax.text(0.03, 0.97, f"({t})", transform=ax.transAxes, va="top")
    a.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.08, 1.0))
    wide_list = ", ".join(f"{C.LABEL[g].replace('$_2$', '₂')} {r}" for g, r in V.loc[V["spread"] > C.EST_WIDE, ["gas", "Re"]].values)
    print(f"velocity estimators disagree by more than {100 * C.EST_WIDE:.0f} % at: {wide_list}")
    fig.tight_layout()
    C.save(fig, "fig_tip_kinematics", pd.concat(data, ignore_index=True),
           f"validation_upwards/figures_tip/fig_tip_kinematics.py; velocity = range of 4 estimators, hollow if range > "
           f"{C.EST_WIDE}; inputs aggregate_annotated/per_recording.csv, frames_cache.parquet",
           f"""# Leading-tip kinematics vs Re

(a) Tip velocity. The 5-frame local velocity is quantised by the whole-pixel tip position (~0.1 mm), so a slow tip
reads exactly 0 on many frames and the answer depends on the estimator. The band is the range over four estimators
(median of all velocity frames, median of moving frames, mean, and the per-event median used before); the line is
their mean. Hollow markers: the estimators disagree by more than {100 * C.EST_WIDE:.0f} % of their mean ({wide_list}).
Velocities there are not resolved and trends should be read from the filled points.
(b) Median arc length over complete arcs. (c) 90th percentile of the tip height above the minimum gap. In (b, c) open
markers are single trials, filled the mean ± sd over trials (3 per condition, 2 for Ar–Air at Re 2250). {len(D)}
recordings; lengths use each set-up's hand-annotated scale. Air Re values as labelled (the possible ×1.225 correction
is not applied).""")


if __name__ == "__main__":
    main()
