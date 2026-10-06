"""Leading-tip kinematics vs Re, per gas: (a) tip velocity, (b) arc length, (c) p90 tip height above the gap.

(a) Tip velocity is drawn as the RANGE over four estimators per condition (C.velocity_estimators: frame median,
median of moving frames, frame mean, the per-event-median headline), because the local velocity (OLS over the frame and ±5 advancing-front points) is quantised
by the integer tip row and at low Re the estimators disagree by up to x3 (code review 2026-09-30). Bar = range;
marker = the mean of the four; a marker is hollow where the range exceeds EST_WIDE of the mean.
(b, c) Per-run values from aggregate_annotated/per_recording.csv; grey open markers single trials, coloured mean +/- sd.
Styled as the J-electrode paper (_common.style): stacked single-column panels sharing Re.

usage: python fig_tip_kinematics.py
"""
import matplotlib.pyplot as plt
import pandas as pd

import _common as C

PANELS = [("arc_len_mm_med", "Arc Length (mm)"), ("tip_height_above_gap_mm_p90", "Tip Height, p90 (mm)")]


def main():
    C.style()
    D = C.per_run()
    V = C.velocity_estimators()
    V["mid"] = V[["med", "med_nz", "mean", "agg"]].mean(axis=1)
    fig, axs = plt.subplots(3, 1, figsize=(C.SINGLE_W, 7.4), sharex=True)
    a = axs[0]
    for gas in C.GASES:
        v = V[V["gas"] == gas].sort_values("Re")
        wide = (v["spread"] > C.EST_WIDE).to_numpy()
        # estimator range as a black-capped bar; hollow marker where it exceeds EST_WIDE of the mean
        for sel, hollow in ((~wide, False), (wide, True)):
            s = v[sel]
            a.errorbar(s["Re"], s["mid"], yerr=[s["mid"] - s["lo"], s["hi"] - s["mid"]],
                       **C.gas_err(gas, hollow=hollow))
    a.set_ylabel("Tip Velocity (m/s)")
    a.set_xlim(0, 9000), a.set_ylim(bottom=0)
    data = [V.assign(quantity="tip_velocity_estimators")]
    data += [C.vs_re_panel(ax, D, col, lab) for ax, (col, lab) in zip(axs[1:], PANELS)]
    for ax in axs[:-1]:
        ax.set_xlabel("")
    for ax, t in zip(axs, "abc"):
        C.panel(ax, t)
    C.legend(a, handles=C.gas_handles(hollow_note=f"range > {100 * C.EST_WIDE:.0f} %"), title="Gas",
             loc="upper left", bbox_to_anchor=(0.1, 1.0))
    wide_list = ", ".join(f"{C.LABEL[g].replace('$_2$', '₂')} {r}" for g, r in V.loc[V["spread"] > C.EST_WIDE, ["gas", "Re"]].values)
    print(f"velocity estimators disagree by more than {100 * C.EST_WIDE:.0f} % at: {wide_list}")
    fig.tight_layout()
    C.save(fig, "fig_tip_kinematics", pd.concat(data, ignore_index=True),
           f"validation_upwards/figures_tip/fig_tip_kinematics.py; velocity = range of 4 estimators, hollow if range > "
           f"{C.EST_WIDE}; inputs aggregate_annotated/per_recording.csv, frames_cache.parquet",
           f"""# Leading-tip kinematics vs Re

(a) Tip velocity. The local velocity (OLS over the frame and ±5 advancing-front points) is quantised by the whole-pixel tip position (~0.1 mm), so a slow tip
reads exactly 0 on many frames and the answer depends on the estimator. The bar is the range over four estimators
(median of all velocity frames, median of moving frames, mean, and the per-event median used before); the marker is
their mean. Hollow markers: the estimators disagree by more than {100 * C.EST_WIDE:.0f} % of their mean ({wide_list}).
Velocities there are not resolved and trends should be read from the filled points.
(b) Median arc length over complete arcs. (c) 90th percentile of the tip height above the minimum gap. In (b, c) small grey
open markers are single trials, coloured markers the mean ± sd over trials (3 per condition, 2 for Ar–Air at Re 2250). {len(D)}
recordings; lengths use each set-up's hand-annotated scale. Air Re values as labelled (the possible ×1.225 correction
is not applied).""")


if __name__ == "__main__":
    main()
