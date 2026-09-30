"""Leading-tip kinematics vs Re, per gas: (a) local tip velocity, (b) arc length, (c) p90 tip height above the gap.

Per-run medians from aggregate_annotated/per_recording.csv (hand scale per setup). Open markers: single trials;
filled: mean +/- sd over trials (n = 3; Ar-Air Re2250 n = 2).

usage: python fig_tip_kinematics.py
"""
import matplotlib.pyplot as plt
import pandas as pd

import _common as C

PANELS = [("local_vel_ms_med", "tip velocity (m/s)"), ("arc_len_mm_med", "arc length (mm)"),
          ("tip_height_above_gap_mm_p90", "tip height, p90 (mm)")]


def main():
    C.style()
    D = C.per_run()
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
    data = [C.vs_re_panel(ax, D, col, lab) for ax, (col, lab) in zip(axs, PANELS)]
    for ax, t in zip(axs, "abc"):
        ax.text(0.03, 0.97, f"({t})", transform=ax.transAxes, va="top")
    axs[0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.08, 1.0))
    fig.tight_layout()
    C.save(fig, "fig_tip_kinematics", pd.concat(data, ignore_index=True),
           "validation_upwards/figures_tip/fig_tip_kinematics.py; input aggregate_annotated/per_recording.csv",
           """# Leading-tip kinematics vs Re

(a) Local tip velocity: median per recording of the leading tip's local speed. (b) Median arc length over
complete arcs. (c) 90th percentile of the tip height above the minimum gap. Open markers are single trials;
filled markers and bars are the mean ± sd over trials (3 per condition, 2 for Ar–Air at Re 2250). 113 recordings;
lengths use each setup's hand-annotated scale (8.905–9.712 px/mm). Air Re values are used as labelled (the Air
folder's note on a possible ×1.225 setpoint correction is not applied).""")


if __name__ == "__main__":
    main()
