"""Discharge regime against Re: the fraction of frames holding a complete arc (attached to both blades), per gas.

Per-run `frac_complete` from aggregate_annotated/per_recording.csv; open markers single trials, filled mean +/- sd.
This is the large-contrast observable (Argon ~0.84 vs Air ~0.40 vs CO2 0.04 at low Re); steps between neighbouring
Re and the Air vs Ar-Air difference are within its detector spread (REPORT.md).

usage: python fig_regime.py
"""
import matplotlib.pyplot as plt

import _common as C


def main():
    C.style()
    D = C.per_run()
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    data = C.vs_re_panel(ax, D, "frac_complete", "complete-arc fraction of frames")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.8))
    fig.tight_layout()
    C.save(fig, "fig_regime", data,
           "validation_upwards/figures_tip/fig_regime.py; input aggregate_annotated/per_recording.csv",
           """# Discharge regime

Fraction of all frames holding a complete arc (attached to both blades within 7 px of the hand-drawn edges, not a
flash), per recording (open) and mean ± sd over trials (filled). Argon glides almost all the time (≈ 0.84) at every
Re; Air and Ar–Air sit near 0.35–0.42; CO₂ and Ar–CO₂ are spot-dominated at low Re (0.04–0.06) and reach ≈ 0.3 only
above Re ≈ 5000. Only large contrasts are resolved: neighbouring-Re steps and the Air vs Ar–Air difference are
within the detector's sensitivity.""")


if __name__ == "__main__":
    main()
