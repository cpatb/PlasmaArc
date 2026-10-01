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
    fig, ax = plt.subplots(figsize=(C.SINGLE_W, 2.9))
    data = C.vs_re_panel(ax, D, "frac_complete", "Complete-Arc Fraction")
    ax.set_ylim(0, 1)
    C.legend(ax, handles=C.gas_handles(), title="Gas", loc="center", bbox_to_anchor=(0.62, 0.62), ncol=2,
             columnspacing=0.8)
    fig.tight_layout()
    C.save(fig, "fig_regime", data,
           "validation_upwards/figures_tip/fig_regime.py; input aggregate_annotated/per_recording.csv",
           """# Discharge regime

Fraction of all frames holding a complete arc (attached to both blades within 7 px of the hand-drawn edges, not a
flash), per recording (small grey open markers) and mean ± sd over trials (coloured). Argon glides almost all the time (≈ 0.84) at every
Re; Air and Ar–Air sit near 0.33–0.42; CO₂ and Ar–CO₂ are spot-dominated at Re 3156 (0.04–0.06) and rise to ≈ 0.3
by Re 4969 (Ar–CO₂) and Re 5875 (CO₂); exact values in data/fig_regime.csv. Only large contrasts are resolved: neighbouring-Re steps and the Air vs Ar–Air difference are
within the detector's sensitivity.""")


if __name__ == "__main__":
    main()
