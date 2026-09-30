"""Leading-tip re-strikes vs Re, per gas: (a) rate, (b) mean cycle time, (c) Mode B fraction.

Re-strikes: qa_restrike.py tip detector, KEPT if the largest single-step fall from the previous extension peak is
>= T_TIP (tip_restrike_threshold.py: 5 % false-detection rule from the tip data). A fall across frames with no
complete arc counts (author, 2026-09-30: the arc fails at height and re-strikes near the minimum gap), so cycle
times include those frames. Mode B: re-strike height above the valley of the pooled height distribution
(tip_modes.py). Open markers: single trials; filled: mean +/- sd over trials.

usage: python fig_tip_restrikes.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import _common as C

PANELS = [("rate_kept_per_s", "re-strike rate (1/s)"), ("cycle_kept_ms", "cycle time (ms)"),
          ("fracB", "Mode B fraction")]


def main():
    C.style()
    D = C.per_run()
    t_tip = float(Path(C.THR / "T_TIP.txt").read_text().strip())
    txt = (C.THR / "tip_modes.txt").read_text(encoding="utf-8").splitlines()
    val = [ln for ln in txt if ln.startswith("valley value used")][0].split(":")[1].strip()
    dmax = [ln for ln in txt if ln.startswith("largest per-run change")][0].split(":")[1].strip()
    nfr = sorted(D["n_frames"].unique())
    nrec = len(D)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
    data = [C.vs_re_panel(ax, D, col, lab) for ax, (col, lab) in zip(axs, PANELS)]
    axs[2].set_ylim(0, 1)
    for ax, t in zip(axs, "abc"):
        ax.text(0.03, 0.97, f"({t})", transform=ax.transAxes, va="top")
    axs[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    C.save(fig, "fig_tip_restrikes", pd.concat(data, ignore_index=True),
           f"validation_upwards/figures_tip/fig_tip_restrikes.py; T_TIP {t_tip} mm; Mode B valley {val}; inputs "
           "threshold/tip_restrikes_kept_by_run.csv, threshold/tip_modes_by_run.csv",
           f"""# Leading-tip re-strikes vs Re

(a) Re-strikes per second of record ({", ".join(f"{int(x):,}" for x in nfr)} frames at 200 kfps). (b) Mean time between successive re-strikes.
(c) Fraction of re-strikes whose new tip sits above {val} of the gap (Mode B: the arc re-strikes partway up
the blades; Mode A: at the gap). A tip re-strike is counted when the tip falls by at least {t_tip} mm in one step,
the smallest threshold at which single-step rises (which breakdown cannot produce) are at most 5 % of such falls in
every gas, pooled over all steps (for Ar–Air the falls between consecutive frames alone fail that bar; see
tip_restrike_threshold_output.txt). The net fall from the previous extension peak must also be at least
{t_tip} mm, with no single-step rise of that size after it (a one-frame spike is not a re-strike). A fall across frames with no complete arc counts, so cycle times include the time between the arc failing
and re-striking. Moving the Mode A/B cut among 1.0, 2.0 and {val} changes any recording's Mode B fraction by at most {dmax}.
Open markers: single trials; filled: mean ± sd over trials. {nrec} recordings.""")


if __name__ == "__main__":
    main()
