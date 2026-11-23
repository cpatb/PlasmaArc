"""Each panel of fig_tip_kinematics and fig_restrike_height as its own high-DPI file (author, 2026-10-28).

Same data and same per-panel drawing as the two parent scripts (fig_tip_kinematics.local_cycle_means / PANELS / RULE,
C.vs_re_panel; fig_restrike_height.BIN_MM and its histogram), one figure per panel, each with its own axis labels and,
for the kinematics panels, its own legend. Written to <ROOT>/figures_tip/panels/ as PNG (DPI) and PDF.

usage: python fig_tip_panels.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as C
import fig_restrike_height as RH
import fig_tip_kinematics as TK

DPI = 600
SIZE = (C.SINGLE_W, 3.2)   # 2.6 in clipped "Mean Max. Penetration Height (mm)"


def out_dir():
    d = C.ROOT / "figures_tip" / "panels"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(fig, name):
    d = out_dir()
    fig.tight_layout()
    fig.savefig(d / f"{name}.png", dpi=DPI, metadata={"Software": None})
    fig.savefig(d / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"wrote {d / name}.png/.pdf")


def kinematics():
    D = C.per_run()
    T = pd.read_csv(C.ROOT / "tip_cycles" / "tip_cycles.csv")
    T = T[T["rule"] == TK.RULE].copy()
    T["local_vel_mean_m_s"] = TK.local_cycle_means(T)
    R = T.groupby("run")[[c for c, _ in TK.PANELS]].mean()
    D = D.drop(columns=[c for c, _ in TK.PANELS], errors="ignore").merge(
        R, left_on="stem", right_index=True, how="left", validate="one_to_one")
    for col, lab in TK.PANELS:
        fig, ax = plt.subplots(figsize=SIZE)
        C.vs_re_panel(ax, D, col, lab)
        C.legend(ax, handles=C.gas_handles(), title="Gas", loc="upper left", fontsize=6, title_fontsize=6)
        save(fig, f"fig_tip_kinematics_{col}")


def restrike_height():
    E = pd.read_csv(C.THR / "tip_restrike_events_kept.csv")
    F = pd.read_csv(C.ROOT / "aggregate_annotated" / "per_recording.csv").set_index("stem")
    E = E[E["kept"]].copy()
    E["gas"] = E["run"].map(F["gas"])
    hmax = float(np.ceil(E["tip_after_mm"].max() / RH.BIN_MM) * RH.BIN_MM)
    bins = np.arange(0, hmax + RH.BIN_MM / 2, RH.BIN_MM)
    for gas in C.GASES:
        h, _ = np.histogram(E.loc[E["gas"] == gas, "tip_after_mm"], bins)
        fig, ax = plt.subplots(figsize=SIZE)
        ax.bar(bins[:-1], h, width=RH.BIN_MM, align="edge", color=C.COLOR[gas], edgecolor=C.EDGE, linewidth=0.5)
        ax.set_yscale("log")
        ax.set_ylim(0.8, 2e3)
        ax.set_xlim(0, hmax)
        C.tag(ax, f"{C.LABEL[gas]}, N = {int(h.sum())}")
        ax.set_xlabel("Re-strike Height Above Gap (mm)")
        ax.set_ylabel("Count")
        save(fig, f"fig_restrike_height_{gas}")


def main():
    C.style()
    kinematics()
    restrike_height()


if __name__ == "__main__":
    main()
