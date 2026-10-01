"""Validation summary for the leading-tip analysis: what was checked and what it found, from the saved outputs.

(a) Scale: hand-annotated px/mm per setup (mean +/- sem over drawings; Argon setup 2 is one drawing, no bar).
(b) Tip re-strike threshold: false-detection estimate (single-step rises / falls >= T) per gas against T, with
    the 5 % bar and T_TIP (tip_restrike_threshold.py).
(c) Blind audit: real (R) / not real (N) for 40 Mode A, 40 Mode B and 20 rejected events (restrike_audit).
(d) Completeness windows (6 of 12 marked, descriptive): fraction of eye-seen re-strikes the pipeline counted,
    Wilson 95 % (restrike_windows).

usage: python fig_validation.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _common as C


def scales():
    rows = {}
    for mp in (C.ROOT / "runs_annotated").rglob("run_metadata.json"):
        c = json.loads(mp.read_text(encoding="utf-8"))["calibration"]
        if c.get("source") == "hand annotation":
            key = " / ".join([c["condition"][0]] + c["condition"][2:])
            rows[key] = dict(setup=key, gas=c["condition"][0], px_per_mm=c["px_per_mm"],
                             sem=c["px_per_mm_sem"] if c["members"]["n"] > 1 else np.nan, n=c["members"]["n"])
    return pd.DataFrame(rows.values()).sort_values("setup")


def main():
    C.style()
    fig, axs = plt.subplots(2, 2, figsize=(C.DOUBLE_W, 6.2))
    (a, b), (c, d) = axs
    out = []
    # (a) scale
    S = scales()
    x = np.arange(len(S))
    a.bar(x, S["px_per_mm"], yerr=S["sem"], color=[C.COLOR[g] for g in S["gas"]], edgecolor=C.EDGE, linewidth=0.8,
          capsize=3, error_kw=dict(elinewidth=0.8, capthick=0.8))
    a.set_xticks(x, [s.replace("45degree_2.4mmGap", "").replace("(1_1)", "").replace("setup1_shifted_right_6px",
                                                                                        "set-up 1").replace("setup2", "set-up 2")
                     for s in S["setup"]], rotation=30, ha="right", fontsize=7)
    a.set_ylabel("Scale (px/mm)")
    a.tick_params(axis="x", which="minor", bottom=False, top=False)
    a.set_ylim(0, 1.15 * S["px_per_mm"].max())
    for xi, n in zip(x, S["n"]):
        a.text(xi, 0.3, f"n={n}", ha="center", fontsize=7, color="black")
    out.append(S.assign(panel="a"))
    # (b) threshold rule
    G = pd.read_csv(C.THR / "tip_threshold_curve.csv")
    t_tip = float((C.THR / "T_TIP.txt").read_text().strip())
    for gas in C.GASES:
        g = G[G["gas"] == gas]
        b.plot(g["T_mm"], 100 * g["fdr"], "-", color=C.COLOR[gas], label=C.LABEL[gas], **C.outlined(1.2))
    b.axhline(5, color="0.4", ls="--", lw=0.8)
    b.axvline(t_tip, color="0.4", ls=":", lw=0.8)
    b.set_xlabel("Single-Step Fall Threshold T (mm)"), b.set_ylabel("False-Detection Estimate (%)")
    b.set_xlim(0, 5), b.set_ylim(0, 40)
    C.legend(b, title="Gas", loc="upper right")
    out.append(G.assign(panel="b"))
    # (c) blind audit
    A = pd.read_csv(C.ROOT / "restrike_audit" / "graded.csv")
    cls = [("KA", "Mode A\n(counted)"), ("KB", "Mode B\n(counted)"), ("REJ", "rejected")]
    for i, (k, lab) in enumerate(cls):
        dd = A[A["cls"] == k]
        r, n = int((dd["q1"] == "R").sum()), int((dd["q1"] == "N").sum())
        c.bar(i, r, color=C.YLGNBU5[1], edgecolor=C.EDGE, linewidth=0.8, label="Real" if i == 0 else None)
        c.bar(i, n, bottom=r, color=C.YLGNBU5[4], edgecolor=C.EDGE, linewidth=0.8, label="Not real" if i == 0 else None)
        c.text(i, r + n + 0.5, f"{r}/{r + n}", ha="center", fontsize=8)
        out.append(pd.DataFrame([{"panel": "c", "class": k, "R": r, "N": n}]))
    c.set_xticks(range(3), [x[1] for x in cls])
    c.set_ylabel("Events Judged Blind")
    c.tick_params(axis="x", which="minor", bottom=False, top=False)
    c.set_ylim(0, 45)
    C.legend(c, loc="upper right")
    # (d) completeness windows (descriptive)
    W = pd.read_csv(C.ROOT / "restrike_windows" / "windows_graded.csv")
    W = W[(W["mode"] == "all") & (W["H"] > 0)].sort_values(["gas", "Re"])
    y = np.arange(len(W))
    d.errorbar(W["C"], y, xerr=[W["C"] - W["C_lo"], W["C_hi"] - W["C"]], fmt="none", ecolor=C.EDGE, elinewidth=0.8,
               capsize=2.5, capthick=0.8)
    for yi, (_, r) in zip(y, W.iterrows()):
        d.plot(r["C"], yi, C.MARK[r["gas"]], ms=6, mfc=C.COLOR[r["gas"]], mec=C.EDGE, mew=0.8)
    d.set_yticks(y, [f"{C.LABEL[g]} Re{re}  ({h} seen)" for g, re, h in zip(W["gas"], W["Re"], W["H"])], fontsize=7)
    d.set_xlim(0, 1), d.set_xlabel("Fraction of Eye-Seen Re-strikes Counted")
    d.tick_params(axis="y", which="minor", left=False, right=False)
    d.set_ylim(-0.6, len(W) - 0.4)
    out.append(W.assign(panel="d"))
    for ax, t in zip(axs.ravel(), "abcd"):
        C.panel(ax, t)
    fig.tight_layout()
    C.save(fig, "fig_validation", pd.concat(out, ignore_index=True),
           f"validation_upwards/figures_tip/fig_validation.py; T_TIP {t_tip} mm; inputs runs_annotated metadata, "
           "threshold/tip_threshold_curve.csv, restrike_audit/graded.csv, restrike_windows/windows_graded.csv",
           f"""# Validation of the leading-tip analysis

(a) Scale per set-up from the author's hand-drawn gap lines (mean ± sem over drawings; n = drawings; Argon set-up 2
rests on one drawing). (b) The tip re-strike threshold: single-step tip rises (which a breakdown cannot produce) as a
fraction of single-step falls, per gas; dashed 5 % bar, dotted T_TIP = {t_tip} mm, the smallest threshold meeting the
bar in every gas. (c) Blind audit by the author: counted re-strikes are real (Mode A 39/40, Mode B 40/40), but 13 of
20 rejected events are real too. (d) Completeness, descriptive (6 of 12 planned windows marked): the pipeline counts
about half to nine-tenths of the re-strikes seen by eye where the arc glides, and 1 of 32 in the spot-dominated
low-Re Ar–CO₂ window. Bars: Wilson 95 %. Consequence: re-strike rates, cycle times and Mode B fractions are not
quoted.""")


if __name__ == "__main__":
    main()
