"""Where the counted re-strikes happen, against Re: the median height of Mode A (at the gap) and Mode B (partway up)
re-strikes, with the p90 height the tip reaches for scale. One panel per gas.

Counted (kept) re-strikes only. They are verified real (blind audit: Mode A 39/40, Mode B 40/40, labels 73/76), but
their COMPLETENESS is not measured per mode, so this shows WHERE re-strikes happen, not how OFTEN. Mode B points are
drawn only where a condition has >= MIN_B Mode B re-strikes pooled over trials.

usage: python fig_restrike_height_vs_re.py
"""
import matplotlib.pyplot as plt
import pandas as pd

import _common as C

MIN_B = 10


def main():
    C.style()
    E = pd.read_csv(C.THR / "tip_restrike_events_kept.csv")
    F = C.per_run().set_index("stem")
    val = float([ln for ln in (C.THR / "tip_modes.txt").read_text(encoding="utf-8").splitlines()
                 if ln.startswith("valley value used")][0].split(":")[1].split()[0])
    n0 = len(E)
    E = E[E["kept"] & E["run"].isin(F.index)].copy()
    E["gas"], E["Re"] = E["run"].map(F["gas"]), E["run"].map(F["Re"])
    E["B"] = E["tip_after_mm"] > val
    print(f"events {n0}, kept in analysed runs {len(E)}; Mode A/B cut {val} mm")
    fig, axs = plt.subplots(1, len(C.GASES), figsize=(13, 3.0), sharey=True)
    rows = []
    for ax, gas in zip(axs, C.GASES):
        d = E[E["gas"] == gas]
        for B, mk, lab in ((False, "o", "Mode A (at gap)"), (True, "s", "Mode B (partway)")):
            g = d[d["B"] == B].groupby("Re")["tip_after_mm"]
            s = pd.DataFrame({"median": g.median(), "n": g.size(), "q25": g.quantile(0.25), "q75": g.quantile(0.75)})
            if B:
                s = s[s["n"] >= MIN_B]
            if s.empty:
                continue
            ax.errorbar(s.index, s["median"], yerr=[s["median"] - s["q25"], s["q75"] - s["median"]], fmt=mk,
                        color=C.COLOR[gas], mfc="none" if B else C.COLOR[gas], ms=4.5, capsize=2, label=lab)
            rows.append(s.reset_index().assign(gas=gas, mode="B" if B else "A"))
        p90 = F[F["gas"] == gas].groupby("Re")["tip_height_above_gap_mm_p90"].mean()
        ax.plot(p90.index, p90.values, "--", color="0.5", lw=1, label="tip height p90")
        rows.append(pd.DataFrame({"Re": p90.index, "median": p90.values, "gas": gas, "mode": "tip_p90"}))
        ax.axhline(val, color="0.7", lw=0.8, ls=":")
        ax.set_title(C.LABEL[gas], fontsize=10)
        ax.set_xlabel(C.RE_LABEL)
        ax.set_xlim(0, 9000)
    axs[0].set_ylabel("re-strike height above gap (mm)")
    ymax = max(r["q75"].max() if "q75" in r else r["median"].max() for r in rows)
    for ax in axs:
        ax.set_ylim(0, 1.05 * ymax)
    axs[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    C.save(fig, "fig_restrike_height_vs_re", pd.concat(rows, ignore_index=True),
           f"validation_upwards/figures_tip/fig_restrike_height_vs_re.py; Mode A/B cut {val} mm; Mode B shown if n >= "
           f"{MIN_B}; input threshold/tip_restrike_events_kept.csv",
           f"""# Re-strike height against Re

Median height of the new leading tip above the minimum gap at each counted re-strike (bars: interquartile range),
Mode A (filled; at the gap, ≤ {val} mm) and Mode B (open; partway up the blades), trials pooled; Mode B drawn where a
condition has ≥ {MIN_B} of them. Dashed: the 90th-percentile tip height, for scale; dotted: the Mode A/B cut.
Counted re-strikes are verified real in a blind audit, but how many are missed per mode is not measured, so this
figure says WHERE re-strikes happen, not how often or in what proportion.""")


if __name__ == "__main__":
    main()
