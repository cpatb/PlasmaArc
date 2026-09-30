"""Shared inputs and style for the BLADE-upwards LEADING-TIP figures (one script per figure).

Inputs (all printed by scripts; nothing typed):
  aggregate_annotated/per_recording.csv                    local tip velocity, arc length, tip height (per run)
  tip_restrike_check/threshold/tip_restrikes_kept_by_run.csv   re-strike rate, cycle (fall >= T_TIP)
  tip_restrike_check/threshold/tip_modes_by_run.csv        Mode B fraction (valley split)
  tip_restrike_check/threshold/tip_restrike_events_kept.csv    per re-strike height (histogram figure)
"""
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import figstyle  # noqa: E402

ROOT = Path(r"E:\BladeReactor_Upwards_analysis")
OUT = Path(r"C:\Users\compl\Desktop\Plasma Arc\BLADE\BLADE_UPWARDS_45deg")   # the figure suite (author, 2026-09-30)
CACHE = ROOT / "figures_tip" / "frames_cache.parquet"                     # derived; rebuilt by build_frames_cache.py
THR = ROOT / "tip_restrike_check" / "threshold"
GASES = ["Argon", "Air", "Ar-Air(1_1)", "Ar-CO2(1_1)", "CO2"]          # fastest to slowest (REPORT.md finding 2)
LABEL = {"Argon": "Ar", "Air": "Air", "Ar-Air(1_1)": "Ar–Air 1:1", "Ar-CO2(1_1)": "Ar–CO$_2$ 1:1", "CO2": "CO$_2$"}
# Okabe-Ito (colourblind-safe categorical)
COLOR = {"Argon": "#0072B2", "Air": "#E69F00", "Ar-Air(1_1)": "#009E73", "Ar-CO2(1_1)": "#CC79A7", "CO2": "#D55E00"}
MARK = {"Argon": "o", "Air": "s", "Ar-Air(1_1)": "^", "Ar-CO2(1_1)": "D", "CO2": "v"}
RE_LABEL = "Re"


def git_hash():
    try:
        h = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                           check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "validation_upwards"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def per_run():
    """One row per analysed recording, every plotted per-run quantity, with N printed before/after each join."""
    F = pd.read_csv(ROOT / "aggregate_annotated" / "per_recording.csv")
    n0 = len(F)
    anchor = F["physical_gap_anchor"].astype("boolean").fillna(False).astype(bool)
    F = F[(F["status"] == "DONE") & (F["tag"].fillna("") == "") & (F["short_complete_frac"].fillna(1.0) < 0.001)
          & anchor]
    print(f"per_recording: {n0} rows, {len(F)} DONE and passing every gate")
    K = pd.read_csv(THR / "tip_restrikes_kept_by_run.csv")[["run", "rate_per_s", "cycle_ms_mean"]].rename(
        columns={"rate_per_s": "rate_kept_per_s", "cycle_ms_mean": "cycle_kept_ms"})   # per_recording has the unfiltered ones
    M = pd.read_csv(THR / "tip_modes_by_run.csv")[["run", "fracB"]]
    D = F.merge(K, left_on="stem", right_on="run", how="left", validate="one_to_one").drop(columns="run")
    D = D.merge(M, left_on="stem", right_on="run", how="left", validate="one_to_one").drop(columns="run")
    for c in ("rate_kept_per_s", "cycle_kept_ms", "fracB"):
        print(f"  {c}: {int(D[c].notna().sum())} of {len(D)} runs have a value")
    unknown = set(D["gas"]) - set(GASES)
    if unknown:
        raise SystemExit(f"unknown gas labels {unknown}")
    return D


ZERO_EPS = 1e-9     # event_velocity writes a stalled tip's 5-frame OLS slope as +/-1e-13: a zero, not a velocity
EST_WIDE = 0.20     # display convention only: an estimator range wider than this fraction of its mean is drawn hollow


def velocity_estimators():
    """Per (gas, Re): the tip velocity by four estimators, and their range.

    The 5-frame local velocity is quantised by the integer tip row (~0.106 mm): a slow tip reads exactly 0 on most
    frames and one-pixel values on the rest (code review 2026-09-30). Its median is then a biased estimator, and which
    estimator is used matters. Reported side by side, no selection:
      med     median over all velocity frames (zeros included)
      med_nz  median over frames with |v| > ZERO_EPS
      mean    mean over all velocity frames (least affected by quantisation for a steady tip)
      agg     the headline number: per recording, median over events of each event's median (aggregate_upwards.py)
    lo / hi = min / max of the four; spread = (hi - lo) / mean of the four; zero = share of exactly-zero frames."""
    D = pd.read_parquet(CACHE, columns=["gas", "Re", "v_ms"])
    D = D[D["v_ms"].notna()]
    nz = D["v_ms"].abs() > ZERO_EPS
    g = D.groupby(["gas", "Re"])["v_ms"]
    T = pd.DataFrame({"med": g.median(), "med_nz": D[nz].groupby(["gas", "Re"])["v_ms"].median(), "mean": g.mean(),
                      "zero": (~nz).groupby([D["gas"], D["Re"]]).mean()})
    T["agg"] = per_run().groupby(["gas", "Re"])["local_vel_ms_med"].mean()
    E = T[["med", "med_nz", "mean", "agg"]]
    T["lo"], T["hi"] = E.min(axis=1), E.max(axis=1)
    T["spread"] = (T["hi"] - T["lo"]) / E.mean(axis=1)
    return T.reset_index()


def by_condition(D, col):
    g = D.groupby(["gas", "Re"])[col]
    return pd.DataFrame({"mean": g.mean(), "sd": g.std(), "n": g.count()}).reset_index()


def vs_re_panel(ax, D, col, ylabel):
    """Per trial (small open markers) and mean +/- sd over trials (filled, joined)."""
    rows = []
    for gas in GASES:
        d = D[D["gas"] == gas]
        ax.plot(d["Re"], d[col], MARK[gas], ms=3, mfc="none", mec=COLOR[gas], mew=0.6, alpha=0.6, ls="none")
        c = by_condition(d, col)
        ax.errorbar(c["Re"], c["mean"], yerr=c["sd"], fmt=MARK[gas] + "-", color=COLOR[gas], ms=4.5, lw=1.1,
                    capsize=2, label=LABEL[gas])
        rows.append(c.assign(quantity=col))
    ax.set_xlabel(RE_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 9000)
    ax.set_ylim(bottom=0)
    return pd.concat(rows, ignore_index=True)


def save(fig, slug, data: pd.DataFrame, provenance: str, caption_md: str):
    """PDF (vector, fixed metadata so it regenerates identically) + PNG 300 dpi + plotted-data CSV + caption .md."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    prov = f"{provenance}; git {git_hash()}"      # no date: the outputs must regenerate identically on any day
    fig.savefig(OUT / f"{slug}.pdf", metadata={"Subject": prov, "CreationDate": None, "ModDate": None})
    fig.savefig(OUT / f"{slug}.png", dpi=300, metadata={"Software": None})
    plt.close(fig)
    with open(OUT / "data" / f"{slug}.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# {prov}\n")
        data.to_csv(f, index=False, float_format="%.6g", lineterminator="\n")
    (OUT / f"{slug}.md").write_text(caption_md.strip() + f"\n\nProvenance: {prov}\n", encoding="utf-8")
    print(f"wrote {OUT / slug}.pdf/.png/.md and data/{slug}.csv")


def style():
    figstyle.apply()
