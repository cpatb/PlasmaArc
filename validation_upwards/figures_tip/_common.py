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
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnchoredText  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import figstyle  # noqa: E402

ROOT = Path(r"E:\BladeReactor_Upwards_analysis")
OUT = ROOT / "figure_suite"   # the figure suite; everything for this dataset lives on the data drive (author, 2026-09-30)
CACHE = ROOT / "figures_tip" / "frames_cache.parquet"                     # derived; rebuilt by build_frames_cache.py
THR = ROOT / "tip_restrike_check" / "threshold"
GASES = ["Argon", "Air", "Ar-Air(1_1)", "Ar-CO2(1_1)", "CO2"]          # fastest to slowest (REPORT.md finding 2)
LABEL = {"Argon": "Ar", "Air": "Air", "Ar-Air(1_1)": "Ar–Air 1:1", "Ar-CO2(1_1)": "Ar–CO$_2$ 1:1", "CO2": "CO$_2$"}

# --- House style of the last J-electrode paper (GAD PAPER 28JUL26), read off its published Origin PDFs
# (Figure2/3/6/7_Origin.pdf, MainText_VelocityNephogram.pdf; fills extracted with PyMuPDF, 2026-10-01):
# Arial; boxed axes with inward major+minor ticks on all four sides; bold "a)" panel labels inside the top-left;
# black-framed square-cornered legends with a title; markers with black edges; black capped error bars; histogram
# bars with black edges; a condition tag ("Re: 9717") in a framed box at the top right; Title Case axis labels;
# the ColorBrewer YlGnBu nozzle colours below, pale = fast (high Re) and navy = slow (low Re); plasma velocity maps
# on a black background. Gas is categorical, so each gas also keeps its own marker shape.
YLGNBU5 = ["#FFFFCC", "#A1DAB4", "#41B6C4", "#2C7FB8", "#253494"]          # exact fills of Figure6_Origin.pdf
COLOR = dict(zip(GASES, YLGNBU5))                                          # fastest gas palest, as Re in the paper
MARK = {"Argon": "o", "Air": "s", "Ar-Air(1_1)": "^", "Ar-CO2(1_1)": "D", "CO2": "v"}
RE_LABEL = "Re"
RE_LIM = (2250, 8594)                                                      # colour range of the Re ramp
EDGE = "black"
MAP_BG = "#1E1E1E"                                                         # nephogram background
SINGLE_W, DOUBLE_W = 3.35, 6.9                                             # column widths (in)


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


ZERO_EPS = 1e-9     # event_velocity writes a stalled tip's local (±5-point) OLS slope as +/-1e-13: a zero, not a velocity
EST_WIDE = 0.20     # display convention only: an estimator range wider than this fraction of its mean is drawn hollow


def velocity_estimators():
    """Per (gas, Re): the tip velocity by four estimators, and their range.

    The local velocity (OLS over the frame and ±5 advancing-front points) is quantised by the integer tip row (~0.106 mm): a slow tip reads exactly 0 on most
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
        ax.plot(d["Re"], d[col], MARK[gas], ms=3, mfc="none", mec="0.55", mew=0.6, ls="none", zorder=2)
        c = by_condition(d, col)
        ax.errorbar(c["Re"], c["mean"], yerr=c["sd"], label=LABEL[gas], **gas_err(gas))
        rows.append(c.assign(quantity=col))
    ax.set_xlabel(RE_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 9000)
    ax.set_ylim(bottom=0)
    return pd.concat(rows, ignore_index=True)


def gas_err(gas, ms=5.5, hollow=False):
    """errorbar kwargs in the paper's style: gas-coloured marker, black edge, black capped bars, no joining line."""
    return dict(fmt=MARK[gas], ms=ms, mfc="white" if hollow else COLOR[gas], mec=EDGE, mew=0.8, ecolor=EDGE,
                elinewidth=0.8, capsize=2.5, capthick=0.8, ls="none", zorder=3)


def re_color(Re):
    """The paper's Re ramp (YlGnBu, navy at low Re, pale yellow at high Re), continuous over RE_LIM."""
    return plt.get_cmap("YlGnBu_r")((Re - RE_LIM[0]) / (RE_LIM[1] - RE_LIM[0]))


def outlined(lw=1.4):
    """Black stroke under a coloured line, so the pale end of YlGnBu stays visible on white (as the paper's black
    marker and bar edges do)."""
    return dict(lw=lw, path_effects=[pe.Stroke(linewidth=lw + 1.0, foreground=EDGE), pe.Normal()])


def panel(ax, letter, x=0.025, y=0.97, color="black"):
    """Bold 'a)' inside the top-left corner."""
    ax.text(x, y, f"{letter})", transform=ax.transAxes, ha="left", va="top", fontweight="bold", color=color,
            zorder=10)


def tag(ax, text, loc="upper right"):
    """Condition tag in a black-framed box (the paper's 'Re: 9717' box)."""
    ax.add_artist(AnchoredText(text, loc=loc, frameon=True, borderpad=0.4, pad=0.25,
                               prop=dict(size=plt.rcParams["legend.fontsize"])))
    ax.artists[-1].patch.set(edgecolor=EDGE, linewidth=0.8, boxstyle="square,pad=0")


def legend(ax, **kw):
    kw.setdefault("loc", "best")
    return ax.legend(**kw)


def gas_handles(hollow_note=None):
    """Legend handles: one per gas (filled marker, black edge)."""
    h = [Line2D([0], [0], marker=MARK[g], ls="none", ms=5.5, mfc=COLOR[g], mec=EDGE, mew=0.8, label=LABEL[g])
         for g in GASES]
    if hollow_note:
        h.append(Line2D([0], [0], marker="o", ls="none", ms=5.5, mfc="white", mec=EDGE, mew=0.8, label=hollow_note))
    return h


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
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
        "font.size": 9, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8, "legend.title_fontsize": 8, "axes.titlesize": 9,
        "axes.linewidth": 0.8, "axes.edgecolor": EDGE,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.width": 0.6, "ytick.minor.width": 0.6,
        "legend.frameon": True, "legend.fancybox": False, "legend.edgecolor": EDGE, "legend.framealpha": 1.0,
        "legend.borderpad": 0.4, "legend.handletextpad": 0.3, "legend.labelspacing": 0.3,
        "legend.handlelength": 1.2, "patch.linewidth": 0.8,
        "lines.markeredgecolor": EDGE, "hatch.linewidth": 0.6,
    })
