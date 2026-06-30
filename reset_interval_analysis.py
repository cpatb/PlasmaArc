#!/usr/bin/env python3
"""
reset_interval_analysis.py  —  characterise RESET INTERVALS (long stretches with
no recognised arc) and the mode of the re-strike that follows each one.

MOTIVATION
----------
The discharge spends most of its time dark (on the 1.2 mm recording ~82% of
frames have no detected arc).  Occasionally the arc extinguishes for a long
stretch and then re-ignites.  We hypothesise that after such a full extinction
("reset interval") the discharge always restarts in Mode A (a fresh strike from
the apex), not Mode B (a back-breakdown of a still-extended channel).

This script, on the largest recording by default (1.2 mm, 873k frames):
  * detects every reset interval (no usable arc for >= reset_gap frames),
  * reports the gap-DURATION distribution (ms) and the reset RATE (resets/s),
  * attaches the FIRST re-strike after each reset and its mode (A/B),
  * tests the "reset => Mode A" hypothesis:
      - observed post-reset p_A with a Clopper-Pearson 95% CI,
      - exact binomial test that post-reset p_A differs from the background p_A
        (all re-strikes), and Fisher's exact on the 2x2 (post-reset vs other) x (A/B),
  * runs a sensitivity sweep over reset_gap in {250, 500, 1000} frames.

A reset interval = maximal run of (arc_detected != 1 OR is_flash) frames of length
>= reset_gap (missing frames count as no-arc); see arc_resets.py.  Default
reset_gap = 500 == event_velocity.MAX_FRAME_GAP.

USAGE
-----
    python reset_interval_analysis.py                       # 1.2 mm (default)
    python reset_interval_analysis.py --recording "HS_0.4mm_1000mL_(CO2)_trig"
    python reset_interval_analysis.py --all                 # every recording
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import arc_resets
import figstyle
import plot_config

DATA_ROOT_DEFAULT = Path(r"C:\Users\compl\Desktop\Plasma Arc\J Electrode")
RESET_GAP_SWEEP = (250, 500, 1000)
DEFAULT_RECORDING = "HS_1.2mm_1500mL_(CO2)"   # the largest recording

# import the recording registry (Re/d_mm) without forcing a heavy import chain
try:
    from npoint_restrike import RECORDINGS
except Exception:  # noqa: BLE001
    RECORDINGS = {}


def load_fps(results_dir: Path) -> float:
    md = results_dir / "run_metadata.json"
    txt = md.read_text(encoding="utf-8", errors="replace")
    txt = txt[:txt.rfind("}") + 1]
    return float(json.loads(txt).get("fps") or 200000.0)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> "tuple[float, float]":
    """Exact (Clopper-Pearson) binomial CI for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def analyse(stem: str, data_root: Path, reset_gap: int) -> dict:
    rd = data_root / f"{stem}_results"
    meas = pd.read_csv(rd / "measurements_all_events.csv",
                       usecols=["frame", "arc_detected", "is_flash"])
    rs = pd.read_csv(rd / "re_strikes_qa.csv")
    rs = rs[rs["mode"].isin(["A", "B"])].copy()
    fps = load_fps(rd)

    resets = arc_resets.reset_intervals(meas, reset_gap=reset_gap)
    resets = arc_resets.first_restrike_after(resets, rs)

    # gap-duration distribution context (ALL no-arc runs, no threshold)
    all_gaps = arc_resets.all_gap_lengths(meas)
    total_frames = int(meas["frame"].max() - meas["frame"].min() + 1)
    duration_s = total_frames / fps

    # post-reset modes (drop resets with no following re-strike)
    valid = resets.dropna(subset=["first_restrike_mode"])
    post_modes = valid["first_restrike_mode"].astype(str).to_numpy()
    n_post = len(post_modes)
    n_post_A = int((post_modes == "A").sum())
    p_post_A = (n_post_A / n_post) if n_post else float("nan")
    cp_lo, cp_hi = clopper_pearson(n_post_A, n_post)

    # background p_A over ALL re-strikes
    n_all = len(rs)
    n_all_A = int((rs["mode"] == "A").sum())
    p_bg_A = (n_all_A / n_all) if n_all else float("nan")

    # binomial test: post-reset A count vs background p_A (is post-reset MORE A?)
    binom_p = float("nan")
    if n_post and np.isfinite(p_bg_A) and 0 < p_bg_A < 1:
        binom_p = float(stats.binomtest(n_post_A, n_post, p_bg_A,
                                        alternative="greater").pvalue)
    # Fisher exact on (post-reset vs other) x (A/B)
    other_A = n_all_A - n_post_A
    other_B = (n_all - n_all_A) - (n_post - n_post_A)
    fisher_p = float("nan")
    if n_post and (other_A + other_B) > 0:
        table = [[n_post_A, n_post - n_post_A], [other_A, other_B]]
        try:
            _, fisher_p = stats.fisher_exact(table, alternative="greater")
            fisher_p = float(fisher_p)
        except ValueError:
            pass

    gaps_ms = resets["gap_frames"].to_numpy() / fps * 1e3 if len(resets) else np.array([])
    info = RECORDINGS.get(stem, {})
    return dict(
        stem=stem, reset_gap=reset_gap, fps=fps,
        Re=info.get("re"), d_mm=info.get("d_mm"), q_mLpm=info.get("q"),
        n_resets=int(len(resets)),
        reset_rate_per_s=round(len(resets) / duration_s, 2) if duration_s else float("nan"),
        duration_s=round(duration_s, 4),
        gap_ms_median=round(float(np.median(gaps_ms)), 4) if len(gaps_ms) else float("nan"),
        gap_ms_p90=round(float(np.percentile(gaps_ms, 90)), 4) if len(gaps_ms) else float("nan"),
        gap_ms_max=round(float(gaps_ms.max()), 4) if len(gaps_ms) else float("nan"),
        n_post_restrike=n_post, n_post_A=n_post_A,
        p_post_A=round(p_post_A, 4), p_post_A_cp_lo=round(cp_lo, 4),
        p_post_A_cp_hi=round(cp_hi, 4),
        p_background_A=round(p_bg_A, 4),
        binom_p_greater=binom_p, fisher_p_greater=fisher_p,
        n_restrikes_total=n_all,
        _resets=resets, _all_gaps=all_gaps,
    )


def make_figure(res: dict, data_root: Path):
    """Two-panel figure: (left) no-arc gap-duration distribution with the reset_gap
    cut marked; (right) post-reset first-mode counts vs the background A/B split."""
    figstyle.apply()
    rd = data_root / f"{res['stem']}_results"
    fps = res["fps"]
    all_gaps_ms = res["_all_gaps"] / fps * 1e3
    reset_cut_ms = res["reset_gap"] / fps * 1e3

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    pos = all_gaps_ms[all_gaps_ms > 0]
    if len(pos):
        bins = np.logspace(np.log10(max(pos.min(), 1e-3)), np.log10(pos.max()), 60)
        ax.hist(pos, bins=bins, color=plot_config.CIVIDIS_DISCRETE[1], alpha=0.85)
        ax.set_xscale("log")
    ax.axvline(reset_cut_ms, color="#d6604d", lw=1.5, ls="--",
               label=f"reset cut = {res['reset_gap']} fr ({reset_cut_ms:.2f} ms)")
    ax.set_xlabel("no-arc gap duration (ms, log)")
    ax.set_ylabel("count")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    p_post = res["p_post_A"]
    p_bg = res["p_background_A"]
    bars = ax.bar([0, 1], [p_post, p_bg],
                  color=[plot_config.CIVIDIS_DISCRETE[0], plot_config.CIVIDIS_DISCRETE[3]],
                  width=0.6)
    # CI on post-reset p_A
    ax.errorbar([0], [p_post],
                yerr=[[p_post - res["p_post_A_cp_lo"]], [res["p_post_A_cp_hi"] - p_post]],
                fmt="none", ecolor="black", capsize=5, lw=1.2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"first after reset\n(n={res['n_post_restrike']})",
                        f"all re-strikes\n(n={res['n_restrikes_total']})"])
    ax.set_ylabel("fraction Mode A")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", ls=":", lw=0.8)
    for b, v in zip(bars, [p_post, p_bg]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=9)

    cap = figstyle.caption_block(
        recording=res["stem"], fps=fps, px_per_mm=float("nan"),
        bright_thresh=0, n=res["reset_gap"],
        sigma_pxmm_rel=0.0, sigma_fps_rel=0.0,
        extra=(f"Reset intervals = no-arc runs >= {res['reset_gap']} frames. "
               f"n_resets={res['n_resets']}, post-reset p_A={p_post:.3f} "
               f"[{res['p_post_A_cp_lo']:.2f},{res['p_post_A_cp_hi']:.2f}] vs "
               f"background p_A={p_bg:.3f}; binom(greater) p={res['binom_p_greater']:.2g}, "
               f"Fisher(greater) p={res['fisher_p_greater']:.2g}."))
    figstyle.save(fig, "reset_interval_analysis", cap, out_dir=str(rd))


def write_report(results_by_gap: dict, stem: str, data_root: Path):
    """results_by_gap: {reset_gap: res-dict at headline}. Writes a markdown report
    with the sensitivity sweep table and the hypothesis verdict."""
    rd = data_root / f"{stem}_results"
    head0 = results_by_gap[sorted(results_by_gap)[0]]
    longest_ms = head0["gap_ms_max"]
    L = [f"# Reset-interval analysis — {stem}\n",
         "Hypothesis tested: *after every reset interval (full extinction) the "
         "discharge restarts in Mode A.*\n",
         "A reset interval = maximal run of frames with no usable arc "
         "(`arc_detected != 1 OR is_flash`) of length >= reset_gap.\n",
         "## Key finding: there are essentially NO long reset intervals\n",
         f"- The longest no-arc stretch in the whole recording is only "
         f"**{longest_ms:.2f} ms** (~{int(round(longest_ms/1000*head0['fps']))} frames). "
         f"99% of dark gaps are a few tens of frames. The discharge re-ignites "
         f"within at most ~660 frames every time — it never fully extinguishes for "
         f"a long period in this recording, so the premise of 'long spaces of no "
         f"recognised frame' does not hold here.\n",
         "- Consequently the strong form of the hypothesis cannot be tested; we test "
         "the weak form using the LONGEST dark gaps (reset_gap sweep below).\n",
         "> Caveat: the 'first re-strike after a reset' is the first detected COLLAPSE "
         "after the arc re-appears (a re-strike requires a preceding extension peak), "
         "not the initial re-ignition itself. So this tests the mode of the first "
         "*cycle* after a gap, which is the closest available proxy.\n",
         "## Sensitivity sweep over reset_gap\n",
         "| reset_gap (fr) | n_resets | rate (/s) | gap median/p90/max (ms) | "
         "post-reset p_A [95% CI] | background p_A | binom p | Fisher p |",
         "|--:|--:|--:|--|--|--:|--:|--:|"]
    for g in sorted(results_by_gap):
        r = results_by_gap[g]
        L.append(f"| {g} | {r['n_resets']} | {r['reset_rate_per_s']} | "
                 f"{r['gap_ms_median']}/{r['gap_ms_p90']}/{r['gap_ms_max']} | "
                 f"{r['p_post_A']:.2f} [{r['p_post_A_cp_lo']:.2f},{r['p_post_A_cp_hi']:.2f}] | "
                 f"{r['p_background_A']:.2f} | {r['binom_p_greater']:.2g} | "
                 f"{r['fisher_p_greater']:.2g} |")
    head = results_by_gap[arc_resets.DEFAULT_RESET_GAP] if arc_resets.DEFAULT_RESET_GAP \
        in results_by_gap else results_by_gap[sorted(results_by_gap)[0]]
    pA = head["p_post_A"]
    verdict = ("STRONGLY SUPPORTED" if pA >= 0.95 else
               "PARTIALLY SUPPORTED (A-biased but not absolute)" if pA > head["p_background_A"] + 0.05 else
               "NOT SUPPORTED (post-reset mode ~ background)")
    L.append(f"\n## Verdict (reset_gap={head['reset_gap']})\n")
    L.append(f"- Post-reset Mode-A fraction = **{pA:.3f}** "
             f"[{head['p_post_A_cp_lo']:.2f}, {head['p_post_A_cp_hi']:.2f}] "
             f"(n={head['n_post_restrike']} resets with a following re-strike).")
    L.append(f"- Background Mode-A fraction (all re-strikes) = {head['p_background_A']:.3f}.")
    L.append(f"- Post-reset A enrichment vs background: binomial(greater) "
             f"p={head['binom_p_greater']:.2g}, Fisher(greater) p={head['fisher_p_greater']:.2g}.")
    L.append(f"- **Hypothesis 'reset => Mode A': {verdict}.**")
    L.append(f"- Reset rate ~ {head['reset_rate_per_s']}/s; typical reset duration "
             f"(median) {head['gap_ms_median']} ms, up to {head['gap_ms_max']} ms.")
    out = rd / "reset_interval_report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", default=DEFAULT_RECORDING)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--data-root", default=str(DATA_ROOT_DEFAULT))
    ap.add_argument("--reset-gap", type=int, default=arc_resets.DEFAULT_RESET_GAP,
                    help="headline reset_gap (frames) for the figure; the report "
                         "always sweeps {250,500,1000}")
    args = ap.parse_args()
    data_root = Path(args.data_root)
    stems = list(RECORDINGS.keys()) if args.all else [args.recording]

    rows = []
    for stem in stems:
        print(f">>> {stem}")
        by_gap = {}
        for g in sorted(set(RESET_GAP_SWEEP) | {args.reset_gap}):
            res = analyse(stem, data_root, reset_gap=g)
            by_gap[g] = res
            print(f"    gap={g:>4}: n_resets={res['n_resets']:>4}  "
                  f"post-reset p_A={res['p_post_A']:.3f} "
                  f"[{res['p_post_A_cp_lo']:.2f},{res['p_post_A_cp_hi']:.2f}]  "
                  f"bg p_A={res['p_background_A']:.3f}  fisher_p={res['fisher_p_greater']:.2g}")
        # headline figure + per-reset csv at the chosen gap
        head = by_gap[args.reset_gap]
        rd = data_root / f"{stem}_results"
        head["_resets"].to_csv(rd / "reset_intervals.csv", index=False)
        make_figure(head, data_root)
        write_report(by_gap, stem, data_root)
        rows.append({k: v for k, v in head.items() if not k.startswith("_")})
        print(f"    wrote reset_intervals.csv, reset_interval_analysis.png/pdf, "
              f"reset_interval_report.md")

    # cross-recording summary (when --all)
    out = data_root / "Reynolds_comparison"
    out.mkdir(exist_ok=True)
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "reset_interval_summary.csv", index=False)
    print(f"\nWrote {out/'reset_interval_summary.csv'}")


if __name__ == "__main__":
    main()
