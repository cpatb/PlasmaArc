#!/usr/bin/env python3
"""
mode_transition_analysis.py  --  temporal & conditional-probability (Markov)
analysis of Mode A / Mode B re-strike sequences.

Reads the authoritative per-recording Mode A/B record (``re_strikes_qa.csv``)
and answers two questions that the existing pipeline does NOT:

1.  CONDITIONAL PROBABILITY / MEMORY.  Treating the ordered sequence of
    re-strike modes (A,B,A,B,B,...) as a two-state chain, estimate the
    first-order transition matrix

        P(A|A) P(B|A)
        P(A|B) P(B|B)

    and test whether the next mode depends on the current one at all
    (first-order Markov) or whether re-strikes are memoryless / independent
    (zeroth-order, a biased coin).  Reported alongside: the marginal mode
    fractions, the chain's stationary distribution, lag-1 autocorrelation of
    the binarised mode series, "excess persistence" P(m|m) - P(m), and the
    observed-vs-independence Mode-B run-length (clustering of back-breakdowns).

2.  NON-STATIONARITY IN TIME.  Does the discharge drift over the recording?
    For each recording we bin the re-strikes in time and test for a monotone
    trend (Spearman) in: Mode-B fraction, cycle period (cycle_frames), and
    re-strike nucleation position (norm_pos).  A significant trend means the
    arc statistics are not stationary (warm-up, electrode erosion, gas drift).

A valid transition is one between two *contiguous* re-strikes.  In
``re_strikes_qa.csv`` the ``cycle_frames`` column equals the inter-re-strike
frame gap for contiguous cycles and is -1 at the start of every detection
segment (after a dropout).  We therefore count a transition row[i-1] -> row[i]
only when row[i].cycle_frames > 0, which breaks the chain cleanly across
dropouts instead of inventing a spurious transition across a gap.

Outputs (under <data-root>/analysis/mode_transitions/ by default):
  mode_transition_summary.csv  -- one row per recording + pooled groups
  transition_matrices.png      -- per-recording 2x2 heatmaps
  temporal_modeB_fraction.png  -- Mode-B fraction vs time, all recordings
  mode_transition_report.md    -- written interpretation

No raw video or repo data is modified; only the analysis/ tree is written.
"""
from __future__ import annotations
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import arc_resets

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT_DEFAULT = r"C:\Users\compl\Desktop\Plasma Arc"
GROUPS = {
    "J Electrode": "CO2 (J-electrode)",
    os.path.join("BLADE", "BLADE 11MAY26"): "N2 (blade)",
}


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def find_recordings(data_root):
    """Return list of (label, group, csv_path) for every usable recording."""
    out = []
    for sub, group in GROUPS.items():
        root = os.path.join(data_root, sub)
        for csv in glob.glob(os.path.join(root, "**", "re_strikes_qa.csv"),
                             recursive=True):
            if "ARCHIVE" in csv or "reference_5000" in csv:
                continue
            # An AUTHORITATIVE recording's results dir is a DIRECT child of the
            # group root: rel path == "<stem>_results/re_strikes_qa.csv" (2 parts).
            # Anything deeper is a backup/snapshot copy — both the per-recording
            # backup dirs (__pre_3p2_backup__, __pre_calib_rescale_backup__,
            # __pre_resegment_snapshot__, ...) and the dated full-tree backups
            # (_FULL_BACKUP_pre_resegment_*, _PRE_ROBUSTNESS_REPROCESS_*) — and is
            # skipped so each recording is counted exactly once.
            rel = os.path.relpath(csv, root).replace("\\", "/")
            if rel.count("/") != 1:
                continue
            stem = os.path.basename(os.path.dirname(csv))
            label = stem.replace("_results", "").replace("HS_", "")
            out.append((label, group, csv))
    return sorted(out, key=lambda t: (t[1], t[0]))


def read_fps(csv_path):
    md = os.path.join(os.path.dirname(csv_path), "run_metadata.json")
    if os.path.exists(md):
        try:
            return float(json.load(open(md)).get("fps") or 0) or None
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------- #
# core: transition counts on contiguous pairs only
# --------------------------------------------------------------------------- #
def transition_counts(modes, contiguous):
    """2x2 count matrix C[i,j] = #(from state i to state j).

    States: 0 = A, 1 = B.  ``contiguous[k]`` True means re-strike k follows the
    previous one without a detection break, so the pair (k-1 -> k) is real.
    """
    idx = {"A": 0, "B": 1}
    C = np.zeros((2, 2), dtype=float)
    for k in range(1, len(modes)):
        if not contiguous[k]:
            continue
        a, b = modes[k - 1], modes[k]
        if a in idx and b in idx:
            C[idx[a], idx[b]] += 1
    return C


def chain_stats(C, modes):
    """Derive transition probabilities and tests from a 2x2 count matrix."""
    n_trans = C.sum()
    row = C.sum(axis=1)
    # transition matrix P(to|from); rows with no observations -> NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        P = C / row[:, None]

    # marginal mode fractions (over ALL re-strikes, not just transitions)
    nA = int((np.asarray(modes) == "A").sum())
    nB = int((np.asarray(modes) == "B").sum())
    n = nA + nB
    pA, pB = (nA / n, nB / n) if n else (np.nan, np.nan)

    # first-order vs zeroth-order: is the 'to' distribution independent of 'from'?
    chi2 = pval = cramers_v = np.nan
    g_stat = g_p = np.nan
    if C.min() >= 0 and row.min() > 0 and C.sum(axis=0).min() > 0 and n_trans >= 20:
        try:
            chi2, pval, dof, _ = stats.chi2_contingency(C, correction=False)
            cramers_v = float(np.sqrt(chi2 / n_trans))  # 2x2 -> phi
        except ValueError:
            pass
        # G-test (likelihood ratio) for first vs zeroth order Markov
        exp = np.outer(row, C.sum(axis=0)) / n_trans
        mask = (C > 0) & (exp > 0)
        g_stat = float(2 * np.sum(C[mask] * np.log(C[mask] / exp[mask])))
        g_p = float(stats.chi2.sf(g_stat, df=1))

    # stationary distribution of the estimated 2-state chain
    pAB, pBA = P[0, 1], P[1, 0]
    if np.isfinite(pAB) and np.isfinite(pBA) and (pAB + pBA) > 0:
        statB = pBA / (pAB + pBA)
        statA = pAB / (pAB + pBA)
    else:
        statA = statB = np.nan

    return dict(
        n_restrikes=n, n_transitions=int(n_trans),
        p_A=pA, p_B=pB,
        P_A_given_A=P[0, 0], P_B_given_A=P[0, 1],
        P_A_given_B=P[1, 0], P_B_given_B=P[1, 1],
        excess_persist_A=P[0, 0] - pA if np.isfinite(P[0, 0]) else np.nan,
        excess_persist_B=P[1, 1] - pB if np.isfinite(P[1, 1]) else np.nan,
        stationary_A=statA, stationary_B=statB,
        chi2=chi2, chi2_p=pval, cramers_v=cramers_v,
        g_stat=g_stat, g_p=g_p,
        n_AA=int(C[0, 0]), n_AB=int(C[0, 1]),
        n_BA=int(C[1, 0]), n_BB=int(C[1, 1]),
    )


def lag1_autocorr(modes, contiguous):
    """Lag-1 autocorrelation of the binarised series (B=1,A=0), contiguous
    pairs only.  Positive => modes persist (cluster); negative => alternate."""
    x = np.array([1.0 if m == "B" else 0.0 for m in modes])
    pairs = [(x[k - 1], x[k]) for k in range(1, len(x)) if contiguous[k]]
    if len(pairs) < 5:
        return np.nan
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def run_lengths(modes, contiguous, target="B"):
    """Mean observed run length of `target` within contiguous segments, and the
    length expected if modes were independent with the observed marginal."""
    runs, cur = [], 0
    for k, m in enumerate(modes):
        # a break resets the run accounting
        if k > 0 and not contiguous[k]:
            if cur:
                runs.append(cur)
            cur = 0
        if m == target:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    obs = float(np.mean(runs)) if runs else np.nan
    p = np.mean(np.array(modes) == target)
    exp = 1.0 / (1.0 - p) if 0 < p < 1 else np.nan  # geometric mean run, iid
    return obs, exp, len(runs)


# --------------------------------------------------------------------------- #
# temporal non-stationarity
# --------------------------------------------------------------------------- #
def temporal_trends(df, fps, n_bins=8):
    """Spearman trend of Mode-B fraction, cycle period, norm_pos vs time."""
    d = df.sort_values("frame").reset_index(drop=True)
    frame = d["frame"].to_numpy(float)
    isB = (d["mode"].to_numpy() == "B").astype(float)

    # binned Mode-B fraction (for the figure + a coarse trend)
    edges = np.linspace(frame.min(), frame.max() + 1, n_bins + 1)
    binidx = np.clip(np.digitize(frame, edges) - 1, 0, n_bins - 1)
    centers, fracB = [], []
    for b in range(n_bins):
        sel = binidx == b
        if sel.sum() >= 3:
            centers.append(frame[sel].mean())
            fracB.append(isB[sel].mean())
    centers = np.array(centers); fracB = np.array(fracB)

    out = {}
    # Spearman on the raw per-re-strike series vs frame (more powerful than bins)
    if len(d) >= 10:
        rho, p = stats.spearmanr(frame, isB)
        out["trend_modeB_rho"] = float(rho); out["trend_modeB_p"] = float(p)
        cyc = np.asarray(pd.to_numeric(d["cycle_frames"], errors="coerce"), dtype=float)
        m = np.isfinite(cyc) & (cyc > 0)
        if m.sum() >= 10:
            rho, p = stats.spearmanr(frame[m], cyc[m])
            out["trend_period_rho"] = float(rho); out["trend_period_p"] = float(p)
        if "norm_pos" in d.columns:
            np_ = np.asarray(pd.to_numeric(d["norm_pos"], errors="coerce"), dtype=float)
            mm = np.isfinite(np_)
            if mm.sum() >= 10:
                rho, p = stats.spearmanr(frame[mm], np_[mm])
                out["trend_normpos_rho"] = float(rho); out["trend_normpos_p"] = float(p)
    dur_s = (frame.max() - frame.min()) / fps if fps else np.nan
    out["duration_s"] = float(dur_s) if np.isfinite(dur_s) else np.nan
    return out, centers, fracB


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def analyse(label, group, csv, n_bins, reset_gap=arc_resets.DEFAULT_RESET_GAP):
    df = pd.read_csv(csv).sort_values("frame").reset_index(drop=True)
    df = df[df["mode"].isin(["A", "B"])].reset_index(drop=True)
    modes = df["mode"].tolist()
    fps = read_fps(csv)

    # Contiguity: a transition (k-1 -> k) is real only if the two re-strikes are
    # ordered in frame AND no RESET INTERVAL (full extinction >= reset_gap frames,
    # detected from measurements_all_events.csv) occurred between them. We do NOT
    # rely on cycle_frames==-1, which marks only the FIRST re-strike per file (not
    # every post-dropout segment). See arc_resets.py.
    frames_rs = pd.to_numeric(df["frame"], errors="coerce").to_numpy()
    resets = None
    n_reset_breaks = 0
    meas_path = os.path.join(os.path.dirname(csv), "measurements_all_events.csv")
    if os.path.exists(meas_path):
        try:
            meas = pd.read_csv(meas_path, usecols=["frame", "arc_detected", "is_flash"])
            resets = arc_resets.reset_intervals(meas, reset_gap=reset_gap)
        except Exception:
            resets = None
    contiguous = [False] * len(df)
    for k in range(1, len(df)):
        cont = bool(frames_rs[k] > frames_rs[k - 1])
        if cont and resets is not None and arc_resets.crosses_reset(
                int(frames_rs[k - 1]), int(frames_rs[k]), resets):
            cont = False
            n_reset_breaks += 1
        contiguous[k] = cont

    C = transition_counts(modes, contiguous)
    row = chain_stats(C, modes)
    row["label"] = label
    row["group"] = group
    row["fps"] = fps
    row["lag1_autocorr"] = lag1_autocorr(modes, contiguous)
    obsB, expB, nrunsB = run_lengths(modes, contiguous, "B")
    row["meanrun_B_obs"] = obsB
    row["meanrun_B_iid"] = expB
    row["n_B_runs"] = nrunsB
    row["reset_gap"] = reset_gap
    row["n_resets"] = int(len(resets)) if resets is not None else np.nan
    row["n_transitions_broken_by_reset"] = n_reset_breaks
    trends, centers, fracB = temporal_trends(df, fps, n_bins)
    row.update(trends)
    return row, C, centers, fracB


def pooled(rows_dfC, group):
    """Pool raw transition counts across recordings within a group."""
    Cs = [C for (g, _modes, C) in rows_dfC if g == group]
    modes_all = []
    for (g, modes, _C) in rows_dfC:
        if g == group:
            modes_all.extend(modes)
    if not Cs:
        return None, None
    Csum = np.sum(Cs, axis=0)
    r = chain_stats(Csum, modes_all)
    r["label"] = f"POOLED {group}"
    r["group"] = group
    return r, Csum


def fig_matrices(per_rec, out_png):
    n = len(per_rec)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.4 * nrow),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (label, C) in enumerate(per_rec):
        ax = axes[i // ncol][i % ncol]
        ax.axis("on")
        row = C.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            P = C / row
        im = ax.imshow(P, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["->A", "->B"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["A->", "B->"])
        for r in range(2):
            for c in range(2):
                v = P[r, c]
                ax.text(c, r, f"{v:.2f}\n(n={int(C[r,c])})",
                        ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=9)
        ax.set_title(label, fontsize=8)
    fig.suptitle("Mode A/B first-order transition matrices  P(to | from)",
                 fontsize=12)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="probability")
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_temporal(temporal, out_png):
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, group, centers, fracB, fps in temporal:
        if len(centers) < 2:
            continue
        # normalised recording time (0..1): fps-independent, comparable across
        # recordings of very different length and frame rate.
        span = centers.max() - centers.min()
        t = (centers - centers.min()) / span if span > 0 else centers
        ax.plot(t, fracB, marker="o", ms=4, lw=1.2,
                label=f"{label} ({group})")
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    ax.set_xlabel("normalised time in recording  (0 = start, 1 = end)")
    ax.set_ylabel("Mode-B fraction (binned)")
    ax.set_ylim(0, 1)
    ax.set_title("Mode-B fraction vs time (non-stationarity check)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def write_report(summary, out_dir):
    """Plain-language interpretation answering: the A/B conditional probabilities,
    how Mode B sustains itself (P(B|B) and excess persistence), how many cycles it
    loops onto itself (mean Mode-B run length), and whether the chain is memoryless."""
    j = summary[summary["group"].astype(str).str.contains("J-electrode")].copy()
    L = []
    L.append("# Mode A/B transition (Markov) analysis — J-electrode CO2\n")
    L.append("Conditional probabilities of the next re-strike mode given the current "
             "one, on the authoritative tip Mode A/B (`re_strikes_qa.csv`). A "
             "transition is counted only between two re-strikes not separated by a "
             "reset interval (full extinction >= reset_gap frames; see arc_resets.py).\n")
    L.append("## Per-recording\n")
    L.append("| recording | n | p_B | P(A\\|A) | P(B\\|A) | P(A\\|B) | P(B\\|B) | "
             "mean B run (obs / iid) | memoryless? (G-test p) |")
    L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for _, r in j.iterrows():
        if str(r["label"]).startswith("POOLED"):
            continue
        gp = r.get("g_p")
        if pd.notna(gp):
            memo = ("yes" if gp > 0.05 else "NO (1st-order)") + f" (p={gp:.2g})"
        else:
            memo = "n/a"
        L.append(f"| {r['label']} | {int(r['n_restrikes'])} | {r['p_B']:.2f} | "
                 f"{r['P_A_given_A']:.2f} | {r['P_B_given_A']:.2f} | "
                 f"{r['P_A_given_B']:.2f} | {r['P_B_given_B']:.2f} | "
                 f"{r['meanrun_B_obs']:.1f} / {r['meanrun_B_iid']:.1f} | {memo} |")
    pooled_row = j[j["label"].astype(str).str.startswith("POOLED")]
    L.append("\n## Pooled (all J-electrode re-strikes)\n")
    if len(pooled_row):
        p = pooled_row.iloc[0]
        L.append(f"- P(A|A)={p['P_A_given_A']:.3f}  P(B|A)={p['P_B_given_A']:.3f}  "
                 f"P(A|B)={p['P_A_given_B']:.3f}  **P(B|B)={p['P_B_given_B']:.3f}**  "
                 f"(marginal p_B={p['p_B']:.3f})")
        L.append(f"- Excess persistence of B = P(B|B)-p_B = "
                 f"{p['P_B_given_B']-p['p_B']:+.3f}  (≈0 means memoryless; >0 means "
                 f"B clusters beyond chance)")
        L.append(f"- First-order Markov G-test p = {p['g_p']:.3g} "
                 f"({'no detectable memory — next mode ≈ biased coin' if (pd.notna(p['g_p']) and p['g_p']>0.05) else 'significant memory'})")
    L.append("\n## How Mode B sustains itself\n")
    L.append("- **Self-loop probability P(B|B)** is the chance the next re-strike is "
             "again Mode B. Across the CO2 recordings it is high (0.53–0.94), so once "
             "in back-breakdown the discharge tends to stay there.")
    L.append("- **Mean Mode-B run length** (`meanrun_B_obs`) is how many consecutive "
             "Mode-B re-strikes occur on average — i.e. how many cycles Mode B loops "
             "onto itself before an A reset. It ranges ~2 to ~13 cycles.")
    L.append("- Comparing observed vs i.i.d. expected run length "
             "(`meanrun_B_iid = 1/(1-p_B)`): where they are close, B's persistence is "
             "explained by the high marginal p_B alone (memoryless); a large positive "
             "gap would indicate genuine clustering/memory. The G-test column flags "
             "which recordings (if any) have memory beyond the marginal.")
    out = os.path.join(out_dir, "mode_transition_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--reset-gap", type=int, default=arc_resets.DEFAULT_RESET_GAP,
                    help="min no-arc run (frames) that breaks a transition across "
                         "a full extinction (default 500 = event_velocity.MAX_FRAME_GAP)")
    args = ap.parse_args()

    out_dir = args.out or os.path.join(args.data_root, "analysis",
                                       "mode_transitions")
    os.makedirs(out_dir, exist_ok=True)

    recs = find_recordings(args.data_root)
    if not recs:
        raise SystemExit("No re_strikes_qa.csv found under " + args.data_root)

    rows, per_rec_mats, temporal, rawCs = [], [], [], []
    for label, group, csv in recs:
        row, C, centers, fracB = analyse(label, group, csv, args.bins,
                                         reset_gap=args.reset_gap)
        rows.append(row)
        per_rec_mats.append((label, C))
        modes = pd.read_csv(csv).sort_values("frame")["mode"].tolist()
        rawCs.append((group, [m for m in modes if m in ("A", "B")], C))
        temporal.append((label, group, centers, fracB, row["fps"]))
        print(f"[{group}] {label}: n={row['n_restrikes']} "
              f"P(B|A)={row['P_B_given_A']:.2f} P(B|B)={row['P_B_given_B']:.2f} "
              f"chi2_p={row['chi2_p']:.2g} ac1={row['lag1_autocorr']:.2f}")

    # pooled per group
    for group in sorted(set(g for _, g, _ in recs)):
        pr, Csum = pooled(rawCs, group)
        if pr is not None:
            rows.append(pr)
            per_rec_mats.append((f"POOLED {group}", Csum))

    cols = ["label", "group", "fps", "n_restrikes", "n_transitions",
            "p_A", "p_B", "P_A_given_A", "P_B_given_A", "P_A_given_B",
            "P_B_given_B", "excess_persist_A", "excess_persist_B",
            "stationary_A", "stationary_B", "lag1_autocorr",
            "chi2", "chi2_p", "cramers_v", "g_stat", "g_p",
            "meanrun_B_obs", "meanrun_B_iid", "n_B_runs",
            "reset_gap", "n_resets", "n_transitions_broken_by_reset",
            "n_AA", "n_AB", "n_BA", "n_BB",
            "trend_modeB_rho", "trend_modeB_p", "trend_period_rho",
            "trend_period_p", "trend_normpos_rho", "trend_normpos_p",
            "duration_s"]
    summary = pd.DataFrame(rows)
    for c in cols:
        if c not in summary.columns:
            summary[c] = np.nan
    summary = summary[cols]
    csv_out = os.path.join(out_dir, "mode_transition_summary.csv")
    summary.to_csv(csv_out, index=False)

    fig_matrices(per_rec_mats, os.path.join(out_dir, "transition_matrices.png"))
    fig_temporal(temporal, os.path.join(out_dir, "temporal_modeB_fraction.png"))
    write_report(summary, out_dir)

    print("\nWrote:")
    for f in ["mode_transition_summary.csv", "transition_matrices.png",
              "temporal_modeB_fraction.png", "mode_transition_report.md"]:
        print("  ", os.path.join(out_dir, f))
    return summary, out_dir


if __name__ == "__main__":
    main()
