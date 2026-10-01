"""Per-recording kinematics of ONE BLADE-upwards recording (showcase_common.STEM), in the figure suite's style
(figures_tip/_common.style): what a single video yields. One figure per quantity, each as PDF + PNG 300 dpi +
plotted-data CSV + caption .md, written to <showcase>/plots.

Inputs are the run's own outputs (nothing recomputed from the cine):
  runs_annotated/.../measurements_all_events.csv   per frame: completeness, flash, arc length, tip row, local velocity
  tip_restrike_check/threshold/tip_restrike_events_kept.csv   kept leading-tip re-strikes (fall >= T_TIP)
  runs_annotated/.../velocity_map_local.npy, velocity_map_count.npy
  aggregate_annotated/per_recording.csv            this run's headline numbers, and every Argon run for context
Anchors (asserted): the per-frame table reproduces this run's per_recording.csv arc-length median, p90 tip height and
complete fraction, and the kept re-strike count of tip_restrikes_kept_by_run.csv.

usage: python make_plots.py
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import showcase_common as S
import _common as C

OUT = S.OUT / "plots"
TAG = "Ar, Re 8594, trial 1"
VBIN, VMAX = 0.1, 12.0          # m/s; the suite's 0.1 m/s bins, first bin centred on 0 (stalled frames)
BIN_PX = 5                      # arc length and tip height are whole pixels: bins of 5 px levels, else a comb
EDGES_H = np.arange(0.0, 22.01, 1.0)
MIN_FRAMES = 100                # per height bin (the suite's rule)
REST_BIN = 0.32                 # mm, the suite's re-strike height bin
NBLOCK = 10                     # stationarity blocks
MAP_BIN, MAP_MIN, MAP_PCT = 0.2, 20, 97
X_RANGE, Y_RANGE = (-12.0, 12.0), (-3.0, 26.0)
FPS = None                      # set from run_metadata.json in main()
BLUE, NAVY, TEAL = C.YLGNBU5[3], C.YLGNBU5[4], C.YLGNBU5[2]


def save(fig, slug, data, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    prov = f"validation_upwards/showcase/make_plots.py ({slug}); run {S.STEM}; git {C.git_hash()}"
    fig.savefig(OUT / f"{slug}.pdf", metadata={"Subject": prov, "CreationDate": None, "ModDate": None})
    fig.savefig(OUT / f"{slug}.png", dpi=300, metadata={"Software": None})
    plt.close(fig)
    with open(OUT / "data" / f"{slug}.csv", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# {prov}\n")
        data.to_csv(f, index=False, float_format="%.6g", lineterminator="\n")
    (OUT / f"{slug}.md").write_text(caption.strip() + f"\n\nProvenance: {prov}\n", encoding="utf-8")
    print(f"wrote {slug}")


def estimators(v, agg):
    nz = np.abs(v) > C.ZERO_EPS
    return {"median (all)": np.median(v), "median (moving)": np.median(v[nz]), "mean": np.mean(v),
            "per-event median (headline)": agg}


def fig_timeseries(d, s, E):
    fig, axs = plt.subplots(3, 1, figsize=(C.DOUBLE_W, 6.4), sharex=True)
    t = d["t_ms"]
    for ax, col, lab in zip(axs, ["h_mm", "L_mm", "v_ms"],
                            ["Tip Height (mm)", "Arc Length (mm)", "Tip Velocity (m/s)"]):
        ax.plot(t, d[col], ",", color=BLUE, rasterized=True)
        ax.axvspan(t.iloc[s], t.iloc[s + S.WIN - 1], color="0.85", zorder=0, lw=0)
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1.05 * np.nanmax(d[col]))
    axs[-1].set_xlabel("Time (ms)")
    axs[-1].set_xlim(0, t.iloc[-1])
    for ax, k in zip(axs, "abc"):
        C.panel(ax, k)
    C.tag(axs[0], TAG)
    fig.tight_layout()
    save(fig, "fig_run_timeseries", d[["frame", "t_ms", "h_mm", "L_mm", "v_ms"]],
         f"""# Whole-record time series

Every complete, non-flash frame of the {d['t_ms'].iloc[-1]:.1f} ms recording ({len(d)} frames at {FPS / 1e3:.0f} kfps): (a) height
of the leading tip above the minimum gap, (b) skeleton arc length, (c) local axial tip velocity (5-frame fit, advancing
front only, so it is blank between glides). Grey band: the 12 ms shown in the videos. Each sawtooth in (a) is one glide:
the tip climbs, then a re-strike drops it back to the gap or partway up the blades.""")


def fig_clip(d, s, E):
    w = d.iloc[s:s + S.WIN]
    t = w["t_ms"] - w["t_ms"].iloc[0]
    e = E[(E["frame"] >= s) & (E["frame"] < s + S.WIN)]
    fig, axs = plt.subplots(3, 1, figsize=(C.SINGLE_W, 6.4), sharex=True)
    for ax, col, lab in zip(axs, ["h_mm", "L_mm", "v_ms"],
                            ["Tip Height (mm)", "Arc Length (mm)", "Tip Velocity (m/s)"]):
        ax.plot(t, w[col], "o", ms=1.4, mfc=BLUE, mec="none", rasterized=True)
        for _, r in e.iterrows():
            tt = (r["frame"] - s) / FPS * 1e3
            ax.axvline(tt, color=NAVY if r["mode"] == "A" else TEAL, ls="--" if r["mode"] == "A" else "-", lw=0.8,
                       zorder=0)
        ax.set_ylabel(lab)
        ax.set_ylim(0, 1.08 * np.nanmax(w[col]))
    axs[-1].set_xlabel("Elapsed Time (ms)")
    axs[-1].set_xlim(0, S.WIN / FPS * 1e3)
    for ax, k in zip(axs, "abc"):
        C.panel(ax, k)
    axs[0].legend(handles=[Line2D([0], [0], color=NAVY, ls="--", lw=0.8, label="A (gap)"),
                           Line2D([0], [0], color=TEAL, lw=0.8, label="B (blades)")], title="Re-strike",
                  loc="upper center")
    fig.tight_layout()
    save(fig, "fig_run_clip", w.assign(t_clip_ms=t)[["frame", "t_clip_ms", "h_mm", "L_mm", "v_ms"]],
         f"""# The video clip

The 12 ms of the videos (frames {s}–{s + S.WIN - 1}, starting {w['t_ms'].iloc[0]:.2f} ms into the record):
(a) tip height above the gap, (b) arc length, (c) local tip velocity, complete non-flash frames only. Vertical lines:
kept leading-tip re-strikes (fall ≥ {S.t_tip_mm()} mm), dashed = Mode A (new tip below the {S.mode_valley_mm()} mm
valley, i.e. at the gap), solid = Mode B (partway up the blades). The window was chosen by a stated rule (most re-strikes
of the rarer mode), not by eye.""")


def fig_distributions(d, agg, m):
    v = d["v_ms"].dropna().to_numpy()
    v = np.where(np.abs(v) <= C.ZERO_EPS, 0.0, v)
    L, h = d["L_mm"].dropna().to_numpy(), d["h_mm"].dropna().to_numpy()
    fig, axs = plt.subplots(3, 1, figsize=(C.SINGLE_W, 7.0))
    vb = np.arange(-VBIN / 2, VMAX + VBIN, VBIN)
    cnt, _ = np.histogram(v, vb)
    axs[0].bar(vb[:-1], cnt / (len(v) * VBIN), width=VBIN, align="edge", color=C.COLOR["Argon"], edgecolor=C.EDGE,
               linewidth=0.3)
    est = estimators(v, agg)
    for (k, val), ls in zip(est.items(), ["-", "--", ":", "-."]):
        axs[0].axvline(val, color="black", ls=ls, lw=0.9, label=f"{k}: {val:.2f}")
    axs[0].legend(loc="upper right", fontsize=6.5, title="Estimator (m/s)")
    axs[0].set_xlabel("Local Tip Velocity (m/s)"), axs[0].set_ylabel("Density (s/m)")
    axs[0].set_xlim(-VBIN / 2, VMAX)
    rows = [pd.DataFrame({"quantity": "v_ms", "lo": vb[:-1], "hi": vb[1:], "count": cnt})]
    pxmm, H, gap_row = float(m["calibration"]["px_per_mm"]), int(m["H"]), float(m["calibration"]["gap_row"])
    for ax, x, off, lab, nm in ((axs[1], L, 0.0, "Arc Length (mm)", "L_mm"),
                                (axs[2], h, -(((H - 1) - gap_row) % 1.0), "Tip Height Above Gap (mm)", "h_mm")):
        # pixel levels sit at (k + off) / pxmm; edges half a level below each block of BIN_PX levels
        lev = x * pxmm - off
        if not np.allclose(lev, np.round(lev), atol=1e-6):
            raise SystemExit(f"{nm} is not on the pixel lattice")
        e = (off - 0.5 + BIN_PX * np.arange(0, np.ceil((lev.max() + 1) / BIN_PX) + 1)) / pxmm
        e = e[e > -BIN_PX / pxmm]
        b = BIN_PX / pxmm
        c, _ = np.histogram(x, e)
        ax.bar(e[:-1], c / (len(x) * b), width=b, align="edge", color=C.COLOR["Argon"], edgecolor=C.EDGE,
               linewidth=0.3)
        ax.axvline(np.median(x), color="black", lw=0.9, label=f"median {np.median(x):.2f}")
        if nm == "h_mm":
            ax.axvline(np.percentile(x, 90), color="black", ls="--", lw=0.9, label=f"p90 {np.percentile(x, 90):.2f}")
        ax.legend(loc="upper right")
        ax.set_xlabel(lab), ax.set_ylabel("Density (1/mm)")
        ax.set_xlim(max(0, e[0]), e[-1])
        rows.append(pd.DataFrame({"quantity": nm, "lo": e[:-1], "hi": e[1:], "count": c}))
    for ax, k in zip(axs, "abc"):
        C.panel(ax, k)
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    zero = float((v == 0).mean())
    n_over = int((v >= vb[-1]).sum())
    print(f"velocity frames beyond the {vb[-1]:.2f} m/s axis (not drawn, kept in the normalisation and the mean): {n_over}, max {v.max():.1f} m/s")
    save(fig, "fig_run_distributions", pd.concat(rows, ignore_index=True),
         f"""# Distributions over the whole recording

(a) Per-frame local tip velocity ({len(v)} advancing-front frames, {VBIN} m/s bins, first bin centred on 0; stalled
share {zero:.3f}), with the four estimators the suite compares; they span {min(est.values()):.2f}–{max(est.values()):.2f}
m/s here. {n_over} frames above {vb[-1]:.2f} m/s (max {v.max():.1f}) are beyond the axis: not drawn, but kept in the
normalisation and in the mean. The velocity is quantised by the whole-pixel tip position, so the bin-to-bin structure is not physical.
(b) Arc length and (c) tip height above the gap over {len(L)} complete non-flash frames (bins of {BIN_PX} whole-pixel levels, {BIN_PX / pxmm:.3f} mm), with the median (and p90 for the
height, the suite's tip-height statistic).""")


def fig_v_vs_h(d):
    x = d[d["v_ms"].notna()].copy()
    x["hb"] = pd.cut(x["h_mm"], EDGES_H)
    x["stall"] = x["v_ms"].abs() <= C.ZERO_EPS
    g = x.groupby("hb", observed=True)
    T = pd.DataFrame({"n": g.size(), "stalled_share": g["stall"].mean(),
                      "v_moving_median": x[~x["stall"]].groupby("hb", observed=True)["v_ms"].median(),
                      "v_moving_p25": x[~x["stall"]].groupby("hb", observed=True)["v_ms"].quantile(0.25),
                      "v_moving_p75": x[~x["stall"]].groupby("hb", observed=True)["v_ms"].quantile(0.75)})
    T = T[T["n"] >= MIN_FRAMES]
    T["h_mid_mm"] = [b.mid for b in T.index]
    fig, axs = plt.subplots(2, 1, figsize=(C.SINGLE_W, 4.8), sharex=True)
    axs[0].fill_between(T["h_mid_mm"], T["v_moving_p25"], T["v_moving_p75"], color=C.COLOR["Argon"],
                        edgecolor="0.55", lw=0.4)
    axs[0].plot(T["h_mid_mm"], T["v_moving_median"], "o-", color=BLUE, mec=C.EDGE, ms=3.5, mew=0.6,
                **C.outlined(1.1))
    axs[1].plot(T["h_mid_mm"], T["stalled_share"], "o-", color=BLUE, mec=C.EDGE, ms=3.5, mew=0.6, **C.outlined(1.1))
    axs[0].set_ylabel("Tip Velocity, Moving (m/s)"), axs[1].set_ylabel("Stalled Share")
    axs[1].set_xlabel("Tip Height Above Gap (mm)")
    axs[0].set_ylim(0, 1.05 * T["v_moving_p75"].max()), axs[1].set_ylim(0, 1)
    axs[1].set_xlim(0, EDGES_H[-1])
    for ax, k in zip(axs, "ab"):
        C.panel(ax, k)
    C.tag(axs[0], TAG)
    fig.tight_layout()
    save(fig, "fig_run_velocity_vs_height", T.reset_index(drop=True),
         f"""# Tip velocity against height

(a) Median local tip velocity over MOVING frames per 1 mm height bin (band: interquartile range of frames, not a
trial scatter: one recording). (b) Share of stalled frames (the whole-pixel tip did not move over the 5-frame fit),
shown separately as in the suite because folding it into the median manufactures a height trend. Bins with fewer than
{MIN_FRAMES} velocity frames are not drawn. Velocities exist only on advancing-front frames, and few are near the gap,
so the lowest bin is the floor of sampling, not the gap.""")


def fig_restrikes(E, d):
    fig, axs = plt.subplots(2, 1, figsize=(C.SINGLE_W, 4.8))
    hb = np.arange(0, np.ceil(E["tip_after_mm"].max() / REST_BIN) * REST_BIN + REST_BIN / 2, REST_BIN)
    c, _ = np.histogram(E["tip_after_mm"], hb)
    axs[0].bar(hb[:-1], c, width=REST_BIN, align="edge", color=C.COLOR["Argon"], edgecolor=C.EDGE, linewidth=0.5)
    axs[0].set_yscale("log"), axs[0].set_ylim(0.8, 2 * c.max())
    axs[0].set_xlabel("Re-strike Height Above Gap (mm)"), axs[0].set_ylabel("Count")
    axs[0].set_xlim(0, hb[-1])
    nA, nB = int((E["mode"] == "A").sum()), int((E["mode"] == "B").sum())
    C.tag(axs[0], f"N = {len(E)} (A {nA}, B {nB})")
    dt = np.diff(np.sort(E["frame"].to_numpy())) / FPS * 1e3
    ib = np.arange(0, np.ceil(dt.max() / 0.5) * 0.5 + 0.5, 0.5)
    ci, _ = np.histogram(dt, ib)
    axs[1].bar(ib[:-1], ci, width=0.5, align="edge", color=C.COLOR["Argon"], edgecolor=C.EDGE, linewidth=0.5)
    axs[1].axvline(np.mean(dt), color="black", lw=0.9, label=f"mean {np.mean(dt):.2f} ms")
    axs[1].axvline(np.median(dt), color="black", ls="--", lw=0.9, label=f"median {np.median(dt):.2f} ms")
    axs[1].legend(loc="upper right")
    axs[1].set_xlabel("Interval Between Counted Re-strikes (ms)"), axs[1].set_ylabel("Count")
    axs[1].set_xlim(0, ib[-1]), axs[1].set_ylim(bottom=0)
    for ax, k in zip(axs, "ab"):
        C.panel(ax, k)
    fig.tight_layout()
    data = pd.concat([pd.DataFrame({"quantity": "restrike_height_mm", "lo": hb[:-1], "hi": hb[1:], "count": c}),
                      pd.DataFrame({"quantity": "interval_ms", "lo": ib[:-1], "hi": ib[1:], "count": ci})])
    save(fig, "fig_run_restrikes", data,
         f"""# Re-strikes in this recording

(a) Height of the new leading tip above the gap at each kept re-strike (fall ≥ {S.t_tip_mm()} mm; {len(E)} events, log
counts, {REST_BIN} mm bins): one peak at the gap (Mode A, {nA}) and a spread partway up the blades (Mode B, {nB}), split
at the suite's {S.mode_valley_mm()} mm valley. These heights come from tip_restrike_check.py, which measures from the
gap row ROUNDED to a whole pixel, so they sit 0.04 mm below the h convention of the other figures (a fraction of a bin). (b) Time between consecutive counted re-strikes. **Not a cycle time:**
the blind audit showed the leading-tip detector misses re-strikes while the old arc still glows above the new one, so
counted intervals merge true cycles; the mean is an UPPER bound on the discharge time and the rate
({len(E) / (len(d) / FPS):.0f} /s) a LOWER bound (LEADING_TIP_FINDINGS.md).""")


def fig_stationarity(d):
    b = np.minimum((d["frame"] * NBLOCK) // len(d), NBLOCK - 1)
    g = d.groupby(b)
    T = pd.DataFrame({"t_mid_ms": g["t_ms"].mean(), "complete_frac": g["valid"].mean(),
                      "L_med_mm": g["L_mm"].median(), "h_p90_mm": g["h_mm"].quantile(0.9),
                      "v_mean_ms": g["v_ms"].mean()})
    fig, axs = plt.subplots(4, 1, figsize=(C.SINGLE_W, 6.6), sharex=True)
    for ax, col, lab in zip(axs, ["complete_frac", "L_med_mm", "h_p90_mm", "v_mean_ms"],
                            ["Complete Fraction", "Arc Length, Median (mm)", "Tip Height, p90 (mm)",
                             "Tip Velocity, Mean (m/s)"]):
        ax.plot(T["t_mid_ms"], T[col], "o-", color=BLUE, mec=C.EDGE, ms=4, mew=0.6, **C.outlined(1.1))
        ax.set_ylabel(lab, fontsize=8)
        ax.set_ylim(0, 1.15 * T[col].max())
    axs[-1].set_xlabel("Time (ms)")
    axs[-1].set_xlim(0, d["t_ms"].iloc[-1])
    for ax, k in zip(axs, "abcd"):
        C.panel(ax, k)
    fig.tight_layout()
    save(fig, "fig_run_stationarity", T.reset_index(names="block"),
         f"""# Stationarity within the recording

The record split into {NBLOCK} equal blocks of {len(d) // NBLOCK} frames: (a) fraction of frames with a complete
arc, (b) median arc length, (c) p90 tip height, (d) mean local tip velocity (the estimator least affected by the
pixel quantisation; stalled frames count as 0). A drift here would mean the whole-record numbers mix two states.""")


def fig_velocity_map(m):
    cal = m["calibration"]
    pxmm = float(cal["px_per_mm"])
    (ra, ca), (rb, cb) = cal["gap_ends"]
    r0, c0 = 0.5 * (ra + rb), 0.5 * (ca + cb)
    v = np.load(S.RUN / "velocity_map_local.npy") / 1000.0
    n = np.load(S.RUN / "velocity_map_count.npy")
    rr, cc = np.nonzero(n > 0)
    x, y, w = (cc - c0) / pxmm, (r0 - rr) / pxmm, n[rr, cc]
    ex = np.arange(X_RANGE[0], X_RANGE[1] + MAP_BIN / 2, MAP_BIN)
    ey = np.arange(Y_RANGE[0], Y_RANGE[1] + MAP_BIN / 2, MAP_BIN)
    s_, _, _ = np.histogram2d(y, x, [ey, ex], weights=v[rr, cc] * w)
    c_, _, _ = np.histogram2d(y, x, [ey, ex], weights=w)
    V = np.where(c_ >= MAP_MIN, s_ / np.maximum(c_, 1), np.nan)
    vmax = float(np.nanpercentile(V, MAP_PCT))
    fig, ax = plt.subplots(figsize=(C.SINGLE_W, 3.9))
    ax.set_facecolor(C.MAP_BG)
    im = ax.imshow(V, origin="lower", extent=[ex[0], ex[-1], ey[0], ey[-1]], cmap="plasma", vmin=0, vmax=vmax,
                   interpolation="nearest")
    z = np.load(S.RUN / "calibration" / f"{S.STEM}_edges_annotated.npz")
    for k in ("A", "B"):
        e = z[k][[0, -1]]
        ax.plot((e[:, 1] - c0) / pxmm, (r0 - e[:, 0]) / pxmm, color="white", lw=1.0)
    ax.axhline(0, color="0.75", lw=0.5, ls=":")
    ax.set_xlim(*X_RANGE), ax.set_ylim(*Y_RANGE), ax.set_aspect("equal")
    ax.set_xlabel("x (mm)"), ax.set_ylabel("Height Above Gap (mm)")
    ax.tick_params(which="both", color="white")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label("Tip Velocity (m/s)")
    C.panel(ax, "a", color="white")
    fig.tight_layout()
    iy, ix = np.nonzero(np.isfinite(V))
    save(fig, "fig_run_velocity_map", pd.DataFrame({"x_mm": 0.5 * (ex[ix] + ex[ix + 1]),
                                                    "y_mm": 0.5 * (ey[iy] + ey[iy + 1]), "v_ms": V[iy, ix],
                                                    "frames": c_[iy, ix]}),
         f"""# Velocity map

Mean leading-tip velocity over the frames in which the arc covered each point ({MAP_BIN} mm bins, coordinates from the
hand-annotated gap centre, white lines: the electrode edges, dotted: the gap height). The value at a point is the TIP
velocity of those frames (the J-suite definition), not the local speed of that piece of arc; stalled frames count as 0.
Dark: fewer than {MAP_MIN} frames. Colour 0 to the {MAP_PCT}th percentile ({vmax:.2f} m/s).""")


def fig_vs_condition():
    A = C.per_run()                                   # the suite's own gates
    A = A[A["gas"] == "Argon"]
    me = A[A["stem"] == S.STEM]
    cols = [("local_vel_ms_med", "Tip Velocity (m/s)"), ("arc_len_mm_med", "Arc Length (mm)"),
            ("tip_height_above_gap_mm_p90", "Tip Height, p90 (mm)")]
    fig, axs = plt.subplots(3, 1, figsize=(C.SINGLE_W, 6.4), sharex=True)
    for ax, (col, lab) in zip(axs, cols):
        ax.plot(A["Re"], A[col], "o", ms=3.5, mfc="none", mec="0.55", mew=0.7, ls="none", label="Argon trials")
        c = C.by_condition(A, col)
        ax.errorbar(c["Re"], c["mean"], yerr=c["sd"], label="mean ± sd", **C.gas_err("Argon"))
        ax.plot(me["Re"], me[col], "*", ms=11, mfc="#E31A1C", mec=C.EDGE, mew=0.7, ls="none", label="this video",
                zorder=5)
        ax.set_ylabel(lab), ax.set_ylim(bottom=0)
    axs[-1].set_xlabel("Re"), axs[-1].set_xlim(0, 9000)
    axs[0].legend(loc="upper left", bbox_to_anchor=(0.08, 1.0))
    for ax, k in zip(axs, "abc"):
        C.panel(ax, k)
    fig.tight_layout()
    save(fig, "fig_run_vs_condition", A[["stem", "Re", "trial"] + [c for c, _ in cols]],
         """# This video against the Argon campaign

Headline per-recording numbers of the suite (aggregate_annotated/per_recording.csv) for every Argon recording (grey
open: single trials; filled: mean ± sd over the 3 trials of each Re), with this video starred. (a) Tip velocity, the
per-event-median headline estimator (at low Re the estimators disagree, see fig_tip_kinematics); (b) median arc length
over complete arcs; (c) p90 tip height above the gap.""")


def main():
    C.style()
    global FPS
    m = S.meta()
    FPS = float(m["fps"])
    d = S.frames_table(m)
    E = S.restrikes()
    s = S.pick_window(d, E)
    F = pd.read_csv(C.ROOT / "aggregate_annotated" / "per_recording.csv")
    r = F[F["stem"] == S.STEM].iloc[0]
    K = pd.read_csv(S.THR / "tip_restrikes_kept_by_run.csv").set_index("run").loc[S.STEM]
    # anchors: this table reproduces the run's published per-run numbers
    chk = {"arc_len_mm_med": (np.nanmedian(d["L_mm"]), r["arc_len_mm_med"]),
           "tip_height_above_gap_mm_p90": (np.nanpercentile(d["h_mm"], 90), r["tip_height_above_gap_mm_p90"]),
           "frac_complete": ((d["arc_complete"] == 1).mean(), r["frac_complete"]),
           "n_kept_restrikes": (len(E), K["n_kept"])}
    for k, (a, b) in chk.items():
        print(f"anchor {k}: here {a:.10g}, published {b:.10g}")
        if not np.isclose(a, b, rtol=0, atol=1e-9):
            raise SystemExit(f"anchor {k} failed")
    fig_timeseries(d, s, E)
    fig_clip(d, s, E)
    fig_distributions(d, float(r["local_vel_ms_med"]), m)
    fig_v_vs_h(d)
    fig_restrikes(E, d)
    fig_stationarity(d)
    fig_velocity_map(m)
    fig_vs_condition()
    json.dump({k: [float(a), float(b)] for k, (a, b) in chk.items()}, open(OUT / "anchors.json", "w"), indent=1)


if __name__ == "__main__":
    main()
