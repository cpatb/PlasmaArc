#!/usr/bin/env python3
"""
overlay_shape_metrics.py  —  Annotated sample frames showing how the per-frame
shape metrics (arc length, tortuosity, and the length/tortuosity asymmetry
indices) are measured on the real video frames.

For each selected frame it re-runs ``analyze_blade.measure_frame`` (the exact
detection used by the pipeline, keep_overlays=True) and draws, on top of the raw
grayscale frame:

  * orange line       — the ordered arc SKELETON (the path whose length feeds
                        arc_length_mm and the numerator of tortuosity)
  * white dashed line — root-to-root CHORD (electrode gap; denominator of the
                        whole-arc tortuosity = path / chord)
  * green dashed line — electrode MIDLINE (bisector of the two blades) that
                        splits the arc into a blade-A half and a blade-B half
  * yellow X          — where the arc crosses the midline (the apex split point)
  * blue / red dots   — the two arc roots (blade-A root / blade-B root)
  * blue/red dashed   — the per-half chords (root -> crossing) used for the
                        per-half tortuosities tort_A, tort_B
  * cyan circle       — arc front tip

A text box reports arc_length_mm, tortuosity, tort_A, tort_B, arclen_asym and
tort_asym for that frame.

Outputs (into --out-dir):
  overlay_anatomy.png            single fully-labelled frame + legend
  overlay_tortuosity_series.png  4 frames spanning low -> high tortuosity
  overlay_asymmetry_series.png   4 frames spanning blade-B side -> blade-A side

Usage:
    python overlay_shape_metrics.py \
        --results ".../_downward_results" \
        --out-dir ".../shape_overlays" \
        --direction down --gap-row 26 --px-per-mm 8.8
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import analyze_blade as ab
import arc_filter

SKEL_C = "#FF9800"   # skeleton path
CHORD_C = "white"    # root-root chord
MID_C = "#00E676"    # electrode midline
CROSS_C = "yellow"   # midline crossing
AROOT_C = "#2979FF"  # blade-A root / A-half chord
BROOT_C = "#FF1744"  # blade-B root / B-half chord
TIP_C = "cyan"       # arc front tip


def resolve_csv(p):
    p = Path(p)
    return p if p.suffix == ".csv" else p / "measurements_all_events.csv"


def detect_stem(tiff_dir):
    tiffs = sorted(Path(tiff_dir).glob("*.tiff"))
    if not tiffs:
        raise FileNotFoundError("No TIFFs in %s" % tiff_dir)
    name = tiffs[0].stem
    parts = name.rsplit("_", 1)
    return parts[0] if (len(parts) == 2 and parts[1].isdigit()) else name


def load_frame(tiff_dir, stem, global_frame, per_tiff):
    chunk = global_frame // per_tiff
    page = global_frame % per_tiff
    path = Path(tiff_dir) / ("%s_%04d.tiff" % (stem, chunk + 1))
    if not path.exists():
        return None
    with tifffile.TiffFile(str(path)) as tf:
        if page >= len(tf.pages):
            return None
        return tf.pages[page].asarray()


def midline_endpoints(midline, H, W):
    """Return two points spanning the frame along the midline direction."""
    if midline is None:
        return None
    M0, u, _ = midline
    M0 = np.asarray(M0, float); u = np.asarray(u, float)
    t = np.linspace(-max(H, W), max(H, W), 2)
    pts = np.array([M0 + tt * u for tt in t])
    return pts  # rows in [:,0], cols in [:,1]


def draw_overlay(ax, raw, m, midline, show_text=True, fontsize=7):
    H, W = raw.shape
    ax.imshow(raw, cmap="gray", aspect="equal")
    ax.set_xticks([]); ax.set_yticks([])

    coords = m.get("_skel_coords", np.empty((0, 2)))
    ta, tb = m.get("tip_a"), m.get("tip_b")

    # electrode midline (green dashed)
    mp = midline_endpoints(midline, H, W)
    if mp is not None:
        ax.plot(mp[:, 1], mp[:, 0], ls="--", lw=1.0, color=MID_C, alpha=0.9, zorder=2)

    # root-root chord (white dashed) — denominator of whole-arc tortuosity
    if ta is not None and tb is not None:
        ax.plot([ta[1], tb[1]], [ta[0], tb[0]], ls="--", lw=1.2,
                color=CHORD_C, alpha=0.9, zorder=3)

    # per-half chords root -> crossing (blue/red dashed)
    xr, xc = m.get("mid_cross_row"), m.get("mid_cross_col")
    if ta is not None and np.isfinite(xr):
        ax.plot([ta[1], xc], [ta[0], xr], ls=":", lw=1.1, color=AROOT_C, zorder=3)
    if tb is not None and np.isfinite(xr):
        ax.plot([tb[1], xc], [tb[0], xr], ls=":", lw=1.1, color=BROOT_C, zorder=3)

    # arc skeleton path (orange)
    if len(coords):
        ax.plot(coords[:, 1], coords[:, 0], "-", lw=1.6, color=SKEL_C, zorder=4)

    # crossing point (yellow X)
    if np.isfinite(xr):
        ax.plot(xc, xr, "x", color=CROSS_C, ms=8, mew=2.0, zorder=6)

    # roots
    if ta is not None:
        ax.plot(ta[1], ta[0], "o", color=AROOT_C, ms=6, mec="white", mew=0.8, zorder=6)
    if tb is not None:
        ax.plot(tb[1], tb[0], "o", color=BROOT_C, ms=6, mec="white", mew=0.8, zorder=6)

    # arc front tip
    tr, tc = m.get("arc_tip_row"), m.get("arc_tip_col")
    if np.isfinite(tr) and np.isfinite(tc):
        ax.plot(tc, tr, "o", color=TIP_C, ms=8, mfc="none", mew=1.5, zorder=6)

    if show_text:
        txt = ("arc len = %.2f mm\n"
               "tortuosity = %.2f\n"
               "  tort_A = %.2f | tort_B = %.2f\n"
               "arclen_asym = %+.2f\n"
               "tort_asym = %+.2f"
               % (m.get("arc_length_mm", np.nan), m.get("tortuosity", np.nan),
                  m.get("tort_a", np.nan), m.get("tort_b", np.nan),
                  m.get("arclen_asym", np.nan), m.get("tort_asym", np.nan)))
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=fontsize,
                va="top", ha="left", color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="none", alpha=0.55))

    # zoom to the arc bbox with margin (frames are 256x256 but arc is small)
    if len(coords):
        r0, c0 = coords.min(0); r1, c1 = coords.max(0)
        pad = 22
        ax.set_xlim(max(0, c0 - pad), min(W, c1 + pad))
        ax.set_ylim(min(H, r1 + pad), max(0, r0 - pad))  # inverted y (image)


def measure(raw, bl, px_per_mm, direction, gap_row):
    return ab.measure_frame(raw, close_r=3, px_per_mm=px_per_mm, top_fraction=0.25,
                            keep_overlays=True, blade_lines=bl,
                            arc_direction=direction, gap_row=gap_row)


def is_valid(m):
    """Accept only cleanly-formed arcs (re-measurement reproduces a real arc)."""
    coords = m.get("_skel_coords", np.empty((0, 2)))
    return (m.get("arc_complete") and len(coords) >= 14
            and np.isfinite(m.get("tortuosity", np.nan))
            and 1.0 <= m["tortuosity"] <= 2.8
            and np.isfinite(m.get("mid_cross_row", np.nan))
            and np.isfinite(m.get("arclen_asym", np.nan))
            and np.isfinite(m.get("tort_asym", np.nan))
            and 3.0 <= m.get("arc_length_mm", 0) <= 28.0)


def build_validated_cache(ok, tiff_dir, stem, per_tiff, bl, px_per_mm,
                          direction, gap_row, n_sample=360):
    """Re-measure an evenly-spaced sample of usable frames; keep only the clean
    ones. Returns a DataFrame of re-measured metrics plus a {frame: (raw, m)}
    cache so frames are never measured twice."""
    frames = ok.sort_values("frame")["frame"].to_numpy()
    if len(frames) > n_sample:
        frames = frames[np.round(np.linspace(0, len(frames) - 1, n_sample)).astype(int)]
    cache, rows = {}, []
    for fr in frames:
        raw = load_frame(tiff_dir, stem, int(fr), per_tiff)
        if raw is None:
            continue
        m = measure(raw, bl, px_per_mm, direction, gap_row)
        if not is_valid(m):
            continue
        cache[int(fr)] = (raw, m)
        rows.append(dict(frame=int(fr), tortuosity=m["tortuosity"],
                         arclen_asym=m["arclen_asym"], tort_asym=m["tort_asym"],
                         arc_length_mm=m["arc_length_mm"]))
    return pd.DataFrame(rows), cache


def pick_by(meas, column, targets):
    """Pick frames whose (re-measured) `column` is nearest each percentile target."""
    sub = meas.copy(); chosen = []
    for q in targets:
        v = np.percentile(sub[column], q)
        idx = (sub[column] - v).abs().idxmin()
        chosen.append(int(sub.loc[idx, "frame"]))
        sub = sub.drop(index=idx)
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="*_results dir (or CSV path)")
    ap.add_argument("--tiff-dir", default=None, help="default: {results}/tiff")
    ap.add_argument("--blade-lines", default=None,
                    help="default: {results}/blade_lines.npy")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--direction", choices=["down", "up"], default="down")
    ap.add_argument("--gap-row", type=int, default=None)
    ap.add_argument("--px-per-mm", type=float, default=8.8)
    ap.add_argument("--per-tiff", type=int, default=10000)
    args = ap.parse_args()

    res = Path(args.results)
    csv = resolve_csv(res)
    tiff_dir = Path(args.tiff_dir) if args.tiff_dir else res / "tiff"
    bl_path = Path(args.blade_lines) if args.blade_lines else res / "blade_lines.npy"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    bl = np.load(bl_path, allow_pickle=True) if bl_path.exists() else None
    stem = detect_stem(tiff_dir)

    df = pd.read_csv(csv)
    df = arc_filter.add_flash_flag(df)
    for c in ["tortuosity", "arclen_asym", "tort_asym", "arc_length_mm"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ok = df[(df["arc_detected"] == 1) & (~df["is_flash"])
            & df["tortuosity"].notna() & df["arclen_asym"].notna()
            & (df["arc_length_mm"] > 3.0)].copy()
    print("usable frames (CSV):", len(ok))

    # Re-measure a sample and keep only clean arcs; select from the VALIDATED set
    # so we never illustrate a degenerate/broken detection.
    meas, cache = build_validated_cache(ok, tiff_dir, stem, args.per_tiff, bl,
                                        args.px_per_mm, args.direction, args.gap_row)
    print("validated clean frames:", len(meas))
    midline = ab.midline_from_blades(bl)

    def get(fr):
        return cache[fr]  # (raw, m)

    # ---- 1) anatomy figure: one clean, clearly asymmetric & tortuous frame ----
    cand = meas[(meas["tortuosity"].between(1.3, 1.9))
                & (meas["arclen_asym"].abs().between(0.18, 0.6))]
    if not len(cand):
        cand = meas
    anat_frame = int(cand.sort_values("arc_length_mm").iloc[len(cand) // 2]["frame"])
    raw, m = get(anat_frame)

    fig, ax = plt.subplots(figsize=(7.5, 6.2), facecolor="white")
    draw_overlay(ax, raw, m, midline, show_text=True, fontsize=9)
    ax.set_title("Shape-metric overlay anatomy  (frame %d, %sward)"
                 % (anat_frame, args.direction), fontsize=11)
    legend = [
        Line2D([0], [0], color=SKEL_C, lw=2, label="arc skeleton (path length)"),
        Line2D([0], [0], color=CHORD_C, lw=2, ls="--",
               label="root-root chord (tortuosity = path / chord)"),
        Line2D([0], [0], color=MID_C, lw=2, ls="--", label="electrode midline (A/B split)"),
        Line2D([0], [0], marker="x", color=CROSS_C, lw=0, mew=2, ms=8,
               label="midline crossing (apex split)"),
        Line2D([0], [0], marker="o", color=AROOT_C, lw=0, ms=7, label="blade-A root"),
        Line2D([0], [0], marker="o", color=BROOT_C, lw=0, ms=7, label="blade-B root"),
        Line2D([0], [0], color=AROOT_C, lw=1.2, ls=":", label="A-half chord (tort_A)"),
        Line2D([0], [0], color=BROOT_C, lw=1.2, ls=":", label="B-half chord (tort_B)"),
        Line2D([0], [0], marker="o", color=TIP_C, lw=0, mfc="none", mew=1.5, ms=8,
               label="arc front tip"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=7.5, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out / "overlay_anatomy.png", dpi=170, facecolor="white")
    plt.close(fig)
    print("saved overlay_anatomy.png (frame %d)" % anat_frame)

    # ---- 2) tortuosity series: low -> high ----
    tort_frames = pick_by(meas, "tortuosity", [8, 45, 80, 97])
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), facecolor="white")
    for ax, fr in zip(axes, tort_frames):
        raw, mm = get(fr)
        draw_overlay(ax, raw, mm, midline, show_text=True)
        ax.set_title("frame %d   tortuosity=%.2f" % (fr, mm.get("tortuosity", np.nan)),
                     fontsize=9)
    fig.suptitle("Tortuosity = skeleton path length / root-root chord  (low -> high)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "overlay_tortuosity_series.png", dpi=160, facecolor="white")
    plt.close(fig)
    print("saved overlay_tortuosity_series.png", tort_frames)

    # ---- 3) asymmetry series: blade-B side -> blade-A side ----
    def pick_near_vals(meas, column, vals):
        sub = meas.copy(); chosen = []
        for v in vals:
            idx = (sub[column] - v).abs().idxmin()
            chosen.append(int(sub.loc[idx, "frame"])); sub = sub.drop(index=idx)
        return chosen
    asym_frames = pick_near_vals(meas, "arclen_asym", [-0.35, -0.12, 0.12, 0.35])
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), facecolor="white")
    for ax, fr in zip(axes, asym_frames):
        raw, mm = get(fr)
        draw_overlay(ax, raw, mm, midline, show_text=True)
        ax.set_title("frame %d   arclen_asym=%+.2f" % (fr, mm.get("arclen_asym", np.nan)),
                     fontsize=9)
    fig.suptitle("Arc-length asymmetry  (- = more length on blade-B side  ->  "
                 "+ = more on blade-A side)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out / "overlay_asymmetry_series.png", dpi=160, facecolor="white")
    plt.close(fig)
    print("saved overlay_asymmetry_series.png", asym_frames)


if __name__ == "__main__":
    main()
