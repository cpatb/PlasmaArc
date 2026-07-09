#!/usr/bin/env python3
"""
flash_filter_qa.py  -  Verify / diagnose the width-based flash filter so you can
decide whether (and which leg) to loosen.

Background
----------
A frame is flagged is_flash if EITHER leg fires (arc_filter, criterion 3):
    eff_width  = area_px / arc_length_px  > FLASH_EFF_WIDTH (default 20)
                 -> a bright blob whose skeleton collapsed (length tiny vs area)
    arc_width  = arc_width_px (FWHM)       > FLASH_ARC_WIDTH (default 16)
                 -> a genuinely thick / motion-blurred channel
These target different failure modes, so they must be judged separately.

The key question for "am I smothering real signal?" is NOT how a frame looks but
whether its TIP carries coherent propagation signal.  This tool answers it with a
trajectory-coherence test: for every detected frame, fit the tip-depth of the
nearby CLEAN frames in the same event and measure the residual of the frame's own
tip.  A flagged frame whose tip lies ON the local clean trajectory is real
propagation signal (its skeleton may still be unusable for shape metrics); one
whose tip scatters off it is a true artifact.

Outputs (into --out-dir)
------------------------
  flash_legs_summary.txt          leg x depth breakdown + coherence test + a
                                  threshold sweep (what loosening each leg admits)
  flash_kymograph_by_leg.png      tip depth vs time, colored by which leg flagged
  flash_kymograph_coherence.png   flagged frames colored on-trajectory vs scatter
  flash_montage_<category>.png    sample frames (raw + binary contour + skeleton)
                                  for each category so you can eyeball blob vs arc

Usage
-----
    python flash_filter_qa.py \
        --results ".../_downward_results" \
        --tiff-dir ".../_downward_results/tiff" --frames-per-tiff 10000 \
        --arc-direction down --px-per-mm 8.8 --out-dir ".../flash_qa_down"

  --eff-width / --arc-width override the thresholds being TESTED (defaults pulled
  from arc_filter), so you can preview a looser cut before committing to it.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import tifffile

import arc_filter
from analyze_blade import binarize

FPS_DEFAULT = 200_000
COHERENCE_PX = 3.0       # tip within this many px of the local clean fit = on-trajectory
NEIGH_W = 8              # +/- frames of clean neighbours used for the local fit


def resolve_csv(p):
    p = Path(p)
    return p if p.suffix == ".csv" else p / "measurements_all_events.csv"


class FrameSource:
    """Fetch a raw frame by GLOBAL frame index from ordered chunk TIFFs."""
    def __init__(self, tiff_dir, frames_per_tiff):
        self.fpf = frames_per_tiff
        self.tiffs = sorted(Path(tiff_dir).glob("*.tiff"))
        self._cache = {}

    def _open(self, k):
        if k not in self._cache:
            self._cache[k] = tifffile.TiffFile(str(self.tiffs[k]))
        return self._cache[k]

    def get(self, gframe):
        if not self.tiffs or not self.fpf:
            return None
        k, page = divmod(int(gframe), self.fpf)
        if k < 0 or k >= len(self.tiffs):
            return None
        try:
            tf = self._open(k)
            return tf.pages[page].asarray() if 0 <= page < len(tf.pages) else None
        except Exception:
            return None

    def close(self):
        for t in self._cache.values():
            try:
                t.close()
            except Exception:
                pass


def tag_frames(df, eff_thr, aw_thr, px_per_mm):
    """Return a detected-frames DataFrame tagged with legs, depth and coherence."""
    for c in ["area_px", "arc_length_px", "arc_width_px", "arc_tip_row",
              "arc_detected", "event_id", "frame"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    det = df[(df["arc_detected"] == 1) & (df["event_id"] >= 0)] \
        .dropna(subset=["arc_tip_row"]).copy()
    det["eff"] = det["area_px"] / det["arc_length_px"].clip(lower=1)
    det["depth_mm"] = det["arc_tip_row"] / px_per_mm
    det["fl_eff"] = det["eff"] > eff_thr
    det["fl_aw"] = det["arc_width_px"] > aw_thr
    det["is_flash"] = det["fl_eff"] | det["fl_aw"]
    det["clean"] = ~det["is_flash"]

    # coherence: residual of each tip vs a local fit of CLEAN neighbours
    det = det.sort_values(["event_id", "frame"])
    resid = np.full(len(det), np.nan)
    pos = 0
    for _, g in det.groupby("event_id"):
        fr = g["frame"].values.astype(float)
        tp = g["arc_tip_row"].values.astype(float)
        clean = g["clean"].values
        n = len(g)
        for i in range(n):
            lo, hi = max(0, i - NEIGH_W), min(n, i + NEIGH_W + 1)
            idx = np.arange(lo, hi)
            idx = idx[clean[lo:hi] & (idx != i)]
            if len(idx) >= 5:
                sl, ic = np.polyfit(fr[idx], tp[idx], 1)
                resid[pos + i] = abs(tp[i] - (sl * fr[i] + ic))
        pos += n
    det["tip_residual_px"] = resid
    det["on_traj"] = det["tip_residual_px"] < COHERENCE_PX
    return det


def category(row):
    if not row["is_flash"]:
        return "clean"
    if row["fl_eff"] and row["fl_aw"]:
        return "both"
    if row["fl_eff"]:
        return "eff_only"
    return "aw_only"


def write_summary(det, eff_thr, aw_thr, path):
    lines = []
    n = len(det)
    fl = det["is_flash"]
    lines.append(f"Detected frames: {n}")
    lines.append(f"Thresholds tested: eff_width>{eff_thr}  arc_width(FWHM)>{aw_thr}")
    lines.append(f"is_flash: {int(fl.sum())} ({100*fl.mean():.1f}%)")
    lines.append(f"  eff_width leg only : {int((det['fl_eff'] & ~det['fl_aw']).sum())}")
    lines.append(f"  arc_width leg only : {int((det['fl_aw'] & ~det['fl_eff']).sum())}")
    lines.append(f"  both legs          : {int((det['fl_eff'] & det['fl_aw']).sum())}")
    lines.append("")
    # coherence per category
    lines.append("Tip-coherence (residual vs local CLEAN trajectory, px; on-traj < "
                 f"{COHERENCE_PX}px):")
    for cat in ["clean", "eff_only", "aw_only", "both"]:
        sub = det[det.apply(category, axis=1) == cat]
        r = sub["tip_residual_px"].dropna()
        if len(r):
            lines.append(f"  {cat:9s}: n={len(sub):6d}  median_resid={r.median():.1f}px  "
                         f"p90={r.quantile(.9):.1f}px  on-traj={100*sub['on_traj'].mean():.0f}%")
    lines.append("")
    lines.append("Interpretation:")
    lines.append("  on-traj flagged frames = real propagation signal (tip is real;")
    lines.append("    skeleton/shape may still be unusable -> keep out of SHAPE metrics,")
    lines.append("    but they are valid for VELOCITY/tip tracking).")
    lines.append("  scattered flagged frames = genuine artifacts -> keep filtered.")
    lines.append("")
    # threshold sweep: how many frames re-enter if each leg is loosened
    lines.append("Threshold sweep (frames that would RE-ENTER if a leg is loosened):")
    for thr in [eff_thr, 30, 50, 100, 1e9]:
        admitted = det[(det["eff"] <= thr) & det["fl_eff"] & ~det["fl_aw"]]
        on = 100 * admitted["on_traj"].mean() if len(admitted) else float("nan")
        tag = "OFF" if thr >= 1e9 else f"{thr:g}"
        lines.append(f"  eff_width cut -> {tag:>5s}: re-enters {len(admitted):6d} eff-only frames  "
                     f"({on:.0f}% on-traj)")
    for thr in [aw_thr, 20, 24, 30, 1e9]:
        admitted = det[(det["arc_width_px"] <= thr) & det["fl_aw"] & ~det["fl_eff"]]
        on = 100 * admitted["on_traj"].mean() if len(admitted) else float("nan")
        tag = "OFF" if thr >= 1e9 else f"{thr:g}"
        lines.append(f"  arc_width cut -> {tag:>5s}: re-enters {len(admitted):6d} aw-only frames  "
                     f"({on:.0f}% on-traj)")
    Path(path).write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nWrote {path}")


def kymograph_by_leg(det, fps, out):
    t = det["frame"] / fps * 1e3
    fig, ax = plt.subplots(figsize=(13, 5))
    cats = [("clean", "#9aa7b1", 2, "clean"),
            ("eff_only", "#FF8F00", 5, "eff_width (blob)"),
            ("aw_only", "#C62828", 7, "arc_width (thick)"),
            ("both", "#6A1B9A", 7, "both")]
    catcol = det.apply(category, axis=1)
    for key, c, s, lbl in cats:
        m = catcol == key
        ax.scatter(t[m], det["depth_mm"][m], s=s, c=c, alpha=0.5, linewidths=0, label=lbl)
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Tip depth (mm)")
    ax.set_title("Flash flags by leg")
    ax.legend(markerscale=2, framealpha=0.9, loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Wrote {out}")


def kymograph_coherence(det, fps, out):
    t = det["frame"] / fps * 1e3
    fig, ax = plt.subplots(figsize=(13, 5))
    cl = det[det["clean"]]
    ax.scatter(cl["frame"] / fps * 1e3, cl["depth_mm"], s=2, c="#cfd8dc",
               alpha=0.4, linewidths=0, label="clean")
    fl = det[det["is_flash"]]
    on = fl[fl["on_traj"]]; off = fl[~fl["on_traj"]]
    ax.scatter(on["frame"] / fps * 1e3, on["depth_mm"], s=6, c="#2E7D32",
               alpha=0.6, linewidths=0, label="flagged, ON trajectory (real signal)")
    ax.scatter(off["frame"] / fps * 1e3, off["depth_mm"], s=8, c="#D32F2F",
               alpha=0.7, linewidths=0, label="flagged, scattered (artifact)")
    ax.invert_yaxis()
    ax.set_xlabel("Time (ms)"); ax.set_ylabel("Tip depth (mm)")
    ax.set_title("Flagged frames: real propagation signal vs artifact")
    ax.legend(markerscale=2, framealpha=0.9, loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"Wrote {out}")


def montage(det, src, thresh, out_dir, cat, n=12):
    sub = det[det.apply(category, axis=1) == cat]
    if cat == "eff_only":  # split blob montage by coherence so the contrast is visible
        sub = pd.concat([sub[sub["on_traj"]].head(n // 2),
                         sub[~sub["on_traj"]].head(n // 2)])
    sub = sub.head(n)
    if len(sub) == 0:
        return
    cols = 4; rows = int(np.ceil(len(sub) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows),
                             facecolor="black", squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), sub.iterrows()):
        raw = src.get(int(r["frame"]))
        if raw is None:
            ax.text(0.5, 0.5, "frame %d\nn/a" % r["frame"], color="w", ha="center")
            continue
        ax.imshow(np.power(raw / max(raw.max(), 1), 0.45), cmap="inferno", vmin=0, vmax=1)
        b = binarize(raw, 3, bright_thresh=thresh)
        if b.any():
            ax.contour(b, levels=[0.5], colors="cyan", linewidths=0.6)
            sk = skeletonize(b)
            yy, xx = np.where(sk)
            ax.scatter(xx, yy, s=0.6, c="#76FF03", linewidths=0)
        tr = r["tip_residual_px"]
        ax.set_title("f%d d=%.0fmm A=%.0f L=%.0f\neff=%.0f w=%.1f res=%s"
                     % (r["frame"], r["depth_mm"], r["area_px"], r["arc_length_px"],
                        r["eff"], r["arc_width_px"],
                        ("%.1f" % tr) if np.isfinite(tr) else "n/a"),
                     color="w", fontsize=6)
    fig.suptitle("Montage: %s frames" % cat, color="w")
    fig.tight_layout()
    p = Path(out_dir) / ("flash_montage_%s.png" % cat)
    fig.savefig(p, dpi=130, facecolor="black"); plt.close(fig)
    print(f"Wrote {p}")


def main():
    ap = argparse.ArgumentParser(description="Flash-filter verification / diagnosis")
    ap.add_argument("--results", required=True, help="*_results dir or CSV")
    ap.add_argument("--tiff-dir", default=None, help="default: {results}/tiff")
    ap.add_argument("--frames-per-tiff", type=int, default=10000)
    ap.add_argument("--px-per-mm", type=float, default=8.8)
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT)
    ap.add_argument("--arc-direction", choices=["down", "up"], default="down")
    ap.add_argument("--thresh", type=int, default=100, help="binarization DN (for montage)")
    ap.add_argument("--eff-width", type=float, default=arc_filter.FLASH_EFF_WIDTH)
    ap.add_argument("--arc-width", type=float, default=arc_filter.FLASH_ARC_WIDTH)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-montage", action="store_true", help="skip the TIFF montages")
    args = ap.parse_args()

    res = Path(args.results)
    csv = resolve_csv(res)
    tiff_dir = Path(args.tiff_dir) if args.tiff_dir else res / "tiff"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv, low_memory=False)
    det = tag_frames(df, args.eff_width, args.arc_width, args.px_per_mm)

    write_summary(det, args.eff_width, args.arc_width, out / "flash_legs_summary.txt")
    kymograph_by_leg(det, args.fps, out / "flash_kymograph_by_leg.png")
    kymograph_coherence(det, args.fps, out / "flash_kymograph_coherence.png")

    if not args.no_montage and tiff_dir.exists():
        src = FrameSource(tiff_dir, args.frames_per_tiff)
        for cat in ["eff_only", "aw_only", "both"]:
            montage(det, src, args.thresh, out, cat)
        src.close()
    elif not args.no_montage:
        print("[skip] montage - tiff dir not found: %s" % tiff_dir)


if __name__ == "__main__":
    main()
