#!/usr/bin/env python3
r"""build_velocity_field.py — arrival-time (eikonal) front-speed velocity FIELD on
the arc CHANNEL. A physically faithful complement to build_velocity_map.py.

NEW, SEPARATE analysis. It does NOT replace or modify velocity_map_local.npy (the
map that goes into the paper). It writes ONLY into <results>/velocity_field_eikonal/
and refuses (asserts) to write any of the protected legacy filenames.

Method
------
A (arrival-time / eikonal): for each discharge event build T(x) = first time the arc
reaches pixel x. The local front-propagation speed is v(x) = 1/|grad T| and the
direction is grad T/|grad T| (the front travels toward LATER arrival). This is a true
per-pixel velocity FIELD of the moving front — unlike velocity_map_local.npy, which
stamps the single tip-speed scalar onto the whole luminous arc body.

B (channel, not plume): T is accumulated on the skeleton (medial axis) of the
binarized arc, dilated CHANNEL_R px and clipped to the mask -> a thin channel band, so
the bright entrained CO2 plume is excluded and we measure the conducting channel's
advance, not the luminous envelope.

Refinements (validated on Re 6478):
  * FRONTIER_TRIM: erode each event's swept region so the ill-defined outward gradient
    at the last-arrived pixels (where 1/|grad T| blows up) is discarded.
  * MIN_COVERAGE: keep only pixels seen by >= this many events — an evidence floor that
    also removes the residual low-N frontier band. Dropped pixels are logged, not
    silently capped.
  * nan-aware Gaussian smoothing of T (quantised in frames) before differentiating.

Binarisation matches analyze_blade exactly, using the per-recording params from
run_metadata.json (bright_thresh, min/max_arc_px); frame->chunk mapping is derived from
the actual tiff page counts (no frames-per-tiff assumption).

Outputs (in <results>/velocity_field_eikonal/):
    velocity_field_speed_ms.npy   per-pixel front speed (m/s), NaN where unpopulated
    velocity_field_vx_ms.npy      lateral velocity component (m/s)
    velocity_field_vy_ms.npy      axial component (+row = away from gap)
    velocity_field_count.npy      events contributing per pixel (evidence map)
    velocity_field.png            current map | eikonal field+quiver | coverage

Usage:
    python build_velocity_field.py --results ".../HS_0.4mm_1000mL_(CO2)_trig_results"
    python build_velocity_field.py --results A B C D \
        --labels "Re 9717" "Re 6478" "Re 4859" "Re 3239" \
        --montage-out ".../velocity_field_eikonal_montage/montage.png"
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from skimage.morphology import skeletonize, disk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_blade import binarize

OUT_SUBDIR = "velocity_field_eikonal"
# Legacy artefacts that must never be written by this script (paper-bound old method).
PROTECTED = {"velocity_map_local.npy", "velocity_map_local.png",
             "velocity_map.npy", "velocity_map_comparison.png"}

# ---- method constants (named, documented) ----
CHANNEL_R     = 2                 # skeleton dilation radius (px) -> channel band half-width
FRONTIER_TRIM = CHANNEL_R + 2     # px eroded off each event's swept region (frontier fix)
SMOOTH_SIGMA  = 2.0               # px, nan-aware Gaussian smoothing of T before grad
MIN_FRAMES    = 8                 # event needs >= this many detected, non-flash frames
MIN_EXT_PX    = 15                # ... and tip must extend >= this (real propagation)
SPEED_CEIL    = 60.0             # m/s; per-event mask above this (unreliable)
MIN_COVERAGE  = 3                 # keep only pixels seen by >= this many events
EPS_GRAD      = 1e-9
DEFAULT_CLOSE_R = 3               # matches build_velocity_map.py's --close-radius default


def load_params(results):
    g = json.load(open(os.path.join(results, "run_metadata.json")))
    return dict(px_per_mm=float(g["px_per_mm"]), H=int(g["height"]), W=int(g["width"]),
                fps=float(g["fps"]), bright_thresh=int(g["bright_thresh"]),
                min_arc_px=int(g["min_arc_px"]), max_arc_px=int(g["max_arc_px"]),
                n_frames=int(g.get("n_frames", -1)))


def build_chunks(tiff_dir, n_frames_expected):
    """Open ordered chunk TIFFs; return (list[TiffFile], offsets) where offsets[i] is
    the global frame index of chunk i's page 0. Frame->chunk mapping is derived from
    real page counts, so no frames-per-tiff assumption is made."""
    files = sorted(os.path.join(tiff_dir, f) for f in os.listdir(tiff_dir)
                   if f.endswith(".tiff"))
    if not files:
        raise SystemExit("no *.tiff in %s" % tiff_dir)
    tfs = [tifffile.TiffFile(p) for p in files]
    counts = [len(t.pages) for t in tfs]
    offsets = list(np.cumsum([0] + counts))
    if n_frames_expected > 0 and offsets[-1] != n_frames_expected:
        print("  [WARN] tiff pages (%d) != run_metadata n_frames (%d)"
              % (offsets[-1], n_frames_expected))
    return tfs, offsets


def page_for(tfs, offsets, gframe):
    i = int(np.searchsorted(offsets, gframe, side="right") - 1)
    if i < 0 or i >= len(tfs):
        return None
    pi = gframe - offsets[i]
    if pi < 0 or pi >= len(tfs[i].pages):
        return None
    return tfs[i].pages[pi].asarray()


def channel_band(raw, p, close_r):
    """Method B: skeleton of the arc mask, dilated to a thin band, clipped to mask."""
    b = binarize(raw, close_r, bright_thresh=p["bright_thresh"],
                 min_arc_px=p["min_arc_px"], max_arc_px=p["max_arc_px"])
    if not b.any():
        return None
    sk = skeletonize(b)
    return binary_dilation(sk, disk(CHANNEL_R)) & b


def nan_gaussian(T, valid, sigma):
    """Normalized (nan-aware) Gaussian smoothing: G(T*valid)/G(valid)."""
    num = gaussian_filter(np.where(valid, T, 0.0), sigma)
    den = gaussian_filter(valid.astype(float), sigma)
    ok = den > 1e-3
    out = np.full_like(T, np.nan)
    out[ok] = num[ok] / den[ok]
    return out, ok


def event_field(T, px_per_mm):
    """From an event arrival-time field T (s; NaN outside), return per-pixel
    (speed_ms, vx_ms, vy_ms, mask). Speed = 1/|grad T| in the +grad T direction."""
    valid = np.isfinite(T)
    Ts, ok = nan_gaussian(T, valid, SMOOTH_SIGMA)
    grad_ok = binary_erosion(ok, iterations=FRONTIER_TRIM)
    gy, gx = np.gradient(np.nan_to_num(Ts, nan=0.0))          # s/px, safe on grad_ok interior
    gmag = np.hypot(gy, gx)
    speed = np.full_like(T, np.nan)
    vx = np.full_like(T, np.nan)
    vy = np.full_like(T, np.nan)
    good = grad_ok & (gmag > EPS_GRAD)
    sp = (1.0 / gmag[good]) / px_per_mm / 1000.0             # px/s -> mm/s -> m/s
    speed[good] = sp
    vy[good] = sp * (gy[good] / gmag[good])                  # +row (down) = away from gap
    vx[good] = sp * (gx[good] / gmag[good])
    m = good & (speed <= SPEED_CEIL)
    speed[~m] = np.nan; vx[~m] = np.nan; vy[~m] = np.nan
    return speed, vx, vy, m


def qualifying_events(results, p):
    df = pd.read_csv(os.path.join(results, "measurements_all_events.csv"))
    for c in ("frame", "arc_tip_row", "arc_tip_col", "event_id", "is_flash", "local_vel_ms"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ev = df[(df.event_id >= 0) & (df.is_flash != 1)].dropna(subset=["arc_tip_row"])
    qual = []
    for eid, g in ev.groupby("event_id"):
        rows = g["arc_tip_row"].to_numpy()
        if len(g) >= MIN_FRAMES and (rows.max() - rows.min()) >= MIN_EXT_PX:
            qual.append((int(eid), g.sort_values("frame")))
    return qual, ev["event_id"].nunique()


def safe_out_dir(results):
    out = os.path.join(results, OUT_SUBDIR)
    assert os.path.basename(out) == OUT_SUBDIR, "refusing to write outside %s" % OUT_SUBDIR
    os.makedirs(out, exist_ok=True)
    return out


def save_npy(out_dir, name, arr):
    assert name not in PROTECTED, "refusing to overwrite protected legacy file %s" % name
    assert os.path.basename(out_dir) == OUT_SUBDIR, "refusing to write outside %s" % OUT_SUBDIR
    np.save(os.path.join(out_dir, name), arr)


def process(results, label, close_r):
    p = load_params(results)
    H, W, ppm = p["H"], p["W"], p["px_per_mm"]
    out_dir = safe_out_dir(results)
    print("\n=== %s  (%s) ===" % (label, os.path.basename(results)))
    print("  NON-DESTRUCTIVE: writing only to %s ; legacy velocity_map_local.npy untouched"
          % out_dir)

    qual, n_total = qualifying_events(results, p)
    print("  qualifying (propagating) events: %d of %d" % (len(qual), n_total))
    tfs, offsets = build_chunks(os.path.join(results, "tiff"), p["n_frames"])

    speed_sum = np.zeros((H, W)); vx_sum = np.zeros((H, W)); vy_sum = np.zeros((H, W))
    cnt = np.zeros((H, W))
    n_used = 0
    tip_check = []
    for _eid, g in qual:
        gframes = g["frame"].astype(int).to_numpy()
        f0 = gframes[0]
        T = np.full((H, W), np.inf)
        for gf in gframes:
            raw = page_for(tfs, offsets, gf)
            if raw is None:
                continue
            band = channel_band(raw, p, close_r)
            if band is None:
                continue
            fresh = band & ~np.isfinite(T)              # first arrival only
            T[fresh] = (gf - f0) / p["fps"]
        T[~np.isfinite(T)] = np.nan
        if np.isfinite(T).sum() < 50:
            continue
        speed, vx, vy, m = event_field(T, ppm)
        if not m.any():
            continue
        speed_sum[m] += speed[m]; vx_sum[m] += vx[m]; vy_sum[m] += vy[m]; cnt[m] += 1
        n_used += 1
        rows = g["arc_tip_row"].to_numpy()
        cols = g["arc_tip_col"].to_numpy()
        lvs = g["local_vel_ms"].to_numpy()
        for tr, tc, lv in zip(rows, cols, lvs):
            if not (np.isfinite(tr) and np.isfinite(tc) and np.isfinite(lv) and lv > 0):
                continue
            tr, tc = int(round(tr)), int(round(tc))
            if 0 <= tr < H and 0 <= tc < W and m[tr, tc]:
                tip_check.append((speed[tr, tc], float(lv)))
    for t in tfs:
        t.close()

    raw_have = cnt > 0
    have = cnt >= MIN_COVERAGE
    dropped = int(raw_have.sum() - have.sum())
    print("  coverage floor >= %d: dropped %d low-N px (%.0f%%) -> %d kept"
          % (MIN_COVERAGE, dropped,
             100 * dropped / max(int(raw_have.sum()), 1), int(have.sum())))
    speed_mean = np.where(have, speed_sum / np.maximum(cnt, 1), np.nan)
    vx_mean = np.where(have, vx_sum / np.maximum(cnt, 1), np.nan)
    vy_mean = np.where(have, vy_sum / np.maximum(cnt, 1), np.nan)

    save_npy(out_dir, "velocity_field_speed_ms.npy", speed_mean)
    save_npy(out_dir, "velocity_field_vx_ms.npy", vx_mean)
    save_npy(out_dir, "velocity_field_vy_ms.npy", vy_mean)
    save_npy(out_dir, "velocity_field_count.npy", cnt)

    # ---- validation ----
    sv = speed_mean[have]
    print("  events used: %d | speed m/s: med=%.2f p95=%.2f max=%.2f"
          % (n_used, np.median(sv), np.percentile(sv, 95), sv.max()))
    print("  direction: %.0f%% of pixels propagate away from gap (+row)"
          % (100 * np.mean(vy_mean[have] > 0)))
    if tip_check:
        arr = np.array(tip_check)
        r = np.median(arr[:, 0]) / np.median(arr[:, 1])
        print("  tip cross-check (n=%d): eikonal med=%.2f vs tip-vel med=%.2f  ratio=%.2f"
              % (len(arr), np.median(arr[:, 0]), np.median(arr[:, 1]), r))

    render_recording(results, out_dir, label, speed_mean, vx_mean, vy_mean, cnt)
    return dict(label=label, speed=speed_mean, vx=vx_mean, vy=vy_mean,
                cnt=cnt, ppm=ppm, results=results)


def _overlay_edges(ax, results):
    ep = os.path.join(results, "electrode_edges.npz")
    if os.path.exists(ep):
        z = np.load(ep)
        ax.plot(z["A"][:, 1], z["A"][:, 0], "-", color="cyan", lw=1.0, alpha=0.7)
        ax.plot(z["B"][:, 1], z["B"][:, 0], "-", color="magenta", lw=1.0, alpha=0.7)


def _quiver(ax, vx, vy, H, W, step=6, scale=6.0):
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    U = vx[::step, ::step]; V = vy[::step, ::step]
    good = np.isfinite(U) & np.isfinite(V)
    ax.quiver(xx[good], yy[good], U[good], V[good], color="white",
              scale_units="xy", angles="xy", scale=scale, width=0.003)


def render_recording(results, out_dir, label, speed, vx, vy, cnt):
    H, W = speed.shape
    cm = plt.get_cmap("plasma").copy(); cm.set_bad("0.12")
    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    curp = os.path.join(results, "velocity_map_local.npy")     # READ-ONLY reference
    if os.path.exists(curp):
        cur = np.load(curp).astype(float) / 1000.0
        ax[0].imshow(np.ma.masked_invalid(cur), cmap=cm, vmin=0.25,
                     vmax=np.nanpercentile(cur, 98), origin="upper")
    ax[0].set_title("CURRENT (paper): tip-speed on arc body")
    vmax = float(np.nanpercentile(speed, 98))
    im = ax[1].imshow(np.ma.masked_invalid(speed), cmap=cm, vmin=0.25, vmax=vmax,
                      origin="upper")
    _quiver(ax[1], vx, vy, H, W)
    ax[1].set_title("NEW: eikonal channel front-speed FIELD + direction")
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04).set_label("front speed (m/s)")
    cc = ax[2].imshow(np.ma.masked_where(cnt == 0, cnt), cmap="viridis", origin="upper")
    ax[2].set_title("coverage (events per pixel)")
    fig.colorbar(cc, ax=ax[2], fraction=0.046, pad=0.04).set_label("N events")
    for a in ax:
        _overlay_edges(a, results); a.set_xticks([]); a.set_yticks([])
    fig.suptitle("%s  —  %s" % (label, os.path.basename(results)), y=0.99)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "velocity_field.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_montage(records, out_path):
    assert os.path.basename(out_path) not in PROTECTED
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pooled = np.concatenate([r["speed"][np.isfinite(r["speed"])] for r in records])
    vmax = float(np.percentile(pooled, 98))
    cm = plt.get_cmap("plasma").copy(); cm.set_bad("0.12")
    n = len(records)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10)) if n == 4 \
        else plt.subplots(1, n, figsize=(5 * n, 5))
    im = None
    for ax, r in zip(np.ravel(axes), records):
        H, W = r["speed"].shape
        im = ax.imshow(np.ma.masked_invalid(r["speed"]), cmap=cm, vmin=0.25, vmax=vmax,
                       origin="upper")
        _quiver(ax, r["vx"], r["vy"], H, W)
        _overlay_edges(ax, r["results"])
        ax.set_title(r["label"]); ax.set_xticks([]); ax.set_yticks([])
    fig.subplots_adjust(right=0.88, wspace=0.15, hspace=0.15)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cax).set_label("front speed (m/s)")
    fig.suptitle("Eikonal channel front-speed field (NEW method) — all Re", y=0.99)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nwrote montage:", out_path)


def main():
    ap = argparse.ArgumentParser(description="Eikonal channel front-speed field (non-destructive)")
    ap.add_argument("--results", nargs="+", required=True, help="one or more *_results dirs")
    ap.add_argument("--labels", nargs="+", default=None, help="per-results display labels")
    ap.add_argument("--close-radius", type=int, default=DEFAULT_CLOSE_R)
    ap.add_argument("--montage-out", default=None, help="path for a combined montage PNG")
    args = ap.parse_args()
    labels = args.labels if args.labels and len(args.labels) == len(args.results) \
        else [os.path.basename(r).replace("_results", "") for r in args.results]

    records = [process(r, lab, args.close_radius) for r, lab in zip(args.results, labels)]
    if args.montage_out:
        render_montage(records, args.montage_out)


if __name__ == "__main__":
    main()
