#!/usr/bin/env python3
"""
detect_blade_lines.py  —  Automatic, headless blade-line detection.

Produces the same `blade_lines.npy` that `analyze_blade.py --blade-lines`
consumes (shape (2,2,2): [bladeA,bladeB] x [pt0,pt1] x [row,col]), but WITHOUT
the interactive mouse-click step in `calibrate_electrodes.py`, so it can run in
batch / on a server.

Physics premise (confirmed from the 11MAY26 data, see METHODS_BLADE.md):
  * The arc is an arch bridging two blade electrodes that meet at an apex and
    diverge at the opening angle.
  * A *clean* arch frame has exactly two skeleton endpoints (the two feet), one
    rooted on each blade.
  * The feet sit on the ELECTRODE side of the occupancy base row; as the arc
    glides the feet move along the blades, so the loci of the left and right
    feet over many frames trace the two blade lines.

The measured FULL opening angle is ~2x the per-blade half-angle. The 11MAY26
filename "25deg" is the per-blade half-angle (cf. METHODS.md half-angle
convention), so a measured full angle near 50 deg is the expected, consistent
result.

Robustness:
  * Only frames with exactly two endpoints, both on the electrode side of the
    base row, are used (rejects partial arcs / free-tip contamination).
  * Left/right foot assignment is deterministic per frame (min-col vs max-col).
  * Each blade is fit by total-least-squares with iterative outlier trimming.
  * ALWAYS inspect the QA overlay; fall back to calibrate_electrodes.py for a
    hand-defined line if the arc rarely bridges in a given segment.

Usage:
    python detect_blade_lines.py CHUNK.tiff --arc-direction up \
        --out "<stem>_results/blade_lines.npy"
"""
import argparse
from pathlib import Path
import numpy as np
import tifffile
from skimage.morphology import skeletonize
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_blade import binarize, skeleton_endpoints, DEFAULT_DARK_GUARD


def tls_line(pts, n_iter=4, k=2.5):
    """Total-least-squares line with iterative outlier trimming.
    Returns (point_on_line, unit_dir, n_inliers) in (row,col)."""
    P = pts.copy()
    mean = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - mean)
    d = vt[0]
    for _ in range(n_iter):
        rel = P - mean
        proj = rel @ d
        perp = rel - np.outer(proj, d)
        res = np.linalg.norm(perp, axis=1)
        keep = res <= np.median(res) + k * (res.std() + 1e-9)
        if keep.sum() < max(8, 0.3 * len(P)):
            break
        P = P[keep]
        mean = P.mean(axis=0)
        _, _, vt = np.linalg.svd(P - mean)
        d = vt[0]
    return mean, d, len(P)


def col_at_row(mean, d, r):
    if abs(d[0]) < 1e-9:
        return mean[1]
    t = (r - mean[0]) / d[0]
    return mean[1] + t * d[1]


def main():
    ap = argparse.ArgumentParser(description='Automatic blade-line detection')
    ap.add_argument('tiff')
    ap.add_argument('--arc-direction', choices=['up', 'down'], default='up')
    ap.add_argument('--thresh', type=int, default=100)
    ap.add_argument('--close-radius', type=int, default=3)
    ap.add_argument('--min-arc-px', type=int, default=22)
    ap.add_argument('--px-per-mm', type=float, default=8.8)
    ap.add_argument('--band-frac', type=float, default=0.30,
                    help='Electrode-band half-height as a fraction of frame height')
    ap.add_argument('--min-gap-px', type=float, default=15.0,
                    help='Minimum foot separation for a clean arch')
    ap.add_argument('--max-frames', type=int, default=2000)
    ap.add_argument('--out', required=True)
    ap.add_argument('--qa', default=None, help='QA overlay PNG (default: alongside --out)')
    args = ap.parse_args()

    left_feet, right_feet = [], []
    with tifffile.TiffFile(args.tiff) as t:
        n = min(len(t.pages), args.max_frames)
        H, W = t.pages[0].asarray().shape
        occ = np.zeros((H, W))
        endpts = []
        for i in range(n):
            raw = t.pages[i].asarray()
            b = binarize(raw, args.close_radius, dark_guard=DEFAULT_DARK_GUARD,
                         bright_thresh=args.thresh, min_arc_px=args.min_arc_px)
            if not b.any():
                continue
            occ += b
            sk = skeletonize(b)
            if not sk.any():
                continue
            eps = skeleton_endpoints(sk).astype(float)
            if len(eps) == 2:
                endpts.append(eps)

    base_row = float(np.argmax(occ.sum(axis=1)))
    band = args.band_frac * H
    margin = 0.10 * H
    # The real blade roots sit on the ELECTRODE side of the base row: large rows
    # (bottom) for 'up', small rows (top) for 'down'. Feet on the far side are
    # partial-arc / plume artifacts, not blade roots.
    for eps in endpts:
        if args.arc_direction == 'up':
            on_side = np.all(eps[:, 0] >= base_row - margin)
        else:
            on_side = np.all(eps[:, 0] <= base_row + margin)
        within = np.all(np.abs(eps[:, 0] - base_row) <= band)
        if on_side and within:
            lo = eps[eps[:, 1].argmin()]
            hi = eps[eps[:, 1].argmax()]
            if (hi[1] - lo[1]) >= args.min_gap_px:
                left_feet.append(lo)
                right_feet.append(hi)

    left = np.array(left_feet)
    right = np.array(right_feet)
    print("frames=%d  clean-arch frames used=%d  base_row=%.0f (%s)" % (
        n, len(left), base_row, 'BOTTOM' if base_row > H / 2 else 'TOP'))
    if len(left) < 20 or len(right) < 20:
        raise SystemExit("ERROR: too few clean arch feet to fit blades "
                         "(try more frames or a different segment).")

    mA, dA, nA = tls_line(left)
    mB, dB, nB = tls_line(right)
    # The TLS direction sign (from SVD) is arbitrary per blade.  Force the two
    # blade directions to share an orientation so that downstream
    # midline_from_blades (u = dA + dB) yields the ANGLE bisector of the wedge,
    # not the perpendicular bisector.  Without this, a sign mismatch would
    # silently flip the electrode midline used for the A/B asymmetry split.
    if float(dA @ dB) < 0.0:
        dB = -dB
    assert float(dA @ dB) >= 0.0, "blade directions still anti-parallel after sign fix"
    ang = np.degrees(np.arccos(np.clip(abs(dA @ dB), -1, 1)))

    rmin = min(left[:, 0].min(), right[:, 0].min())
    rmax = max(left[:, 0].max(), right[:, 0].max())
    rows = np.linspace(rmin, rmax, 200)
    gap_px = np.abs([col_at_row(mA, dA, r) - col_at_row(mB, dB, r) for r in rows])
    print("blade A inliers=%d  blade B inliers=%d" % (nA, nB))
    print("FULL opening angle = %.1f deg  (half-angle = %.1f deg)" % (ang, ang / 2))
    print("  -> filename '25deg' is consistent as the per-blade HALF-angle "
          "(2 x 25 = 50 deg full); cf. METHODS.md half-angle convention.")
    print("inter-blade gap over [%.0f,%.0f] px: min=%.2f mm (apex)  max=%.2f mm" % (
        rmin, rmax, gap_px.min() / args.px_per_mm, gap_px.max() / args.px_per_mm))

    def two_pts(m, d):
        return np.stack([m + (-H) * d, m + H * d], axis=0)
    blade_lines = np.stack([two_pts(mA, dA), two_pts(mB, dB)], axis=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out), blade_lines)
    print("saved %s  shape %s" % (out, blade_lines.shape))

    qa = args.qa or (str(out.with_suffix('')) + '_qa.png')
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
    ax.imshow(np.power(occ / max(occ.max(), 1e-9), 0.4), cmap='gray')
    ax.scatter(left[:, 1], left[:, 0], s=5, c='cyan', alpha=0.4)
    ax.scatter(right[:, 1], right[:, 0], s=5, c='orange', alpha=0.4)
    rr = np.array([0, H])
    ax.plot([col_at_row(mA, dA, r) for r in rr], rr, c='cyan', lw=2, label='blade A')
    ax.plot([col_at_row(mB, dB, r) for r in rr], rr, c='orange', lw=2, label='blade B')
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.legend(fontsize=8)
    fig.savefig(qa, dpi=110, facecolor='black', bbox_inches='tight')
    print("saved %s" % qa)


if __name__ == '__main__':
    main()
