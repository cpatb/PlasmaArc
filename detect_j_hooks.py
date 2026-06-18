#!/usr/bin/env python3
"""
detect_j_hooks.py  —  J-hook electrode detection by arc-attachment tracking.

The J-electrode reactor uses two hooked (J-shaped) electrodes with the hooks
FACING each other.  Rather than assume a straight-blade geometry, this detects
the electrode edges empirically: the arc can only attach to electrode metal, so
the set of all arc attachment sites (skeleton endpoints / roots), accumulated
across many frames, traces out the two electrode edges.  The facing hook tips
are the closest points of the two edge clouds; their separation should match the
known minimum gap (default 2.4 mm), which is used as a built-in sanity check.

Writes (into the recording's <stem>_results/ folder):
    electrode_edges.npz   A, B (curved edge point clouds), A_tip, B_tip (hook tips)
    j_geometry.json       px_per_mm, gap_row, min_gap, electrode_model, counts

run_j_co2.py auto-loads electrode_edges.npz and threads --electrode-edges through
analyze_blade.py (curved-edge attachment test).  This replaces the straight-blade
calibrate_j_electrodes.py for the J-hook geometry.

USAGE
-----
    python detect_j_hooks.py --results "<...>/HS_0.8mm_1500mL_..._results" \
        --px-per-mm 8.75 --gap-mm 2.4
    # high-flow / bright-plume recordings: raise the core seed so the plume does
    # not merge the two hooks, and/or reuse a clean recording's geometry:
    python detect_j_hooks.py --results "<...>_results" --seed 235
    python detect_j_hooks.py --results "<...>_results" \
        --reuse "<...>/HS_0.8mm_1500mL_..._results/electrode_edges.npz"

NOTE ON HIGH-FLOW RECORDINGS
    When the CO2 plume is bright and nearly always on, per-frame attachment feet
    are noisy and the two hooks can merge, collapsing the detected gap.  The
    electrodes are the SAME hardware across the sweep, so if the detected gap is
    implausible this script WARNS and you should pass --reuse with the
    electrode_edges.npz from a clean low-flow recording (co-location is easy to
    confirm by overlaying the edges on this recording's max-projection).
"""
import argparse, json
from pathlib import Path
import numpy as np
import tifffile
from scipy.ndimage import label, convolve, gaussian_filter
from skimage.morphology import skeletonize


def load_stack(results_dir, n_max=6000):
    tiffs = sorted((Path(results_dir) / 'tiff').glob('*.tiff'))
    if not tiffs:
        raise SystemExit('no tiff/ chunks in %s — run conversion first' % results_dir)
    frames = []
    for t in tiffs:
        with tifffile.TiffFile(str(t)) as tf:
            for p in tf.pages:
                frames.append(p.asarray())
                if len(frames) >= n_max:
                    break
        if len(frames) >= n_max:
            break
    return np.stack(frames, 0)


def _endpoints(sk):
    k = np.ones((3, 3), np.uint8); k[1, 1] = 0
    nb = convolve(sk.astype(np.uint8), k, mode='constant')
    r, c = np.where(sk & (nb == 1))
    return np.column_stack([r, c]) if r.size else np.empty((0, 2), int)


def detect_edges(stack, seed, core_lo=20, core_hi=400, dens_pct=80):
    """Accumulate arc attachment endpoints (DN>seed, arc-core sized) and split the
    density into the two electrode edge clouds. Returns A,B,A_tip,B_tip,gap_px."""
    core = (stack > seed).sum(axis=(1, 2))
    use = np.where((core >= core_lo) & (core <= core_hi))[0]
    pts = []
    for fi in use:
        b = stack[fi] > seed
        lbl, n = label(b)
        if n == 0:
            continue
        s = np.bincount(lbl.ravel()); s[0] = 0
        b = lbl == s.argmax()
        if b.sum() < core_lo:
            continue
        for e in _endpoints(skeletonize(b)):
            pts.append(e)
    pts = np.array(pts)
    if len(pts) < 50:
        raise SystemExit('too few attachment points (%d) — lower --seed' % len(pts))
    H, W = stack.shape[1:]
    heat, _, _ = np.histogram2d(pts[:, 0], pts[:, 1], bins=[H, W], range=[[0, H], [0, W]])
    hs = gaussian_filter(heat, 1.0)
    em = hs > np.percentile(hs[hs > 0], dens_pct)
    lbl, n = label(em)
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    keep = [k for k in np.argsort(sizes)[::-1] if sizes[k] > 0][:2]
    if len(keep) < 2:
        raise SystemExit('could not separate two electrode edges — adjust --seed/--dens-pct')
    E = [np.column_stack(np.where(lbl == k)) for k in keep]
    E.sort(key=lambda e: e[:, 1].mean())
    A, B = E
    rng = np.random.default_rng(0)
    ai = A[rng.choice(len(A), min(500, len(A)), replace=False)]
    bi = B[rng.choice(len(B), min(500, len(B)), replace=False)]
    D = np.linalg.norm(ai[:, None, :] - bi[None], axis=2)
    i, j = np.unravel_index(D.argmin(), D.shape)
    return A, B, ai[i], bi[j], float(np.linalg.norm(ai[i] - bi[j]))


def main():
    ap = argparse.ArgumentParser(description='Detect J-hook electrodes by arc-attachment tracking')
    ap.add_argument('--results', required=True, help='recording <stem>_results/ folder (with tiff/)')
    ap.add_argument('--px-per-mm', type=float, default=8.75)
    ap.add_argument('--gap-mm', type=float, default=2.4, help='known minimum hook gap (sanity check)')
    ap.add_argument('--arc-direction', choices=['down', 'up'], default='down')
    ap.add_argument('--seed', type=int, default=200,
                    help='DN seed for the arc CORE; raise for bright-plume high-flow runs')
    ap.add_argument('--n-frames', type=int, default=6000)
    ap.add_argument('--reuse', default=None,
                    help='reuse electrode_edges.npz from another (clean) recording instead '
                         'of detecting (electrodes are shared hardware across the sweep)')
    args = ap.parse_args()
    rd = Path(args.results)

    if args.reuse:
        z = np.load(args.reuse)
        A, B, At, Bt = z['A'], z['B'], z['A_tip'], z['B_tip']
        gap = float(np.hypot(*(np.asarray(At) - np.asarray(Bt))))
        print('Reusing geometry from %s' % args.reuse)
    else:
        stack = load_stack(rd, args.n_frames)
        A, B, At, Bt, gap = detect_edges(stack, args.seed)

    gap_mm = gap / args.px_per_mm
    gap_row = (int(np.floor(0.5 * (At[0] + Bt[0]))) - 2 if args.arc_direction == 'down'
               else int(np.ceil(0.5 * (At[0] + Bt[0]))) + 2)
    np.savez(rd / 'electrode_edges.npz', A=A, B=B, A_tip=At, B_tip=Bt)
    geom = dict(px_per_mm=args.px_per_mm, gap_row=int(gap_row),
                min_gap_mm=args.gap_mm, min_gap_px=round(gap, 2),
                detected_gap_mm=round(gap_mm, 3),
                electrode_model='J-hook edge point-set',
                arc_direction=args.arc_direction,
                a_tip=[round(float(At[0]), 1), round(float(At[1]), 1)],
                b_tip=[round(float(Bt[0]), 1), round(float(Bt[1]), 1)],
                n_edge_pts_A=int(len(A)), n_edge_pts_B=int(len(B)),
                reused_from=args.reuse)
    json.dump(geom, open(rd / 'j_geometry.json', 'w'), indent=2)
    print('hook tips A=(%.0f,%.0f) B=(%.0f,%.0f)  gap=%.1f px = %.2f mm  gap_row=%d'
          % (At[0], At[1], Bt[0], Bt[1], gap, gap_mm, gap_row))
    print('edge points: A=%d  B=%d   wrote electrode_edges.npz + j_geometry.json' % (len(A), len(B)))
    # sanity check against the known gap
    if not args.reuse and abs(gap_mm - args.gap_mm) > 0.4 * args.gap_mm:
        print('  [WARN] detected gap %.2f mm is far from the known %.2f mm — the plume '
              'likely corrupted detection. Re-run with a higher --seed, or --reuse a clean '
              "recording's electrode_edges.npz (same electrodes across the sweep)." % (gap_mm, args.gap_mm))
    else:
        print('  gap check OK (within 40%% of known %.2f mm)' % args.gap_mm)


if __name__ == '__main__':
    main()
