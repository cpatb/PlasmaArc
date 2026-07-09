#!/usr/bin/env python3
"""
segment_electrodes.py  —  HAND-segment the two J-hook electrode edges per video.

The J electrodes are hooked (J-shaped) with the hooks facing each other, and the
bright CO2 plume makes automatic detection unreliable, so the trustworthy way to
define the geometry is to draw the two electrode edges by hand on a guide image.
This opens a napari viewer (or a matplotlib fallback) showing, as layers:

    • max projection           — the arc envelope over the recording
    • attachment density       — where the arc actually attaches (= electrode metal)
    • a bright sample frame     — a single clear arc frame

You draw ONE path along each electrode's facing edge (two paths total), then
close the window.  The paths are densified into edge point-clouds and saved, with
the facing hook tips taken as the closest points of the two edges.

ONE FILE PER VIDEO — NOT carried across recordings.  The output is written into
THIS recording's results folder and stamped with the recording name; reusing it
for another video is intentionally discouraged (each electrode_edges.npz is
unique to its recording).

Outputs (into <stem>_results/):
    electrode_edges.npz   A, B (edge point clouds), A_tip, B_tip (facing hooks)
    j_geometry.json       px_per_mm, gap_row, detected gap, hand_segmented=True,
                          recording=<stem>

run_j_co2.py auto-loads electrode_edges.npz and threads --electrode-edges through
analyze_blade.py (curved-edge attachment test).

USAGE (on a machine with a display)
-----------------------------------
    pip install "napari[all]"                # one-time, if not installed
    python segment_electrodes.py --results "<...>/HS_0.8mm_..._results" --px-per-mm 8.75
    # matplotlib fallback (no napari): click vertices, press Enter between electrodes
    python segment_electrodes.py --results "<...>_results" --backend matplotlib
"""
import argparse, json
from pathlib import Path
import numpy as np
import tifffile


# ── guide-image construction ────────────────────────────────────────────────

def load_stack(results_dir, n_max=8000, across=True):
    """Load up to n_max frames as the nucleation guide stack.

    With across=True (default) the frames are sampled EVENLY across the WHOLE
    recording (every k-th frame over all chunks), so the attachment-density guide
    reflects the entire run rather than only its first n_max frames. This matters
    for long recordings — e.g. the 1.2 mm video's first 6000 frames are barely
    30 ms of a 4.4 s run. Set across=False to use the original first-n behaviour.
    """
    tiffs = sorted((Path(results_dir) / 'tiff').glob('*.tiff'))
    if not tiffs:
        raise SystemExit('no tiff/ chunks in %s — run conversion first' % results_dir)
    counts = []
    for t in tiffs:
        with tifffile.TiffFile(str(t)) as tf:
            counts.append(len(tf.pages))
    total = int(sum(counts))
    if across and total > n_max:
        targets = set(np.linspace(0, total - 1, n_max).astype(int).tolist())
    else:
        targets = set(range(min(n_max, total)))
    frames, gi = [], 0
    for t, n in zip(tiffs, counts):
        with tifffile.TiffFile(str(t)) as tf:
            for pi in range(n):
                if gi in targets:
                    frames.append(tf.pages[pi].asarray())
                gi += 1
    print('  guide stack: %d frames %s (recording has %d total)'
          % (len(frames), 'sampled evenly across the run' if across
             else 'from the start of the run', total))
    return np.stack(frames, 0)


def _stretch(img, lo_p=1.0, hi_p=99.5):
    """Percentile contrast-stretch an image to 0..255 so dim, low-contrast frames
    become visible. Used for the guide-image layers only (never for measurement)."""
    img = np.asarray(img, float)
    lo, hi = np.percentile(img, [lo_p, hi_p])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0


def _clahe(img, clip_limit=0.01):
    """Adaptive local-contrast enhancement (CLAHE) -> 0..255. Unlike a global stretch,
    this boosts contrast region-by-region, so a dark electrode silhouette against a
    dark background becomes visible. Falls back to a global stretch if scikit-image
    is unavailable."""
    a = np.asarray(img, float)
    a = (a - a.min()) / max(a.max() - a.min(), 1e-6)
    try:
        from skimage.exposure import equalize_adapthist
        return equalize_adapthist(a, clip_limit=clip_limit) * 255.0
    except Exception:
        return _stretch(img)


def _accumulate_endpoints(frame, seed, core_lo, core_hi, heat, pts, k, label, convolve, skeletonize):
    """For one arc-core frame, skeletonise the largest bright blob and append its
    skeleton END-POINTS (the arc feet = where the channel attaches to electrode metal)
    to `heat` and `pts`. No-op for frames without a valid arc core."""
    n_core = int((frame > seed).sum())
    if not (core_lo <= n_core <= core_hi):
        return
    b = frame > seed
    lbl, n = label(b)
    if n == 0:
        return
    s = np.bincount(lbl.ravel()); s[0] = 0
    b = lbl == s.argmax()
    if b.sum() < core_lo:
        return
    sk = skeletonize(b)
    nb = convolve(sk.astype(np.uint8), k, mode='constant')
    rr, cc = np.where(sk & (nb == 1))
    for r, c in zip(rr, cc):
        heat[r, c] += 1
        pts.append((r, c))


def _finish_nucleation(pts, heat, gaussian_filter):
    pts = np.array(pts, float) if pts else np.empty((0, 2))
    heat = gaussian_filter(heat, 1.0)
    if len(pts):
        dens = heat[pts[:, 0].astype(int), pts[:, 1].astype(int)]
    else:
        dens = np.empty((0,))
    return pts, dens, heat


def compute_nucleation(stack, seed=200, core_lo=20, core_hi=400):
    """Track arc nucleation/attachment sites across the frames of `stack`.
    Returns pts (M,2 [row,col]), dens (M, recurrence density), heat (H,W density map).
    The point cloud is what you trace the two electrode edges through."""
    from scipy.ndimage import label, convolve, gaussian_filter
    from skimage.morphology import skeletonize
    H, W = stack.shape[1:]
    heat = np.zeros((H, W), float); pts = []
    k = np.ones((3, 3), np.uint8); k[1, 1] = 0
    for fi in range(len(stack)):
        _accumulate_endpoints(stack[fi], seed, core_lo, core_hi, heat, pts, k,
                              label, convolve, skeletonize)
    return _finish_nucleation(pts, heat, gaussian_filter)


def compute_nucleation_all(results_dir, seed=200, core_lo=20, core_hi=400):
    """Stream EVERY frame of the recording (chunk by chunk, low memory) and collect
    ALL arc-attachment points — denser than the sampled guide stack, but slower
    (it reads the whole video). Same return contract as compute_nucleation."""
    from scipy.ndimage import label, convolve, gaussian_filter
    from skimage.morphology import skeletonize
    tiffs = sorted((Path(results_dir) / 'tiff').glob('*.tiff'))
    heat = None; pts = []; k = np.ones((3, 3), np.uint8); k[1, 1] = 0
    nseen = 0
    for t in tiffs:
        with tifffile.TiffFile(str(t)) as tf:
            for p in tf.pages:
                f = p.asarray()
                if heat is None:
                    heat = np.zeros(f.shape, float)
                _accumulate_endpoints(f, seed, core_lo, core_hi, heat, pts, k,
                                      label, convolve, skeletonize)
                nseen += 1
                if nseen % 50000 == 0:
                    print('    ...%d frames processed, %d points so far' % (nseen, len(pts)))
    print('  streamed ALL %d frames -> %d attachment points' % (nseen, len(pts)))
    return _finish_nucleation(pts, heat if heat is not None else np.zeros((1, 1)), gaussian_filter)


# ── pure geometry: paths -> edge point clouds + facing tips (unit-testable) ──

def _densify(path, step=1.0):
    """Sample points at ~step px spacing along a polyline path (N,2) [row,col]."""
    path = np.asarray(path, float)
    if len(path) < 2:
        return path
    out = []
    for a, b in zip(path[:-1], path[1:]):
        d = np.hypot(*(b - a))
        n = max(int(np.ceil(d / step)), 1)
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(a + t * (b - a))
    out.append(path[-1])
    return np.array(out)


def paths_to_edges(paths_a, paths_b, step=1.0):
    """Convert hand-drawn paths for electrode A and B into edge point clouds plus
    the facing hook tips (closest points of the two clouds).

    paths_a / paths_b : list of (N,2) vertex arrays [row,col] (≥1 path each).
    Returns A, B, A_tip, B_tip, gap_px.
    """
    A = np.vstack([_densify(p, step) for p in paths_a]) if paths_a else np.empty((0, 2))
    B = np.vstack([_densify(p, step) for p in paths_b]) if paths_b else np.empty((0, 2))
    if len(A) == 0 or len(B) == 0:
        raise SystemExit('Need a path drawn for BOTH electrodes (got A=%d, B=%d pts).'
                         % (len(A), len(B)))
    # facing tips = closest pair between the two edges
    D = np.linalg.norm(A[:, None, :] - B[None], axis=2)
    i, j = np.unravel_index(D.argmin(), D.shape)
    return A, B, A[i], B[j], float(D[i, j])


def save_geometry(results_dir, A, B, A_tip, B_tip, gap_px, gap_mm,
                  arc_direction, recording):
    rd = Path(results_dir)
    gap_row = (int(np.floor(0.5 * (A_tip[0] + B_tip[0]))) - 2 if arc_direction == 'down'
               else int(np.ceil(0.5 * (A_tip[0] + B_tip[0]))) + 2)
    # px/mm is DERIVED from the segmented apex gap and the known electrode gap
    # (3.2 mm) — the only calibration available for the archival data. It is
    # never an external/hardcoded value, so j_geometry.json is self-consistent.
    px_per_mm = round(gap_px / gap_mm, 4) if gap_mm > 0 else None
    np.savez(rd / 'electrode_edges.npz', A=A, B=B, A_tip=A_tip, B_tip=B_tip)
    geom = dict(px_per_mm=px_per_mm, gap_row=int(gap_row),
                min_gap_mm=gap_mm, min_gap_px=round(gap_px, 2),
                electrode_model='J-hook edge point-set (hand-segmented)',
                hand_segmented=True, recording=recording,
                arc_direction=arc_direction,
                a_tip=[round(float(A_tip[0]), 1), round(float(A_tip[1]), 1)],
                b_tip=[round(float(B_tip[0]), 1), round(float(B_tip[1]), 1)],
                n_edge_pts_A=int(len(A)), n_edge_pts_B=int(len(B)))
    json.dump(geom, open(rd / 'j_geometry.json', 'w'), indent=2)
    print('\nSaved electrode_edges.npz + j_geometry.json -> %s' % rd)
    print('  hook tips A=(%.0f,%.0f) B=(%.0f,%.0f)  gap=%.1f px = %.2f mm  -> px/mm=%.4f  gap_row=%d'
          % (A_tip[0], A_tip[1], B_tip[0], B_tip[1], gap_px, gap_mm, px_per_mm, gap_row))


# ── napari UI ────────────────────────────────────────────────────────────────

def run_napari(guides, density, pts, dens):
    """guides: ordered dict {layer_name: (image, visible_bool)} from best electrode-
    contrast view to least."""
    import napari
    v = napari.Viewer(title='Trace each electrode edge through the nucleation points, then close')
    for name, (img, vis) in guides.items():
        v.add_image(img, name=name, colormap='gray', contrast_limits=[0, 255], visible=vis)
    v.add_image(density, name='attachment density', colormap='inferno',
                opacity=0.6, blending='additive', visible=True)
    if len(pts):
        d = dens / dens.max() if dens.max() > 0 else dens
        v.add_points(pts, name='nucleation points', size=3, opacity=0.7,
                     face_color=d, face_colormap='viridis', border_width=0)
    la = v.add_shapes(name='electrode A', shape_type='path', edge_color='cyan', edge_width=2)
    lb = v.add_shapes(name='electrode B', shape_type='path', edge_color='lime', edge_width=2)
    print('\nINSTRUCTIONS:')
    print('  • Background layers (top of the list) show the electrode metal — toggle')
    print('    "electrode contrast (CLAHE)" / "median background" / "variance" to taste.')
    print('  • "nucleation points" + "attachment density" mark where arcs attach.')
    print('  1. Select "electrode A", Path tool, click vertices along the FULL edge. Esc to finish.')
    print('  2. Select "electrode B" and trace the other electrode the same way.')
    print('  3. Close the window to save.\n')
    napari.run()
    paths_a = [np.asarray(d)[:, -2:] for d in la.data]   # last 2 dims = (row,col)
    paths_b = [np.asarray(d)[:, -2:] for d in lb.data]
    return paths_a, paths_b


def run_matplotlib(guides, density, pts, dens):
    """Fallback: nucleation points on the best electrode-contrast guide; click vertices
    for electrode A, press Enter, then electrode B, press Enter."""
    import matplotlib.pyplot as plt
    bg = next(iter(guides.values()))[0]      # first guide = the electrode-contrast view
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bg, cmap='gray', vmin=0, vmax=255, origin='upper')
    if len(pts):
        ax.scatter(pts[:, 1], pts[:, 0], s=4, c=dens, cmap='viridis', alpha=0.6)
    out = []
    for name, col in [('electrode A', 'cyan'), ('electrode B', 'lime')]:
        ax.set_title('Click vertices along %s edge — ENTER to finish' % name, color=col)
        fig.canvas.draw()
        clk = plt.ginput(n=-1, timeout=0)      # (x,y) = (col,row)
        path = np.array([[y, x] for x, y in clk])
        out.append([path] if len(path) else [])
        if len(path):
            ax.plot(path[:, 1], path[:, 0], '-', color=col, lw=2)
        fig.canvas.draw()
    plt.close(fig)
    return out[0], out[1]


def main():
    ap = argparse.ArgumentParser(description='Hand-segment the two J-hook electrode edges')
    ap.add_argument('--results', required=True, help='recording <stem>_results/ folder (with tiff/)')
    ap.add_argument('--gap-mm', type=float, default=3.2,
                    help='known minimum electrode gap in mm (confirmed 3.2; was 2.4); '
                         'px/mm is derived as '
                         'segmented_gap_px / gap_mm (no external px/mm input).')
    ap.add_argument('--arc-direction', choices=['down', 'up'], default='down')
    ap.add_argument('--seed', type=int, default=200, help='DN seed for the attachment-density guide')
    ap.add_argument('--n-frames', type=int, default=8000,
                    help='frames sampled for the nucleation guide (more = denser cloud)')
    ap.add_argument('--first-frames', action='store_true',
                    help='sample the FIRST n-frames instead of evenly across the whole recording')
    ap.add_argument('--all-frames', action='store_true',
                    help='compute nucleation points over EVERY frame of the recording '
                         '(complete cloud; slower — streams the whole video)')
    ap.add_argument('--backend', choices=['napari', 'matplotlib'], default='napari')
    ap.add_argument('--step', type=float, default=1.0, help='edge point spacing (px)')
    args = ap.parse_args()

    rd = Path(args.results)
    recording = rd.name.replace('_results', '')
    edges_path = rd / 'electrode_edges.npz'
    if edges_path.exists():
        print('[NOTE] %s already exists — it will be overwritten with your new tracing.'
              % edges_path)

    print('Loading frames + building guide images for %s ...' % recording)
    stack = load_stack(rd, args.n_frames, across=not args.first_frames)
    maxproj = stack.max(0).astype(float)
    sample = stack[int((stack > 180).sum(axis=(1, 2)).argmax())].astype(float)
    medproj = np.median(stack, axis=0).astype(float)     # static scene = arcs removed
    stdproj = stack.astype(float).std(axis=0)            # variance: static electrode = dark
    # guide layers in display order; (image, visible-by-default). The first entry is the
    # default background (and the matplotlib-fallback canvas).
    guides = {
        'electrode contrast (CLAHE)': (_clahe(medproj), True),
        'median background':          (_stretch(medproj), False),
        'variance (std)':             (_stretch(stdproj), False),
        'max projection (enhanced)':  (_stretch(maxproj), False),
        'sample frame (enhanced)':    (_stretch(sample), False),
        'sample arc frame (raw)':     (sample, False),
    }
    if args.all_frames:
        print('  computing nucleation over the WHOLE recording (slower) ...')
        pts, dens, density = compute_nucleation_all(rd, args.seed)
    else:
        pts, dens, density = compute_nucleation(stack, args.seed)
    print('  %d nucleation/attachment points tracked (the tracing guide).' % len(pts))

    if args.backend == 'napari':
        try:
            paths_a, paths_b = run_napari(guides, density, pts, dens)
        except ImportError:
            print('[WARN] napari not installed (pip install "napari[all]"). '
                  'Falling back to matplotlib clicker.')
            paths_a, paths_b = run_matplotlib(guides, density, pts, dens)
    else:
        paths_a, paths_b = run_matplotlib(guides, density, pts, dens)

    A, B, A_tip, B_tip, gap = paths_to_edges(paths_a, paths_b, step=args.step)
    save_geometry(rd, A, B, A_tip, B_tip, gap, args.gap_mm,
                  args.arc_direction, recording)
    print('\nNext: python run_j_co2.py --cine "%s.cine" --from-stage analyze '
          '(auto-loads this electrode_edges.npz)' % recording)


if __name__ == '__main__':
    main()
