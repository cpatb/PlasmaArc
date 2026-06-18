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


def compute_nucleation(stack, seed=200, core_lo=20, core_hi=400):
    """Track arc nucleation / attachment sites across ALL frames.

    For every arc-core frame (a bright blob of core_lo..core_hi px above `seed`),
    skeletonise and take the skeleton END-POINTS — these are the arc feet where
    the channel attaches to the electrode metal.  Returns:
        pts   : (M,2) [row,col] every attachment point across the recording
        dens  : (M,)  local recurrence density at each point (how often the arc
                attaches there) — high where the electrode edge is
        heat  : (H,W) smoothed density map (a guide image)
    The point cloud is what you trace the two electrode edges through.
    """
    from scipy.ndimage import label, convolve, gaussian_filter
    from skimage.morphology import skeletonize
    H, W = stack.shape[1:]
    core = (stack > seed).sum(axis=(1, 2))
    use = np.where((core >= core_lo) & (core <= core_hi))[0]
    heat = np.zeros((H, W), float)
    k = np.ones((3, 3), np.uint8); k[1, 1] = 0
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
        sk = skeletonize(b)
        nb = convolve(sk.astype(np.uint8), k, mode='constant')
        rr, cc = np.where(sk & (nb == 1))
        for r, c in zip(rr, cc):
            heat[r, c] += 1
            pts.append((r, c))
    pts = np.array(pts, float) if pts else np.empty((0, 2))
    heat = gaussian_filter(heat, 1.0)
    if len(pts):
        ri = pts[:, 0].astype(int); ci = pts[:, 1].astype(int)
        dens = heat[ri, ci]
    else:
        dens = np.empty((0,))
    return pts, dens, heat


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


def save_geometry(results_dir, A, B, A_tip, B_tip, gap_px, px_per_mm, gap_mm,
                  arc_direction, recording):
    rd = Path(results_dir)
    gap_row = (int(np.floor(0.5 * (A_tip[0] + B_tip[0]))) - 2 if arc_direction == 'down'
               else int(np.ceil(0.5 * (A_tip[0] + B_tip[0]))) + 2)
    np.savez(rd / 'electrode_edges.npz', A=A, B=B, A_tip=A_tip, B_tip=B_tip)
    geom = dict(px_per_mm=px_per_mm, gap_row=int(gap_row),
                min_gap_mm=gap_mm, min_gap_px=round(gap_px, 2),
                detected_gap_mm=round(gap_px / px_per_mm, 3),
                electrode_model='J-hook edge point-set (hand-segmented)',
                hand_segmented=True, recording=recording,
                arc_direction=arc_direction,
                a_tip=[round(float(A_tip[0]), 1), round(float(A_tip[1]), 1)],
                b_tip=[round(float(B_tip[0]), 1), round(float(B_tip[1]), 1)],
                n_edge_pts_A=int(len(A)), n_edge_pts_B=int(len(B)))
    json.dump(geom, open(rd / 'j_geometry.json', 'w'), indent=2)
    print('\nSaved electrode_edges.npz + j_geometry.json -> %s' % rd)
    print('  hook tips A=(%.0f,%.0f) B=(%.0f,%.0f)  gap=%.1f px = %.2f mm  gap_row=%d'
          % (A_tip[0], A_tip[1], B_tip[0], B_tip[1], gap_px, gap_px / px_per_mm, gap_row))
    if abs(gap_px / px_per_mm - gap_mm) > 0.5 * gap_mm:
        print('  [NOTE] segmented gap %.2f mm differs a lot from the nominal %.2f mm — '
              'check the two paths are along the FACING edges.' % (gap_px / px_per_mm, gap_mm))


# ── napari UI ────────────────────────────────────────────────────────────────

def run_napari(sample, maxproj, density, pts, dens):
    import napari
    v = napari.Viewer(title='Trace each electrode edge through the nucleation points, then close')
    v.add_image(sample, name='sample arc frame', colormap='gray', contrast_limits=[0, 255])
    v.add_image(maxproj, name='max projection', colormap='gray',
                contrast_limits=[0, 255], visible=False)
    v.add_image(density, name='attachment density', colormap='inferno',
                opacity=0.5, blending='additive', visible=False)
    # THE nucleation point cloud — every arc-attachment site, the tracing guide.
    if len(pts):
        d = dens / dens.max() if dens.max() > 0 else dens
        v.add_points(pts, name='nucleation points', size=3, opacity=0.7,
                     face_color=d, face_colormap='viridis', border_width=0)
    la = v.add_shapes(name='electrode A', shape_type='path', edge_color='cyan', edge_width=2)
    lb = v.add_shapes(name='electrode B', shape_type='path', edge_color='lime', edge_width=2)
    print('\nINSTRUCTIONS:')
    print('  • The "nucleation points" layer shows every arc-attachment site across')
    print('    the recording (bright = the arc attaches there often = electrode edge).')
    print('  1. Select "electrode A", pick the Path tool, click vertices THROUGH the')
    print('     point band of one electrode. Finish the path with Esc.')
    print('  2. Select "electrode B" and trace the other electrode the same way.')
    print('  3. Close the window to save.\n')
    napari.run()
    paths_a = [np.asarray(d)[:, -2:] for d in la.data]   # last 2 dims = (row,col)
    paths_b = [np.asarray(d)[:, -2:] for d in lb.data]
    return paths_a, paths_b


def run_matplotlib(sample, maxproj, density, pts, dens):
    """Fallback: nucleation points scattered on the frame; click vertices for
    electrode A, press Enter, then electrode B, press Enter."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(sample, cmap='gray', vmin=0, vmax=255, origin='upper')
    if len(pts):
        ax.scatter(pts[:, 1], pts[:, 0], s=4, c=dens, cmap='viridis', alpha=0.6)
    out = []
    for name, col in [('electrode A', 'cyan'), ('electrode B', 'lime')]:
        ax.set_title('Click vertices along %s edge — ENTER to finish' % name, color=col)
        fig.canvas.draw()
        pts = plt.ginput(n=-1, timeout=0)      # (x,y) = (col,row)
        path = np.array([[y, x] for x, y in pts])
        out.append([path] if len(path) else [])
        if len(path):
            ax.plot(path[:, 1], path[:, 0], '-', color=col, lw=2)
        fig.canvas.draw()
    plt.close(fig)
    return out[0], out[1]


def main():
    ap = argparse.ArgumentParser(description='Hand-segment the two J-hook electrode edges')
    ap.add_argument('--results', required=True, help='recording <stem>_results/ folder (with tiff/)')
    ap.add_argument('--px-per-mm', type=float, default=8.75)
    ap.add_argument('--gap-mm', type=float, default=2.4, help='nominal min hook gap (for the report)')
    ap.add_argument('--arc-direction', choices=['down', 'up'], default='down')
    ap.add_argument('--seed', type=int, default=200, help='DN seed for the attachment-density guide')
    ap.add_argument('--n-frames', type=int, default=6000)
    ap.add_argument('--backend', choices=['napari', 'matplotlib'], default='napari')
    ap.add_argument('--step', type=float, default=1.0, help='edge point spacing (px)')
    args = ap.parse_args()

    rd = Path(args.results)
    recording = rd.name.replace('_results', '')
    edges_path = rd / 'electrode_edges.npz'
    if edges_path.exists():
        print('[NOTE] %s already exists — it will be overwritten with your new tracing.'
              % edges_path)

    print('Loading frames + tracking nucleation points for %s ...' % recording)
    stack = load_stack(rd, args.n_frames)
    maxproj = stack.max(0).astype(float)
    sample = stack[int((stack > 180).sum(axis=(1, 2)).argmax())].astype(float)
    pts, dens, density = compute_nucleation(stack, args.seed)
    print('  %d nucleation/attachment points tracked (the tracing guide).' % len(pts))

    if args.backend == 'napari':
        try:
            paths_a, paths_b = run_napari(sample, maxproj, density, pts, dens)
        except ImportError:
            print('[WARN] napari not installed (pip install "napari[all]"). '
                  'Falling back to matplotlib clicker.')
            paths_a, paths_b = run_matplotlib(sample, maxproj, density, pts, dens)
    else:
        paths_a, paths_b = run_matplotlib(sample, maxproj, density, pts, dens)

    A, B, A_tip, B_tip, gap = paths_to_edges(paths_a, paths_b, step=args.step)
    save_geometry(rd, A, B, A_tip, B_tip, gap, args.px_per_mm, args.gap_mm,
                  args.arc_direction, recording)
    print('\nNext: python run_j_co2.py --cine "%s.cine" --from-stage analyze '
          '(auto-loads this electrode_edges.npz)' % recording)


if __name__ == '__main__':
    main()
