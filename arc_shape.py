"""
arc_shape.py  —  N-point arc shape tracking and normal velocity decomposition

The existing pipeline tracks only the advancing arc tip.  This script treats
the arc as a continuous deformable filament and computes velocity at N
equally-spaced points along the arc skeleton — resolving both the forward
propagation and the lateral (normal) deformation of the arc body.

For each pair of consecutive detected frames within the same discharge event:
  1. Extract the ordered skeleton and resample to N points by arc length.
  2. Orient the chain so point 0 is always the electrode-proximal end.
  3. Decompose the frame-to-frame displacement at each point into components
     parallel (tangential) and perpendicular (normal) to the local arc.

The normal velocity is physically distinct from the tip velocity: it captures
how the arc body bends and sweeps laterally, driven by the gas flow and
electromagnetic forces, independent of how fast the tip advances.

Usage:
    python arc_shape.py measurements_all.csv tiff_dir [output_dir]
        [--n-points 10] [--max-frame-gap 5] [--fps 200000]
        [--px-per-mm 9.167]

    measurements_all.csv  — combined frame CSV from the TIFF analysis pipeline
    tiff_dir              — directory containing the .tiff files
    output_dir            — where to write results (default: same as CSV)

Outputs:
    arc_shape_velocities.csv   — per-frame-pair, per-segment normal/tangential
                                  velocity and arc-position data
    arc_shape_density.png      — hexbin of normal velocity vs arc position
    arc_shape_mean_profile.png — mean ± std normal velocity vs position
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import tifffile
from scipy.ndimage import label, binary_closing, convolve
from skimage.morphology import skeletonize, disk

warnings.filterwarnings('ignore')

# ---
# Configuration (must match the values used when measurements_all.csv was made)
# ---
FPS          = 200_000
PX_PER_MM   = 9.167
N_POINTS     = 10        # number of equidistant arc points per frame
CLOSE_RADIUS = 3         # morphological closing radius (px)
DARK_GUARD    = 35       # frames with max DN < this are skipped immediately
BRIGHT_THRESH = 80       # fixed DN threshold; pixels above this count as arc signal
MIN_ARC_PX    = 75       # minimum bright-pixel count to accept as a real arc
MAX_ARC_PX    = 3000     # maximum credible arc area (thin channel only)
MAX_FRAME_GAP = 5        # only compute velocities for frame pairs ≤ this gap

# ---
# Image processing  (mirrors analyze_blade.py; kept local to avoid import path issues)
# ---

def binarize(raw_uint8):
    """Fixed-threshold binarization: pixels above BRIGHT_THRESH DN, morphological
    closing, largest connected component.  Returns None if the result is too
    small, too large, or the frame is below the dark guard."""
    if raw_uint8.max() < DARK_GUARD:
        return None
    b = raw_uint8 > BRIGHT_THRESH
    if b.sum() < MIN_ARC_PX:
        return None
    b = binary_closing(b, disk(CLOSE_RADIUS))

    labeled, n = label(b)
    if n == 0:
        return None

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest = sizes.argmax()
    area = sizes[largest]
    if area < MIN_ARC_PX or area > MAX_ARC_PX:
        return None

    return labeled == largest


def ordered_path(skel):
    """
    Return skeleton pixels as an (M, 2) array ordered along the arc from one
    endpoint to the other.  Returns None if the skeleton is too small.
    """
    coords = np.argwhere(skel)
    if len(coords) < 3:
        return None

    H, W = skel.shape
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    deg_img = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)

    ep_mask = skel & (deg_img == 1)
    eps = np.argwhere(ep_mask)
    start = tuple(eps[0]) if len(eps) >= 1 else tuple(coords[0])

    n = len(coords)
    path_r = np.empty(n, dtype=np.int32)
    path_c = np.empty(n, dtype=np.int32)
    visited = skel.copy().astype(bool)

    r, c = start
    visited[r, c] = False
    path_r[0] = r
    path_c[0] = c

    for i in range(1, n):
        r0, c0 = path_r[i-1], path_c[i-1]
        found = False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r0 + dr, c0 + dc
                if 0 <= nr < H and 0 <= nc < W and visited[nr, nc]:
                    path_r[i] = nr
                    path_c[i] = nc
                    visited[nr, nc] = False
                    found = True
                    break
            if found:
                break
        if not found:
            path_r = path_r[:i]
            path_c = path_c[:i]
            break

    return np.stack([path_r, path_c], axis=1)

# ---
# Arc geometry
# ---

def resample_arc(path, n_points):
    """
    Resample an ordered skeleton path to n_points equally spaced by arc length.

    Uses linear interpolation on cumulative arc length, so the returned points
    are not constrained to original pixel positions.  Returns None if the path
    is too short.
    """
    if path is None or len(path) < 3:
        return None

    diffs  = np.diff(path.astype(float), axis=0)
    seg_l  = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_l  = np.concatenate([[0.0], np.cumsum(seg_l)])
    total  = cum_l[-1]

    if total < 2.0:
        return None

    s = np.linspace(0.0, total, n_points)
    row = np.interp(s, cum_l, path[:, 0].astype(float))
    col = np.interp(s, cum_l, path[:, 1].astype(float))
    return np.stack([row, col], axis=1)   # (n_points, 2)


def orient_arc(pts, electrode_row):
    """
    Flip the point chain so that point 0 is always the electrode-proximal end.
    'Electrode-proximal' means closer to electrode_row (smallest row value).
    """
    if pts[0, 0] > pts[-1, 0]:   # point 0 is further from electrode than point N-1
        return pts[::-1].copy()
    return pts


def unit_tangents(pts):
    """Unit tangent at each point using central differences (one-sided at endpoints)."""
    n = len(pts)
    t = np.zeros((n, 2))
    t[1:-1] = pts[2:] - pts[:-2]
    t[0]    = pts[1]  - pts[0]
    t[-1]   = pts[-1] - pts[-2]
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    return t / norms


def decompose_velocity(pts_t, pts_t1, tangents, dt):
    """
    Decompose frame-to-frame displacement at each arc point into normal and
    tangential components.

    The normal vector is the tangent rotated 90° (left-hand normal when
    traveling from electrode to arc tip).  Positive normal velocity = arc body
    moving laterally away from the electrode axis; positive tangential velocity
    = motion in the direction of arc propagation.

    Returns v_normal and v_tangential in px/s.
    """
    normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    disp    = pts_t1 - pts_t   # px
    v_n = np.einsum('ij,ij->i', disp, normals) / dt
    v_t = np.einsum('ij,ij->i', disp, tangents) / dt
    return v_n, v_t

# ---
# Main processing loop
# ---

def process_file(tiff_path, frame_df, electrode_row, n_points, fps, px_per_mm):
    """
    Process one TIFF file: extract arc skeletons for detected frames,
    compute inter-frame normal velocities, return a list of records.

    Reads the TIFF in a single sequential pass.
    """
    stem    = tiff_path.stem
    dt_base = 1.0 / fps
    records = []

    # Restrict to complete-arc frames only: the arc bridges both electrodes,
    # the skeleton has reliable endpoints, and the path covers the full channel.
    complete = frame_df[(frame_df['arc_detected'] == 1) &
                        (pd.to_numeric(frame_df.get('arc_complete', 0),
                                       errors='coerce').fillna(0) == 1)]
    det       = complete.set_index('frame')
    det_set   = set(det.index.tolist())

    print(f"  {stem}: {len(det_set)} complete-arc frames in CSV")

    # Store the last successfully resampled arc for inter-frame comparison
    prev_frame   = None
    prev_pts     = None
    prev_event   = None

    with tifffile.TiffFile(str(tiff_path)) as tif:
        n_pages = len(tif.pages)
        print(f"  {stem}: {n_pages} pages in TIFF")

        for fi, page in enumerate(tif.pages):
            if fi not in det_set:
                prev_pts = None
                continue

            raw = page.asarray()
            if raw.ndim == 3:
                raw = raw.mean(axis=2).astype(np.uint8)
            elif raw.dtype != np.uint8:
                raw = ((raw.astype(float) - raw.min()) /
                       max(raw.max() - raw.min(), 1) * 255).astype(np.uint8)

            binary = binarize(raw)
            if binary is None:
                prev_pts = None
                continue

            skel = skeletonize(binary)
            path = ordered_path(skel)
            pts  = resample_arc(path, n_points)
            if pts is None:
                prev_pts = None
                continue

            pts = orient_arc(pts, electrode_row)

            event_id = int(det.loc[fi, 'event_id']) if 'event_id' in det.columns else -1

            if (prev_pts is not None
                    and prev_event == event_id
                    and event_id >= 0
                    and (fi - prev_frame) <= MAX_FRAME_GAP):

                dt      = (fi - prev_frame) * dt_base
                tangents = unit_tangents(prev_pts)
                v_n, v_t = decompose_velocity(prev_pts, pts, tangents, dt)

                arc_total = float(np.sum(np.hypot(np.diff(pts[:, 0]),
                                                   np.diff(pts[:, 1]))))

                for seg in range(n_points):
                    # Convert px/s → m/s via calibration
                    records.append({
                        'file'           : stem,
                        'frame_t'        : prev_frame,
                        'frame_t1'       : fi,
                        'event_id'       : event_id,
                        'seg_idx'        : seg,
                        'norm_pos'       : round(seg / max(n_points - 1, 1), 4),
                        'v_normal_ms'    : round(v_n[seg] / px_per_mm / 1000, 6),
                        'v_tangential_ms': round(v_t[seg] / px_per_mm / 1000, 6),
                        'arc_length_px'  : round(arc_total, 2),
                    })

            prev_frame = fi
            prev_pts   = pts
            prev_event = event_id

    print(f"  {stem}: {len(records)} velocity records written")
    return records

# ---
# Figures
# ---

def plot_density(df, out_dir, n_points):
    """
    Hexbin density of local normal velocity vs normalised arc position.
    One panel per recording file.  Overlaid curves show p10/p50/p90.
    """
    df    = df[df['v_normal_ms'].notna()].copy()
    files = list(df['file'].unique())
    vcap  = float(np.percentile(df['v_normal_ms'].abs(), 99))

    bins  = np.linspace(0, 1, n_points + 1)
    bctrs = 0.5 * (bins[:-1] + bins[1:])

    fig, axes = plt.subplots(1, len(files), figsize=(5 * len(files), 5), sharey=True)
    if len(files) == 1:
        axes = [axes]

    for ax, fname in zip(axes, files):
        sub = df[df['file'] == fname]

        hb = ax.hexbin(sub['norm_pos'], sub['v_normal_ms'],
                       gridsize=35, cmap='RdBu_r', bins='log',
                       extent=[0, 1, -vcap, vcap])
        plt.colorbar(hb, ax=ax, label='log10(count)')
        ax.axhline(0, color='k', lw=0.8, ls='--')

        for pct, lw, ls, lbl in [(10, 1.0, '--', 'p10'),
                                  (50, 2.0, '-',  'p50'),
                                  (90, 1.0, '--', 'p90')]:
            pvals = [
                np.percentile(sub[(sub['norm_pos'] >= lo) &
                                  (sub['norm_pos'] <  hi)]['v_normal_ms'], pct)
                if len(sub[(sub['norm_pos'] >= lo) &
                           (sub['norm_pos'] <  hi)]) >= 3 else np.nan
                for lo, hi in zip(bins[:-1], bins[1:])
            ]
            pvals = np.array(pvals)
            ok = ~np.isnan(pvals)
            if ok.any():
                ax.plot(bctrs[ok], np.clip(pvals[ok], -vcap, vcap),
                        color='#FF6F00', lw=lw, ls=ls, label=lbl)

        run_label = f'Run {files.index(fname) + 1}'
        ax.set_xlabel('Normalised arc position  (0 = electrode,  1 = tip)')
        if ax is axes[0]:
            ax.set_ylabel('Normal velocity (m/s)')
        ax.set_xlim(0, 1)
        ax.set_ylim(-vcap, vcap)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout()
    out = out_dir / 'arc_shape_density.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


def plot_mean_profile(df, out_dir, n_points):
    """
    Mean ± 1 std normal velocity vs normalised arc position, per file.
    The tangential velocity (approximately the tip velocity) is shown for
    comparison — it should match the event_velocity.py regression.
    """
    df   = df[df['v_normal_ms'].notna()].copy()
    segs = np.arange(n_points)
    norm_positions = segs / max(n_points - 1, 1)

    files = list(df['file'].unique())
    fig, axes = plt.subplots(1, len(files), figsize=(5 * len(files), 4), sharey=True)
    if len(files) == 1:
        axes = [axes]

    for ax, fname in zip(axes, files):
        sub = df[df['file'] == fname]

        mn_n  = sub.groupby('seg_idx')['v_normal_ms'].mean().reindex(segs)
        std_n = sub.groupby('seg_idx')['v_normal_ms'].std().reindex(segs)
        mn_t  = sub.groupby('seg_idx')['v_tangential_ms'].mean().reindex(segs)

        ax.fill_between(norm_positions, mn_n - std_n, mn_n + std_n,
                        alpha=0.25, color='#1565C0')
        ax.plot(norm_positions, mn_n, color='#1565C0', lw=2, label='Normal (mean +/- 1sigma)')
        ax.plot(norm_positions, mn_t, color='#C62828', lw=1.5, ls='--',
                label='Tangential (mean)')
        ax.axhline(0, color='k', lw=0.6, ls=':')

        run_label = f'Run {files.index(fname) + 1}'
        ax.set_xlabel('Normalised arc position  (0 = electrode,  1 = tip)')
        if ax is axes[0]:
            ax.set_ylabel('Velocity (m/s)')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout()
    out = out_dir / 'arc_shape_mean_profile.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")

# ---
# Entry point
# ---

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv',      help='measurements_all.csv (with event_id column)')
    ap.add_argument('tiff_dir', help='Directory containing the .tiff source files')
    ap.add_argument('out_dir',  nargs='?', default=None,
                    help='Output directory (default: same directory as csv)')
    ap.add_argument('--n-points',     type=int,   default=N_POINTS)
    ap.add_argument('--fps',          type=float, default=FPS)
    ap.add_argument('--px-per-mm',    type=float, default=PX_PER_MM)
    ap.add_argument('--max-frame-gap',type=int,   default=MAX_FRAME_GAP,
                    help='Max frame gap between consecutive pairs (default 5)')
    args = ap.parse_args()

    csv_path  = Path(args.csv)
    tiff_dir  = Path(args.tiff_dir)
    out_dir   = Path(args.out_dir) if args.out_dir else csv_path.parent
    out_dir.mkdir(exist_ok=True)

    fps      = args.fps
    px_mm    = args.px_per_mm
    n_points = args.n_points

    print(f"\n{'=' * 55}")
    print(f"  Arc shape tracking  ({n_points} points per frame)")
    print(f"  CSV      : {csv_path}")
    print(f"  TIFFs    : {tiff_dir}")
    print(f"  Output   : {out_dir}")
    print(f"  FPS      : {fps:.0f}    px/mm : {px_mm}")
    print(f"{'=' * 55}\n")

    if not csv_path.exists():
        print(f"[ERROR] Not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if 'event_id' not in df.columns:
        print("[ERROR] CSV is missing event_id column — run event_velocity.py first.")
        sys.exit(1)

    all_records = []

    for stem, file_df in df.groupby('file'):
        # Locate the corresponding TIFF
        tiff_path = tiff_dir / f'{stem}.tiff'
        if not tiff_path.exists():
            tiff_path = tiff_dir / f'{stem}.tif'
        if not tiff_path.exists():
            print(f"  [SKIP] TIFF not found for {stem}")
            continue

        # Electrode row for this file (used to orient the arc chain)
        det = file_df[file_df['arc_detected'] == 1]
        if len(det) == 0:
            continue
        electrode_row = float(np.nanpercentile(
            pd.to_numeric(det['arc_tip_row'], errors='coerce').dropna(), 5))

        print(f"Processing {stem}  (electrode_row ≈ {electrode_row:.1f} px)")
        records = process_file(tiff_path, file_df, electrode_row,
                               n_points, fps, px_mm)
        all_records.extend(records)

    if not all_records:
        print("[WARN] No velocity records produced.")
        return

    out_df = pd.DataFrame(all_records)
    csv_out = out_dir / 'arc_shape_velocities.csv'
    out_df.to_csv(csv_out, index=False)
    print(f"\nSaved: {csv_out}  ({len(out_df):,} records)")
    print(f"  Files: {out_df['file'].nunique()}")
    print(f"  Events: {out_df['event_id'].nunique()}")
    print(f"  Frame pairs: {out_df.groupby(['file','frame_t']).ngroups:,}")

    vn = out_df['v_normal_ms']
    vt = out_df['v_tangential_ms']
    print(f"\n  Normal velocity     mean={vn.mean():.3f}  "
          f"IQR={vn.quantile(.25):.3f}–{vn.quantile(.75):.3f} m/s")
    print(f"  Tangential velocity mean={vt.mean():.3f}  "
          f"IQR={vt.quantile(.25):.3f}–{vt.quantile(.75):.3f} m/s")

    plot_density(out_df, out_dir, n_points)
    plot_mean_profile(out_df, out_dir, n_points)

    print(f"\nDone.  Output: {out_dir}")

if __name__ == '__main__':
    main()
