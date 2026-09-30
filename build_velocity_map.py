#!/usr/bin/env python3
"""
build_velocity_map.py  -  2-D velocity map from the LOCAL (time-resolved) velocity.

The local sliding-window velocity (`local_vel_ms`, from event_velocity.py) is the
physically meaningful, time-resolved arc-tip speed. This builds the 2-D velocity
map from it, replacing the noisy central-difference map analyze_blade.py makes in
Stage 2.

Because `local_vel_ms` only exists after Stage 4 while the arc masks live in the
frames, this runs *after* Stage 4 and re-reads the TIFF(s) to re-binarise each
frame (same operators as analyze_blade.py), then accumulates the frame's local
velocity onto its arc pixels:

    velocity_map_local[pixel] = mean over frames of local_vel_ms, where the arc
                                occupied that pixel on an advancing-front frame.

Multi-chunk: pass --tiff-dir (all *.tiff in it are treated as ordered chunks) and
--frames-per-tiff so global frame numbers map to the right chunk/page. The whole
recording is accumulated into ONE map. For a single chunk use --tiff (+ optional
--frame-offset).

Outputs (in --out-dir):
    velocity_map_local.npy / .png        primary map (local velocity, npy in mm/s)
    velocity_map_comparison.png          local vs central-difference, side by side

Usage (build):
    python build_velocity_map.py --csv ".../measurements_all_events.csv" \
        --tiff-dir ".../tiff" --frames-per-tiff 10000 --out-dir "..." --px-per-mm 8.8
    python build_velocity_map.py --csv ... --tiff ".../chunk_0001.tiff" --out-dir ...

Usage (compare two pre-built maps, e.g. downward vs upward):
    python build_velocity_map.py --compare \
        --down-map ".../downward_results/velocity_map_local.npy" \
        --up-map   ".../upward_results/velocity_map_local.npy" \
        --compare-out ".../mode_compare/velocity_map_distribution_comparison.png" \
        --px-per-mm 8.8 --pct 97
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# NOTE: `tifffile` and `from analyze_blade import binarize` are imported lazily
# inside the build path (main) rather than at module top.  This keeps the
# lightweight --compare mode (which only reads pre-built .npy maps) usable
# without the heavy tifffile/scipy/skimage stack that the build path needs.

FPS_DEFAULT = 200_000


def _load_map_ms(path):
    """Load a velocity_map_local.npy and return it in m/s.

    The pipeline stores these maps in mm/s (build_velocity_map.py saves
    `vmap * 1000`).  We detect that scale heuristically (finite max >> a few
    hundred) and divide by 1000 so all comparison plots are in m/s.
    Unpopulated pixels are NaN and are preserved.
    """
    arr = np.load(path).astype(float)
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmax(finite) > 100:   # stored in mm/s -> convert
        arr = arr / 1000.0
    return arr


def compare_maps(args):
    """Compare two pre-built 2-D velocity maps (e.g. downward vs upward).

    Produces a single figure with, on a shared spatial extent and a shared
    pooled color/value scale:
      - top row:     the two velocity maps side by side (one shared colorbar)
      - bottom-left: up-minus-down difference map (diverging, symmetric scale)
      - bottom-right: overlaid, area-normalized histograms of the per-pixel
                      velocities, with median guide lines
    """
    if not args.down_map or not args.up_map:
        raise SystemExit('compare mode requires --down-map and --up-map.')

    down_p, up_p = Path(args.down_map), Path(args.up_map)
    for p in (down_p, up_p):
        if not p.exists():
            raise SystemExit('Map not found: %s' % p)

    down = _load_map_ms(down_p)
    up   = _load_map_ms(up_p)
    if down.shape != up.shape:
        raise SystemExit('Maps differ in shape (%s vs %s) - cannot build a '
                         'pixel-wise difference. Rebuild both at the same '
                         'image size.' % (down.shape, up.shape))

    d_vals = down[np.isfinite(down)]
    u_vals = up[np.isfinite(up)]
    if d_vals.size == 0 or u_vals.size == 0:
        raise SystemExit('One of the maps has no populated pixels.')

    # ---- shared bounds: vmin=0, vmax = pooled percentile across BOTH maps ----
    pooled = np.concatenate([d_vals, u_vals])
    vmax = float(np.percentile(pooled, args.pct))
    vmax = max(vmax, 1e-3)
    print('Shared color/value scale: vmin=0  vmax=%.2f m/s '
          '(pooled p%.0f over %d populated pixels)'
          % (vmax, args.pct, pooled.size))
    for label, vals in [(args.down_label, d_vals), (args.up_label, u_vals)]:
        print('  %-9s: n=%d  mean=%.2f  mean=%.2f  p95=%.2f  max=%.2f m/s'
              % (label, vals.size, np.mean(vals), vals.mean(),
                 np.percentile(vals, 95), vals.max()))

    H, W = down.shape
    ext = [0, W / args.px_per_mm, H / args.px_per_mm, 0]   # (l, r, bottom, top), mm

    fig, axes = plt.subplots(2, 2, figsize=(13, 12))

    # --- top row: the two maps, identical extent + identical vmin/vmax ---
    im0 = axes[0, 0].imshow(down, cmap='plasma', vmin=0, vmax=vmax,
                            origin='upper', extent=ext, aspect='equal')
    im1 = axes[0, 1].imshow(up, cmap='plasma', vmin=0, vmax=vmax,
                            origin='upper', extent=ext, aspect='equal')
    for ax in (axes[0, 0], axes[0, 1]):
        ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    cb = fig.colorbar(im1, ax=[axes[0, 0], axes[0, 1]],
                      fraction=0.046, pad=0.04)
    cb.set_label('Local arc velocity (m/s)')

    # --- bottom-left: difference map (up - down), symmetric diverging scale ---
    # The reactor is physically flipped between the upward and downward runs, so
    # rotate the upward map 180 deg first to register the two before differencing.
    up_for_diff   = np.rot90(up, 2) if args.diff_rotate_up else up
    up_diff_label = args.up_label + (' (rot 180)' if args.diff_rotate_up else '')
    both = np.isfinite(down) & np.isfinite(up_for_diff)
    diff = np.full_like(down, np.nan)
    diff[both] = up_for_diff[both] - down[both]
    if both.any():
        dlim = float(np.percentile(np.abs(diff[both]), 95))
        dlim = max(dlim, 1e-3)
    else:
        dlim = 1.0
    imd = axes[1, 0].imshow(diff, cmap='RdBu_r', vmin=-dlim, vmax=dlim,
                            origin='upper', extent=ext, aspect='equal')
    axes[1, 0].set_xlabel('x (mm)'); axes[1, 0].set_ylabel('y (mm)')
    cbd = fig.colorbar(imd, ax=axes[1, 0], fraction=0.046, pad=0.04)
    cbd.set_label('delta velocity (m/s)')

    # --- bottom-right: overlaid histograms on the shared value range ---
    bins = np.linspace(0, vmax, args.bins + 1)
    axh = axes[1, 1]
    for label, vals, color in [(args.down_label, d_vals, 'tab:blue'),
                               (args.up_label,   u_vals, 'tab:orange')]:
        axh.hist(np.clip(vals, 0, vmax), bins=bins, density=True,
                 histtype='stepfilled', alpha=0.45, color=color,
                 label=label)
        axh.axvline(np.mean(vals), color=color, linestyle='--', linewidth=1.5)
    axh.set_xlim(0, vmax)
    axh.set_xlabel('Local arc velocity (m/s)')
    axh.set_ylabel('Probability density')
    axh.legend()


    out = Path(args.compare_out) if args.compare_out \
        else Path('velocity_map_distribution_comparison.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Saved comparison figure: %s' % out)

    # ------------------------------------------------------------------
    # Also emit each panel as its own standalone figure in the same dir.
    # Same shared bounds (vmax / dlim / bins) so they stay directly comparable.
    # ------------------------------------------------------------------
    out_dir = out.parent

    def _safe(label):
        return ''.join(c if (c.isalnum() or c in '-_.') else '_' for c in label)

    saved = [out]

    # 1) downward map (own figure)
    f, a = plt.subplots(figsize=(6, 7))
    im = a.imshow(down, cmap='plasma', vmin=0, vmax=vmax,
                  origin='upper', extent=ext, aspect='equal')
    a.set_xlabel('x (mm)'); a.set_ylabel('y (mm)')
    f.colorbar(im, ax=a, fraction=0.046, pad=0.04).set_label('Local arc velocity (m/s)')
    pth = out_dir / ('velocity_map_%s.png' % _safe(args.down_label))
    f.tight_layout(); f.savefig(pth, dpi=150, bbox_inches='tight'); plt.close(f)
    saved.append(pth)

    # 2) upward map (own figure)
    f, a = plt.subplots(figsize=(6, 7))
    im = a.imshow(up, cmap='plasma', vmin=0, vmax=vmax,
                  origin='upper', extent=ext, aspect='equal')
    a.set_xlabel('x (mm)'); a.set_ylabel('y (mm)')
    f.colorbar(im, ax=a, fraction=0.046, pad=0.04).set_label('Local arc velocity (m/s)')
    pth = out_dir / ('velocity_map_%s.png' % _safe(args.up_label))
    f.tight_layout(); f.savefig(pth, dpi=150, bbox_inches='tight'); plt.close(f)
    saved.append(pth)

    # 3) difference map (own figure)
    f, a = plt.subplots(figsize=(6, 7))
    im = a.imshow(diff, cmap='RdBu_r', vmin=-dlim, vmax=dlim,
                  origin='upper', extent=ext, aspect='equal')
    a.set_xlabel('x (mm)'); a.set_ylabel('y (mm)')
    f.colorbar(im, ax=a, fraction=0.046, pad=0.04).set_label('delta velocity (m/s)')
    pth = out_dir / ('velocity_map_difference_%s-%s.png'
                     % (_safe(args.up_label), _safe(args.down_label)))
    f.tight_layout(); f.savefig(pth, dpi=150, bbox_inches='tight'); plt.close(f)
    saved.append(pth)

    # 4) velocity distribution histogram (own figure)
    f, a = plt.subplots(figsize=(7, 5))
    for label, vals, color in [(args.down_label, d_vals, 'tab:blue'),
                               (args.up_label,   u_vals, 'tab:orange')]:
        a.hist(np.clip(vals, 0, vmax), bins=bins, density=True,
               histtype='stepfilled', alpha=0.45, color=color,
               label=label)
        a.axvline(np.mean(vals), color=color, linestyle='--', linewidth=1.5)
    a.set_xlim(0, vmax)
    a.set_xlabel('Local arc velocity (m/s)')
    a.set_ylabel('Probability density')
    a.legend()
    pth = out_dir / 'velocity_distribution_hist.png'
    f.tight_layout(); f.savefig(pth, dpi=150, bbox_inches='tight'); plt.close(f)
    saved.append(pth)

    print('Saved %d standalone panel figures in %s' % (len(saved) - 1, out_dir))
    for pth in saved[1:]:
        print('  %s' % pth.name)


def main():
    ap = argparse.ArgumentParser(description='2-D velocity map from local_vel_ms')
    # --- build mode (default): construct a velocity_map_local.npy from a CSV+TIFF ---
    ap.add_argument('--csv', help='measurements_all_events.csv (has local_vel_ms) [build mode]')
    ap.add_argument('--tiff', help='single chunk TIFF')
    ap.add_argument('--tiff-dir', help='directory of ordered chunk TIFFs (multi-chunk)')
    ap.add_argument('--cine', help='read frames straight from this .cine (global frame = cine frame)')
    ap.add_argument('--dark-guard', type=float, default=None, help='binarize dark guard (12-bit value for --cine)')
    ap.add_argument('--background', default=None, help='.npy background subtracted from cine frames')
    ap.add_argument('--low-thresh', type=float, default=None, help='hysteresis low threshold (with --thresh as high)')
    ap.add_argument('--low-frac', type=float, default=None, help='low threshold floor as a fraction of frame max')
    ap.add_argument('--frames-per-tiff', type=int, default=10000,
                    help='chunk size used at conversion (for global-frame->page mapping)')
    ap.add_argument('--out-dir', help='output dir [build mode]')
    ap.add_argument('--px-per-mm', type=float, default=8.8)
    ap.add_argument('--fps', type=float, default=FPS_DEFAULT)
    ap.add_argument('--thresh', type=float, default=100,
                    help='DN threshold for RE-binarising frames to paint velocity onto. '
                         'MUST match the analyze stage (run_j_co2 passes the per-recording '
                         'bright_thresh) or the wrong pixels are painted / the high-flow '
                         'plume blob is rejected and the map comes out empty.')
    ap.add_argument('--close-radius', type=int, default=3)
    ap.add_argument('--min-arc-px', type=int, default=22)
    ap.add_argument('--max-arc-px', type=int, default=3000,
                    help='Max arc AREA (px) for re-binarisation — MUST match the analyze '
                         'stage. The BLADE default 3000 rejects the bright high-flow CO2 '
                         'plume at low --thresh, which left the velocity map empty; '
                         'run_j_co2 passes the per-recording value.')
    ap.add_argument('--frame-offset', type=int, default=0,
                    help='global frame of page 0 (single-TIFF mode only)')
    ap.add_argument('--central-map', default=None)

    # --- compare mode: contrast two pre-built velocity_map_local.npy maps ---
    ap.add_argument('--compare', action='store_true',
                    help='Compare two pre-built velocity maps (e.g. downward vs upward) '
                         'on identical spatial extent and a shared, pooled color/value scale. '
                         'Requires --down-map and --up-map.')
    ap.add_argument('--down-map', help='velocity_map_local.npy for the "downward" run [compare mode]')
    ap.add_argument('--up-map',   help='velocity_map_local.npy for the "upward" run [compare mode]')
    ap.add_argument('--down-label', default='downward', help='legend/title label for --down-map')
    ap.add_argument('--up-label',   default='upward',   help='legend/title label for --up-map')
    ap.add_argument('--compare-out', default=None,
                    help='output PNG path for the comparison figure '
                         '[compare mode; default: ./velocity_map_distribution_comparison.png]')
    ap.add_argument('--pct', type=float, default=97.0,
                    help='percentile (pooled across BOTH maps) used to set the shared vmax / '
                         'histogram upper range [compare mode; default: 97]')
    ap.add_argument('--bins', type=int, default=60,
                    help='number of histogram bins [compare mode; default: 60]')
    ap.add_argument('--diff-rotate-up', action=argparse.BooleanOptionalAction, default=True,
                    help='rotate the upward map 180 deg before differencing so the two '
                         'physically-flipped recordings overlay (affects the difference '
                         'panel only; the side-by-side maps stay as captured) [default: on]')
    args = ap.parse_args()

    # Dispatch: compare mode reads pre-built maps and returns early.
    if args.compare:
        compare_maps(args)
        return

    # ---------------------------- build mode ----------------------------
    if not args.csv or not args.out_dir:
        raise SystemExit('build mode requires --csv and --out-dir '
                         '(or use --compare with --down-map/--up-map).')
    from cine_frames import open_frames   # lazy: only the build path reads frames
    from analyze_blade import binarize   # lazy: only the build path needs the CV stack

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    if 'local_vel_ms' not in df.columns:
        raise SystemExit("ERROR: 'local_vel_ms' not in CSV - run event_velocity.py (Stage 4) first.")
    df['local_vel_ms'] = pd.to_numeric(df['local_vel_ms'], errors='coerce')
    df['frame'] = pd.to_numeric(df['frame'], errors='coerce')
    # Apply the shared flash criterion (arc_filter, criterion 3) before painting,
    # so the map cannot show frames excluded elsewhere. local_vel_ms is already
    # flash-free upstream; this is the explicit, documented guard at this stage.
    import arc_filter
    df = df[~arc_filter.flash_mask(df)]
    # global-frame -> local velocity (advancing-front frames only have a value)
    vel_by_global = {int(r['frame']): float(r['local_vel_ms'])
                     for _, r in df.dropna(subset=['local_vel_ms']).iterrows()
                     if float(r['local_vel_ms']) > 0}

    # resolve chunk list: (tiff_path, frame_offset)
    chunks = []
    if args.tiff_dir:
        tiffs = sorted(Path(args.tiff_dir).glob('*.tiff'))
        if not tiffs:
            raise SystemExit('No TIFFs in %s' % args.tiff_dir)
        chunks = [(t, k * args.frames_per_tiff) for k, t in enumerate(tiffs)]
    elif args.tiff:
        chunks = [(Path(args.tiff), args.frame_offset)]
    elif args.cine:
        chunks = [(Path(args.cine), 0)]
    else:
        raise SystemExit('Provide --tiff, --tiff-dir or --cine.')

    vel_sum = vel_cnt = None
    used = 0
    for tpath, offset in chunks:
        t = open_frames(str(tpath), background=args.background)
        if True:
            n = t.n
            H, W = t.shape
            if vel_sum is None:
                vel_sum = np.zeros((H, W)); vel_cnt = np.zeros((H, W))
            for page in range(n):
                v = vel_by_global.get(page + offset)
                if v is None:
                    continue
                raw = t.frame(page)
                kw = {} if args.dark_guard is None else {'dark_guard': args.dark_guard}
                kw['low_thresh'] = args.low_thresh
                kw['low_frac'] = args.low_frac
                b = binarize(raw, args.close_radius, bright_thresh=args.thresh,
                             min_arc_px=args.min_arc_px, max_arc_px=args.max_arc_px, **kw)
                if b.any():
                    vel_sum[b] += v; vel_cnt[b] += 1
                    used += 1
        print('  chunk %s (offset %d): cumulative advancing-front frames used=%d'
              % (tpath.name, offset, used))

    H, W = vel_sum.shape
    vmap = np.full((H, W), np.nan)   # m/s
    m = vel_cnt > 0
    vmap[m] = vel_sum[m] / vel_cnt[m]
    np.save(out_dir / 'velocity_map_local.npy', vmap * 1000.0)   # mm/s for pipeline consistency
    np.save(out_dir / 'velocity_map_count.npy', vel_cnt)          # frames per pixel (for a minimum-count mask)
    n_px = int(m.sum())
    print("Local-velocity map: %d advancing-front frames contributed, %d pixels populated"
          % (used, n_px))
    if n_px:
        f = vmap[m]
        print("  local velocity over map: mean=%.2f  p95=%.2f m/s" % (np.mean(f), np.percentile(f, 95)))

    ext = [0, W / args.px_per_mm, H / args.px_per_mm, 0]
    fig, ax = plt.subplots(figsize=(6, 6))
    vmax = np.nanpercentile(vmap, 95) if n_px else 1.0
    im = ax.imshow(vmap, cmap='plasma', vmin=0, vmax=max(vmax, 0.01), origin='upper', extent=ext)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Local arc velocity (m/s)')
    ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
    fig.tight_layout(); fig.savefig(out_dir / 'velocity_map_local.png', dpi=150); plt.close(fig)

    # comparison vs central-difference (m/s display)
    central = None
    cpath = Path(args.central_map) if args.central_map else None
    if cpath is None:
        cands = list(out_dir.glob('*_results/velocity_map.npy'))
        if cands:
            cpath = cands[0]
    if cpath and cpath.exists():
        central = np.load(cpath)
        if np.nanmax(central) > 100:
            central = central / 1000.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, dat, title in [(axes[0], central, 'central-difference (raw, Stage 2, chunk 1)'),
                           (axes[1], vmap, 'local sliding-window (primary, all chunks)')]:
        if dat is None:
            ax.text(0.5, 0.5, 'central map not found', ha='center', va='center'); ax.axis('off'); continue
        vm = np.nanpercentile(dat, 95) if np.isfinite(dat).any() else 1.0
        im = ax.imshow(dat, cmap='plasma', vmin=0, vmax=max(vm, 0.01), origin='upper', extent=ext)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('m/s')
    fig.tight_layout(); fig.savefig(out_dir / 'velocity_map_comparison.png', dpi=150); plt.close(fig)
    print("Saved velocity_map_local.npy/.png and velocity_map_comparison.png in %s" % out_dir)


if __name__ == '__main__':
    main()
