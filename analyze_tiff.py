"""
analyze_tiff.py
---------------
Main analysis script for plasma arc .tiff recordings.

Usage
~~~~~
    python analyze_tiff.py recording.tiff
    python analyze_tiff.py recording.tiff --fps 200000
    python analyze_tiff.py recording.tiff --electrode-gap-mm 2.4 --opening-angle 15
    python analyze_tiff.py recording.tiff --qa-interval 200 --out-dir results/
    python analyze_tiff.py recording.tiff --pixel-size 0.02  # mm/px explicit

Outputs (all in <out-dir>/)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
  measurements.csv        – per-frame arc metrics
  summary_dashboard.png   – multi-panel time-series + nephrogram
  nephrogram_vertical.png – raw vertical nephrogram image
  nephrogram_horizontal.png
  velocity_map.png        – velocity distribution map
  qa_frames/qa_NNNNNN.png – annotated QA frames every --qa-interval frames
  METHODS.md              – living methods document (copied/updated alongside code)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from plasma_utils.io import load_recording
from plasma_utils.image import (
    normalize_frame, binarize, measure_frame,
    estimate_px_per_mm, ArcMeasurement,
)
from plasma_utils.nephrogram import (
    build_nephrogram, radon_velocity,
    velocity_distribution_map, detect_cycles,
)
from plasma_utils.qa import save_qa_frames, save_summary_dashboard


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Plasma arc analysis: TIFF → measurements + nephrogram + QA frames',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('tiff', help='Input .tiff (or .cine) recording')
    p.add_argument('--fps', type=float, default=None,
                   help='Frame rate (Hz). Auto-detected from TIFF metadata if not set. '
                        'Falls back to 200000.')
    p.add_argument('--pixel-size', type=float, default=None,
                   help='Physical pixel size (mm/px). If omitted, calibrated '
                        'automatically from --electrode-gap-mm.')
    p.add_argument('--electrode-gap-mm', type=float, default=2.4,
                   help='Minimum physical electrode separation (mm).')
    p.add_argument('--opening-angle', type=float, default=15.0,
                   help='Electrode opening half-angle (degrees). Stored in metadata.')
    p.add_argument('--close-r', type=int, default=3,
                   help='Morphological closing radius (px).')
    p.add_argument('--open-r', type=int, default=2,
                   help='Morphological opening radius (px).')
    p.add_argument('--qa-interval', type=int, default=200,
                   help='Save a QA frame every N frames.')
    p.add_argument('--out-dir', type=str, default=None,
                   help='Output directory (default: <recording_stem>_analysis/).')
    p.add_argument('--nephrogram-axis', choices=['vertical', 'horizontal', 'both'],
                   default='vertical',
                   help='Spatial projection axis for nephrogram.')
    p.add_argument('--no-velocity-map', action='store_true',
                   help='Skip (slow) velocity distribution map computation.')
    p.add_argument('--max-frames', type=int, default=None,
                   help='Process only the first N frames (for quick tests).')
    p.add_argument('--batch-size', type=int, default=500,
                   help='Process frames in batches to manage memory.')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Frame processing
# ---------------------------------------------------------------------------

def process_frames(
    frames: np.ndarray,
    fps: float,
    close_r: int,
    open_r: int,
    px_per_mm_guess: float | None = None,
) -> tuple[list[ArcMeasurement], np.ndarray]:
    """Run binarise + measure on every frame.

    Returns
    -------
    measurements : list of ArcMeasurement
    norm_frames  : (N, H, W) float32 array of normalised frames
    """
    N = len(frames)
    norm_frames = np.empty((N,) + frames[0].shape, dtype=np.float32)
    measurements: list[ArcMeasurement] = []

    for i, raw in enumerate(tqdm(frames, desc='Measuring frames', unit='fr')):
        gray = normalize_frame(raw)
        norm_frames[i] = gray
        binary = binarize(gray, close_r, open_r)
        m = measure_frame(gray, binary, px_per_mm=px_per_mm_guess,
                          width_sample_step=5,
                          close_r=close_r, open_r=open_r)
        measurements.append(m)

    return measurements, norm_frames


# ---------------------------------------------------------------------------
# Build DataFrame
# ---------------------------------------------------------------------------

def measurements_to_df(
    measurements: list[ArcMeasurement],
    fps: float,
    px_per_mm: float | None,
) -> pd.DataFrame:
    rows = []
    for i, m in enumerate(measurements):
        rows.append({
            'frame': i,
            'time_s': i / fps if fps > 0 else i,
            'arc_detected': m.arc_detected,
            'area_px': m.area_px,
            'arc_length_px': m.arc_length_px,
            'arc_length_mm': m.arc_length_mm,
            'arc_width_px_mean': m.arc_width_px_mean,
            'arc_width_px_median': m.arc_width_px_median,
            'arc_width_mm_mean': m.arc_width_mm_mean,
            'arc_width_mm_median': m.arc_width_mm_median,
            'mean_intensity': m.mean_intensity,
            'peak_intensity': m.peak_intensity,
            'integrated_intensity': m.integrated_intensity,
            'centroid_row': m.centroid_row,
            'centroid_col': m.centroid_col,
            'tip_a_row': m.tip_a_row,
            'tip_a_col': m.tip_a_col,
            'tip_b_row': m.tip_b_row,
            'tip_b_col': m.tip_b_col,
            'electrode_gap_px': m.electrode_gap_px,
            'electrode_gap_mm': m.electrode_gap_mm,
            'curvature_mean': m.curvature_mean,
            'skeleton_n_pixels': m.skeleton_n_pixels,
            'bbox_row_min': m.bbox_row_min,
            'bbox_row_max': m.bbox_row_max,
            'bbox_col_min': m.bbox_col_min,
            'bbox_col_max': m.bbox_col_max,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    t0 = time.time()

    tiff_path = Path(args.tiff)
    if not tiff_path.exists():
        sys.exit(f"ERROR: File not found: {tiff_path}")

    out_dir = Path(args.out_dir) if args.out_dir else tiff_path.parent / (tiff_path.stem + '_analysis')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # ---- load frames ----
    print(f"Loading {tiff_path.name} …")
    frames, fps_meta = load_recording(tiff_path, default_fps=200_000)
    fps = args.fps if args.fps is not None else fps_meta
    print(f"  Frames: {len(frames)}  |  FPS: {fps:,.0f}  |  Shape: {frames[0].shape}")

    if args.max_frames:
        frames = frames[:args.max_frames]
        print(f"  (truncated to {args.max_frames} frames for this run)")

    N = len(frames)

    # ---- pixel calibration (first pass, no px_per_mm yet) ----
    print("Pass 1/2: measuring frames for calibration …")
    measurements_pass1, norm_frames = process_frames(
        frames, fps, args.close_r, args.open_r, px_per_mm_guess=None,
    )

    # derive px/mm
    if args.pixel_size is not None:
        px_per_mm = 1.0 / args.pixel_size   # args.pixel_size is mm/px
        print(f"  Using explicit pixel size: {args.pixel_size:.4f} mm/px → {px_per_mm:.2f} px/mm")
    else:
        px_per_mm = estimate_px_per_mm(
            measurements_pass1,
            electrode_gap_mm=args.electrode_gap_mm,
            percentile=5.0,
        )
        if px_per_mm:
            print(f"  Auto-calibrated: {px_per_mm:.2f} px/mm  "
                  f"(from {args.electrode_gap_mm:.2f} mm electrode gap)")
        else:
            print("  WARNING: could not auto-calibrate px/mm (too few detected arcs).")

    # ---- pass 2: re-measure with calibration ----
    print("Pass 2/2: re-measuring with physical calibration …")
    measurements: list[ArcMeasurement] = []
    for i, raw in enumerate(tqdm(norm_frames, desc='Re-measuring', unit='fr')):
        binary = binarize(raw, args.close_r, args.open_r)
        m = measure_frame(raw, binary, px_per_mm=px_per_mm,
                          width_sample_step=5,
                          close_r=args.close_r, open_r=args.open_r)
        measurements.append(m)

    df = measurements_to_df(measurements, fps, px_per_mm)
    csv_path = out_dir / 'measurements.csv'
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ---- detection rate ----
    det_rate = df['arc_detected'].mean() * 100
    print(f"  Arc detected in {det_rate:.1f}% of frames")

    # ---- nephrogram(s) ----
    print("Building nephrogram …")
    axes_to_run = (['vertical', 'horizontal']
                   if args.nephrogram_axis == 'both'
                   else [args.nephrogram_axis])
    neph_results = {}
    for ax_name in axes_to_run:
        ng = build_nephrogram(norm_frames, axis=ax_name, mode='max',
                              normalise=True, smooth_sigma=0.8)
        neph_results[ax_name] = ng
        ng_img_path = out_dir / f'nephrogram_{ax_name}.png'
        _save_nephrogram_image(ng, ng_img_path, fps, px_per_mm, axis=ax_name)
        print(f"  Saved: {ng_img_path}")

    primary_ng = neph_results[axes_to_run[0]]

    # ---- cycle detection ----
    print("Detecting propagation cycles …")
    cycle_info = detect_cycles(primary_ng, fps, min_cycle_frames=int(fps * 1e-5))
    n_cycles = len(cycle_info['cycle_frames'])
    print(f"  Detected {n_cycles} cycles  |  "
          f"mean period: {cycle_info['mean_period_s']*1e6:.1f} µs  |  "
          f"mean freq: {cycle_info['mean_frequency_Hz']:.0f} Hz")

    # ---- velocity map ----
    v_map_result = None
    if not args.no_velocity_map and N >= 20:
        print("Computing velocity distribution map (may be slow) …")
        try:
            window_fr = max(20, min(200, N // 20))
            step_fr   = max(5,  window_fr // 5)
            times_v, v_edges, v_map = velocity_distribution_map(
                primary_ng, fps, px_per_mm,
                window_frames=window_fr, step_frames=step_fr,
            )
            v_map_result = (times_v, v_edges, v_map)
            _save_velocity_map(times_v, v_edges, v_map, out_dir / 'velocity_map.png',
                               px_per_mm=px_per_mm, dpi=150)
            print(f"  Saved: {out_dir / 'velocity_map.png'}")
        except Exception as exc:
            print(f"  WARNING: velocity map failed ({exc}); skipping.")

    # ---- radon velocity (whole recording) ----
    if N >= 10:
        try:
            rv = radon_velocity(primary_ng, fps, px_per_mm)
            v_unit = 'mm/s' if px_per_mm else 'px/frame'
            print(f"  Dominant propagation velocity: "
                  f"{rv['dominant_velocity']:.0f} {v_unit}  "
                  f"(θ = {rv['dominant_angle_deg']:.1f}°)")
        except Exception as exc:
            print(f"  WARNING: Radon velocity failed ({exc})")

    # ---- summary dashboard ----
    print("Saving summary dashboard …")
    dash_path = out_dir / 'summary_dashboard.png'
    save_summary_dashboard(
        df, dash_path, fps,
        px_per_mm=px_per_mm,
        nephrogram=primary_ng,
        v_map_result=v_map_result,
        cycle_info=cycle_info,
        dpi=150,
    )
    print(f"  Saved: {dash_path}")

    # ---- QA frames ----
    print(f"Saving QA frames (every {args.qa_interval} frames) …")
    saved_qa = save_qa_frames(
        norm_frames, measurements, out_dir, fps,
        px_per_mm=px_per_mm,
        qa_interval=args.qa_interval,
        close_r=args.close_r, open_r=args.open_r,
        dpi=150,
    )
    print(f"  Saved {len(saved_qa)} QA frames to {out_dir / 'qa_frames'}/")

    # ---- print summary statistics ----
    elapsed = time.time() - t0
    _print_summary(df, fps, px_per_mm, args.electrode_gap_mm, cycle_info, elapsed)

    print(f"\nAll outputs saved to: {out_dir}/")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_nephrogram_image(
    ng: np.ndarray,
    path: Path,
    fps: float,
    px_per_mm: float | None,
    axis: str = 'vertical',
    dpi: int = 150,
):
    """Save the 2-D nephrogram as a publication-quality PNG."""
    N, S = ng.shape
    t_max = N / fps if fps > 0 else N
    t_max_us = t_max * 1e6

    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    extent = [0, S if px_per_mm is None else S / px_per_mm, t_max_us, 0]
    img = ax.imshow(ng, aspect='auto', cmap='inferno', origin='upper',
                    extent=extent, vmin=0, vmax=1)
    cbar = plt.colorbar(img, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Normalised intensity', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=8)

    x_label = f'Spatial position ({"mm" if px_per_mm else "px"})'
    ax.set_xlabel(x_label, color='white', fontsize=10)
    ax.set_ylabel('Time (us)', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#555577')

    plt.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_velocity_map(
    times: np.ndarray,
    v_edges: np.ndarray,
    v_map: np.ndarray,
    path: Path,
    px_per_mm: float | None = None,
    dpi: int = 150,
):
    v_unit = 'mm/s' if px_per_mm else 'px/frame'
    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    extent = [v_edges[0], v_edges[-1], times[-1] * 1e6, times[0] * 1e6]
    img = ax.imshow(v_map, aspect='auto', cmap='hot', origin='upper', extent=extent)
    cbar = plt.colorbar(img, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Norm. spectral weight', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=8)
    ax.set_xlabel(f'Velocity ({v_unit})', color='white', fontsize=10)
    ax.set_ylabel('Time (us)', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#555577')
    plt.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def _print_summary(df, fps, px_per_mm, electrode_gap_mm, cycle_info, elapsed):
    det = df[df['arc_detected']]
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Total frames       : {len(df)}")
    print(f"  Frame rate         : {fps:,.0f} Hz")
    print(f"  Recording duration : {len(df)/fps*1e3:.2f} ms  ({len(df)/fps*1e6:.0f} µs)")
    print(f"  Arc detected       : {len(det)} / {len(df)} frames ({len(det)/len(df)*100:.1f}%)")
    if px_per_mm:
        print(f"  Calibration        : {px_per_mm:.2f} px/mm  ({1/px_per_mm:.4f} mm/px)")
        print(f"  Physical scale ref : {electrode_gap_mm:.2f} mm electrode gap")
    col_l = 'arc_length_mm' if (px_per_mm and 'arc_length_mm' in det.columns) else 'arc_length_px'
    col_w = 'arc_width_mm_mean' if (px_per_mm and 'arc_width_mm_mean' in det.columns) else 'arc_width_px_mean'
    u = 'mm' if px_per_mm else 'px'
    if len(det):
        print(f"  Arc length range   : {det[col_l].min():.2f} – {det[col_l].max():.2f} {u}")
        print(f"  Arc width (mean)   : {det[col_w].mean():.2f} ± {det[col_w].std():.2f} {u}")
        print(f"  Peak brightness    : {det['peak_intensity'].max():.4f}")
        print(f"  Mean brightness    : {det['mean_intensity'].mean():.4f}")
    n_cyc = len(cycle_info.get('cycle_frames', []))
    if n_cyc > 0:
        print(f"  Propagation cycles : {n_cyc}")
        print(f"  Mean cycle period  : {cycle_info['mean_period_s']*1e6:.1f} µs")
        print(f"  Mean cycle freq    : {cycle_info['mean_frequency_Hz']:.0f} Hz")
    print(f"  Wall time          : {elapsed:.1f} s")
    print("=" * 60)


if __name__ == '__main__':
    main()
