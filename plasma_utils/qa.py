"""
plasma_utils/qa.py
------------------
Quality-assurance frame rendering for plasma arc analysis.

Each QA frame overlays the following on a false-colour (grayscale) render
of the raw normalised image:

  * Arc mask outline (semi-transparent magenta fill + white contour)
  * Skeleton centreline (cyan)
  * Arc tip markers (yellow circles)
  * Intensity-weighted centroid (green cross)
  * Perpendicular width sample lines (thin orange segments)
  * Bounding box (dashed white rectangle)
  * Text panel: frame number, time, length, width, brightness, gap

Visual style: dark background, bright overlays, no axes clutter.
Consistent with high-speed imaging publications.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

from plasma_utils.image import (
    ArcMeasurement,
    normalize_frame,
    binarize,
    _skeleton_ordered_coords,
    _fwhm_perpendicular,
)
from skimage.morphology import skeletonize


# ---------------------------------------------------------------------------
# Single QA frame
# ---------------------------------------------------------------------------

def render_qa_frame(
    raw_frame: np.ndarray,
    meas: ArcMeasurement,
    frame_idx: int,
    fps: float,
    px_per_mm: Optional[float] = None,
    close_r: int = 3,
    open_r: int = 2,
    width_sample_step: int = 5,
    dpi: int = 150,
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """Render a single annotated QA frame.

    Parameters
    ----------
    raw_frame  : 2-D array (any dtype) — the original frame
    meas       : :class:`ArcMeasurement` for this frame
    frame_idx  : integer frame index
    fps        : recording frame rate (for time label)
    px_per_mm  : optional calibration for physical labels
    close_r, open_r, width_sample_step : passed to binarize / width routines

    Returns
    -------
    matplotlib Figure (caller is responsible for saving / closing)
    """
    gray = normalize_frame(raw_frame)
    binary = binarize(gray, close_r, open_r)
    skel = skeletonize(binary)
    path = _skeleton_ordered_coords(skel)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')

    # --- base image (grayscale, gamma-stretched for visibility) ---
    disp = np.power(np.clip(gray, 0, 1), 0.5)   # gamma=0.5 brightens dim regions
    ax.imshow(disp, cmap='gray', vmin=0, vmax=1, origin='upper', interpolation='nearest')

    H, W = gray.shape

    # --- arc mask (semi-transparent magenta fill) ---
    if binary.any():
        mask_rgba = np.zeros((H, W, 4), dtype=np.float32)
        mask_rgba[binary, 0] = 0.9   # R
        mask_rgba[binary, 2] = 0.9   # B  → magenta
        mask_rgba[binary, 3] = 0.25  # alpha
        ax.imshow(mask_rgba, origin='upper', interpolation='nearest')

    # --- skeleton (cyan) ---
    if skel.any():
        skel_rows, skel_cols = np.where(skel)
        ax.scatter(skel_cols, skel_rows, c='cyan', s=1, linewidths=0,
                   alpha=0.8, rasterized=True, zorder=3)

    # --- perpendicular width samples (orange segments) ---
    if path is not None and len(path) >= 3:
        sample_indices = range(0, len(path), max(1, width_sample_step))
        for i in sample_indices:
            r, c = path[i]
            if 0 < i < len(path) - 1:
                tan = path[i + 1].astype(float) - path[i - 1].astype(float)
            elif i == 0:
                tan = path[1].astype(float) - path[0].astype(float)
            else:
                tan = path[-1].astype(float) - path[-2].astype(float)
            tn = np.linalg.norm(tan)
            if tn < 1e-9:
                continue
            tan /= tn
            perp = np.array([-tan[1], tan[0]])
            hw = 12   # half-width of displayed sample line (px)
            r0 = r + hw * perp[0]; c0 = c + hw * perp[1]
            r1 = r - hw * perp[0]; c1 = c - hw * perp[1]
            ax.plot([c0, c1], [r0, r1], color='#FF8800', lw=0.8, alpha=0.7, zorder=4)

    # --- arc tips (yellow circles) ---
    if meas.arc_detected and not np.isnan(meas.tip_a_row):
        ax.scatter([meas.tip_a_col, meas.tip_b_col],
                   [meas.tip_a_row, meas.tip_b_row],
                   c='yellow', s=60, zorder=6, edgecolors='black', linewidths=0.5,
                   label='Tips')
        # electrode gap line
        ax.plot([meas.tip_a_col, meas.tip_b_col],
                [meas.tip_a_row, meas.tip_b_row],
                color='yellow', lw=0.6, ls='--', alpha=0.5, zorder=5)

    # --- centroid (green cross) ---
    if meas.arc_detected and not np.isnan(meas.centroid_row):
        ax.scatter([meas.centroid_col], [meas.centroid_row],
                   marker='+', c='lime', s=120, lw=1.5, zorder=7)

    # --- bounding box (dashed white) ---
    if meas.arc_detected:
        bw = meas.bbox_col_max - meas.bbox_col_min
        bh = meas.bbox_row_max - meas.bbox_row_min
        rect = mpatches.Rectangle(
            (meas.bbox_col_min, meas.bbox_row_min), bw, bh,
            linewidth=0.8, edgecolor='white', facecolor='none',
            linestyle='--', zorder=6,
        )
        ax.add_patch(rect)

    # --- text panel ---
    t_s = frame_idx / fps if fps > 0 else 0
    t_us = t_s * 1e6
    lines = [
        f"Frame {frame_idx:06d}   t = {t_us:.2f} µs",
        "",
    ]
    if meas.arc_detected:
        if px_per_mm and not np.isnan(meas.arc_length_mm):
            lines.append(f"Length : {meas.arc_length_mm:.3f} mm")
            lines.append(f"Width  : {meas.arc_width_mm_mean:.3f} mm (mean FWHM)")
        else:
            lines.append(f"Length : {meas.arc_length_px:.0f} px")
            lines.append(f"Width  : {meas.arc_width_px_mean:.1f} px (mean FWHM)")
        lines.append(f"Bright : {meas.mean_intensity:.4f} (norm.)")
        lines.append(f"Peak   : {meas.peak_intensity:.4f}")
        lines.append(f"Integr.: {meas.integrated_intensity:.2f}")
        if not np.isnan(meas.electrode_gap_mm) and px_per_mm:
            lines.append(f"Gap    : {meas.electrode_gap_mm:.3f} mm")
        else:
            lines.append(f"Gap    : {meas.electrode_gap_px:.1f} px")
        lines.append(f"Curve  : {meas.curvature_mean:.4f} rad/px")
        lines.append(f"Area   : {meas.area_px} px²")
    else:
        lines.append("No arc detected")

    text_str = "\n".join(lines)
    ax.text(
        0.01, 0.99, text_str,
        transform=ax.transAxes, va='top', ha='left',
        fontsize=7, color='white',
        fontfamily='monospace',
        bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=3),
        zorder=10,
    )

    # --- legend ---
    handles = [
        mpatches.Patch(color='cyan', label='Skeleton'),
        mpatches.Patch(color='#FF6699', alpha=0.5, label='Arc mask'),
        plt.Line2D([0], [0], color='#FF8800', lw=1, label='Width sample'),
        plt.Line2D([0], [0], marker='o', color='yellow', ms=5,
                   linestyle='none', label='Tips'),
        plt.Line2D([0], [0], marker='+', color='lime', ms=6, lw=1.5,
                   linestyle='none', label='Centroid'),
    ]
    ax.legend(handles=handles, loc='lower right', fontsize=6,
              framealpha=0.6, facecolor='black', edgecolor='none',
              labelcolor='white')

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    plt.tight_layout(pad=0.2)
    return fig


# ---------------------------------------------------------------------------
# Batch QA frame writer
# ---------------------------------------------------------------------------

def save_qa_frames(
    frames: np.ndarray,
    measurements: list[ArcMeasurement],
    out_dir: Path,
    fps: float,
    px_per_mm: Optional[float] = None,
    qa_interval: int = 100,
    close_r: int = 3,
    open_r: int = 2,
    width_sample_step: int = 5,
    dpi: int = 150,
) -> list[Path]:
    """Save QA frames every *qa_interval* frames.

    Parameters
    ----------
    frames       : (N, H, W) float32 normalised frame stack
    measurements : list of ArcMeasurement (one per frame)
    out_dir      : directory for output PNGs
    fps          : recording frame rate
    px_per_mm    : optional spatial calibration
    qa_interval  : save every this many frames (and always save frame 0)

    Returns
    -------
    list of saved file paths
    """
    out_dir = Path(out_dir)
    (out_dir / 'qa_frames').mkdir(parents=True, exist_ok=True)

    N = len(frames)
    indices = list(range(0, N, qa_interval))
    if N - 1 not in indices:
        indices.append(N - 1)

    saved = []
    for idx in indices:
        fig = render_qa_frame(
            frames[idx], measurements[idx], idx, fps,
            px_per_mm=px_per_mm,
            close_r=close_r, open_r=open_r,
            width_sample_step=width_sample_step,
            dpi=dpi,
        )
        path = out_dir / 'qa_frames' / f'qa_{idx:06d}.png'
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        saved.append(path)

    return saved


# ---------------------------------------------------------------------------
# Summary dashboard
# ---------------------------------------------------------------------------

def save_summary_dashboard(
    df,            # pandas DataFrame from analyze_tiff.py
    out_path: Path,
    fps: float,
    px_per_mm: Optional[float] = None,
    nephrogram: Optional[np.ndarray] = None,
    v_map_result: Optional[tuple] = None,
    cycle_info: Optional[dict] = None,
    dpi: int = 150,
) -> Path:
    """Save a multi-panel summary figure.

    Panels
    ------
    Row 1: Arc length, Arc width (mean FWHM)
    Row 2: Mean brightness, Integrated intensity
    Row 3: Centroid row position, Electrode gap
    Row 4: Nephrogram (if provided)
    Row 5: Velocity distribution map (if provided)
    """
    import pandas as pd

    t = df['time_s'].values
    use_mm = px_per_mm is not None and 'arc_length_mm' in df.columns

    n_rows = 3
    if nephrogram is not None:
        n_rows += 1
    if v_map_result is not None:
        n_rows += 1

    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3.5 * n_rows), dpi=dpi)
    fig.patch.set_facecolor('#111111')

    def _style(ax, xlabel='', ylabel='', title=''):
        ax.set_facecolor('#1a1a2e')
        ax.tick_params(colors='#cccccc', labelsize=8)
        ax.xaxis.label.set_color('#cccccc')
        ax.yaxis.label.set_color('#cccccc')
        if title:
            pass
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=8)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    t_label = 'Time (s)'

    # Row 0: length + width
    col_len = 'arc_length_mm' if use_mm else 'arc_length_px'
    col_wid = 'arc_width_mm_mean' if use_mm else 'arc_width_px_mean'
    unit_l = 'mm' if use_mm else 'px'

    ax = axes[0, 0]
    ax.plot(t, df[col_len], color='#00BFFF', lw=0.6)
    _style(ax, t_label, f'Length ({unit_l})', 'Arc Length')

    ax = axes[0, 1]
    ax.plot(t, df[col_wid], color='#FF8C00', lw=0.6)
    _style(ax, t_label, f'Width FWHM ({unit_l})', 'Arc Width (mean FWHM)')

    # Row 1: brightness
    ax = axes[1, 0]
    ax.plot(t, df['mean_intensity'], color='#ADFF2F', lw=0.6)
    _style(ax, t_label, 'Norm. intensity', 'Mean Brightness')

    ax = axes[1, 1]
    ax.plot(t, df['integrated_intensity'], color='#DA70D6', lw=0.6)
    _style(ax, t_label, 'Sum (norm. px)', 'Integrated Intensity')

    # Row 2: centroid + gap
    ax = axes[2, 0]
    ax.plot(t, df['centroid_row'], color='#7FFFD4', lw=0.6)
    _style(ax, t_label, 'Row (px)', 'Centroid Row Position')

    col_gap = 'electrode_gap_mm' if (use_mm and 'electrode_gap_mm' in df.columns) else 'electrode_gap_px'
    ax = axes[2, 1]
    ax.plot(t, df[col_gap], color='#FFD700', lw=0.6)
    _style(ax, t_label, f'Gap ({unit_l})', 'Electrode Gap')

    row = 3

    # Nephrogram panel
    if nephrogram is not None:
        ax_ng = axes[row, 0]
        extent = [0, nephrogram.shape[1], t[-1] if len(t) else nephrogram.shape[0], 0]
        ax_ng.imshow(
            nephrogram, aspect='auto', cmap='inferno',
            origin='upper', extent=extent,
        )
        ax_ng.set_xlabel('Spatial position (px)', fontsize=8)
        ax_ng.set_ylabel('Time (s)', fontsize=8)
        ax_ng.tick_params(colors='#cccccc', labelsize=8)
        for spine in ax_ng.spines.values():
            spine.set_edgecolor('#444466')
        ax_ng.set_facecolor('#1a1a2e')

        # cycle markers
        if cycle_info and len(cycle_info.get('cycle_frames', [])) > 0:
            ct = cycle_info['cycle_times_s']
            ax_ng.hlines(ct, 0, nephrogram.shape[1], colors='cyan',
                         lw=0.5, ls='--', alpha=0.6)
            ax_ng.text(
                0.01, 0.01,
                f"Cycles: {len(ct)}  f={cycle_info['mean_frequency_Hz']:.1f} Hz",
                transform=ax_ng.transAxes, color='cyan', fontsize=7,
                va='bottom',
            )

        # temporal signal (right panel)
        ax_ts = axes[row, 1]
        if cycle_info:
            sig = cycle_info['temporal_signal']
            t_sig = np.linspace(t[0] if len(t) else 0, t[-1] if len(t) else 1, len(sig))
            ax_ts.plot(t_sig, sig, color='#ADFF2F', lw=0.6)
            if len(cycle_info['cycle_times_s']) > 0:
                ax_ts.vlines(cycle_info['cycle_times_s'], sig.min(), sig.max(),
                             colors='cyan', lw=0.5, ls='--', alpha=0.7)
            _style(ax_ts, t_label, 'Intensity', 'Cycle Detection Signal')
        else:
            ax_ts.axis('off')
        row += 1

    # Velocity distribution map
    if v_map_result is not None:
        times_v, v_edges, v_map = v_map_result
        ax_vd = axes[row, 0]
        v_centres = 0.5 * (v_edges[:-1] + v_edges[1:])
        v_unit = 'mm/s' if px_per_mm else 'px/frame'
        extent_v = [v_edges[0], v_edges[-1], times_v[-1], times_v[0]]
        ax_vd.imshow(
            v_map, aspect='auto', cmap='hot',
            origin='upper', extent=extent_v,
        )
        ax_vd.set_xlabel(f'Velocity ({v_unit})', fontsize=8)
        ax_vd.set_ylabel('Time (s)', fontsize=8)
        ax_vd.tick_params(colors='#cccccc', labelsize=8)
        for spine in ax_vd.spines.values():
            spine.set_edgecolor('#444466')
        ax_vd.set_facecolor('#1a1a2e')

        # mean velocity vs time
        ax_mv = axes[row, 1]
        mean_v = np.average(
            v_centres[np.newaxis, :].repeat(v_map.shape[0], axis=0),
            weights=v_map + 1e-12, axis=1,
        )
        ax_mv.plot(times_v, mean_v, color='#FF6347', lw=0.8)
        _style(ax_mv, t_label, f'Mean v ({v_unit})', 'Mean Propagation Velocity')
        row += 1

    plt.tight_layout(pad=0.5)
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path
