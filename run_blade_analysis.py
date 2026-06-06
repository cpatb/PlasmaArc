"""
run_blade_analysis.py
---------------------
Batch analysis of ADCB_0.8mmNoz_1000mL_N2_{1,2,3}.tiff from the BLADE dataset.

Reads each TIFF in streaming page-by-page mode (avoids loading 1.6 GB into RAM
all at once), computes per-frame arc metrics, then builds nephrogram and
outputs all results into BLADE_analysis/.

Run:
    python run_blade_analysis.py
"""

import sys, warnings, time, datetime
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tifffile
from tqdm import tqdm

from plasma_utils.image import (
    normalize_frame, binarize, measure_frame,
    estimate_px_per_mm, ArcMeasurement
)
from plasma_utils.nephrogram import (
    build_nephrogram, detect_cycles, radon_velocity,
    velocity_distribution_map
)
from plasma_utils.qa import save_qa_frames, save_summary_dashboard

# ── Config ─────────────────────────────────────────────────────────────────
BLADE_DIR   = Path.home() / 'OneDrive' / 'Desktop' / 'BLADE'
_date       = datetime.date.today().isoformat()
OUT_DIR     = BLADE_DIR / _date
FPS         = 200_000
ELEC_GAP_MM = 2.4
OPEN_ANGLE  = 15.0        # degrees (metadata only, stored in methods doc)
CLOSE_R     = 3
OPEN_R      = 2
QA_INTERVAL = 500         # one QA frame every 500 = every 2.5 ms
CALIB_N     = 2000        # frames to use for calibration pass
NEPH_SMOOTH = 0.8         # Gaussian σ for nephrogram smoothing (px)

FILES = [
    BLADE_DIR / 'ADCB_0.8mmNoz_1000mL_N2_1.tiff',
    BLADE_DIR / 'ADCB_0.8mmNoz_1000mL_N2_2.tiff',
    BLADE_DIR / 'ADCB_0.8mmNoz_1000mL_N2_3.tiff',
]

_ROI_PATH = BLADE_DIR / 'roi_polygon.npy'

# ── Helpers ─────────────────────────────────────────────────────────────────

def stream_frames(tiff_path, start=0, stop=None):
    """Yield (index, uint8_frame) from a multi-page TIFF without loading all."""
    with tifffile.TiffFile(str(tiff_path)) as tif:
        n = len(tif.pages)
        stop_ = n if stop is None else min(stop, n)
        for i in range(start, stop_):
            yield i, tif.pages[i].asarray()


def _load_roi(shape):
    """Return ROI boolean mask or None if roi_polygon.npy doesn't exist."""
    if not _ROI_PATH.exists():
        return None
    from plasma_utils.roi import load_roi_mask
    return load_roi_mask(_ROI_PATH, shape)


def measure_all_frames(tiff_path, px_per_mm, desc=""):
    """Stream-measure every frame; return (measurements, norm_frames_array)."""
    with tifffile.TiffFile(str(tiff_path)) as tif:
        n_frames = len(tif.pages)
        H, W     = tif.pages[0].asarray().shape[:2]

    roi = _load_roi((H, W))
    if roi is not None:
        print(f'  ROI mask active: {roi.sum()} / {roi.size} px')

    measurements = []
    norm_stack   = np.empty((n_frames, H, W), dtype=np.float32)

    with tifffile.TiffFile(str(tiff_path)) as tif:
        for i, page in enumerate(tqdm(tif.pages, desc=desc, unit='fr')):
            raw    = page.asarray()
            gray   = normalize_frame(raw)
            norm_stack[i] = gray
            binary = binarize(gray, CLOSE_R, OPEN_R)
            if roi is not None:
                binary = binary & roi
            m = measure_frame(gray, binary, px_per_mm=px_per_mm,
                              width_sample_step=5)
            measurements.append(m)

    return measurements, norm_stack


def meas_to_df(measurements, fps, file_offset_frames=0):
    rows = []
    for i, m in enumerate(measurements):
        fi = i + file_offset_frames
        rows.append(dict(
            frame=fi, time_s=fi/fps,
            arc_detected=m.arc_detected,
            area_px=m.area_px,
            arc_length_px=m.arc_length_px,
            arc_length_mm=m.arc_length_mm,
            arc_width_px_mean=m.arc_width_px_mean,
            arc_width_px_median=m.arc_width_px_median,
            arc_width_mm_mean=m.arc_width_mm_mean,
            arc_width_mm_median=m.arc_width_mm_median,
            mean_intensity=m.mean_intensity,
            peak_intensity=m.peak_intensity,
            integrated_intensity=m.integrated_intensity,
            centroid_row=m.centroid_row,
            centroid_col=m.centroid_col,
            tip_a_row=m.tip_a_row, tip_a_col=m.tip_a_col,
            tip_b_row=m.tip_b_row, tip_b_col=m.tip_b_col,
            electrode_gap_px=m.electrode_gap_px,
            electrode_gap_mm=m.electrode_gap_mm,
            curvature_mean=m.curvature_mean,
            skeleton_n_pixels=m.skeleton_n_pixels,
            bbox_row_min=m.bbox_row_min, bbox_row_max=m.bbox_row_max,
            bbox_col_min=m.bbox_col_min, bbox_col_max=m.bbox_col_max,
        ))
    return pd.DataFrame(rows)


def save_nephrogram_png(ng, path, fps, px_per_mm, label="", axis='vertical'):
    N, S = ng.shape
    t_max_us = N / fps * 1e6
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    x_mm = S / px_per_mm if px_per_mm else S
    extent = [0, x_mm, t_max_us, 0]
    im = ax.imshow(ng, aspect='auto', cmap='inferno', origin='upper',
                   extent=extent, vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Norm. intensity', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=8)
    xlabel = f'{"Vertical" if axis=="vertical" else "Horizontal"} position ({"mm" if px_per_mm else "px"})'
    ax.set_xlabel(xlabel, color='white', fontsize=10)
    ax.set_ylabel('Time (us)', color='white', fontsize=10)
    ax.tick_params(colors='white', labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor('#555577')
    plt.tight_layout()
    fig.savefig(path, dpi=150, facecolor='#0a0a0a')
    plt.close(fig)


def save_timeseries(df, path, px_per_mm, label=""):
    use_mm = px_per_mm is not None
    t = df['time_s'].values * 1e6   # µs

    fig, axes = plt.subplots(4, 2, figsize=(14, 14), dpi=150)
    fig.patch.set_facecolor('#111111')
    pairs = [
        ('arc_length_mm' if use_mm else 'arc_length_px', f'Arc Length ({"mm" if use_mm else "px"})', '#00BFFF'),
        ('arc_width_mm_mean' if use_mm else 'arc_width_px_mean', f'Width FWHM mean ({"mm" if use_mm else "px"})', '#FF8C00'),
        ('mean_intensity', 'Mean Brightness (norm.)', '#ADFF2F'),
        ('peak_intensity', 'Peak Brightness (norm.)', '#FF69B4'),
        ('integrated_intensity', 'Integrated Intensity', '#DA70D6'),
        ('centroid_row', 'Centroid Row (px)', '#7FFFD4'),
        ('electrode_gap_mm' if use_mm else 'electrode_gap_px', f'Electrode Gap ({"mm" if use_mm else "px"})', '#FFD700'),
        ('curvature_mean', 'Mean Curvature (rad/px)', '#FF6347'),
    ]
    for ax, (col, ylabel, color) in zip(axes.flat, pairs):
        ax.set_facecolor('#1a1a2e')
        if col in df.columns:
            vals = df[col].values
            ax.plot(t, vals, color=color, lw=0.4, alpha=0.7)
            # smoothed
            w = min(201, len(vals)//10*2+1)
            if w > 3:
                from scipy.ndimage import uniform_filter1d
                smooth = uniform_filter1d(np.where(np.isnan(vals), 0, vals).astype(float), w)
                ax.plot(t, smooth, color='white', lw=1.2, alpha=0.9)
        ax.set_ylabel(ylabel, color='#cccccc', fontsize=8)
        ax.set_xlabel('Time (us)', color='#cccccc', fontsize=8)
        ax.tick_params(colors='#cccccc', labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor('#444466')
    plt.tight_layout()
    fig.savefig(path, dpi=150, facecolor='#111111', bbox_inches='tight')
    plt.close(fig)


def save_qa_montage(norm_frames, measurements, path, fps, px_per_mm, n_qa=12):
    """Save a quick montage of evenly-spaced QA frames (no full QA overhead)."""
    from skimage.morphology import skeletonize
    N = len(norm_frames)
    idxs = np.linspace(0, N-1, n_qa, dtype=int)
    cols = 4; rows = (n_qa + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3), dpi=150)
    fig.patch.set_facecolor('#0a0a0a')
    for ax in axes.flat: ax.set_facecolor('#0a0a0a'); ax.axis('off')
    for ax, idx in zip(axes.flat, idxs):
        g = norm_frames[idx]
        b = binarize(g, CLOSE_R, OPEN_R)
        skel = skeletonize(b)
        m = measurements[idx]
        disp = np.power(np.clip(g, 0, 1), 0.5)
        rgb = np.stack([disp]*3, axis=-1)
        rgb[b, 0] = np.maximum(rgb[b, 0], 0.7)
        rgb[b, 2] = np.maximum(rgb[b, 2], 0.7)
        rgb[skel] = [0, 1, 1]
        ax.imshow(rgb, origin='upper')
        t_us = idx / fps * 1e6
        lbl = f't={t_us:.1f}µs'
        if px_per_mm and m.arc_detected:
            lbl += f'\nL={m.arc_length_mm:.2f}mm'
        elif m.arc_detected:
            lbl += f'\nL={m.arc_length_px:.0f}px'
    plt.tight_layout(pad=0.3)
    fig.savefig(path, dpi=150, facecolor='#0a0a0a')
    plt.close(fig)
    return path


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    px_per_mm_global = None   # set from file 1, reused for 2 & 3
    all_dfs = []
    t_total_start = time.time()

    for file_idx, tiff_path in enumerate(FILES):
        file_num = file_idx + 1
        print(f"\n{'='*60}")
        print(f"  FILE {file_num}/3: {tiff_path.name}")
        print(f"{'='*60}")
        t0 = time.time()

        file_out = OUT_DIR / f'file{file_num}'
        file_out.mkdir(exist_ok=True)

        # ── Calibration pass (file 1 only) ─────────────────────────────────
        if file_idx == 0:
            print(f"  Calibration pass ({CALIB_N} frames)…")
            calib_meas = []
            _calib_roi = _load_roi((256, 256))
            for _, raw in stream_frames(tiff_path, stop=CALIB_N):
                g = normalize_frame(raw)
                b = binarize(g, CLOSE_R, OPEN_R)
                if _calib_roi is not None:
                    b = b & _calib_roi
                m = measure_frame(g, b, px_per_mm=None, width_sample_step=10)
                calib_meas.append(m)
            px_per_mm_global = estimate_px_per_mm(
                calib_meas, electrode_gap_mm=ELEC_GAP_MM,
                frame_height=256, top_fraction=0.30
            )
            if px_per_mm_global:
                print(f"  Calibration: {px_per_mm_global:.2f} px/mm  "
                      f"({1/px_per_mm_global*1000:.1f} µm/px)  "
                      f"FOV = {256/px_per_mm_global:.1f} mm")
            else:
                print("  WARNING: calibration failed; no physical units.")

        # ── Full measurement pass ───────────────────────────────────────────
        print(f"  Measuring all frames…")
        measurements, norm_frames = measure_all_frames(
            tiff_path, px_per_mm_global, desc=f"  File {file_num}"
        )
        N = len(measurements)

        # offset for global frame index
        offset = file_idx * N
        df = meas_to_df(measurements, FPS, file_offset_frames=offset)
        df.to_csv(file_out / 'measurements.csv', index=False)
        all_dfs.append(df)

        det_pct = df['arc_detected'].mean() * 100
        print(f"  Arc detected: {det_pct:.1f}%  |  frames: {N:,}")

        # ── Time-series plots ────────────────────────────────────────────────
        print("  Saving time-series plots…")
        save_timeseries(df, file_out / 'timeseries.png',
                        px_per_mm_global, label=tiff_path.name)

        # ── Nephrogram ──────────────────────────────────────────────────────
        print("  Building nephrogram…")
        ng_v = build_nephrogram(norm_frames, axis='vertical',
                                mode='max', normalise=True, smooth_sigma=NEPH_SMOOTH)
        ng_h = build_nephrogram(norm_frames, axis='horizontal',
                                mode='max', normalise=True, smooth_sigma=NEPH_SMOOTH)
        save_nephrogram_png(ng_v, file_out / 'nephrogram_vertical.png',
                            FPS, px_per_mm_global, label=tiff_path.stem, axis='vertical')
        save_nephrogram_png(ng_h, file_out / 'nephrogram_horizontal.png',
                            FPS, px_per_mm_global, label=tiff_path.stem, axis='horizontal')

        # ── Cycle detection ──────────────────────────────────────────────────
        print("  Detecting cycles…")
        cycles = detect_cycles(ng_v, FPS, min_cycle_frames=int(FPS * 5e-6))
        n_cyc = len(cycles['cycle_frames'])
        if n_cyc > 1:
            print(f"  Cycles: {n_cyc}  |  mean period: "
                  f"{cycles['mean_period_s']*1e3:.2f} ms  |  "
                  f"freq: {cycles['mean_frequency_Hz']:.0f} Hz")
        else:
            print(f"  Cycles: {n_cyc} (too few for statistics)")

        # ── Radon velocity ───────────────────────────────────────────────────
        try:
            rv = radon_velocity(ng_v, FPS, px_per_mm_global)
            v_unit = 'mm/s' if px_per_mm_global else 'px/fr'
            print(f"  Dominant propagation velocity: "
                  f"{rv['dominant_velocity']:.0f} {v_unit}  "
                  f"(θ = {rv['dominant_angle_deg']:.1f}°)")
            fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=100)
            fig2.patch.set_facecolor('#0a0a0a')
            ax2.set_facecolor('#0a0a0a')
            ax2.imshow(rv['sinogram'].T, aspect='auto', cmap='hot', origin='lower',
                       extent=[rv['angles_deg'][0], rv['angles_deg'][-1],
                                0, rv['sinogram'].shape[0]])
            ax2.set_xlabel('Projection angle (deg)', color='white', fontsize=9)
            ax2.set_ylabel('Radon position', color='white', fontsize=9)
            ax2.tick_params(colors='white')
            ax2.axvline(rv['dominant_angle_deg'], color='cyan', lw=1.2, ls='--')
            fig2.tight_layout()
            fig2.savefig(file_out / 'radon_sinogram.png', dpi=100, facecolor='#0a0a0a')
            plt.close(fig2)
        except Exception as e:
            print(f"  WARNING: Radon failed: {e}")

        # ── Velocity distribution map ────────────────────────────────────────
        print("  Computing velocity distribution map…")
        try:
            window_fr = max(50, min(500, N // 40))
            step_fr   = max(10, window_fr // 5)
            tv, ve, vm = velocity_distribution_map(
                ng_v, FPS, px_per_mm_global,
                window_frames=window_fr, step_frames=step_fr,
                v_bins=80,
            )
            v_unit = 'mm/s' if px_per_mm_global else 'px/fr'
            fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
            fig3.patch.set_facecolor('#0a0a0a')
            for ax3 in axes3:
                ax3.set_facecolor('#1a1a2e')
            extent_v = [ve[0], ve[-1], tv[-1]*1e6, tv[0]*1e6]
            im3 = axes3[0].imshow(vm, aspect='auto', cmap='hot',
                                   origin='upper', extent=extent_v)
            plt.colorbar(im3, ax=axes3[0]).set_label('Norm. weight', color='white')
            axes3[0].set_xlabel(f'Velocity ({v_unit})', color='white', fontsize=9)
            axes3[0].set_ylabel('Time (us)', color='white', fontsize=9)
            axes3[0].tick_params(colors='white')
            vc = 0.5 * (ve[:-1] + ve[1:])
            mean_v = np.average(
                vc[np.newaxis, :].repeat(vm.shape[0], 0),
                weights=vm + 1e-12, axis=1)
            axes3[1].plot(tv * 1e6, mean_v, color='#FF6347', lw=1)
            axes3[1].set_xlabel('Time (us)', color='white', fontsize=9)
            axes3[1].set_ylabel(f'Mean velocity ({v_unit})', color='white', fontsize=9)
            axes3[1].tick_params(colors='white')
            for ax3 in axes3:
                for sp in ax3.spines.values():
                    sp.set_edgecolor('#444466')
            fig3.tight_layout()
            fig3.savefig(file_out / 'velocity_map.png', dpi=150, facecolor='#0a0a0a')
            plt.close(fig3)
            print(f"  Velocity map saved.")
        except Exception as e:
            print(f"  WARNING: Velocity map failed: {e}")

        # ── QA frame montage (lightweight) ──────────────────────────────────
        print("  Saving QA montage…")
        save_qa_montage(norm_frames, measurements,
                        file_out / 'qa_montage.png', FPS, px_per_mm_global)

        # ── Detailed QA frames ───────────────────────────────────────────────
        print(f"  Saving detailed QA frames (every {QA_INTERVAL} frames)…")
        saved = save_qa_frames(
            norm_frames, measurements, file_out, FPS,
            px_per_mm=px_per_mm_global,
            qa_interval=QA_INTERVAL,
            close_r=CLOSE_R, open_r=OPEN_R,
            dpi=120,
        )
        print(f"  {len(saved)} QA frames saved.")

        elapsed = time.time() - t0
        print(f"  File {file_num} done in {elapsed:.1f}s")

    # ── Combined / all-files outputs ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  COMBINED ANALYSIS (all 3 files)")
    print(f"{'='*60}")

    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all.to_csv(OUT_DIR / 'measurements_all.csv', index=False)
    print(f"  Combined CSV: {len(df_all):,} rows  →  measurements_all.csv")

    save_timeseries(df_all, OUT_DIR / 'timeseries_all.png',
                    px_per_mm_global, label="All 3 recordings")

    print(f"\nTotal wall time: {(time.time()-t_total_start)/60:.1f} min")
    print(f"All outputs in: {OUT_DIR}")
    print("DONE.")


if __name__ == '__main__':
    main()
