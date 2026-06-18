"""
fourier_analysis.py  —  Frequency-domain analysis of arc dynamics

Applies FFT to three time series extracted from measurements_all.csv:
  1. Arc detection rate  (binary 0/1 per frame → power spectrum)
  2. Arc tip depth       (tip_row in mm, NaNs interpolated)
  3. Arc column length   (arc_length_mm, NaNs interpolated)

Expected dominant frequency: ~50 Hz AC supply (or 100 Hz for full-wave effects)
and harmonics.

Outputs:
  fourier_detection.png   — power spectrum of arc on/off signal
  fourier_tip_depth.png   — power spectrum of tip depth time series
  fourier_arc_length.png  — power spectrum of arc length time series

Usage:
    python fourier_analysis.py [measurements_all.csv] [output_dir]
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

import arc_filter  # single source of truth for flash / validity criteria

# =============================================================================
# CONFIG
# =============================================================================

FPS       = 200_000    # frames per second
PX_PER_MM = 9.167      # BLADE default — overridden by metadata.txt when present

# Frequency range to display in plots (Hz)
FREQ_MIN  = 0
FREQ_MAX  = 500        # well above AC harmonics; increase if you see aliasing

# Number of top peaks to label on each plot
N_PEAKS   = 5

# =============================================================================
# LOAD + PREPARE
# =============================================================================

def read_metadata_px_per_mm(csv_path):
    """Read px_per_mm from a sibling metadata.txt (written by analyze_cine.py)."""
    meta_path = Path(csv_path).parent / 'metadata.txt'
    if not meta_path.exists():
        return None
    for line in meta_path.read_text().splitlines():
        if ':' in line:
            k, v = line.split(':', 1)
            if k.strip() == 'px_per_mm':
                try:
                    return float(v.strip())
                except (ValueError, TypeError):
                    return None
    return None


def load(csv_path):
    df = pd.read_csv(csv_path)
    df['arc_detected'] = pd.to_numeric(df['arc_detected'], errors='coerce').fillna(0)
    df['arc_tip_row']  = pd.to_numeric(df['arc_tip_row'],  errors='coerce')
    df['frame']        = pd.to_numeric(df['frame'],        errors='coerce')
    for col in ('arc_length_mm', 'arc_length_px'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ('local_vel_ms',):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # no velocity cap
    # Criterion 3: flashes are not valid arcs — treat them as gaps (like
    # non-detections) so every spectrum is built on the same valid frame set.
    flash = arc_filter.flash_mask(df)
    df.loc[flash, 'arc_detected'] = 0
    for col in ('arc_tip_row', 'arc_length_mm', 'arc_length_px', 'local_vel_ms'):
        if col in df.columns:
            df.loc[flash, col] = np.nan
    if 'file' in df.columns:
        print(f"Loaded {len(df):,} rows from {df['file'].nunique()} file(s)")
    else:
        print(f"Loaded {len(df):,} rows")
        df['file'] = 'unknown'
    return df


def make_timeseries(df, column, fill='interpolate'):
    """
    Build a continuous per-frame array for one column.
    Frames where arc was not detected (NaN) are filled by linear interpolation,
    which is standard practice before FFT on gapped physiological/physical signals.
    Set fill='zero' to zero-fill instead.
    """
    # Collapse duplicate frame numbers (e.g. from CSV merge) by averaging,
    # then reindex to a regular integer grid for FFT.
    series = df.groupby('frame')[column].mean().sort_index()
    # Fill to a regular integer index
    full_idx = pd.RangeIndex(int(series.index.min()), int(series.index.max()) + 1)
    series   = series.reindex(full_idx)

    if fill == 'interpolate':
        series = series.interpolate(method='linear', limit_direction='both')
    else:
        series = series.fillna(0)

    return series.values.astype(float)


def do_fft(signal, fps):
    """
    Compute one-sided power spectrum, normalised to the signal mean.

    Normalisation: divide the signal by its mean μ before the FFT so that
    the resulting spectrum represents squared fractional fluctuations (σ/μ)²
    per frequency bin.  This makes spectra comparable across runs that have
    different absolute signal levels (e.g. different detection rates or arc
    lengths) and is standard practice for dimensionless spectral analysis.

    The DC component is removed after normalisation (signal/μ − 1), so the
    zero-frequency bin carries no power and all plotted power represents
    genuine oscillatory content.

    Returns (freqs_hz, power) arrays, both clipped to FREQ_MAX.
    The power axis has units of (fractional amplitude)² — dimensionless.
    """
    n   = len(signal)
    mu  = signal.mean()
    if abs(mu) > 1e-10:
        signal = signal / mu      # express as multiples of mean
    signal = signal - signal.mean()   # remove DC (≈ 1 before this, ≈ 0 after)

    # Hann window to suppress spectral leakage from finite record
    window = np.hanning(n)
    yf     = np.fft.rfft(signal * window)
    power  = (np.abs(yf) ** 2) / n
    freqs  = np.fft.rfftfreq(n, d=1.0 / fps)

    mask   = (freqs >= FREQ_MIN) & (freqs <= FREQ_MAX)
    return freqs[mask], power[mask]

# =============================================================================
# PLOTTING
# =============================================================================

def label_peaks(ax, freqs, power):
    """Label the top N peaks with their frequency."""
    peaks, _ = find_peaks(power, height=power.max() * 0.01,
                          distance=int(5 / (freqs[1] - freqs[0])) + 1)
    if not len(peaks):
        return
    top_n = np.argsort(power[peaks])[-N_PEAKS:][::-1]
    for idx in top_n:
        pk = peaks[idx]
        ax.annotate(f'{freqs[pk]:.1f} Hz',
                    xy=(freqs[pk], power[pk]),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=8,
                    arrowprops=dict(arrowstyle='->', lw=0.7, color='#555'))


def plot_spectrum(freqs, power, title, out_path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogy(freqs, power, color='#1565C0', linewidth=0.8)
    label_peaks(ax, freqs, power)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalised power')
    # ax.set_title(title)
    ax.set_xlim(FREQ_MIN, FREQ_MAX)
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_all_runs(freq_list, power_list, labels, title, out_path):
    colors = ['#1565C0', '#E65100', '#2E7D32']
    fig, ax = plt.subplots(figsize=(9, 4))

    for freqs, power, label, col in zip(freq_list, power_list, labels, colors):
        ax.semilogy(freqs, power, color=col, linewidth=0.8,
                    alpha=0.8, label=label)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalised power')
    # ax.set_title(title)
    ax.set_xlim(FREQ_MIN, FREQ_MAX)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")

def plot_bymode_runs(freq_A_list, power_A_list, freq_B_list, power_B_list,
                     labels, title, out_path):
    """Overlay Mode-A and Mode-B spectra for multiple runs on one axes."""
    colors_A = ['#1565C0', '#0277BD', '#01579B']
    colors_B = ['#C62828', '#AD1457', '#6A1B9A']
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, label in enumerate(labels):
        ca = colors_A[i % len(colors_A)]
        cb = colors_B[i % len(colors_B)]
        if freq_A_list[i] is not None:
            ax.semilogy(freq_A_list[i], power_A_list[i], color=ca, linewidth=0.8,
                        alpha=0.8, label=f'{label} Mode A')
        if freq_B_list[i] is not None:
            ax.semilogy(freq_B_list[i], power_B_list[i], color=cb, linewidth=0.8,
                        alpha=0.8, linestyle='--', label=f'{label} Mode B')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalised power')
    ax.set_xlim(FREQ_MIN, FREQ_MAX)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def print_peak_table(freqs, power, label, n=5):
    peaks, _ = find_peaks(power, height=power.max() * 0.01,
                          distance=int(5 / (freqs[1] - freqs[0])) + 1)
    if not len(peaks):
        return
    top = np.argsort(power[peaks])[-n:][::-1]
    print(f"  {label}:")
    for rank, idx in enumerate(top, 1):
        pk  = peaks[idx]
        pct = 100 * power[pk] / power.sum()
        print(f"    #{rank}  {freqs[pk]:7.1f} Hz   power share {pct:.2f}%")

# =============================================================================
# MAIN
# =============================================================================

def main():
    global PX_PER_MM, FPS, FREQ_MIN, FREQ_MAX
    import argparse
    ap = argparse.ArgumentParser(description='Fourier analysis of arc dynamics')
    ap.add_argument('csv',         nargs='?', default=None,
                    help='Path to measurements_all.csv (or measurements.csv)')
    ap.add_argument('out_dir',     nargs='?', default=None,
                    help='Output directory (default: same as CSV)')
    ap.add_argument('--px-per-mm', type=float, default=None,
                    help='Spatial calibration px/mm (overrides metadata.txt and '
                         'module default %.4f)' % PX_PER_MM)
    ap.add_argument('--fps',       type=float, default=None,
                    help='Frame rate in Hz (default: %d)' % FPS)
    ap.add_argument('--freq-min',  type=float, default=None,
                    help='Low edge of the displayed/analysed spectrum in Hz (default %g).'
                         % FREQ_MIN)
    ap.add_argument('--freq-max',  type=float, default=None,
                    help='High edge of the displayed spectrum in Hz (default %g, tuned for '
                         '50 Hz mains harmonics). The J gliding-arc CYCLE is kHz-scale, so '
                         'raise this (e.g. 20000) to see the cycle peak — run_j_co2.py does.'
                         % FREQ_MAX)
    ap.add_argument('--mode-csv',  default=None,
                    help='Optional CSV with cycle_mode column for mode-stratified FFT '
                         '(e.g. measurements_all_events.csv).  If the main CSV already '
                         'has a cycle_mode column this flag is not needed.')
    args = ap.parse_args()

    if args.fps is not None:
        FPS = args.fps
    if args.freq_min is not None:
        FREQ_MIN = args.freq_min
    if args.freq_max is not None:
        FREQ_MAX = args.freq_max

    script_dir = Path(__file__).parent.resolve()
    csv_path   = Path(args.csv) if args.csv else script_dir / 'measurements_all.csv'
    out_dir    = Path(args.out_dir) if args.out_dir else csv_path.parent

    if not csv_path.exists():
        print(f"[ERROR] Not found: {csv_path}"); sys.exit(1)
    out_dir.mkdir(exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Fourier analysis")
    print(f"  CSV    : {csv_path}")
    print(f"  Output : {out_dir}")
    print(f"{'='*55}\n")

    # px_per_mm: CLI flag > metadata.txt sibling > module default
    if args.px_per_mm is not None:
        px_per_mm = args.px_per_mm
    else:
        px_per_mm = read_metadata_px_per_mm(csv_path) or PX_PER_MM

    df     = load(csv_path)

    # Merge mode labels from --mode-csv if provided, or use cycle_mode from main CSV
    if args.mode_csv:
        mode_df = pd.read_csv(args.mode_csv)
        mode_col = 'cycle_mode' if 'cycle_mode' in mode_df.columns else (
                   'mode' if 'mode' in mode_df.columns else None)
        if mode_col and 'file' in mode_df.columns and 'frame' in mode_df.columns:
            mode_df = mode_df[['file', 'frame', mode_col]].rename(
                columns={mode_col: '_cycle_mode'})
            mode_df['frame'] = pd.to_numeric(mode_df['frame'], errors='coerce')
            df['frame']      = pd.to_numeric(df['frame'], errors='coerce')
            df = df.merge(mode_df, on=['file', 'frame'], how='left')
            df['cycle_mode'] = df['_cycle_mode']
            df.drop(columns=['_cycle_mode'], inplace=True)
            print(f"Mode labels merged from {args.mode_csv}")
        else:
            print("[WARN] --mode-csv provided but missing required columns "
                  "(file, frame, cycle_mode/mode) — mode-stratified FFT skipped.")

    has_mode = 'cycle_mode' in df.columns and df['cycle_mode'].notna().any()

    files  = df['file'].unique()
    labels = [f'Run {i+1}' for i in range(len(files))]

    # ── Per-signal accumulators across runs ───────────────────────────────────
    det_spectra    = []   # (freqs, power) per run — detection rate
    depth_spectra  = []   # tip depth
    length_spectra = []   # arc length
    # Mode-stratified accumulators (None when mode data unavailable)
    depth_spectra_A  = []
    depth_spectra_B  = []
    length_spectra_A = []
    length_spectra_B = []

    print(f"{'─'*55}")
    print(f"  Top {N_PEAKS} spectral peaks per signal per run")
    print(f"  px_per_mm = {px_per_mm}")
    print(f"{'─'*55}")

    for fname, label in zip(files, labels):
        fdf = df[df['file'] == fname].copy()
        print(f"\n{label}  ({len(fdf):,} frames)")

        # 1. Detection rate: raw binary signal, zero-fill gaps
        det_sig      = make_timeseries(fdf, 'arc_detected', fill='zero')
        f_det, p_det = do_fft(det_sig, FPS)
        det_spectra.append((f_det, p_det))
        print_peak_table(f_det, p_det, 'detection')

        # 2. Tip depth: use mm if calibrated, fall back to raw pixel row
        if px_per_mm is not None:
            fdf['_depth'] = fdf['arc_tip_row'] / px_per_mm
            depth_label   = 'tip depth (mm)'
        else:
            fdf['_depth'] = fdf['arc_tip_row']
            depth_label   = 'tip depth (px)'
        depth_sig      = make_timeseries(fdf, '_depth', fill='interpolate')
        f_dep, p_dep   = do_fft(depth_sig, FPS)
        depth_spectra.append((f_dep, p_dep))
        print_peak_table(f_dep, p_dep, depth_label)

        # 3. Arc length: prefer mm if valid (non-NaN), fall back to px
        use_mm = (
            'arc_length_mm' in fdf.columns
            and fdf['arc_length_mm'].notna().sum() > 10
        )
        if use_mm:
            len_col   = 'arc_length_mm'
            len_label = 'arc length (mm)'
        elif 'arc_length_px' in fdf.columns:
            len_col   = 'arc_length_px'
            len_label = 'arc length (px)'
        else:
            len_col   = None
            len_label = 'arc length'

        if len_col is not None:
            len_sig        = make_timeseries(fdf, len_col, fill='interpolate')
            f_len, p_len   = do_fft(len_sig, FPS)
            length_spectra.append((f_len, p_len))
            print_peak_table(f_len, p_len, len_label)
        else:
            print("  [SKIP] No arc-length column available.")
            length_spectra.append(None)

        # Individual per-run plots
        plot_spectrum(f_det, p_det,
                      f'Detection-rate spectrum -- {label}',
                      out_dir / f'fourier_detection_{label.replace(" ","")}.png')
        plot_spectrum(f_dep, p_dep,
                      f'Tip-depth spectrum -- {label}',
                      out_dir / f'fourier_tipdepth_{label.replace(" ","")}.png')
        if length_spectra[-1] is not None:
            f_len, p_len = length_spectra[-1]
            plot_spectrum(f_len, p_len,
                          f'Arc-length spectrum -- {label}',
                          out_dir / f'fourier_arclength_{label.replace(" ","")}.png')

        # ── Mode-stratified FFT for this run ─────────────────────────────────
        if has_mode:
            run_tag = label.replace(' ', '')
            for mode_val, depth_acc, length_acc in [
                    ('A', depth_spectra_A, length_spectra_A),
                    ('B', depth_spectra_B, length_spectra_B)]:
                mdf = fdf[fdf['cycle_mode'] == mode_val].copy()
                if len(mdf) < 20:
                    depth_acc.append(None)
                    length_acc.append(None)
                    print(f"  [Mode {mode_val}] too few frames ({len(mdf)}) — skipping")
                    continue

                # Tip depth by mode
                if px_per_mm is not None:
                    mdf = mdf.copy()
                    mdf['_depth'] = mdf['arc_tip_row'] / px_per_mm
                depth_sig_m   = make_timeseries(mdf, '_depth', fill='interpolate')
                f_dep_m, p_dep_m = do_fft(depth_sig_m, FPS)
                depth_acc.append((f_dep_m, p_dep_m))
                plot_spectrum(f_dep_m, p_dep_m,
                              f'Tip-depth Mode {mode_val} -- {label}',
                              out_dir / f'fourier_tipdepth_mode{mode_val}_{run_tag}.png')
                print_peak_table(f_dep_m, p_dep_m, f'tip depth Mode {mode_val}')

                # Arc length by mode
                if len_col is not None:
                    len_sig_m     = make_timeseries(mdf, len_col, fill='interpolate')
                    f_len_m, p_len_m = do_fft(len_sig_m, FPS)
                    length_acc.append((f_len_m, p_len_m))
                    plot_spectrum(f_len_m, p_len_m,
                                  f'Arc-length Mode {mode_val} -- {label}',
                                  out_dir / f'fourier_arclength_mode{mode_val}_{run_tag}.png')
                    print_peak_table(f_len_m, p_len_m, f'arc length Mode {mode_val}')
                else:
                    length_acc.append(None)

            # Combined overlay A vs B for this run — tip depth
            run_A_dep = depth_spectra_A[-1]
            run_B_dep = depth_spectra_B[-1]
            if run_A_dep is not None or run_B_dep is not None:
                fig, ax = plt.subplots(figsize=(9, 4))
                if run_A_dep is not None:
                    ax.semilogy(run_A_dep[0], run_A_dep[1], color='#1565C0',
                                linewidth=0.8, label='Mode A')
                if run_B_dep is not None:
                    ax.semilogy(run_B_dep[0], run_B_dep[1], color='#C62828',
                                linewidth=0.8, linestyle='--', label='Mode B')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Normalised power')
                ax.set_xlim(FREQ_MIN, FREQ_MAX)
                ax.legend(fontsize=9, frameon=False)
                ax.grid(alpha=0.2)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig.tight_layout()
                p_out = out_dir / f'fourier_tipdepth_bymode_{run_tag}.png'
                fig.savefig(p_out, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved: {p_out}")

            # Combined overlay A vs B for this run — arc length
            run_A_len = length_spectra_A[-1]
            run_B_len = length_spectra_B[-1]
            if run_A_len is not None or run_B_len is not None:
                fig, ax = plt.subplots(figsize=(9, 4))
                if run_A_len is not None:
                    ax.semilogy(run_A_len[0], run_A_len[1], color='#1565C0',
                                linewidth=0.8, label='Mode A')
                if run_B_len is not None:
                    ax.semilogy(run_B_len[0], run_B_len[1], color='#C62828',
                                linewidth=0.8, linestyle='--', label='Mode B')
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Normalised power')
                ax.set_xlim(FREQ_MIN, FREQ_MAX)
                ax.legend(fontsize=9, frameon=False)
                ax.grid(alpha=0.2)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                fig.tight_layout()
                p_out = out_dir / f'fourier_arclength_bymode_{run_tag}.png'
                fig.savefig(p_out, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved: {p_out}")
        else:
            depth_spectra_A.append(None)
            depth_spectra_B.append(None)
            length_spectra_A.append(None)
            length_spectra_B.append(None)

    # Combined overlay plots (filter out None entries from skipped arc-length)
    valid_len = [(s, l) for s, l in zip(length_spectra, labels) if s is not None]

    plot_all_runs(
        [s[0] for s in det_spectra],
        [s[1] for s in det_spectra],
        labels,
        'Detection-rate spectrum -- all runs',
        out_dir / 'fourier_detection_allruns.png')

    plot_all_runs(
        [s[0] for s in depth_spectra],
        [s[1] for s in depth_spectra],
        labels,
        'Tip-depth spectrum -- all runs',
        out_dir / 'fourier_tipdepth_allruns.png')

    if valid_len:
        plot_all_runs(
            [s[0] for s, _ in valid_len],
            [s[1] for s, _ in valid_len],
            [l for _, l in valid_len],
            'Arc-length spectrum -- all runs',
            out_dir / 'fourier_arclength_allruns.png')
    else:
        print("  [SKIP] Arc-length spectrum -- no valid data.")

    # Mode-stratified all-runs combined overlays
    if has_mode:
        fa_dep  = [s[0] if s else None for s in depth_spectra_A]
        pa_dep  = [s[1] if s else None for s in depth_spectra_A]
        fb_dep  = [s[0] if s else None for s in depth_spectra_B]
        pb_dep  = [s[1] if s else None for s in depth_spectra_B]
        if any(x is not None for x in fa_dep) or any(x is not None for x in fb_dep):
            plot_bymode_runs(fa_dep, pa_dep, fb_dep, pb_dep, labels,
                             'Tip-depth by mode — all runs',
                             out_dir / 'fourier_tipdepth_bymode_allruns.png')

        fa_len  = [s[0] if s else None for s in length_spectra_A]
        pa_len  = [s[1] if s else None for s in length_spectra_A]
        fb_len  = [s[0] if s else None for s in length_spectra_B]
        pb_len  = [s[1] if s else None for s in length_spectra_B]
        if any(x is not None for x in fa_len) or any(x is not None for x in fb_len):
            plot_bymode_runs(fa_len, pa_len, fb_len, pb_len, labels,
                             'Arc-length by mode — all runs',
                             out_dir / 'fourier_arclength_bymode_allruns.png')

    print("\n" + "=" * 55)
    print("  Done.  Look for peaks at multiples of 50 Hz.")
    print("  If dominant peak is at 100 Hz, the discharge fires")
    print("  once per half-cycle of 50 Hz AC (both polarities).")
    print("=" * 55 + "\n")


if __name__ == '__main__':
    main()
