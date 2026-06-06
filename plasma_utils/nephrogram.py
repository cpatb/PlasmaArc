"""
plasma_utils/nephrogram.py
--------------------------
Nephrogram (space–time kymograph) and velocity-distribution map for
plasma arc recordings.

What is a nephrogram here?
~~~~~~~~~~~~~~~~~~~~~~~~~~
A *nephrogram* is a 2D space–time image:
  - **Rows** → time (frame index, or physical time if fps is known)
  - **Columns** → spatial position along the arc-propagation axis

Each row is a 1-D intensity projection of one frame onto the chosen
spatial axis.  Stacking these rows produces an image in which the
slope of bright features directly encodes the arc propagation velocity:

    v  =  Δ(spatial axis)  /  Δt   [mm/s or px/frame]

Two projection modes are supported:

1. ``'vertical'`` (default) – column-wise maximum / mean projected onto
   the *row* (y) axis.  Captures the arc reaching downward between the
   electrodes (CO₂ flow pushes the arc down; cycles visible as the arc
   re-strokes at the electrode tips and then extends).

2. ``'horizontal'`` – row-wise projection onto the *column* (x) axis.
   Useful when the arc drifts laterally.

Velocity extraction
~~~~~~~~~~~~~~~~~~~
Bright streaks in the nephrogram correspond to arc features propagating
through space.  We extract velocities with a Radon / Hough-line approach:

1. Compute the Radon transform of the nephrogram (or a Gaussian-blurred
   version to suppress noise).
2. Find the angle θ that maximises the Radon sinogram energy.
3. Convert θ → slope → velocity.

All functions return plain NumPy arrays so they can be saved, displayed,
or fed into downstream analysis without any hidden state.
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Core projection  (single frame → 1-D profile)
# ---------------------------------------------------------------------------

def frame_projection(
    frame: np.ndarray,
    axis: str = 'vertical',
    mode: str = 'max',
) -> np.ndarray:
    """Reduce a 2-D frame to a 1-D spatial profile.

    Parameters
    ----------
    frame : 2-D float32 array (H × W), values in [0, 1]
    axis  : ``'vertical'``  → project onto row-axis (length = H)
            ``'horizontal'``→ project onto col-axis (length = W)
    mode  : ``'max'``  – column/row maximum
            ``'mean'`` – column/row mean
            ``'sum'``  – column/row sum (use for integrated power)

    Returns
    -------
    profile : 1-D float32 array
    """
    if axis == 'vertical':
        reduce_ax = 1   # reduce columns → result indexed by row
    elif axis == 'horizontal':
        reduce_ax = 0   # reduce rows → result indexed by col
    else:
        raise ValueError(f"axis must be 'vertical' or 'horizontal', got {axis!r}")

    if mode == 'max':
        return frame.max(axis=reduce_ax).astype(np.float32)
    elif mode == 'mean':
        return frame.mean(axis=reduce_ax).astype(np.float32)
    elif mode == 'sum':
        return frame.sum(axis=reduce_ax).astype(np.float32)
    else:
        raise ValueError(f"mode must be 'max', 'mean', or 'sum', got {mode!r}")


# ---------------------------------------------------------------------------
# Build the full nephrogram from a frame stack
# ---------------------------------------------------------------------------

def build_nephrogram(
    frames: np.ndarray,
    axis: str = 'vertical',
    mode: str = 'max',
    normalise: bool = True,
    smooth_sigma: float = 0.0,
) -> np.ndarray:
    """Build a 2-D nephrogram from a (N, H, W) frame stack.

    Parameters
    ----------
    frames       : (N, H, W) float32 array, values in [0, 1]
    axis         : spatial projection axis (see :func:`frame_projection`)
    mode         : aggregation mode (see :func:`frame_projection`)
    normalise    : if True, stretch the final image to [0, 1]
    smooth_sigma : Gaussian blur σ applied *after* building (spatial–temporal
                   smoothing to suppress jitter in the velocity estimate)

    Returns
    -------
    nephrogram : 2-D float32 array, shape (N, spatial_length)
        Rows = frames (time); columns = spatial position.
    """
    profiles = np.stack(
        [frame_projection(f, axis=axis, mode=mode) for f in frames],
        axis=0,
    )   # (N, spatial_length)

    if smooth_sigma > 0:
        profiles = gaussian_filter(profiles.astype(np.float64), sigma=smooth_sigma).astype(np.float32)

    if normalise:
        lo, hi = profiles.min(), profiles.max()
        if hi > lo:
            profiles = (profiles - lo) / (hi - lo)

    return profiles.astype(np.float32)


# ---------------------------------------------------------------------------
# Velocity extraction via Radon sinogram
# ---------------------------------------------------------------------------

def radon_velocity(
    nephrogram: np.ndarray,
    fps: float,
    px_per_mm: Optional[float] = None,
    angle_range_deg: Tuple[float, float] = (-80, 80),
    n_angles: int = 360,
    smooth_sigma: float = 1.5,
    top_n: int = 5,
) -> dict:
    """Estimate propagation velocities from the nephrogram via Radon transform.

    The nephrogram is treated as an image in (time, space) coordinates.
    A bright streak in the (space × time) image makes angle α from the
    horizontal time axis.  skimage's radon captures this streak at
    theta = (90° - α), so the dominant peak angle theta satisfies:

        v  =  Δspace / Δtime  =  cot(theta)  ×  fps / px_per_mm   [mm/s]

    Note: the slope is cot(theta) = cos(theta)/sin(theta), NOT tan(theta).
    Small theta → fast feature; large theta (→90°) → stationary feature.

    Parameters
    ----------
    nephrogram      : (N_frames, N_spatial) float32 nephrogram
    fps             : recording frame rate (Hz)
    px_per_mm       : spatial calibration.  If None velocities in px/frame.
    angle_range_deg : search range for propagation angles
    n_angles        : number of projection angles to test
    smooth_sigma    : Gaussian smoothing of sinogram before peak search
    top_n           : return this many peak velocities (sorted by strength)

    Returns
    -------
    dict with keys:
        'dominant_velocity'        : float  (mm/s or px/frame)
        'dominant_angle_deg'       : float
        'velocity_candidates'      : list of (velocity, angle_deg, strength)
        'sinogram'                 : 2-D array (n_angles, n_radon_spatial)
        'angles_deg'               : 1-D array of tested angles
    """
    from skimage.transform import radon

    # Radon expects (H, W) with H = spatial, W = time for our orientation
    # Transpose so spatial is vertical (rows) and time is horizontal (cols)
    img = nephrogram.T.astype(np.float32)   # (N_spatial, N_frames)

    angles = np.linspace(angle_range_deg[0], angle_range_deg[1], n_angles)
    sinogram = radon(img, theta=angles, circle=False)

    if smooth_sigma > 0:
        sinogram_s = gaussian_filter(sinogram.astype(np.float64), sigma=smooth_sigma).astype(np.float32)
    else:
        sinogram_s = sinogram

    # Find top-N angle indices by column energy
    col_energy = (sinogram_s ** 2).sum(axis=0)   # (n_angles,)
    top_idx = np.argsort(col_energy)[::-1][:top_n]

    # Convert angle → velocity
    #
    # Convention check (verified numerically with synthetic streaks):
    #
    #   img is transposed so that rows = space, cols = time.
    #   skimage's radon at theta measures angle of the integration line from
    #   the *vertical* (space) axis, i.e. theta = 0 captures vertical lines
    #   (instantaneous events) and theta = 90 captures horizontal lines
    #   (stationary features).
    #
    #   A feature moving at v px/frame traces a diagonal whose angle from
    #   the time axis is arctan(v), equivalently (90° - arctan(v)) from the
    #   space/vertical axis.  Therefore the Radon peak appears at:
    #
    #       theta_peak = 90° - arctan(v)   →   v = cot(theta_peak)
    #
    #   The correct slope is cot(theta) = cos(theta)/sin(theta), NOT tan(theta).
    #   Using tan gives the reciprocal (confirmed: fast streak → small theta →
    #   tan≈0 but cot≈v_true).
    #
    dt = 1.0 / fps if fps > 0 else 1.0
    dx_per_px = (1.0 / px_per_mm) if px_per_mm else 1.0   # mm/px or 1

    candidates = []
    for idx in top_idx:
        theta = float(angles[idx])
        theta_rad = np.deg2rad(theta)
        sin_t = np.sin(theta_rad)
        # Guard against theta ≈ 0 (instantaneous / infinite-velocity event)
        if abs(sin_t) < 1e-6:
            slope_px_per_frame = np.inf * np.sign(np.cos(theta_rad))
        else:
            slope_px_per_frame = np.cos(theta_rad) / sin_t   # cot(theta)  [px / frame]
        velocity = slope_px_per_frame * dx_per_px / dt        # mm/s or px/frame
        candidates.append((velocity, theta, float(col_energy[idx])))

    dominant_velocity, dominant_angle, _ = candidates[0]

    return {
        'dominant_velocity': dominant_velocity,
        'dominant_angle_deg': dominant_angle,
        'velocity_candidates': candidates,
        'sinogram': sinogram,
        'angles_deg': angles,
    }


# ---------------------------------------------------------------------------
# Velocity-distribution map (spatial histogram of instantaneous velocities)
# ---------------------------------------------------------------------------

def velocity_distribution_map(
    nephrogram: np.ndarray,
    fps: float,
    px_per_mm: Optional[float] = None,
    window_frames: int = 50,
    step_frames: int = 10,
    v_bins: int = 100,
    v_range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a time-resolved velocity distribution map.

    Slides a temporal window over the nephrogram and estimates the
    dominant propagation velocity in each window using the local
    Radon transform.

    Parameters
    ----------
    nephrogram    : (N_frames, N_spatial) nephrogram
    fps, px_per_mm: as above
    window_frames : temporal window length
    step_frames   : hop size between windows
    v_bins        : number of velocity histogram bins
    v_range       : (v_min, v_max) in mm/s.  Auto-set if None.

    Returns
    -------
    times    : 1-D float array of window centre times (s)
    v_edges  : 1-D float array of velocity bin edges (mm/s)
    v_map    : 2-D float array (n_windows, v_bins) — normalised histogram
               of candidate velocities per window  → the "velocity
               distribution nephrogram"
    """
    from skimage.transform import radon

    N = nephrogram.shape[0]
    centres = list(range(window_frames // 2, N - window_frames // 2, step_frames))
    if not centres:
        centres = [N // 2]

    # establish v_range on the full nephrogram first
    if v_range is None:
        full = radon_velocity(nephrogram, fps, px_per_mm, top_n=20)
        all_v = [abs(v) for v, _, _ in full['velocity_candidates']]
        vmax = max(all_v) * 1.2 if all_v else 1e6
        v_range = (-vmax, vmax)

    v_edges = np.linspace(v_range[0], v_range[1], v_bins + 1)
    v_map = np.zeros((len(centres), v_bins), dtype=np.float32)
    times = np.array([c / fps for c in centres], dtype=np.float64)

    for i, centre in enumerate(centres):
        start = max(0, centre - window_frames // 2)
        end = min(N, start + window_frames)
        window_ng = nephrogram[start:end]
        if window_ng.shape[0] < 10:
            continue
        res = radon_velocity(window_ng, fps, px_per_mm, top_n=30)
        for v, _, strength in res['velocity_candidates']:
            bin_idx = np.searchsorted(v_edges, v) - 1
            if 0 <= bin_idx < v_bins:
                v_map[i, bin_idx] += strength

        # normalise row
        row_sum = v_map[i].sum()
        if row_sum > 0:
            v_map[i] /= row_sum

    return times, v_edges, v_map


# ---------------------------------------------------------------------------
# Cycle detection in the nephrogram
# ---------------------------------------------------------------------------

def detect_cycles(
    nephrogram: np.ndarray,
    fps: float,
    spatial_range: Optional[Tuple[int, int]] = None,
    min_cycle_frames: int = 10,
) -> dict:
    """Detect arc propagation cycles from a 1-D temporal signal derived
    from the nephrogram.

    Strategy: integrate the nephrogram intensity over a spatial window
    (near the arc midpoint, where propagation is most visible) to get a
    1-D temporal signal.  Find peaks in this signal to identify
    individual propagation cycles.

    Parameters
    ----------
    nephrogram      : (N, S) nephrogram
    fps             : recording fps
    spatial_range   : (start, end) column indices of the spatial window.
                      Defaults to the central 20% of spatial extent.
    min_cycle_frames: minimum spacing between detected cycle peaks

    Returns
    -------
    dict with keys:
        'cycle_frames'     : array of frame indices at which cycles peak
        'cycle_times_s'    : corresponding times in seconds
        'mean_period_s'    : mean cycle period (s)
        'mean_frequency_Hz': mean cycle frequency (Hz)
        'temporal_signal'  : 1-D float array used for detection
    """
    from scipy.signal import find_peaks

    N, S = nephrogram.shape
    if spatial_range is None:
        lo = int(S * 0.4)
        hi = int(S * 0.6)
    else:
        lo, hi = spatial_range
    lo, hi = max(0, lo), min(S, hi)
    if hi <= lo:
        hi = S

    signal = nephrogram[:, lo:hi].mean(axis=1).astype(np.float64)

    peaks, props = find_peaks(
        signal,
        distance=min_cycle_frames,
        prominence=signal.std() * 0.5,
    )

    mean_period = float(np.diff(peaks).mean() / fps) if len(peaks) > 1 else float('nan')
    mean_freq = 1.0 / mean_period if mean_period > 0 else float('nan')

    # FFT-based frequency spectrum (more robust than peak counting for
    # noisy or quasi-periodic signals, and reveals harmonics)
    fft_result = fft_frequency_spectrum(signal, fps)

    return {
        'cycle_frames': peaks,
        'cycle_times_s': peaks / fps,
        'mean_period_s': mean_period,
        'mean_frequency_Hz': mean_freq,
        'temporal_signal': signal,
        'fft': fft_result,                  # see fft_frequency_spectrum()
    }


# ---------------------------------------------------------------------------
# FFT-based frequency analysis
# ---------------------------------------------------------------------------

def fft_frequency_spectrum(
    signal: np.ndarray,
    fps: float,
    detrend: bool = True,
    window: str = 'hann',
) -> dict:
    """Compute a one-sided power spectrum of a 1-D temporal signal via FFT.

    This is the recommended way to characterise the cyclical behaviour of
    arc propagation signals (arc length, centroid position, nephrogram
    temporal average, etc.).  Advantages over ``find_peaks``:

    * Returns the **full frequency spectrum** including harmonics.
    * Robust to amplitude variations across cycles.
    * Directly gives the dominant frequency even for quasi-periodic signals.

    Parameters
    ----------
    signal : 1-D float array
        Time-domain signal sampled at *fps* frames per second.
    fps    : float
        Sampling rate in Hz (frames per second).
    detrend : bool
        If True, remove the linear trend before computing the FFT (avoids
        spectral leakage from slow drift).  Recommended for centroid_row
        and arc_length signals that drift over the recording.
    window : str
        Windowing function applied before FFT.  ``'hann'`` (default) gives
        good frequency resolution and low sidelobe leakage.
        Also accepts ``'hamming'``, ``'blackman'``, ``'none'``.

    Returns
    -------
    dict with keys:
        'freqs_Hz'        : 1-D array, positive frequencies in Hz
        'power'           : 1-D array, one-sided power spectral density
                            (arbitrary units; useful for relative comparison)
        'dominant_freq_Hz': float, frequency of the highest power peak
        'dominant_period_s': float, 1 / dominant_freq_Hz
        'harmonics'       : list of (freq_Hz, power) for the top-5 peaks
    """
    from scipy.signal import find_peaks as _find_peaks

    sig = np.asarray(signal, dtype=np.float64).copy()
    N = len(sig)
    if N < 4:
        return {
            'freqs_Hz': np.array([]),
            'power': np.array([]),
            'dominant_freq_Hz': float('nan'),
            'dominant_period_s': float('nan'),
            'harmonics': [],
        }

    # Detrend: remove linear ramp (handles DC offset + slow drift)
    if detrend:
        sig -= np.polyval(np.polyfit(np.arange(N), sig, 1), np.arange(N))

    # Apply window to reduce spectral leakage
    _windows = {
        'hann':     np.hanning(N),
        'hamming':  np.hamming(N),
        'blackman': np.blackman(N),
        'none':     np.ones(N),
    }
    win = _windows.get(window, np.hanning(N))
    sig_w = sig * win

    # One-sided FFT power spectrum
    fft_vals = np.fft.rfft(sig_w)
    freqs    = np.fft.rfftfreq(N, d=1.0 / fps)   # Hz
    power    = (np.abs(fft_vals) ** 2) / (win.sum() ** 2)  # normalised PSD
    # Double the one-sided spectrum (except DC and Nyquist) to conserve energy
    if N % 2 == 0:
        power[1:-1] *= 2
    else:
        power[1:] *= 2

    # Find dominant frequency (skip DC at index 0)
    if len(power) > 1:
        peak_indices, _ = _find_peaks(power[1:])
        peak_indices += 1   # offset back to full-spectrum index
        if len(peak_indices) == 0:
            dominant_idx = int(np.argmax(power[1:])) + 1
        else:
            dominant_idx = peak_indices[int(np.argmax(power[peak_indices]))]
        dominant_freq = float(freqs[dominant_idx])
    else:
        dominant_freq = float('nan')

    dominant_period = (1.0 / dominant_freq) if dominant_freq > 0 else float('nan')

    # Top-5 spectral peaks (skip DC)
    if len(power) > 1:
        all_peaks, _ = _find_peaks(power[1:])
        all_peaks += 1
        if len(all_peaks) == 0:
            all_peaks = np.array([int(np.argmax(power[1:])) + 1])
        sorted_peaks = all_peaks[np.argsort(power[all_peaks])[::-1]]
        harmonics = [(float(freqs[p]), float(power[p])) for p in sorted_peaks[:5]]
    else:
        harmonics = []

    return {
        'freqs_Hz':          freqs,
        'power':             power,
        'dominant_freq_Hz':  dominant_freq,
        'dominant_period_s': dominant_period,
        'harmonics':         harmonics,
    }


def fft_from_csv_column(
    times_s: np.ndarray,
    values: np.ndarray,
    fps: float,
    detrend: bool = True,
    window: str = 'hann',
) -> dict:
    """Convenience wrapper: compute FFT spectrum from a CSV time-series column.

    Use this to analyse the cyclic behaviour of any per-frame metric
    (arc_length_mm, centroid_row, electrode_gap_mm, mean_intensity, etc.)
    read from measurements.csv.

    Parameters
    ----------
    times_s : 1-D array of frame timestamps (seconds)
    values  : 1-D array of the metric values (same length as times_s)
    fps     : recording frame rate in Hz
    detrend, window : passed to :func:`fft_frequency_spectrum`

    Returns
    -------
    dict from :func:`fft_frequency_spectrum`

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.read_csv('measurements.csv')
    >>> det = df[df['arc_detected'] == 1]
    >>> result = fft_from_csv_column(det['time_s'].values,
    ...                              det['arc_length_mm'].values, fps=200_000)
    >>> print(f"Dominant arc cycle: {result['dominant_freq_Hz']:.1f} Hz  "
    ...       f"({result['dominant_period_s']*1e3:.2f} ms)")
    """
    # Interpolate onto a uniform grid (handles NaNs and missing frames)
    mask = np.isfinite(values)
    if mask.sum() < 4:
        return fft_frequency_spectrum(np.zeros(4), fps)
    t_uniform = np.arange(times_s[mask][0], times_s[mask][-1], 1.0 / fps)
    v_uniform = np.interp(t_uniform, times_s[mask], values[mask])
    return fft_frequency_spectrum(v_uniform, fps, detrend=detrend, window=window)
