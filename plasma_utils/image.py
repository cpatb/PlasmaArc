"""
plasma_utils/image.py
---------------------
Per-frame image processing for plasma arc analysis.

Pipeline per frame
~~~~~~~~~~~~~~~~~~
1. **Normalise** – stretch 16-/12-bit raw to float32 [0, 1].
2. **Threshold** – Otsu on the bright pixels (arc is a small bright object on
   a dark background; we use a percentile pre-filter to avoid bias).
3. **Morphological cleanup** – closing fills small gaps in the arc channel;
   opening removes isolated speckle.
4. **Skeletonise** – medial-axis thinning gives a 1-pixel-wide arc centreline.
5. **Measurements**:
   - Arc length    : skeleton pixel count × pixel pitch
   - Arc width     : FWHM of perpendicular intensity profiles sampled at
                     evenly spaced skeleton points
   - Brightness    : mean, peak, and integrated intensity inside the arc mask
   - Tip positions : endpoints of the skeleton (degree-1 nodes)
   - Centroid      : intensity-weighted centroid of arc pixels
   - Bounding box  : tight axis-aligned rectangle around the arc mask
   - Curvature     : mean absolute angular change along the skeleton (rad/px)
   - Electrode gap : estimated as Euclidean distance between the two brightest
                     skeleton endpoints; converted to mm via px_per_mm.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import label as nd_label
from skimage.filters import threshold_otsu
from skimage.morphology import (
    closing, opening, skeletonize, disk
)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ArcMeasurement:
    # --- raw counts (pixels) ---
    area_px: int = 0
    arc_length_px: float = 0.0
    arc_width_px_mean: float = float('nan')
    arc_width_px_median: float = float('nan')

    # --- physical (mm) — set to nan if px_per_mm unknown ---
    arc_length_mm: float = float('nan')
    arc_width_mm_mean: float = float('nan')
    arc_width_mm_median: float = float('nan')

    # --- brightness ---
    mean_intensity: float = float('nan')
    peak_intensity: float = float('nan')
    integrated_intensity: float = float('nan')   # sum of arc-pixel values

    # --- geometry ---
    centroid_row: float = float('nan')
    centroid_col: float = float('nan')
    bbox_row_min: int = 0
    bbox_row_max: int = 0
    bbox_col_min: int = 0
    bbox_col_max: int = 0

    # --- arc tips (row, col) ---
    tip_a_row: float = float('nan')
    tip_a_col: float = float('nan')
    tip_b_row: float = float('nan')
    tip_b_col: float = float('nan')
    electrode_gap_px: float = float('nan')
    electrode_gap_mm: float = float('nan')

    # --- shape ---
    curvature_mean: float = float('nan')   # mean |Δθ| along skeleton (rad/px)
    skeleton_n_pixels: int = 0

    # --- flags ---
    arc_detected: bool = False


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Return frame as float32 in [0, 1].

    Works for 8-bit, 12-bit, 16-bit, or already-float inputs.
    """
    f = frame.astype(np.float32)
    lo, hi = f.min(), f.max()
    if hi <= lo:
        return np.zeros_like(f)
    return (f - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------

def binarize(
    gray: np.ndarray,
    close_r: int = 3,
    open_r: int = 2,
    otsu_percentile: float = 50.0,
) -> np.ndarray:
    """Binary arc mask via Otsu thresholding + morphological cleanup.

    The arc occupies a very small fraction of the frame (dark background).
    A raw Otsu over the full image is biased toward the background; we
    therefore run Otsu only on the upper *otsu_percentile* of pixel values,
    which concentrates the threshold around the arc-intensity peak.

    Parameters
    ----------
    gray : float32 [0,1] frame
    close_r : closing disk radius (px) – bridges small gaps in the arc
    open_r  : opening disk radius (px) – removes isolated speckle
    otsu_percentile : float
        Only pixels above this intensity percentile are fed to Otsu.
        Default 50 (median) works well for dark-background arc images.
    """
    bright_pixels = gray[gray > np.percentile(gray, otsu_percentile)]
    if bright_pixels.size < 10:
        return np.zeros(gray.shape, dtype=bool)

    thresh = threshold_otsu(bright_pixels)
    binary = gray > thresh

    if close_r > 0:
        binary = closing(binary, disk(close_r))
    if open_r > 0:
        binary = opening(binary, disk(open_r))

    # keep only the largest connected component (the arc channel)
    labeled, n = nd_label(binary)
    if n == 0:
        return binary
    sizes = np.bincount(labeled.ravel())[1:]
    biggest = np.argmax(sizes) + 1
    return labeled == biggest


# ---------------------------------------------------------------------------
# Width measurement helper
# ---------------------------------------------------------------------------

def _fwhm_perpendicular(
    gray: np.ndarray,
    row: float,
    col: float,
    tangent: Tuple[float, float],
    half_width: int = 20,
) -> float:
    """Full-width at half-maximum of the intensity profile perpendicular to
    the arc skeleton at point (row, col).

    Parameters
    ----------
    gray      : normalised float32 frame
    row, col  : skeleton point coordinates
    tangent   : (drow, dcol) unit tangent to the arc at this point
    half_width: number of pixels to sample either side of the skeleton

    Returns
    -------
    fwhm_px : float  (nan if the peak is too dim or profile is ill-defined)
    """
    H, W = gray.shape
    # perpendicular direction (rotate tangent 90°)
    perp = np.array([-tangent[1], tangent[0]], dtype=float)
    norm = np.linalg.norm(perp)
    if norm < 1e-9:
        return float('nan')
    perp /= norm

    offsets = np.arange(-half_width, half_width + 1, dtype=float)
    sample_rows = row + offsets * perp[0]
    sample_cols = col + offsets * perp[1]

    # bounds check
    valid = (
        (sample_rows >= 0) & (sample_rows < H - 1) &
        (sample_cols >= 0) & (sample_cols < W - 1)
    )
    if valid.sum() < 5:
        return float('nan')

    # bilinear interpolation
    r0 = sample_rows[valid].astype(int)
    c0 = sample_cols[valid].astype(int)
    dr = sample_rows[valid] - r0
    dc = sample_cols[valid] - c0
    r1 = np.clip(r0 + 1, 0, H - 1)
    c1 = np.clip(c0 + 1, 0, W - 1)

    profile = (
        gray[r0, c0] * (1 - dr) * (1 - dc) +
        gray[r0, c1] * (1 - dr) * dc +
        gray[r1, c0] * dr * (1 - dc) +
        gray[r1, c1] * dr * dc
    )

    peak = profile.max()
    if peak < 0.05:          # arc too dim — probably noise
        return float('nan')

    half_max = peak / 2.0
    above = profile >= half_max
    if not above.any():
        return float('nan')

    # FWHM = distance between first and last crossing of half-max
    idx = np.where(above)[0]
    fwhm = float(idx[-1] - idx[0]) + 1.0
    return fwhm


# ---------------------------------------------------------------------------
# Skeleton → ordered path  (fast, numpy-based)
# ---------------------------------------------------------------------------

def _skeleton_ordered_coords(skeleton: np.ndarray) -> np.ndarray | None:
    """Return skeleton pixel coordinates as an (N, 2) array ordered along the
    arc from one endpoint to the other.

    Fast implementation: uses a 2-D boolean lookup array for O(1) neighbour
    checks, walks greedily from one endpoint to the other.

    Returns None if the skeleton is empty.
    """
    coords = np.argwhere(skeleton)   # (N, 2)
    N = len(coords)
    if N == 0:
        return None
    if N == 1:
        return coords

    H, W = skeleton.shape

    # Pre-compute degree of each skeleton pixel using the boolean image
    # (8-connected convolution trick with scipy — fast for 256×256)
    from scipy.ndimage import convolve
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    deg_img = convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)

    # Endpoints: degree == 1 in skeleton
    ep_mask = (skeleton > 0) & (deg_img == 1)
    eps = np.argwhere(ep_mask)
    if len(eps) >= 1:
        start_r, start_c = int(eps[0, 0]), int(eps[0, 1])
    else:
        start_r, start_c = int(coords[0, 0]), int(coords[0, 1])

    # Greedy walk using the boolean skeleton image as neighbour lookup
    path_r = np.empty(N, dtype=np.int32)
    path_c = np.empty(N, dtype=np.int32)
    visited = skeleton.copy().astype(bool)   # copy — we'll mark as we visit

    r, c = start_r, start_c
    visited[r, c] = False   # mark visited by clearing from the copy
    path_r[0] = r
    path_c[0] = c

    for i in range(1, N):
        # find unvisited 8-connected neighbours
        r0, r1 = max(r-1, 0), min(r+2, H)
        c0, c1 = max(c-1, 0), min(c+2, W)
        patch = visited[r0:r1, c0:c1]
        nbs = np.argwhere(patch)
        if len(nbs) == 0:
            # disconnection — use nearest remaining point
            rem = np.argwhere(visited)
            if len(rem) == 0:
                path_r = path_r[:i]; path_c = path_c[:i]
                break
            dists = np.abs(rem[:,0]-r) + np.abs(rem[:,1]-c)
            best = np.argmin(dists)
            r, c = int(rem[best,0]), int(rem[best,1])
        else:
            nr, nc = int(nbs[0,0]+r0), int(nbs[0,1]+c0)
            r, c = nr, nc
        visited[r, c] = False
        path_r[i] = r
        path_c[i] = c

    return np.stack([path_r, path_c], axis=1)   # (N, 2)


# ---------------------------------------------------------------------------
# Main measurement function
# ---------------------------------------------------------------------------

def measure_frame(
    gray: np.ndarray,
    binary: np.ndarray,
    px_per_mm: Optional[float] = None,
    width_sample_step: int = 5,
    close_r: int = 3,
    open_r: int = 2,
    fast: bool = False,
) -> ArcMeasurement:
    """Compute all arc measurements for a single frame.

    Parameters
    ----------
    gray : float32 [0,1] normalised frame
    binary : bool mask of the arc
    px_per_mm : physical calibration (pixels per millimetre).
                If None all physical quantities are set to nan.
    width_sample_step : sample the skeleton every this many pixels for
                        perpendicular FWHM width estimates
    fast : bool
        If True, skip perpendicular FWHM width sampling and ordered path
        construction. Width is approximated as area / skeleton_length.
        Tips are found from the degree-1 convolution result directly.
        This is ~8× faster and suitable for bulk processing.
        FWHM-based width can be computed on a subsample of frames separately.
    """
    m = ArcMeasurement()

    if not binary.any():
        return m

    m.arc_detected = True
    m.area_px = int(np.sum(binary))

    # --- bounding box ---
    rows_b, cols_b = np.where(binary)
    m.bbox_row_min = int(rows_b.min())
    m.bbox_row_max = int(rows_b.max())
    m.bbox_col_min = int(cols_b.min())
    m.bbox_col_max = int(cols_b.max())

    # --- intensity statistics ---
    arc_vals = gray[binary]
    m.mean_intensity = float(arc_vals.mean())
    m.peak_intensity = float(arc_vals.max())
    m.integrated_intensity = float(arc_vals.sum())

    # --- intensity-weighted centroid (vectorised) ---
    weights = gray * binary
    total_w = float(weights.sum())
    if total_w > 0:
        r_idx, c_idx = np.indices(gray.shape)
        m.centroid_row = float((r_idx * weights).sum() / total_w)
        m.centroid_col = float((c_idx * weights).sum() / total_w)

    # --- skeleton ---
    skel = skeletonize(binary)
    m.skeleton_n_pixels = int(skel.sum())
    m.arc_length_px = float(m.skeleton_n_pixels)

    # ── fast path ──────────────────────────────────────────────────────────
    if fast:
        # Width approximation: mean perpendicular width ≈ area / length
        if m.arc_length_px > 0:
            w_approx = float(m.area_px) / m.arc_length_px
            m.arc_width_px_mean = w_approx
            m.arc_width_px_median = w_approx

        # Tips: degree-1 skeleton pixels via convolution (no DFS walk)
        from scipy.ndimage import convolve
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
        deg_img = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
        ep_mask = skel & (deg_img == 1)
        eps = np.argwhere(ep_mask)
        if len(eps) >= 2:
            m.tip_a_row, m.tip_a_col = float(eps[0, 0]), float(eps[0, 1])
            m.tip_b_row, m.tip_b_col = float(eps[-1, 0]), float(eps[-1, 1])
            m.electrode_gap_px = float(np.hypot(
                m.tip_b_row - m.tip_a_row,
                m.tip_b_col - m.tip_a_col,
            ))
        elif len(eps) == 1:
            m.tip_a_row, m.tip_a_col = float(eps[0, 0]), float(eps[0, 1])
        # curvature: not computed in fast mode
        _apply_calibration(m, px_per_mm)
        return m

    # ── full path (ordered walk) ──────────────────────────────────────────
    path = _skeleton_ordered_coords(skel)
    if path is None or len(path) < 3:
        _apply_calibration(m, px_per_mm)
        return m

    # --- tip positions ---
    m.tip_a_row, m.tip_a_col = float(path[0, 0]), float(path[0, 1])
    m.tip_b_row, m.tip_b_col = float(path[-1, 0]), float(path[-1, 1])
    m.electrode_gap_px = float(np.hypot(
        m.tip_b_row - m.tip_a_row,
        m.tip_b_col - m.tip_a_col,
    ))

    # --- curvature ---
    diffs = np.diff(path.astype(float), axis=0)
    angles = np.arctan2(diffs[:, 0], diffs[:, 1])
    delta_theta = np.diff(angles)
    delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
    m.curvature_mean = float(np.abs(delta_theta).mean()) if len(delta_theta) else float('nan')

    # --- perpendicular width profiles ---
    widths: list[float] = []
    sample_indices = range(0, len(path), max(1, width_sample_step))
    for i in sample_indices:
        r, c = path[i]
        if 0 < i < len(path) - 1:
            tan = path[i + 1].astype(float) - path[i - 1].astype(float)
        elif i == 0:
            tan = path[1].astype(float) - path[0].astype(float)
        else:
            tan = path[-1].astype(float) - path[-2].astype(float)
        tan_norm = np.linalg.norm(tan)
        if tan_norm < 1e-9:
            continue
        tan = tan / tan_norm
        w = _fwhm_perpendicular(gray, float(r), float(c), (tan[0], tan[1]))
        if not np.isnan(w):
            widths.append(w)

    if widths:
        m.arc_width_px_mean = float(np.mean(widths))
        m.arc_width_px_median = float(np.median(widths))
    elif m.arc_length_px > 0:
        # fallback: area / length approximation
        w_approx = float(m.area_px) / m.arc_length_px
        m.arc_width_px_mean = w_approx
        m.arc_width_px_median = w_approx

    _apply_calibration(m, px_per_mm)
    return m


def _apply_calibration(m: 'ArcMeasurement', px_per_mm: Optional[float]) -> None:
    """Apply px_per_mm calibration to all physical fields in-place."""
    if px_per_mm is not None and px_per_mm > 0:
        mm_per_px = 1.0 / px_per_mm
        m.arc_length_mm = m.arc_length_px * mm_per_px
        if not np.isnan(m.arc_width_px_mean):
            m.arc_width_mm_mean = m.arc_width_px_mean * mm_per_px
        if not np.isnan(m.arc_width_px_median):
            m.arc_width_mm_median = m.arc_width_px_median * mm_per_px
        if not np.isnan(m.electrode_gap_px):
            m.electrode_gap_mm = m.electrode_gap_px * mm_per_px


# ---------------------------------------------------------------------------
# Calibration helper
# ---------------------------------------------------------------------------

def estimate_px_per_mm(
    measurements: list[ArcMeasurement],
    electrode_gap_mm: float = 2.4,
    percentile: float = 5.0,
    frame_height: int = 256,
    top_fraction: float = 0.30,
) -> Optional[float]:
    """Estimate px/mm from the electrode tip separation over the measurement series.

    Strategy
    --------
    In this experimental geometry the electrode roots are at the TOP of the
    frame.  The arc skeleton has its endpoints ('tips') at the electrode
    contact points.  We therefore:

    1. Filter to frames where **both** skeleton endpoints are in the top
       *top_fraction* of the frame height (i.e. rows < H × top_fraction).
       This selects frames where the arc hasn't elongated yet, and the
       endpoint-to-endpoint distance corresponds to the physical electrode
       separation.
    2. Take the **median** of the resulting gap_px values.  The median is
       robust to frames where path-ordering produces slightly asymmetric tips.
    3. Return  median_gap_px / electrode_gap_mm.

    Fallback
    --------
    If fewer than 5 frames pass the top-tip filter, fall back to the
    *percentile*-th percentile of all detected gaps (legacy behaviour).

    Parameters
    ----------
    measurements     : list of ArcMeasurement objects
    electrode_gap_mm : physical minimum electrode separation (mm).
    percentile       : percentile for the legacy fallback (default 5).
    frame_height     : image height in pixels (default 256).
    top_fraction     : tip rows must be < frame_height × top_fraction.
    """
    row_thresh = frame_height * top_fraction

    # Primary: both tips in top portion
    top_gaps = [
        m.electrode_gap_px for m in measurements
        if (m.arc_detected
            and not np.isnan(m.electrode_gap_px)
            and m.electrode_gap_px > 5          # reject sub-pixel artefacts
            and m.tip_a_row < row_thresh
            and m.tip_b_row < row_thresh)
    ]

    if len(top_gaps) >= 5:
        median_gap = float(np.median(top_gaps))
        if median_gap > 0:
            return median_gap / electrode_gap_mm

    # Fallback: percentile of all gaps
    gaps = [m.electrode_gap_px for m in measurements
            if m.arc_detected and not np.isnan(m.electrode_gap_px) and m.electrode_gap_px > 0]
    if not gaps:
        return None
    min_gap_px = float(np.percentile(gaps, percentile))
    if min_gap_px <= 0:
        return None
    return min_gap_px / electrode_gap_mm   # px per mm


def measure(binary: np.ndarray) -> Tuple[int, int]:
    """Legacy measure returning (area_px, arc_length_px)."""
    from skimage.morphology import skeletonize
    area = int(np.sum(binary))
    arc_length = int(np.sum(skeletonize(binary)))
    return area, arc_length
