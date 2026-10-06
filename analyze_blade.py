#!/usr/bin/env python3
"""
analyze_blade.py
================
Standalone plasma-arc analysis script for TIFF recordings.

Usage
-----
    python analyze_blade.py recording.tiff
    python analyze_blade.py recording.tiff --out-dir ./my_results
    python analyze_blade.py recording.tiff --fps 200000 --threshold 50
    python analyze_blade.py recording.tiff --px-per-mm 4.71   # skip auto-calib

Dependencies (pip install)
--------------------------
    numpy scipy scikit-image tifffile matplotlib pandas

What it produces (in --out-dir)
--------------------------------
    measurements.csv          per-frame metrics: arc length, width, tip
                              velocity, electrode gap, brightness, curvature
    velocity_map.png / .npy   2-D spatial map: each pixel coloured by
                              the average arc-tip speed when the arc
                              was present at that location
    timeseries.png            5-panel time series plot
    qa_frames/qa_NNNNNN.png   annotated QA frame every --qa-interval frames
"""

import sys, os, argparse, time, warnings, csv
import numpy as np
from cine_frames import open_frames
from skimage.filters import apply_hysteresis_threshold
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import label, binary_closing, convolve
from skimage.morphology import skeletonize, disk

import arc_filter  # single source of truth for flash / validity criteria
from arc_shape import resample_arc  # N-point whole-arc reconstruction (npoint_restrike's resampling)
from skimage.graph import route_through_array

ROUTE_MARGIN_PX = 2       # crop margin around the skeleton for the foot-to-foot route (speed only)


def skeleton_route(skel, a, b):
    """Shortest 8-connected path ALONG the skeleton from pixel a to pixel b ((row, col) ints), as an (M, 2)
    array, or None if they are not joined by skeleton pixels. Minimum-cost path (skimage MCP, geometric
    steps) on cost 1 over skeleton pixels; off-skeleton pixels are forbidden (infinite cost)."""
    if a is None or b is None:
        return None
    rr, cc = np.nonzero(skel)
    r0, c0 = max(rr.min() - ROUTE_MARGIN_PX, 0), max(cc.min() - ROUTE_MARGIN_PX, 0)
    r1, c1 = rr.max() + ROUTE_MARGIN_PX + 1, cc.max() + ROUTE_MARGIN_PX + 1
    sub = skel[r0:r1, c0:c1]
    s = (int(a[0]) - r0, int(a[1]) - c0)
    e = (int(b[0]) - r0, int(b[1]) - c0)
    if not (0 <= s[0] < sub.shape[0] and 0 <= s[1] < sub.shape[1] and sub[s]
            and 0 <= e[0] < sub.shape[0] and 0 <= e[1] < sub.shape[1] and sub[e]):
        return None
    cost = np.where(sub, 1.0, np.inf)
    try:
        path, total = route_through_array(cost, s, e, fully_connected=True, geometric=True)
    except ValueError:
        return None
    if not np.isfinite(total):
        return None
    return np.asarray(path) + np.array([r0, c0])

warnings.filterwarnings('ignore')

# ── defaults ───────────────────────────────────────────────────────────────────
DEFAULT_FPS           = 200_000
DEFAULT_CLOSE_RADIUS  = 3        # morphological closing disk radius
DEFAULT_ELECTRODE_GAP = 2.4      # known minimum electrode gap (mm)
DEFAULT_PX_PER_MM     = 9.167    # ground-truth: 22 px measured = 2.4 mm gap
DEFAULT_DARK_GUARD    = 35       # frames with max DN < this are skipped immediately
DEFAULT_BRIGHT_THRESH = 80       # pixels above this DN are considered arc signal
DEFAULT_MAX_ARC_PX    = 3000     # maximum credible arc area (thin channel only)

# Minimum arc area: the arc body must span at least the electrode gap.
DEFAULT_MIN_ARC_PX = int(DEFAULT_ELECTRODE_GAP * DEFAULT_PX_PER_MM)  # ≈ 22 px

# Tolerance for blade-line attachment check (perpendicular distance in px).
# Must be < half the apex gap (22 px for ADCB, ~8.8 px/mm × 2.4 mm) so the
# two blade tolerance bands never overlap.
ELEC_TOL_PX = 4
DEFAULT_QA_INTERVAL   = 500      # save one QA frame every N frames
DEFAULT_TOP_FRACTION  = 0.25     # top fraction of frame for electrode detection

# ── image processing ───────────────────────────────────────────────────────────

def binarize(raw_uint8, close_r,
             dark_guard=DEFAULT_DARK_GUARD,
             bright_thresh=DEFAULT_BRIGHT_THRESH,
             min_arc_px=DEFAULT_MIN_ARC_PX,
             max_arc_px=DEFAULT_MAX_ARC_PX,
             low_thresh=None,
             low_frac=None):
    """
    Fixed-threshold arc detection: pixels above bright_thresh DN are arc
    candidates; morphological closing fills gaps; the largest connected
    component is kept if its area is within the credible range.

    The threshold is FIXED (per recording) rather than adaptive (Otsu) for
    two reasons.  First, a per-recording threshold is calibrated against the
    arc-vs-background separation measured directly from the data (for the
    BLADE N2 reactor this is ~80 DN; for the J-electrode CO2 sweep it is
    bright_thresh=210, chosen from the arc/plume local-intensity QA in
    qa_arc_local_intensity_j.py, where the arc core saturates near 255 while
    the surrounding plume sits below ~150).  Second, per-frame adaptive
    thresholds amplify noise on dark inter-arc frames by stretching the
    histogram, creating spurious detections that corrupt event detection and
    Mode A/B classification.  The caller is responsible for passing the
    bright_thresh appropriate to the recording; this function does not assume
    a single dataset-wide value.

    Parameters
    ----------
    dark_guard    : frames whose max DN < this are skipped (no signal at all)
    bright_thresh : fixed DN threshold; pixels above this count as arc signal
    min_arc_px    : reject mask if total bright area < this (isolated speckle)
    max_arc_px    : reject mask if total bright area > this (noise flood)
    low_thresh    : if given, HYSTERESIS thresholding (Canny 1986;
                    skimage.filters.apply_hysteresis_threshold): pixels > low_thresh
                    connected to a pixel > bright_thresh.  For 12-bit cine input, where
                    brightness varies along one arc (bright apex/roots, dim legs) and no
                    single threshold keeps the whole arc.  None = original fixed threshold.
    """
    if int(raw_uint8.max()) < dark_guard:
        return np.zeros(raw_uint8.shape, bool)
    if low_thresh is None:
        b = raw_uint8 > bright_thresh
    else:
        # low_frac: the arc edge is where brightness falls to low_frac of THIS frame's peak, never
        # below the absolute artefact floor low_thresh (a glow halo around a saturated filament
        # otherwise joins the mask; OVERNIGHT_LOG V-H3 on Argon)
        lo = low_thresh if low_frac is None else max(low_thresh, low_frac * float(raw_uint8.max()))
        b = apply_hysteresis_threshold(raw_uint8, lo, max(bright_thresh, lo))
    if b.sum() < min_arc_px:
        return np.zeros(raw_uint8.shape, bool)
    if low_thresh is not None:
        # Same closing + largest component, computed on the bounding box of b padded by
        # close_r + 2.  Closing cannot reach further than close_r from b, and scipy pads with
        # 0 at the crop edge exactly as it does at the frame edge, so the result is identical
        # (checked frame-by-frame in test_cine_frames) at ~1/4 of the cost.
        rr = np.nonzero(b.any(1))[0]; cc = np.nonzero(b.any(0))[0]
        pad = close_r + 2
        r0, r1 = max(rr[0] - pad, 0), min(rr[-1] + pad + 1, b.shape[0])
        c0, c1 = max(cc[0] - pad, 0), min(cc[-1] + pad + 1, b.shape[1])
        sub = binary_closing(b[r0:r1, c0:c1], structure=disk(close_r))
        lab, n = label(sub)
        if n == 0:
            return np.zeros(raw_uint8.shape, bool)
        sz = np.bincount(lab.ravel()); sz[0] = 0
        mask = np.zeros(b.shape, bool)
        mask[r0:r1, c0:c1] = lab == sz.argmax()
        area = int(mask.sum())
        if area < min_arc_px or area > max_arc_px:
            return np.zeros(raw_uint8.shape, bool)
        return mask
    b = binary_closing(b, structure=disk(close_r))
    labeled, n = label(b)
    if n == 0:
        return np.zeros(raw_uint8.shape, bool)
    sz = np.bincount(labeled.ravel()); sz[0] = 0
    mask = labeled == sz.argmax()
    area = int(mask.sum())
    if area < min_arc_px or area > max_arc_px:
        return np.zeros(raw_uint8.shape, bool)
    return mask

def _bbox_crop(img, margin):
    """(img[r0:r1, c0:c1], r0, c0): the bounding box of img's nonzero pixels plus `margin` px (clipped to the
    frame). SPEED ONLY: every caller does a local (3x3) operation with zero outside the image, which is
    identical on this crop because everything outside the box is zero."""
    rr, cc = np.nonzero(img)
    if rr.size == 0:
        return img, 0, 0
    r0, c0 = max(int(rr.min()) - margin, 0), max(int(cc.min()) - margin, 0)
    return img[r0:int(rr.max()) + margin + 1, c0:int(cc.max()) + margin + 1], r0, c0


def _skel_neighbours(skel):
    """3x3 neighbour count of every skeleton pixel (zero padding), on the skeleton's bounding box."""
    sub, r0, c0 = _bbox_crop(skel, 1)
    kern = np.ones((3,3), np.uint8); kern[1,1] = 0
    return sub, convolve(sub.astype(np.uint8), kern, mode='constant'), r0, c0


def skeleton_endpoints(skel):
    sub, nb, r0, c0 = _skel_neighbours(skel)
    r, c = np.where(sub & (nb == 1))
    return np.column_stack([r + r0, c + c0]) if r.size else np.empty((0,2), int)

def find_electrode_tips(eps, H, top_fraction):
    """Widest-separated endpoint pair in the top fraction of the frame."""
    if eps.shape[0] < 2: return None, None
    cands = eps[eps[:,0] <= int(H * top_fraction)]
    if cands.shape[0] < 2:
        idx = np.argsort(eps[:,0]); cands = eps[idx[:10]]
    if cands.shape[0] < 2: return None, None
    best, ta, tb = -1, cands[0], cands[1]
    for i in range(len(cands)):
        for j in range(i+1, len(cands)):
            s = abs(int(cands[i,1]) - int(cands[j,1]))
            if s > best: best, ta, tb = s, cands[i], cands[j]
    return ta, tb

def _dist_to_line(pt, p0, p1):
    """Perpendicular distance from pt=(row,col) to the infinite line through p0–p1."""
    d = np.asarray(p1, float) - np.asarray(p0, float)
    n = np.linalg.norm(d)
    if n < 1e-9:
        return float(np.linalg.norm(np.asarray(pt, float) - np.asarray(p0, float)))
    return float(abs(np.cross(d / n, np.asarray(pt, float) - np.asarray(p0, float))))


def _proj_param_px(pt, p0, p1):
    """Signed arc-length (px) of pt's projection onto the line p0→p1, measured
    from p0.  0 = at p0, L = at p1 (L = |p1−p0|).  Used for the finite-segment
    (J-electrode) attachment gate."""
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return 0.0, 0.0
    t = float(np.dot(np.asarray(pt, float) - p0, d) / L)
    return t, L


def _on_segment(pt, p0, p1, margin_px):
    """True if pt projects onto the finite segment p0–p1 (± margin_px past
    either end)."""
    t, L = _proj_param_px(pt, p0, p1)
    return (-margin_px) <= t <= (L + margin_px)


def _arc_complete_blade_lines(eps, blade_lines, tol_px,
                              finite_segments=False, seg_margin_px=0.0):
    """
    Assign skeleton endpoints to blade A and blade B using perpendicular
    distance to each blade line.

    Parameters
    ----------
    eps             : (N,2) array of skeleton endpoints [row, col]
    blade_lines     : (2,2,2) array  [blade_A, blade_B] × [pt0, pt1] × [row, col]
    tol_px          : maximum perpendicular distance (px) to be considered attached
    finite_segments : J-electrode mode.  When True the two points of each blade
                      are treated as the FINITE electrode segment (pt0 = tip-end
                      near the apex, pt1 = far-end), and a foot only counts as
                      attached if its perpendicular distance ≤ tol_px AND its
                      projection falls within the segment (± seg_margin_px).
                      This rejects a free arc tip that lies on the extrapolated
                      blade line but BEYOND the physical electrode end.  When
                      False (BLADE 11MAY26 default) the blades are infinite lines
                      and only the perpendicular distance is used — behaviour is
                      byte-for-byte unchanged.
    seg_margin_px   : tolerance (px) past either clicked end (finite mode only).

    Returns
    -------
    ta, tb          : arrays (2,) — endpoint assigned to blade A and B
    attached_a/b    : bool — True if attached to respective blade
    """
    if eps.shape[0] < 2:
        return None, None, False, False

    bl_a, bl_b = blade_lines[0], blade_lines[1]
    dists_a = np.array([_dist_to_line(ep, bl_a[0], bl_a[1]) for ep in eps])
    dists_b = np.array([_dist_to_line(ep, bl_b[0], bl_b[1]) for ep in eps])

    if finite_segments:
        on_a = np.array([_on_segment(ep, bl_a[0], bl_a[1], seg_margin_px) for ep in eps])
        on_b = np.array([_on_segment(ep, bl_b[0], bl_b[1], seg_margin_px) for ep in eps])
    else:
        on_a = np.ones(len(eps), bool)
        on_b = np.ones(len(eps), bool)

    # Find the assignment of two distinct endpoints to (blade A, blade B) that
    # minimises the total perpendicular distance.
    best_score = np.inf
    ta = tb = None
    att_a = att_b = False
    for ia in range(len(eps)):
        for ib in range(len(eps)):
            if ia == ib:
                continue
            score = dists_a[ia] + dists_b[ib]
            if score < best_score:
                best_score = score
                ta        = eps[ia]
                tb        = eps[ib]
                att_a     = bool(dists_a[ia] <= tol_px and on_a[ia])
                att_b     = bool(dists_b[ib] <= tol_px and on_b[ib])
    return ta, tb, att_a, att_b


def _min_dist_to_points(pt, pts):
    """Euclidean distance from pt=(row,col) to the NEAREST point in pts (M,2)."""
    d = np.asarray(pts, float) - np.asarray(pt, float)
    return float(np.sqrt((d[:, 0] ** 2 + d[:, 1] ** 2).min()))


def _arc_complete_edge_points(eps, edge_a, edge_b, tol_px):
    """
    Curved-electrode attachment test for the J-hook geometry.

    Instead of perpendicular distance to a straight blade LINE (which mis-models a
    curved J-hook), this measures each arc-foot's distance to the nearest point of
    the actual detected electrode EDGE point-set (built by tracking arc attachment
    sites across all frames).  A foot counts as attached to an electrode if its
    nearest-edge-point distance ≤ tol_px.  The two distinct skeleton endpoints that
    minimise (dist→edge_A + dist→edge_B) are assigned to electrodes A and B.

    Parameters
    ----------
    eps             : (N,2) skeleton endpoints [row, col]
    edge_a, edge_b  : (Ma,2)/(Mb,2) electrode-edge point clouds [row, col]
    tol_px          : max distance (px) to the nearest edge point to count attached

    Returns ta, tb, attached_a, attached_b  (same contract as
    _arc_complete_blade_lines), so the rest of measure_frame is unchanged.
    """
    if eps.shape[0] < 2 or edge_a is None or edge_b is None \
       or len(edge_a) == 0 or len(edge_b) == 0:
        return None, None, False, False
    dists_a = np.array([_min_dist_to_points(ep, edge_a) for ep in eps])
    dists_b = np.array([_min_dist_to_points(ep, edge_b) for ep in eps])
    best_score = np.inf
    ta = tb = None
    att_a = att_b = False
    for ia in range(len(eps)):
        for ib in range(len(eps)):
            if ia == ib:
                continue
            score = dists_a[ia] + dists_b[ib]
            if score < best_score:
                best_score = score
                ta, tb = eps[ia], eps[ib]
                att_a  = bool(dists_a[ia] <= tol_px)
                att_b  = bool(dists_b[ib] <= tol_px)
    return ta, tb, att_a, att_b


def arc_front_tip(binary, direction='down'):
    """
    Arc propagation front pixel.
    direction='down' → bottommost pixel (row argmax); reactor right-side up.
    direction='up'   → topmost pixel   (row argmin); reactor inverted (upward video).
    """
    if not binary.any(): return np.nan, np.nan
    r, c = np.where(binary)
    idx  = r.argmax() if direction == 'down' else r.argmin()
    return float(r[idx]), float(c[idx])

def arc_front_row_extent(binary, direction='down'):
    """
    Column extent of the lit pixels in the front row (the row arc_front_tip returns).
    arc_front_tip takes the FIRST pixel of that row in raster order, i.e. the LEFTMOST,
    and the front row holds more than one lit pixel on most frames, so its column is
    biased left. Returns (leftmost col, rightmost col, mean col of the row's lit pixels).
    """
    if not binary.any(): return np.nan, np.nan, np.nan
    r, c = np.where(binary)
    row = r.max() if direction == 'down' else r.min()
    cc = c[r == row]
    return float(cc.min()), float(cc.max()), float(cc.mean())

def fwhm_perpendicular(gray_n, coords, sample_step=10, half_width=20):
    H, W = gray_n.shape; n = len(coords)
    if n < 3: return np.nan
    widths = []
    for idx in range(0, n, sample_step):
        i = min(max(idx,1), n-2)
        tang = coords[i+1] - coords[i-1]
        tn = np.linalg.norm(tang)
        if tn < 1e-6: continue
        tang /= tn; perp = np.array([-tang[1], tang[0]])
        r0, c0 = coords[i]; ts = np.arange(-half_width, half_width+1)
        rs = r0 + ts*perp[0]; cs = c0 + ts*perp[1]
        prof = np.zeros(len(ts))
        for k, (r, c) in enumerate(zip(rs, cs)):
            ri, ci = int(np.floor(r)), int(np.floor(c))
            if ri<0 or ri+1>=H or ci<0 or ci+1>=W: continue
            dr, dc = r-ri, c-ci
            prof[k] = (gray_n[ri,ci]*(1-dr)*(1-dc) + gray_n[ri,ci+1]*(1-dr)*dc
                     + gray_n[ri+1,ci]*dr*(1-dc)   + gray_n[ri+1,ci+1]*dr*dc)
        pk = prof.max()
        if pk < 1e-6: continue
        ab = np.where(prof >= pk/2)[0]
        if len(ab) >= 2: widths.append(ab[-1]-ab[0]+1)
    return float(np.mean(widths)) if widths else np.nan

def ordered_skeleton_coords(skel):
    pts = set(map(tuple, np.column_stack(np.where(skel))))
    if not pts: return np.empty((0,2))
    sub, nb, r0, c0 = _skel_neighbours(skel)
    er, ec = np.where(sub & (nb==1))
    eps = list(zip(er + r0, ec + c0))
    start = eps[0] if eps else next(iter(pts))
    order, visited = [start], {start}
    while True:
        r, c = order[-1]; found = None
        for dr in (-1,0,1):
            for dc in (-1,0,1):
                if dr==dc==0: continue
                p = (r+dr, c+dc)
                if p in pts and p not in visited: found=p; break
            if found: break
        if found is None: break
        order.append(found); visited.add(found)
    return np.array(order, float)

# ── tortuosity & midline asymmetry ──────────────────────────────────────────────

def _polyline_length(P):
    """Euclidean path length of an ordered (N,2) polyline (px)."""
    if P is None or len(P) < 2:
        return 0.0
    d = np.diff(np.asarray(P, float), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def midline_from_blades(blade_lines):
    """Build the fixed midline between the two electrodes from blade_lines.

    blade_lines : (2,2,2) array  [blade_A, blade_B] × [pt_top, pt_bot] × [row, col]

    The midline is the mean of the two blade lines — the line down the centre of
    the electrode gap.  Returns (M0, u, sideA_sign) where M0 is a point on the
    midline (row,col), u is the unit direction (row,col), and sideA_sign (±1)
    orients the signed offset so that positive = the blade-A side.
    Returns None if blade_lines is unusable.
    """
    if blade_lines is None:
        return None
    bl = np.asarray(blade_lines, float)
    if bl.shape != (2, 2, 2):
        return None
    A0, A1 = bl[0, 0], bl[0, 1]
    B0, B1 = bl[1, 0], bl[1, 1]
    # Orient blade B the same way as blade A so the midline direction
    # u = (dA + dB)/2 is the ANGLE bisector of the wedge — NOT the perpendicular
    # bisector.  The stored endpoint order (the SVD sign in detect_blade_lines.py)
    # is otherwise arbitrary and a mismatched sign would silently flip the
    # midline.  Guard here so any blade_lines.npy (auto or hand-made) is safe.
    dA = A1 - A0
    dB = B1 - B0
    if float(dA[0] * dB[0] + dA[1] * dB[1]) < 0.0:
        B0, B1 = B1, B0
    M_top = 0.5 * (A0 + B0)
    M_bot = 0.5 * (A1 + B1)
    u = M_bot - M_top
    n = np.hypot(u[0], u[1])
    if n < 1e-9:
        return None
    u = u / n
    # signed offset of blade-A midpoint; sign chosen so A-side is positive
    pA = 0.5 * (A0 + A1)
    dpA = pA - M_top
    offA = u[1] * dpA[0] - u[0] * dpA[1]
    sideA_sign = 1.0 if offA >= 0 else -1.0
    return M_top, u, sideA_sign


def midline_from_edge_tips(a_tip, b_tip, arc_direction='down'):
    """Midline for the curved J-hook geometry, anchored on the two FIXED facing
    hook tips (a_tip, b_tip = electrode A/B tip-ends near the gap).

    The midline starts at the midpoint of the two hook tips and runs in the arc
    glide direction (down for arc_direction='down', i.e. increasing row), so the
    A/B asymmetry split is anchored on the fixed electrode gap rather than on the
    wobbling per-frame arc.  Returns (M0, u, sideA_sign), positive offset on the
    electrode-A side, or None if the tips coincide / are missing."""
    if a_tip is None or b_tip is None:
        return None
    a_tip = np.asarray(a_tip, float); b_tip = np.asarray(b_tip, float)
    g = b_tip - a_tip
    n = np.hypot(g[0], g[1])
    if n < 1e-6:
        return None
    M0 = 0.5 * (a_tip + b_tip)
    # glide direction: perpendicular to the gap, pointing to larger rows (down)
    u = np.array([-g[1], g[0]]) / n
    if (arc_direction == 'down' and u[0] < 0) or (arc_direction == 'up' and u[0] > 0):
        u = -u
    dpa = a_tip - M0
    offA = u[1] * dpa[0] - u[0] * dpa[1]
    sideA_sign = 1.0 if offA >= 0 else -1.0
    return M0, u, sideA_sign


def midline_from_tips(ta, tb):
    """Fallback midline: perpendicular bisector of the two arc roots ta–tb.

    Returns (M0, u, sideA_sign) with positive offset on the ta (blade-A) side,
    or None if the roots coincide.
    """
    if ta is None or tb is None:
        return None
    ta = np.asarray(ta, float); tb = np.asarray(tb, float)
    g = tb - ta
    n = np.hypot(g[0], g[1])
    if n < 1e-6:
        return None
    M0 = 0.5 * (ta + tb)
    # midline runs perpendicular to the gap g
    u = np.array([-g[1], g[0]]) / n
    dpa = ta - M0
    offA = u[1] * dpa[0] - u[0] * dpa[1]
    sideA_sign = 1.0 if offA >= 0 else -1.0
    return M0, u, sideA_sign


def _signed_offsets(P, M0, u, sideA_sign):
    """Signed perpendicular offset of each (row,col) point from the midline.
    Positive = blade-A side."""
    dp = np.asarray(P, float) - np.asarray(M0, float)
    return sideA_sign * (u[1] * dp[:, 0] - u[0] * dp[:, 1])


def arc_asymmetry(coords, ta, tb, midline, px_per_mm=None):
    """Split the arc at the electrode midline and measure length / tortuosity
    asymmetry between the blade-A half and the blade-B half.

    coords  : ordered (N,2) skeleton polyline [row,col]
    ta, tb  : arc roots assigned to blade A and blade B
    midline : (M0, u, sideA_sign) from midline_from_blades / midline_from_tips

    Returns a dict of:
      arclen_a_px, arclen_b_px : path length of each half (px)
      arclen_asym              : (L_A − L_B)/(L_A + L_B)  (+ = more on A side)
      tort_a, tort_b           : per-half arc-chord tortuosity (root→crossing)
      tort_asym                : (τ_A − τ_B)/(τ_A + τ_B)   (+ = A half twistier)
      mid_cross_row/col        : where the arc crosses the midline (apex split)
    All NaN when the arc is too short, the roots are missing, or no midline.
    """
    nan = np.nan
    out = dict(arclen_a_px=nan, arclen_b_px=nan, arclen_asym=nan,
               tort_a=nan, tort_b=nan, tort_asym=nan,
               mid_cross_row=nan, mid_cross_col=nan)
    if midline is None or coords is None or len(coords) < 3 or ta is None or tb is None:
        return out
    M0, u, sideA_sign = midline
    P = np.asarray(coords, float)
    ta = np.asarray(ta, float); tb = np.asarray(tb, float)
    # orient the path so P[0] is the blade-A root
    if np.hypot(*(P[0] - ta)) > np.hypot(*(P[-1] - ta)):
        P = P[::-1]
    s = _signed_offsets(P, M0, u, sideA_sign)
    seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    Ltot = float(cum[-1])
    if Ltot < 1e-6:
        return out
    # midline crossings (sign changes / zeros) with interpolated path position
    crossings = []
    for i in range(len(s) - 1):
        if s[i] == 0.0:
            crossings.append((cum[i], P[i]))
        elif s[i] * s[i + 1] < 0:
            t = s[i] / (s[i] - s[i + 1])
            X = P[i] + t * (P[i + 1] - P[i])
            crossings.append((cum[i] + t * seg[i], X))
    if not crossings:
        # arc lies entirely on one side of the midline → maximally asymmetric
        on_A = s.mean() > 0
        out['arclen_a_px'] = Ltot if on_A else 0.0
        out['arclen_b_px'] = 0.0 if on_A else Ltot
        out['arclen_asym'] = 1.0 if on_A else -1.0
        return out
    # dominant crossing = the one nearest the path midpoint (the apex)
    poss = np.array([c[0] for c in crossings])
    X = crossings[int(np.argmin(np.abs(poss - Ltot / 2.0)))][1]
    L_A = float(poss[int(np.argmin(np.abs(poss - Ltot / 2.0)))])
    L_B = Ltot - L_A
    chord_A = max(float(np.hypot(*(P[0] - X))), 1.0)
    chord_B = max(float(np.hypot(*(P[-1] - X))), 1.0)
    tort_A = L_A / chord_A
    tort_B = L_B / chord_B
    out['arclen_a_px'] = L_A
    out['arclen_b_px'] = L_B
    out['arclen_asym'] = (L_A - L_B) / (L_A + L_B) if (L_A + L_B) > 0 else nan
    out['tort_a'] = tort_A
    out['tort_b'] = tort_B
    out['tort_asym'] = (tort_A - tort_B) / (tort_A + tort_B) if (tort_A + tort_B) > 0 else nan
    out['mid_cross_row'] = float(X[0])
    out['mid_cross_col'] = float(X[1])
    return out

# ── calibration ────────────────────────────────────────────────────────────────

class FrameWindow:
    """frames [start, start+count) of a frame source, renumbered from 0."""
    def __init__(self, src, start, count):
        if start < 0 or count <= 0 or start + count > src.n:
            raise ValueError('window [%d,%d) outside [0,%d)' % (start, start + count, src.n))
        self.src, self.start, self.n = src, int(start), int(count)
        self.shape, self.defects = src.shape, getattr(src, 'defects', None)
    def frame(self, i):
        if not 0 <= i < self.n:
            raise IndexError(i)
        return self.src.frame(self.start + i)


def calibrate_px_per_mm(tif, n_frames, electrode_gap_mm,
                         close_r, top_fraction, sample_every=20):
    """
    Find px/mm from the known minimum electrode gap.
    Samples frames every `sample_every` steps, finds the two electrode-tip
    skeleton endpoints (widest pair in top fraction of frame), collects
    their pixel separations. The 10th percentile corresponds to the
    shortest arc (at breakdown) = electrode_gap_mm.
    NOTE: auto-calibration is retained as a fallback but the default pipeline
    uses DEFAULT_PX_PER_MM (ground-truth hand measurement: 22 px = 2.4 mm).
    """
    print("  Calibrating scale (sampling every %d frames)..." % sample_every)
    gaps = []
    for i in range(0, n_frames, sample_every):
        raw = tif.frame(i)
        b   = binarize(raw, close_r)
        if not b.any(): continue
        sk  = skeletonize(b)
        if not sk.any(): continue
        eps = skeleton_endpoints(sk)
        ta, tb = find_electrode_tips(eps, raw.shape[0], top_fraction)
        if ta is None: continue
        g = float(np.hypot(ta[0]-tb[0], ta[1]-tb[1]))
        if g > 2: gaps.append(g)
    if len(gaps) < 3:
        print("  WARNING: calibration failed, defaulting to 4.0 px/mm")
        return 4.0
    min_gap = np.percentile(gaps, 10)
    pxmm    = min_gap / electrode_gap_mm
    print("  %d samples | 10th-pct gap = %.1f px → %.2f px/mm (%.0f µm/px)" % (
        len(gaps), min_gap, pxmm, 1000/pxmm))
    return pxmm

# ── per-frame measurement ──────────────────────────────────────────────────────

def measure_frame(raw, close_r, px_per_mm, top_fraction,
                   keep_overlays=False,
                   dark_guard=DEFAULT_DARK_GUARD,
                   bright_thresh=DEFAULT_BRIGHT_THRESH,
                   min_arc_px=DEFAULT_MIN_ARC_PX,
                   max_arc_px=DEFAULT_MAX_ARC_PX,
                   blade_lines=None,
                   elec_tol_px=ELEC_TOL_PX,
                   arc_direction='down',
                   gap_row=None,
                   finite_segments=False,
                   seg_margin_px=0.0,
                   electrode_edges=None,
                   flash_eff_width=arc_filter.FLASH_EFF_WIDTH,
                   flash_arc_width=arc_filter.FLASH_ARC_WIDTH,
                   allow_tip_midline_fallback=False,
                   low_thresh=None, low_frac=None, npoint_n=0):
    # electrode_edges (J-hook geometry): (edge_a, edge_b, a_tip, b_tip) where
    # edge_a/edge_b are (M,2) point clouds of the detected electrode edges and
    # a_tip/b_tip are the facing hook tips.  When supplied it OVERRIDES the
    # straight blade_lines attachment test with the curved edge-point test and
    # anchors the asymmetry midline on the fixed hook tips.
    H, W = raw.shape
    gray = raw.astype(np.float32)
    lo, hi = gray.min(), gray.max()
    gray_n = (gray-lo)/(hi-lo) if hi>lo else gray
    binary = binarize(raw, close_r,
                      dark_guard=dark_guard, bright_thresh=bright_thresh,
                      min_arc_px=min_arc_px, max_arc_px=max_arc_px,
                      low_thresh=low_thresh, low_frac=low_frac)

    m = dict(arc_detected=False, arc_complete=False, area_px=0,
             arc_length_px=0, arc_length_mm=np.nan,
             arc_width_px=np.nan, arc_width_mm=np.nan,
             mean_intensity=np.nan, peak_intensity=np.nan,
             integrated_intensity=np.nan,
             centroid_row=np.nan, centroid_col=np.nan,
             arc_tip_row=np.nan, arc_tip_col=np.nan,
             arc_tip_col_left=np.nan, arc_tip_col_right=np.nan, arc_tip_col_mean=np.nan,
             tip_a=None, tip_b=None,
             electrode_gap_px=np.nan, electrode_gap_mm=np.nan,
             curvature_mean=np.nan, bbox=None,
             path_len_px=np.nan, tortuosity=np.nan, total_curvature_rad=np.nan,
             arclen_a_px=np.nan, arclen_b_px=np.nan, arclen_asym=np.nan,
             tort_a=np.nan, tort_b=np.nan, tort_asym=np.nan,
             mid_cross_row=np.nan, mid_cross_col=np.nan,
             is_flash=False)
    if keep_overlays:
        m['_binary'] = binary; m['_skel_coords'] = np.empty((0,2))
    if not binary.any(): return m

    m['arc_detected'] = True; m['area_px'] = int(binary.sum())
    # Radiometry uses RAW detector counts (DN, 0-255), NOT the per-frame
    # min-max-normalised gray_n.  gray_n is rescaled independently every frame
    # (its max maps to 1.0), so intensities taken from it are not comparable
    # between frames or recordings — peak_intensity would be ~1.0 on every
    # detected frame and integrated/mean intensity would track contrast, not
    # radiated power.  gray_n is kept only for the FWHM width profile (a purely
    # relative shape measurement).
    ri, ci = np.where(binary)
    wr = raw[binary].astype(np.float64); wsr = wr.sum()
    # ── electrode-gap cutoff (flash rejection) ────────────────────────────────
    # A real arc must not extend past the user-defined electrode-gap row into the
    # electrode/flash zone.  'down': electrodes at SMALL rows, so any arc pixel
    # above the line (row < gap_row) means the bright blob has bloomed back over
    # the electrodes — a flash.  'up': electrodes at LARGE rows, discard side is
    # row > gap_row.  Reject the WHOLE frame (treat as no arc) so it drops out of
    # detection, velocity and the 2-D map alike. Uses the full binarized mask,
    # which the rest of the pipeline already treats as "the arc".
    if gap_row is not None:
        crosses = (int(ri.min()) < gap_row) if arc_direction == 'down' \
                  else (int(ri.max()) > gap_row)
        if crosses:
            m['arc_detected'] = False
            return m
    # Store sparse arc-pixel coordinates so the 2-D velocity map can be
    # accumulated in this single pass — no second read of the TIFF.
    m['_arc_rc'] = (ri.astype(np.int16), ci.astype(np.int16))
    m['mean_intensity'] = float(wr.mean())          # mean arc DN (0-255)
    m['peak_intensity'] = float(wr.max())           # brightest arc DN (0-255)
    m['integrated_intensity'] = float(wr.sum())     # sum of arc DN (DN*px)
    if wsr > 0:
        m['centroid_row'] = float((ri*wr).sum()/wsr)
        m['centroid_col'] = float((ci*wr).sum()/wsr)
    m['bbox'] = (int(ri.min()), int(ci.min()), int(ri.max()), int(ci.max()))
    _tip_r, _tip_c = arc_front_tip(binary, arc_direction)
    # Store arc_tip_row in consistent "distance from electrode" coordinates so
    # event_velocity.py works identically for both orientations.
    # downward: electrode is at SMALL rows; arc_tip_row = bottommost pixel
    #           → tip row increases as arc extends.  No flip needed.
    # upward:   electrode is at LARGE rows; arc_tip_row would DECREASE as arc
    #           extends.  Flip so tip row increases away from electrode → same
    #           convention, event_velocity.py advancing-front logic unchanged.
    if arc_direction == 'up' and not np.isnan(float(_tip_r)):
        _tip_r = float(H - 1 - _tip_r)
    m['arc_tip_row'], m['arc_tip_col'] = _tip_r, _tip_c
    (m['arc_tip_col_left'], m['arc_tip_col_right'],
     m['arc_tip_col_mean']) = arc_front_row_extent(binary, arc_direction)

    # skeletonize on the mask's bounding box (+2 px of zeros), pasted back: speed only, byte-identical skeleton
    _sub, _r0, _c0 = _bbox_crop(binary, 2)
    skel = np.zeros_like(binary, dtype=bool)
    skel[_r0:_r0 + _sub.shape[0], _c0:_c0 + _sub.shape[1]] = skeletonize(_sub)
    if not skel.any(): return m
    coords = ordered_skeleton_coords(skel)
    if keep_overlays: m['_skel_coords'] = coords
    # Arc length = TOTAL skeleton pixel count (all branches).  Previously this was
    # len(ordered_skeleton_coords), which follows only ONE branch and so grossly
    # under-measures FORKED arcs — that collapsed length blew up the flash
    # eff_width = area/arc_length_px metric and silently dropped ~5-14% of real
    # deep forked arcs as "flash" (see QA report / METHODS §6.6).  skel.sum() is
    # branch-complete and robust: a genuine saturation blob still has a short
    # medial axis (high eff_width → flagged), while a forked thin arc keeps its
    # full length (low eff_width → kept).
    m['arc_length_px'] = int(skel.sum())
    if px_per_mm: m['arc_length_mm'] = m['arc_length_px']/px_per_mm
    # Euclidean path length (diagonal-aware), single-branch (ordered coords).
    m['path_len_px'] = _polyline_length(coords)
    if len(coords) >= 3:
        d = np.diff(coords, axis=0)
        ang = np.arctan2(d[:,0], d[:,1])
        dt  = np.diff(ang); dt = (dt+np.pi)%(2*np.pi)-np.pi
        m['curvature_mean'] = float(np.abs(dt).mean())
        m['total_curvature_rad'] = float(np.abs(dt).sum())
        wm = fwhm_perpendicular(gray_n, coords)
        m['arc_width_px'] = wm
        if px_per_mm and not np.isnan(wm): m['arc_width_mm'] = wm/px_per_mm
    # Canonical flash flag (arc_filter criterion 3), written to the CSV so every
    # downstream stage reads one consistent value rather than recomputing it.
    m['is_flash'] = arc_filter.is_flash(m['area_px'], m['arc_length_px'],
                                        m['arc_width_px'],
                                        eff_width=flash_eff_width,
                                        arc_width=flash_arc_width)
    eps = skeleton_endpoints(skel)

    # ── fully-connected check ─────────────────────────────────────────────────
    # A physically meaningful arc must bridge BOTH electrodes.
    # If blade_lines.npy is available, use perpendicular distance to each blade
    # line (ELEC_TOL_PX tolerance).  Without it, fall back to the widest-pair
    # heuristic restricted to the electrode end of the frame.
    # For direction='up' the electrodes are at the BOTTOM of the frame, so the
    # heuristic must search there instead of the top.
    if electrode_edges is not None:
        # J-hook geometry: attachment judged against the actual detected curved
        # electrode-edge point clouds (not a straight line).
        edge_a, edge_b = electrode_edges[0], electrode_edges[1]
        ta, tb, att_a, att_b = _arc_complete_edge_points(
            eps, edge_a, edge_b, elec_tol_px)
        arc_complete = att_a and att_b
    elif blade_lines is not None:
        ta, tb, att_a, att_b = _arc_complete_blade_lines(
            eps, blade_lines, elec_tol_px,
            finite_segments=finite_segments, seg_margin_px=seg_margin_px)
        arc_complete = att_a and att_b
    else:
        if arc_direction == 'up':
            # flip eps vertically so find_electrode_tips (which looks at small row
            # indices) targets the actual electrode end
            eps_flipped = eps.copy(); eps_flipped[:, 0] = H - 1 - eps_flipped[:, 0]
            ta_f, tb_f = find_electrode_tips(eps_flipped, H, top_fraction)
            # map back to real coordinates
            ta = np.array([H - 1 - ta_f[0], ta_f[1]]) if ta_f is not None else None
            tb = np.array([H - 1 - tb_f[0], tb_f[1]]) if tb_f is not None else None
        else:
            ta, tb = find_electrode_tips(eps, H, top_fraction)
        arc_complete = (ta is not None) and (tb is not None)

    m['tip_a'], m['tip_b'] = ta, tb

    if not arc_complete:
        m['arc_detected'] = False
        m['arc_complete'] = False
        return m

    m['arc_complete'] = True
    if npoint_n:
        # N points equally spaced (resample_arc, npoint_restrike's) along the skeleton route FROM the foot on
        # electrode A TO the foot on electrode B, RAW (row, col); point 0 is therefore always on A. The J suite's
        # ordered_path starts at the first endpoint in raster order, which for a two-footed BLADE arch is often a
        # spur at the apex: on Air Re2250 t1 its ends were not the two attached feet in 20 % of frames.
        pts = resample_arc(skeleton_route(skel, ta, tb), npoint_n)
        m['_npts'] = None if pts is None else pts.astype(np.float32)
    g = float(np.hypot(ta[0]-tb[0], ta[1]-tb[1]))
    m['electrode_gap_px'] = g
    if px_per_mm: m['electrode_gap_mm'] = g/px_per_mm

    # ── tortuosity (arc-chord ratio) ──────────────────────────────────────────
    # path length along the skeleton / straight-line distance between the roots.
    if not np.isnan(m['path_len_px']) and g > 1e-6:
        m['tortuosity'] = m['path_len_px'] / g

    # ── midline asymmetry ─────────────────────────────────────────────────────
    # The split midline MUST be the predefined bisector of the two BLADE
    # electrodes (a fixed line per recording), so the A/B split is anchored on
    # the geometry rather than on the arc itself.  When blade_lines is missing we
    # do NOT silently substitute the per-frame perpendicular bisector of the two
    # arc roots (that line wobbles frame-to-frame and is not the electrode
    # bisector): the asymmetry/per-half-tortuosity metrics are left NaN instead.
    # Pass allow_tip_midline_fallback=True only for exploratory use without a
    # blade-line calibration, never for results.
    if electrode_edges is not None and len(electrode_edges) >= 4:
        # J-hook: anchor the midline on the two FIXED facing hook tips.
        midline = midline_from_edge_tips(electrode_edges[2], electrode_edges[3],
                                         arc_direction=arc_direction)
    else:
        midline = midline_from_blades(blade_lines)
    if midline is None and allow_tip_midline_fallback:
        midline = midline_from_tips(ta, tb)
    asym = arc_asymmetry(coords, ta, tb, midline, px_per_mm)
    m.update(asym)
    return m

# ── QA renderer ────────────────────────────────────────────────────────────────

def render_qa_frame(raw, m, frame_idx, fps, px_per_mm, vel_mm_s=np.nan):
    H, W = raw.shape
    gray = raw.astype(np.float32); lo,hi = gray.min(),gray.max()
    gn = (gray-lo)/(hi-lo) if hi>lo else gray
    fig, ax = plt.subplots(figsize=(6,6), facecolor='black')
    ax.set_facecolor('black')
    ax.imshow(np.power(gn,0.5), cmap='gray', vmin=0, vmax=1, origin='upper')
    if m.get('arc_detected'):
        # _binary is always present here (QA frames are measured with
        # keep_overlays=True); fall back lazily instead of recomputing eagerly.
        b = m.get('_binary')
        if b is None:
            b = binarize(raw, DEFAULT_CLOSE_RADIUS)
        ov = np.zeros((*gn.shape,4)); ov[b,0]=1; ov[b,2]=1; ov[b,3]=0.25
        ax.imshow(ov, origin='upper')
        coords = m.get('_skel_coords', np.empty((0,2)))
        if len(coords):
            ax.scatter(coords[:,1], coords[:,0], s=0.5, c='cyan', linewidths=0)
        if len(coords) >= 3:
            n = len(coords)
            for idx in range(0,n,10):
                i = min(max(idx,1),n-2)
                tang = coords[i+1]-coords[i-1]; tn = np.linalg.norm(tang)
                if tn<1e-6: continue
                tang/=tn; perp=np.array([-tang[1],tang[0]])
                r0,c0 = coords[i]
                for sg in (-1,1):
                    ax.plot([c0,c0+sg*12*perp[1]],[r0,r0+sg*12*perp[0]],
                            '-', color='orange', lw=0.6, alpha=0.7)
        ta,tb = m.get('tip_a'),m.get('tip_b')
        if ta is not None and tb is not None:
            ax.scatter([ta[1],tb[1]],[ta[0],tb[0]],s=40,c='yellow',zorder=5)
            ax.plot([ta[1],tb[1]],[ta[0],tb[0]],'--',color='yellow',lw=1)
        tr,tc = m.get('arc_tip_row',np.nan),m.get('arc_tip_col',np.nan)
        if not np.isnan(float(tr)): ax.plot(tc,tr,'x',color='red',ms=10,mew=2,zorder=6)
        cr,cc = m.get('centroid_row',np.nan),m.get('centroid_col',np.nan)
        if not np.isnan(float(cr)): ax.plot(cc,cr,'+',color='lime',ms=10,mew=2)
        if m.get('bbox'):
            r0,c0,r1,c1 = m['bbox']
            ax.add_patch(mpatches.Rectangle((c0,r0),c1-c0,r1-r0,
                edgecolor='white',facecolor='none',linestyle='--',linewidth=1))
    t_us = frame_idx/fps*1e6
    def f(v,fmt='%.2f'): return (fmt%float(v)) if not np.isnan(float(v)) else 'n/a'
    lines = ["Frame %d  t=%.1fµs"%(frame_idx,t_us),
             "detected : %s"%m.get('arc_detected',False),
             "length   : %s px  %s mm"%(f(m['arc_length_px'],'%.0f'),f(m['arc_length_mm'])),
             "width    : %s px  %s mm"%(f(m['arc_width_px'],'%.1f'),f(m['arc_width_mm'],'%.3f')),
             "gap      : %s px  %s mm"%(f(m['electrode_gap_px'],'%.1f'),f(m['electrode_gap_mm'])),
             "I_mean   : %s"%f(m['mean_intensity'],'%.3f'),
             "curvature: %s rad/px"%f(m['curvature_mean'],'%.4f'),
             "tip_vel  : %s m/s"%f(vel_mm_s/1000 if not np.isnan(float(vel_mm_s)) else np.nan,'%.1f')]
    ax.text(3,3,'\n'.join(lines),color='white',fontsize=6,va='top',ha='left',
            fontfamily='monospace',
            bbox=dict(boxstyle='square,pad=0.2',facecolor='black',alpha=0.6))
    ax.set_xlim(0,W); ax.set_ylim(H,0); ax.axis('off')
    fig.tight_layout(pad=0)
    return fig

# ── main ───────────────────────────────────────────────────────────────────────

CSV_FIELDS = ['file','frame','time_s','arc_detected','arc_complete','is_flash','area_px',
    'arc_length_px','arc_length_mm','arc_width_px','arc_width_mm',
    'mean_intensity','peak_intensity','integrated_intensity',
    'centroid_row','centroid_col','arc_tip_row','arc_tip_col','arc_tip_vel_mm_s',
    'tip_a_row','tip_a_col','tip_b_row','tip_b_col',
    'electrode_gap_px','electrode_gap_mm','curvature_mean',
    'path_len_px','tortuosity','total_curvature_rad',
    'arclen_a_px','arclen_b_px','arclen_asym',
    'tort_a','tort_b','tort_asym','mid_cross_row','mid_cross_col',
    'bbox_r0','bbox_c0','bbox_r1','bbox_c1',
    'arc_tip_col_left','arc_tip_col_right','arc_tip_col_mean']

def _f(v):
    if v is None: return ''
    try: return '' if np.isnan(float(v)) else v
    except (ValueError, TypeError): return v

def main():
    ap = argparse.ArgumentParser(description="Plasma arc TIFF analysis")
    ap.add_argument('tiff',              help="Input TIFF file")
    ap.add_argument('--out-dir',         default=None,
                    help="Output directory (default: <tiff_name>_results/)")
    ap.add_argument('--fps',             type=float, default=DEFAULT_FPS)
    ap.add_argument('--frame-start',     type=int,   default=0,
                    help='first cine frame of a chunk (with --frame-count)')
    ap.add_argument('--frame-count',     type=int,   default=None,
                    help='analyse only this many frames from --frame-start (local numbering)')
    ap.add_argument('--close-radius',    type=int,   default=DEFAULT_CLOSE_RADIUS,
                    help="Morphological closing radius in px (default 3)")
    ap.add_argument('--electrode-gap',   type=float, default=DEFAULT_ELECTRODE_GAP,
                    help="Known minimum electrode gap in mm (default 2.4)")
    ap.add_argument('--px-per-mm',       type=float, default=DEFAULT_PX_PER_MM,
                    help="Spatial scale in px/mm (default %.4f, ground-truth "
                         "22px=2.4mm hand measurement; pass 0 to auto-calibrate)"
                         % DEFAULT_PX_PER_MM)
    ap.add_argument('--dark-guard',      type=float,   default=DEFAULT_DARK_GUARD,
                    help="Skip frames whose max pixel is below this DN (default %d)" %
                         DEFAULT_DARK_GUARD)
    ap.add_argument('--background',     default=None,
                    help='.npy per-pixel background subtracted from every cine frame (cine_frames.measure_background)')
    ap.add_argument('--low-frac',       type=float,   default=None,
                    help='hysteresis low threshold is max(--low-thresh, low_frac * frame max)')
    ap.add_argument('--low-thresh',     type=float,   default=None,
                    help='hysteresis low threshold (DN); with it --bright-thresh is the seed '
                         '(high) threshold. Default None = single fixed threshold.')
    ap.add_argument('--bright-thresh',  type=float,   default=DEFAULT_BRIGHT_THRESH,
                    help="Fixed DN threshold: pixels above this are arc candidates "
                         "(default %d; replaces adaptive Otsu)" % DEFAULT_BRIGHT_THRESH)
    ap.add_argument('--min-arc-px',      type=int,   default=DEFAULT_MIN_ARC_PX,
                    help="Minimum bright-pixel AREA (px) to accept as a real arc. "
                         "Default %d px = int(2.4 mm x 9.167 px/mm), frozen at the "
                         "ADCB scale; at the 11MAY26 scale (8.8 px/mm) it is just a "
                         "speckle/noise floor, NOT a 'spans the electrode gap' test "
                         "(a 4-5 mm apex gap is ~40-44 px). Pass an explicit value to "
                         "tie it to this recording's geometry." % DEFAULT_MIN_ARC_PX)
    ap.add_argument('--max-arc-px',      type=int,   default=DEFAULT_MAX_ARC_PX,
                    help="Maximum credible arc AREA (px). A mask larger than this is "
                         "rejected as a noise flood / flash bloom. Default %d is tuned "
                         "for the thin BLADE N2 arc; the wide high-flow CO2 J arc needs "
                         "a larger value (a 27 px-wide arc spanning the frame can exceed "
                         "3000 px and be wrongly rejected) — run_j_co2.py raises it." %
                         DEFAULT_MAX_ARC_PX)
    ap.add_argument('--top-fraction',    type=float, default=DEFAULT_TOP_FRACTION,
                    help="Fraction of frame height used for electrode-tip search (default 0.25)")
    ap.add_argument('--arc-direction',   type=str,   default='down',
                    choices=['down', 'up'],
                    help="'down' = arc front is bottommost pixel (default, reactor upright). "
                         "'up' = arc front is topmost pixel (reactor inverted/upward video).")
    ap.add_argument('--blade-lines',     type=str,   default=None,
                    help="Path to blade_lines.npy (shape 2×2×2) defining the two electrode "
                         "blade lines.  When supplied, arc_complete uses perpendicular "
                         "distance to each blade line rather than the widest-pair heuristic.")
    ap.add_argument('--electrode-edges', type=str,   default=None,
                    help="Path to an .npz with arrays A,B (curved electrode-edge point "
                         "clouds, each (M,2) [row,col]) and A_tip,B_tip (facing hook tips), "
                         "built by tracking arc attachment sites across frames. When given, "
                         "arc_complete is judged by distance to the nearest EDGE POINT "
                         "(faithful for curved J-hook electrodes) and the asymmetry midline "
                         "is anchored on the hook tips. Overrides --blade-lines.")
    ap.add_argument('--elec-tol-px',     type=int,   default=ELEC_TOL_PX,
                    help="Max distance (px) from skeleton endpoint to the electrode (blade "
                         "line OR edge point-set) to count as attached (default %d)" % ELEC_TOL_PX)
    ap.add_argument('--finite-segments', action='store_true',
                    help="J-electrode geometry: treat each blade in blade_lines.npy as a "
                         "FINITE segment (pt0 = tip-end near the apex, pt1 = far-end) "
                         "rather than an infinite line.  A foot must project onto the "
                         "segment (± --segment-margin-px) to count as attached, rejecting "
                         "free tips that lie on the extrapolated line beyond the physical "
                         "electrode end.  Off by default (BLADE infinite-blade behaviour).")
    ap.add_argument('--segment-margin-px', type=float, default=6.0,
                    help="Tolerance (px) past either clicked electrode end for the "
                         "--finite-segments attachment gate (default 6).")
    ap.add_argument('--flash-arc-width', type=float, default=arc_filter.FLASH_ARC_WIDTH,
                    help="FWHM arc width (px) above which a frame is flagged is_flash at "
                         "the SOURCE. Default %.0f is the thin BLADE N2 arc; the wide "
                         "high-flow CO2 J arc is ~27 px, so raise it (e.g. 45) or the real "
                         "arc is wrongly flagged as flash and starved from event fitting." %
                         arc_filter.FLASH_ARC_WIDTH)
    ap.add_argument('--flash-eff-width', type=float, default=arc_filter.FLASH_EFF_WIDTH,
                    help="Effective width area/length (px) above which a frame is is_flash "
                         "(saturated blob). Default %.0f." % arc_filter.FLASH_EFF_WIDTH)
    ap.add_argument('--npoint-n',        type=int,   default=0,
                    help="also write npoint_xy.npz: N points equally spaced along each detected arc "
                         "(raw row,col; npoint_restrike's reconstruction, from this pass's skeleton). 0 = off")
    ap.add_argument('--no-figures',      action='store_true',
                    help="skip the per-run velocity_map.png / timeseries.png (numbers and .npy unchanged)")
    ap.add_argument('--qa-interval',     type=int,   default=DEFAULT_QA_INTERVAL,
                    help="Save a QA frame every N frames (default 500)")
    ap.add_argument('--gap-row',         type=int,   default=None,
                    help="Electrode-gap row (px). Frames whose arc mask crosses past "
                         "this row into the electrode/flash zone are rejected as flashes "
                         "(arc_detected=0). For --arc-direction down the discard side is "
                         "rows ABOVE the line (row < gap_row); for up it is rows BELOW "
                         "(row > gap_row). Default None = disabled.")
    args = ap.parse_args()

    tiff_path  = args.tiff
    stem       = os.path.splitext(os.path.basename(tiff_path))[0]
    out_dir    = args.out_dir or (stem + "_results")
    qa_dir     = os.path.join(out_dir, "qa_frames")
    os.makedirs(qa_dir, exist_ok=True)

    fps        = args.fps
    close_r    = args.close_radius
    top_frac   = args.top_fraction
    qa_int     = args.qa_interval
    dark_guard    = args.dark_guard
    bright_thresh = args.bright_thresh
    min_arc_px    = args.min_arc_px
    max_arc_px    = args.max_arc_px
    elec_tol_px   = args.elec_tol_px
    arc_direction = args.arc_direction

    # ── load blade lines ─────────────────────────────────────────────────────
    blade_lines = None
    if args.blade_lines:
        bl_path = args.blade_lines
        if os.path.exists(bl_path):
            blade_lines = np.load(bl_path)
            print("Blade lines : %s  (shape %s)" % (bl_path, blade_lines.shape))
            print("  A: (%.0f,%.0f) → (%.0f,%.0f)" % (
                  blade_lines[0,0,0], blade_lines[0,0,1],
                  blade_lines[0,1,0], blade_lines[0,1,1]))
            print("  B: (%.0f,%.0f) → (%.0f,%.0f)" % (
                  blade_lines[1,0,0], blade_lines[1,0,1],
                  blade_lines[1,1,0], blade_lines[1,1,1]))
            print("  Attachment tolerance: %d px" % elec_tol_px)
        else:
            print("WARNING: --blade-lines path not found: %s" % bl_path)
            print("         Falling back to widest-pair electrode heuristic.")

    # ── load curved electrode edges (J-hook geometry; overrides blade_lines) ──
    electrode_edges = None
    if args.electrode_edges:
        if os.path.exists(args.electrode_edges):
            _z = np.load(args.electrode_edges)
            electrode_edges = (np.asarray(_z['A'], float), np.asarray(_z['B'], float),
                               np.asarray(_z['A_tip'], float), np.asarray(_z['B_tip'], float))
            print("Electrode edges : %s" % args.electrode_edges)
            print("  A edge pts: %d   B edge pts: %d" % (len(electrode_edges[0]), len(electrode_edges[1])))
            print("  hook tips A=(%.0f,%.0f) B=(%.0f,%.0f)  gap=%.1f px" % (
                  electrode_edges[2][0], electrode_edges[2][1],
                  electrode_edges[3][0], electrode_edges[3][1],
                  float(np.hypot(*(electrode_edges[2]-electrode_edges[3])))))
            print("  Attachment: nearest-edge-point ≤ %d px (curved J-hook test)" % elec_tol_px)
        else:
            print("WARNING: --electrode-edges path not found: %s" % args.electrode_edges)

    print("=" * 60)
    print("Input  : %s" % tiff_path)
    print("Output : %s" % out_dir)

    tif = open_frames(tiff_path, background=args.background)   # .cine: 12-bit, defect-corrected, header frame count
    if hasattr(tif, 'fps') and abs(tif.fps - fps) > 0.5:
        raise SystemExit('ERROR: --fps %.0f disagrees with the cine header %.0f' % (fps, tif.fps))
    if args.frame_count is not None:
        # a chunk of a cine: local frame i = cine frame frame_start + i, exactly as
        # a TIFF chunk would be, so run_*.merge_csvs offsets it back to global
        tif = FrameWindow(tif, args.frame_start, args.frame_count)
        print("Window : cine frames [%d, %d)" % (tif.start, tif.start + tif.n))
    if True:
        n_frames = tif.n
        H, W     = tif.shape
        if getattr(tif, 'defects', None) is not None:
            print("Defects: %d stuck px replaced by neighbour median (values %s)" % (
                  tif.defects.n, tif.defects.values))
        print("Frames : %d  |  Resolution : %d×%d  |  FPS : %.0f" % (
              n_frames, H, W, fps))

        # calibration
        if args.px_per_mm and args.px_per_mm > 0:
            px_per_mm = args.px_per_mm
            print("Scale  : %.4f px/mm (%.1f µm/px) [ground-truth]" % (
                  px_per_mm, 1000/px_per_mm))
        else:
            px_per_mm = calibrate_px_per_mm(
                tif, n_frames, args.electrode_gap,
                close_r, top_frac)

        print("Threshold : fixed DN>%g  |  dark-guard DN<%g  |  arc area %d–%d px" % (
              bright_thresh, dark_guard, min_arc_px, max_arc_px))
        print("Direction : arc front is %s" % (
              "BOTTOMMOST pixel (reactor upright)" if arc_direction == 'down'
              else "TOPMOST pixel (reactor inverted)"))

        # ── pass 1: measure every frame ──────────────────────────────
        print("\nMeasuring all %d frames..." % n_frames)
        tip_rows = np.full(n_frames, np.nan)
        tip_cols = np.full(n_frames, np.nan)
        meas_all = [None] * n_frames
        qa_store = {}
        np_frames, np_pts = [], []

        t0 = time.time()
        for i in range(n_frames):
            raw = tif.frame(i)
            keep = (i % qa_int == 0)
            m    = measure_frame(raw, close_r, px_per_mm,
                                  top_frac, keep_overlays=keep,
                                  dark_guard=dark_guard,
                                  bright_thresh=bright_thresh,
                                  min_arc_px=min_arc_px,
                                  max_arc_px=max_arc_px,
                                  blade_lines=blade_lines,
                                  elec_tol_px=elec_tol_px,
                                  arc_direction=arc_direction,
                                  gap_row=args.gap_row,
                                  finite_segments=args.finite_segments,
                                  seg_margin_px=args.segment_margin_px,
                                  electrode_edges=electrode_edges,
                                  flash_eff_width=args.flash_eff_width,
                                  flash_arc_width=args.flash_arc_width,
                                  low_thresh=args.low_thresh, low_frac=args.low_frac,
                                  npoint_n=args.npoint_n)
            if args.npoint_n and m['arc_detected'] and m.get('_npts') is not None:
                np_frames.append(i)
                np_pts.append(m.pop('_npts'))
            meas_all[i] = m
            if m['arc_detected']:
                tip_rows[i] = m['arc_tip_row']
                tip_cols[i] = m['arc_tip_col']
            if keep:
                qa_store[i] = raw.copy()
            if (i+1) % 5000 == 0:
                rate = (i+1)/(time.time()-t0)
                print("  [%d/%d]  %.0f fr/s  ETA %.1f min" % (
                    i+1, n_frames, rate, (n_frames-i-1)/rate/60))
                sys.stdout.flush()

        if args.npoint_n:
            # frames are LOCAL to this window (as the CSV's 'frame' column); raw (row, col), float32
            np.savez(os.path.join(out_dir, "npoint_xy.npz"), n=args.npoint_n, H=H,
                     frames=np.asarray(np_frames, np.int64),
                     pts=(np.stack(np_pts) if np_pts else np.zeros((0, args.npoint_n, 2), np.float32)))

        # ── arc-tip velocity (central difference) ────────────────────
        tip_vels = np.full(n_frames, np.nan)  # mm/s
        for i in range(1, n_frames-1):
            rp,cp,rn,cn = tip_rows[i-1],tip_cols[i-1],tip_rows[i+1],tip_cols[i+1]
            if any(np.isnan([rp,cp,rn,cn])): continue
            tip_vels[i] = (np.hypot(rn-rp,cn-cp)/px_per_mm)/(2.0/fps)

        # ── write CSV ─────────────────────────────────────────────────
        csv_path = os.path.join(out_dir, "measurements.csv")
        with open(csv_path, 'w', newline='') as cf:
            w = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
            w.writeheader()
            for i, m in enumerate(meas_all):
                ta,tb,bb = m['tip_a'],m['tip_b'],m.get('bbox')
                w.writerow({'file':stem,'frame':i,'time_s':i/fps,
                    'arc_detected':int(m['arc_detected']),
                    'arc_complete':int(m.get('arc_complete', False)),
                    'is_flash':int(m.get('is_flash', False)),
                    'area_px':_f(m['area_px']),
                    'arc_length_px':_f(m['arc_length_px']),'arc_length_mm':_f(m['arc_length_mm']),
                    'arc_width_px':_f(m['arc_width_px']),'arc_width_mm':_f(m['arc_width_mm']),
                    'mean_intensity':_f(m['mean_intensity']),'peak_intensity':_f(m['peak_intensity']),
                    'integrated_intensity':_f(m['integrated_intensity']),
                    'centroid_row':_f(m['centroid_row']),'centroid_col':_f(m['centroid_col']),
                    'arc_tip_row':_f(m['arc_tip_row']),'arc_tip_col':_f(m['arc_tip_col']),
                    'arc_tip_vel_mm_s':_f(tip_vels[i]),
                    'tip_a_row':ta[0] if ta is not None else '',
                    'tip_a_col':ta[1] if ta is not None else '',
                    'tip_b_row':tb[0] if tb is not None else '',
                    'tip_b_col':tb[1] if tb is not None else '',
                    'electrode_gap_px':_f(m['electrode_gap_px']),
                    'electrode_gap_mm':_f(m['electrode_gap_mm']),
                    'curvature_mean':_f(m['curvature_mean']),
                    'path_len_px':_f(m['path_len_px']),
                    'tortuosity':_f(m['tortuosity']),
                    'total_curvature_rad':_f(m['total_curvature_rad']),
                    'arclen_a_px':_f(m['arclen_a_px']),'arclen_b_px':_f(m['arclen_b_px']),
                    'arclen_asym':_f(m['arclen_asym']),
                    'tort_a':_f(m['tort_a']),'tort_b':_f(m['tort_b']),
                    'tort_asym':_f(m['tort_asym']),
                    'mid_cross_row':_f(m['mid_cross_row']),'mid_cross_col':_f(m['mid_cross_col']),
                    'bbox_r0':bb[0] if bb else '','bbox_c0':bb[1] if bb else '',
                    'bbox_r1':bb[2] if bb else '','bbox_c1':bb[3] if bb else '',
                    'arc_tip_col_left':_f(m['arc_tip_col_left']),
                    'arc_tip_col_right':_f(m['arc_tip_col_right']),
                    'arc_tip_col_mean':_f(m['arc_tip_col_mean'])})
        print("CSV: %s" % csv_path)

        # ── save QA frames ────────────────────────────────────────────
        print("Saving %d QA frames..." % len(qa_store))
        for fi, raw_qa in sorted(qa_store.items()):
            m_qa = measure_frame(raw_qa, close_r, px_per_mm,
                                  top_frac, keep_overlays=True,
                                  dark_guard=dark_guard,
                                  bright_thresh=bright_thresh,
                                  min_arc_px=min_arc_px,
                                  max_arc_px=max_arc_px,
                                  blade_lines=blade_lines,
                                  elec_tol_px=elec_tol_px,
                                  arc_direction=arc_direction,
                                  gap_row=args.gap_row,
                                  finite_segments=args.finite_segments,
                                  seg_margin_px=args.segment_margin_px,
                                  electrode_edges=electrode_edges,
                                  flash_eff_width=args.flash_eff_width,
                                  flash_arc_width=args.flash_arc_width,
                                  low_thresh=args.low_thresh, low_frac=args.low_frac)
            fig  = render_qa_frame(raw_qa, m_qa, fi, fps, px_per_mm, tip_vels[fi])
            fig.savefig(os.path.join(qa_dir,"qa_%06d.png"%fi),
                        dpi=120, bbox_inches='tight', facecolor='black')
            plt.close(fig)

        # ── 2-D velocity map (built in the single measurement pass) ───
        # Reuse the arc-pixel coordinates stored by measure_frame instead of
        # re-reading and re-binarizing every frame. Only frames with a valid
        # central-difference tip velocity (complete arcs) contribute, which is
        # identical behaviour to the previous second-pass implementation.
        print("Building 2-D velocity map (single-pass)...")
        vel_sum = np.zeros((H,W), np.float64)
        vel_cnt = np.zeros((H,W), np.float64)
        for i in range(n_frames):
            v = tip_vels[i]
            if np.isnan(v): continue
            rc = meas_all[i].get('_arc_rc')
            if rc is None: continue
            rr, cc = rc
            vel_sum[rr, cc] += v; vel_cnt[rr, cc] += 1
        del meas_all  # free memory now that the velocity map is built

        vel_map = np.full((H,W), np.nan)
        mask = vel_cnt > 0
        vel_map[mask] = vel_sum[mask]/vel_cnt[mask]
        np.save(os.path.join(out_dir,"velocity_map.npy"), vel_map)

    if n_frames > 0 and not args.no_figures:
        fig, ax = plt.subplots(figsize=(7,7))
        vmax = np.nanpercentile(vel_map, 98)
        im = ax.imshow(vel_map/1000, cmap='plasma', vmin=0,
                       vmax=max(vmax/1000,0.01), origin='upper',
                       extent=[0,W/px_per_mm,H/px_per_mm,0])
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Mean arc-tip velocity (m/s)")
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir,"velocity_map.png"), dpi=150)
        plt.close(fig)

        # ── time-series plot ──────────────────────────────────────────
        import pandas as pd
        df     = pd.read_csv(csv_path)
        df_det = df[df['arc_detected']==1]
        t_ms   = df_det['time_s']*1000
        fig, axes = plt.subplots(5,1,figsize=(14,12),sharex=True)
        for ax,(col,lbl,clr) in zip(axes,[
            ('arc_length_mm','Arc length (mm)','steelblue'),
            ('electrode_gap_mm','Electrode gap (mm)','orange'),
            ('arc_tip_vel_mm_s','Tip velocity (mm/s)','tomato'),
            ('mean_intensity','Mean intensity','gold'),
            ('curvature_mean','Curvature (rad/px)','orchid')]):
            if col in df_det.columns:
                ax.plot(t_ms, df_det[col], lw=0.5, color=clr, alpha=0.85)
            ax.set_ylabel(lbl, fontsize=8); ax.set_facecolor('#111111')
            for sp in ax.spines.values(): sp.set_color('#444')
            ax.tick_params(colors='#aaa',labelsize=7); ax.yaxis.label.set_color('#ccc')
        axes[-1].set_xlabel("Time (ms)")
        fig.patch.set_facecolor('#1a1a1a'); fig.tight_layout()
        fig.savefig(os.path.join(out_dir,"timeseries.png"), dpi=150)
        plt.close(fig)

    elapsed = (time.time()-t0)/60
    print("\nDone in %.1f min. Results in: %s" % (elapsed, out_dir))

if __name__ == '__main__':
    main()
