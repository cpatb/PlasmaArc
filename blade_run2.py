#!/usr/bin/env python3
"""
Plasma arc analysis — chunked, checkpointed.

Usage:
  python blade_run2.py calibrate  <file_idx>
  python blade_run2.py measure    <file_idx> <chunk>
  python blade_run2.py velocity   <file_idx>
  python blade_run2.py velmap     <file_idx> <chunk>
  python blade_run2.py finalize   <file_idx>
  python blade_run2.py combine
"""
import sys, os, time, csv, datetime
import numpy as np
import tifffile

from plasma_utils.image import normalize_frame, binarize, measure_frame as _pm_measure

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_home     = os.path.expanduser('~')
BLADE_DIR = os.path.join(_home, 'OneDrive', 'Desktop', 'BLADE')
# run_pipeline.py sets BLADE_OUT_BASE so all subprocesses share the same
# output directory regardless of what time/date they happen to start.
OUT_BASE  = (os.environ.get('BLADE_OUT_BASE')
             or os.path.join(BLADE_DIR, datetime.date.today().isoformat()))
STATE_DIR = os.path.join(BLADE_DIR, 'pipeline_state')

os.makedirs(STATE_DIR, exist_ok=True)
os.makedirs(OUT_BASE, exist_ok=True)

TIFF_FILES = [
    os.path.join(BLADE_DIR, 'ADCB_0.8mmNoz_1000mL_N2_1.tiff'),
    os.path.join(BLADE_DIR, 'ADCB_0.8mmNoz_1000mL_N2_2.tiff'),
    os.path.join(BLADE_DIR, 'ADCB_0.8mmNoz_1000mL_N2_3.tiff'),
]
STEMS = [
    'ADCB_0.8mmNoz_1000mL_N2_1',
    'ADCB_0.8mmNoz_1000mL_N2_2',
    'ADCB_0.8mmNoz_1000mL_N2_3',
]

# ── constants ─────────────────────────────────────────────────────────────────
FPS        = 200_000
PX_PER_MM  = 9.167
DARK_GUARD = 35      # frames with raw max below this DN are skipped as dark
MIN_ARC_PX = 5
MAX_ARC_PX = 3000
CLOSE_R      = 3
CHUNK_SIZE   = 4000
# Re-stroke events cause the arc tip to teleport from the bottom to the top
# of the frame, producing ~5000 m/s velocity spikes.  Any computed value above
# this physical cap is treated as an artefact and set to NaN.
MAX_VEL_MM_S = 500_000   # 500 m/s expressed in mm/s

# ── ROI ───────────────────────────────────────────────────────────────────────
_ROI_PATH = os.path.join(BLADE_DIR, 'roi_polygon.npy')
_roi_mask = None   # lazy-loaded on first frame


def _get_roi_mask(shape):
    global _roi_mask
    if _roi_mask is None and os.path.exists(_ROI_PATH):
        from plasma_utils.roi import load_roi_mask
        _roi_mask = load_roi_mask(_ROI_PATH, shape)
        if _roi_mask is not None:
            print(f'  [ROI] mask loaded: {_roi_mask.sum()} / {_roi_mask.size} px active')
    return _roi_mask


# ── blade lines ───────────────────────────────────────────────────────────────
# Set with:  python calibrate_electrodes.py
# Shape (2, 2, 2): [blade_A, blade_B] × [point_0, point_1] × [row, col]
# Each blade is a razor edge — the arc can attach at any point along its
# length, so arc_complete is based on perpendicular distance to the blade LINE,
# not proximity to a fixed tip position.
_BLADE_PATH  = os.path.join(BLADE_DIR, 'blade_lines.npy')
_blade_lines = None    # lazy-loaded; None if file absent
ELEC_TOL_PX  = 4      # px — max perpendicular distance from a blade LINE for a
                       # skeleton endpoint to count as "attached" to that blade.
                       # A truly contacting endpoint should land within ~2-4 px of
                       # the blade edge after skeletonisation (binarisation fringing
                       # + discrete-grid localisation noise).  Well below the 11 px
                       # half-gap so the two blade bands never overlap.


def _get_blade_lines():
    """Return blade-line array (2,2,2) or None if not calibrated."""
    global _blade_lines
    if _blade_lines is None and os.path.exists(_BLADE_PATH):
        _blade_lines = np.load(_BLADE_PATH)
        print(f'  [BLADE] lines loaded:'
              f' A=({_blade_lines[0,0,0]:.0f},{_blade_lines[0,0,1]:.0f})'
              f'→({_blade_lines[0,1,0]:.0f},{_blade_lines[0,1,1]:.0f})'
              f'  B=({_blade_lines[1,0,0]:.0f},{_blade_lines[1,0,1]:.0f})'
              f'→({_blade_lines[1,1,0]:.0f},{_blade_lines[1,1,1]:.0f})'
              f'  tol={ELEC_TOL_PX} px')
    return _blade_lines


def _dist_to_line(pr, pc, blade):
    """Perpendicular distance from (pr, pc) to the infinite line through
    blade[0] and blade[1], where blade has shape (2, 2) = [[r0,c0],[r1,c1]].

    Using the infinite line (not clamped to the segment) because the arc can
    attach anywhere along a blade of unknown length, and we only care which
    blade the endpoint is closest to.
    """
    r1, c1 = blade[0]
    r2, c2 = blade[1]
    dr, dc = r2 - r1, c2 - c1
    length = np.hypot(dr, dc)
    if length < 1e-9:
        return np.hypot(pr - r1, pc - c1)
    # |dc*(pr-r1) - dr*(pc-c1)| / length  (cross-product formula)
    return abs(dc * (pr - r1) - dr * (pc - c1)) / length


# ── image processing ──────────────────────────────────────────────────────────

def _binarize_raw(raw: np.ndarray) -> np.ndarray:
    """Dark-frame guard + normalize + binarize + optional ROI mask."""
    if int(raw.max()) < DARK_GUARD:
        return np.zeros(raw.shape, dtype=bool)
    mask = binarize(normalize_frame(raw), close_r=CLOSE_R, open_r=0)
    roi = _get_roi_mask(raw.shape)
    if roi is not None:
        mask = mask & roi
    area = int(mask.sum())
    if area < MIN_ARC_PX or area > MAX_ARC_PX:
        return np.zeros(raw.shape, dtype=bool)
    return mask


_EMPTY_DICT = dict(
    arc_detected=False, arc_complete=False, area_px=0, arc_length_px=0,
    arc_length_mm=float('nan'), arc_width_px=float('nan'), arc_width_mm=float('nan'),
    mean_intensity=float('nan'), peak_intensity=float('nan'),
    integrated_intensity=float('nan'),
    centroid_row=float('nan'), centroid_col=float('nan'),
    arc_tip_row=float('nan'), arc_tip_col=float('nan'), arc_tip_vel_mm_s=float('nan'),
    tip_a_row=float('nan'), tip_a_col=float('nan'), tip_a_attached=False,
    tip_b_row=float('nan'), tip_b_col=float('nan'), tip_b_attached=False,
    electrode_gap_px=float('nan'), electrode_gap_mm=float('nan'),
    curvature_mean=float('nan'),
    bbox_r0=float('nan'), bbox_c0=float('nan'),
    bbox_r1=float('nan'), bbox_c1=float('nan'),
)


def measure_frame(raw: np.ndarray) -> dict:
    """Measure arc properties for one raw frame; return a CSV-schema dict."""
    if int(raw.max()) < DARK_GUARD:
        return dict(_EMPTY_DICT)
    # Normalize once and reuse for both binarization and measurement.
    gray   = normalize_frame(raw)
    binary = binarize(gray, close_r=CLOSE_R, open_r=0)
    roi    = _get_roi_mask(raw.shape)
    if roi is not None:
        binary = binary & roi
    area = int(binary.sum())
    if area < MIN_ARC_PX or area > MAX_ARC_PX or not binary.any():
        return dict(_EMPTY_DICT)
    m = _pm_measure(gray, binary, px_per_mm=PX_PER_MM, width_sample_step=10)
    # bottommost arc pixel — drives kymograph depth in visualize_blade
    ri, ci  = np.where(binary)
    bot     = ri.argmax()
    # Complete (bridging) arc: endpoints are separated more horizontally than
    # vertically — both tips sit at electrode height on opposite sides of the gap.
    # Partial arc (leader/streamer): one tip is floating below → vertical
    # separation dominates. Works for any gap size or electrode V-angle.
    _ta_r, _ta_c = float(m.tip_a_row), float(m.tip_a_col)
    _tb_r, _tb_c = float(m.tip_b_row), float(m.tip_b_col)
    _tips_valid = (m.arc_detected
                   and not (np.isnan(_ta_r) or np.isnan(_ta_c)
                            or np.isnan(_tb_r) or np.isnan(_tb_c)))

    blades = _get_blade_lines()
    tip_a_attached = False   # endpoint on blade surface (within ELEC_TOL_PX)
    tip_b_attached = False
    if blades is not None and _tips_valid:
        # Blade-line criterion: each skeleton endpoint must lie within
        # ELEC_TOL_PX perpendicular distance of a blade edge.  Each electrode
        # is a razor blade of finite length, so the contact point slides along
        # the blade as the arc moves — perpendicular line distance is the
        # correct metric, not point proximity to a fixed tip location.
        # Optimal 2-tip → 2-blade assignment (try both permutations):
        d_aa = _dist_to_line(_ta_r, _ta_c, blades[0])   # tip_a ↔ blade A
        d_ab = _dist_to_line(_ta_r, _ta_c, blades[1])   # tip_a ↔ blade B
        d_ba = _dist_to_line(_tb_r, _tb_c, blades[0])   # tip_b ↔ blade A
        d_bb = _dist_to_line(_tb_r, _tb_c, blades[1])   # tip_b ↔ blade B
        if d_aa + d_bb <= d_ab + d_ba:   # best assignment: a→A, b→B
            tip_a_attached = d_aa < ELEC_TOL_PX
            tip_b_attached = d_bb < ELEC_TOL_PX
        else:                            # best assignment: a→B, b→A
            tip_a_attached = d_ab < ELEC_TOL_PX
            tip_b_attached = d_ba < ELEC_TOL_PX
        arc_complete = tip_a_attached and tip_b_attached
    else:
        # Fallback heuristic — no calibration file present.
        # Only reliable for horizontally symmetric electrode layouts.
        arc_complete = _tips_valid and abs(_tb_c - _ta_c) > abs(_tb_r - _ta_r)
        # Can't determine individual attachment without blade calibration
        tip_a_attached = arc_complete
        tip_b_attached = arc_complete
    return dict(
        arc_detected=m.arc_detected, arc_complete=arc_complete, area_px=m.area_px,
        arc_length_px=m.arc_length_px, arc_length_mm=m.arc_length_mm,
        arc_width_px=m.arc_width_px_mean, arc_width_mm=m.arc_width_mm_mean,
        mean_intensity=m.mean_intensity, peak_intensity=m.peak_intensity,
        integrated_intensity=m.integrated_intensity,
        centroid_row=m.centroid_row, centroid_col=m.centroid_col,
        arc_tip_row=float(ri[bot]), arc_tip_col=float(ci[bot]),
        arc_tip_vel_mm_s=float('nan'),
        tip_a_row=m.tip_a_row, tip_a_col=m.tip_a_col, tip_a_attached=int(tip_a_attached),
        tip_b_row=m.tip_b_row, tip_b_col=m.tip_b_col, tip_b_attached=int(tip_b_attached),
        electrode_gap_px=m.electrode_gap_px, electrode_gap_mm=m.electrode_gap_mm,
        curvature_mean=m.curvature_mean,
        bbox_r0=float(m.bbox_row_min), bbox_c0=float(m.bbox_col_min),
        bbox_r1=float(m.bbox_row_max), bbox_c1=float(m.bbox_col_max),
    )


# ── CSV schema ─────────────────────────────────────────────────────────────────
CSV_FIELDS = ['frame', 'time_s', 'arc_detected', 'arc_complete', 'area_px',
    'arc_length_px', 'arc_length_mm', 'arc_width_px', 'arc_width_mm',
    'mean_intensity', 'peak_intensity', 'integrated_intensity',
    'centroid_row', 'centroid_col', 'arc_tip_row', 'arc_tip_col', 'arc_tip_vel_mm_s',
    'tip_a_row', 'tip_a_col', 'tip_a_attached',
    'tip_b_row', 'tip_b_col', 'tip_b_attached',
    'electrode_gap_px', 'electrode_gap_mm', 'curvature_mean',
    'bbox_r0', 'bbox_c0', 'bbox_r1', 'bbox_c1']

def _f(v):
    if v is None: return ''
    try: return '' if (isinstance(v, float) and np.isnan(v)) else v
    except (ValueError, TypeError): return v


# ── phases ─────────────────────────────────────────────────────────────────────
def phase_calibrate(fi):
    print('Calibration: %.4f px/mm  (ground-truth 22 px = 2.4 mm)' % PX_PER_MM)
    np.save(os.path.join(STATE_DIR, 'f%d_px_per_mm.npy' % fi), np.array([PX_PER_MM]))
    print('NEXT: python blade_run2.py measure %d 0' % fi)


def phase_measure(fi, chunk):
    path  = TIFF_FILES[fi]
    start = chunk * CHUNK_SIZE
    t0    = time.time()
    with tifffile.TiffFile(path) as tif:
        n_frames = len(tif.pages)
        end  = min(start + CHUNK_SIZE, n_frames)
        H, W = tif.pages[0].asarray().shape[:2]
        tip_r = (np.load(os.path.join(STATE_DIR, 'f%d_tip_rows.npy' % fi))
                 if chunk > 0 else np.full(n_frames, np.nan))
        tip_c = (np.load(os.path.join(STATE_DIR, 'f%d_tip_cols.npy' % fi))
                 if chunk > 0 else np.full(n_frames, np.nan))
        # col_front[i, c] = bottommost arc pixel row in column c at frame i,
        # NaN if no arc in that column.  Used for per-column front velocity.
        col_front_f = os.path.join(STATE_DIR, 'f%d_col_front.npy' % fi)
        col_front = (np.load(col_front_f)
                     if chunk > 0 and os.path.exists(col_front_f)
                     else np.full((n_frames, W), np.nan))
        _rows = np.arange(H, dtype=float)[:, np.newaxis]  # (H,1) broadcast helper
        csv_chunk = os.path.join(STATE_DIR, 'f%d_meas_chunk%d.csv' % (fi, chunk))
        with open(csv_chunk, 'w', newline='') as cf:
            w = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
            if chunk == 0: w.writeheader()
            for i in range(start, end):
                raw = tif.pages[i].asarray()
                m   = measure_frame(raw)
                tip_r[i] = m['arc_tip_row'] if m['arc_detected'] else np.nan
                tip_c[i] = m['arc_tip_col'] if m['arc_detected'] else np.nan
                # Per-column front: bottommost arc pixel row per column.
                # _binarize_raw duplicates work done inside measure_frame, but
                # measure_frame does not expose the binary mask, and the extra
                # binarize call is cheap compared to TIFF page I/O.
                if m['arc_detected']:
                    binary = _binarize_raw(raw)
                    if binary.any():
                        col_any = binary.any(axis=0)               # (W,) bool
                        masked  = np.where(binary, _rows, -1.0)    # (H,W)
                        col_max = masked.max(axis=0)               # (W,) max row
                        col_front[i, :] = np.where(col_any, col_max, np.nan)
                w.writerow({'frame': i, 'time_s': i / FPS,
                    'arc_detected': int(m['arc_detected']),
                    **{k: _f(m[k]) for k in CSV_FIELDS
                       if k not in ('frame', 'time_s', 'arc_detected', 'arc_tip_vel_mm_s')}})
                if (i - start + 1) % 2000 == 0:
                    rate = (i - start + 1) / (time.time() - t0)
                    print('  [%d/%d]  %.0f fr/s' % (i + 1, n_frames, rate))
                    sys.stdout.flush()
        np.save(os.path.join(STATE_DIR, 'f%d_tip_rows.npy' % fi), tip_r)
        np.save(os.path.join(STATE_DIR, 'f%d_tip_cols.npy' % fi), tip_c)
        np.save(col_front_f, col_front)
    n_chunks = (n_frames + CHUNK_SIZE - 1) // CHUNK_SIZE
    nxt = chunk + 1
    if nxt < n_chunks:
        print('NEXT: python blade_run2.py measure %d %d' % (fi, nxt))
    else:
        print('NEXT: python blade_run2.py velocity %d' % fi)


def phase_velocity(fi):
    with tifffile.TiffFile(TIFF_FILES[fi]) as tif:
        n_frames = len(tif.pages)
    tip_r   = np.load(os.path.join(STATE_DIR, 'f%d_tip_rows.npy' % fi))
    tip_c   = np.load(os.path.join(STATE_DIR, 'f%d_tip_cols.npy' % fi))
    tip_vel = np.full(n_frames, np.nan)

    # Gap-spanning velocity.
    #
    # For each detected frame i, use the nearest detected frames
    # j (immediately before i) and k (immediately after i) regardless of
    # how many blank frames lie between them.  Blank frames represent
    # unseen continuous arc motion -- not zero displacement -- so the
    # time interval is (k-j) frames, not 2 frames.
    #
    # Formula:
    #   v[i] = hypot(tip_r[k]-tip_r[j], tip_c[k]-tip_c[j])
    #          / (k-j) / PX_PER_MM * FPS          [mm/s]
    #
    # Positivity constraint:
    #   The arc front propagates downward (increasing row index).
    #   If tip_r[k] < tip_r[j] the arc jumped back toward the top --
    #   this is a re-strike (new cycle), not backward propagation.
    #   Such frames are set NaN.
    #
    # Safety cap: any velocity > MAX_VEL_MM_S is still an artefact -> NaN.
    det      = np.where(~np.isnan(tip_r))[0]   # sorted detected-frame indices
    n_upward = 0
    n_capped = 0
    if len(det) >= 3:
        for idx in range(1, len(det) - 1):
            i = det[idx]
            j = det[idx - 1]
            k = det[idx + 1]
            dr = tip_r[k] - tip_r[j]   # positive = downward = forward
            if dr < 0:                  # re-strike: tip jumped back to top
                n_upward += 1
                continue
            dc  = tip_c[k] - tip_c[j]
            vel = np.hypot(dr, dc) / (k - j) / PX_PER_MM * FPS   # mm/s
            if vel > MAX_VEL_MM_S:
                n_capped += 1
                continue
            tip_vel[i] = vel

    np.save(os.path.join(STATE_DIR, 'f%d_tip_vels.npy' % fi), tip_vel)
    det_v = ~np.isnan(tip_vel)
    if det_v.any():
        v_ms = tip_vel[det_v] / 1000.0
        print('Velocity: %d valid | mean %.0f m/s | 95th %.0f m/s | max %.0f m/s' %
              (det_v.sum(), np.mean(v_ms), np.percentile(v_ms, 95), v_ms.max()))
    else:
        print('Velocity: no valid frames.')
    if n_upward:
        print('  %d frames skipped: upward tip motion (re-strike)' % n_upward)
    if n_capped:
        print('  %d frames capped: >%.0f m/s' % (n_capped, MAX_VEL_MM_S / 1000.0))

    # Per-column front velocity.
    #
    # col_front[i, c] = bottommost arc pixel row in column c at frame i.
    # The same gap-spanning + positivity + cap logic is applied per column.
    # Result: col_vels[i, c] in mm/s, saved as f{fi}_col_vels.npy.
    col_front_f = os.path.join(STATE_DIR, 'f%d_col_front.npy' % fi)
    if os.path.exists(col_front_f):
        col_front = np.load(col_front_f)        # (n_frames, n_cols)
        n_cols    = col_front.shape[1]
        col_vels  = np.full((n_frames, n_cols), np.nan)
        n_col_upward = 0
        n_col_capped = 0
        for c in range(n_cols):
            cdet = np.where(~np.isnan(col_front[:, c]))[0]
            if len(cdet) < 3:
                continue
            for idx in range(1, len(cdet) - 1):
                i = cdet[idx]
                j = cdet[idx - 1]
                k = cdet[idx + 1]
                dr = col_front[k, c] - col_front[j, c]
                if dr < 0:
                    n_col_upward += 1
                    continue
                vel = dr / (k - j) / PX_PER_MM * FPS   # mm/s (dr >= 0)
                if vel > MAX_VEL_MM_S:
                    n_col_capped += 1
                    continue
                col_vels[i, c] = vel
        np.save(os.path.join(STATE_DIR, 'f%d_col_vels.npy' % fi), col_vels)
        n_valid_col = int((~np.isnan(col_vels)).sum())
        print('Col vels: %d valid (frame,col) pairs | %d upward | %d capped' %
              (n_valid_col, n_col_upward, n_col_capped))
    else:
        print('Col front not found; skipping per-column velocity.')
    print('NEXT: python blade_run2.py velmap %d 0' % fi)


def phase_velmap(fi, chunk):
    path  = TIFF_FILES[fi]
    start = chunk * CHUNK_SIZE
    t0    = time.time()
    with tifffile.TiffFile(path) as tif:
        n_frames = len(tif.pages)
        end     = min(start + CHUNK_SIZE, n_frames)
        tip_vel = np.load(os.path.join(STATE_DIR, 'f%d_tip_vels.npy' % fi))
        vsum_f  = os.path.join(STATE_DIR, 'f%d_velmap_sum.npy' % fi)
        vcnt_f  = os.path.join(STATE_DIR, 'f%d_velmap_cnt.npy' % fi)
        H, W    = tif.pages[0].asarray().shape
        vsum    = np.load(vsum_f) if os.path.exists(vsum_f) else np.zeros((H, W))
        vcnt    = np.load(vcnt_f) if os.path.exists(vcnt_f) else np.zeros((H, W))
        for i in range(start, end):
            if np.isnan(tip_vel[i]): continue
            raw  = tif.pages[i].asarray()
            mask = _binarize_raw(raw)
            if not mask.any(): continue
            vsum[mask] += tip_vel[i]
            vcnt[mask] += 1
            if (i - start + 1) % 2000 == 0:
                rate = (i - start + 1) / (time.time() - t0)
                print('  velmap [%d/%d]  %.0f fr/s' % (i + 1, n_frames, rate))
                sys.stdout.flush()
        np.save(vsum_f, vsum); np.save(vcnt_f, vcnt)
    n_chunks = (n_frames + CHUNK_SIZE - 1) // CHUNK_SIZE
    nxt = chunk + 1
    if nxt < n_chunks:
        print('NEXT: python blade_run2.py velmap %d %d' % (fi, nxt))
    else:
        print('NEXT: python blade_run2.py finalize %d' % fi)


def phase_finalize(fi):
    stem    = STEMS[fi]
    out_dir = os.path.join(OUT_BASE, stem + '_results')
    os.makedirs(out_dir, exist_ok=True)

    with tifffile.TiffFile(TIFF_FILES[fi]) as tif:
        n_frames = len(tif.pages)

    tip_vel = np.load(os.path.join(STATE_DIR, 'f%d_tip_vels.npy' % fi))

    csv_out  = os.path.join(out_dir, 'measurements.csv')
    n_chunks = (n_frames + CHUNK_SIZE - 1) // CHUNK_SIZE
    with open(csv_out, 'w', newline='') as cf:
        w = csv.DictWriter(cf, fieldnames=CSV_FIELDS)
        w.writeheader()
        for chunk in range(n_chunks):
            chunk_f = os.path.join(STATE_DIR, 'f%d_meas_chunk%d.csv' % (fi, chunk))
            if not os.path.exists(chunk_f): continue
            with open(chunk_f) as rf:
                reader = csv.DictReader(rf,
                    fieldnames=(CSV_FIELDS if chunk > 0 else None))
                for row in reader:
                    if row['frame'] == 'frame': continue
                    frame_i = int(row['frame'])
                    row['arc_tip_vel_mm_s'] = (
                        '' if np.isnan(tip_vel[frame_i]) else tip_vel[frame_i])
                    w.writerow(row)
    print('Wrote: %s' % csv_out)

    vsum_f = os.path.join(STATE_DIR, 'f%d_velmap_sum.npy' % fi)
    vcnt_f = os.path.join(STATE_DIR, 'f%d_velmap_cnt.npy' % fi)
    if os.path.exists(vsum_f):
        vsum = np.load(vsum_f); vcnt = np.load(vcnt_f)
        vmap = np.where(vcnt > 0, vsum / vcnt, np.nan)
        np.save(os.path.join(out_dir, stem + '_velocity_map.npy'), vmap)
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        vm   = vmap / 1000
        vmax = np.nanpercentile(vm, 97)
        im   = ax.imshow(vm, cmap='plasma', vmin=0, vmax=max(vmax, 0.1), origin='upper')
        plt.colorbar(im, ax=ax, label='Arc-tip velocity (m/s)')
        ax.set_xlabel('x (px)'); ax.set_ylabel('y (px)')
        fig.savefig(os.path.join(out_dir, stem + '_velocity_map.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)
        print('Velocity map saved.')
    nxt_msg = ('NEXT: python blade_run2.py finalize %d' % (fi + 1)
               if fi < 2 else 'All files done. NEXT: python blade_run2.py combine')
    print(nxt_msg)


def phase_combine():
    import pandas as pd
    dfs = []
    for stem in STEMS:
        csv_f = os.path.join(OUT_BASE, stem + '_results', 'measurements.csv')
        if not os.path.exists(csv_f):
            print('MISSING:', csv_f); continue
        df = pd.read_csv(csv_f)
        df.insert(0, 'file', stem)
        dfs.append(df)
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        out = os.path.join(OUT_BASE, 'measurements_all.csv')
        combined.to_csv(out, index=False)
        det = (combined['arc_detected'] == 1).sum()
        print('Combined %d rows | %d detected (%.1f%%)' %
              (len(combined), det, 100 * det / len(combined)))
        print('Saved:', out)
    else:
        print('No CSVs found.')


# ── dispatch ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    phase = sys.argv[1]
    if   phase == 'calibrate': phase_calibrate(int(sys.argv[2]))
    elif phase == 'measure':   phase_measure(int(sys.argv[2]), int(sys.argv[3]))
    elif phase == 'velocity':  phase_velocity(int(sys.argv[2]))
    elif phase == 'velmap':    phase_velmap(int(sys.argv[2]), int(sys.argv[3]))
    elif phase == 'finalize':  phase_finalize(int(sys.argv[2]))
    elif phase == 'combine':   phase_combine()
    else: print('Unknown phase:', phase); sys.exit(1)
