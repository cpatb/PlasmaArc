#!/usr/bin/env python3
"""
qa_check.py  —  Verify binarization and velocity on the first N frames.

Renders an annotated QA image for each of the first --n-frames frames of
every TIFF, using *exactly* the same binarization parameters as the main
pipeline (blade_run2.py / run_pipeline.py).  Check these images to confirm:

  • The arc mask (magenta) correctly covers the arc channel
  • The skeleton (cyan) runs along the arc centreline
  • Tips (yellow) sit at the electrode contact points
  • The metrics panel shows plausible length / width / gap numbers
  • Velocity values (printed to console) are in a physically sensible range

Usage:
    python qa_check.py                    # first 100 frames, all 3 TIFFs
    python qa_check.py --n-frames 50      # first 50 frames only
    python qa_check.py --file 0           # file 1 only (0-indexed)
"""

import argparse, datetime
from pathlib import Path

import numpy as np
import tifffile
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from plasma_utils.image import normalize_frame, binarize, measure_frame
from plasma_utils.qa    import render_qa_frame

# ── match blade_run2.py exactly ───────────────────────────────────────────────
BLADE_DIR  = Path.home() / 'OneDrive' / 'Desktop' / 'BLADE'
FPS        = 200_000
PX_PER_MM  = 9.167
CLOSE_R    = 3
OPEN_R     = 0      # blade_run2.py uses open_r=0; QA must match
DARK_GUARD = 35
MIN_ARC_PX = 5
MAX_ARC_PX = 3000
ELEC_TOL_PX = 10    # must match blade_run2.ELEC_TOL_PX
                    # kept < half the 22 px apex gap so bands never overlap

# Physical velocity cap: re-stroke events cause the tip to jump the full
# frame height in one frame, producing ~5000 m/s artefacts.  Anything above
# this cap is flagged as unphysical.
MAX_VEL_M_S = 500     # m/s  (raise if your arc is genuinely faster)

TIFF_FILES = [
    BLADE_DIR / 'ADCB_0.8mmNoz_1000mL_N2_1.tiff',
    BLADE_DIR / 'ADCB_0.8mmNoz_1000mL_N2_2.tiff',
    BLADE_DIR / 'ADCB_0.8mmNoz_1000mL_N2_3.tiff',
]

# ── blade lines (optional) ────────────────────────────────────────────────────
# Written by calibrate_electrodes.py; shape (2,2,2):
#   [blade_A, blade_B] × [point_0, point_1] × [row, col]
_BLADE_LINES_PATH = BLADE_DIR / 'blade_lines.npy'
_blade_lines = None
if _BLADE_LINES_PATH.exists():
    _blade_lines = np.load(str(_BLADE_LINES_PATH))
    print(f'Blade lines loaded:'
          f' A=({_blade_lines[0,0,0]:.0f},{_blade_lines[0,0,1]:.0f})'
          f'→({_blade_lines[0,1,0]:.0f},{_blade_lines[0,1,1]:.0f})'
          f'  B=({_blade_lines[1,0,0]:.0f},{_blade_lines[1,0,1]:.0f})'
          f'→({_blade_lines[1,1,0]:.0f},{_blade_lines[1,1,1]:.0f})'
          f'  tol={ELEC_TOL_PX} px')


# ── helpers ───────────────────────────────────────────────────────────────────

def _binarize(raw: np.ndarray) -> np.ndarray:
    """Dark-guard + normalize + binarize, identical to blade_run2._binarize_raw."""
    if int(raw.max()) < DARK_GUARD:
        return np.zeros(raw.shape, dtype=bool)
    gray = normalize_frame(raw)
    mask = binarize(gray, close_r=CLOSE_R, open_r=OPEN_R)
    area = int(mask.sum())
    if area < MIN_ARC_PX or area > MAX_ARC_PX:
        return np.zeros(raw.shape, dtype=bool)
    return mask


def run_qa(tiff_path: Path, out_dir: Path, n_frames: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(str(tiff_path)) as tif:
        total  = len(tif.pages)
        n      = min(n_frames, total)
        H, W   = tif.pages[0].asarray().shape[:2]

        # ── pass 1: measure ──────────────────────────────────────────────────
        tip_rows = np.full(n, np.nan)
        tip_cols = np.full(n, np.nan)
        meas_all = []

        print(f"\n  {tiff_path.name}  ({n} frames)")
        for i in tqdm(range(n), desc='  measuring', unit='fr'):
            raw    = tif.pages[i].asarray()
            gray   = normalize_frame(raw)
            binary = _binarize(raw)
            m      = measure_frame(gray, binary, px_per_mm=PX_PER_MM,
                                   width_sample_step=5)
            meas_all.append(m)
            if m.arc_detected:
                # bottommost arc pixel — same definition as blade_run2
                ri, ci = np.where(binary)
                bot = ri.argmax()
                tip_rows[i] = float(ri[bot])
                tip_cols[i] = float(ci[bot])

        # ── central-difference velocity (same formula as the fixed pipeline) ──
        tip_vels_ms = np.full(n, np.nan)   # m/s for display
        for i in range(1, n - 1):
            rp, cp = tip_rows[i-1], tip_cols[i-1]
            rn, cn = tip_rows[i+1], tip_cols[i+1]
            if any(np.isnan(v) for v in [rp, cp, rn, cn]):
                continue
            disp_mm = np.hypot(rn - rp, cn - cp) / PX_PER_MM
            vel_ms  = disp_mm / (2.0 / FPS) / 1000.0   # mm/s → m/s
            if vel_ms <= MAX_VEL_M_S:
                tip_vels_ms[i] = vel_ms

        # ── velocity summary ──────────────────────────────────────────────────
        valid_v = tip_vels_ms[np.isfinite(tip_vels_ms)]
        n_det   = sum(1 for m in meas_all if m.arc_detected)
        print(f"  arc detected : {n_det}/{n} frames  ({100*n_det/n:.0f}%)")
        if len(valid_v) > 0:
            print(f"  tip velocity : mean {np.mean(valid_v):.1f} m/s  "
                  f"| 95th {np.percentile(valid_v, 95):.1f} m/s  "
                  f"| max {valid_v.max():.1f} m/s")
            n_capped = int((np.isnan(tip_vels_ms) & ~np.isnan(tip_rows[1:-1])).sum())
            if n_capped:
                print(f"  !! {n_capped} velocity values capped as re-stroke artefacts "
                      f"(>{MAX_VEL_M_S} m/s)")
        else:
            print("  tip velocity : no valid values in these frames")

        # ── pass 2: render QA frames ──────────────────────────────────────────
        print(f"  rendering {n} QA frames → {out_dir.name}/")
        for i in tqdm(range(n), desc='  rendering', unit='fr'):
            raw = tif.pages[i].asarray()
            fig = render_qa_frame(
                raw, meas_all[i], i, FPS,
                px_per_mm=PX_PER_MM,
                close_r=CLOSE_R,
                open_r=OPEN_R,        # ← matches pipeline
            )
            ax = fig.axes[0]

            # ── blade line overlay (cyan=A, orange=B) ────────────────────
            if _blade_lines is not None:
                _blade_colors = ['#00FFFF', '#FF8800']
                _blade_lbls   = ['A', 'B']
                H_fr, W_fr = raw.shape[:2]
                for k in range(2):
                    r1, c1 = _blade_lines[k, 0]
                    r2, c2 = _blade_lines[k, 1]
                    col_k  = _blade_colors[k]
                    # Direction vector along blade (normalised)
                    dr, dc = r2 - r1, c2 - c1
                    length = np.hypot(dr, dc)
                    # Extend line to frame edges for display
                    if length > 1e-9:
                        # Extend far beyond frame so line always crosses it
                        scale = max(H_fr, W_fr) * 2
                        r_lo = r1 - (dr / length) * scale
                        c_lo = c1 - (dc / length) * scale
                        r_hi = r1 + (dr / length) * scale
                        c_hi = c1 + (dc / length) * scale
                        ax.plot([c_lo, c_hi], [r_lo, r_hi],
                                color=col_k, lw=1.2, alpha=0.7,
                                zorder=8, clip_on=True)
                        # Tolerance bands: parallel lines offset ±ELEC_TOL_PX
                        perp_r = -dc / length
                        perp_c =  dr / length
                        for sign in (+1, -1):
                            ro = sign * ELEC_TOL_PX * perp_r
                            co = sign * ELEC_TOL_PX * perp_c
                            ax.plot([c_lo + co, c_hi + co],
                                    [r_lo + ro, r_hi + ro],
                                    color=col_k, lw=0.6, ls='--',
                                    alpha=0.4, zorder=7, clip_on=True)
                    # Label near midpoint of the clicked segment
                    rm, cm = (r1 + r2) / 2, (c1 + c2) / 2
                    ax.annotate(f'B{_blade_lbls[k]}', (cm, rm),
                                color=col_k, fontsize=6,
                                ha='center', va='bottom',
                                fontfamily='monospace', zorder=10)

            # ── velocity title annotation ─────────────────────────────────
            v = tip_vels_ms[i]
            v_str = f"{v:.1f} m/s" if np.isfinite(v) else "n/a"
            fig.savefig(out_dir / f'qa_{i:04d}.png',
                        dpi=120, facecolor=fig.get_facecolor())
            plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="QA check on first N frames")
    ap.add_argument('--n-frames', type=int, default=100,
                    help='Number of frames to process per file (default 100)')
    ap.add_argument('--file', type=int, default=None,
                    help='Process one file only: 0, 1, or 2 (default: all)')
    args = ap.parse_args()

    out_base = BLADE_DIR / datetime.date.today().isoformat() / 'qa_check'
    files    = TIFF_FILES if args.file is None else [TIFF_FILES[args.file]]

    print(f"QA check — {args.n_frames} frames per file")
    print(f"Binarization: close_r={CLOSE_R}, open_r={OPEN_R} "
          f"(matches blade_run2.py)")
    print(f"Velocity cap: {MAX_VEL_M_S} m/s")
    print(f"Output: {out_base}\n")

    for tiff_path in files:
        if not tiff_path.exists():
            print(f"  MISSING: {tiff_path}")
            continue
        run_qa(tiff_path, out_base / tiff_path.stem, args.n_frames)

    print("\nDone.  Open %s to inspect QA frames." % out_base)


if __name__ == '__main__':
    main()
