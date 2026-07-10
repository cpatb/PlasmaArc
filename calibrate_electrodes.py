#!/usr/bin/env python3
"""
calibrate_electrodes.py  —  Define razor-blade lines for arc_complete detection.

Each electrode is a razor blade of finite length, arranged in an inverted V
(∧) with a 15-degree included angle and a 2.4 mm minimum gap at the apex.
The plasma arc can attach at any point along a blade's length, so arc_complete
is determined by the perpendicular distance of each skeleton endpoint to its
respective BLADE LINE — not to a fixed electrode tip position.

This script shows a frame from a TIFF and asks you to click TWO points on
each blade edge (four clicks total).  Those four points define two straight
lines (one per blade) which are saved to:

    ~/OneDrive/Desktop/BLADE/blade_lines.npy
    shape (2, 2, 2):  [blade_A, blade_B] × [point_0, point_1] × [row, col]

blade_run2.py and qa_check.py load this file automatically.

Click order
-----------
  Click 1 — any point on Blade A (left blade edge)
  Click 2 — a second point on Blade A (further along the edge)
  Click 3 — any point on Blade B (right blade edge)
  Click 4 — a second point on Blade B

Tip: use a frame where both blade edges are faintly visible even without the
arc.  Frames near the start of the recording often show the blade outlines.
Run with  --show-only  to browse frames without saving.

Usage:
    # Original ADCB files (uses TIFF_FILES list)
    python calibrate_electrodes.py              # first ADCB TIFF, auto-find frame
    python calibrate_electrodes.py --file 1     # second ADCB TIFF
    python calibrate_electrodes.py --frame 500  # specific frame index

    # Any TIFF (e.g. 11MAY26 chunks)
    python calibrate_electrodes.py --tiff "path/to/chunk.tiff"
    python calibrate_electrodes.py --tiff "path/to/chunk.tiff" --frame 1000
    python calibrate_electrodes.py --tiff "path/to/chunk.tiff" \\
        --out "path/to/blade_lines_11may26.npy"

    python calibrate_electrodes.py --show-only  # display only, no save
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile
import electrode_geometry as eg
import matplotlib.pyplot as plt

# ── config (mirrors blade_run2.py) ────────────────────────────────────────────
BLADE_DIR   = Path.home() / 'OneDrive' / 'Desktop' / 'BLADE'
DARK_GUARD  = 35
OUT_PATH    = BLADE_DIR / 'blade_lines.npy'
ELEC_TOL_PX = 10    # tolerance band half-width drawn on-screen for reference.
                    # < half the 22 px apex gap — bands from the two blades
                    # must never overlap or arc_complete becomes trivially true.

TIFF_FILES = [
    BLADE_DIR / 'ADCB_0.8mmNoz_1000mL_N2_1.tiff',
    BLADE_DIR / 'ADCB_0.8mmNoz_1000mL_N2_2.tiff',
    BLADE_DIR / 'ADCB_0.8mmNoz_1000mL_N2_3.tiff',
]

_BLADE_COLORS = ['#00FFFF', '#FF8800']   # cyan = A, orange = B
_BLADE_LABELS = ['Blade A', 'Blade B']


def _extend_line_to_frame(r1, c1, r2, c2, H, W, pad=20):
    """Return two endpoints of the infinite line (r1,c1)-(r2,c2) clipped to
    the frame boundary [0-H] × [0-W] for drawing purposes."""
    dr, dc = r2 - r1, c2 - c1
    if abs(dc) < 1e-9 and abs(dr) < 1e-9:
        return (c1, r1), (c2, r2)
    pts = []
    for t in np.linspace(-10, 10, 400):
        rr = r1 + t * dr
        cc = c1 + t * dc
        if -pad <= rr <= H + pad and -pad <= cc <= W + pad:
            pts.append((cc, rr))
    if len(pts) < 2:
        return (c1, r1), (c2, r2)
    return pts[0], pts[-1]


def _draw_blade(ax, points, color, label, H, W):
    """Draw a blade line and its tolerance band on ax."""
    if len(points) < 2:
        return
    (r1, c1), (r2, c2) = points
    p0, p1 = _extend_line_to_frame(r1, c1, r2, c2, H, W)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
            color=color, lw=1.5, zorder=8, label=label)
    # tolerance band: two parallel lines offset by ELEC_TOL_PX
    dr, dc = r2 - r1, c2 - c1
    length = np.hypot(dr, dc)
    if length > 1e-9:
        perp_r, perp_c = -dc / length, dr / length
        for sign in (+1, -1):
            roff = sign * ELEC_TOL_PX * perp_r
            coff = sign * ELEC_TOL_PX * perp_c
            q0, q1 = _extend_line_to_frame(r1 + roff, c1 + coff,
                                           r2 + roff, c2 + coff, H, W)
            ax.plot([q0[0], q1[0]], [q0[1], q1[1]],
                    color=color, lw=0.7, ls='--', alpha=0.45, zorder=7)


def main() -> None:
    ap = argparse.ArgumentParser(
        description='Click two points per blade to define blade lines')
    ap.add_argument('--tiff',      type=str,  default=None,
                    help='Path to any TIFF file (overrides --file).  Use this '
                         'for recordings other than the hardcoded ADCB list, '
                         'e.g. 11MAY26 chunks.')
    ap.add_argument('--file',      type=int,  default=0,
                    help='ADCB TIFF file index 0/1/2 (default 0; ignored when '
                         '--tiff is supplied)')
    ap.add_argument('--frame',     type=int,  default=None,
                    help='Frame index (default: first arc frame)')
    ap.add_argument('--out',       type=str,  default=None,
                    help='Output path for blade_lines.npy  '
                         '(default: %s)' % OUT_PATH)
    ap.add_argument('--show-only', action='store_true',
                    help='Display frame without saving')
    args = ap.parse_args()

    # Resolve output path
    out_path = Path(args.out) if args.out else OUT_PATH

    # Resolve input TIFF
    if args.tiff:
        tiff_path = Path(args.tiff)
    else:
        tiff_path = TIFF_FILES[args.file]
    if not tiff_path.exists():
        print(f'ERROR: TIFF not found:\n  {tiff_path}')
        return

    print(f'Loading  {tiff_path.name}')
    with tifffile.TiffFile(str(tiff_path)) as tif:
        frame_idx = (args.frame if args.frame is not None
                     else eg.find_first_arc_frame(tif))
        raw = tif.pages[frame_idx].asarray()
    H, W = raw.shape[:2]
    print(f'  Frame {frame_idx}  max DN = {raw.max()}')

    # ── display ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    disp = np.sqrt(raw.astype(float) / max(float(raw.max()), 1))
    ax.imshow(disp, cmap='gray', vmin=0, vmax=1, origin='upper')
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.axis('off')

    # Show previously saved lines if they exist
    if out_path.exists():
        prev = np.load(str(out_path))
        for k in range(2):
            _draw_blade(ax, [tuple(prev[k, 0]), tuple(prev[k, 1])],
                        _BLADE_COLORS[k], f'{_BLADE_LABELS[k]} (saved)', H, W)
        ax.legend(loc='lower right', fontsize=7, framealpha=0.6,
                  facecolor='black', edgecolor='none', labelcolor='white')
        print('  Existing blade lines loaded.')
        print(f'    A: ({prev[0,0,0]:.0f},{prev[0,0,1]:.0f}) → '
              f'({prev[0,1,0]:.0f},{prev[0,1,1]:.0f})')
        print(f'    B: ({prev[1,0,0]:.0f},{prev[1,0,1]:.0f}) → '
              f'({prev[1,1,0]:.0f},{prev[1,1,1]:.0f})')
        print('  Click to replace, or close to keep.')

    plt.tight_layout(pad=0.3)

    # ── click sequence: 2 points per blade ────────────────────────────────────
    # clicks[0:2] = blade A,  clicks[2:4] = blade B
    clicks: list[tuple[float, float]] = []
    blade_pts: list[list[tuple]] = [[], []]   # blade_pts[k] = list of (row,col)

    def on_click(event):
        if event.inaxes != ax or event.button != 1 or len(clicks) >= 4:
            return
        row, col = float(event.ydata), float(event.xdata)
        n = len(clicks)
        clicks.append((row, col))
        k = n // 2       # 0 = blade A, 1 = blade B
        blade_pts[k].append((row, col))

        ax.scatter([col], [row], c=_BLADE_COLORS[k], s=120, zorder=10,
                   edgecolors='white', linewidths=1.0)

        # Once we have both points for a blade, draw the line
        if len(blade_pts[k]) == 2:
            _draw_blade(ax, blade_pts[k], _BLADE_COLORS[k],
                        _BLADE_LABELS[k], H, W)

        if n + 1 < 4:
            pass
        else:
            pass
            # draw inter-blade gap line at the apex region
            # (just a visual aid; not saved)

        fig.canvas.draw_idle()

    if not args.show_only:
        fig.canvas.mpl_connect('button_press_event', on_click)

    plt.show()

    # ── save ───────────────────────────────────────────────────────────────────
    if args.show_only:
        print('--show-only: nothing saved.')
        return

    if len(clicks) < 4:
        print(f'\nOnly {len(clicks)}/4 clicks recorded — blade lines NOT updated.')
        return

    blade_lines = np.array([
        [clicks[0], clicks[1]],   # blade A: point0, point1
        [clicks[2], clicks[3]],   # blade B: point0, point1
    ])   # shape (2, 2, 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), blade_lines)
    print('\nSaved blade lines -> ' + str(out_path))
    for k in range(2):
        r1, c1 = blade_lines[k, 0]
        r2, c2 = blade_lines[k, 1]
        # Angle from vertical — expect ~7.5 deg for each blade in a 15-deg V
        ang_deg = np.degrees(np.arctan2(abs(c2 - c1), abs(r2 - r1)))
        print('  %s: (%.0f,%.0f) -> (%.0f,%.0f)  |  %.1f deg from vertical (expect ~7.5)'
              % (_BLADE_LABELS[k], r1, c1, r2, c2, ang_deg))
    print('  Tolerance band: +/-%d px' % ELEC_TOL_PX)
    print('\nDelete pipeline_state/ and re-run the pipeline to apply the new criterion.')


if __name__ == '__main__':
    main()
