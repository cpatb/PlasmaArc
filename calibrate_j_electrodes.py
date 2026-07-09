#!/usr/bin/env python3
"""
calibrate_j_electrodes.py  —  Define the J-electrode geometry (finite blades +
minimum gap) for the maintained BLADE-style analysis path.

WHY A SEPARATE CALIBRATOR FROM calibrate_electrodes.py
------------------------------------------------------
`calibrate_electrodes.py` captures two *infinite* blade lines (4 clicks anywhere
along each edge).  That is correct for the 11MAY26 BLADE reactor, where the
blades are long and the arc feet glide across most of the frame, so the
extrapolated line is the electrode everywhere it matters.

The J-electrode reactor is different (cf. METHODS.md §1):

    * two razor blades in an inverted V (∧), ~15 deg included angle,
    * a small **minimum gap** at the apex (3.2 mm), and
    * blades of **finite length** — each has a physical TIP near the apex and a
      FAR end where it leaves the field of view / mount.

For this geometry two things must be specified that the BLADE calibrator does
not capture, and that the user flagged explicitly:

  1. **Electrode ends** — the finite segment of each blade.  A free arc tip that
     happens to lie on the *extrapolated* blade line BEYOND the physical blade
     must NOT count as "attached".  We therefore store each electrode as a
     finite SEGMENT (tip-end → far-end) and, in analyze_blade.py with
     --finite-segments, require the arc foot to project ONTO that segment.

  2. **Minimum gap distance** — the apex gap (default 3.2 mm).  Clicking the two
     blade TIP-ends gives the gap directly: their pixel separation is the
     minimum gap in px, which (with the known mm value) yields px/mm, and their
     mean row yields the electrode-gap cutoff row used for flash rejection
     (analyze_blade.py --gap-row) and as the apex anchor for Mode A/B.

CLICK ORDER (4 clicks, TIP-END FIRST for each blade)
----------------------------------------------------
  Click 1 — Blade A TIP-END   (the point nearest the apex / minimum gap)
  Click 2 — Blade A FAR-END    (the other end of the visible blade)
  Click 3 — Blade B TIP-END   (nearest the apex)
  Click 4 — Blade B FAR-END

Outputs (default: alongside, in the recording's results folder)
---------------------------------------------------------------
  blade_lines.npy   shape (2,2,2): [bladeA,bladeB] x [tip_end, far_end] x [row,col]
                    — SAME format analyze_blade.py --blade-lines already consumes,
                      with the convention pt0 = tip-end (apex side).
  j_geometry.json   the J-specific extras derived from the clicks:
                      { min_gap_mm, min_gap_px, px_per_mm, gap_row,
                        arc_direction, tip_end_a/b, far_end_a/b,
                        finite_segments: true, segment_margin_px }

The companion orchestrator run_j_co2.py reads j_geometry.json to thread
--gap-row, --px-per-mm, --finite-segments and --blade-lines through the same
stages as run_11may26.py.

USAGE
-----
    # interactive (needs a display) — first arc frame auto-found
    python calibrate_j_electrodes.py --tiff "<...>_results/tiff/<stem>_0001.tiff"

    # specific frame, custom outputs, known gap value
    python calibrate_j_electrodes.py --tiff CHUNK.tiff --frame 1200 \
        --gap-mm 3.2 --arc-direction down \
        --out  "<...>_results/blade_lines.npy" \
        --geom-out "<...>_results/j_geometry.json"

    # headless / scripted (no display): pass the 4 points as row,col pairs,
    # tip-end first per blade:  Atip ; Afar ; Btip ; Bfar
    python calibrate_j_electrodes.py --tiff CHUNK.tiff \
        --points "30,150 ; 90,120 ; 32,172 ; 92,205" --gap-mm 3.2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile

DARK_GUARD = 35
DEFAULT_GAP_MM = 3.2          # confirmed true minimum electrode gap
                             # (hand_calibration_authoritative_25JUN26; was 2.4)
DEFAULT_SEG_MARGIN_PX = 6     # how far past a clicked end a foot may still attach
_BLADE_COLORS = ['#00FFFF', '#FF8800']   # cyan = A, orange = B
_BLADE_LABELS = ['Blade A', 'Blade B']


# ── helpers ───────────────────────────────────────────────────────────────────

def _find_first_arc_frame(tif, n_search=1000):
    for i in range(min(n_search, len(tif.pages))):
        if int(tif.pages[i].asarray().max()) >= DARK_GUARD:
            return i
    return 0


def _parse_points(s):
    """'r,c ; r,c ; r,c ; r,c' -> list of 4 (row,col) tuples (tip,far per blade)."""
    parts = [p for p in s.replace('\n', ' ').split(';') if p.strip()]
    pts = []
    for p in parts:
        rc = p.replace(',', ' ').split()
        if len(rc) != 2:
            raise ValueError("each point must be 'row,col'; got %r" % p)
        pts.append((float(rc[0]), float(rc[1])))
    if len(pts) != 4:
        raise ValueError("need exactly 4 points (Atip;Afar;Btip;Bfar), got %d" % len(pts))
    return pts


def derive_geometry(clicks, gap_mm, arc_direction, seg_margin_px,
                    px_per_mm_override=None):
    """clicks = [A_tip, A_far, B_tip, B_far] each (row,col).

    Returns (blade_lines (2,2,2), geom_dict).  Pure function — unit-testable,
    no display needed.
    """
    A_tip, A_far, B_tip, B_far = (np.asarray(c, float) for c in clicks)

    blade_lines = np.array([[A_tip, A_far],
                            [B_tip, B_far]])            # (2,2,2), pt0 = tip-end

    min_gap_px = float(np.hypot(*(A_tip - B_tip)))
    if px_per_mm_override is not None:
        px_per_mm = float(px_per_mm_override)
    else:
        px_per_mm = min_gap_px / float(gap_mm) if gap_mm > 0 else None

    # Electrode-gap cutoff row (flash rejection + apex anchor).
    #   'down'  electrodes at SMALL rows; reject arc pixels ABOVE the tips
    #           (row < gap_row), so place gap_row just above the tip-ends.
    #   'up'    electrodes at LARGE rows; reject below (row > gap_row), so place
    #           gap_row just below the tip-ends.
    tip_mean_row = 0.5 * (A_tip[0] + B_tip[0])
    if arc_direction == 'down':
        gap_row = int(np.floor(tip_mean_row)) - 2
    else:
        gap_row = int(np.ceil(tip_mean_row)) + 2

    # per-blade angle from vertical (expect ~7.5 deg each for a 15 deg V)
    def ang_from_vertical(p_tip, p_far):
        d = p_far - p_tip
        return float(np.degrees(np.arctan2(abs(d[1]), abs(d[0]))))

    geom = dict(
        min_gap_mm=float(gap_mm),
        min_gap_px=round(min_gap_px, 3),
        px_per_mm=round(px_per_mm, 4) if px_per_mm else None,
        gap_row=int(gap_row),
        arc_direction=arc_direction,
        finite_segments=True,
        segment_margin_px=int(seg_margin_px),
        tip_end_a=[round(float(A_tip[0]), 2), round(float(A_tip[1]), 2)],
        far_end_a=[round(float(A_far[0]), 2), round(float(A_far[1]), 2)],
        tip_end_b=[round(float(B_tip[0]), 2), round(float(B_tip[1]), 2)],
        far_end_b=[round(float(B_far[0]), 2), round(float(B_far[1]), 2)],
        blade_a_angle_deg=round(ang_from_vertical(A_tip, A_far), 2),
        blade_b_angle_deg=round(ang_from_vertical(B_tip, B_far), 2),
    )
    return blade_lines, geom


def _save(blade_lines, geom, out_path, geom_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), blade_lines)
    Path(geom_path).parent.mkdir(parents=True, exist_ok=True)
    with open(geom_path, 'w') as f:
        json.dump(geom, f, indent=2)
    print('\nSaved blade segments -> %s' % out_path)
    print('Saved J geometry     -> %s' % geom_path)
    print('  min gap   : %.2f px = %.2f mm  -> px/mm = %s' % (
        geom['min_gap_px'], geom['min_gap_mm'], geom['px_per_mm']))
    print('  gap_row   : %d  (arc_direction=%s)' % (geom['gap_row'], geom['arc_direction']))
    print('  blade A   : %.1f deg from vertical   blade B: %.1f deg  (expect ~7.5 each)'
          % (geom['blade_a_angle_deg'], geom['blade_b_angle_deg']))
    incl = geom['blade_a_angle_deg'] + geom['blade_b_angle_deg']
    print('  included  : ~%.1f deg  (expect ~15)' % incl)


# ── interactive picking ───────────────────────────────────────────────────────

def _pick_interactive(raw, frame_idx):
    import matplotlib.pyplot as plt
    H, W = raw.shape[:2]
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#0a0a0a'); ax.set_facecolor('#0a0a0a')
    disp = np.sqrt(raw.astype(float) / max(float(raw.max()), 1))
    ax.imshow(disp, cmap='gray', vmin=0, vmax=1, origin='upper')
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
    ax.set_title('Frame %d — click  A-tip, A-far, B-tip, B-far  (tip-end FIRST)'
                 % frame_idx, color='white', fontsize=10)

    step_labels = [
        'Click 1: BLADE A TIP-END (nearest the apex/min gap)',
        'Click 2: BLADE A FAR-END',
        'Click 3: BLADE B TIP-END (nearest the apex/min gap)',
        'Click 4: BLADE B FAR-END',
    ]
    print('\n' + '\n'.join('  ' + s for s in step_labels))
    clicks = []

    def on_click(event):
        if event.inaxes != ax or event.button != 1 or len(clicks) >= 4:
            return
        row, col = float(event.ydata), float(event.xdata)
        k = len(clicks) // 2
        clicks.append((row, col))
        ax.scatter([col], [row], c=_BLADE_COLORS[k], s=120, zorder=10,
                   edgecolors='white', linewidths=1.0,
                   marker='*' if len(clicks) % 2 == 1 else 'o')
        if len(clicks) % 2 == 0:                      # both ends of a blade done
            (r0, c0), (r1, c1) = clicks[-2], clicks[-1]
            ax.plot([c0, c1], [r0, r1], color=_BLADE_COLORS[k], lw=1.6, zorder=8)
        if len(clicks) == 4:                          # draw the min-gap line
            (rat, cat) = clicks[0]; (rbt, cbt) = clicks[2]
            ax.plot([cat, cbt], [rat, rbt], color='gold', lw=1.2, ls='--', zorder=9)
        if len(clicks) < 4:
            ax.set_xlabel(step_labels[len(clicks)], color='yellow', fontsize=9)
        else:
            ax.set_xlabel('4/4 clicks — close the window to save', color='lime', fontsize=9)
        fig.canvas.draw_idle()

    ax.set_xlabel(step_labels[0], color='yellow', fontsize=9)
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.tight_layout(pad=0.4)
    plt.show()
    return clicks


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Define J-electrode finite blades + minimum gap')
    ap.add_argument('--tiff', required=True, help='TIFF chunk to display/sample')
    ap.add_argument('--frame', type=int, default=None,
                    help='Frame index (default: first arc frame)')
    ap.add_argument('--gap-mm', type=float, default=DEFAULT_GAP_MM,
                    help='Known minimum electrode gap in mm (default %.1f)' % DEFAULT_GAP_MM)
    ap.add_argument('--px-per-mm', type=float, default=None,
                    help='Override px/mm instead of deriving it from the clicked '
                         'tip-end separation and --gap-mm.')
    ap.add_argument('--arc-direction', choices=['down', 'up'], default='down',
                    help="Arc propagation direction (J CO2 reactor: 'down').")
    ap.add_argument('--segment-margin-px', type=int, default=DEFAULT_SEG_MARGIN_PX,
                    help='How far past a clicked electrode end a foot may still '
                         'count as attached (default %d).' % DEFAULT_SEG_MARGIN_PX)
    ap.add_argument('--points', type=str, default=None,
                    help="Headless mode: 4 points 'Atip ; Afar ; Btip ; Bfar' as "
                         "row,col pairs. Skips the display.")
    ap.add_argument('--out', type=str, default=None,
                    help='blade_lines.npy output (default: alongside --tiff in '
                         'the parent results folder).')
    ap.add_argument('--geom-out', type=str, default=None,
                    help='j_geometry.json output (default: next to --out).')
    args = ap.parse_args()

    tiff_path = Path(args.tiff)
    if not tiff_path.exists():
        print('ERROR: TIFF not found:\n  %s' % tiff_path); return

    # Default outputs: the recording results folder (tiff/ -> parent)
    results_dir = tiff_path.parent.parent if tiff_path.parent.name == 'tiff' \
        else tiff_path.parent
    out_path = Path(args.out) if args.out else results_dir / 'blade_lines.npy'
    geom_path = Path(args.geom_out) if args.geom_out else out_path.with_name('j_geometry.json')

    if args.points:
        clicks = _parse_points(args.points)
        print('Headless mode — using supplied points (tip-end first per blade).')
    else:
        with tifffile.TiffFile(str(tiff_path)) as tif:
            frame_idx = args.frame if args.frame is not None else _find_first_arc_frame(tif)
            raw = tif.pages[frame_idx].asarray()
        print('Loaded %s  frame %d  max DN=%d' % (tiff_path.name, frame_idx, raw.max()))
        clicks = _pick_interactive(raw, frame_idx)
        if len(clicks) < 4:
            print('\nOnly %d/4 clicks — nothing saved.' % len(clicks)); return

    blade_lines, geom = derive_geometry(
        clicks, args.gap_mm, args.arc_direction, args.segment_margin_px,
        px_per_mm_override=args.px_per_mm)
    _save(blade_lines, geom, out_path, geom_path)
    print('\nNext: run_j_co2.py reads j_geometry.json and threads these through '
          'analyze_blade.py (--blade-lines, --finite-segments, --gap-row, --px-per-mm).')


if __name__ == '__main__':
    main()
