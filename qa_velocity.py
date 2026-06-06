#!/usr/bin/env python3
"""
qa_velocity.py  —  Ground-truth QA for the event-velocity regression.

Purpose: verify that the velocities written by event_velocity.py are real and
not fitting artifacts. For a selection of events it produces, per event, a row
of panels:

  [ trajectory + fit ]  [ frame @ start ]  [ frame @ mid ]  [ frame @ end ]

and three INDEPENDENT velocity estimates that should agree:
  1. regression  — the pipeline's advancing-front least-squares slope
  2. chord       — (tip_end - tip_start) / (t_end - t_start) over the front
  3. local-median— median of the per-frame sliding-window local_vel_ms
If these three numbers agree and the fit overlays the points (high R^2), the
velocity is trustworthy. The frame panels overlay the *recorded* arc tip
(arc_tip_row/col from the CSV) on the raw image, so you can confirm by eye that
the arc tip was detected correctly and genuinely moved that far in that time.

No fabricated data: every number is recomputed from the per-frame CSV and the
TIFF the pipeline analysed.

Usage:
    python qa_velocity.py \
        --csv   "<results>/measurements_all_events.csv" \
        --events "<results>/event_summary.csv" \
        --tiff  "<results>/tiff/<stem>_0001.tiff" \
        --arc-direction down --px-per-mm 8.8 \
        --out   "<results>/qa_velocity.png"

    # select specific events instead of the auto pick (fastest/slowest/worst-R2)
    python qa_velocity.py ... --event-ids 15 31
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

from analyze_blade import binarize, skeleton_endpoints, arc_front_tip

FPS_DEFAULT = 200_000


def advancing_front(rows):
    """Indices where the tip reaches a new maximum (monotonic envelope)."""
    keep, runmax = [], -np.inf
    for i, r in enumerate(rows):
        if r >= runmax:
            keep.append(i); runmax = r
    return np.array(keep, int)


def select_events(ev, ids):
    if ids:
        return ev[ev['event_id'].isin(ids)].copy()
    pick = set()
    down = ev[ev['vel_ms'] > 0]
    if len(down):
        pick.add(int(down.loc[down['vel_ms'].idxmax(), 'event_id']))   # fastest
        pick.add(int(down.loc[down['vel_ms'].idxmin(), 'event_id']))   # slowest
        pick.add(int(down.loc[down['r2'].idxmin(), 'event_id']))       # worst fit
        med = down['vel_ms'].mean()
        pick.add(int(down.loc[(down['vel_ms'] - med).abs().idxmin(), 'event_id']))  # median
    return ev[ev['event_id'].isin(pick)].copy()


def main():
    ap = argparse.ArgumentParser(description='Ground-truth QA for event-velocity regression')
    ap.add_argument('--csv', required=True, help='measurements_all_events.csv')
    ap.add_argument('--events', required=True, help='event_summary.csv')
    ap.add_argument('--tiff', required=True, help='chunk TIFF the pipeline analysed')
    ap.add_argument('--arc-direction', choices=['up', 'down'], required=True)
    ap.add_argument('--px-per-mm', type=float, default=8.8)
    ap.add_argument('--fps', type=float, default=FPS_DEFAULT)
    ap.add_argument('--thresh', type=int, default=100)
    ap.add_argument('--frame-offset', type=int, default=0,
                    help='Global frame number of TIFF page 0 (for multi-chunk runs)')
    ap.add_argument('--event-ids', type=int, nargs='*', default=None)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    ev = pd.read_csv(args.events)
    for c in ['arc_tip_row', 'arc_tip_col', 'local_vel_ms']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    sel = select_events(ev, args.event_ids)
    sel = sel.sort_values('vel_ms', ascending=False).reset_index(drop=True)
    if len(sel) == 0:
        raise SystemExit('No events to QA.')

    import tifffile
    tif = tifffile.TiffFile(args.tiff)
    H, W = tif.pages[0].asarray().shape

    nrows = len(sel)
    fig, axes = plt.subplots(nrows, 4, figsize=(18, 4.2 * nrows),
                             squeeze=False, facecolor='white')

    print('event  reg_vel  chord_vel  local_med   R2   |  agree?')
    for ri, (_, e) in enumerate(sel.iterrows()):
        eid = int(e['event_id'])
        g = df[df['event_id'] == eid].sort_values('frame')
        g = g.dropna(subset=['arc_tip_row'])
        frames = g['frame'].values.astype(float)
        rows = g['arc_tip_row'].values.astype(float)
        erow = float(e['electrode_row'])
        # advancing front (same definition the pipeline uses)
        front = advancing_front(rows)
        ff, rf = frames[front], rows[front]
        t_ms_all = frames / args.fps * 1e3
        dist_all = (rows - erow) / args.px_per_mm
        t_ms_f = ff / args.fps * 1e3
        dist_f = (rf - erow) / args.px_per_mm

        # (1) regression on advancing front (reproduces the pipeline)
        slope, intercept = np.polyfit(ff, rf, 1)
        reg_vel = slope * args.fps / args.px_per_mm / 1000
        pred = slope * ff + intercept
        ss_res = np.sum((rf - pred) ** 2); ss_tot = np.sum((rf - rf.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        # (2) chord velocity over the front
        dt_s = (ff[-1] - ff[0]) / args.fps
        chord_vel = ((rf[-1] - rf[0]) / args.px_per_mm / 1000) / dt_s if dt_s > 0 else np.nan
        # (3) median local sliding-window velocity
        loc = g['local_vel_ms'].dropna() if 'local_vel_ms' in g.columns else pd.Series([], dtype=float)
        local_med = float(loc[loc > 0].mean()) if len(loc[loc > 0]) else np.nan

        vals = [v for v in [reg_vel, chord_vel, local_med] if not np.isnan(v)]
        agree = (max(vals) - min(vals)) / max(np.median(vals), 1e-9) < 0.25 if len(vals) >= 2 else False
        print('%5d  %7.3f  %8.3f  %8s  %.3f  |  %s' % (
            eid, reg_vel, chord_vel,
            ('%.3f' % local_med) if not np.isnan(local_med) else 'n/a', r2,
            'YES' if agree else 'check'))

        # ---- trajectory panel ----
        ax = axes[ri][0]
        ax.scatter(t_ms_all, dist_all, s=10, c='#bbbbbb', label='all detected')
        ax.scatter(t_ms_f, dist_f, s=16, c='#1565C0', label='advancing front')
        xs = np.array([ff.min(), ff.max()])
        ax.plot(xs / args.fps * 1e3, ((slope * xs + intercept) - erow) / args.px_per_mm,
                'r-', lw=2, label='regression fit')
        ax.set_xlabel('time (ms)'); ax.set_ylabel('tip distance from electrode (mm)')
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # ---- frame panels at start / mid / end of the advancing front ----
        sel_idx = [front[0], front[len(front) // 2], front[-1]]
        for pj, fi_local in enumerate(sel_idx):
            ax = axes[ri][pj + 1]
            gframe = int(frames[fi_local])
            page = gframe - args.frame_offset
            row_rec = rows[fi_local]
            col_rec = float(g.iloc[fi_local]['arc_tip_col']) if 'arc_tip_col' in g.columns else np.nan
            # un-flip the recorded row back to raw image coordinates
            raw_row = (H - 1 - row_rec) if args.arc_direction == 'up' else row_rec
            if 0 <= page < len(tif.pages):
                raw = tif.pages[page].asarray()
                ax.imshow(np.power(raw / max(raw.max(), 1), 0.45), cmap='inferno', vmin=0, vmax=1)
                b = binarize(raw, 3, bright_thresh=args.thresh)
                if b.any():
                    ax.contour(b, levels=[0.5], colors='cyan', linewidths=0.5)
                    sk = skeletonize(b)
                    rr, cc = np.where(sk); ax.scatter(cc, rr, s=0.3, c='lime')
                if not np.isnan(col_rec):
                    ax.plot(col_rec, raw_row, 'x', c='red', ms=12, mew=2.5)
                ax.axhline((H - 1 - erow) if args.arc_direction == 'up' else erow,
                           color='yellow', lw=0.6, ls='--')
            else:
                ax.text(0.5, 0.5, 'frame %d\nnot in this TIFF\n(use --frame-offset)' % gframe,
                        ha='center', va='center')
            ax.axis('off')

    tif.close()
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out, dpi=110, facecolor='white')
    plt.close(fig)
    print('\nSaved %s' % args.out)
    print('Read the panels left-to-right: the red x is the RECORDED tip; confirm it '
          'sits on the arc front and advances from start to end. The three velocity '
          'estimates (reg/chord/local) agreeing is the quantitative cross-check.')


if __name__ == '__main__':
    main()
