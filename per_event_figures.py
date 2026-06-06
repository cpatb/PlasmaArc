"""
per_event_figures.py  —  Per-event velocity profile figures

Reads event_summary.csv and per-event CSVs from events/ and generates
a two-panel figure for each event:
  Top    — arc-tip row vs frame (trajectory within the event)
  Bottom — local velocity vs normalised arc position (within-event dynamics)

Both panels are annotated with event metadata: mode, velocity, R², duration.

Usage:
    python per_event_figures.py [events_dir] [--max N] [--mode A|B|all]

    events_dir   path to the events/ subdirectory produced by event_velocity.py
                 (default: looks for events/ next to this script or in CWD)
    --max N      limit to N events (default: all)
    --mode       only plot events of given mode (default: all)

Outputs (written into events_dir/figures/):
    event_NNNN.png   — one figure per event
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Appearance ────────────────────────────────────────────────────────────────
MODE_COLOR  = {'A': '#1565C0', 'B': '#C62828', 'unknown': '#546E7A'}
ALPHA_DOTS  = 0.55
FPS         = 200_000


def frame_to_ms(frames):
    """Convert frame count to milliseconds at FPS."""
    return np.asarray(frames, dtype=float) / FPS * 1000


def plot_event(event_row, frames_df, dyn_df, out_path):
    """
    Two-panel figure for a single event.

    Parameters
    ----------
    event_row  : Series  — row from events_index.csv
    frames_df  : DataFrame  — event_NNNN_frames.csv
    dyn_df     : DataFrame or None  — event_NNNN_dynamics.csv (may be absent)
    out_path   : Path  — where to save the figure
    """
    eid    = int(event_row['event_id'])
    mode   = str(event_row.get('mode', 'unknown'))
    vel    = float(event_row.get('vel_ms', np.nan))
    r2     = float(event_row.get('r2', np.nan))
    dur_fr = int(event_row.get('duration_frames', 0))
    dur_ms = dur_fr / FPS * 1000
    nn     = event_row.get('norm_nucleation_pos', np.nan)
    ext_px = event_row.get('extension_px', np.nan)
    color  = MODE_COLOR.get(mode, '#546E7A')

    has_dyn = dyn_df is not None and len(dyn_df) > 0

    fig, axes = plt.subplots(
        2 if has_dyn else 1, 1,
        figsize=(8, 7 if has_dyn else 4),
        squeeze=False
    )
    axes = axes.flatten()

    # ── Panel 1: arc-tip row vs time (frame trajectory) ──────────────────────
    ax = axes[0]
    det = frames_df[frames_df['arc_detected'] == 1].copy()
    if 'arc_tip_row' in det.columns and len(det) > 0:
        det = det.sort_values('frame')
        t_ms = frame_to_ms(det['frame'] - det['frame'].iloc[0])

        # Split into complete / partial arc if column exists
        if 'arc_complete' in det.columns:
            comp = det[det['arc_complete'].astype(float) == 1]
            part = det[det['arc_complete'].astype(float) == 0]
            if len(comp):
                ax.scatter(frame_to_ms(comp['frame'] - det['frame'].iloc[0]),
                           comp['arc_tip_row'],
                           s=6, alpha=ALPHA_DOTS, color=color, label='Complete arc')
            if len(part):
                ax.scatter(frame_to_ms(part['frame'] - det['frame'].iloc[0]),
                           part['arc_tip_row'],
                           s=4, alpha=ALPHA_DOTS * 0.6, color='#90A4AE',
                           label='Partial arc')
        else:
            ax.scatter(t_ms, det['arc_tip_row'],
                       s=6, alpha=ALPHA_DOTS, color=color)

        # Overlay the advancing-front regression line
        if not np.isnan(vel):
            # Reconstruct slope in px/frame from vel_ms
            try:
                px_per_mm = float(frames_df.get('px_per_mm', [9.167]).iloc[0]) \
                            if 'px_per_mm' in frames_df.columns else 9.167
            except Exception:
                px_per_mm = 9.167
            slope_px_fr = vel * 1000 * px_per_mm / FPS   # m/s → px/frame
            f0 = det['frame'].iloc[0]
            f_range = np.array([det['frame'].min(), det['frame'].max()], dtype=float)
            r_pred  = float(det['arc_tip_row'].iloc[0]) + slope_px_fr * (f_range - f_range[0])
            t_pred  = frame_to_ms(f_range - f0)
            ax.plot(t_pred, r_pred, '--', color=color, lw=1.5, alpha=0.8,
                    label=f'Regression slope  ({vel:.2f} m/s)')

        ax.invert_yaxis()   # row 0 is electrode apex (top of image)
        ax.set_xlabel('Time within event (ms)')
        ax.set_ylabel('Arc-tip row (px)  down = deeper')
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    title_parts = [
        f'Event {eid:04d}  |  Mode {mode}',
        f'vel = {vel:.2f} m/s   R² = {r2:.2f}   duration = {dur_ms:.2f} ms ({dur_fr} fr)',
    ]
    if not np.isnan(nn):
        title_parts.append(
            f'norm_nucleation = {nn:.3f}   extension = {ext_px:.0f} px')

    # ── Panel 2: local velocity vs normalised position ────────────────────────
    if has_dyn:
        ax2 = axes[1]
        dyn_pos = dyn_df[dyn_df['vel_ms'] > 0].copy()
        if len(dyn_pos) > 0:
            ax2.scatter(dyn_pos['norm_pos'], dyn_pos['vel_ms'],
                        s=8, alpha=0.55, color=color)
            # Rolling median for clarity
            dyn_pos = dyn_pos.sort_values('norm_pos')
            if len(dyn_pos) >= 5:
                roll_med = (dyn_pos.set_index('norm_pos')['vel_ms']
                            .rolling(window=5, center=True, min_periods=3)
                            .mean())
                ax2.plot(roll_med.index, roll_med.values,
                         color=color, lw=2, alpha=0.85, label='Rolling mean')
            ax2.set_xlabel('Normalised arc position  (0 = nucleation,  1 = max extent)')
            ax2.set_ylabel('Local arc-tip velocity (m/s)')
            ax2.set_xlim(0, 1)
            ax2.set_ylim(bottom=0)
            ax2.legend(fontsize=8, frameon=False)
            ax2.grid(alpha=0.25)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
        else:
            ax2.text(0.5, 0.5, 'No advancing-front dynamics data',
                     ha='center', va='center', transform=ax2.transAxes,
                     color='grey', fontsize=10)
            ax2.axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('events_dir', nargs='?', default=None,
                        help='Path to events/ directory (default: auto-detect)')
    parser.add_argument('--max', type=int, default=None,
                        help='Maximum number of events to plot')
    parser.add_argument('--mode', choices=['A', 'B', 'all'], default='all',
                        help='Only plot events of given mode (default: all)')
    args = parser.parse_args()

    # ── Locate events directory ───────────────────────────────────────────────
    if args.events_dir:
        ev_dir = Path(args.events_dir)
    else:
        script_dir = Path(__file__).parent.resolve()
        candidates = [script_dir / 'events', Path.cwd() / 'events']
        ev_dir = next((p for p in candidates if p.is_dir()), None)
        if ev_dir is None:
            print("[ERROR] Could not find events/ directory. "
                  "Pass path explicitly or run event_velocity.py first.")
            sys.exit(1)

    index_path = ev_dir / 'events_index.csv'
    if not index_path.exists():
        print(f"[ERROR] Not found: {index_path}")
        sys.exit(1)

    events = pd.read_csv(index_path)
    # Back-compat: the per-event mode is now written as 'mode_eventstart_legacy'
    # (the authoritative Mode A/B lives per re-strike in re_strikes_qa.csv).
    if 'mode' not in events.columns and 'mode_eventstart_legacy' in events.columns:
        events['mode'] = events['mode_eventstart_legacy']
    print(f"Loaded {len(events)} events from {index_path}")

    # ── Filter by mode ────────────────────────────────────────────────────────
    if args.mode != 'all' and 'mode' in events.columns:
        events = events[events['mode'] == args.mode].copy()
        print(f"Filtered to Mode {args.mode}: {len(events)} events")

    if args.max is not None:
        events = events.head(args.max)
        print(f"Limited to first {len(events)} events")

    # ── Output directory ──────────────────────────────────────────────────────
    fig_dir = ev_dir / 'figures'
    fig_dir.mkdir(exist_ok=True)
    print(f"Writing figures → {fig_dir}/")

    # ── Generate per-event figures ────────────────────────────────────────────
    n_ok = n_skip = 0
    for _, row in events.iterrows():
        eid = int(row['event_id'])

        frames_path = ev_dir / f'event_{eid:04d}_frames.csv'
        dyn_path    = ev_dir / f'event_{eid:04d}_dynamics.csv'
        out_path    = fig_dir / f'event_{eid:04d}.png'

        if not frames_path.exists():
            n_skip += 1
            continue

        frames_df = pd.read_csv(frames_path)
        dyn_df    = pd.read_csv(dyn_path) if dyn_path.exists() else None

        try:
            plot_event(row, frames_df, dyn_df, out_path)
            n_ok += 1
            if n_ok % 20 == 0:
                print(f"  ... {n_ok} figures written")
        except Exception as exc:
            print(f"  [WARN] Event {eid:04d}: {exc}")
            n_skip += 1

    print(f"\nDone.  {n_ok} figures written, {n_skip} skipped → {fig_dir}/")


if __name__ == '__main__':
    main()
