#!/usr/bin/env python3
"""
qa_modes.py  —  First-pass Mode A / Mode B characterization & QA.

Mode A : re-initiation AT / very close to the minimum electrode gap (apex).
Mode B : re-initiation AWAY from the apex (back-breakdown along the warm channel).

For EVERY primary event (not only the velocity-fitted ones), this computes the
re-initiation location as a dimensionless, calibration-free ratio:

    norm_nucleation_pos = (tip_row_start - electrode_row) / L_eff_px

where tip_row_start is the arc-tip row at the event's first frame, electrode_row
is the apex (min-gap) row, and L_eff_px is the typical arc range. An event is
Mode B if norm_nucleation_pos > --threshold, else Mode A.

It produces a QA figure to judge how well this separates events:
  A  distribution of norm_nucleation_pos with the threshold (bimodal -> good A/B
     separation; unimodal -> the split is somewhat arbitrary)
  B  threshold sensitivity: Mode-B fraction vs threshold (how robust is the call)
  C  norm_nucleation_pos vs inter-event gap, coloured by mode (true Mode B is
     expected at HIGH position AND SHORT gap — warm channel not yet cooled)
  D+ montage of the actual re-initiation frames (first frame of sampled A and B
     events) with the apex line and detected tip, so you can SEE whether the
     Mode-B events really re-struck away from the gap.

Also writes event_modes_qa.csv (one row per primary event).

Usage:
    python qa_modes.py \
        --csv  "<results>/measurements_all_events.csv" \
        --tiff "<results>/tiff/<stem>_0001.tiff" \
        --arc-direction down --px-per-mm 8.8 --threshold 0.15 \
        --out  "<results>/qa_modes.png"
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.morphology import skeletonize

from analyze_blade import binarize, skeleton_endpoints, arc_front_tip

FPS_DEFAULT = 200_000
MODE_COLOR = {'A': '#1565C0', 'B': '#C62828'}


def per_event_table(df, slack):
    """One row per primary event: re-initiation location + geometry."""
    det = df[df['arc_detected'] == 1].dropna(subset=['arc_tip_row']).copy()
    det = det[det['event_id'] >= 0]
    rows = []
    for fname, fg in det.groupby('file'):
        tip = fg['arc_tip_row'].astype(float)
        # apex (min-gap) and effective range — same definitions as event_velocity
        if 'arc_complete' in fg.columns:
            comp = fg[pd.to_numeric(fg['arc_complete'], errors='coerce').fillna(0) == 1]['arc_tip_row']
            erow = float(np.percentile(comp, 5)) if len(comp) >= 20 else float(np.percentile(tip, 5))
        else:
            erow = float(np.percentile(tip, 5))
        l_eff = max(float(np.percentile(tip, 90)) - erow, 1.0)
        for eid, g in fg.groupby('event_id'):
            g = g.sort_values('frame')
            f0 = g.iloc[0]
            tip_start = float(f0['arc_tip_row'])
            npos = float(np.clip((tip_start - erow) / l_eff, 0, 1))
            rows.append(dict(
                file=fname, event_id=int(eid),
                frame_start=int(g['frame'].iloc[0]), frame_end=int(g['frame'].iloc[-1]),
                duration=int(g['frame'].iloc[-1] - g['frame'].iloc[0]),
                n_frames=len(g),
                tip_start_px=round(tip_start, 1),
                tip_start_mm_from_apex=round((tip_start - erow), 1),
                norm_nucleation_pos=round(npos, 4),
                electrode_row=round(erow, 1), l_eff_px=round(l_eff, 1),
                arc_len_start_mm=round(float(f0['arc_length_mm']), 3)
                if 'arc_length_mm' in g.columns and pd.notna(f0['arc_length_mm']) else np.nan,
                tip_col_start=float(f0['arc_tip_col']) if 'arc_tip_col' in g.columns else np.nan,
            ))
    ev = pd.DataFrame(rows).sort_values(['file', 'frame_start']).reset_index(drop=True)
    # inter-event gap (frames since previous event ended, same file)
    ev['inter_event_gap'] = np.nan
    for fname, g in ev.groupby('file'):
        if len(g) > 1:
            gaps = g['frame_start'].values[1:] - g['frame_end'].values[:-1]
            ev.loc[g.index[1:], 'inter_event_gap'] = gaps
    ev['mode'] = np.where(ev['norm_nucleation_pos'] > slack, 'B', 'A')
    return ev


def main():
    ap = argparse.ArgumentParser(description='First-pass Mode A/B characterization QA')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--tiff', required=True)
    ap.add_argument('--arc-direction', choices=['up', 'down'], required=True)
    ap.add_argument('--px-per-mm', type=float, default=8.8)
    ap.add_argument('--fps', type=float, default=FPS_DEFAULT)
    ap.add_argument('--thresh', type=int, default=100, help='binarization DN threshold')
    ap.add_argument('--threshold', type=float, default=0.15,
                    help='norm_nucleation_pos above which an event is Mode B')
    ap.add_argument('--frame-offset', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for c in ['arc_tip_row', 'arc_tip_col', 'arc_length_mm']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    ev = per_event_table(df, args.threshold)
    if len(ev) == 0:
        raise SystemExit('No primary events found.')

    n_a = int((ev['mode'] == 'A').sum()); n_b = int((ev['mode'] == 'B').sum())
    print('Primary events: %d   Mode A: %d (%.0f%%)   Mode B: %d (%.0f%%)' % (
        len(ev), n_a, 100 * n_a / len(ev), n_b, 100 * n_b / len(ev)))
    print('mean norm_nucleation_pos  A: %.3f   B: %.3f' % (
        ev.loc[ev['mode'] == 'A', 'norm_nucleation_pos'].mean(),
        ev.loc[ev['mode'] == 'B', 'norm_nucleation_pos'].mean()))
    out_csv = Path(args.out).with_name('event_modes_qa.csv')
    ev.to_csv(out_csv, index=False)
    print('Wrote %s' % out_csv)

    # ---- figure ----
    import tifffile
    tif = tifffile.TiffFile(args.tiff)
    H, W = tif.pages[0].asarray().shape

    fig = plt.figure(figsize=(16, 9), facecolor='white')
    gs = gridspec.GridSpec(2, 6, height_ratios=[1, 1.1])

    # A. distribution
    axA = fig.add_subplot(gs[0, 0:2])
    axA.hist(ev['norm_nucleation_pos'], bins=20, color='#607D8B', edgecolor='white')
    axA.axvline(args.threshold, color='red', ls='--', lw=2, label='threshold=%.2f' % args.threshold)
    axA.set_xlabel('norm_nucleation_pos\n(0 = apex / min gap,  1 = max extent)')
    axA.legend(fontsize=8)

    # B. threshold sensitivity
    axB = fig.add_subplot(gs[0, 2:4])
    ths = np.linspace(0.05, 0.5, 40)
    fracB = [100 * (ev['norm_nucleation_pos'] > th).mean() for th in ths]
    axB.plot(ths, fracB, color='#C62828')
    axB.axvline(args.threshold, color='red', ls='--', lw=1)
    axB.set_xlabel('threshold'); axB.set_ylabel('% events Mode B')
    axB.grid(alpha=0.3)

    # C. nucleation pos vs inter-event gap
    axC = fig.add_subplot(gs[0, 4:6])
    sub = ev.dropna(subset=['inter_event_gap'])
    for mode, g in sub.groupby('mode'):
        axC.scatter(g['norm_nucleation_pos'], g['inter_event_gap'],
                    c=MODE_COLOR.get(mode), s=30, alpha=0.7, label='Mode %s' % mode)
    axC.axvline(args.threshold, color='red', ls='--', lw=1)
    axC.set_xlabel('norm_nucleation_pos'); axC.set_ylabel('inter-event gap (frames)')
    axC.legend(fontsize=8); axC.grid(alpha=0.3)

    # D. re-initiation frame montage: 3 most apex-like (A) + 3 most displaced (B)
    ev_sorted = ev.sort_values('norm_nucleation_pos')
    picks = list(ev_sorted.head(3).itertuples()) + list(ev_sorted.tail(3).itertuples())
    for j, e in enumerate(picks):
        ax = fig.add_subplot(gs[1, j])
        page = int(e.frame_start) - args.frame_offset
        erow = e.electrode_row
        raw_apex = (H - 1 - erow) if args.arc_direction == 'up' else erow
        raw_tip = (H - 1 - e.tip_start_px) if args.arc_direction == 'up' else e.tip_start_px
        if 0 <= page < len(tif.pages):
            raw = tif.pages[page].asarray()
            ax.imshow(np.power(raw / max(raw.max(), 1), 0.45), cmap='inferno', vmin=0, vmax=1)
            b = binarize(raw, 3, bright_thresh=args.thresh)
            if b.any():
                ax.contour(b, levels=[0.5], colors='cyan', linewidths=0.5)
            ax.axhline(raw_apex, color='yellow', lw=1, ls='--')
            if not np.isnan(e.tip_col_start):
                ax.plot(e.tip_col_start, raw_tip, 'x', c='red', ms=10, mew=2)
        else:
            ax.text(0.5, 0.5, 'frame %d not in TIFF' % e.frame_start, ha='center', va='center')
        ax.axis('off')
    tif.close()

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=110, facecolor='white')
    plt.close(fig)
    print('Saved %s' % args.out)
    print('Interpretation: Mode-A frames (left) should show the arc rooted AT the '
          'yellow apex line; Mode-B frames (right) should show it re-struck ABOVE it. '
          'If panel A is bimodal the split is meaningful; if unimodal, treat the '
          'threshold as a soft cut and lean on the continuous norm_nucleation_pos.')


if __name__ == '__main__':
    main()
