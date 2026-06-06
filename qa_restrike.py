#!/usr/bin/env python3
"""
qa_restrike.py  —  Re-strike–based Mode A / Mode B characterization (non-circular).

Mode A : re-initiation AT / very close to the minimum electrode gap (apex).
Mode B : re-initiation AWAY from the apex (partial re-ignition / back-breakdown).

A "re-strike" is each re-initiation of the arc — every local MINIMUM of the
arc-tip trajectory (the bottom of a retreat, where the next advance begins).
This is detected directly from the tip motion, anchored on the ARC EXTENSIONS
(prominent local maxima); the re-strike is the minimum BETWEEN two consecutive
extensions. That captures shallow partial retreats (Mode B back-breakdowns that
re-ignite before the arc falls back to the gap), which a retreat-prominence
detector would discard.

(Contrast with an "event" in event_velocity.py: a new event is only declared
when the tip returns to the apex, so events ≈ Mode-A cycles and Mode B re-strikes
are hidden inside an event. Hence Mode A/B must be measured per re-strike.)

For each re-strike we report two calibration-free measures (classify with
--method):
  abs  norm_pos     = (tip_min - apex) / L_eff       0 = apex, 1 = max extent
                      Mode B if norm_pos > --threshold
  rel  retreat_frac = (prev_extension - tip_min) / (prev_extension - apex)
                      1 = fell all the way back to the gap; 0 = no retreat
                      Mode B if retreat_frac < --rel-threshold

Outputs (next to --out): re_strikes_qa.csv and the QA figure.

Montage frames: pass a single --tiff (+ --frame-offset) OR --tiff-dir (+
--frames-per-tiff) so frames from any chunk resolve (chunk=frame//fpf).

Usage:
    python qa_restrike.py --csv ... --tiff-dir .../tiff --frames-per-tiff 10000 \
        --arc-direction down --px-per-mm 8.8 --method abs --threshold 0.10 --out qa_restrike.png
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
import tifffile

from analyze_blade import binarize

FPS_DEFAULT = 200_000
MODE_COLOR = {'A': '#1565C0', 'B': '#C62828'}


class FrameSource:
    """Fetch a raw frame by GLOBAL frame index from a single TIFF or a directory
    of ordered chunk TIFFs."""
    def __init__(self, tiff=None, tiff_dir=None, frames_per_tiff=None, frame_offset=0):
        self.fpf = frames_per_tiff
        self.frame_offset = frame_offset
        self._cache = {}
        if tiff_dir:
            self.mode = 'dir'
            self.tiffs = sorted(Path(tiff_dir).glob('*.tiff'))
        elif tiff:
            self.mode = 'single'
            self.tiffs = [Path(tiff)]
        else:
            self.mode = None
            self.tiffs = []
        if self.tiffs:
            with tifffile.TiffFile(str(self.tiffs[0])) as t:
                self.H, self.W = t.pages[0].asarray().shape
        else:
            self.H = self.W = 256

    def _open(self, k):
        if k not in self._cache:
            self._cache[k] = tifffile.TiffFile(str(self.tiffs[k]))
        return self._cache[k]

    def get(self, gframe):
        if not self.tiffs:
            return None
        if self.mode == 'dir':
            if not self.fpf:
                return None
            k = int(gframe) // self.fpf
            page = int(gframe) % self.fpf
        else:
            k = 0
            page = int(gframe) - self.frame_offset
        if k < 0 or k >= len(self.tiffs):
            return None
        try:
            tf = self._open(k)
            if page < 0 or page >= len(tf.pages):
                return None
            return tf.pages[page].asarray()
        except Exception:
            return None

    def close(self):
        for t in self._cache.values():
            try:
                t.close()
            except Exception:
                pass


def apex_and_range(tip, complete=None):
    if complete is not None and complete.sum() >= 20:
        erow = float(np.percentile(tip[complete], 5))
    else:
        erow = float(np.percentile(tip, 5))
    l_eff = max(float(np.percentile(tip, 90)) - erow, 1.0)
    return erow, l_eff


def detect_restrikes(tip, erow, l_eff, ext_prom_frac, min_cycle, smooth):
    """Anchor on extension peaks; the re-strike is the minimum between
    consecutive extensions."""
    sig = tip.astype(float)
    if smooth > 1:
        sig = uniform_filter1d(sig, size=int(smooth))
    ext_prom = max(ext_prom_frac * l_eff, 1.0)
    peaks, _ = find_peaks(sig, prominence=ext_prom, distance=max(int(min_cycle), 1))
    out = []
    for i in range(1, len(peaks)):
        a, b = peaks[i - 1], peaks[i]
        k = a + int(np.argmin(sig[a:b + 1]))
        out.append(dict(idx=k, prev_ext=float(sig[a])))
    return out, peaks, ext_prom


def classify(npos, rfrac, method, thr_abs, thr_rel):
    if method == 'rel':
        return 'B' if rfrac < thr_rel else 'A'
    return 'B' if npos > thr_abs else 'A'


def pct_line(name, v):
    if len(v) == 0:
        return f"  {name}: (none)"
    q = np.percentile(v, [10, 25, 50, 75, 90])
    return ("  %-12s p10=%.3f p25=%.3f p50=%.3f p75=%.3f p90=%.3f (min %.3f max %.3f)"
            % (name, q[0], q[1], q[2], q[3], q[4], v.min(), v.max()))


def main():
    ap = argparse.ArgumentParser(description='Re-strike-based Mode A/B characterization')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--tiff', help='single chunk TIFF for montage')
    ap.add_argument('--tiff-dir', help='directory of ordered chunk TIFFs (multi-chunk montage)')
    ap.add_argument('--frames-per-tiff', type=int, default=None,
                    help='chunk size (required with --tiff-dir for global-frame->page)')
    ap.add_argument('--arc-direction', choices=['up', 'down'], required=True)
    ap.add_argument('--px-per-mm', type=float, default=8.8)
    ap.add_argument('--fps', type=float, default=FPS_DEFAULT)
    ap.add_argument('--thresh', type=int, default=100, help='binarization DN threshold')
    ap.add_argument('--method', choices=['abs', 'rel'], default='abs')
    ap.add_argument('--threshold', type=float, default=0.10)
    ap.add_argument('--rel-threshold', type=float, default=0.70)
    ap.add_argument('--ext-prominence-frac', type=float, default=0.10)
    ap.add_argument('--min-cycle-frames', type=int, default=5)
    ap.add_argument('--smooth', type=int, default=3)
    ap.add_argument('--frame-offset', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    for c in ['arc_tip_row', 'arc_tip_col', 'arc_complete']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    det = df[(df['arc_detected'] == 1)].dropna(subset=['arc_tip_row']).sort_values(['file', 'frame'])

    recs = []
    traj = {}
    for fname, g in det.groupby('file'):
        frames = g['frame'].values.astype(int)
        tip = g['arc_tip_row'].values.astype(float)
        col = g['arc_tip_col'].values.astype(float) if 'arc_tip_col' in g.columns else np.full(len(g), np.nan)
        complete = (pd.to_numeric(g['arc_complete'], errors='coerce').fillna(0).values == 1) \
            if 'arc_complete' in g.columns else None
        erow, l_eff = apex_and_range(tip, complete)
        rk, peaks, ext_prom = detect_restrikes(tip, erow, l_eff,
                                               args.ext_prominence_frac, args.min_cycle_frames, args.smooth)
        traj[fname] = (frames, tip, erow, l_eff, [r['idx'] for r in rk], peaks)
        prev_f = None
        for r in rk:
            k = r['idx']
            npos = float(np.clip((tip[k] - erow) / l_eff, 0, 1))
            denom = max(r['prev_ext'] - erow, 1e-6)
            rfrac = float(np.clip((r['prev_ext'] - tip[k]) / denom, 0, 1))
            mode = classify(npos, rfrac, args.method, args.threshold, args.rel_threshold)
            cyc = int(frames[k] - prev_f) if prev_f is not None else -1
            recs.append(dict(file=fname, frame=int(frames[k]),
                             tip_row=round(float(tip[k]), 1),
                             tip_col=float(col[k]) if not np.isnan(col[k]) else np.nan,
                             dist_from_apex_mm=round((tip[k] - erow) / args.px_per_mm, 2),
                             prev_extension_mm=round((r['prev_ext'] - erow) / args.px_per_mm, 2),
                             norm_pos=round(npos, 4), retreat_frac=round(rfrac, 4),
                             cycle_frames=cyc, mode=mode,
                             apex_row=round(erow, 1), l_eff_px=round(l_eff, 1)))
            prev_f = frames[k]

    rs = pd.DataFrame(recs)
    if len(rs) == 0:
        raise SystemExit('No re-strikes detected — lower --ext-prominence-frac.')

    n_a = int((rs['mode'] == 'A').sum()); n_b = int((rs['mode'] == 'B').sum())
    cyc = rs[rs['cycle_frames'] > 0]['cycle_frames']
    print('Re-strikes detected: %d   Mode A: %d (%.0f%%)   Mode B: %d (%.0f%%)   [method=%s]' % (
        len(rs), n_a, 100 * n_a / len(rs), n_b, 100 * n_b / len(rs), args.method))
    print('Distributions (use these to choose the cut):')
    print(pct_line('norm_pos', rs['norm_pos'].values))
    print(pct_line('retreat_frac', rs['retreat_frac'].values))
    if len(cyc):
        print('Cycle period: mean %.0f frames (%.1f us, %.1f kHz)' % (
            cyc.mean(), cyc.mean() / args.fps * 1e6, args.fps / cyc.mean() / 1e3))
    out_csv = Path(args.out).with_name('re_strikes_qa.csv')
    rs.to_csv(out_csv, index=False)
    print('Wrote %s' % out_csv)

    # ---- figure ----
    src = FrameSource(tiff=args.tiff, tiff_dir=args.tiff_dir,
                      frames_per_tiff=args.frames_per_tiff, frame_offset=args.frame_offset)
    H, W = src.H, src.W

    fig = plt.figure(figsize=(16, 9), facecolor='white')
    gs = gridspec.GridSpec(2, 6, height_ratios=[1, 1.05])

    axA = fig.add_subplot(gs[0, 0:4])
    fname0 = max(traj, key=lambda f: len(traj[f][0]))
    frames, tip, erow, l_eff, ridx, peaks = traj[fname0]
    t_ms = frames / args.fps * 1e3
    dist_mm = (tip - erow) / args.px_per_mm
    axA.plot(t_ms, dist_mm, color='#90A4AE', lw=0.7, label='arc-tip distance from apex')
    if len(peaks):
        axA.scatter(t_ms[peaks], dist_mm[peaks], c='#2E7D32', s=18, marker='^',
                    label='extension peak', zorder=4)
    rs0 = rs[rs['file'] == fname0]
    for mode, gg in rs0.groupby('mode'):
        mk = np.clip(np.searchsorted(frames, gg['frame'].values), 0, len(frames) - 1)
        axA.scatter(t_ms[mk], dist_mm[mk], c=MODE_COLOR[mode], s=45, zorder=5,
                    label='re-strike Mode %s' % mode)
    axA.axhline(0, color='gold', lw=1, ls='--', label='apex (min gap)')
    if args.method == 'abs':
        axA.axhline(args.threshold * l_eff / args.px_per_mm, color='red', lw=1, ls=':',
                    label='A/B threshold')
    axA.set_xlabel('time (ms)'); axA.set_ylabel('tip distance from apex (mm)')
    axA.legend(fontsize=7, ncol=3); axA.grid(alpha=0.3)

    axB = fig.add_subplot(gs[0, 4:6])
    axB.hist(rs['norm_pos'], bins=20, color='#607D8B', edgecolor='white')
    if args.method == 'abs':
        axB.axvline(args.threshold, color='red', ls='--', lw=2, label='thr=%.2f' % args.threshold)
        axB.legend(fontsize=8)
    axB.set_xlabel('re-strike norm_pos (0=apex, 1=max)'); axB.set_ylabel('re-strikes')

    axC = fig.add_subplot(gs[1, 0:2])
    axC.hist(rs['retreat_frac'], bins=20, color='#8D6E63', edgecolor='white')
    if args.method == 'rel':
        axC.axvline(args.rel_threshold, color='red', ls='--', lw=2,
                    label='thr=%.2f' % args.rel_threshold)
        axC.legend(fontsize=8)
    axC.set_xlabel('retreat_frac (1=back to apex, 0=none)'); axC.set_ylabel('re-strikes')

    rs_sorted = rs.sort_values('norm_pos')
    picks = list(rs_sorted.head(2).itertuples()) + list(rs_sorted.tail(2).itertuples())
    for j, e in enumerate(picks):
        ax = fig.add_subplot(gs[1, 2 + j])
        raw_apex = (H - 1 - e.apex_row) if args.arc_direction == 'up' else e.apex_row
        raw_tip = (H - 1 - e.tip_row) if args.arc_direction == 'up' else e.tip_row
        raw = src.get(e.frame)
        if raw is not None:
            ax.imshow(np.power(raw / max(raw.max(), 1), 0.45), cmap='inferno', vmin=0, vmax=1)
            b = binarize(raw, 3, bright_thresh=args.thresh)
            if b.any():
                ax.contour(b, levels=[0.5], colors='cyan', linewidths=0.5)
            ax.axhline(raw_apex, color='yellow', lw=1, ls='--')
            if not np.isnan(e.tip_col):
                ax.plot(e.tip_col, raw_tip, 'x', c='red', ms=10, mew=2)
        else:
            ax.text(0.5, 0.5, 'frame %d\nnot available' % e.frame, ha='center', va='center')
        ax.axis('off')
    src.close()

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=110, facecolor='white')
    plt.close(fig)
    print('Saved %s' % args.out)


if __name__ == '__main__':
    main()
