"""
long_video_probe.py — go / no-go decision tool for processing a long .cine.

Before committing 12+ hours of compute on a 100+ GB recording, extract 3
widely-spaced 10 000-frame chunks (first / middle / last by default), run
analyze_blade + an inline event regression on each, and compare the headline
metrics across the chunks. If the three chunks agree within their per-chunk
statistical errors, full processing won't reveal a drift the existing data
missed -- STOP. If they diverge, there is a real long-timescale signal to
extract -- GO.

The probe uses pycine random-access (run_11may26._read_frames_at) so it
does NOT pre-convert the whole file. Disk + RAM cost is the 3 chunks only.

Calibration: by default, the probe looks for a sibling ``<stem>_results/``
directory next to the .cine and reads its ``j_geometry.json`` /
``run_metadata.json`` / ``electrode_edges.npz`` if present. Override any of
the detection parameters with CLI flags.

Outputs (in ``<cine>.parent/<stem>_probe/``):
    chunk_<k>/                       per-chunk TIFF + measurements.csv
    long_video_probe.csv             one row per chunk (stats)
    long_video_probe.{pdf,png,txt}   3-panel comparison figure + verdict

Usage:
    python long_video_probe.py --cine "<path to the long .cine>"
    python long_video_probe.py --cine "..."  --chunk-size 10000
    python long_video_probe.py --cine "..."  --chunks 0,500000,1700000
"""
from __future__ import annotations
import argparse, json, subprocess, sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import figstyle


PLASMA_ARC = Path(__file__).parent
CHUNK_SIZE_DEFAULT = 10_000


def _load_pycine_helpers():
    """Import the same pycine random-access path the production conversion uses.
    Done lazily so the probe still parses --help without pycine installed."""
    sys.path.insert(0, str(PLASMA_ARC))
    import run_11may26 as blade
    from run_11may26 import (probe_cine, _read_frames_at,
                              correct_bad_pixels_stack,
                              N_PERCENTILE_SAMPLE, SATURATED)
    from pycine.file import read_header
    return dict(blade=blade, probe_cine=probe_cine,
                _read_frames_at=_read_frames_at,
                correct_bad_pixels_stack=correct_bad_pixels_stack,
                read_header=read_header,
                N_PERCENTILE_SAMPLE=N_PERCENTILE_SAMPLE,
                SATURATED=SATURATED)


def normalisation_params(cine_path, n_frames, helpers):
    """Sample frames evenly across the WHOLE recording and return (lo, hi).
    Matches the production normalisation in run_11may26.convert_chunked so
    the per-chunk TIFFs the probe writes are directly comparable to a future
    full-file conversion."""
    header = helpers['read_header'](str(cine_path))
    N = helpers['N_PERCENTILE_SAMPLE']
    idx = np.linspace(0, n_frames - 1, N, dtype=int).tolist()
    raw = list(helpers['_read_frames_at'](cine_path, idx, header))
    corrected = helpers['correct_bad_pixels_stack'](np.stack(raw, 0))
    SAT = helpers['SATURATED']
    lo = float(np.percentile(corrected, SAT / 2))
    hi = float(np.percentile(corrected, 100.0 - SAT / 2))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi, header


def extract_chunk_to_tiff(cine_path, start_frame, n_frames, lo, hi, out_tiff,
                          helpers, header):
    """Pull frames [start_frame, start_frame + n_frames) out of the .cine via
    random-access and write a single 8-bit TIFF identical in format to what
    run_11may26.convert_chunked would have produced."""
    import tifffile
    indices = list(range(int(start_frame), int(start_frame) + int(n_frames)))
    raw = list(helpers['_read_frames_at'](cine_path, indices, header))
    corrected = helpers['correct_bad_pixels_stack'](np.stack(raw, 0))
    scale = 255.0 / (hi - lo)
    with tifffile.TiffWriter(str(out_tiff), bigtiff=False) as tw:
        for frame in corrected:
            tw.write(np.clip((frame - lo) * scale, 0, 255).astype(np.uint8),
                     photometric='minisblack', contiguous=True)


def discover_calibration(cine_path):
    """Try to find a sibling <stem>_results/ directory with calibration."""
    parent = cine_path.parent
    stem   = cine_path.stem
    candidates = [parent / (stem + '_results'),
                  parent / (stem + '_results') / 'tiff' / 'calibration.txt']
    res = parent / (stem + '_results')
    meta = res / 'run_metadata.json'
    geom = res / 'j_geometry.json'
    edges = res / 'electrode_edges.npz'
    blade_lines = res / 'blade_lines.npy'
    out = dict(found=res.exists())
    if meta.exists():
        m = json.loads(meta.read_text())
        out['fps']           = int(m.get('fps', 200000))
        out['px_per_mm']     = float(m.get('px_per_mm', 9.167))
        out['bright_thresh'] = int(m.get('bright_thresh', 210))
        out['gap_row']       = m.get('gap_row')
        out['arc_direction'] = m.get('arc_direction', 'down')
        out['max_arc_px']    = int(m.get('max_arc_px', 6553))
        out['flash_arc_width'] = m.get('flash_arc_width')
    if geom.exists():
        g = json.loads(geom.read_text())
        out['px_per_mm']     = float(g.get('px_per_mm', out.get('px_per_mm', 9.167)))
    if edges.exists():
        out['electrode_edges'] = edges
    elif blade_lines.exists():
        out['blade_lines'] = blade_lines
    return out


def run_analyze_blade(tiff_path, out_dir, fps, px_per_mm, bright_thresh,
                       max_arc_px, arc_direction, gap_row, electrode_edges,
                       blade_lines, flash_arc_width):
    """Subprocess-call the production analyze_blade.py on a single TIFF."""
    cmd = [sys.executable, str(PLASMA_ARC / 'analyze_blade.py'), str(tiff_path),
            '--out-dir',       str(out_dir),
            '--fps',           str(fps),
            '--px-per-mm',     str(px_per_mm),
            '--bright-thresh', str(bright_thresh),
            '--arc-direction', str(arc_direction),
            '--max-arc-px',    str(int(max_arc_px))]
    if electrode_edges is not None:
        cmd += ['--electrode-edges', str(electrode_edges)]
    elif blade_lines is not None:
        cmd += ['--blade-lines', str(blade_lines)]
    if gap_row is not None:
        cmd += ['--gap-row', str(int(gap_row))]
    if flash_arc_width is not None:
        cmd += ['--flash-arc-width', str(flash_arc_width)]
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8']       = '1'
    print('   $ ' + ' '.join([str(c) for c in cmd[1:]]))
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise RuntimeError(f'analyze_blade.py failed on {tiff_path.name}')


def inline_event_stats(meas_csv, fps, px_per_mm):
    """Replicate the advancing-front event regression from event_velocity.py
    on a single chunk's measurements.csv. Returns median vel_ms and a few
    summary stats used by the probe figure."""
    NUCLEATION_SLACK_FRAC = 0.15
    MIN_EVENT_FRAMES = 5

    df = pd.read_csv(meas_csv)
    df = df[(df['arc_detected'] == 1) & (df['arc_complete'] == 1)
           & (df['is_flash'] == 0) & df['arc_tip_row'].notna()]
    if len(df) == 0:
        return dict(n_valid_frames=0, n_events_fitted=0,
                    median_v_ms=float('nan'), SE_median_v=float('nan'),
                    mode_A_frac=float('nan'), SE_modeA=float('nan'),
                    median_arclen_asym=float('nan'), SE_arclen_asym=float('nan'),
                    median_abs_arclen_asym=float('nan'),
                    detection_rate=0.0)
    df = df.sort_values('frame')
    frames = df['frame'].to_numpy(dtype=float)
    rows   = df['arc_tip_row'].to_numpy(dtype=float)
    electrode_row = float(np.percentile(rows, 5))
    typical_max   = float(np.percentile(rows, 90))
    full_range    = max(typical_max - electrode_row, 1.0)
    nucl_thresh   = electrode_row + NUCLEATION_SLACK_FRAC * full_range

    # event ids
    eids = [0]; eid = 0
    for k in range(1, len(frames)):
        gap = frames[k] - frames[k-1]
        if gap > 500 or rows[k] <= nucl_thresh:
            eid += 1
        eids.append(eid)
    eids = np.array(eids)

    vels = []
    norm_nucleation = []
    for e in np.unique(eids):
        f = frames[eids == e]; r = rows[eids == e]
        keep = []; m = -np.inf
        for i, rv in enumerate(r):
            if rv >= m: keep.append(i); m = rv
        if len(keep) < MIN_EVENT_FRAMES: continue
        f2 = f[keep]; r2 = r[keep]
        slope, _ = np.polyfit(f2, r2, 1)
        v_ms = slope * fps / px_per_mm / 1000.0
        if v_ms > 0: vels.append(v_ms)
        norm_nucleation.append(
            (float(r2[0]) - electrode_row) / full_range)

    vels = np.array(vels)
    nn   = np.array(norm_nucleation)
    n_events_fitted = len(vels)
    median_v = float(np.median(vels)) if len(vels) else float('nan')
    SE_med_v = (1.253 * float(np.std(vels, ddof=1)) / np.sqrt(len(vels))
                if len(vels) > 1 else float('nan'))

    if len(nn) > 0:
        n_a = int((nn <= NUCLEATION_SLACK_FRAC).sum())
        n_t = int(len(nn))
        modeA = n_a / n_t
        SE_mA = float(np.sqrt(modeA * (1 - modeA) / n_t)) if n_t > 1 else float('nan')
    else:
        modeA = float('nan'); SE_mA = float('nan')

    asym_series = pd.to_numeric(df.get('arclen_asym'), errors='coerce').dropna()
    if len(asym_series) >= 30:
        med_a   = float(asym_series.median())
        med_aa  = float(np.median(np.abs(asym_series)))
        SE_med  = 1.253 * float(asym_series.std(ddof=1)) / np.sqrt(len(asym_series))
    else:
        med_a = med_aa = SE_med = float('nan')

    return dict(n_valid_frames=int(len(df)),
                n_events_fitted=n_events_fitted,
                median_v_ms=median_v, SE_median_v=SE_med_v,
                mode_A_frac=modeA, SE_modeA=SE_mA,
                median_arclen_asym=med_a, SE_arclen_asym=SE_med,
                median_abs_arclen_asym=med_aa,
                detection_rate=float(len(df)) /
                                (max(frames[-1] - frames[0] + 1, 1)))


MIN_EVENTS_PER_CHUNK_RELIABLE = 25


def verdict(stats_df):
    """Decide GO / CONSIDER / STOP from the per-chunk medians + their SEs.

    Criterion (per metric):
        spread_in_sigmas = (max - min) / mean(SE)
        spread > 3   -> drift signal clearly above noise  -> GO
        1 < ... <= 3 -> ambiguous                         -> CONSIDER
        spread <= 1  -> chunks indistinguishable          -> STOP
    Combined verdict is the WORST (most cautious) of the three metrics.

    GUARD: if ANY chunk has fewer than MIN_EVENTS_PER_CHUNK_RELIABLE fitted
    events, the per-chunk SE is itself too unstable to compare; the verdict
    is downgraded to UNRELIABLE and the user is told to enlarge --chunk-size.
    """
    if (stats_df['n_events_fitted'] < MIN_EVENTS_PER_CHUNK_RELIABLE).any():
        per_metric = {k: ('UNRELIABLE', float('nan'))
                       for k in ('median_v_ms', 'mode_A_frac',
                                  'median_abs_arclen_asym')}
        return 'UNRELIABLE', per_metric
    out = {}
    for key, label in [('median_v_ms',           'median event velocity (m/s)'),
                        ('mode_A_frac',          'Mode A fraction'),
                        ('median_abs_arclen_asym','|median arclen_asym|')]:
        vals = stats_df[key].to_numpy(dtype=float)
        SE_col = {'median_v_ms': 'SE_median_v',
                  'mode_A_frac': 'SE_modeA',
                  'median_abs_arclen_asym': 'SE_arclen_asym'}[key]
        SEs  = stats_df[SE_col].to_numpy(dtype=float)
        if np.isfinite(vals).sum() < 2 or np.isfinite(SEs).sum() < 1:
            out[key] = ('n/a', float('nan'))
            continue
        SE_avg = float(np.nanmean(SEs)) if np.isfinite(SEs).any() else 0.0
        if SE_avg <= 0:
            out[key] = ('n/a', float('nan'))
            continue
        spread = float(np.nanmax(vals) - np.nanmin(vals))
        spread_sigma = spread / SE_avg
        lvl = ('GO'       if spread_sigma > 3 else
               'CONSIDER' if spread_sigma > 1 else 'STOP')
        out[key] = (lvl, spread_sigma)
    overall = 'STOP'
    for v, _ in out.values():
        if v == 'GO':     overall = 'GO'
        elif v == 'CONSIDER' and overall != 'GO': overall = 'CONSIDER'
    return overall, out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cine', required=True, type=str)
    ap.add_argument('--out',  type=str, default=None,
                     help='Output dir (default: <cine_parent>/<stem>_probe)')
    ap.add_argument('--chunk-size', type=int, default=CHUNK_SIZE_DEFAULT,
                     help=f'Frames per chunk. Default {CHUNK_SIZE_DEFAULT}.')
    ap.add_argument('--chunks', type=str, default=None,
                     help='Comma-separated chunk START frames (default: first, '
                          'middle, last 10k-frame chunks across the recording).')
    # Detection params — overridable, otherwise auto-discovered.
    ap.add_argument('--px-per-mm',   type=float, default=None)
    ap.add_argument('--bright-thresh', type=int, default=None)
    ap.add_argument('--max-arc-px',  type=int, default=None)
    ap.add_argument('--gap-row',     type=int, default=None)
    ap.add_argument('--arc-direction', choices=['up', 'down'], default=None)
    ap.add_argument('--flash-arc-width', type=float, default=None)
    ap.add_argument('--electrode-edges', type=str, default=None)
    ap.add_argument('--blade-lines',     type=str, default=None)
    args = ap.parse_args()

    cine_path = Path(args.cine)
    if not cine_path.exists():
        print(f'[ERROR] not found: {cine_path}'); sys.exit(1)
    out_dir = (Path(args.out) if args.out
               else cine_path.parent / (cine_path.stem + '_probe'))
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print(f'  long_video_probe  ->  {cine_path.name}')
    print('=' * 70)

    helpers = _load_pycine_helpers()
    n_frames, fps_probe, H, W = helpers['probe_cine'](cine_path)
    print(f'  n_frames = {n_frames:,}   {H}x{W}   fps = {fps_probe}')

    cal = discover_calibration(cine_path)
    if cal.get('found'):
        print(f'  auto-discovered calibration from <stem>_results/')
    fps          = int(cal.get('fps') or fps_probe)
    px_per_mm    = float(args.px_per_mm if args.px_per_mm is not None
                          else cal.get('px_per_mm', 9.167))
    bright_thresh = int(args.bright_thresh if args.bright_thresh is not None
                          else cal.get('bright_thresh', 210))
    max_arc_px   = int(args.max_arc_px if args.max_arc_px is not None
                          else cal.get('max_arc_px', max(3000, int(0.10 * H * W))))
    gap_row      = (args.gap_row if args.gap_row is not None
                      else cal.get('gap_row'))
    arc_direction = (args.arc_direction if args.arc_direction is not None
                       else cal.get('arc_direction', 'down'))
    flash_arc_width = (args.flash_arc_width if args.flash_arc_width is not None
                        else cal.get('flash_arc_width'))
    electrode_edges = (Path(args.electrode_edges)
                       if args.electrode_edges is not None
                       else cal.get('electrode_edges'))
    blade_lines     = (Path(args.blade_lines)
                       if args.blade_lines is not None
                       else cal.get('blade_lines'))

    print(f'  fps               = {fps}')
    print(f'  px/mm             = {px_per_mm:.4f}')
    print(f'  bright_thresh     = {bright_thresh}')
    print(f'  max_arc_px        = {max_arc_px}')
    print(f'  arc_direction     = {arc_direction}')
    print(f'  gap_row           = {gap_row}')
    print(f'  flash_arc_width   = {flash_arc_width}')
    print(f'  electrode_edges   = {electrode_edges}')
    print(f'  blade_lines       = {blade_lines}')

    if args.chunks:
        starts = [int(x.strip()) for x in args.chunks.split(',')]
    else:
        cs = args.chunk_size
        starts = [0,
                   max(0, (n_frames - cs) // 2),
                   max(0, n_frames - cs)]
    labels = (['first', 'middle', 'last']
              if len(starts) == 3 else [f'chunk_{i}' for i in range(len(starts))])
    cs = args.chunk_size
    if any(s + cs > n_frames for s in starts):
        print(f'[WARN] some chunks reach past the file end; trimming.')
        starts = [min(s, n_frames - cs) for s in starts]
    print(f'\n  chunk starts: {starts}  (size {cs} each)')

    print('\n  Computing whole-file normalisation (lo, hi) ...')
    lo, hi, header = normalisation_params(cine_path, n_frames, helpers)
    print(f'    lo = {lo:.1f}   hi = {hi:.1f}')

    rows = []
    for i, s in enumerate(starts):
        chunk_dir = out_dir / f'chunk_{i:02d}_f{s:08d}'
        chunk_dir.mkdir(parents=True, exist_ok=True)
        tiff_path = chunk_dir / f'{cine_path.stem}_probe_{i:02d}.tiff'
        print(f'\n  ── Chunk {i} ({labels[i]}): frames [{s}, {s+cs})')
        if not tiff_path.exists():
            print(f'     extracting -> {tiff_path.name}')
            extract_chunk_to_tiff(cine_path, s, cs, lo, hi, tiff_path,
                                   helpers, header)
        else:
            print(f'     [skip] tiff exists')
        ckpt = chunk_dir / 'measurements.csv'
        if not ckpt.exists() or ckpt.stat().st_size < 100:
            run_analyze_blade(tiff_path, chunk_dir, fps, px_per_mm,
                              bright_thresh, max_arc_px, arc_direction,
                              gap_row, electrode_edges, blade_lines,
                              flash_arc_width)
        else:
            print(f'     [skip] measurements.csv exists')
        st = inline_event_stats(ckpt, fps, px_per_mm)
        st.update(dict(chunk_index=i, label=labels[i], start_frame=int(s),
                        end_frame=int(s + cs),
                        time_s_start=int(s) / float(fps),
                        time_s_end=int(s + cs) / float(fps)))
        rows.append(st)
        print(f'     n_events={st["n_events_fitted"]}  '
              f'median_v={st["median_v_ms"]:.2f}±{st["SE_median_v"]:.2f} m/s  '
              f'ModeA={st["mode_A_frac"]:.3f}  '
              f'|asym|={st["median_abs_arclen_asym"]:.3f}')

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'long_video_probe.csv', index=False)
    overall, per_metric = verdict(df)

    # ── Figure ─────────────────────────────────────────────────────────────
    figstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    palette = ['#00204d', '#5d7588', '#fde725']

    def panel(ax, ycol, yerr_col, ylabel, decision_key):
        for i, r in df.iterrows():
            ax.errorbar(r['time_s_start'] * 1000.0, r[ycol], yerr=r[yerr_col],
                         fmt='o', ms=10, capsize=4,
                         color=palette[int(i) % len(palette)], mec='black', mew=0.6,
                         label=r['label'])
        ax.set_xlabel('Chunk start time (ms)')
        ax.set_ylabel(ylabel)
        lvl, sigma = per_metric.get(decision_key, ('n/a', float('nan')))
        ax.set_title(f'{ylabel}\nspread = {sigma:.2f} σ  ({lvl})', fontsize=9)
        ax.grid(alpha=0.3)
    panel(axes[0], 'median_v_ms', 'SE_median_v',
           'Median event velocity (m/s)', 'median_v_ms')
    panel(axes[1], 'mode_A_frac', 'SE_modeA',
           'Mode A fraction (event-start)', 'mode_A_frac')
    panel(axes[2], 'median_abs_arclen_asym', 'SE_arclen_asym',
           '|median arclen_asym|', 'median_abs_arclen_asym')
    axes[0].legend(frameon=False, fontsize=8, loc='best')

    color = {'GO':'#C62828', 'CONSIDER':'#F57C00',
              'STOP':'#2E7D32', 'UNRELIABLE':'#9E9E9E'}.get(overall, '#000')
    fig.suptitle(f'Long-video probe — {cine_path.name}      '
                  f'VERDICT: {overall}',
                  fontsize=11, color=color)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    cap = (
        f'Three-chunk probe across a long .cine, comparing the three headline '
        f'metrics that would drive a "process the whole file" decision: median '
        f'event velocity, Mode A fraction (event-start classifier from '
        f'norm_nucleation_pos), and median |arclen_asym| (left-right arc '
        f'symmetry).\n\n'
        f'Per-metric verdict: ratio of max-min spread to the mean per-chunk SE. '
        f'>3 sigma = clear drift signal (GO); 1-3 = ambiguous (CONSIDER); '
        f'<=1 = chunks indistinguishable (STOP). Overall verdict is the worst '
        f'(most cautious) of the three.\n\n'
        f'Recording   : {cine_path.name}\n'
        f'Total frames: {n_frames:,}    chunk size: {cs:,}    '
        f'duration covered: {n_frames/fps:.3f} s\n'
        f'fps={fps}   px/mm={px_per_mm:.4f}   bright_thresh={bright_thresh}\n\n'
        f'Per-chunk verdicts:\n'
        + '\n'.join(f'  {k:30s}  spread = {sigma:.2f} sigma   ->  {lvl}'
                     for k, (lvl, sigma) in per_metric.items())
        + f'\nOverall verdict: {overall}\n\n'
        f'STOP   : full processing will not change the conclusion -- the '
        f'existing short-recording results already capture the dynamics at '
        f'all timescales the long file would reveal.\n'
        f'CONSIDER: borderline. Consider 1-2 more probes (e.g. 5 chunks '
        f'instead of 3) before committing the full processing budget.\n'
        f'GO     : at least one headline metric drifts significantly across '
        f'the recording. Full processing is justified -- run the standard '
        f'run_j_co2.py pipeline on the whole file.\n'
        f'UNRELIABLE: at least one chunk had fewer than '
        f'{MIN_EVENTS_PER_CHUNK_RELIABLE} fitted events; per-chunk SE is too '
        f'noisy to compare. Re-run the probe with a larger --chunk-size (e.g. '
        f'25000-50000) so each chunk averages over more events.')
    figstyle.save(fig, str(out_dir / 'long_video_probe'), cap)
    figstyle.verify(str(out_dir / 'long_video_probe'))

    print('\n' + '=' * 70)
    print(f'  OVERALL VERDICT: {overall}')
    for k, (lvl, sigma) in per_metric.items():
        print(f'    {k:30s}  spread = {sigma:.2f} sigma   ->  {lvl}')
    print(f'\n  Probe outputs:  {out_dir}')


if __name__ == '__main__':
    main()
