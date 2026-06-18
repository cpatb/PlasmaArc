"""
run_j_co2.py  —  J-electrode (CO2) analysis pipeline, BLADE-style
==================================================================

Runs the *maintained* BLADE 11MAY26 analysis path on the J-electrode CO2
recording, so the J data is analysed "basically the same way" as BLADE.  The
only differences are the reactor geometry, which the J configuration specifies
explicitly via an interactive calibration step:

    * electrode ENDS  — each blade is a FINITE segment (tip-end near the apex,
      far-end), captured by calibrate_j_electrodes.py and enforced in
      analyze_blade.py with --finite-segments.
    * minimum GAP     — the ~2.4 mm apex gap, which sets the px/mm scale, the
      electrode-gap flash-cutoff row (--gap-row) and the Mode-A/B apex anchor.

The conversion and every downstream stage are the SAME code as run_11may26.py
(imported, not re-implemented): convert .cine → 8-bit TIFF chunks, analyze each
chunk, merge, events + re-strike Mode A/B, Fourier, 2-D local-velocity map and
paper figures.

PREREQUISITE — run the interactive calibration ONCE first
---------------------------------------------------------
    # 1) convert a first chunk so there is a frame to click on
    python run_j_co2.py --from-stage convert --max-frames 10000   # writes tiff/...0001.tiff
    # 2) click the four electrode ends (tip-end first per blade)
    python calibrate_j_electrodes.py \
        --tiff "<...>/AD_Cer_..._results/tiff/AD_Cer_..._0001.tiff" \
        --gap-mm 2.4 --arc-direction down
    # 3) run the rest
    python run_j_co2.py --from-stage analyze

Or, if you already have a frame to click on, run the calibration first and then
`python run_j_co2.py` for the full pipeline.

This is a SWEEP (nozzle x flow, with repeats) in
    C:\\Users\\compl\\OneDrive\\Desktop\\Plasma Arc\\J Electrode
Each recording is an INDEPENDENT measurement with its own dimensions, frame rate
(auto-detected per file — the HS 'trig' recordings are NOT assumed 200 kHz) and
its own electrode calibration in its <stem>_results/ folder.  Pick one with
--cine (bare file name is fine).

USAGE
-----
    cd "C:\\Users\\compl\\Github\\Plasma Arc"
    python run_j_co2.py                                   # lists the recordings
    python run_j_co2.py --cine "HS_0.4mm_1500mL_(CO2)_trig.cine" --from-stage convert
    python run_j_co2.py --cine "HS_0.4mm_1500mL_(CO2)_trig.cine" --from-stage analyze
    python run_j_co2.py --cine "...cine" --max-frames 20000   # smoke-test
    python run_j_co2.py --cine "...cine" --bright-thresh 90   # tune detection

Outputs (under  J electrode arc / <stem>_results/)
    tiff/               numbered 8-bit TIFF chunks
    blade_lines.npy     finite electrode segments (from calibrate_j_electrodes.py)
    j_geometry.json     min gap, px/mm, gap_row, finite-segment flags
    measurements.csv    merged per-frame metrics
    event_summary.csv / measurements_all_events.csv
    re_strikes_qa.csv   authoritative Mode A/B
    velocity_map_local.* + paper figures
"""

import sys, os, json, argparse
from pathlib import Path

# Reuse the EXACT BLADE conversion / merge / subprocess helpers so the J path is
# the same code, not a fork.  (run_11may26 imports Drop Impact's pycine at module
# load; that is available on the workstation.)
import run_11may26 as blade
from run_11may26 import convert_chunked, run, PX_PER_MM as BLADE_PX_PER_MM

PLASMA_ARC = Path(__file__).parent


def _resolve_j_dir():
    """Locate the J Electrode folder robustly.  Windows OneDrive Known-Folder-Move
    can put the Desktop at either ~/Desktop or ~/OneDrive/Desktop, so try both
    (and a couple of common roots) and return the first that exists."""
    home = Path.home()
    cands = [home / 'Desktop' / 'Plasma Arc' / 'J Electrode',
             home / 'OneDrive' / 'Desktop' / 'Plasma Arc' / 'J Electrode',
             Path('C:/Users/compl/Desktop/Plasma Arc/J Electrode'),
             Path('C:/Users/compl/OneDrive/Desktop/Plasma Arc/J Electrode')]
    for c in cands:
        if c.exists():
            return c
    return cands[0]   # sensible default if none are present yet


J_DIR      = _resolve_j_dir()

# No single default recording: this is a parameter sweep (nozzle x flow, with
# repeats).  Each recording is treated as an INDEPENDENT measurement with its own
# dimensions, frame rate and electrode calibration — pick one with --cine.
CINE_DEFAULT = None

# J-electrode defaults (METHODS.md §1).  px/mm and gap_row are normally taken
# from j_geometry.json written by calibrate_j_electrodes.py; these are fallbacks.
J_PX_PER_MM_DEFAULT = 9.167   # fallback only; per-recording value comes from calibration
J_ARC_DIRECTION     = 'down'  # inverted-V at top, CO2 arc extends downward
J_BRIGHT_THRESH     = 210     # chosen from arc-vs-plume local-intensity QA (qa_arc_local_intensity_j.py):
                              # arc core saturates ~255, plume sits below ~150; 210 keeps the
                              # whole arc and rejects the plume across the sweep.

STAGES = ['convert', 'analyze', 'merge', 'events', 'fourier', 'visualize']


def load_geometry(out_root):
    """Read j_geometry.json + the electrode definition.  Supports BOTH geometries:
      • straight blades  -> blade_lines.npy   (calibrate_j_electrodes.py)
      • curved J-hooks    -> electrode_edges.npz (detect_j_hooks.py)
    Returns a dict (with _blade_lines and/or _electrode_edges paths) or None if no
    calibration has been done yet."""
    geom_path  = out_root / 'j_geometry.json'
    blade_path = out_root / 'blade_lines.npy'
    edges_path = out_root / 'electrode_edges.npz'
    if not geom_path.exists() or not (blade_path.exists() or edges_path.exists()):
        return None
    with open(geom_path) as f:
        g = json.load(f)
    g['_blade_lines']     = blade_path if blade_path.exists() else None
    g['_electrode_edges'] = edges_path if edges_path.exists() else None
    return g


def read_calibration_txt(tiff_dir):
    """Parse px/mm from a hand-written tiff/calibration.txt. Accepts lines like
    '21 px = 2.4 mm' or '8.75 px/mm'. Returns float px/mm or None."""
    import re
    p = Path(tiff_dir) / 'calibration.txt'
    if not p.exists():
        return None
    txt = p.read_text()
    # Prefer the physical 'N px = M mm' statement — most reliable, and avoids
    # mis-reading a value embedded in an expression like '30/2.4 px/mm = 12.5'.
    m = re.search(r'([0-9]*\.?[0-9]+)\s*(?:px|pixels)\s*=\s*([0-9]*\.?[0-9]+)\s*mm', txt)
    if m and float(m.group(2)) > 0:
        return float(m.group(1)) / float(m.group(2))
    # else the RESULT of an explicit '= X px/mm'
    m = re.search(r'=\s*([0-9]*\.?[0-9]+)\s*px\s*/\s*mm', txt)
    if m:
        return float(m.group(1))
    # else a bare 'X px/mm'
    m = re.search(r'([0-9]*\.?[0-9]+)\s*px\s*/\s*mm', txt)
    if m:
        return float(m.group(1))
    return None


def read_threshold_txt(tiff_dir):
    """Parse a per-recording detection threshold from tiff/threshold_suggested.txt.

    qa_arc_local_intensity_j.py writes this file when its arc/plume separation
    analysis finds a clean gap, so the detection threshold is data-driven and
    auditable per recording rather than a single sweep-wide constant.  Accepts a
    bare integer or a 'threshold = N' / 'bright_thresh: N' line.  Returns int or
    None."""
    import re
    p = Path(tiff_dir) / 'threshold_suggested.txt'
    if not p.exists():
        return None
    txt = p.read_text()
    m = re.search(r'(?:thresh(?:old)?|bright[_-]?thresh)\s*[:=]\s*([0-9]+)', txt, re.I)
    if m:
        return int(m.group(1))
    m = re.search(r'\b([0-9]{2,3})\b', txt)   # else a bare DN value
    return int(m.group(1)) if m else None


def analyze_chunks_j(tiff_paths, out_root, px_per_mm, bright_thresh, gap_row,
                     blade_lines, finite_segments, seg_margin_px, fps,
                     max_arc_px=None, electrode_edges=None,
                     flash_arc_width=None, flash_eff_width=None,
                     arc_direction=J_ARC_DIRECTION):
    """Run analyze_blade.py on each chunk with the J geometry flags.

    Mirrors run_11may26.analyze_chunks but additionally passes --finite-segments
    / --segment-margin-px, a J --bright-thresh, and the per-recording --fps (the
    HS 'trig' recordings are NOT assumed to be 200 kHz).
    """
    out_dirs = []
    for tp in tiff_paths:
        tp      = Path(tp)
        out_dir = tp.parent.parent / (tp.stem + '_results')
        ckpt    = out_dir / 'measurements.csv'
        if ckpt.exists() and ckpt.stat().st_size > 100:
            print('  [skip] %s (measurements.csv exists)' % tp.name)
        else:
            cmd = [PLASMA_ARC / 'analyze_blade.py', str(tp),
                   '--out-dir',       str(out_dir),
                   '--fps',           str(fps),
                   '--px-per-mm',     str(px_per_mm),
                   '--bright-thresh', str(bright_thresh),
                   '--arc-direction', arc_direction]
            if max_arc_px is not None:
                cmd += ['--max-arc-px', str(int(max_arc_px))]
            if electrode_edges is not None:                   # curved J-hook geometry
                cmd += ['--electrode-edges', str(electrode_edges)]
            elif blade_lines is not None:                     # straight-blade geometry
                cmd += ['--blade-lines', str(blade_lines)]
                if finite_segments:
                    cmd += ['--finite-segments',
                            '--segment-margin-px', str(seg_margin_px)]
            if gap_row is not None:                           # flash cutoff optional
                cmd += ['--gap-row', str(gap_row)]
            if flash_arc_width is not None:                   # wide CO2 arc: raise from 16 px
                cmd += ['--flash-arc-width', str(flash_arc_width)]
            if flash_eff_width is not None:
                cmd += ['--flash-eff-width', str(flash_eff_width)]
            run(cmd, label='analyze_blade.py [J]  %s' % tp.name)
        out_dirs.append(out_dir)
    return out_dirs


def merge_csvs_fps(out_dirs, merged_csv, frames_per_tiff, recording_stem, fps):
    """Like run_11may26.merge_csvs but offsets time_s with the per-recording fps
    instead of a hardcoded 200 kHz (the HS 'trig' recordings may differ)."""
    import pandas as pd
    dfs = []
    for k, d in enumerate(out_dirs):
        csv = Path(d) / 'measurements.csv'
        if not csv.exists():
            print('  [WARN] missing %s — skipping chunk %d' % (csv, k))
            continue
        df = pd.read_csv(csv)
        df['file']   = recording_stem
        df['frame']  = df['frame']  + k * frames_per_tiff
        df['time_s'] = df['time_s'] + k * frames_per_tiff / float(fps)
        dfs.append(df)
    if not dfs:
        print('[ERROR] no chunk CSVs found to merge'); sys.exit(1)
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(merged_csv, index=False)
    print('  Merged %d chunks → %s  (%d frames, fps=%d)' % (
          len(dfs), merged_csv, len(merged), fps))


def write_run_metadata(out_root, cine_path, n_frames, H, W, fps,
                       px_per_mm, pxmm_src, bright_thresh, bt_src, max_arc_px,
                       use_geom, gap_row, finite, seg_marg, blade_lines, args,
                       sigma_pxmm_rel=None, sigma_fps_rel=None):
    """Write run_metadata.json capturing every calibration/parameter actually
    used, plus the code version and timestamp, so each result is reproducible
    and self-documenting.  fourier_analysis.py already looks for a sibling
    metadata; this is the authoritative provenance record for the J pipeline."""
    import subprocess, datetime
    try:
        code_version = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=str(PLASMA_ARC),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        code_version = 'unknown'
    meta = dict(
        recording=cine_path.name,
        n_frames=int(n_frames), height=int(H), width=int(W),
        fps=int(fps),
        px_per_mm=round(float(px_per_mm), 5), px_per_mm_source=pxmm_src,
        bright_thresh=int(bright_thresh), bright_thresh_source=bt_src,
        max_arc_px=int(max_arc_px),
        arc_direction=J_ARC_DIRECTION,
        geometry_mode='full' if use_geom else 'first-pass (--no-geometry)',
        gap_row=(int(gap_row) if gap_row is not None else None),
        finite_segments=bool(finite),
        segment_margin_px=int(seg_marg),
        blade_lines=(str(blade_lines) if blade_lines is not None else None),
        local_vel_n=int(args.local_vel_n),
        flash_arc_width=args.flash_arc_width,
        restrike_method=args.restrike_method,
        restrike_threshold=args.restrike_threshold,
        restrike_prominence_frac=args.restrike_prominence_frac,
        freq_max_hz=float(args.freq_max),
        sigma_pxmm_rel=(None if sigma_pxmm_rel is None else float(sigma_pxmm_rel)),
        sigma_fps_rel=(None  if sigma_fps_rel  is None else float(sigma_fps_rel)),
        code_version=code_version,
        written_utc=datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
    )
    p = out_root / 'run_metadata.json'
    with open(p, 'w') as f:
        json.dump(meta, f, indent=2)
    # Also write the px_per_mm in the key:value form fourier_analysis.py reads.
    with open(out_root / 'metadata.txt', 'w') as f:
        f.write('px_per_mm: %.5f\nfps: %d\nbright_thresh: %d\n'
                % (px_per_mm, fps, bright_thresh))
    print('  provenance     : %s  (code %s, fps=%d)' % (p.name, code_version, fps))


def main():
    ap = argparse.ArgumentParser(
        description='J-electrode CO2 pipeline (BLADE-style)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--cine', type=str, default=None,
                    help='CINE file (default: auto-discover in the J electrode arc folder)')
    ap.add_argument('--from-stage', choices=STAGES, default='convert')
    ap.add_argument('--max-frames', type=int, default=None,
                    help='Process only the first N frames (smoke-test)')
    ap.add_argument('--frames-per-tiff', type=int, default=blade.FRAMES_PER_TIFF,
                    help='Frames per TIFF chunk written by the conversion stage')
    ap.add_argument('--px-per-mm', type=float, default=None,
                    help='Override px/mm (default: from j_geometry.json, else %.3f)'
                         % J_PX_PER_MM_DEFAULT)
    ap.add_argument('--bright-thresh', type=int, default=None,
                    help='Fixed DN detection threshold for analyze_blade.py. Default: '
                         'per-recording value from j_geometry.json ("bright_thresh") or '
                         'tiff/threshold_suggested.txt (written by qa_arc_local_intensity_j.py), '
                         'else the sweep fallback %d.' % J_BRIGHT_THRESH)
    ap.add_argument('--max-arc-px', type=int, default=None,
                    help='Maximum credible arc AREA (px) before a mask is rejected as a '
                         'noise flood. Default: max(3000, 10%% of frame area) so the wide '
                         'high-flow CO2 arc is not clipped (BLADE default 3000 is too small).')
    ap.add_argument('--freq-max', type=float, default=20000.0,
                    help='Upper edge (Hz) of the Fourier display range. Default 20 kHz to '
                         'capture the kHz gliding-arc cycle (the 500 Hz module default only '
                         'shows mains harmonics).')
    ap.add_argument('--gap-row', type=int, default=None,
                    help='Override electrode-gap cutoff row (default: from j_geometry.json)')
    ap.add_argument('--no-finite-segments', action='store_true',
                    help='Disable the finite-segment attachment gate (treat blades as '
                         'infinite lines, like BLADE).')
    ap.add_argument('--no-geometry', action='store_true',
                    help='FIRST-PASS mode: ignore any reactor geometry (blade_lines/'
                         'j_geometry.json) and run with no flash gap-row cutoff. '
                         'arc_complete falls back to the widest-pair heuristic. px/mm '
                         'still comes from tiff/calibration.txt if present.')
    ap.add_argument('--local-vel-n', type=int, default=10,
                    help='Sliding-window width for the local OLS velocity (Stage 4).')
    ap.add_argument('--flash-arc-width', type=float, default=None,
                    help='FWHM arc-width (px) above which a frame is flagged is_flash. Now '
                         'threaded to BOTH analyze_blade (source flag) AND event_velocity. '
                         'Default = BLADE 16 px (too small for the wide high-flow CO2 arc, '
                         'measured ~34-41 px). Use ~50 for the high-flow 1500mL runs or the '
                         'real arc is wrongly excluded and 0 events fit.')
    ap.add_argument('--flash-eff-width', type=float, default=None,
                    help='Effective width area/length (px) above which a frame is is_flash. '
                         'Threaded to analyze_blade + event_velocity. Default BLADE 20 px; '
                         'use ~45 for high-flow runs.')
    ap.add_argument('--restrike-method', choices=['abs', 'rel'], default='abs')
    ap.add_argument('--restrike-threshold', type=float, default=0.10)
    ap.add_argument('--restrike-prominence-frac', type=float, default=0.10)
    ap.add_argument('--velmap-vmax', type=float, default=None)
    ap.add_argument('--pxmm-rel-err', type=float, default=None,
                    help='Relative 1-sigma uncertainty on px/mm threaded into '
                         'event_velocity. Default from uncertainty.py (5%%); '
                         'override per-recording when hand-segmentation slop is known.')
    ap.add_argument('--fps-rel-err', type=float, default=None,
                    help='Relative 1-sigma uncertainty on fps threaded into '
                         'event_velocity / fourier_analysis. Default 1e-3.')
    args = ap.parse_args()

    # ── resolve recording ────────────────────────────────────────────────────
    # Each recording is an independent measurement; you MUST pick one with --cine
    # (accepts a full path or just the file name within the J Electrode folder).
    if not args.cine:
        cands = sorted(J_DIR.glob('*.cine'))
        print('[ERROR] no --cine given. This is a sweep — choose a recording:')
        for c in cands:
            print('    --cine "%s"' % c.name)
        sys.exit(1)
    cine_path = Path(args.cine)
    if not cine_path.exists():
        cine_path = J_DIR / args.cine            # bare file name within J_DIR
    if not cine_path.exists():
        print('[ERROR] recording not found: %s' % args.cine); sys.exit(1)

    stem     = cine_path.stem
    out_root = J_DIR / (stem + '_results')
    tiff_dir = out_root / 'tiff'
    out_root.mkdir(parents=True, exist_ok=True)

    from_idx = STAGES.index(args.from_stage)
    def active(s): return STAGES.index(s) >= from_idx

    # Per-recording frame rate + dimensions (NOT assumed 200 kHz / 256x256).
    n_frames, fps, H, W = blade.probe_cine(cine_path)

    print('\n' + '=' * 60)
    print('  J-ELECTRODE CO2 PIPELINE')
    print('  File   : %s' % cine_path.name)
    print('  Frames : %d   Dims: %dx%d   FPS: %d' % (n_frames, H, W, fps))
    print('  Out    : %s' % out_root)
    print('=' * 60)
    if fps != 200_000:
        print('  [NOTE] fps=%d (not 200 kHz) — threaded through analyze, merge, '
              'event-velocity, re-strike AND Fourier so velocities and frequencies '
              'are correct.' % fps)

    # ── Stage 1: convert ─────────────────────────────────────────────────────
    if active('convert'):
        print('\n[STAGE 1]  Convert .cine -> TIFF chunks')
        tiff_paths = convert_chunked(cine_path, tiff_dir,
                                     max_frames=args.max_frames,
                                     frames_per_tiff=args.frames_per_tiff)
    else:
        tiff_paths = sorted(tiff_dir.glob('*.tiff'))
        print('[skip] convert  (%d TIFFs already in %s)' % (len(tiff_paths), tiff_dir))
    if not tiff_paths:
        print('[ERROR] no TIFF files in %s' % tiff_dir); sys.exit(1)

    # ── px/mm resolution: CLI > j_geometry.json > tiff/calibration.txt > default
    geom = load_geometry(out_root)
    cal_pxmm = read_calibration_txt(tiff_dir)
    px_per_mm = (args.px_per_mm or (geom or {}).get('px_per_mm')
                 or cal_pxmm or J_PX_PER_MM_DEFAULT)
    pxmm_src = ('CLI' if args.px_per_mm else
                'j_geometry.json' if (geom and geom.get('px_per_mm')) else
                'tiff/calibration.txt' if cal_pxmm else 'DEFAULT')

    # ── bright_thresh resolution: CLI > j_geometry.json > threshold_suggested.txt
    #    > sweep fallback (J_BRIGHT_THRESH).  Per-recording where available so the
    #    detection threshold is data-driven and auditable, not one global constant.
    geom_bt = (geom or {}).get('bright_thresh')
    txt_bt  = read_threshold_txt(tiff_dir)
    bright_thresh = (args.bright_thresh if args.bright_thresh is not None
                     else geom_bt if geom_bt is not None
                     else txt_bt if txt_bt is not None
                     else J_BRIGHT_THRESH)
    bt_src = ('CLI' if args.bright_thresh is not None else
              'j_geometry.json' if geom_bt is not None else
              'tiff/threshold_suggested.txt' if txt_bt is not None else 'DEFAULT')

    # ── max_arc_px resolution: CLI > 10% of frame area (>= BLADE 3000).  The wide
    #    high-flow CO2 arc can exceed the BLADE 3000 px cap and be wrongly rejected.
    max_arc_px = (args.max_arc_px if args.max_arc_px is not None
                  else max(3000, int(0.10 * H * W)))

    use_geom = (geom is not None) and (not args.no_geometry)

    # Electrode geometry is REQUIRED for every plasma-arc video.  Hand-segment the
    # two electrode edges with segment_electrodes.py (nucleation-point-guided) so
    # the curved-edge attachment gate, gap-row flash cutoff and asymmetry midline
    # are anchored on the real electrodes.  --no-geometry is the explicit opt-out
    # for a quick first-pass look only (not for results).
    if active('analyze') and geom is None and not args.no_geometry:
        print('\n[ERROR] No electrode geometry for this recording — it is REQUIRED.\n'
              '        Hand-segment the two electrode edges first (one file per video):\n'
              '            python segment_electrodes.py --results "%s" --px-per-mm %.3f\n'
              '        That writes electrode_edges.npz, which this stage auto-loads.\n'
              '        (Pass --no-geometry to force a geometry-free first-pass run instead.)'
              % (out_root, px_per_mm))
        sys.exit(1)

    if active('analyze'):
        print('\n[STAGE 2]  Analyze TIFF chunks')
        print('  px/mm          : %.4f  (from %s)' % (px_per_mm, pxmm_src))
        print('  bright-thresh  : %d  (from %s)' % (bright_thresh, bt_src))
        print('  max-arc-px     : %d' % max_arc_px)
        if use_geom:
            gap_row  = args.gap_row if args.gap_row is not None else geom.get('gap_row')
            finite   = (not args.no_finite_segments) and bool(geom.get('finite_segments', True))
            seg_marg = geom.get('segment_margin_px', 6)
            blade_lines     = geom.get('_blade_lines')
            electrode_edges = geom.get('_electrode_edges')
            if electrode_edges is not None:
                print('  mode           : full geometry (curved J-hook edges)')
                print('  electrode edges: %s' % electrode_edges)
                print('  gap_row        : %s   attachment: nearest-edge-point' % gap_row)
            else:
                print('  mode           : full geometry (straight blades)')
                print('  gap_row        : %s   finite segments: %s (margin %s px)'
                      % (gap_row, finite, seg_marg))
                print('  blade segments : %s' % blade_lines)
        else:
            # First pass: no reactor geometry, no flash gap-row cutoff.
            # arc_complete falls back to the widest-pair heuristic; is_flash is
            # still computed (area/length/width) but flash-zone crossings are
            # NOT rejected. Add geometry later and re-run --from-stage analyze.
            blade_lines = None; gap_row = None; finite = False; seg_marg = 6
            electrode_edges = None
            why = 'forced by --no-geometry' if args.no_geometry else \
                  'no electrode_edges.npz/blade_lines.npy/j_geometry.json yet'
            print('  mode           : FIRST PASS — no geometry, no flash cutoff (%s)' % why)
            print('                   arc_complete uses widest-pair heuristic; '
                  'flashes flagged but not rejected.')
        if args.flash_arc_width is not None:
            print('  flash-arc-width: %.0f px (wide-CO2-arc override; default 16)' % args.flash_arc_width)
        chunk_dirs = analyze_chunks_j(
            tiff_paths, out_root, px_per_mm, bright_thresh, gap_row,
            blade_lines, finite, seg_marg, fps, max_arc_px=max_arc_px,
            electrode_edges=electrode_edges,
            flash_arc_width=args.flash_arc_width, flash_eff_width=args.flash_eff_width)
        write_run_metadata(out_root, cine_path, n_frames, H, W, fps,
                           px_per_mm, pxmm_src, bright_thresh, bt_src, max_arc_px,
                           use_geom, gap_row, finite, seg_marg, blade_lines, args,
                           sigma_pxmm_rel=args.pxmm_rel_err,
                           sigma_fps_rel=args.fps_rel_err)
    else:
        chunk_dirs = [tiff_dir.parent / (Path(tp).stem + '_results') for tp in tiff_paths]
        print('[skip] analyze  (px/mm=%.4f from %s)' % (px_per_mm, pxmm_src))

    # ── Stage 3: merge ───────────────────────────────────────────────────────
    merged_csv = out_root / 'measurements.csv'
    if active('merge'):
        print('\n[STAGE 3]  Merge per-chunk CSVs')
        merge_csvs_fps(chunk_dirs, merged_csv, args.frames_per_tiff, stem, fps)
    else:
        print('[skip] merge  (%s)' % merged_csv)
    if not merged_csv.exists():
        print('[ERROR] merged CSV missing: %s' % merged_csv); sys.exit(1)

    # ── Stage 4: events + re-strike Mode A/B ─────────────────────────────────
    if active('events'):
        print('\n[STAGE 4]  Event detection + velocity + re-strike Mode A/B')
        ev_cmd = [PLASMA_ARC / 'event_velocity.py', str(merged_csv), str(out_root),
                  '--px-per-mm', str(px_per_mm), '--fps', str(fps),
                  '--local-vel-n', str(args.local_vel_n)]
        if args.flash_arc_width is not None:
            ev_cmd += ['--flash-arc-width', str(args.flash_arc_width)]
        if args.flash_eff_width is not None:
            ev_cmd += ['--flash-eff-width', str(args.flash_eff_width)]
        if args.pxmm_rel_err is not None:
            ev_cmd += ['--pxmm-rel-err', str(args.pxmm_rel_err)]
        if args.fps_rel_err is not None:
            ev_cmd += ['--fps-rel-err',  str(args.fps_rel_err)]
        run(ev_cmd, label='event_velocity.py (local N=%d)' % args.local_vel_n)
        event_csv = out_root / 'measurements_all_events.csv'
        if event_csv.exists():
            run([PLASMA_ARC / 'qa_restrike.py',
                 '--csv', str(event_csv), '--tiff-dir', str(tiff_dir),
                 '--frames-per-tiff', str(args.frames_per_tiff),
                 '--arc-direction', J_ARC_DIRECTION, '--px-per-mm', str(px_per_mm),
                 '--fps', str(fps),
                 '--method', args.restrike_method,
                 '--threshold', str(args.restrike_threshold),
                 '--ext-prominence-frac', str(args.restrike_prominence_frac),
                 '--out', str(out_root / 'qa_restrike.png')],
                label='qa_restrike.py (Mode A/B)')
        else:
            print('  [skip] re-strike Mode A/B — events CSV missing')

    # ── Stage 5: Fourier ─────────────────────────────────────────────────────
    if active('fourier'):
        print('\n[STAGE 5]  Fourier analysis')
        fourier_csv = out_root / 'measurements_all_events.csv'
        if not fourier_csv.exists():
            fourier_csv = merged_csv
        run([PLASMA_ARC / 'fourier_analysis.py', str(fourier_csv), str(out_root),
             '--px-per-mm', str(px_per_mm), '--fps', str(fps),
             '--freq-max', str(args.freq_max)], label='fourier_analysis.py')

    # ── Stage 6: visualization ───────────────────────────────────────────────
    if active('visualize'):
        print('\n[STAGE 6]  Visualization')
        event_csv = out_root / 'measurements_all_events.csv'
        viz_csv   = event_csv if event_csv.exists() else merged_csv
        if event_csv.exists():
            run([PLASMA_ARC / 'build_velocity_map.py', '--csv', str(event_csv),
                 '--tiff-dir', str(tiff_dir), '--frames-per-tiff', str(args.frames_per_tiff),
                 '--out-dir', str(out_root), '--px-per-mm', str(px_per_mm),
                 '--thresh', str(bright_thresh), '--max-arc-px', str(int(max_arc_px))],
                label='build_velocity_map.py (local-velocity 2-D map)')
        else:
            print('  [skip] local velocity map — no events CSV')
        viz_cmd = [PLASMA_ARC / 'visualize_blade.py', str(viz_csv), str(out_root),
                   '--px-per-mm', str(px_per_mm), '--arc-direction', J_ARC_DIRECTION]
        if args.velmap_vmax is not None:
            viz_cmd += ['--velmap-vmax', str(args.velmap_vmax)]
        run(viz_cmd, label='visualize_blade.py')

    print('\nDone.  Results: %s' % out_root)


if __name__ == '__main__':
    main()
