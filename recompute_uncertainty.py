"""
recompute_uncertainty.py — derive vel_ms_err_total from existing artifacts.

Adds vel_ms_err_stat / vel_ms_err_sys / vel_ms_err_total columns to
event_summary.csv WITHOUT rerunning the (expensive) pipeline. Mathematically
equivalent to having run the modified event_velocity.py against the same
calibration — uncertainty is a pure derived column.

Reads ``<results_dir>/event_summary.csv`` + ``<results_dir>/run_metadata.json``
and writes ``<results_dir>/event_summary_with_unc.csv``. Refuses to compute
if the metadata is missing ``fps`` or ``px_per_mm``.

Usage:
    python recompute_uncertainty.py --results "<dir>"
    python recompute_uncertainty.py --root "<J Electrode/>"
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

import uncertainty


def recompute_one(results_dir, sigma_pxmm_rel=None, sigma_fps_rel=None,
                  verbose=True):
    """Recompute uncertainty for a single _results/ dir. Returns True on success."""
    results_dir = Path(results_dir)
    es_csv      = results_dir / 'event_summary.csv'
    meta_json   = results_dir / 'run_metadata.json'

    if not es_csv.exists():
        if verbose:
            print(f"  [skip] {results_dir.name}: no event_summary.csv")
        return False
    if not meta_json.exists():
        print(f"  [ERROR] {results_dir.name}: missing run_metadata.json — refusing "
              f"to default fps/px_per_mm.")
        return False
    meta = json.loads(meta_json.read_text())
    if 'fps' not in meta or 'px_per_mm' not in meta:
        print(f"  [ERROR] {results_dir.name}: run_metadata.json missing fps/px_per_mm.")
        return False

    df = pd.read_csv(es_csv)
    if 'vel_ms' not in df.columns:
        if verbose:
            print(f"  [skip] {results_dir.name}: event_summary.csv has no vel_ms.")
        return False

    # If the metadata already records a per-recording sigma budget (because
    # event_velocity.py was invoked with --pxmm-rel-err / --fps-rel-err and
    # run_j_co2.py logged it), prefer that over the CLI default — keeps the
    # post-hoc recomputation consistent with whatever the pipeline used.
    # Use `is not None` (not `or`) so an explicit 0.0 in the metadata — a
    # deliberately-suppressed systematic — is honoured instead of silently
    # falling back to the 5%/0.1% default.
    def _pick(cli, key, default):
        if cli is not None:
            return cli
        m = meta.get(key)
        return m if m is not None else default
    s_pxmm = float(_pick(sigma_pxmm_rel, 'sigma_pxmm_rel', uncertainty.SIGMA_PXMM_REL))
    s_fps  = float(_pick(sigma_fps_rel,  'sigma_fps_rel',  uncertainty.SIGMA_FPS_REL))

    v   = df['vel_ms'].to_numpy(dtype=float)
    sst = (df['vel_ms_err'].to_numpy(dtype=float)
           if 'vel_ms_err' in df.columns
           else np.full_like(v, np.nan))
    sst = np.where(np.isfinite(sst), sst, 0.0)
    s_sys   = np.sqrt((v * s_pxmm) ** 2 + (v * s_fps) ** 2)
    s_total = np.sqrt(sst ** 2 + s_sys ** 2)

    df['vel_ms_err_stat']  = np.round(sst,     4)
    df['vel_ms_err_sys']   = np.round(s_sys,   4)
    df['vel_ms_err_total'] = np.round(s_total, 4)

    out = results_dir / 'event_summary_with_unc.csv'
    df.to_csv(out, index=False)

    if verbose:
        med_v   = float(np.nanmedian(v[v > 0])) if (v > 0).any() else float('nan')
        med_sys = float(np.nanmedian(s_sys[v > 0])) if (v > 0).any() else float('nan')
        med_tot = float(np.nanmedian(s_total[v > 0])) if (v > 0).any() else float('nan')
        print(f"  [ok]   {results_dir.name}: n={len(df)}  median v={med_v:.2f} m/s  "
              f"sys={med_sys:.3f}  total={med_tot:.3f}  "
              f"(sigma_pxmm={s_pxmm*100:.2f}%, sigma_fps={s_fps*100:.3f}%)")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--results', type=str, help='Single <stem>_results directory')
    g.add_argument('--root',    type=str, help='Root containing *_results dirs '
                                               '(e.g. J Electrode/)')
    ap.add_argument('--pxmm-rel-err', type=float, default=None,
                    help='Override relative px/mm sigma (default from metadata or %.4f)'
                         % uncertainty.SIGMA_PXMM_REL)
    ap.add_argument('--fps-rel-err',  type=float, default=None,
                    help='Override relative fps sigma (default from metadata or %.4f)'
                         % uncertainty.SIGMA_FPS_REL)
    args = ap.parse_args()

    if args.results:
        targets = [Path(args.results)]
    else:
        root    = Path(args.root)
        # Plain glob; ALL *_results dirs, including the per-chunk ones, are fine
        # to recompute — they just won't have event_summary if not run.
        targets = sorted(d for d in root.glob('*_results') if d.is_dir())

    if not targets:
        print(f"[ERROR] no results directories found in {args.root or args.results}")
        sys.exit(1)

    print(f"Recomputing uncertainty for {len(targets)} directory(ies):")
    n_ok = 0
    for d in targets:
        if recompute_one(d, sigma_pxmm_rel=args.pxmm_rel_err,
                            sigma_fps_rel=args.fps_rel_err):
            n_ok += 1
    print(f"\nDone. {n_ok} / {len(targets)} updated.")


if __name__ == '__main__':
    main()
