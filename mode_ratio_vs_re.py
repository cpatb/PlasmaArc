"""
mode_ratio_vs_re.py — Mode A vs Mode B re-strike ratio as a function of Re.

Companion to reynolds_comparison.py (Mode-B fraction) but expressed as the
explicit A/B count ratio (and log10 of that ratio), which is the natural way
to say "is this recording more A-like or more B-like?".

Source data: re_strikes_qa.csv per recording (one row per re-strike), with
the 'mode' column already classified A or B by qa_restrike.py.

Definitions:
    R_AB    = N_A / N_B
    log10_R = log10(R_AB)
        log10_R > 0  -> more A re-strikes than B (gap-anchored)
        log10_R < 0  -> more B re-strikes than A (back-breakdown dominated)
    sigma on R_AB propagates from the Poisson SEs on N_A and N_B
    (sqrt-counting; no correlation correction, which is conservative here).

Outputs (in <root>/Reynolds_comparison/):
    mode_ratio_per_recording.csv
    mode_ratio_vs_re.{pdf,png,txt}      A/B ratio (log scale) vs Re
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import figstyle
import plot_config
import reynolds_comparison as rc
import sweep_utils


def per_recording_row(results_dir, sigma_d_rel, sigma_q_rel):
    meta_path = results_dir / 'run_metadata.json'
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    cond = rc.parse_condition(meta.get('recording', results_dir.name))
    if cond is None:
        return None
    d_mm, q_mLpm, is_rep = cond
    re_out = rc.compute_re(d_mm, q_mLpm, sigma_d_rel=sigma_d_rel,
                                          sigma_q_rel=sigma_q_rel)

    p = results_dir / 're_strikes_qa.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if 'mode' not in df.columns or len(df) == 0:
        return None
    n_A = int((df['mode'] == 'A').sum())
    n_B = int((df['mode'] == 'B').sum())
    n   = n_A + n_B
    if n == 0:
        return None

    # Poisson SE for each count, then propagate to R_AB = N_A/N_B
    s_NA = np.sqrt(max(n_A, 1))
    s_NB = np.sqrt(max(n_B, 1))
    if n_B == 0 or n_A == 0:
        R_AB = np.nan; sigma_R = np.nan
        log10_R = np.nan; sigma_log10_R = np.nan
    else:
        R_AB = n_A / n_B
        # (sigma_R/R)^2 = (sigma_NA/N_A)^2 + (sigma_NB/N_B)^2
        rel_sq  = (s_NA / n_A) ** 2 + (s_NB / n_B) ** 2
        sigma_R = R_AB * np.sqrt(rel_sq)
        log10_R       = float(np.log10(R_AB))
        sigma_log10_R = float(np.sqrt(rel_sq) / np.log(10.0))

    return dict(
        recording=meta.get('recording', results_dir.name),
        d_mm=d_mm, q_mLpm=q_mLpm, is_repeat=int(is_rep),
        Re=re_out['Re'], sigma_Re=re_out['sigma_Re'], Ma=re_out['Ma'],
        n_A=n_A, n_B=n_B, n_restrikes=n,
        R_AB=R_AB, sigma_R_AB=sigma_R,
        log10_R_AB=log10_R, sigma_log10_R_AB=sigma_log10_R,
        frac_B=n_B / n,
    )


def aggregate(df):
    """Per (d, Q) condition, sum counts across repeats and recompute the ratio."""
    rows = []
    for (d, q), grp in df.groupby(['d_mm', 'q_mLpm']):
        nA = int(grp['n_A'].sum()); nB = int(grp['n_B'].sum())
        n  = nA + nB
        if n == 0:
            continue
        if nA == 0 or nB == 0:
            R_AB = np.nan; sigma_R = np.nan
            log10_R = np.nan; sigma_log10_R = np.nan
        else:
            R_AB = nA / nB
            rel  = np.sqrt(1.0 / nA + 1.0 / nB)   # Poisson
            sigma_R       = R_AB * rel
            log10_R       = float(np.log10(R_AB))
            sigma_log10_R = float(rel / np.log(10.0))
        rows.append(dict(
            d_mm=d, q_mLpm=q, n_repeats=len(grp),
            Re=float(grp['Re'].mean()),
            sigma_Re=float(grp['sigma_Re'].mean()),
            Ma=float(grp['Ma'].mean()),
            n_A=nA, n_B=nB, n_restrikes=n,
            R_AB=R_AB, sigma_R_AB=sigma_R,
            log10_R_AB=log10_R, sigma_log10_R_AB=sigma_log10_R,
        ))
    return pd.DataFrame(rows).sort_values('Re')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', required=True, type=str)
    ap.add_argument('--out',  type=str, default=None)
    ap.add_argument('--sigma-d-rel', type=float, default=rc.SIGMA_D_REL_DEFAULT)
    ap.add_argument('--sigma-q-rel', type=float, default=rc.SIGMA_Q_REL_DEFAULT)
    args = ap.parse_args()
    root    = Path(args.root)
    out_dir = Path(args.out) if args.out else (root / 'Reynolds_comparison')
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [d for d in sweep_utils.iter_sweep_dirs(root)
               if (d / 're_strikes_qa.csv').exists()]
    rows = []
    for d in targets:
        r = per_recording_row(d, args.sigma_d_rel, args.sigma_q_rel)
        if r is None:
            print(f"  [skip] {d.name}")
            continue
        rows.append(r)
        print(f"  [ok]   {d.name}: d={r['d_mm']:.2f}mm Q={r['q_mLpm']:.0f}mL/min  "
              f"N_A={r['n_A']:4d} N_B={r['n_B']:4d}  "
              f"R_AB={r['R_AB']:.3f}+-{r['sigma_R_AB']:.3f}  "
              f"log10={r['log10_R_AB']:+.3f}  Re={r['Re']:7.0f}")
    if not rows:
        print('[ERROR] no recordings with re_strikes_qa.csv found.')
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'mode_ratio_per_recording.csv', index=False)
    df_agg = aggregate(df)
    df_agg.to_csv(out_dir / 'mode_ratio_summary.csv', index=False)
    print('\nAggregated per (d, Q):')
    print(df_agg[['d_mm', 'q_mLpm', 'n_repeats', 'n_A', 'n_B',
                  'R_AB', 'sigma_R_AB', 'log10_R_AB', 'Re']].to_string(index=False))

    # ── plot: log10(R_AB) vs Re, per-recording (faint) and per-condition (bold)
    figstyle.apply()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    for d_mm, g in df.groupby('d_mm'):
        col = plot_config.nozzle_color(d_mm)
        ax.errorbar(g['Re'], g['log10_R_AB'],
                    xerr=g['sigma_Re'], yerr=g['sigma_log10_R_AB'],
                    fmt='o', ms=6, capsize=3, mfc='none',
                    color=col, mec=col, alpha=0.55,
                    label=None)
    for d_mm, g in df_agg.groupby('d_mm'):
        col = plot_config.nozzle_color(d_mm)
        ax.errorbar(g['Re'], g['log10_R_AB'],
                    xerr=g['sigma_Re'], yerr=g['sigma_log10_R_AB'],
                    fmt='s', ms=10, capsize=4,
                    color=col, mec='#000000', mew=0.6,
                    label=f'd = {d_mm:g} mm  (combined)')

    ax.axhline(0.0, color='#000000', lw=0.6,
                label='log10(R) = 0  (equal A and B)')
    ax.set_xlabel('Reynolds number (CO₂ at 295 K, 101 kPa)')
    ax.set_ylabel('log10( N_A / N_B )')
    # Show the raw ratio on a secondary axis for intuition
    sec = ax.secondary_yaxis('right',
        functions=(lambda y: 10 ** y, lambda r: np.log10(np.maximum(r, 1e-9))))
    sec.set_ylabel('N_A / N_B  (raw ratio)')
    ax.legend(frameon=False, fontsize=9, loc='best',
              title='Nozzle diameter', title_fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title('Mode A / Mode B re-strike ratio vs Re', fontsize=10)
    fig.tight_layout()

    cap = ('Per re-strike Mode A/B classification from qa_restrike.py '
           '(authoritative per-restrike, NOT the legacy per-event start '
           'classifier).\n\n'
           'Open circles: each recording individually.\n'
           'Filled squares: counts summed across repeats per (d, Q) condition.\n'
           'Error bars: Poisson sqrt(N_A) and sqrt(N_B) propagated into log10 '
           'space.\n\n'
           'Mode A = re-initiation at the apex gap (gap-anchored).\n'
           'Mode B = re-initiation away from the apex (back-breakdown / '
           'partial re-ignition).\n\n'
           'CONFOUND: changing nozzle diameter changes both Re AND reactor '
           'geometry; a trend along the Re axis conflates fluid-dynamic and '
           'geometric effects. CO2 properties at T=295.15 K, P=101.325 kPa: '
           'rho=1.815 kg/m^3 (CRC), mu=1.48e-5 Pa.s (Lemmon & Jacobsen 2004).')
    figstyle.save(fig, str(out_dir / 'mode_ratio_vs_re'), cap)
    figstyle.verify(str(out_dir / 'mode_ratio_vs_re'))
    print(f"\nWrote: {out_dir / 'mode_ratio_vs_re'}.pdf / .png / .txt")


if __name__ == '__main__':
    main()
