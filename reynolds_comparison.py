"""
reynolds_comparison.py — Reynolds-number comparison across the J-electrode CO2
sweep.

The pipeline (event_velocity, qa_restrike, fourier_analysis) reports the arc-
side metrics; this module supplies the missing **flow-side** Reynolds axis
and aggregates everything into a single repeat-aware figure-set.

Definitions
-----------
Characteristic length    L = nozzle diameter d                  [m]
Characteristic velocity  U = bulk gas velocity v_bulk = Q / A   [m/s]
                                    where A = pi/4 d^2
Fluid                    CO2 at room conditions (T = 295.15 K, P = 101.325 kPa)
Density                  rho_CO2 = 1.815 kg/m^3         (CRC Handbook 90th ed.)
Dyn. viscosity           mu_CO2 = 1.48e-5 Pa.s         (Lemmon & Jacobsen 2004,
                                                        J Phys Chem Ref Data)

Re = rho * v_bulk * d / mu      (incompressible-flow Reynolds)

Mach caveat: v_bulk = 132.7 m/s at d=0.4 mm / Q=1000 mL/min approaches sonic
(a_CO2 ~ 270 m/s at 295 K, Ma ~ 0.49). The Re value above the small-nozzle
corner is the *incompressible* equivalent; flow is firmly compressible there.

Uncertainty
-----------
Default 1-sigma relative uncertainties (override at CLI):
    sigma_d/d   = 5 %    (machining tolerance, conservative)
    sigma_Q/Q   = 5 %    (typical MFC spec, no calibration trace on disk)
    sigma_T     = 2 K    (room-condition assumption)
    sigma_P     = 1 kPa
These propagate into v_bulk and through ideal-gas rho into Re.

Repeat-awareness
----------------
Per (d, Q) condition, repeats are aggregated:
    y_mean = mean across repeats of the per-recording median
    sigma_y = sqrt( SE_repeats^2 + sigma_sys_recording^2 )
where sigma_sys_recording is the px/mm + fps systematic that does NOT
sqrt(N)-suppress within a single calibration.

Confound (printed in every caption)
-----------------------------------
Changing nozzle diameter changes BOTH Re and reactor geometry (impingement
distance, divergence angle). The arc-velocity-vs-Re relation therefore
conflates fluid-dynamic and geometric effects; isolating Re alone requires
fixed-d, swept-Q, but the sweep only has two Q per d. The plots are reported
with that caveat front-and-centre.

Outputs (in --out / "Reynolds_comparison/")
    reynolds_arc_velocity.{pdf,png,txt}
    reynolds_mode_b_fraction.{pdf,png,txt}
    reynolds_cycle_frequency.{pdf,png,txt}
    reynolds_summary.csv
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import figstyle
import plot_config
import sweep_utils


# CO2 properties at room conditions (T = 295.15 K, P = 101.325 kPa)
T_REF        = 295.15          # K
P_REF        = 101_325.0       # Pa
RHO_CO2_REF  = 1.815           # kg/m^3   (CRC Handbook 90th ed.)
MU_CO2_REF   = 1.48e-5         # Pa.s     (Lemmon & Jacobsen 2004)
SOUND_CO2    = 270.0           # m/s, approx for the Mach caveat

# Conservative default 1-sigma flow/geometry uncertainties (overridable on CLI).
SIGMA_D_REL_DEFAULT = 0.05
SIGMA_Q_REL_DEFAULT = 0.05
SIGMA_T_K           = 2.0
SIGMA_P_PA          = 1000.0


_FNAME_RE = re.compile(r'HS_([0-9.]+)mm_([0-9]+)mL', re.IGNORECASE)


def parse_condition(stem):
    """Return (d_mm, Q_mLpm, is_repeat) from a recording stem like
    'HS_0.4mm_1500mL_(CO2)_trig_repeat'."""
    m = _FNAME_RE.search(stem)
    if not m:
        return None
    d_mm   = float(m.group(1))
    q_mLpm = float(m.group(2))
    is_rep = ('repeat' in stem.lower())
    return d_mm, q_mLpm, is_rep


def compute_re(d_mm, q_mLpm,
               sigma_d_rel=SIGMA_D_REL_DEFAULT,
               sigma_q_rel=SIGMA_Q_REL_DEFAULT,
               sigma_T=SIGMA_T_K, sigma_P=SIGMA_P_PA):
    """Return dict with v_bulk_ms, Re, sigma_Re, Ma, and intermediate values.

    Linear propagation of uncertainty:
        v = (Q / (pi/4 d^2));   Re = rho v d / mu
        (sigma_v/v)^2 = (sigma_Q/Q)^2 + (2 sigma_d/d)^2
        (sigma_Re/Re)^2 = (sigma_v/v)^2 + (sigma_d/d)^2 + (sigma_rho/rho)^2
                          + (sigma_mu/mu)^2
    Density follows ideal gas, rho = P M / R T (M_CO2 = 44.01 g/mol):
        (sigma_rho/rho)^2 = (sigma_T/T)^2 + (sigma_P/P)^2
    Viscosity Sutherland slope around 295 K gives ~3 %/K — at sigma_T=2 K
    that's ~0.6 % which we fold in as a flat sigma_mu/mu = 0.02 (Lemmon &
    Jacobsen cite ~2 % combined property uncertainty).
    """
    d = d_mm * 1e-3
    Q = q_mLpm * 1e-6 / 60.0
    A = np.pi / 4.0 * d ** 2
    v = Q / A

    rho     = RHO_CO2_REF
    mu      = MU_CO2_REF
    Re      = rho * v * d / mu
    Ma      = v / SOUND_CO2

    rel_v_sq    = sigma_q_rel ** 2 + (2 * sigma_d_rel) ** 2
    rel_rho_sq  = (sigma_T / T_REF) ** 2 + (sigma_P / P_REF) ** 2
    rel_mu_sq   = 0.02 ** 2
    rel_Re_sq   = rel_v_sq + sigma_d_rel ** 2 + rel_rho_sq + rel_mu_sq
    sigma_Re    = Re * np.sqrt(rel_Re_sq)
    sigma_v     = v * np.sqrt(rel_v_sq)

    return dict(
        d_mm=d_mm, q_mLpm=q_mLpm,
        d_m=d, Q_m3s=Q, A_m2=A,
        v_bulk_ms=v, sigma_v_bulk_ms=sigma_v,
        rho=rho, mu=mu,
        Re=Re, sigma_Re=sigma_Re, Ma=Ma,
    )


def load_event_metrics(results_dir):
    """Return (n, median_v_ms, mean_v_ms, SE_v_ms, sigma_v_sys_ms, source_csv)."""
    p = results_dir / 'event_summary_with_unc.csv'
    if not p.exists():
        p = results_dir / 'event_summary.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if 'vel_ms' not in df.columns:
        return None
    df = df[df['vel_ms'] > 0]
    if len(df) == 0:
        return None
    v = df['vel_ms'].to_numpy(dtype=float)
    med  = float(np.median(v))
    mean = float(np.mean(v))
    SE   = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float('nan')
    if 'vel_ms_err_sys' in df.columns:
        sys_med = float(np.median(df['vel_ms_err_sys']))
    else:
        sys_med = float('nan')
    return dict(n=len(v), median_v_ms=med, mean_v_ms=mean,
                SE_v_ms=SE, sigma_v_sys_ms=sys_med, source_csv=str(p.name))


def load_mode_b_fraction(results_dir):
    """Read re_strikes_qa.csv and return Mode-B fraction + SE (Wilson)."""
    p = results_dir / 're_strikes_qa.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if 'mode' not in df.columns or len(df) == 0:
        return None
    n   = len(df)
    n_b = int((df['mode'] == 'B').sum())
    if n == 0:
        return None
    p_b = n_b / n
    SE  = float(np.sqrt(p_b * (1 - p_b) / n))
    return dict(n_restrikes=n, n_b=n_b, frac_b=p_b, SE_frac_b=SE)


def load_cycle_metrics(results_dir, fps):
    """Re-strike cycle statistics from ``cycle_frames`` (period, in frames).

    Reports the cycle both as a duration (ms; the more intuitive read) and as a
    frequency (Hz), so the figure can plot either via ``plot_config.CYCLE``.

        cycle duration  = 1000 * cycle_frames / fps   [ms]
        cycle frequency = fps / cycle_frames          [Hz]
    """
    p = results_dir / 're_strikes_qa.csv'
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if 'cycle_frames' not in df.columns:
        return None
    cyc = pd.to_numeric(df['cycle_frames'], errors='coerce').dropna()
    cyc = cyc[cyc > 0]
    if len(cyc) < 3:
        return None
    n = int(len(cyc))
    mean_frames = float(cyc.mean())
    # duration: mean period in ms, SE = SE of the mean period scaled to ms
    dur_mean = 1000.0 * mean_frames / fps
    dur_med  = 1000.0 * float(cyc.median()) / fps
    SE_dur   = 1000.0 * float(cyc.std(ddof=1)) / fps / np.sqrt(n)
    # frequency: kept for back-compat and the CSV
    f_mean = float(fps / mean_frames)
    f_med  = float(fps / float(cyc.median()))
    SE_f   = float(fps * cyc.std(ddof=1) / (mean_frames ** 2) / np.sqrt(n))
    return dict(n_cycles=n,
                dur_mean_ms=dur_mean, dur_med_ms=dur_med, SE_dur_ms=SE_dur,
                f_mean_Hz=f_mean, f_med_Hz=f_med, SE_f_Hz=SE_f)


def per_recording_row(results_dir, sigma_d_rel, sigma_q_rel):
    """Build one summary row for a recording, or None if it can't be measured."""
    meta_path = results_dir / 'run_metadata.json'
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    cond = parse_condition(meta.get('recording', results_dir.name))
    if cond is None:
        return None
    d_mm, q_mLpm, is_rep = cond
    re_out = compute_re(d_mm, q_mLpm, sigma_d_rel=sigma_d_rel,
                                       sigma_q_rel=sigma_q_rel)
    em = load_event_metrics(results_dir)
    mb = load_mode_b_fraction(results_dir)
    cf = load_cycle_metrics(results_dir, fps=float(meta.get('fps', 200000)))

    row = dict(
        recording=meta.get('recording', results_dir.name),
        d_mm=d_mm, q_mLpm=q_mLpm, is_repeat=int(is_rep),
        v_bulk_ms=re_out['v_bulk_ms'],
        sigma_v_bulk_ms=re_out['sigma_v_bulk_ms'],
        Re=re_out['Re'], sigma_Re=re_out['sigma_Re'], Ma=re_out['Ma'],
        rho=re_out['rho'], mu=re_out['mu'],
        n_events=(em['n']            if em else 0),
        median_v_ms=(em['median_v_ms']   if em else np.nan),
        SE_v_ms=(em['SE_v_ms']           if em else np.nan),
        sigma_v_sys_ms=(em['sigma_v_sys_ms'] if em else np.nan),
        n_restrikes=(mb['n_restrikes']   if mb else 0),
        frac_b=(mb['frac_b']             if mb else np.nan),
        SE_frac_b=(mb['SE_frac_b']       if mb else np.nan),
        n_cycles=(cf['n_cycles']         if cf else 0),
        dur_mean_ms=(cf['dur_mean_ms']   if cf else np.nan),
        SE_dur_ms=(cf['SE_dur_ms']       if cf else np.nan),
        f_mean_Hz=(cf['f_mean_Hz']       if cf else np.nan),
        SE_f_Hz=(cf['SE_f_Hz']           if cf else np.nan),
    )
    return row


def aggregate_per_condition(df):
    """Per (d_mm, q_mLpm), aggregate across repeats."""
    rows = []
    for (d, q), grp in df.groupby(['d_mm', 'q_mLpm']):
        n_rep = len(grp)
        med   = grp['median_v_ms'].mean()
        # systematic does NOT sqrt(N)-suppress within shared calibration —
        # we conservatively take the mean sys across repeats (each had its
        # own calibration so they aren't perfectly correlated either).
        sys_v = grp['sigma_v_sys_ms'].mean()
        # repeat-spread SE adds in quadrature on top of sys.
        if n_rep > 1:
            SE_rep = grp['median_v_ms'].std(ddof=1) / np.sqrt(n_rep)
        else:
            SE_rep = float(grp['SE_v_ms'].iloc[0]) if pd.notnull(grp['SE_v_ms'].iloc[0]) else 0.0
        tot_v = float(np.sqrt(SE_rep ** 2 + (sys_v if pd.notnull(sys_v) else 0.0) ** 2))
        rows.append(dict(
            d_mm=d, q_mLpm=q, n_repeats=n_rep,
            Re=float(grp['Re'].mean()),
            sigma_Re=float(grp['sigma_Re'].mean()),
            Ma=float(grp['Ma'].mean()),
            median_v_ms=float(med),
            sigma_v_ms_total=tot_v,
            sigma_v_ms_stat=float(SE_rep),
            sigma_v_ms_sys=float(sys_v) if pd.notnull(sys_v) else 0.0,
            frac_b=float(grp['frac_b'].mean()),
            SE_frac_b=float(grp['SE_frac_b'].mean()),
            dur_mean_ms=float(grp['dur_mean_ms'].mean()),
            SE_dur_ms=float(grp['SE_dur_ms'].mean()),
            f_mean_Hz=float(grp['f_mean_Hz'].mean()),
            SE_f_Hz=float(grp['SE_f_Hz'].mean()),
        ))
    return pd.DataFrame(rows).sort_values(['Re'])


def plot_versus_re(df_agg, y_col, y_err_col, ylabel, stem, caption, out_dir):
    figstyle.apply()
    m = plot_config.MARKER
    fig, ax = plt.subplots(figsize=plot_config.figsize(stem))
    for d_mm, g in df_agg.groupby('d_mm'):
        col = plot_config.nozzle_color(d_mm)
        ax.errorbar(g['Re'], g[y_col],
                    xerr=g['sigma_Re'], yerr=g[y_err_col],
                    fmt=m['fmt'], ms=m['size'], capsize=m['capsize'],
                    color=col, mec=m['edge_color'], mew=m['edge_width'],
                    label=f'd = {d_mm:g} mm')
    ax.set_xlabel(plot_config.LABELS['reynolds_x'])
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=plot_config.STYLE['legend_size'],
              title=plot_config.LABELS['legend_title'],
              title_fontsize=plot_config.STYLE['legend_size'], loc='best')
    ax.grid(alpha=m['grid_alpha'])
    fig.tight_layout()
    figstyle.save(fig, str(out_dir / stem), caption)
    figstyle.verify(str(out_dir / stem))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root',  required=True, type=str,
                    help='J Electrode root containing *_results dirs.')
    ap.add_argument('--out',   type=str, default=None,
                    help='Output directory (default: <root>/Reynolds_comparison).')
    ap.add_argument('--sigma-d-rel', type=float, default=SIGMA_D_REL_DEFAULT)
    ap.add_argument('--sigma-q-rel', type=float, default=SIGMA_Q_REL_DEFAULT)
    args = ap.parse_args()
    root    = Path(args.root)
    out_dir = Path(args.out) if args.out else (root / 'Reynolds_comparison')
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [d for d in sweep_utils.iter_sweep_dirs(root)
               if not d.name.endswith('_chunked_results')
               and ((d / 'event_summary.csv').exists()
                    or (d / 'event_summary_with_unc.csv').exists())]
    print(f"reynolds_comparison: {len(targets)} recording(s)")

    rows = []
    for d in targets:
        r = per_recording_row(d, args.sigma_d_rel, args.sigma_q_rel)
        if r is None:
            print(f"  [skip] {d.name}")
            continue
        rows.append(r)
        print(f"  [ok]   {d.name}: d={r['d_mm']:.2f}mm Q={r['q_mLpm']:.0f}mL/min  "
              f"v_bulk={r['v_bulk_ms']:6.1f} m/s  Re={r['Re']:7.0f}±{r['sigma_Re']:.0f}  "
              f"Ma={r['Ma']:.2f}  med_v_arc={r['median_v_ms']:.2f}±{r['sigma_v_sys_ms']:.2f}  "
              f"frac_B={r['frac_b']:.3f}")
    if not rows:
        print('[ERROR] no recordings could be measured.')
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'reynolds_per_recording.csv', index=False)

    df_agg = aggregate_per_condition(df)
    df_agg.to_csv(out_dir / 'reynolds_summary.csv', index=False)

    print(f"\nAggregated to {len(df_agg)} (d, Q) condition(s):")
    print(df_agg[['d_mm', 'q_mLpm', 'n_repeats', 'Re', 'Ma',
                  'median_v_ms', 'sigma_v_ms_total',
                  'frac_b', 'dur_mean_ms', 'f_mean_Hz']].to_string(index=False))

    cap_common = (
        'CO2 at T=295.15 K, P=101.325 kPa: rho=1.815 kg/m^3 (CRC), mu=1.48e-5 Pa.s '
        '(Lemmon & Jacobsen 2004). v_bulk = Q / (pi/4 d^2). Re = rho v_bulk d / mu. '
        'X-error: sigma_d=5%, sigma_Q=5%, sigma_T=2K, sigma_P=1kPa, sigma_mu=2%.\n'
        'Y-error: repeat-SE in quadrature with the per-recording px/mm+fps systematic '
        '(NOT averaged down, since the systematic is coherent within one calibration).\n'
        'CONFOUND: changing nozzle diameter changes BOTH Re AND geometry (impingement, '
        'divergence). A trend visible here therefore conflates fluid-dynamic and geometric '
        'effects; isolating Re alone requires fixed-d, swept-Q across many flows.\n'
        'COMPRESSIBILITY: v_bulk hits 132 m/s in the small-nozzle high-flow corner '
        '(Ma~0.49 in CO2 at 295 K); the incompressible-Re value is reported but the '
        'flow is firmly compressible there.'
    )

    plot_versus_re(df_agg, 'median_v_ms', 'sigma_v_ms_total',
                   ylabel='Median event arc velocity (m/s)',
                   stem='reynolds_arc_velocity',
                   caption='Arc velocity vs Re — repeat-aware, with systematic uncertainty.\n\n'
                           + cap_common,
                   out_dir=out_dir)
    plot_versus_re(df_agg, 'frac_b', 'SE_frac_b',
                   ylabel='Mode B re-strike fraction',
                   stem='reynolds_mode_b_fraction',
                   caption='Mode B re-strike fraction vs Re.\n\n' + cap_common,
                   out_dir=out_dir)
    # Re-strike cycling panel. By default this reports the cycle DURATION (ms);
    # flip plot_config.CYCLE['metric'] to 'frequency_hz' for the old Hz plot.
    cyc = plot_config.CYCLE
    if cyc['metric'] == 'frequency_hz':
        plot_versus_re(df_agg, 'f_mean_Hz', 'SE_f_Hz',
                       ylabel=cyc['frequency_label'],
                       stem=cyc['frequency_stem'],
                       caption='Re-strike cycle frequency (fps / mean cycle_frames) vs Re.\n\n'
                               + cap_common,
                       out_dir=out_dir)
    else:
        plot_versus_re(df_agg, 'dur_mean_ms', 'SE_dur_ms',
                       ylabel=cyc['duration_label'],
                       stem=cyc['duration_stem'],
                       caption='Re-strike cycle duration (1000 * mean cycle_frames / fps) vs Re.\n'
                               'This is the mean time between successive arc re-strikes.\n\n'
                               + cap_common,
                       out_dir=out_dir)
    print(f"\nWrote figures + sidecars in {out_dir}")


if __name__ == '__main__':
    main()
