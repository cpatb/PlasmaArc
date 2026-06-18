"""
uncertainty.py — systematic + statistical uncertainty propagation for the J pipeline.

Single source of truth for the *non*-statistical uncertainty budget that the
event/Fourier stages were missing: the OLS slope SE in ``vel_ms_err`` is only
the random fitting error.  Reported velocities and frequencies must additionally
carry the systematic calibration uncertainty from px/mm and fps.

Velocity (m/s) from an OLS slope (px / frame):

    v = slope * FPS / PX_PER_MM / 1000

Linear propagation (slope, FPS, PX_PER_MM uncorrelated):

    sigma_v^2 = (FPS / PX_PER_MM / 1000)^2 * sigma_slope^2
              + (v / FPS)^2                * sigma_FPS^2
              + (v / PX_PER_MM)^2          * sigma_PXMM^2

equivalently, in relative form:

    (sigma_v / |v|)^2 = (sigma_v_stat / |v|)^2
                      + (sigma_FPS / FPS)^2
                      + (sigma_PXMM / PX_PER_MM)^2

Frequency (Hz) from an FFT bin index k of an N-sample signal:

    f = k * FPS / N      ->     sigma_f / f = sigma_FPS / FPS

DEFAULT BUDGET (justified in METHODS.md and ROBUSTNESS_REPORT.md)

  SIGMA_FPS_REL  = 1e-3   (0.1%)  — Phantom HS crystal oscillator is ~50-100 ppm;
                                   0.1% is conservative for trig-mode recordings.
  SIGMA_PXMM_REL = 0.05   (5%)    — calibration gap is ~22-30 px wide for a 2.4 mm
                                   apex gap, with two clicks at ~1 px sub-pixel
                                   localization each: sqrt(2) px / 26.56 px ≈ 5.3%.
                                   Hand-segmented J-hook calibrations may exceed
                                   this slightly; override with --pxmm-rel-err.

These are RELATIVE 1-sigma uncertainties. Override per-recording by passing the
matching CLI flags to ``run_j_co2.py`` / ``event_velocity.py`` / ``fourier_analysis.py``
so the budget is auditable in ``run_metadata.json``.
"""
from __future__ import annotations
import numpy as np

SIGMA_FPS_REL  = 1e-3
SIGMA_PXMM_REL = 0.05


def velocity_uncertainty(v_ms, vel_stat_err,
                         sigma_pxmm_rel=SIGMA_PXMM_REL,
                         sigma_fps_rel=SIGMA_FPS_REL):
    """Quadrature combination of statistical + px/mm + fps uncertainty.

    Parameters
    ----------
    v_ms : float or array
        Velocity in m/s.
    vel_stat_err : float or array
        Statistical (e.g. OLS slope SE) 1-sigma uncertainty on v_ms, in m/s.
        Pass 0 (or NaN) for velocities that have no statistical component
        (e.g. a single frame-difference Δs/Δt); the returned total is then
        the systematic floor.
    sigma_pxmm_rel, sigma_fps_rel : float
        Relative 1-sigma uncertainty on the px/mm calibration and the frame
        rate.

    Returns
    -------
    sigma_v_total : float or array (m/s)
    """
    v   = np.asarray(v_ms,         dtype=float)
    sst = np.asarray(vel_stat_err, dtype=float)
    sst = np.where(np.isfinite(sst), sst, 0.0)
    sys_sq = (v * sigma_pxmm_rel) ** 2 + (v * sigma_fps_rel) ** 2
    return np.sqrt(sst ** 2 + sys_sq)


def velocity_uncertainty_relative(v_ms, vel_stat_err,
                                  sigma_pxmm_rel=SIGMA_PXMM_REL,
                                  sigma_fps_rel=SIGMA_FPS_REL):
    """Same as ``velocity_uncertainty`` but returned as a relative 1-sigma."""
    v = np.asarray(v_ms, dtype=float)
    s = velocity_uncertainty(v_ms, vel_stat_err, sigma_pxmm_rel, sigma_fps_rel)
    out = np.where(np.abs(v) > 0, s / np.maximum(np.abs(v), 1e-30), np.nan)
    return out


def freq_uncertainty(f_hz, sigma_fps_rel=SIGMA_FPS_REL):
    """1-sigma uncertainty on an FFT-derived frequency, m/s units of fps only.

    f = k * FPS / N has no spatial-calibration dependence, so the budget reduces
    to the fps relative uncertainty applied to f.
    """
    f = np.asarray(f_hz, dtype=float)
    return np.abs(f) * sigma_fps_rel


def pxmm_uncertainty_from_calibration(min_gap_px,
                                       click_uncertainty_px=1.0):
    """Estimate px/mm relative uncertainty from the apex-gap calibration.

    The pixel gap g_px is the L2 distance between two clicked tips; each click
    has an isotropic localisation uncertainty c (default 1 px sub-pixel).
    The variance on g_px is then 2 c^2, so:

        sigma_pxmm / pxmm = sqrt(2) * c / g_px

    Returns a float (relative 1-sigma).  The 2.4 mm reference gap itself is
    treated as exact (the J electrodes are machined to 2.4 mm).
    """
    if min_gap_px is None or not np.isfinite(min_gap_px) or min_gap_px <= 0:
        return SIGMA_PXMM_REL
    return float(np.sqrt(2.0) * click_uncertainty_px / float(min_gap_px))


__all__ = [
    'SIGMA_FPS_REL', 'SIGMA_PXMM_REL',
    'velocity_uncertainty', 'velocity_uncertainty_relative', 'freq_uncertainty',
    'pxmm_uncertainty_from_calibration',
]
