"""
sweep_utils.py — sweep-level recording-exclusion mechanism.

A ``<stem>_results`` directory may carry a ``.excluded_from_sweep`` sentinel
file. Cross-recording aggregators (reynolds_comparison, mode_ratio_vs_re,
asymmetry_vs_time, warmup_time, temporal_velocity_profiles when aggregating)
skip those directories by default. Per-recording diagnostics
(recompute_uncertainty, verify_fps_threading) still process them, because
per-recording data is honestly captured and worth keeping visible.

Exclude a recording from the sweep by writing::

    <stem>_results/.excluded_from_sweep

with a paragraph inside explaining WHY (so the choice is documented and
auditable, not silent). Read the rationale from any consumer with
``read_exclusion_rationale(results_dir)``.
"""
from __future__ import annotations
from pathlib import Path

EXCLUSION_SENTINEL = '.excluded_from_sweep'


def is_excluded(results_dir):
    """True iff the directory carries the EXCLUSION_SENTINEL file."""
    return (Path(results_dir) / EXCLUSION_SENTINEL).exists()


def iter_sweep_dirs(root, include_excluded=False):
    """Yield ``*_results`` directories under ``root``, sorted, skipping any
    that carry EXCLUSION_SENTINEL (unless ``include_excluded=True``).
    """
    root = Path(root)
    for d in sorted(root.glob('*_results')):
        if not d.is_dir():
            continue
        if is_excluded(d) and not include_excluded:
            continue
        yield d


def read_exclusion_rationale(results_dir):
    """Return the contents of EXCLUSION_SENTINEL, or '' if absent."""
    p = Path(results_dir) / EXCLUSION_SENTINEL
    return p.read_text(encoding='utf-8') if p.exists() else ''


__all__ = ['EXCLUSION_SENTINEL', 'is_excluded',
            'iter_sweep_dirs', 'read_exclusion_rationale']
