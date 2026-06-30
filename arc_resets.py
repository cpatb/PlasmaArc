#!/usr/bin/env python3
"""
arc_resets.py  —  shared detection of "reset intervals": long stretches where no
usable arc is recognised.

A reset interval is a maximal run of consecutive frames with NO usable arc
(``arc_detected != 1`` OR ``is_flash == 1``, and missing frames count as no-arc)
of length >= ``reset_gap`` frames.  Physically these are full extinctions: the
discharge goes dark long enough that the next re-strike begins a fresh sequence
rather than continuing the previous cycle.

Used by:
  * ``mode_transition_analysis.py`` — to break the Markov chain across resets
    (two re-strikes separated by a reset are NOT a within-sequence transition);
  * ``reset_interval_analysis.py`` — to characterise reset duration/frequency and
    the mode of the first re-strike after each reset.

``measurements_all_events.csv`` (one row per processed frame, including
``arc_detected==0`` rows) is the source.  The functions are robust to missing
frames (treated as no-arc) by working over the full integer frame range.

Default ``reset_gap = 500`` frames == ``event_velocity.MAX_FRAME_GAP`` (the
existing "force a new primary event" dropout boundary, ~2.5 ms at 200 kfps), so
the reset definition is consistent with the rest of the pipeline rather than
invented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_RESET_GAP = 500


def invalid_per_frame(meas_df: pd.DataFrame) -> "tuple[np.ndarray, np.ndarray]":
    """Return (frames_full, invalid_full) over the full contiguous frame range.

    ``invalid_full[i]`` is True where frame ``frames_full[i]`` has no usable arc:
    ``arc_detected != 1`` OR ``is_flash == 1``; frames absent from ``meas_df`` are
    treated as invalid (no data == no arc).
    """
    d = meas_df[["frame"]].copy()
    fr = pd.to_numeric(meas_df["frame"], errors="coerce").to_numpy()
    det = pd.to_numeric(meas_df.get("arc_detected", 0), errors="coerce").fillna(0).to_numpy()
    flash = pd.to_numeric(meas_df.get("is_flash", 0), errors="coerce").fillna(0).to_numpy()
    good = np.isfinite(fr)
    fr = fr[good].astype(np.int64)
    inv = (det[good] != 1) | (flash[good] == 1)

    if len(fr) == 0:
        return np.zeros(0, np.int64), np.zeros(0, bool)
    fmin, fmax = int(fr.min()), int(fr.max())
    frames_full = np.arange(fmin, fmax + 1, dtype=np.int64)
    invalid_full = np.ones(len(frames_full), dtype=bool)   # default invalid (missing => no arc)
    invalid_full[fr - fmin] = inv
    return frames_full, invalid_full


def _true_runs(mask: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Start/end (inclusive) indices of maximal True runs in a boolean array."""
    if mask.size == 0:
        return np.zeros(0, int), np.zeros(0, int)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    starts = edges[0::2]
    ends = edges[1::2] - 1
    return starts, ends


def reset_intervals(meas_df: pd.DataFrame, reset_gap: int = DEFAULT_RESET_GAP) -> pd.DataFrame:
    """All reset intervals (no-arc runs >= ``reset_gap`` frames).

    Returns a DataFrame sorted by ``start_frame`` with columns
    ``[start_frame, end_frame, gap_frames]`` (end_frame is the last no-arc frame).
    """
    frames_full, invalid_full = invalid_per_frame(meas_df)
    s, e = _true_runs(invalid_full)
    if len(s) == 0:
        return pd.DataFrame(columns=["start_frame", "end_frame", "gap_frames"])
    start_f = frames_full[s]
    end_f = frames_full[e]
    gap = end_f - start_f + 1
    keep = gap >= reset_gap
    out = pd.DataFrame(dict(start_frame=start_f[keep], end_frame=end_f[keep],
                            gap_frames=gap[keep]))
    return out.sort_values("start_frame").reset_index(drop=True)


def all_gap_lengths(meas_df: pd.DataFrame) -> np.ndarray:
    """Lengths (frames) of EVERY no-arc run (no threshold) — for the duration
    distribution / bimodality check that motivates the reset_gap choice."""
    frames_full, invalid_full = invalid_per_frame(meas_df)
    s, e = _true_runs(invalid_full)
    if len(s) == 0:
        return np.zeros(0, np.int64)
    return (frames_full[e] - frames_full[s] + 1).astype(np.int64)


def crosses_reset(f_prev: int, f_cur: int, resets: pd.DataFrame) -> bool:
    """True if a reset interval starts strictly between ``f_prev`` and ``f_cur``.

    Used to break a re-strike->re-strike transition that spans a full extinction.
    """
    if resets is None or len(resets) == 0 or f_cur <= f_prev:
        return False
    starts = resets["start_frame"].to_numpy()
    # any reset whose start is in (f_prev, f_cur)
    lo = np.searchsorted(starts, f_prev, side="right")
    hi = np.searchsorted(starts, f_cur, side="left")
    return hi > lo


def first_restrike_after(resets: pd.DataFrame, restrikes: pd.DataFrame,
                         frame_col: str = "frame", mode_col: str = "mode") -> pd.DataFrame:
    """Attach the first re-strike with frame > reset end to each reset interval.

    ``restrikes`` must have ``frame`` and ``mode`` columns. Returns ``resets`` with
    added ``first_restrike_frame``, ``first_restrike_mode``, ``frames_to_restrike``
    (NaN if no re-strike follows — kept for the duration distribution, dropped from
    the mode test by the caller).
    """
    out = resets.copy().reset_index(drop=True)
    if len(out) == 0:
        out["first_restrike_frame"] = []
        out["first_restrike_mode"] = []
        out["frames_to_restrike"] = []
        return out
    rs = restrikes.sort_values(frame_col).reset_index(drop=True)
    rf = pd.to_numeric(rs[frame_col], errors="coerce").to_numpy()
    rm = rs[mode_col].to_numpy()
    end = out["end_frame"].to_numpy()
    pos = np.searchsorted(rf, end, side="right")   # first re-strike with frame > end
    f_after = np.where(pos < len(rf), rf[np.clip(pos, 0, len(rf) - 1)], np.nan)
    m_after = np.where(pos < len(rf), rm[np.clip(pos, 0, len(rf) - 1)], None)
    out["first_restrike_frame"] = f_after
    out["first_restrike_mode"] = m_after
    out["frames_to_restrike"] = f_after - end
    return out
