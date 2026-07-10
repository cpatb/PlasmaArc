#!/usr/bin/env python3
"""
electrode_geometry.py  —  shared primitives for the electrode-geometry producers.

The five geometry producers (BLADE: calibrate_electrodes / detect_blade_lines;
J: calibrate_j_electrodes / detect_j_hooks / segment_electrodes) each rolled their
own copies of the same three small computations. Those copies are collected here,
VERBATIM (no algorithm change), so every producer shares one implementation and
the behaviour is identical by construction.

This module is deliberately numpy-only — it does not import the measurement core
(analyze_blade). The one other shared primitive, `skeleton_endpoints`, already has
a canonical home in analyze_blade and is imported from there by the producers that
need it (as detect_blade_lines already does); it is intentionally NOT duplicated
here.

Scope note: this is the Option-A (producer-side) unification. The measurement
consumer analyze_blade.measure_frame is unchanged; on-disk artifact schemas,
field names, values and rounding are unchanged. See
docs/specs/2026-07-09-electrode-geometry-engine-design.md.
"""
from __future__ import annotations

import numpy as np

# The dark-frame guard shared by the interactive calibrators (frames whose max DN
# is below this hold no arc and are skipped when auto-finding a frame to click on).
DARK_GUARD = 35


def find_first_arc_frame(tif, n_search: int = 1000, dark_guard: int = DARK_GUARD) -> int:
    """Index of the first frame whose max DN reaches `dark_guard` (else 0).

    Extracted verbatim from calibrate_electrodes._find_first_arc_frame /
    calibrate_j_electrodes._find_first_arc_frame (byte-identical copies).
    `tif` is an open tifffile.TiffFile (uses .pages).
    """
    for i in range(min(n_search, len(tif.pages))):
        if int(tif.pages[i].asarray().max()) >= dark_guard:
            return i
    return 0


def gap_row_from_tips(tip_a_row, tip_b_row, arc_direction: str) -> int:
    """Electrode-gap cutoff row from the two hook/blade tip rows.

    'down' electrodes sit at SMALL rows -> reject arc pixels above the tips, so the
    cutoff is placed two rows above the mean tip row: floor(mean) - 2.
    'up' electrodes sit at LARGE rows -> ceil(mean) + 2.

    Extracted verbatim from the identical formula in calibrate_j_electrodes (L134-138),
    detect_j_hooks (L135-136) and segment_electrodes.save_geometry (L212-213).
    """
    mean_row = 0.5 * (tip_a_row + tip_b_row)
    if arc_direction == 'down':
        return int(np.floor(mean_row)) - 2
    return int(np.ceil(mean_row)) + 2


def nearest_facing_tips(A, B, subsample=None, seed: int = 0):
    """Facing tips = the closest pair of points between the two edge point-clouds.

    Returns (tip_a, tip_b, gap_px), each tip a (row, col) point.

    subsample=None  -> exact closest pair over the full clouds
                       (matches segment_electrodes.paths_to_edges, L204-206).
    subsample=k     -> draw min(k, len) points from each cloud with a seeded
                       np.random.default_rng(seed) BEFORE the pairwise argmin, in the
                       order A-then-B (matches detect_j_hooks.detect_edges, L102-107,
                       preserving its exact determinism). The returned tips are the
                       SUBSAMPLED points, exactly as the original did.
    """
    A = np.asarray(A)
    B = np.asarray(B)
    if subsample is not None:
        rng = np.random.default_rng(seed)
        A = A[rng.choice(len(A), min(subsample, len(A)), replace=False)]
        B = B[rng.choice(len(B), min(subsample, len(B)), replace=False)]
    D = np.linalg.norm(A[:, None, :] - B[None], axis=2)
    i, j = np.unravel_index(D.argmin(), D.shape)
    return A[i], B[j], float(D[i, j])
