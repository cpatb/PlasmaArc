#!/usr/bin/env python3
"""
Regression tests for the extracted electrode_geometry primitives.

Each test compares the shared helper against a VERBATIM re-implementation of the
original inline code it replaces, and against golden values captured on baseline
c3efc88. numpy-only; run with `python test_electrode_geometry.py` or pytest.
"""
import numpy as np

import electrode_geometry as eg


# ── gap_row_from_tips vs the original floor/ceil ±2 formula ───────────────────

def _orig_gap_row(a_row, b_row, arc_direction):
    # verbatim from calibrate_j_electrodes L134-138 / detect_j_hooks L135-136 /
    # segment_electrodes L212-213
    m = 0.5 * (a_row + b_row)
    if arc_direction == 'down':
        return int(np.floor(m)) - 2
    return int(np.ceil(m)) + 2


def test_gap_row_matches_original():
    rng = np.random.default_rng(1)
    for _ in range(200):
        a, b = rng.uniform(0, 256, 2)
        for ad in ('down', 'up'):
            assert eg.gap_row_from_tips(a, b, ad) == _orig_gap_row(a, b, ad)


def test_gap_row_golden():
    # golden clicks tip rows 30 and 32 -> mean 31
    assert eg.gap_row_from_tips(30.0, 32.0, 'down') == 29
    assert eg.gap_row_from_tips(30.0, 32.0, 'up') == 33


# ── nearest_facing_tips: full vs the original inline argmin ────────────────────

def _orig_full(A, B):
    # verbatim from segment_electrodes.paths_to_edges L204-206
    D = np.linalg.norm(A[:, None, :] - B[None], axis=2)
    i, j = np.unravel_index(D.argmin(), D.shape)
    return A[i], B[j], float(D[i, j])


def _orig_subsample(A, B, k=500, seed=0):
    # verbatim from detect_j_hooks.detect_edges L102-107
    rng = np.random.default_rng(seed)
    ai = A[rng.choice(len(A), min(k, len(A)), replace=False)]
    bi = B[rng.choice(len(B), min(k, len(B)), replace=False)]
    D = np.linalg.norm(ai[:, None, :] - bi[None], axis=2)
    i, j = np.unravel_index(D.argmin(), D.shape)
    return ai[i], bi[j], float(np.linalg.norm(ai[i] - bi[j]))


def test_nearest_full_matches_original():
    rng = np.random.default_rng(2)
    for _ in range(20):
        A = rng.uniform(0, 256, (rng.integers(5, 300), 2))
        B = rng.uniform(0, 256, (rng.integers(5, 300), 2))
        ta, tb, g = eg.nearest_facing_tips(A, B)
        ota, otb, og = _orig_full(A, B)
        assert np.array_equal(ta, ota) and np.array_equal(tb, otb)
        assert g == og


def test_nearest_subsample_matches_original():
    rng = np.random.default_rng(3)
    for _ in range(20):
        A = rng.uniform(0, 256, (rng.integers(600, 1200), 2))
        B = rng.uniform(0, 256, (rng.integers(600, 1200), 2))
        ta, tb, g = eg.nearest_facing_tips(A, B, subsample=500, seed=0)
        ota, otb, og = _orig_subsample(A, B, k=500, seed=0)
        assert np.array_equal(ta, ota) and np.array_equal(tb, otb)
        assert g == og


# ── find_first_arc_frame on a fake tif ────────────────────────────────────────

class _FakePage:
    def __init__(self, m):
        self._m = m

    def asarray(self):
        return np.array([[self._m]], dtype=np.uint8)


class _FakeTif:
    def __init__(self, maxes):
        self.pages = [_FakePage(m) for m in maxes]


def test_find_first_arc_frame():
    assert eg.find_first_arc_frame(_FakeTif([0, 10, 34, 35, 200])) == 3
    assert eg.find_first_arc_frame(_FakeTif([0, 10, 20])) == 0            # none reach guard
    assert eg.find_first_arc_frame(_FakeTif([200]), dark_guard=35) == 0


# ── derive_geometry full-output golden (calibrate_j_electrodes, unchanged) ─────

def test_derive_geometry_golden():
    import calibrate_j_electrodes as cj
    clicks = [(30.0, 150.0), (90.0, 120.0), (32.0, 172.0), (92.0, 205.0)]
    bl, geom = cj.derive_geometry(clicks, 3.2, 'down', 6, px_per_mm_override=None)
    assert abs(bl.sum() - 891.0) < 1e-9 and bl.shape == (2, 2, 2)
    assert geom['gap_row'] == 29
    assert geom['min_gap_px'] == 22.091
    assert geom['px_per_mm'] == 6.9034
    assert geom['finite_segments'] is True


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in fns:
        fn()
        print('ok:', fn.__name__)
    print('ALL %d TESTS PASSED' % len(fns))
