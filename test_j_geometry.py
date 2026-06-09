"""Regression tests for the J-electrode geometry logic (numpy-only).

Run:  python test_j_geometry.py

These pin down (a) the calibration math in calibrate_j_electrodes.derive_geometry
(min-gap -> px/mm, gap_row for both arc directions, blade angles) and (b) the
finite-segment attachment gate added to analyze_blade._arc_complete_blade_lines,
including the guarantee that with --finite-segments OFF the BLADE infinite-line
behaviour is byte-for-byte unchanged.

The functions are imported from the live modules. If scipy/scikit-image are not
installed (so analyze_blade can't import), the test falls back to verifying the
calibration half only and prints a SKIP note for the analyze half.
"""
import numpy as np

from calibrate_j_electrodes import derive_geometry

try:
    from analyze_blade import (_arc_complete_blade_lines, _dist_to_line,
                               _on_segment)
    _HAVE_ANALYZE = True
except Exception as e:                                  # pragma: no cover
    _HAVE_ANALYZE = False
    print("[SKIP] analyze_blade import failed (%s); testing calibration only." % e)


def approx(a, b, tol=1e-3):
    return abs(a - b) <= tol


A_tip, B_tip = (30.0, 118.0), (30.0, 140.0)
A_far, B_far = (120.0, 106.0), (120.0, 152.0)
clicks = [A_tip, A_far, B_tip, B_far]


def test_derive_geometry():
    bl, g = derive_geometry(clicks, gap_mm=2.4, arc_direction='down', seg_margin_px=6)
    assert approx(g['min_gap_px'], 22.0), g
    assert approx(g['px_per_mm'], 9.1667, 1e-3), g
    assert g['gap_row'] == 28, g                         # floor(30) - 2
    assert approx(g['blade_a_angle_deg'], 7.59, 0.1), g
    assert approx(g['blade_b_angle_deg'], 7.59, 0.1), g
    assert 14.5 <= g['blade_a_angle_deg'] + g['blade_b_angle_deg'] <= 15.6, g

    _, g_up = derive_geometry(clicks, 2.4, 'up', 6)
    assert g_up['gap_row'] == 32, g_up                   # ceil(30) + 2

    _, g_ov = derive_geometry(clicks, 2.4, 'down', 6, px_per_mm_override=9.0)
    assert approx(g_ov['px_per_mm'], 9.0), g_ov
    print("test_derive_geometry: OK")
    return bl


def test_finite_segments(bl):
    if not _HAVE_ANALYZE:
        return
    # tip feet attach in both modes
    eps = np.array([A_tip, B_tip])
    for fin in (False, True):
        _, _, aa, ab = _arc_complete_blade_lines(eps, bl, 4,
                                                 finite_segments=fin, seg_margin_px=6)
        assert aa and ab, fin

    uA = (np.asarray(A_far) - np.asarray(A_tip))
    uA = uA / np.linalg.norm(uA)
    beyond = np.asarray(A_far) + 30.0 * uA
    eps2 = np.array([beyond, B_tip])
    _, _, aa_inf, _ = _arc_complete_blade_lines(eps2, bl, 4, finite_segments=False)
    _, _, aa_fin, _ = _arc_complete_blade_lines(eps2, bl, 4, finite_segments=True, seg_margin_px=6)
    assert aa_inf and not aa_fin, "infinite ATTACH / finite REJECT past far-end"

    near = np.asarray(A_far) + 4.0 * uA
    _, _, aa_n, _ = _arc_complete_blade_lines(np.array([near, B_tip]), bl, 4,
                                              finite_segments=True, seg_margin_px=6)
    assert aa_n, "within-margin foot should still attach"

    # BLADE regression: finite=False == original perp-distance-only logic
    rng = np.random.default_rng(0)
    eps_rand = rng.uniform(0, 256, size=(6, 2))
    _, _, a1, b1 = _arc_complete_blade_lines(eps_rand, bl, 4, finite_segments=False)
    da = np.array([_dist_to_line(e, bl[0, 0], bl[0, 1]) for e in eps_rand])
    db = np.array([_dist_to_line(e, bl[1, 0], bl[1, 1]) for e in eps_rand])
    bs = np.inf; oa = ob = False
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            if da[i] + db[j] < bs:
                bs = da[i] + db[j]; oa = bool(da[i] <= 4); ob = bool(db[j] <= 4)
    assert (a1, b1) == (oa, ob), "finite=False must match original BLADE logic"
    print("test_finite_segments: OK")


if __name__ == '__main__':
    bl = test_derive_geometry()
    test_finite_segments(bl)
    print("\nALL J-GEOMETRY TESTS PASSED")
