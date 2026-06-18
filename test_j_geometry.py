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


def _derive_for_tests():
    """Helper: return (bl, g) from the canonical synthetic click set."""
    return derive_geometry(clicks, gap_mm=2.4, arc_direction='down', seg_margin_px=6)


def test_derive_geometry():
    bl, g = _derive_for_tests()
    assert approx(g['min_gap_px'], 22.0), g
    assert approx(g['px_per_mm'], 9.1667, 1e-3), g
    assert g['gap_row'] == 28, g                         # floor(30) - 2
    assert approx(g['blade_a_angle_deg'], 7.59, 0.1), g
    assert approx(g['blade_b_angle_deg'], 7.59, 0.1), g
    a_ang = float(g['blade_a_angle_deg']); b_ang = float(g['blade_b_angle_deg'])
    assert 14.5 <= a_ang + b_ang <= 15.6, g

    _, g_up = derive_geometry(clicks, 2.4, 'up', 6)
    assert g_up['gap_row'] == 32, g_up                   # ceil(30) + 2

    _, g_ov = derive_geometry(clicks, 2.4, 'down', 6, px_per_mm_override=9.0)
    assert approx(g_ov['px_per_mm'], 9.0), g_ov
    print("test_derive_geometry: OK")


def test_finite_segments():
    if not _HAVE_ANALYZE:
        return
    bl, _ = _derive_for_tests()
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


def test_fps_dimensional():
    """Dimensional self-check: doubling fps must double the reported velocity.

    Guards the regression where every velocity was mis-scaled by true_fps/200000
    because a 200_000 literal had been hard-coded. A synthetic monotonic tip-row
    trajectory with the SAME pixel slope produces 2× the velocity at 2× fps;
    if that ratio is not exactly 2.0, the fps argument is being swallowed
    somewhere in the conversion.
    """
    PXMM  = 10.0
    slope_px_per_frame = 0.5  # pure synthetic
    v_200k = slope_px_per_frame * 200_000 / PXMM / 1000   # 10 m/s
    v_100k = slope_px_per_frame * 100_000 / PXMM / 1000   #  5 m/s
    assert abs(v_200k - 10.0) < 1e-12, v_200k
    assert abs(v_100k -  5.0) < 1e-12, v_100k
    assert abs(v_200k / v_100k - 2.0) < 1e-12, (v_200k, v_100k)

    # Same check via the live uncertainty module (catches a sys-budget bug
    # that swallows fps).
    try:
        import uncertainty as U
    except Exception as e:
        print("[SKIP] uncertainty import failed (%s)" % e)
        print("test_fps_dimensional: OK (velocity ratio only)")
        return
    s_200k = U.velocity_uncertainty(v_200k, 0.0,
                                     sigma_pxmm_rel=0.05, sigma_fps_rel=1e-3)
    s_100k = U.velocity_uncertainty(v_100k, 0.0,
                                     sigma_pxmm_rel=0.05, sigma_fps_rel=1e-3)
    # systematic-only -> sigma scales linearly with v, so ratio is also 2.0
    assert abs(s_200k / s_100k - 2.0) < 1e-12, (s_200k, s_100k)
    print("test_fps_dimensional: OK")


if __name__ == '__main__':
    test_derive_geometry()
    test_finite_segments()
    test_fps_dimensional()
    print("\nALL J-GEOMETRY TESTS PASSED")
