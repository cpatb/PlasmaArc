"""Tests for rig_annotation geometry on a synthetic V of two blades with a known gap and opening."""
import numpy as np
import pytest

import rig_annotation as ra


def _v_rig(gap_px=20.0, half_angle_deg=45.0, length=80.0, apex=(230.0, 110.0)):
    """Two straight blades opening UPWARD (to smaller rows) from tips gap_px apart."""
    r0, c0 = apex
    th = np.radians(half_angle_deg)
    tipA = np.array([r0, c0 - gap_px / 2])
    tipB = np.array([r0, c0 + gap_px / 2])
    endA = tipA + length * np.array([-np.cos(th), -np.sin(th)])     # up and left
    endB = tipB + length * np.array([-np.cos(th), +np.sin(th)])     # up and right
    return [tipA.tolist(), endA.tolist()], [tipB.tolist(), endB.tolist()], tipA, tipB


def test_gap_line_sets_px_per_mm_hand_computed():
    A, B, ta, tb = _v_rig(gap_px=24.0)
    ann = dict(electrode_A=A, electrode_B=B, gap_line=[ta.tolist(), tb.tolist()], gap_mm=2.4)
    g = ra.geometry(ann)
    assert g["gap_px"] == pytest.approx(24.0)
    assert g["px_per_mm"] == pytest.approx(10.0)           # 24 px / 2.4 mm
    assert g["gap_from_edges_px"] == pytest.approx(24.0, abs=1e-6)
    assert g["gap_source"] == "drawn gap line"


def test_opening_angle_and_bisector():
    A, B, ta, tb = _v_rig(half_angle_deg=45.0)
    g = ra.geometry(dict(electrode_A=A, electrode_B=B, gap_mm=2.4))
    assert g["opening_deg"] == pytest.approx(90.0, abs=1e-6)
    # symmetric V opening upward: bisector points straight up (-row), through the gap midpoint
    assert np.allclose(g["bisector_dir"], [-1.0, 0.0], atol=1e-9)
    assert np.allclose(g["bisector_point"], [230.0, 110.0])
    # a discriminating case: a 30 deg half angle must give 60 deg, not 90
    A2, B2, _, _ = _v_rig(half_angle_deg=30.0)
    assert ra.geometry(dict(electrode_A=A2, electrode_B=B2, gap_mm=2.4))["opening_deg"] == pytest.approx(60.0, abs=1e-6)


def test_direction_of_tracing_does_not_matter():
    A, B, ta, tb = _v_rig()
    g1 = ra.geometry(dict(electrode_A=A, electrode_B=B, gap_mm=2.4))
    g2 = ra.geometry(dict(electrode_A=A[::-1], electrode_B=B[::-1], gap_mm=2.4))
    assert g1["opening_deg"] == pytest.approx(g2["opening_deg"])
    assert np.allclose(g1["bisector_dir"], g2["bisector_dir"])


def test_edges_only_falls_back_to_closest_approach():
    A, B, ta, tb = _v_rig(gap_px=21.0)
    g = ra.geometry(dict(electrode_A=A, electrode_B=B, gap_mm=2.4))
    assert g["gap_px"] == pytest.approx(21.0, abs=1e-6)
    assert "closest approach" in g["gap_source"]


def test_nothing_drawn_raises():
    with pytest.raises(ValueError):
        ra.geometry(dict(gap_mm=2.4))


def test_edges_npz_assigns_tips_to_the_right_electrode(tmp_path):
    A, B, ta, tb = _v_rig()
    ann = dict(electrode_A=A, electrode_B=B, gap_line=[tb.tolist(), ta.tolist()], gap_mm=2.4)   # drawn B->A
    p = ra.edges_npz(ann, tmp_path / "e.npz")
    z = np.load(p)
    assert np.allclose(z["A_tip"], ta) and np.allclose(z["B_tip"], tb)
    assert z["A"].shape[1] == 2 and len(z["A"]) > 100


def test_save_load_roundtrip_recomputes(tmp_path):
    A, B, ta, tb = _v_rig(gap_px=24.0)
    ann = dict(electrode_A=A, electrode_B=B, gap_line=[ta.tolist(), tb.tolist()], gap_mm=2.4)
    ra.save(ann, tmp_path / "r.json")
    back = ra.load(tmp_path / "r.json")
    assert back["derived"]["px_per_mm"] == pytest.approx(10.0)


def test_straight_blades_extend_from_the_cutoff_to_the_border():
    # hand-computed: A cutoff (230, 100) going up-left at 45 deg meets col 0 at row 130; B cutoff (230, 124)
    # going up-right meets col 255 at row 230 - 131 = 99
    ann = dict(electrode_A=[[200.0, 70.0], [190.0, 60.0]], electrode_B=[[200.0, 154.0], [190.0, 164.0]],
               gap_line=[[230.0, 124.0], [230.0, 100.0]], gap_mm=2.4)
    sb = ra.straight_blades(ann, (256, 256))
    assert np.allclose(sb["A_tip"], [230, 100]) and np.allclose(sb["B_tip"], [230, 124])
    assert np.allclose(sb["A"][-1], [130, 0]) and np.allclose(sb["B"][-1], [99, 255])
    assert sb["A"][:, 0].max() <= 230 + 1e-9 and sb["B"][:, 0].max() <= 230 + 1e-9   # nothing below the gap
    assert sb["opening_deg"] == pytest.approx(90.0)
    assert np.hypot(*np.diff(sb["A"], axis=0).T).max() <= ra.DENSIFY_STEP_PX + 1e-9


def test_straight_blades_refuse_a_gap_line_on_one_blade():
    ann = dict(electrode_A=[[200.0, 70.0], [190.0, 60.0]], electrode_B=[[200.0, 154.0], [190.0, 164.0]],
               gap_line=[[230.0, 100.0], [220.0, 90.0]], gap_mm=2.4)
    with pytest.raises(ValueError):
        ra.straight_blades(ann, (256, 256))
