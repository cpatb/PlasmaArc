"""Tests for run_blade_upwards' hand-annotation lookup, per-condition averaging and stop rules (no cine needed)."""
import json

import numpy as np
import pytest

import rig_annotation as ra
import run_blade_upwards as rbu

AIR = rbu.DATA_ROOT / "Air" / "45degree_2.4mmGap"
CINE = AIR / "Re2250" / "BLADE_ADCB_Air_Re2250_06mmNozzle_trial1.cine"
CINE_RE4063 = AIR / "Re4063" / "BLADE_ADCB_Air_Re4063_06mmNozzle_trial2.cine"
CINE_ARGON = rbu.DATA_ROOT / "Argon" / "45degree_2.4mmGap" / "Re2250" / "BLADE_ADCB_Argon_Re2250_06mmNozzle_trial1.cine"
# a symmetric V: gap line from col 100 to col 124 at row 230 = 24 px, i.e. 10 px/mm at 2.4 mm
V = dict(electrode_A=[[230.0, 100.0], [180.0, 50.0]], electrode_B=[[230.0, 124.0], [180.0, 174.0]],
         gap_line=[[230.0, 100.0], [230.0, 124.0]])


def write(tmp_path, name, cines, shape=(256, 256), gap_mm=2.4, **over):
    ann = dict(cines=[str(c) for c in cines], image_shape=list(shape), gap_mm=gap_mm, **dict(V, **over))
    ra.save(ann, tmp_path / name)


@pytest.fixture
def adir(tmp_path, monkeypatch):
    monkeypatch.setattr(rbu, "ANNOT_DIR", tmp_path)
    return tmp_path


def test_hand_scale_is_the_drawn_gap(adir):
    write(adir, "rig_a.json", [CINE])
    cal, problems, flags = rbu.annotation_calibration(CINE, 256, 256)
    assert problems == [] and flags == []
    assert np.isclose(cal["px_per_mm"], 24.0 / 2.4)        # hand-computed anchor: 24 px / 2.4 mm
    assert np.isclose(cal["gap_row"], 230.0)
    assert np.isclose(cal["full_opening_angle_deg"], 90.0)
    assert cal["source"] == "hand annotation" and cal["members"]["n"] == 1


def test_missing_annotation_stops(adir):
    cal, problems, _ = rbu.annotation_calibration(CINE, 256, 256)
    assert cal is None and "no hand annotation" in problems[0]


def test_another_Re_of_the_same_condition_is_used(adir):
    write(adir, "rig_b.json", [CINE_RE4063])
    cal, problems, _ = rbu.annotation_calibration(CINE, 256, 256)
    assert problems == [] and np.isclose(cal["px_per_mm"], 10.0)


def test_another_gas_is_not_used(adir):
    write(adir, "rig_b.json", [CINE_ARGON])
    cal, problems, _ = rbu.annotation_calibration(CINE, 256, 256)
    assert cal is None and "no hand annotation" in problems[0]


def test_members_are_averaged_hand_computed(adir):
    # member 1: ends at cols 100 / 124 (24 px); member 2: ends at 102 / 122 (20 px, each 1.41 px off its line).
    # Mean ends (230, 101) and (230, 123) -> 22 px -> 22 / 2.4 px/mm.
    write(adir, "rig_a.json", [CINE])
    write(adir, "rig_b.json", [CINE_RE4063], gap_line=[[230.0, 102.0], [230.0, 122.0]])
    cal, problems, flags = rbu.annotation_calibration(CINE, 256, 256)
    assert problems == [] and cal["members"]["n"] == 2
    assert np.isclose(cal["gap_px"], 22.0) and np.isclose(cal["px_per_mm"], 22.0 / 2.4)
    assert np.allclose(cal["members"]["gap_px"], [24.0, 20.0])
    assert np.isclose(cal["px_per_mm_sem"], np.std([24.0, 20.0], ddof=1) / np.sqrt(2) / 2.4)


def test_outlying_member_is_flagged_and_kept(adir):
    write(adir, "rig_a.json", [CINE])
    write(adir, "rig_b.json", [CINE])
    write(adir, "rig_c.json", [CINE_RE4063], gap_line=[[230.0, 102.0], [230.0, 122.0]])   # 20 px, -17 %
    cal, problems, flags = rbu.annotation_calibration(CINE, 256, 256)
    assert problems == [] and len(flags) == 1 and "rig_c.json" in flags[0]
    assert np.isclose(cal["gap_px"], (124 + 124 + 122) / 3 - (100 + 100 + 102) / 3)      # kept, not dropped


def test_an_annotation_spanning_two_conditions_stops(adir):
    write(adir, "rig_a.json", [CINE, CINE_ARGON])
    cal, problems, _ = rbu.annotation_calibration(CINE, 256, 256)
    assert cal is None and "2 conditions" in problems[0]


def test_md5_changes_when_a_member_is_added(adir):
    write(adir, "rig_a.json", [CINE])
    m1 = rbu.annotation_md5(rbu.find_annotations(CINE)[0])
    write(adir, "rig_b.json", [CINE_RE4063])
    assert rbu.annotation_md5(rbu.find_annotations(CINE)[0]) != m1


def test_wrong_image_size_stops(adir):
    write(adir, "rig_a.json", [CINE], shape=(512, 512))
    _, problems, _ = rbu.annotation_calibration(CINE, 256, 256)
    assert any("image" in p for p in problems)


def test_gap_mm_disagreeing_with_folder_stops(adir):
    write(adir, "rig_a.json", [CINE], gap_mm=3.2)
    _, problems, _ = rbu.annotation_calibration(CINE, 256, 256)
    assert any("gap_mm" in p for p in problems)


def test_short_edges_drawn_far_up_the_blade_are_extended_to_the_gap(adir):
    # edges drawn only well up each blade (the typical real annotation): the lines still reach the gap
    write(adir, "rig_a.json", [CINE], electrode_A=[[200.0, 70.0], [180.0, 50.0]],
          electrode_B=[[200.0, 154.0], [180.0, 174.0]])
    cal, problems, flags = rbu.annotation_calibration(CINE, 256, 256)
    assert problems == [] and flags == []
    assert np.isclose(cal["cloud_sep_min_px"], 24.0)


def test_gap_end_off_its_blade_line_is_flagged_not_stopped(adir):
    write(adir, "rig_a.json", [CINE], gap_line=[[230.0, 100.0], [230.0, 131.0]])   # B end 7/sqrt2 = 4.95 px off
    cal, problems, flags = rbu.annotation_calibration(CINE, 256, 256)
    assert problems == [] and len(flags) == 1 and "blade-B" in flags[0]


def test_a_swapped_drawing_averages_as_if_unswapped(adir):
    # discriminating: were A/B not relabelled, averaging a swapped and an unswapped drawing would put both
    # mean ends at the gap midpoint (gap 0 px) instead of reproducing the 24 px V
    write(adir, "rig_a.json", [CINE])
    write(adir, "rig_b.json", [CINE_RE4063], electrode_A=V["electrode_B"], electrode_B=V["electrode_A"],
          gap_line=V["gap_line"][::-1])
    cal, problems, _ = rbu.annotation_calibration(CINE, 256, 256)
    assert problems == [] and np.isclose(cal["gap_px"], 24.0)
    assert np.isclose(cal["full_opening_angle_deg"], 90.0)


def test_incomplete_annotation_stops(adir):
    (adir / "rig_a.json").write_text(json.dumps(dict(cines=[str(CINE)], image_shape=[256, 256], gap_mm=2.4,
                                                     schema=ra.SCHEMA, electrode_A=None, electrode_B=None,
                                                     gap_line=None)))
    cal, problems, _ = rbu.annotation_calibration(CINE, 256, 256)
    assert cal is None and "incomplete" in problems[0]


def test_output_goes_to_a_separate_tree():
    assert rbu.out_dir_for(CINE, annotated=True).parts[-5] == "runs_annotated"
    assert rbu.out_dir_for(CINE).parts[-5] == "runs"
