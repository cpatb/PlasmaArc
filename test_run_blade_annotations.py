"""Tests for run_blade_upwards' hand-annotation lookup and stop rules (no cine needed)."""
import json

import numpy as np
import pytest

import rig_annotation as ra
import run_blade_upwards as rbu

CINE = rbu.DATA_ROOT / "Air" / "45degree_2.4mmGap" / "Re2250" / "BLADE_ADCB_Air_Re2250_06mmNozzle_trial1.cine"
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
    assert cal["source"] == "hand annotation"


def test_missing_annotation_stops(adir):
    cal, problems, _ = rbu.annotation_calibration(CINE, 256, 256)
    assert cal is None and "no hand annotation" in problems[0]


def test_other_video_is_not_matched(adir):
    write(adir, "rig_b.json", [CINE.with_name("BLADE_ADCB_Argon_Re2250_06mmNozzle_trial1.cine")])
    assert rbu.find_annotation(CINE)[0] is None


def test_two_annotations_for_one_video_stop(adir):
    write(adir, "rig_a.json", [CINE])
    write(adir, "rig_b.json", [CINE])
    path, why = rbu.find_annotation(CINE)
    assert path is None and "2 annotations" in why


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
    assert np.allclose(cal["gap_end_offsets_px"], 0.0, atol=1e-9)
    assert np.isclose(cal["cloud_sep_min_px"], 24.0)


def test_gap_end_off_its_blade_line_is_flagged_not_stopped(adir):
    write(adir, "rig_a.json", [CINE], gap_line=[[230.0, 100.0], [230.0, 131.0]])   # B end 7/sqrt2 = 4.95 px off
    cal, problems, flags = rbu.annotation_calibration(CINE, 256, 256)
    assert problems == [] and len(flags) == 1 and "blade-B" in flags[0]
    assert np.isclose(cal["gap_end_offsets_px"][1], 7 / np.sqrt(2))


def test_swapped_layers_are_relabelled_left_is_A(adir):
    write(adir, "rig_a.json", [CINE], electrode_A=V["electrode_B"], electrode_B=V["electrode_A"])
    cal, problems, _ = rbu.annotation_calibration(CINE, 256, 256)
    assert problems == [] and cal["ab_swapped_in_drawing"] is True


def test_incomplete_annotation_stops(adir):
    (adir / "rig_a.json").write_text(json.dumps(dict(cines=[str(CINE)], image_shape=[256, 256], gap_mm=2.4,
                                                     schema=ra.SCHEMA, electrode_A=None, electrode_B=None,
                                                     gap_line=None)))
    cal, problems, _ = rbu.annotation_calibration(CINE, 256, 256)
    assert cal is None and "incomplete" in problems[0]


def test_output_goes_to_a_separate_tree():
    assert rbu.out_dir_for(CINE, annotated=True).parts[-5] == "runs_annotated"
    assert rbu.out_dir_for(CINE).parts[-5] == "runs"
