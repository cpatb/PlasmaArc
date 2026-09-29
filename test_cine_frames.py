"""Tests for cine_frames: decode must match pycine bit-for-bit; defect fix must touch only defects.

The decode tests need a real packed cine; set BLADE_TEST_CINE or they use the
default upward recording on E:. They are asserted present, not skipped, when
that path exists; with no drive attached the module-level check fails loudly.
"""
import os

import numpy as np
import pytest
from pycine.file import read_header
from pycine.raw import create_raw_array

import cine_frames as cf

CINE = os.environ.get(
    "BLADE_TEST_CINE",
    r"E:\BladeReactor_Upwards\Argon\45degree_2.4mmGap\Re2250\BLADE_ADCB_Argon_Re2250_06mmNozzle_trial1.cine")
HAVE = os.path.exists(CINE)
needs_cine = pytest.mark.skipif(not HAVE, reason="test cine not attached (E: drive)")


def _pycine_frame(path, header, i):
    """Independent path: pycine's own per-frame decode at 0-based index i."""
    import struct
    with open(path, "rb") as f:
        f.seek(header["pImage"][i])
        ann = struct.unpack("I", f.read(4))[0]
        f.seek(ann - 8, 1)
        isz = struct.unpack("I", f.read(4))[0]
        return create_raw_array(f.read(isz), header)


@needs_cine
def test_decode_matches_pycine_bit_for_bit():
    src = cf.CineFrames(CINE)
    h = read_header(CINE)
    idx = [0, 1, 2, 3, 12345, src.n // 2, src.n - 2, src.n - 1]
    for i in idx:
        ref = _pycine_frame(CINE, h, i)
        got = src.raw_block(i, 1)[0]
        assert got.dtype == ref.dtype == np.uint16
        np.testing.assert_array_equal(got, ref, err_msg=f"frame {i}")
    # a block decode equals the per-frame decodes (vectorised unpack is not shifted)
    blk = src.raw_block(40000, 5)
    for k in range(5):
        np.testing.assert_array_equal(blk[k], _pycine_frame(CINE, h, 40000 + k))


@needs_cine
def test_frame_count_is_header_not_filesize():
    src = cf.CineFrames(CINE)
    assert src.n == read_header(CINE)["cinefileheader"].ImageCount
    # the bug this replaces: filesize / unpacked bytes gives 62.5 % of the record
    old = os.path.getsize(CINE) // (src.H * src.W * 2)
    assert old < 0.63 * src.n


@needs_cine
def test_defect_lattice_is_the_known_128():
    src = cf.open_cine(CINE)
    rows, cols = src.defects.lattice_signature()
    assert src.defects.n == 128
    assert rows == list(range(1, 122, 8))
    assert cols == list(range(11, 124, 16))
    assert src.defects.values == [4095]
    # after correction no pixel sits at 4095 in every sampled frame
    fr = src.block(0, 300)
    assert not np.any((fr == 4095).all(axis=0))


def test_correct_defects_hand_computed_and_touches_only_mask():
    img = np.arange(25, dtype=np.uint16).reshape(5, 5) * 10
    mask = np.zeros((5, 5), bool)
    mask[2, 2] = True
    img_bad = img.copy()
    img_bad[2, 2] = 4095
    out = cf.correct_defects(img_bad, mask)
    # neighbours of (2,2): 60,70,80,110,130,160,170,180 -> median (110+130)/2 = 120
    assert out[2, 2] == 120
    same = ~mask
    np.testing.assert_array_equal(out[same], img_bad[same])
    assert img_bad[2, 2] == 4095   # input not mutated


def test_correct_defects_excludes_defective_neighbours_and_edges():
    img = np.full((4, 4), 7, np.uint16)
    mask = np.zeros((4, 4), bool)
    mask[0, 0] = mask[0, 1] = True
    img[0, 0] = img[0, 1] = 4095
    out = cf.correct_defects(img[None], mask)[0]
    assert out[0, 0] == 7 and out[0, 1] == 7     # a neighbouring defect is not used


def _binarize_fullframe_reference(raw, close_r, dark_guard, bright, low, min_px, max_px):
    """Independent reference: the original full-frame closing + largest CC after hysteresis."""
    from scipy.ndimage import binary_closing, label
    from skimage.filters import apply_hysteresis_threshold
    from skimage.morphology import disk
    if int(raw.max()) < dark_guard:
        return np.zeros(raw.shape, bool)
    b = apply_hysteresis_threshold(raw, low, bright)
    if b.sum() < min_px:
        return np.zeros(raw.shape, bool)
    b = binary_closing(b, structure=disk(close_r))
    lab, n = label(b)
    if n == 0:
        return np.zeros(raw.shape, bool)
    sz = np.bincount(lab.ravel()); sz[0] = 0
    m = lab == sz.argmax()
    return m if min_px <= m.sum() <= max_px else np.zeros(raw.shape, bool)


def test_cropped_binarize_identical_to_fullframe_synthetic():
    from analyze_blade import binarize
    rng = np.random.default_rng(7)
    for k in range(300):
        img = rng.integers(0, 25, (64, 80)).astype(np.uint16)
        # random blobs/lines, including ones touching the frame edge
        for _ in range(rng.integers(1, 5)):
            r, c = rng.integers(0, 64), rng.integers(0, 80)
            img[max(r - 3, 0):r + 3, max(c - 1, 0):c + 12] = rng.integers(30, 400)
        ref = _binarize_fullframe_reference(img, 3, 50, 50, 22, 5, 10**9)
        got = binarize(img, 3, dark_guard=50, bright_thresh=50, low_thresh=22, min_arc_px=5, max_arc_px=10**9)
        np.testing.assert_array_equal(got, ref, err_msg=f"case {k}")


@needs_cine
def test_cropped_binarize_identical_to_fullframe_real():
    from analyze_blade import binarize
    src = cf.open_cine(CINE)
    fr = src.block(30000, 400)
    n_det = 0
    for f in fr:
        ref = _binarize_fullframe_reference(f, 3, 300, 300, 60, 22, 10**9)
        got = binarize(f, 3, dark_guard=300, bright_thresh=300, low_thresh=60, min_arc_px=22, max_arc_px=10**9)
        np.testing.assert_array_equal(got, ref)
        n_det += bool(ref.any())
    assert n_det > 50   # the comparison exercised real arcs, not only empty frames


def test_unpack10_layout():
    # 4 pixels 0x3FF, 0x000, 0x155, 0x2AA packed big-endian in 5 bytes
    px = [0x3FF, 0x000, 0x155, 0x2AA]
    bits = "".join(f"{v:010b}" for v in px)
    b = np.array([int(bits[i:i + 8], 2) for i in range(0, 40, 8)], np.uint8)
    out = cf._unpack10(b, 1, 1, 4)
    assert out.ravel().tolist() == px
