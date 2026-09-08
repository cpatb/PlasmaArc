"""A backup is not a recording: `mode_transition_analysis.find_recordings`.

The rule has been in master since 2026-06-29 and had no test. It matters because there are
45 `re_strikes_qa.csv` files under the data tree and 8 real recordings -- the rest are
per-recording backup dirs, two dated whole-tree backups, the figure package's `_tree`, and
ARCHIVE. An orphan worktree copy of this module lacks the rule; running both on the same
tree with the detector held fixed takes the pooled CO2 P(B|B) from 0.657 (persistent) to
0.418 (anti-persistent) against P(B|A) 0.629 -> 0.586, so the duplication REVERSES the sign
of the memory effect rather than merely inflating n.

See ORPHAN_WORKTREE_2026-09-08.md.
"""
from __future__ import annotations

import glob
import os

import pytest

import mode_transition_analysis as mta

DATA_ROOT = mta.DATA_ROOT_DEFAULT
needs_data = pytest.mark.skipif(not os.path.isdir(DATA_ROOT),
                                reason="Plasma Arc data tree not mounted")

BACKUP_MARKERS = ("__pre_3p2_backup__", "__pre_calib_rescale_backup__",
                  "__pre_resegment_snapshot__", "_FULL_BACKUP_pre_resegment",
                  "_PRE_ROBUSTNESS_REPROCESS", "_tree", "ARCHIVE")


@needs_data
def test_the_tree_really_does_contain_far_more_copies_than_recordings():
    """If this stops being true the test below is measuring nothing."""
    n_csv = sum(len(glob.glob(os.path.join(DATA_ROOT, sub, "**", "re_strikes_qa.csv"),
                              recursive=True))
                for sub in mta.GROUPS)
    assert n_csv > 20, n_csv


@needs_data
def test_discovery_finds_eight_recordings():
    recs = mta.find_recordings(DATA_ROOT)
    assert len(recs) == 8, sorted(r[0] for r in recs)
    assert sum(1 for r in recs if r[1] == "CO2 (J-electrode)") == 6
    assert sum(1 for r in recs if r[1] == "N2 (blade)") == 2


@needs_data
def test_no_discovered_path_sits_under_a_backup_or_snapshot_directory():
    for label, _group, csv in mta.find_recordings(DATA_ROOT):
        parts = csv.replace("\\", "/").split("/")[:-1]
        for marker in BACKUP_MARKERS:
            assert not any(marker in p for p in parts), (label, csv, marker)


@needs_data
def test_no_label_is_a_backup_directory_name():
    """Phantom recordings appear this way: the label is the parent's basename, so a
    backup directory becomes a 'recording' called `__pre_3p2_backup__`."""
    labels = {r[0] for r in mta.find_recordings(DATA_ROOT)}
    for bad in ("__pre_3p2_backup__", "__pre_calib_rescale_backup__",
                "__pre_resegment_snapshot__"):
        assert bad not in labels


@needs_data
def test_labels_are_unique_so_nothing_is_pooled_twice():
    recs = mta.find_recordings(DATA_ROOT)
    keys = [(r[1], r[0]) for r in recs]
    assert len(keys) == len(set(keys)), sorted(keys)


@needs_data
def test_a_legal_recording_name_with_underscores_survives():
    """A fail-closed filter must be shown NOT to reject something legal. Every real name
    here contains underscores; the rule keys on DEPTH, not on the name."""
    labels = {r[0] for r in mta.find_recordings(DATA_ROOT)}
    assert any("_repeat_weird" in x for x in labels), labels
    assert any("SLPM-N2" in x for x in labels), labels


@needs_data
def test_the_permissive_filter_would_reintroduce_the_duplication():
    """The discriminating half: with only the pre-2026-06-29 exclusions the count must
    jump back up. Without this, the test above passes for any filter that happens to
    return 8, including a wrong one."""
    src_root = os.path.join(DATA_ROOT, "J Electrode")
    all_csv = glob.glob(os.path.join(src_root, "**", "re_strikes_qa.csv"), recursive=True)
    permissive = [c for c in all_csv
                  if "ARCHIVE" not in c and "reference_5000" not in c]
    strict = [r[2] for r in mta.find_recordings(DATA_ROOT)
              if r[1] == "CO2 (J-electrode)"]
    assert len(permissive) > 4 * len(strict), (len(permissive), len(strict))
