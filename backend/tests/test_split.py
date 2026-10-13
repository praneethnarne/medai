"""Splits must not leak.

Two photographs of the same lesion on opposite sides of a split make the test
set partly a memory test, and every metric computed afterwards is inflated. This
is the failure mode that silently invalidates a whole project, so it is pinned
here rather than trusted.
"""

import collections
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from data import labels  # noqa: E402
from data.split import group_key  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATASETS = ["ham10000", "pad_ufes_20"]


def load(name):
    manifest = ROOT / "data" / name / "metadata.json"
    splits = ROOT / "data" / name / "splits.json"
    if not (manifest.exists() and splits.exists()):
        pytest.skip(f"{name} not downloaded or not split")
    return json.loads(manifest.read_text()), json.loads(splits.read_text())


@pytest.mark.parametrize("name", DATASETS)
def test_no_group_spans_two_splits(name):
    """The core guarantee: one patient (or lesion) lives in exactly one split."""
    records, splits = load(name)
    unit = splits["split_unit"]
    where = collections.defaultdict(set)
    for r in records:
        where[group_key(r, unit)].add(splits["assignment"][r["isic_id"]])
    straddling = {g: s for g, s in where.items() if len(s) > 1}
    assert not straddling, f"{len(straddling)} {unit}s span multiple splits"


@pytest.mark.parametrize("name", DATASETS)
def test_every_image_assigned_exactly_once(name):
    records, splits = load(name)
    assignment = splits["assignment"]
    ids = [r["isic_id"] for r in records]
    assert len(assignment) == len(records)
    assert set(assignment) == set(ids)
    assert set(assignment.values()) == {"train", "val", "test"}


@pytest.mark.parametrize("name", DATASETS)
def test_every_triage_class_present_in_every_split(name):
    """A split missing a class makes its recall undefined rather than zero."""
    records, splits = load(name)
    seen = collections.defaultdict(set)
    for r in records:
        seen[splits["assignment"][r["isic_id"]]].add(labels.triage_of(r))
    for split in ("train", "val", "test"):
        assert seen[split] == set(labels.TRIAGE), f"{split} missing {set(labels.TRIAGE) - seen[split]}"


@pytest.mark.parametrize("name", DATASETS)
def test_splits_are_reproducible(name):
    """Re-running the splitter must reproduce the committed assignment."""
    from data.split import split_one
    _, before = load(name)
    split_one(name)
    after = json.loads((ROOT / "data" / name / "splits.json").read_text())
    assert before["assignment"] == after["assignment"]
