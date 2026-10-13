"""The label mapping must cover every value both datasets actually contain.

This is the test that stops a silent relabelling: if the ISIC archive renames a
category or adds one, these fail loudly rather than letting records slip into
the wrong class.
"""

import json
import pathlib

import pytest

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from data import labels  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATASETS = ["ham10000", "pad_ufes_20"]


def load(name):
    path = ROOT / "data" / name / "metadata.json"
    if not path.exists():
        pytest.skip(f"{name} manifest not downloaded")
    return json.loads(path.read_text())


@pytest.mark.parametrize("name", DATASETS)
def test_every_record_maps_to_a_triage_label(name):
    for record in load(name):
        assert labels.triage_of(record) in labels.TRIAGE


@pytest.mark.parametrize("name", DATASETS)
def test_every_record_maps_to_a_lesion_type(name):
    for record in load(name):
        assert labels.lesion_type_of(record) in labels.LESION_TYPES


def test_shared_types_appear_in_both_datasets():
    seen = {name: {labels.lesion_type_of(r) for r in load(name)} for name in DATASETS}
    for lesion_type in labels.SHARED_TYPES:
        for name in DATASETS:
            assert lesion_type in seen[name], f"{lesion_type} missing from {name}"


def test_ham_only_types_are_absent_from_pad():
    pad_types = {labels.lesion_type_of(r) for r in load("pad_ufes_20")}
    for lesion_type in labels.HAM_ONLY_TYPES:
        assert lesion_type not in pad_types


def test_unknown_label_is_fatal():
    bogus = {"metadata": {"clinical": {"diagnosis_1": "Spicy", "diagnosis_2": "Spicy"}}}
    with pytest.raises(labels.UnknownLabel):
        labels.triage_of(bogus)
    with pytest.raises(labels.UnknownLabel):
        labels.lesion_type_of(bogus)


def test_every_lesion_type_has_a_display_name():
    for lesion_type in labels.LESION_TYPES:
        assert labels.DISPLAY_NAMES[lesion_type]
