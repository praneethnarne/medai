import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
ROOT = pathlib.Path(__file__).resolve().parents[2]

from data.groups import GROUPS, SCIN_TO_GROUP  # noqa: E402

needs_scin = pytest.mark.skipif(not (ROOT / "data/scin/manifest.json").exists(),
                                reason="SCIN manifest not built")
needs_images = pytest.mark.skipif(
    not (ROOT / "data/scin/images").is_dir() or not (ROOT / "data/pad_ufes_20/images").is_dir(),
    reason="SCIN / PAD-UFES-20 images not downloaded")


def test_mapping_only_targets_known_groups():
    assert set(SCIN_TO_GROUP.values()) <= set(GROUPS)


@needs_scin
def test_scin_splits_keep_each_case_on_one_side():
    manifest = json.loads((ROOT / "data/scin/manifest.json").read_text())
    assign = json.loads((ROOT / "data/scin/splits.json").read_text())["assignment"]
    mapped_cases = {r["case_id"] for r in manifest if r["group"]}
    assert mapped_cases == set(assign)            # every mapped case has exactly one split
    assert set(assign.values()) == {"train", "val", "test"}


@needs_scin
@needs_images
def test_group_dataset_has_every_group_in_every_split():
    pytest.importorskip("torch")
    from data.group_dataset import GroupDataset

    for split in ("train", "val", "test"):
        counts = GroupDataset(split).class_counts()
        assert (counts > 0).all(), (split, counts)


@needs_scin
@needs_images
def test_sources_flag_pad_images_as_spot_growth_only():
    pytest.importorskip("torch")
    from data.group_dataset import GroupDataset

    ds = GroupDataset("val")
    spot = ds.class_to_idx["spot_growth"]
    for (_, t, src) in ds.items:
        if src == "pad_ufes_20":
            assert t == spot
