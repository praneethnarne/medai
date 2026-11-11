"""JointDataset pools collections without moving any record across splits."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from data.dataset import ISICDataset, JointDataset


def test_joint_is_the_sum_of_its_parts():
    parts = ["ham10000", "pad_ufes_20"]
    joint = JointDataset(parts, "train", target="triage")
    singles = [ISICDataset(c, "train", target="triage") for c in parts]
    assert len(joint) == sum(len(s) for s in singles)
    assert joint.targets() == [t for s in singles for t in s.targets()]
    assert joint.class_counts().tolist() == sum(s.class_counts() for s in singles).tolist()
    assert joint.source_counts() == {c: len(s) for c, s in zip(parts, singles)}


def test_joint_splits_do_not_overlap():
    parts = ["ham10000", "pad_ufes_20"]
    ids = {sp: {r["isic_id"] for p in JointDataset(parts, sp).parts for r in p.records}
           for sp in ("train", "val", "test")}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]
