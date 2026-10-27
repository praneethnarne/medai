"""Torch datasets over the downloaded ISIC collections.

Reads the committed manifest and split assignment, so what a dataset yields is
fully determined by files in the repo -- no directory scanning, no implicit
ordering, nothing that changes when the filesystem does.

Targets come from `labels`: triage (primary) and lesion type (secondary).
"""

from __future__ import annotations

import json
import pathlib
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from data import labels  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]

# ImageNet statistics -- the backbones we fine-tune were pretrained with these.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


def build_transforms(train: bool, image_size: int = IMAGE_SIZE):
    """Geometry and colour jitter only.

    Phone-capture augmentation (blur, white balance, JPEG artefacts) belongs to
    the cross-domain phase and is deliberately not here: the baseline has to be
    measured before we start changing what it sees.
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.85, 1.18)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class ISICDataset(Dataset):
    """One split of one collection.

    target: "triage" -> 3 classes, the primary task
            "lesion_type" -> 8 classes, secondary
    """

    def __init__(
        self,
        collection: str,
        split: Literal["train", "val", "test"],
        target: Literal["triage", "lesion_type"] = "triage",
        image_size: int = IMAGE_SIZE,
        shared_types_only: bool = False,
    ):
        base = ROOT / "data" / collection
        records = json.loads((base / "metadata.json").read_text())
        assignment = json.loads((base / "splits.json").read_text())["assignment"]

        self.collection = collection
        self.split = split
        self.target = target
        self.classes = (labels.TRIAGE if target == "triage" else labels.LESION_TYPES)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.records = [
            r for r in records
            if assignment[r["isic_id"]] == split
            and (not shared_types_only or labels.is_shared_type(r))
        ]
        self.images = base / "images"
        self.transform = build_transforms(train=(split == "train"), image_size=image_size)

    def label_of(self, record) -> str:
        return (labels.triage_of(record) if self.target == "triage"
                else labels.lesion_type_of(record))

    def targets(self) -> list[int]:
        """All integer targets, in order -- for class weights and samplers,
        without paying to decode a single image."""
        return [self.class_to_idx[self.label_of(r)] for r in self.records]

    def class_counts(self) -> torch.Tensor:
        counts = torch.zeros(len(self.classes))
        for t in self.targets():
            counts[t] += 1
        return counts

    def class_weights(self, scheme: str = "balanced") -> torch.Tensor:
        """Per-class loss weights.

        "balanced" is sklearn's definition, n_total / (n_classes * n_c), which
        leaves the expected loss scale unchanged. An earlier version rescaled
        these to mean 1 across classes, which is wrong when one class is tiny:
        HAM10000's indeterminate class has 109 training images against benign's
        6,802, and that rescaling drove the majority weight down to 0.04 while
        leaving a 62:1 spread.

        "sqrt" softens the same weights for cases where balanced is so extreme
        that a handful of rare-class images dominate every gradient step. Which
        one to use is an empirical question, so both are available and the
        choice is recorded in the run config.
        """
        counts = self.class_counts()
        present = counts > 0
        weights = torch.zeros(len(self.classes))
        weights[present] = counts.sum() / (present.sum() * counts[present])
        if scheme == "sqrt":
            weights[present] = weights[present].sqrt()
        elif scheme != "balanced":
            raise ValueError(f"unknown weighting scheme: {scheme!r}")
        return weights

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        record = self.records[i]
        path = self.images / f"{record['isic_id']}.jpg"
        with Image.open(path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        return tensor, self.class_to_idx[self.label_of(record)]
