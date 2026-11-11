"""Torch datasets over the downloaded ISIC collections.

Reads the committed manifest and split assignment, so what a dataset yields is
fully determined by files in the repo -- no directory scanning, no implicit
ordering, nothing that changes when the filesystem does.

Targets come from `labels`: triage (primary) and lesion type (secondary).
"""

from __future__ import annotations

import io
import json
import pathlib
import random
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageFilter
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


class PhoneCapture:
    """Simulate what a phone camera does to a skin photo.

    Each effect is applied independently at random: a white-balance cast
    (per-channel gain), focus blur, loss of resolution (down- then up-scale),
    and JPEG compression. Off by default -- it is a hypothesis to be tested
    against a run without it, not an assumed improvement.
    """

    def __init__(self, p_cast=0.7, p_blur=0.5, p_lowres=0.4, p_jpeg=0.7):
        self.p_cast, self.p_blur, self.p_lowres, self.p_jpeg = p_cast, p_blur, p_lowres, p_jpeg

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() < self.p_cast:
            arr = np.asarray(img, dtype=np.float32)
            gains = np.array([random.uniform(0.85, 1.15) for _ in range(3)], dtype=np.float32)
            img = Image.fromarray(np.clip(arr * gains, 0, 255).astype(np.uint8))
        if random.random() < self.p_blur:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.8)))
        if random.random() < self.p_lowres:
            w, h = img.size
            f = random.uniform(0.3, 0.8)
            img = img.resize((max(1, int(w * f)), max(1, int(h * f))), Image.BILINEAR).resize((w, h), Image.BILINEAR)
        if random.random() < self.p_jpeg:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(30, 85))
            buf.seek(0)
            with Image.open(buf) as j:
                img = j.convert("RGB")
        return img


def build_transforms(train: bool, image_size: int = IMAGE_SIZE, phone_aug: bool = False):
    """Geometry and colour jitter, plus phone-capture effects when asked.

    The baseline was measured without phone augmentation; it is switched on
    per run (--phone-aug) so its effect can be measured rather than assumed.
    """
    if train:
        return transforms.Compose([
            *([PhoneCapture(), transforms.RandomPerspective(0.2, p=0.3)] if phone_aug else []),
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
        phone_aug: bool = False,
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
        self.transform = build_transforms(train=(split == "train"), image_size=image_size,
                                          phone_aug=phone_aug and split == "train")

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


class JointDataset(Dataset):
    """Several collections' copies of one split, pooled -- for joint training.

    Each record keeps its own collection's split assignment (patient- or lesion-
    level), so pooling never moves a lesion across train / val / test. Class
    weights are computed on the pooled counts, so HAM10000's benign majority and
    PAD-UFES-20's malignant majority (the inverted priors) are balanced together.
    """

    def __init__(self, collections: list[str], split, target="triage", **kw):
        self.parts = [ISICDataset(c, split, target=target, **kw) for c in collections]
        self.collection = "+".join(collections)
        self.split, self.target = split, target
        self.classes = self.parts[0].classes
        self.class_to_idx = self.parts[0].class_to_idx
        self._offsets = []
        n = 0
        for part in self.parts:
            self._offsets.append(n)
            n += len(part)
        self._n = n

    def targets(self) -> list[int]:
        return [t for part in self.parts for t in part.targets()]

    def source_counts(self) -> dict[str, int]:
        return {part.collection: len(part) for part in self.parts}

    class_counts = ISICDataset.class_counts
    class_weights = ISICDataset.class_weights

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int):
        for part, off in zip(reversed(self.parts), reversed(self._offsets)):
            if i >= off:
                return part[i - off]
        raise IndexError(i)
