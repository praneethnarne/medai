"""Level 1 dataset: phone photos from PAD-UFES-20 and SCIN, labelled by group.

Every PAD-UFES-20 image is a spot or growth; SCIN supplies the other groups plus a small
number of spots and growths of its own. Both are phone photography, which keeps the
model in its primary domain (ADR 0004).

Known shortcut risk: almost all spot_growth images come from PAD-UFES-20, so a model
could learn "looks like a PAD photo" instead of "is a spot or growth". `sources` lets
the evaluation report SCIN's own spot_growth images separately to check for that.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import Dataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from data.dataset import IMAGE_SIZE, build_transforms  # noqa: E402
from data.groups import GROUPS  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]


class GroupDataset(Dataset):
    def __init__(self, split: Literal["train", "val", "test"],
                 image_size: int = IMAGE_SIZE, phone_aug: bool = False,
                 sources: tuple[str, ...] = ("pad_ufes_20", "scin")):
        self.split = split
        self.classes = GROUPS
        self.class_to_idx = {c: i for i, c in enumerate(GROUPS)}
        self.items: list[tuple[pathlib.Path, int, str]] = []   # path, target, source

        if "pad_ufes_20" in sources:
            base = ROOT / "data" / "pad_ufes_20"
            records = json.loads((base / "metadata.json").read_text())
            assign = json.loads((base / "splits.json").read_text())["assignment"]
            t = self.class_to_idx["spot_growth"]
            self.items += [(base / "images" / f"{r['isic_id']}.jpg", t, "pad_ufes_20")
                           for r in records if assign[r["isic_id"]] == split]

        if "scin" in sources:
            base = ROOT / "data" / "scin"
            manifest = json.loads((base / "manifest.json").read_text())
            assign = json.loads((base / "splits.json").read_text())["assignment"]
            self.items += [(base / "images" / r["image"], self.class_to_idx[r["group"]], "scin")
                           for r in manifest
                           if r["group"] and assign.get(r["case_id"]) == split]

        # SCIN has a known missing file (its README lists one); a download can also be
        # incomplete. Drop absent images rather than crash mid-epoch, and say how many.
        present = [it for it in self.items if it[0].exists()]
        self.missing = len(self.items) - len(present)
        if self.missing:
            print(f"GroupDataset[{split}]: {self.missing} listed image(s) not on disk, skipped")
        self.items = present

        self.transform = build_transforms(train=(split == "train"), image_size=image_size,
                                          phone_aug=phone_aug and split == "train")

    @property
    def sources(self) -> list[str]:
        return [s for _, _, s in self.items]

    def targets(self) -> list[int]:
        return [t for _, t, _ in self.items]

    def class_counts(self) -> torch.Tensor:
        counts = torch.zeros(len(self.classes))
        for t in self.targets():
            counts[t] += 1
        return counts

    def class_weights(self, scheme: str = "balanced") -> torch.Tensor:
        counts = self.class_counts()
        present = counts > 0
        w = torch.zeros(len(self.classes))
        w[present] = counts.sum() / (present.sum() * counts[present])
        if scheme == "sqrt":
            w[present] = w[present].sqrt()
        elif scheme != "balanced":
            raise ValueError(f"unknown weighting scheme: {scheme!r}")
        return w

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, target, _ = self.items[i]
        with Image.open(path) as img:
            x = self.transform(img.convert("RGB"))
        return x, target
