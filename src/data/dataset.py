"""Strict image, mask and description matching for QaTa-COV19-v2."""
import csv
import random
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset
from monai.transforms import (Compose, EnsureChannelFirstd, LoadImaged,
                              NormalizeIntensityd, RandZoomd, Resized, ToTensord)


def group_key(mask_name: str) -> str:
    """Group repeated sub-S... acquisitions by patient; retain unknown IDs as unique."""
    match = re.match(r"mask_(sub-[^_]+)", mask_name)
    return match.group(1) if match else mask_name


def patient_group_partition(names, seed=2026):
    """Return deterministic disjoint train/validation patient-group keys."""
    groups = sorted({group_key(name) for name in names})
    random.Random(seed).shuffle(groups)
    boundary = int(0.8 * len(groups))
    return set(groups[:boundary]), set(groups[boundary:])


def read_manifest(csv_path: Path, root: Path, split: str, seed=2026):
    csv_path, root = Path(csv_path), Path(root)
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation or test")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Annotation CSV missing: {csv_path}")
    folder = root / ("Test" if split == "test" else "Train")
    for name in ("Images", "GTs"):
        if not (folder / name).is_dir():
            raise FileNotFoundError(f"QaTa dataset directory missing: {folder / name}")
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["Image", "Description"]:
            raise ValueError(f"Expected Image,Description columns in {csv_path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No annotations in {csv_path}")
    names = [r["Image"] for r in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate image annotations in {csv_path}")
    result = []
    for row in rows:
        name, prompt = row["Image"], row["Description"].strip()
        if not name.startswith("mask_") or Path(name).name != name or not prompt:
            raise ValueError(f"Invalid mask name or empty prompt: {name!r}")
        image = folder / "Images" / name.removeprefix("mask_")
        mask = folder / "GTs" / name
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"Missing image/mask pair: {image}, {mask}")
        result.append({"name": name, "image": image, "mask": mask, "prompt": prompt})
    if split != "test":
        train_groups, validation_groups = patient_group_partition(
            (row["name"] for row in result), seed)
        chosen = train_groups if split == "train" else validation_groups
        result = [row for row in result if group_key(row["name"]) in chosen]
        if not result:
            raise ValueError(f"No records in patient-grouped {split} split")
    return result


def image_transform(size=224):
    # Matches the released model's MONAI/PILReader orientation and normalization.
    return Compose([LoadImaged(["image"], reader="PILReader"),
                    EnsureChannelFirstd(["image"]),
                    Resized(["image"], spatial_size=(size, size), mode="bicubic"),
                    NormalizeIntensityd(["image"], channel_wise=True),
                    ToTensord(["image"])])


def paired_transform(size=224, augment=False):
    ops = [LoadImaged(["image", "gt"], reader="PILReader"),
           EnsureChannelFirstd(["image", "gt"])]
    if augment:
        ops.append(RandZoomd(["image", "gt"], min_zoom=0.95, max_zoom=1.2,
                             mode=("bicubic", "nearest"), prob=0.1))
    ops += [Resized(["image"], spatial_size=(size, size), mode="bicubic"),
            Resized(["gt"], spatial_size=(size, size), mode="nearest"),
            NormalizeIntensityd(["image"], channel_wise=True),
            ToTensord(["image", "gt"])]
    return Compose(ops)


class QaTaDataset(Dataset):
    def __init__(self, csv_path, root, split, tokenizer, image_size=224, augment=False,
                 max_text_length=24, seed=2026):
        self.records = read_manifest(csv_path, root, split, seed=seed)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.transform = paired_transform(image_size, augment and split == "train")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        try:
            item = self.transform({"image": str(record["image"]), "gt": str(record["mask"])})
        except Exception as exc:
            raise ValueError(f"Cannot decode image/mask pair {record['name']}: {exc}") from exc
        mask = (item["gt"] > 0).to(torch.float32)
        image = item["image"].float()
        if image.ndim != 3 or mask.ndim != 3 or image.shape[0] != 1 or mask.shape[0] != 1 or image.shape[-2:] != mask.shape[-2:]:
            raise ValueError(f"Invalid image/mask dimensions for {record['name']}")
        if not torch.isfinite(image).all() or not torch.isfinite(item["gt"]).all():
            raise FloatingPointError(f"Non-finite image/mask for {record['name']}")
        tokens = self.tokenizer(record["prompt"], padding="max_length",
                                max_length=self.max_text_length, truncation=True,
                                return_attention_mask=True, return_tensors="pt")
        text = {key: tokens[key].squeeze(0) for key in ("input_ids", "attention_mask")}
        return (image, text), mask
