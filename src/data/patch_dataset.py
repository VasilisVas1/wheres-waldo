"""PyTorch Dataset over the materialized patch corpus from notebook 02.

Train-split patches get live augmentation applied per `__getitem__` call (so
each epoch sees a different augmented version of the same underlying crop);
validation-split patches are returned as-is, since a validation metric has to
be measured against a fixed target to be comparable across epochs.
"""

from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.data.augmentation import augment_patch, build_train_augmentation

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(image: np.ndarray) -> torch.Tensor:
    normalized = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(normalized.transpose(2, 0, 1)).float()


class PatchDataset(Dataset):
    def __init__(
        self,
        patch_manifest: pd.DataFrame,
        fold_assignment: pd.DataFrame,
        val_fold: int,
        split: str,
        augmenter: A.Compose | None = None,
    ):
        assert split in ("train", "val")
        merged = patch_manifest.merge(fold_assignment, on="scene_id", how="left")
        is_val_scene = merged["val_fold"] == val_fold
        self.rows = (merged[is_val_scene] if split == "val" else merged[~is_val_scene]).reset_index(
            drop=True
        )
        self.split = split
        self.augmenter = augmenter if augmenter is not None else build_train_augmentation()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows.iloc[idx]
        image = np.array(Image.open(row["path"]).convert("RGB"))
        if self.split == "train":
            image = augment_patch(image, self.augmenter)
        return to_tensor(image), torch.tensor(float(row["label"]))
