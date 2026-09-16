"""Patch-level augmentation for the stage-1 classifier.

Applied on top of the jittered crops from patch_generation.py. Each transform
targets a specific failure mode we'd otherwise have too little data to learn
to ignore, given only 19 source scenes:

- Flip / rotate: Waldo can be facing/oriented in ways not present in our 19
  scenes; the classifier should recognize the same visual pattern regardless
  of mirroring.
- Color jitter: puzzle pages are scanned/printed with variable brightness and
  color balance; without this, the classifier could latch onto exact color
  values from our tiny training set rather than the shape/texture pattern.
- Scale jitter (RandomResizedCrop): our sliding window will run at multiple
  scales at inference time (notebook 03), so training patches need to
  simulate "Waldo slightly too big/small for the window" too.
"""

from __future__ import annotations

import albumentations as A
import numpy as np

from src.data.patch_generation import PATCH_SIZE


def build_train_augmentation() -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.Rotate(limit=25, border_mode=0, p=0.5),
            A.RandomResizedCrop(
                size=(PATCH_SIZE, PATCH_SIZE), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=0.7
            ),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.7),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
        ]
    )


def augment_patch(image: np.ndarray, augmenter: A.Compose) -> np.ndarray:
    return augmenter(image=image)["image"]
