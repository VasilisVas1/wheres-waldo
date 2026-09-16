"""Turn full scenes into fixed-size patches for the stage-1 sliding-window
classifier: positive crops (centered on a Waldo box) and negative crops
(random regions that don't overlap any box).

Design choices, and why:

- We generate a *bounded* number of negatives per scene (NEGATIVES_PER_SCENE),
  not every possible sliding-window position. Notebook 01 measured the true
  imbalance at roughly 1:2345 — training on that ratio directly would mean a
  trivial "always predict not-Waldo" classifier gets ~99.96% accuracy on the
  training set itself, giving the optimizer almost no useful gradient signal
  from positives. Undersampling negatives during training (while still
  keeping them the majority class) is standard practice for this kind of
  imbalance; we make up for it at evaluation time by running the classifier
  over *every* window of a held-out scene (true sliding-window inference),
  not just the undersampled set.
- We only generate one round of "easy" random negatives here. Hard-negative
  mining (specifically sampling crops the current model gets wrong) needs a
  trained model to mine against, so that happens in notebook 03 as a second
  round, bootstrapped from this first round's model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.eval.box_utils import iou

PATCH_SIZE = 64  # fixed input size for the stage-1 CNN classifier
NEGATIVE_IOU_THRESHOLD = 0.1  # a crop overlapping a box by more than this isn't a clean negative
NEGATIVES_PER_SCENE = 40
POSITIVE_CROPS_PER_BOX = 6  # jittered positive crops per ground-truth box


@dataclass
class Patch:
    image: np.ndarray  # HxWx3 uint8, already resized to PATCH_SIZE
    label: int  # 1 = waldo, 0 = not waldo
    scene_id: str
    source_box: tuple[int, int, int, int] | None  # None for negatives


def _crop_and_resize(image: Image.Image, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    crop = image.crop((x1, y1, x2, y2)).resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR)
    return np.array(crop.convert("RGB"))


def generate_positive_patches(
    image: Image.Image, boxes: list[tuple[int, int, int, int]], scene_id: str, rng: np.random.Generator
) -> list[Patch]:
    """For each ground-truth box, take several jittered square crops around it.

    Jittering (random padding + a small center offset) is the positive-class
    augmentation: it teaches the classifier that Waldo doesn't have to be
    perfectly centered or tightly framed in a window to count as a match,
    which matters because a real sliding window will rarely land exactly on
    the box.
    """
    patches = []
    for box in boxes:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        side = max(bw, bh)

        for _ in range(POSITIVE_CROPS_PER_BOX):
            pad = rng.uniform(0.15, 0.6) * side
            jitter_x = rng.uniform(-0.2, 0.2) * side
            jitter_y = rng.uniform(-0.2, 0.2) * side
            cx, cy = (x1 + x2) / 2 + jitter_x, (y1 + y2) / 2 + jitter_y
            half = side / 2 + pad

            crop_box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
            patches.append(
                Patch(
                    image=_crop_and_resize(image, crop_box),
                    label=1,
                    scene_id=scene_id,
                    source_box=box,
                )
            )
    return patches


def generate_negative_patches(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    scene_id: str,
    rng: np.random.Generator,
    n: int = NEGATIVES_PER_SCENE,
) -> list[Patch]:
    """Sample random square crops that don't meaningfully overlap any Waldo box.

    Crop sizes are drawn from a range around typical box sizes rather than a
    single fixed size, since a real sliding window will be run at multiple
    scales (notebook 03) and the classifier needs negative examples at those
    scales too, not just one.
    """
    W, H = image.width, image.height
    boxes_arr = [np.array(b) for b in boxes]

    patches = []
    attempts = 0
    while len(patches) < n and attempts < n * 20:
        attempts += 1
        side = rng.integers(20, 120)
        if side >= min(W, H):
            continue
        x1 = rng.integers(0, W - side)
        y1 = rng.integers(0, H - side)
        candidate = np.array([x1, y1, x1 + side, y1 + side])

        if boxes_arr and max(iou(candidate, b) for b in boxes_arr) > NEGATIVE_IOU_THRESHOLD:
            continue

        patches.append(
            Patch(
                image=_crop_and_resize(image, tuple(candidate)),
                label=0,
                scene_id=scene_id,
                source_box=None,
            )
        )
    return patches
