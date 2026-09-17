"""Multi-scale sliding-window inference: turn a trained patch classifier into
a detector over a full scene.

This is the "smarter, learned version of slide + classify + NMS" the project
brief points to as the conceptual bridge to YOLO (notebook 04). Every step is
explicit here rather than hidden in a framework call:

1. Slide square windows of several sizes across the image (multi-scale,
   because we don't know Waldo's size in a new scene ahead of time).
2. Classify every window (batched, for speed on CPU).
3. Keep windows above a confidence threshold.
4. Collapse overlapping detections with the from-scratch NMS in
   `src/eval/box_utils.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.data.patch_dataset import to_tensor
from src.data.patch_generation import PATCH_SIZE
from src.eval.box_utils import non_max_suppression

DEFAULT_WINDOW_SIZES = (32, 48, 64, 96, 128)
DEFAULT_STRIDE_FRACTION = 0.5
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    score: float


def windows_for_scale(width: int, height: int, size: int, stride: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for y in range(0, max(1, height - size + 1), stride):
        for x in range(0, max(1, width - size + 1), stride):
            boxes.append((x, y, x + size, y + size))
    return boxes


@torch.no_grad()
def detect(
    model: nn.Module,
    image: Image.Image,
    window_sizes: tuple[int, ...] = DEFAULT_WINDOW_SIZES,
    stride_fraction: float = DEFAULT_STRIDE_FRACTION,
    score_threshold: float = 0.5,
    nms_iou_threshold: float = 0.2,
    batch_size: int = 256,
    device: str = DEFAULT_DEVICE,
) -> list[Detection]:
    model.to(device)
    model.eval()
    W, H = image.width, image.height

    all_boxes: list[tuple[int, int, int, int]] = []
    for size in window_sizes:
        stride = max(1, int(size * stride_fraction))
        all_boxes.extend(windows_for_scale(W, H, size, stride))

    all_scores = np.zeros(len(all_boxes), dtype=np.float32)
    for start in range(0, len(all_boxes), batch_size):
        batch_boxes = all_boxes[start : start + batch_size]
        crops = [
            image.crop(b).resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR) for b in batch_boxes
        ]
        tensors = torch.stack([to_tensor(np.array(c.convert("RGB"))) for c in crops]).to(device)
        logits = model(tensors)
        scores = torch.sigmoid(logits).cpu().numpy()
        all_scores[start : start + len(batch_boxes)] = scores

    keep_mask = all_scores >= score_threshold
    kept_boxes = np.array(all_boxes)[keep_mask]
    kept_scores = all_scores[keep_mask]

    if len(kept_boxes) == 0:
        return []

    keep_idx = non_max_suppression(kept_boxes, kept_scores, iou_threshold=nms_iou_threshold)
    return [Detection(box=tuple(kept_boxes[i]), score=float(kept_scores[i])) for i in keep_idx]
