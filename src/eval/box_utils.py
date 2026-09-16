"""
Bounding box math used throughout the project: IoU and non-max suppression.

These are implemented by hand (rather than imported from torchvision.ops or
ultralytics internals) because understanding what they compute is one of this
project's explicit learning goals — see notebooks/03_train_sliding_window.ipynb
for how they're used to turn a heatmap of window scores into a single box.

Boxes are (x1, y1, x2, y2) in absolute pixel coordinates, x1<x2 and y1<y2.
"""

from __future__ import annotations

import numpy as np


def box_area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes.

    IoU = area(A ∩ B) / area(A ∪ B). It's 0 for boxes that don't overlap and 1
    for identical boxes, which is what makes it a good scale-invariant measure
    of "did the model point at the right place" for both NMS (suppressing
    duplicate detections of the same object) and evaluation (is a predicted
    box close enough to the ground truth to count as a correct detection).
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = box_area(box_a) + box_area(box_b) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized IoU between every box in boxes_a and every box in boxes_b.

    Returns an (len(boxes_a), len(boxes_b)) matrix. Used for evaluation
    (matching predicted boxes to ground-truth boxes) where a python double
    loop over hundreds of candidate windows would be too slow.
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)

    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    with np.errstate(divide="ignore", invalid="ignore"):
        ious = np.where(union_area > 0, inter_area / union_area, 0.0)
    return ious


def non_max_suppression(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.3
) -> list[int]:
    """Greedy NMS: collapse a pile of overlapping candidate boxes into one
    box per real object.

    A sliding-window classifier fires on many overlapping windows around the
    same true Waldo (shifted by a few pixels, at several scales), each with
    its own confidence score. Non-max suppression is the standard fix:

    1. Take the highest-scoring remaining box.
    2. Keep it, and discard every other remaining box that overlaps it by
       more than `iou_threshold` (i.e. is almost certainly the same
       detection, not a different object).
    3. Repeat until no boxes remain.

    Returns the indices (into `boxes`/`scores`) of the boxes that survive,
    ordered from highest to lowest score.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        current = order[0]
        keep.append(int(current))

        if order.size == 1:
            break

        rest = order[1:]
        current_iou = np.array([iou(boxes[current], boxes[i]) for i in rest])
        order = rest[current_iou <= iou_threshold]

    return keep
