"""Detection metrics: precision/recall @ IoU threshold, and mean localization
error. Deliberately not plain accuracy — notebook 01 established that a
trivial "predict nothing" model would score ~99.96% accuracy under the true
window-level class imbalance, so accuracy can't distinguish a useful detector
from a useless one here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.eval.box_utils import iou


@dataclass
class DetectionMatch:
    true_positives: int
    false_positives: int
    false_negatives: int
    localization_errors: list[float]  # center-distance (px) for matched TPs


def match_detections(
    pred_boxes: list[tuple[int, int, int, int]],
    gt_boxes: list[tuple[int, int, int, int]],
    iou_threshold: float = 0.3,
) -> DetectionMatch:
    """Greedily match predicted boxes to ground-truth boxes.

    A predicted box counts as a true positive if its IoU with some
    not-yet-matched ground-truth box exceeds `iou_threshold` (each ground
    truth box can only be matched once, so duplicate detections of the same
    Waldo don't count as extra true positives).
    """
    matched_gt = set()
    tp = 0
    localization_errors = []

    for pred in pred_boxes:
        best_iou, best_gt_idx = 0.0, -1
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            score = iou(np.array(pred), np.array(gt))
            if score > best_iou:
                best_iou, best_gt_idx = score, gt_idx

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
            gt = gt_boxes[best_gt_idx]
            pred_center = ((pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2)
            gt_center = ((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2)
            localization_errors.append(float(np.hypot(*(np.array(pred_center) - np.array(gt_center)))))

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return DetectionMatch(true_positives=tp, false_positives=fp, false_negatives=fn, localization_errors=localization_errors)


def precision_recall(matches: list[DetectionMatch]) -> tuple[float, float]:
    tp = sum(m.true_positives for m in matches)
    fp = sum(m.false_positives for m in matches)
    fn = sum(m.false_negatives for m in matches)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def mean_localization_error(matches: list[DetectionMatch]) -> float | None:
    errors = [e for m in matches for e in m.localization_errors]
    return float(np.mean(errors)) if errors else None
