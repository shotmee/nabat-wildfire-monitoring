"""Competition score, computed exactly like the organizers compute it:
pixels from every chip are pooled into one confusion matrix per class
(micro-averaging) *before* precision/recall/F1/IoU are derived - not
averaged per-chip and then across chips.

Score = 0.35 * F1_af + 0.35 * IoU_burn + 0.30 * mIoU_sev
"""
from __future__ import annotations

import numpy as np


class ConfusionAccumulator:
    """Pools TP/FP/FN across an arbitrary number of update() calls (= chips)."""

    def __init__(self, classes: tuple[int, ...]):
        self.classes = tuple(classes)
        self.tp = {c: 0 for c in self.classes}
        self.fp = {c: 0 for c in self.classes}
        self.fn = {c: 0 for c in self.classes}

    def update(self, pred: np.ndarray, true: np.ndarray, valid: np.ndarray | None = None) -> None:
        if valid is None:
            valid = np.ones_like(true, dtype=bool)
        for c in self.classes:
            p = (pred == c) & valid
            t = (true == c) & valid
            self.tp[c] += int(np.sum(p & t))
            self.fp[c] += int(np.sum(p & ~t))
            self.fn[c] += int(np.sum(~p & t))

    def iou(self, c: int) -> float:
        tp, fp, fn = self.tp[c], self.fp[c], self.fn[c]
        denom = tp + fp + fn
        return 1.0 if denom == 0 else tp / denom

    def precision_recall_f1(self, c: int) -> tuple[float, float, float]:
        tp, fp, fn = self.tp[c], self.fp[c], self.fn[c]
        if tp + fp == 0 and tp + fn == 0:
            return 1.0, 1.0, 1.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return precision, recall, f1


def binarize_burn(mask: np.ndarray) -> np.ndarray:
    """BS mask (0/1/2/3) -> binary burned/not-burned mask, for IoU_burn."""
    return (mask >= 1).astype(np.uint8)


def f1_af(acc: ConfusionAccumulator) -> float:
    return acc.precision_recall_f1(1)[2]


def iou_burn(acc_binary: ConfusionAccumulator) -> float:
    """acc_binary must have been fed binarize_burn() masks with classes=(1,)."""
    return acc_binary.iou(1)


def miou_severity(acc_sev: ConfusionAccumulator) -> float:
    """acc_sev must have been fed the raw 0..3 masks with classes=(1, 2, 3)."""
    return sum(acc_sev.iou(c) for c in (1, 2, 3)) / 3


def score(f1_af_value: float, iou_burn_value: float, miou_sev_value: float) -> float:
    return 0.35 * f1_af_value + 0.35 * iou_burn_value + 0.30 * miou_sev_value
