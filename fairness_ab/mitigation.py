"""Two standard mitigations, implemented from scratch (no fairness library).

1. Reweighing (Kamiran & Calders 2012, pre-processing): weight each (group, label) cell by
   P(group) * P(label) / P(group, label) so group and label look statistically independent in training.
2. Per-group thresholds (post-processing): choose a decision threshold for each group on a validation
   split so true-positive rates match (equal opportunity) -- no retraining.
"""
from __future__ import annotations

import numpy as np


def reweighing_weights(y: np.ndarray, group: np.ndarray) -> np.ndarray:
    w = np.ones(len(y), dtype=float)
    for g in np.unique(group):
        for lab in np.unique(y):
            cell = (group == g) & (y == lab)
            if cell.any():
                w[cell] = (group == g).mean() * (y == lab).mean() / cell.mean()
    return w


def equal_opportunity_thresholds(score, y, group, base_threshold: float = 0.5):
    """Keep group 0's threshold at base; pick group 1's so its TPR matches group 0's (on the data given)."""
    m0, m1 = group == 0, group == 1
    target = (score[m0 & (y == 1)] >= base_threshold).mean()
    cands = np.unique(np.quantile(score[m1 & (y == 1)], np.linspace(0, 1, 201)))
    best = min(cands, key=lambda t: abs((score[m1 & (y == 1)] >= t).mean() - target))
    return {0: base_threshold, 1: float(best)}


def apply_thresholds(score, group, thr: dict):
    t = np.where(group == 1, thr[1], thr[0])
    return (score >= t).astype(int)
