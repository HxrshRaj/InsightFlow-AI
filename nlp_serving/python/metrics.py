from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def summarise(y, p, threshold: float = 0.5) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"threshold": threshold, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "precision": float(precision_score(y, pred)), "recall": float(recall_score(y, pred)),
            "f1": float(f1_score(y, pred)), "auc": float(roc_auc_score(y, p))}
