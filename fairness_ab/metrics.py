from __future__ import annotations

import numpy as np


def group_rates(y, pred, group) -> dict:
    out = {}
    for g in (0, 1):
        m = group == g
        yt, yp = y[m], pred[m]
        out[g] = {
            "n": int(m.sum()),
            "selection_rate": float(yp.mean()),
            "tpr": float(yp[yt == 1].mean()) if (yt == 1).any() else float("nan"),
            "fpr": float(yp[yt == 0].mean()) if (yt == 0).any() else float("nan"),
            "accuracy": float((yp == yt).mean()),
        }
    return out


def disparities(y, pred, group) -> dict:
    r = group_rates(y, pred, group)
    return {
        # group 1 = female, group 0 = male; differences are female minus male
        "demographic_parity_diff": r[1]["selection_rate"] - r[0]["selection_rate"],
        "equal_opportunity_diff": r[1]["tpr"] - r[0]["tpr"],
        "fpr_diff": r[1]["fpr"] - r[0]["fpr"],
        "equalized_odds_gap": max(abs(r[1]["tpr"] - r[0]["tpr"]), abs(r[1]["fpr"] - r[0]["fpr"])),
        "accuracy": float((pred == y).mean()),
    }
