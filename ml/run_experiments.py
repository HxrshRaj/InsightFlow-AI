"""Reproduce every number in docs/pytorch-model.md. Writes ml/results.json."""
import json
import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import train_pytorch as t

SEEDS = range(5)
out = {"hyperparameters": {k: list(v) if isinstance(v, tuple) else v for k, v in t.HP.items()}, "variants": {}}
for v in ["raw", "leaky", "fixed"]:
    runs = [t.run(v, s, verbose=False) for s in SEEDS]
    out["variants"][v] = {
        "runs": runs,
        **{f"{k}_{m}_{f}": float(getattr(np, f)([r[k][m] for r in runs]))
           for k in ("val", "test") for m in ("rmse", "r2") for f in ("mean", "std")},
    }
# Reference baselines on the identical seed-0 split (train-only scaling)
X_tr, X_va, X_te, y_tr, y_va, y_te = t.make_splits("fixed", 0)
out["baselines_seed0"] = {
    "predict_train_mean": t.metrics(np.full_like(y_te, y_tr.mean()), y_te),
    "ridge": t.metrics(Ridge().fit(X_tr, y_tr).predict(X_te), y_te),
}
json.dump(out, open(os.path.join(os.path.dirname(__file__), "results.json"), "w"), indent=2)
for v, d in out["variants"].items():
    print(v, {k: round(x, 4) for k, x in d.items() if k != "runs"})
print(out["baselines_seed0"])
