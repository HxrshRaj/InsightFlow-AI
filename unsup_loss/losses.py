"""Loss functions compared in PyTorch (all written out, not library presets where it matters).

Classification (imbalanced: 24% positive): BCE, class-weighted BCE, focal loss, hinge.
Regression with heavy-tailed target noise: MSE, MAE, Huber.

python -m unsup_loss.losses -> unsup_loss/results/losses.json
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from fairness_ab import data


def bce(logit, y, pos_weight=None):
    return F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)


def focal(logit, y, gamma=2.0, alpha=0.25):
    p = torch.sigmoid(logit)
    pt = torch.where(y == 1, p, 1 - p)
    at = torch.where(y == 1, torch.full_like(p, alpha), torch.full_like(p, 1 - alpha))
    return (-at * (1 - pt) ** gamma * torch.log(pt.clamp_min(1e-8))).mean()


def hinge(logit, y):
    return torch.clamp(1 - (2 * y - 1) * logit, min=0).mean()


def huber(pred, y, delta=1.0):
    return F.huber_loss(pred, y, delta=delta)


def fit(X, y, loss_fn, epochs=80, lr=3e-3, seed=0, hidden=32):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 1024):
            b = perm[i:i + 1024]
            opt.zero_grad()
            loss_fn(net(X[b]).squeeze(1), y[b]).backward()
            opt.step()
    return lambda Xn: net(Xn).squeeze(1).detach()


def classification(seed):
    df = data.load()
    tr, te = data.split(df, seed)
    Xtr = data.design_matrix(tr)
    Xte = torch.tensor(data.design_matrix(te, Xtr.columns).to_numpy(np.float32))
    Xtr_t = torch.tensor(Xtr.to_numpy(np.float32))
    ytr, yte = torch.tensor(tr["y"].to_numpy(np.float32)), te["y"].to_numpy()
    pw = torch.tensor((1 - ytr.mean()) / ytr.mean())
    losses = {"bce": bce, "bce_class_weighted": lambda l, y: bce(l, y, pw), "focal": focal, "hinge": hinge}
    res = {}
    for name, fn in losses.items():
        score = fit(Xtr_t, ytr, fn, seed=seed)(Xte).numpy()
        pred = (score > 0).astype(int)  # logit 0 == p 0.5 for the BCE family; margin 0 for hinge
        res[name] = {"roc_auc": roc_auc_score(yte, score), "pr_auc": average_precision_score(yte, score),
                     "f1_at_default_threshold": f1_score(yte, pred), "predicted_positive_rate": float(pred.mean())}
    return res


def regression(seed):
    """Target: hours_per_week (z-scored). Training labels get heavy-tailed noise (10% of rows get +/- 5 sd error);
    the test labels stay clean, so the table shows robustness to label noise."""
    df = data.load()
    tr, te = data.split(df, seed)
    Xtr = data.design_matrix(tr).drop(columns=["hours_per_week"], errors="ignore")
    Xte = data.design_matrix(te, Xtr.columns)
    mu, sd = tr["hours_per_week"].mean(), tr["hours_per_week"].std()
    ytr = ((tr["hours_per_week"] - mu) / sd).to_numpy(np.float32)
    yte = ((te["hours_per_week"] - mu) / sd).to_numpy(np.float32)
    rng = np.random.default_rng(seed)
    bad = rng.random(len(ytr)) < 0.10
    ytr_noisy = ytr + bad * rng.choice([-5.0, 5.0], len(ytr)).astype(np.float32)
    Xtr_t, Xte_t, ytr_t = (torch.tensor(Xtr.to_numpy(np.float32)), torch.tensor(Xte.to_numpy(np.float32)),
                           torch.tensor(ytr_noisy))
    res = {}
    for name, fn in {"mse": F.mse_loss, "mae": F.l1_loss, "huber": huber}.items():
        pred = fit(Xtr_t, ytr_t, fn, seed=seed)(Xte_t).numpy()
        res[name] = {"mae_sd_units": float(np.abs(pred - yte).mean()), "rmse_sd_units": float(np.sqrt(((pred - yte) ** 2).mean()))}
    return res


def main(seeds=(0, 1, 2)):
    agg = lambda runs: {m: {k: {"mean": float(np.mean([r[m][k] for r in runs])), "std": float(np.std([r[m][k] for r in runs]))}
                            for k in runs[0][m]} for m in runs[0]}
    out = {"seeds": list(seeds), "classification": agg([classification(s) for s in seeds]),
           "regression_noisy_labels": agg([regression(s) for s in seeds])}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    json.dump(main(), open("unsup_loss/results/losses.json", "w"), indent=2)
