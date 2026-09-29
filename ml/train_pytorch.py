"""Feedforward PyTorch regressor on California Housing (standalone; see docs/pytorch-model.md).

Variants (--variant) exist so the debugging cycle in the docs is reproducible:
  raw    : features fed to the network unscaled            (bug 1: normalization)
  leaky  : StandardScaler fit on ALL rows before splitting (bug 2: train/test leakage)
  fixed  : StandardScaler fit on the training split only   (correct)
"""
import argparse
import json

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

HP = dict(hidden=(64, 32), lr=1e-3, batch_size=256, epochs=200, patience=15, weight_decay=0.0)


def make_splits(variant, seed):
    X, y = fetch_california_housing(return_X_y=True)
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    if variant == "leaky":  # BUG: statistics computed from val/test rows too
        X = StandardScaler().fit_transform(X).astype(np.float32)
    # 70 / 15 / 15 split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=seed)
    if variant == "fixed":
        sc = StandardScaler().fit(X_tr)
        X_tr, X_va, X_te = (sc.transform(a).astype(np.float32) for a in (X_tr, X_va, X_te))
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def build_model(n_in, hidden):
    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def metrics(pred, y):
    mse = float(np.mean((pred - y) ** 2))
    return dict(rmse=mse ** 0.5, r2=1 - mse / float(np.var(y)))


def run(variant, seed=0, verbose=True, epochs=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_tr, X_va, X_te, y_tr, y_va, y_te = make_splits(variant, seed)
    model = build_model(X_tr.shape[1], HP["hidden"])
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    loss_fn = nn.MSELoss()
    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    Xv, Xe = torch.from_numpy(X_va), torch.from_numpy(X_te)
    best_val, best_state, bad, history = float("inf"), None, 0, []
    for epoch in range(1, (epochs or HP["epochs"]) + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(Xt), HP["batch_size"]):
            idx = perm[i:i + HP["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]).squeeze(1), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val = float(loss_fn(model(Xv).squeeze(1), torch.from_numpy(y_va)))
        history.append((total / len(Xt), val))
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"[{variant}] epoch {epoch:3d}  train_mse={history[-1][0]:.4f}  val_mse={val:.4f}")
        if val < best_val - 1e-5:
            best_val, bad = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= HP["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():  # test set touched exactly once, after model selection
        p_va = model(Xv).squeeze(1).numpy()
        p_te = model(Xe).squeeze(1).numpy()
    return dict(variant=variant, seed=seed, epochs_run=len(history),
                train_mse_last=history[-1][0], val=metrics(p_va, y_va), test=metrics(p_te, y_te))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["raw", "leaky", "fixed"], default="fixed")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    for s in a.seeds:
        print(json.dumps(run(a.variant, s, epochs=a.epochs), indent=2))
