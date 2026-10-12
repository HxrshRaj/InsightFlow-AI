"""Unsupervised structure in the Adult records: PCA, k-means (k chosen and checked for stability),
and anomaly detection with a PyTorch autoencoder vs IsolationForest.

python -m unsup_loss.unsupervised -> unsup_loss/results/unsupervised.json
"""
from __future__ import annotations

import json

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import adjusted_rand_score, silhouette_score

from fairness_ab import data


def autoencoder_scores(X: np.ndarray, hidden=16, epochs=60, seed=0) -> np.ndarray:
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    d = Xt.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.ReLU(), torch.nn.Linear(64, hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, 64), torch.nn.ReLU(), torch.nn.Linear(64, d))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            b = Xt[perm[i:i + 512]]
            opt.zero_grad()
            torch.nn.functional.mse_loss(net(b), b).backward()
            opt.step()
    with torch.no_grad():
        return ((net(Xt) - Xt) ** 2).mean(1).numpy()


def main(seed=0):
    df = data.load()
    X = data.design_matrix(df).to_numpy(dtype=np.float32)
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(X), 8000, replace=False)  # subsample for silhouette (O(n^2))
    out = {"rows": len(X), "features": X.shape[1]}

    pca = PCA().fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    out["pca"] = {"components_for_80pct_variance": int(np.searchsorted(cum, 0.80) + 1),
                  "components_for_90pct_variance": int(np.searchsorted(cum, 0.90) + 1)}
    Z = PCA(n_components=out["pca"]["components_for_90pct_variance"], random_state=seed).fit_transform(X)

    out["kmeans"] = {}
    for k in (2, 3, 4, 5, 6, 8):
        labs = [KMeans(k, n_init=3, random_state=s).fit(Z[sub]).labels_ for s in range(3)]
        sil = silhouette_score(Z[sub], labs[0])
        stab = float(np.mean([adjusted_rand_score(labs[0], labs[i]) for i in (1, 2)]))
        out["kmeans"][k] = {"silhouette": float(sil), "stability_ari_across_seeds": stab}
    best = max(out["kmeans"], key=lambda k: out["kmeans"][k]["silhouette"])
    out["kmeans_best_k_by_silhouette"] = int(best)
    km = KMeans(int(best), n_init=5, random_state=seed).fit(Z)
    prof = []
    for c in range(int(best)):
        m = km.labels_ == c
        prof.append({"cluster": c, "share": float(m.mean()), "positive_income_rate": float(df["y"][m].mean()),
                     "female_share": float(df["female"][m].mean()), "mean_age": float(df["age"][m].mean())})
    out["cluster_profiles_not_used_for_fitting"] = prof

    ae = autoencoder_scores(X, seed=seed)
    iso = -IsolationForest(random_state=seed, n_estimators=200).fit(X).score_samples(X)
    top = lambda s: set(np.argsort(s)[-int(0.01 * len(s)):])
    out["anomaly"] = {"spearman_autoencoder_vs_isolation_forest": float(spearmanr(ae, iso)[0]),
                      "top1pct_overlap": float(len(top(ae) & top(iso)) / len(top(ae)))}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    json.dump(main(), open("unsup_loss/results/unsupervised.json", "w"), indent=2)
