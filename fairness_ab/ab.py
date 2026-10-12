"""A/B testing toolkit: z-test, sample-size/power, paired-bootstrap offline model comparison,
and a simulation showing why peeking at a running test inflates false positives.

python -m fairness_ab.ab  -> fairness_ab/results/ab_results.json
"""
from __future__ import annotations

import json
import math

import numpy as np
from scipy import stats


def two_proportion_ztest(x1: int, n1: int, x2: int, n2: int, alpha: float = 0.05) -> dict:
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se0 if se0 > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    zc = stats.norm.ppf(1 - alpha / 2)
    return {"p1": p1, "p2": p2, "diff": p2 - p1, "z": z, "p_value": p, "ci": (p2 - p1 - zc * se, p2 - p1 + zc * se)}


def sample_size_per_arm(p_base: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-arm n to detect an absolute lift `mde_abs` over baseline rate `p_base` (two-sided, two proportions)."""
    p2 = p_base + mde_abs
    pbar = (p_base + p2) / 2
    za, zb = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)
    num = (za * math.sqrt(2 * pbar * (1 - pbar)) + zb * math.sqrt(p_base * (1 - p_base) + p2 * (1 - p2))) ** 2
    return math.ceil(num / mde_abs**2)


def paired_bootstrap(metric, y, pred_a, pred_b, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap the metric difference (B - A) over the SAME resampled examples (paired)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = metric(y[idx], pred_b[idx]) - metric(y[idx], pred_a[idx])
    return {"observed_diff": float(metric(y, pred_b) - metric(y, pred_a)),
            "ci95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))}


def peeking_false_positive_rate(n_sims=4000, n_total=4000, looks=10, p=0.1, alpha=0.05, seed=0) -> dict:
    """A/A tests (both arms identical, true effect = 0). Compare error rates for:
    one look at the end, stopping at the first significant peek, and a Bonferroni-corrected peek rule."""
    rng = np.random.default_rng(seed)
    checkpoints = np.linspace(n_total // looks, n_total, looks, dtype=int)
    hits = {"single_look": 0, "peek_any_p<0.05": 0, "peek_bonferroni": 0}
    for _ in range(n_sims):
        a, b = rng.random(n_total) < p, rng.random(n_total) < p
        pvals = []
        for n in checkpoints:
            pvals.append(two_proportion_ztest(int(a[:n].sum()), n, int(b[:n].sum()), n)["p_value"])
        hits["single_look"] += pvals[-1] < alpha
        hits["peek_any_p<0.05"] += min(pvals) < alpha
        hits["peek_bonferroni"] += min(pvals) < alpha / looks
    return {k: v / n_sims for k, v in hits.items()} | {"n_sims": n_sims, "looks": looks, "arm_size": n_total}


def offline_model_ab(seed=0) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    from fairness_ab import data

    df = data.load()
    train, test = data.split(df, seed)
    Xtr = data.design_matrix(train)
    Xte = data.design_matrix(test, Xtr.columns)
    ytr, yte = train["y"].to_numpy(), test["y"].to_numpy()
    a = LogisticRegression(max_iter=1000).fit(Xtr, ytr).predict(Xte)
    b = HistGradientBoostingClassifier(random_state=seed).fit(Xtr, ytr).predict(Xte)
    acc = lambda y, p: float((y == p).mean())
    out = {"test_rows": int(len(yte)), "acc_A_logreg": acc(yte, a), "acc_B_gradient_boosting": acc(yte, b)}
    out["accuracy_diff_B_minus_A"] = paired_bootstrap(acc, yte, a, b, seed=seed)
    # McNemar exact-ish check on discordant pairs
    b01 = int(((a == yte) & (b != yte)).sum())  # A right, B wrong
    b10 = int(((a != yte) & (b == yte)).sum())  # A wrong, B right
    out["mcnemar"] = {"A_right_B_wrong": b01, "A_wrong_B_right": b10,
                      "p_value": float(stats.binomtest(b10, b10 + b01, 0.5).pvalue)}
    return out


def main():
    res = {
        "z_test_example": two_proportion_ztest(480, 4000, 540, 4000),
        "sample_size_per_arm": {"baseline_10pct_detect_+1pt": sample_size_per_arm(0.10, 0.01),
                                "baseline_10pct_detect_+2pt": sample_size_per_arm(0.10, 0.02)},
        "offline_model_comparison": offline_model_ab(),
        "peeking_simulation_A_vs_A": peeking_false_positive_rate(),
    }
    print(json.dumps(res, indent=2))
    json.dump(res, open("fairness_ab/results/ab_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
