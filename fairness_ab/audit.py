"""Bias audit + mitigation on UCI Adult, sensitive attribute = sex.

python -m fairness_ab.audit  -> fairness_ab/results/fairness_audit.json
Repeats over several random splits and reports mean +/- std.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from fairness_ab import data, mitigation, metrics


def one_split(df, seed):
    train, test = data.split(df, seed)
    # validation slice of train used ONLY to pick the post-processing thresholds
    fit, val = data.split(train, seed + 1000, test_frac=0.25)
    Xf = data.design_matrix(fit)
    cols = Xf.columns
    Xv, Xt = data.design_matrix(val, cols), data.design_matrix(test, cols)
    yf, yv, yt = fit["y"].to_numpy(), val["y"].to_numpy(), test["y"].to_numpy()
    gf, gv, gt = fit["female"].to_numpy(), val["female"].to_numpy(), test["female"].to_numpy()

    res = {}
    base = LogisticRegression(max_iter=1000).fit(Xf, yf)
    res["baseline"] = metrics.disparities(yt, base.predict(Xt), gt)

    rw = LogisticRegression(max_iter=1000).fit(Xf, yf, sample_weight=mitigation.reweighing_weights(yf, gf))
    res["reweighing"] = metrics.disparities(yt, rw.predict(Xt), gt)

    thr = mitigation.equal_opportunity_thresholds(base.predict_proba(Xv)[:, 1], yv, gv)
    pred = mitigation.apply_thresholds(base.predict_proba(Xt)[:, 1], gt, thr)
    res["group_thresholds"] = metrics.disparities(yt, pred, gt)
    res["group_thresholds"]["female_threshold"] = thr[1]
    res["_rates_baseline"] = metrics.group_rates(yt, base.predict(Xt), gt)
    return res


def main(path="fairness_ab/adult.csv", seeds=(0, 1, 2, 3, 4)):
    df = data.load(path)
    runs = [one_split(df, s) for s in seeds]
    out = {"rows": int(len(df)), "female_share": float(df["female"].mean()),
           "positive_rate_male": float(df.loc[df.female == 0, "y"].mean()),
           "positive_rate_female": float(df.loc[df.female == 1, "y"].mean()), "seeds": list(seeds), "methods": {}}
    for name in ("baseline", "reweighing", "group_thresholds"):
        out["methods"][name] = {
            k: {"mean": float(np.mean([r[name][k] for r in runs])), "std": float(np.std([r[name][k] for r in runs]))}
            for k in runs[0][name]
        }
    out["baseline_group_rates_last_split"] = runs[-1]["_rates_baseline"]
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    res = main()
    with open("fairness_ab/results/fairness_audit.json", "w") as f:
        json.dump(res, f, indent=2)
