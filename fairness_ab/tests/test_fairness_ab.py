import numpy as np

from fairness_ab import ab, metrics, mitigation


def test_reweighing_makes_label_independent_of_group():
    rng = np.random.default_rng(0)
    g = rng.integers(0, 2, 5000)
    y = (rng.random(5000) < np.where(g == 1, 0.1, 0.3)).astype(int)
    w = mitigation.reweighing_weights(y, g)
    rate = [np.average(y[g == k], weights=w[g == k]) for k in (0, 1)]
    assert abs(rate[0] - rate[1]) < 1e-9
    assert abs(w.sum() - len(y)) < 1e-6


def test_equal_opportunity_thresholds_match_tpr():
    rng = np.random.default_rng(1)
    g = rng.integers(0, 2, 20000)
    y = rng.integers(0, 2, 20000)
    score = np.clip(0.5 * y + rng.normal(0.25, 0.2, 20000) - 0.1 * g, 0, 1)
    thr = mitigation.equal_opportunity_thresholds(score, y, g)
    pred = mitigation.apply_thresholds(score, g, thr)
    r = metrics.group_rates(y, pred, g)
    assert abs(r[0]["tpr"] - r[1]["tpr"]) < 0.01


def test_disparity_zero_for_identical_predictions_across_groups():
    y = np.array([1, 0, 1, 0]); g = np.array([0, 0, 1, 1]); p = np.array([1, 0, 1, 0])
    d = metrics.disparities(y, p, g)
    assert d["demographic_parity_diff"] == 0 and d["equalized_odds_gap"] == 0


def test_ztest_detects_known_difference_and_null():
    assert ab.two_proportion_ztest(480, 4000, 540, 4000)["p_value"] < 0.05
    assert ab.two_proportion_ztest(500, 4000, 500, 4000)["p_value"] > 0.99


def test_sample_size_matches_textbook_value():
    assert 14000 < ab.sample_size_per_arm(0.10, 0.01) < 15500  # ~14.75k per arm for 10% -> 11%


def test_paired_bootstrap_ci_contains_observed_diff():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 2000)
    a = np.where(rng.random(2000) < 0.80, y, 1 - y)
    b = np.where(rng.random(2000) < 0.85, y, 1 - y)
    r = ab.paired_bootstrap(lambda yy, pp: float((yy == pp).mean()), y, a, b, n_boot=500)
    assert r["ci95"][0] <= r["observed_diff"] <= r["ci95"][1]


def test_peeking_inflates_false_positives():
    r = ab.peeking_false_positive_rate(n_sims=600, n_total=2000, looks=10, seed=3)
    assert r["peek_any_p<0.05"] > 2 * r["single_look"]
    assert r["peek_bonferroni"] < r["peek_any_p<0.05"]
