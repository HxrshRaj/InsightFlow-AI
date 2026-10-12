# fairness_ab/ — bias audit, mitigation, and A/B experimentation

Standalone module (not part of the Streamlit app). Dataset: UCI Adult / Census Income, 48,842 rows, predict
income > $50K. Sensitive attribute: sex (33% female; positive rate 30.4% male vs 10.9% female).

```
bash fairness_ab/download.sh
python -m fairness_ab.audit      # -> results/fairness_audit.json
python -m fairness_ab.ab         # -> results/ab_results.json
pytest fairness_ab/tests
```

## 1. Bias audit and mitigation (`audit.py`, `mitigation.py`, `metrics.py`)

Logistic regression, sex NOT used as a feature, 5 random 70/30 splits (mean ± std). Differences are female minus male.

| Method | Demographic-parity diff | Equal-opportunity (TPR) diff | Equalized-odds gap | Accuracy |
|---|---|---|---|---|
| Baseline | -0.173 ± 0.002 | -0.087 ± 0.015 | 0.089 | 0.852 ± 0.002 |
| Reweighing (pre-processing) | -0.089 ± 0.004 | **+0.136 ± 0.021** | 0.136 | 0.847 ± 0.002 |
| Per-group threshold (post-processing) | -0.150 ± 0.009 | **-0.001 ± 0.040** | 0.058 | 0.851 ± 0.003 |

What this shows:
- **Leaving sex out is not enough**: the baseline still selects women far less often (-17 pts), because the remaining
  features, and the labels themselves, still carry the group difference. (Which features drive it was not analyzed.)
- **Reweighing** cuts the demographic-parity gap roughly in half for a 0.5-pt accuracy cost, but over-corrects on
  true-positive rate (women now get higher TPR than men, +13.6 pts). Fixing one fairness metric can break another.
- **Per-group thresholds** equalize TPR almost exactly with no accuracy loss, but leave a selection-rate gap.
  Different fairness definitions cannot all be satisfied at once when base rates differ; the choice is a policy
  decision, not a modeling one.
- Both methods are implemented from scratch (Kamiran & Calders reweighing; TPR-matching thresholds picked on a
  held-out validation slice). Caveat: Adult is a 1994 census sample, and "sex" is recorded as binary.

## 2. A/B testing toolkit (`ab.py`)

- Two-proportion z-test with CI; per-arm sample-size / power calculation (10% baseline, detect +1 pt: **14,751 per
  arm**; detect +2 pts: 3,841).
- **Offline model A/B** on the same test set (14,652 rows): logistic regression (A) vs gradient boosting (B):
  accuracy 0.849 vs 0.868, paired-bootstrap 95% CI of the difference [+1.4, +2.3] pts; McNemar test on discordant
  predictions (326 vs 596) p < 1e-18.
- **Peeking simulation** (A/A tests, no true effect, 4,000 simulations, 10 looks): a single end-of-test look gives a
  4.9% false-positive rate (as designed); stopping at the first p < 0.05 peek gives **19.7%**; a Bonferroni-corrected
  rule gives 2.5%. This is a simulation of the statistics, not an analysis of real user traffic.

## Limits
- No real online experiment was run; the A/B section is statistical tooling plus an offline comparison.
- Single dataset, single sensitive attribute, linear baseline; no intersectional (sex x race) analysis.
