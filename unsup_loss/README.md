# unsup_loss/ — unsupervised structure and loss-function comparison

Same UCI Adult data as `fairness_ab/` (48,842 rows, 106 one-hot/numeric features). Run:

```
bash fairness_ab/download.sh
python -m unsup_loss.unsupervised     # -> results/unsupervised.json
python -m unsup_loss.losses           # -> results/losses.json
pytest unsup_loss/tests
```

## Unsupervised (`unsupervised.py`)
- **PCA**: 13 components keep 80% of variance, 22 keep 90%.
- **k-means on the 22 PCA components** (8,000-row subsample for silhouette; each k fit with 3 seeds):
  silhouette is low everywhere (0.11 to 0.14; best k = 6 at 0.143), so the data has **weak cluster structure**.
  Stability (adjusted Rand index across seeds) varies a lot (k=2: 0.97, k=3: 0.45, k=5: 0.88, k=6: 0.77), so I
  report both instead of trusting silhouette alone.
- Cluster profiles (income, sex and age were not used for fitting) differ clearly: e.g. a 25% cluster with mean age 25
  and 1.6% high income vs a 23% cluster with 46% high income; one tiny cluster (0.5%) is 100% high income.
- **Anomaly detection**: PyTorch autoencoder (reconstruction error) vs IsolationForest: Spearman 0.69, but only 27%
  overlap in the top 1% flagged, so the two detectors disagree on what is anomalous. An autoencoder trained long
  enough also starts to memorize outliers it was trained on (observed in a unit test), so epochs matter.

## Loss functions (`losses.py`, PyTorch, 3 seeds, mean ± std)
Classification, 24% positives, small MLP:

| Loss | ROC-AUC | PR-AUC | F1 @ default threshold | Predicted-positive rate |
|---|---|---|---|---|
| BCE | 0.908 ± 0.001 | 0.772 ± 0.005 | 0.672 | 20.7% |
| BCE, class-weighted | 0.908 ± 0.002 | 0.772 ± 0.004 | 0.682 | 35.5% |
| Focal (gamma 2, alpha 0.25) | 0.903 ± 0.002 | 0.763 ± 0.005 | 0.515 | 10.3% |
| Hinge | 0.888 ± 0.003 | 0.757 ± 0.003 | 0.657 | 19.2% |

Class weighting and focal loss mainly move the decision threshold, not the ranking quality: AUC is about equal for
BCE and weighted BCE. Focal's low F1 is a calibration/threshold effect (it under-predicts positives), not worse
ranking; its AUC is within 0.005 of BCE. Plain BCE was already the best ranker here.

Regression (predict `hours_per_week`; 10% of TRAINING labels corrupted by +/-5 sd, test labels clean):

| Loss | MAE (sd units) | RMSE (sd units) |
|---|---|---|
| MSE | 0.682 ± 0.008 | 0.946 ± 0.013 |
| MAE (L1) | **0.556 ± 0.005** | 0.887 ± 0.010 |
| Huber (delta 1) | 0.606 ± 0.006 | **0.880 ± 0.010** |

MSE is the least robust to the corrupted labels; L1 wins on MAE and Huber wins narrowly on RMSE. The label noise
is synthetic and is injected on purpose to show the effect.

## Limits
Single dataset; one small MLP architecture; clusters are weak, so cluster "profiles" are descriptive, not
segments to act on; thresholds were not tuned per loss.
