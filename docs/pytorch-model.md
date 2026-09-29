# PyTorch model: training, a real bug, and honest evaluation

> **Scope note.** InsightFlow AI's app (Streamlit, validation, cleaning, dashboards, LLM insights) has no
> prediction task and no other training pipeline. This is a **standalone module** in `ml/`, trained on the public
> **California Housing** dataset (20,640 rows, 8 numeric features, target = median house value in $100k) via
> `sklearn.datasets.fetch_california_housing`. It is not wired into `app.py`.

## Setup

| | |
|---|---|
| Task | Regression: predict `MedHouseVal` |
| Split | 70 / 15 / 15 train / val / test, `random_state = seed` |
| Model | MLP `8 -> 64 -> ReLU -> 32 -> ReLU -> 1` |
| Loss / optimizer | `MSELoss`, Adam, lr `1e-3`, weight decay 0 |
| Batch size / epochs | 256 / max 200, early stopping on **validation** MSE (patience 15), best-val weights restored |
| Test set | Evaluated once, after model selection |
| Seeds | 0-4 (mean ± std over 5 seeds); single-seed numbers are seed 0 |

Reproduce everything: `pip install -r ml/requirements.txt && cd ml && python run_experiments.py`
(writes `ml/results.json`; a from-scratch rerun gave identical numbers). `python diagnose_raw.py` reproduces the
diagnosis table below.

Reference points (seed 0 test set): predicting the train mean gives RMSE 1.156, R² -0.001; Ridge regression gives RMSE 0.741, R² 0.589.

## The bug

**Found naturally, not deliberately introduced.** My first training script fed the raw features to the network
with no scaling at all (variant `raw`). I did not plan this one; it was the first thing I ran.

**Symptoms (seed 0, `raw`).** Training was unstable. Train MSE went 0.66 (epoch 30) -> 3.53 (epoch 40) -> 0.67
(epoch 50) -> 0.79 (epoch 70), validation MSE swung between 0.63 and 1.88, and early stopping ended at epoch 79.
Final: val RMSE 0.778 / R² 0.545, test RMSE 0.773 / R² 0.553. That's *worse than the Ridge baseline* (R² 0.589), which
is a strong signal the network was not fitting properly.

**Wrong first diagnoses.** Spiky loss looks like "learning rate too high", and slow progress looks like "needs
more epochs". I tested both (seed 0, validation metrics):

| Run | epochs | val RMSE | val R² |
|---|---|---|---|
| raw (baseline) | 79 | 0.778 | 0.545 |
| raw, lr 1e-4 (LR hypothesis) | 129 | 0.794 | 0.526 |
| raw, 300 epochs, no early stop (epochs hypothesis) | 300 | 0.682 | 0.650 |
| raw, lr 1e-4, 300 epochs | 300 | 0.756 | 0.570 |
| **fixed (scaler fit on train only)** | 113 | **0.542** | **0.779** |

Lower LR did not help. More epochs helped partly but was still far from the fixed result, and cost ~4x the compute.
Neither hypothesis explained the instability.

**Root cause.** Feature scales differ by orders of magnitude (Population up to 35,682 and AveOccup up to 1,243, versus
MedInc ~4 and AveBedrms ~1). Unscaled inputs make gradients dominated by a few large-magnitude features, so a single Adam
step size is wrong for most inputs. It's an input-normalization problem, not an optimization-schedule problem.

**Fix.** `StandardScaler` fit on the **training split only**, applied to val and test (variant `fixed`).

## Before / after (same 5 seeds, same splits, same hyperparameters)

| Variant | val RMSE | test RMSE | test R² |
|---|---|---|---|
| `raw` (bug) | 0.762 ± 0.022 | 0.755 ± 0.026 | 0.562 ± 0.034 |
| `fixed` | 0.523 ± 0.014 | 0.528 ± 0.004 | 0.786 ± 0.008 |

Seed 0 alone: test RMSE 0.773 -> 0.522, test R² 0.553 -> 0.796. Test RMSE improves by ~30% across all seeds and the run-to-run
spread also shrinks (0.026 -> 0.004). The fix was verified by re-running from scratch, not by a single lucky seed.

## Data leakage: I checked it, and here it did *not* matter

The brief suggested scaler leakage as a debugging case. I tested it (variant `leaky`: `StandardScaler` fit on all rows
before splitting) and it **did not produce inflated metrics** on this dataset:

| Variant | val RMSE | test RMSE | test R² |
|---|---|---|---|
| `leaky` | 0.523 ± 0.013 | 0.532 ± 0.006 | 0.783 ± 0.011 |
| `fixed` | 0.523 ± 0.014 | 0.528 ± 0.004 | 0.786 ± 0.008 |

The difference is within seed noise. With 20k rows, mean/std estimated from 70% vs 100% of the data are nearly identical, so
this leak barely leaks. It's still the wrong pattern (and could matter with small data, or with target-aware
preprocessing), and `fixed` is the correct pipeline, but I'm not claiming a leakage bug that didn't show up in the numbers.

## What this demonstrates

- A model that "trains without errors" can be badly wrong: the `raw` run completed, produced a plausible-looking R² of 0.56, and would have shipped
  had I not compared against a simple baseline and looked at the loss curve.
- Diagnose before tuning: the two intuitive fixes (LR, epochs) each cost compute and did not address the cause.
- Correct evaluation means fitting preprocessing on train only, selecting on validation, touching test once, reporting mean ± std over seeds,
  and being willing to report that a suspected problem (scaler leakage) had no measurable effect.
- Caveat: the diagnosis table used validation metrics, but I also looked at seed-0 test numbers during exploration; the reported test
  numbers are from the final scripted runs and were not used to choose hyperparameters.
