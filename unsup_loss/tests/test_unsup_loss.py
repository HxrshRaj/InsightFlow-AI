import numpy as np
import torch

from unsup_loss import losses, unsupervised


def test_focal_reduces_to_scaled_bce_when_gamma_zero():
    logit = torch.tensor([2.0, -1.0, 0.5, -3.0]); y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    p = torch.sigmoid(logit)
    expected = -(0.25 * y * torch.log(p) + 0.75 * (1 - y) * torch.log(1 - p)).mean()
    assert torch.isclose(losses.focal(logit, y, gamma=0.0, alpha=0.25), expected, atol=1e-6)


def test_focal_downweights_easy_examples():
    easy = losses.focal(torch.tensor([6.0]), torch.tensor([1.0]), alpha=0.5)
    hard = losses.focal(torch.tensor([-2.0]), torch.tensor([1.0]), alpha=0.5)
    assert hard > 100 * easy


def test_hinge_zero_beyond_margin():
    assert losses.hinge(torch.tensor([2.0, -2.0]), torch.tensor([1.0, 0.0])) == 0


def test_huber_between_mae_and_mse_behaviour():
    pred, y = torch.tensor([10.0]), torch.tensor([0.0])
    assert losses.huber(pred, y) < torch.nn.functional.mse_loss(pred, y)
    assert abs(losses.huber(pred, y).item() - 9.5) < 1e-6  # delta * (|e| - delta/2)


def test_autoencoder_scores_flag_planted_outlier():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (400, 6)).astype(np.float32)
    X[:, 3:] = X[:, :3]  # redundant structure the autoencoder can learn
    X[0] = [5, -5, 5, -5, 5, -5]  # breaks the structure
    s = unsupervised.autoencoder_scores(X, hidden=4, epochs=60)
    assert s[0] == s.max()
