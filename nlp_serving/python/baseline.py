"""TF-IDF + logistic regression baseline, scoring done by hand so the exact same arithmetic
runs in Go (term counts * idf, L2 normalise, dot with coefficients, sigmoid)."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from python.model_format import LinearTextModel
from python.tokenizer import analyze


def train(texts, y, min_df: int = 2, C: float = 10.0):
    vec = TfidfVectorizer(analyzer=analyze, min_df=min_df, norm="l2", smooth_idf=True)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced", solver="liblinear")
    clf.fit(X, y)
    return vec, clf


def to_model(vec: TfidfVectorizer, clf: LogisticRegression) -> LinearTextModel:
    vocab = vec.vocabulary_
    terms = [None] * len(vocab)
    for t, i in vocab.items():
        terms[i] = t
    return LinearTextModel(terms, [float(x) for x in vec.idf_],
                           [float(x) for x in clf.coef_[0]], float(clf.intercept_[0]))


def score(model: LinearTextModel, text: str) -> float:
    """Reference scorer mirroring the Go implementation line by line."""
    index = {t: i for i, t in enumerate(model.terms)}
    counts = Counter(t for t in analyze(text) if t in index)
    weights = {index[t]: c * model.idf[index[t]] for t, c in counts.items()}
    norm = math.sqrt(sum(w * w for w in weights.values()))
    z = model.bias
    if norm > 0:
        z += sum(model.coef[i] * w / norm for i, w in weights.items())
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


def probabilities(vec, clf, texts) -> np.ndarray:
    return clf.predict_proba(vec.transform(texts))[:, 1]
