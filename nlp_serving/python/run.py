"""Train baseline + TensorFlow model, write metrics, predictions, model files and Go golden data.

    python -m python.run            (from nlp_serving/)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from python import baseline, metrics, model_format
from python.data import load, split
from python.tf_model import fit_predict

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TESTDATA = ROOT / "go" / "testdata"
EDGE_CASES = [
    "", "   ", "FREE entry!!! Call 08000 now", "Ünïcödé tëxt with ASCII words and 12345 numbers",
    "a b c d e", "WINNER!! claim your prize today", "ok see you at 5", "x" * 500,
]


def main():
    RESULTS.mkdir(exist_ok=True)
    TESTDATA.mkdir(parents=True, exist_ok=True)
    df = load()
    train, test = split(df)

    vec, clf = baseline.train(train["text"], train["y"])
    p_base = baseline.probabilities(vec, clf, test["text"])
    mdl = baseline.to_model(vec, clf)

    p_tf = fit_predict(train["text"], train["y"], test["text"])

    out = {"dataset": {"rows": len(df), "spam": int(df["y"].sum()), "train": len(train),
                       "test": len(test)},
           "tfidf_logreg": metrics.summarise(test["y"], p_base),
           "tensorflow_keras": metrics.summarise(test["y"], p_tf),
           "n_features": len(mdl.terms)}
    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({"y": test["y"].values, "p_tfidf": p_base, "p_tf": p_tf}).to_csv(
        RESULTS / "test_predictions.csv", index=False)

    (TESTDATA / "model_le.bin").write_bytes(model_format.write(mdl, "little"))
    (TESTDATA / "model_be.bin").write_bytes(model_format.write(mdl, "big"))
    sample = list(test["text"].head(200)) + EDGE_CASES
    golden = [{"text": t, "p": baseline.score(mdl, t)} for t in sample]
    (TESTDATA / "golden.json").write_text(json.dumps(golden))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
