import math
import numpy as np
import pytest

from python import baseline, data, model_format as mf, tokenizer


def tiny_model():
    return mf.LinearTextModel(["free", "win", "free win", "héllo"], [1.5, 2.0, 3.0, 1.0], [0.7, 1.1, -0.3, 0.2], -0.4)


def test_tokenizer_rules():
    assert tokenizer.tokens("A b CD-ef 12") == ["cd", "ef", "12"]
    assert tokenizer.analyze("Free WIN!") == ["free", "win", "free win"]
    assert tokenizer.tokens("café") == ["caf"]  # non-ASCII bytes are separators by spec
    assert tokenizer.analyze("") == []


@pytest.mark.parametrize("order", ["little", "big"])
def test_round_trip(order):
    m = tiny_model()
    out = mf.read(mf.write(m, order))
    assert out == m


def test_orders_differ_on_disk_but_decode_same():
    m = tiny_model()
    le, be = mf.write(m, "little"), mf.write(m, "big")
    assert le != be and len(le) == len(be)
    assert mf.read(le) == mf.read(be)


def test_reader_rejects_bad_input():
    good = mf.write(tiny_model())
    with pytest.raises(ValueError):
        mf.read(b"XXXX" + good[4:])
    with pytest.raises(ValueError):
        mf.read(good + b"\0")
    with pytest.raises(ValueError):
        mf.write(mf.LinearTextModel(["a"], [1.0], [], 0.0))


def test_score_matches_sklearn():
    df = data.load().iloc[:1500]
    texts, labels = list(df["text"]), df["y"].to_numpy()
    vec, clf = baseline.train(texts, labels)
    model = baseline.to_model(vec, clf)
    ref = baseline.probabilities(vec, clf, texts[:100])
    mine = np.array([baseline.score(model, t) for t in texts[:100]])
    assert np.max(np.abs(ref - mine)) < 1e-9


def test_score_no_known_terms_is_bias_only():
    m = tiny_model()
    p = baseline.score(m, "zzzz qqqq")
    assert math.isclose(p, 1 / (1 + math.exp(0.4)), rel_tol=1e-12)
