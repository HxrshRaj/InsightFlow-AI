"""UCI Adult (Census Income) loader. Source: UCI ML Repository (CC BY 4.0); fetched by download.sh."""
from __future__ import annotations

import numpy as np
import pandas as pd

COLS = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
        "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
NUM = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CAT = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "native_country"]


def load(path: str = "fairness_ab/adult.csv") -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLS, skipinitialspace=True, na_values="?")
    df[CAT] = df[CAT].fillna("Unknown")
    df["y"] = (df["income"].str.strip().str.rstrip(".") == ">50K").astype(int)
    df["female"] = (df["sex"] == "Female").astype(int)
    return df


def design_matrix(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """One-hot encode. The sensitive attribute (sex) is NOT a feature -- 'fairness through unawareness',
    which the audit shows is not sufficient on its own (correlated features still leak it)."""
    X = pd.get_dummies(df[NUM + CAT], columns=CAT, dtype=float)
    X[NUM] = (X[NUM] - X[NUM].mean()) / X[NUM].std()
    return X if columns is None else X.reindex(columns=columns, fill_value=0.0)


def split(df: pd.DataFrame, seed: int, test_frac: float = 0.3):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * test_frac)
    return df.iloc[idx[n_test:]].reset_index(drop=True), df.iloc[idx[:n_test]].reset_index(drop=True)
