from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA = Path(__file__).resolve().parent.parent / "data" / "sms.tsv"
SEED = 42


def load() -> pd.DataFrame:
    if not DATA.exists():
        raise FileNotFoundError(f"{DATA} missing: run nlp_serving/download_data.sh")
    df = pd.read_csv(DATA, sep="\t", header=None, names=["label", "text"], quoting=3,
                     encoding="utf-8").dropna()
    df["y"] = (df["label"] == "spam").astype(int)
    return df.reset_index(drop=True)


def split(df: pd.DataFrame):
    """Stratified 80/20 split with a fixed seed, shared by every model and the R check."""
    return train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["y"])
