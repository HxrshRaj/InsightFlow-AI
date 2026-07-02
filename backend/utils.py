"""Shared utility functions for InsightFlow AI."""

from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

import pandas as pd


def load_csv(source: BinaryIO | str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(source)


def format_file_size(size_bytes: int) -> str:
    """Format byte count as a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    return f"{size_bytes / (1024 ** 3):.1f} GB"


def get_dataframe_preview(df: pd.DataFrame, rows: int = 5) -> pd.DataFrame:
    """Return the first N rows of a DataFrame."""
    return df.head(rows)
