"""Data cleaning module for InsightFlow AI."""

from __future__ import annotations

import pandas as pd


class DataCleaner:
    """Handles data cleaning and preprocessing operations."""

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates()

    def fill_missing_numeric(self, df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
        cleaned = df.copy()
        numeric_cols = cleaned.select_dtypes(include="number").columns

        for col in numeric_cols:
            if strategy == "mean":
                cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
            elif strategy == "median":
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())
            elif strategy == "zero":
                cleaned[col] = cleaned[col].fillna(0)

        return cleaned

    def strip_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        string_cols = cleaned.select_dtypes(include="object").columns
        for col in string_cols:
            cleaned[col] = cleaned[col].astype(str).str.strip()
        return cleaned
