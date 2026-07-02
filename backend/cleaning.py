"""Data cleaning module for InsightFlow AI."""

from __future__ import annotations

from typing import Any

import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows without mutating the original dataframe."""
    return df.drop_duplicates().reset_index(drop=True)


def fill_missing_numeric(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Fill numeric missing values using the median by default."""
    cleaned = df.copy()
    numeric_cols = cleaned.select_dtypes(include="number").columns

    for column in numeric_cols:
        if cleaned[column].isnull().any():
            if strategy == "mean":
                fill_value = cleaned[column].mean()
            elif strategy == "median":
                fill_value = cleaned[column].median()
            elif strategy == "zero":
                fill_value = 0
            else:
                fill_value = cleaned[column].median()
            cleaned[column] = cleaned[column].fillna(fill_value)

    return cleaned


def fill_missing_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Fill categorical/object missing values using the mode or a fallback."""
    cleaned = df.copy()
    categorical_cols = cleaned.select_dtypes(include=["object", "string", "category"]).columns

    for column in categorical_cols:
        if cleaned[column].isnull().any():
            mode_values = cleaned[column].mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
            cleaned[column] = cleaned[column].fillna(fill_value)

    return cleaned


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns and convert low-cardinality objects to category."""
    cleaned = df.copy()

    for column in cleaned.columns:
        series = cleaned[column]

        if pd.api.types.is_integer_dtype(series) and not pd.api.types.is_bool_dtype(series):
            cleaned[column] = pd.to_numeric(series, downcast="integer")
        elif pd.api.types.is_float_dtype(series):
            cleaned[column] = pd.to_numeric(series, downcast="float")
        elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            unique_count = series.nunique(dropna=True)
            unique_ratio = unique_count / max(1, len(series))
            if unique_count < 20 or unique_ratio < 0.5:
                cleaned[column] = series.astype("category")

    return cleaned


def calculate_memory(df: pd.DataFrame) -> float:
    """Return dataframe memory usage in megabytes."""
    return float(df.memory_usage(deep=True).sum() / (1024**2))


def generate_cleaning_log(original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> list[str]:
    """Generate human-readable cleaning actions for the UI."""
    log: list[str] = []

    duplicates_removed = int(original_df.duplicated().sum())
    if duplicates_removed > 0:
        log.append(f"✓ Removed {duplicates_removed} duplicate rows")

    original_numeric_missing = int(original_df.select_dtypes(include="number").isna().sum().sum())
    cleaned_numeric_missing = int(cleaned_df.select_dtypes(include="number").isna().sum().sum())
    numeric_missing_fixed = max(0, original_numeric_missing - cleaned_numeric_missing)
    if numeric_missing_fixed > 0:
        log.append(f"✓ Filled {numeric_missing_fixed} numeric missing values")

    original_categorical_missing = int(
        original_df.select_dtypes(include=["object", "string", "category"]).isna().sum().sum()
    )
    cleaned_categorical_missing = int(
        cleaned_df.select_dtypes(include=["object", "string", "category"]).isna().sum().sum()
    )
    categorical_missing_fixed = max(0, original_categorical_missing - cleaned_categorical_missing)
    if categorical_missing_fixed > 0:
        log.append(f"✓ Filled {categorical_missing_fixed} categorical missing values")

    optimized_columns = sum(
        1
        for column in cleaned_df.columns
        if str(cleaned_df[column].dtype) != str(original_df[column].dtype)
    )
    if optimized_columns > 0:
        log.append(f"✓ Optimized {optimized_columns} columns")

    if not log:
        log.append("✓ No additional cleaning changes were required")

    return log


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Run the complete cleaning workflow and return the cleaned dataframe and report."""
    original_df = df.copy()
    cleaned_df = remove_duplicates(original_df)
    cleaned_df = fill_missing_numeric(cleaned_df)
    cleaned_df = fill_missing_categorical(cleaned_df)
    cleaned_df = optimize_dtypes(cleaned_df)

    before_memory = calculate_memory(original_df)
    after_memory = calculate_memory(cleaned_df)
    memory_saved_mb = round(max(0.0, before_memory - after_memory), 2)

    before_missing = int(original_df.isna().sum().sum())
    after_missing = int(cleaned_df.isna().sum().sum())
    missing_values_fixed = max(0, before_missing - after_missing)
    duplicates_removed = int(original_df.duplicated().sum())
    optimized_columns = sum(
        1
        for column in cleaned_df.columns
        if str(cleaned_df[column].dtype) != str(original_df[column].dtype)
    )

    def quality_score(frame: pd.DataFrame) -> float:
        if len(frame) == 0:
            return 0.0
        penalty = (int(frame.isna().sum().sum()) + int(frame.duplicated().sum())) / len(frame) * 100
        return round(max(0.0, 100.0 - penalty), 1)

    cleaning_summary: dict[str, Any] = {
        "rows_before": int(len(original_df)),
        "rows_after": int(len(cleaned_df)),
        "duplicates_removed": duplicates_removed,
        "missing_values_fixed": missing_values_fixed,
        "memory_saved_mb": memory_saved_mb,
        "memory_saved": memory_saved_mb,
        "columns_optimized": optimized_columns,
        "quality_score_before": quality_score(original_df),
        "quality_score_after": quality_score(cleaned_df),
    }
    cleaning_log = generate_cleaning_log(original_df, cleaned_df)

    return cleaned_df, cleaning_summary, cleaning_log


class DataCleaner:
    """Handles data cleaning and preprocessing operations."""

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return remove_duplicates(df)

    def fill_missing_numeric(self, df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
        return fill_missing_numeric(df, strategy=strategy)

    def fill_missing_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        return fill_missing_categorical(df)

    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        return optimize_dtypes(df)

    def strip_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        string_cols = cleaned.select_dtypes(include="object").columns
        for column in string_cols:
            cleaned[column] = cleaned[column].astype(str).str.strip()
        return cleaned

    def clean_dataset(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
        return clean_dataset(df)
