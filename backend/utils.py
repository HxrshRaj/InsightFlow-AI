"""Shared utility functions for InsightFlow AI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, BinaryIO

import pandas as pd


@dataclass
class DatasetSummary:
    row_count: int
    column_count: int
    memory_mb: float
    missing_values: int
    duplicate_rows: int
    numeric_count: int
    categorical_count: int
    column_info: pd.DataFrame


def load_csv(uploaded_file: BinaryIO) -> pd.DataFrame:
    """Load an uploaded CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded file contains no data.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Unable to parse CSV file: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(
            "Unable to decode file. Please ensure the file uses UTF-8 encoding."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Failed to load CSV file: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")

    return df


def calculate_memory(df: pd.DataFrame) -> float:
    """Return total DataFrame memory usage in megabytes."""
    return float(df.memory_usage(deep=True).sum() / (1024**2))


def calculate_quality_score(df: pd.DataFrame) -> float:
    """Return a simple quality score based on missing values and duplicates."""
    if df is None or len(df) == 0:
        return 0.0

    penalty = (int(df.isnull().sum().sum()) + int(df.duplicated().sum())) / len(df) * 100
    return round(max(0.0, 100.0 - penalty), 1)


def get_dataframe_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Return a reusable dictionary of core dataframe metrics."""
    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "memory_mb": calculate_memory(df),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "quality_score": calculate_quality_score(df),
    }


def get_dataset_summary(df: pd.DataFrame) -> DatasetSummary:
    """Compute summary statistics and column metadata for a dataset."""
    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include=["object", "category", "bool", "string"]).columns

    column_info = pd.DataFrame(
        {
            "Column Name": df.columns.astype(str),
            "Data Type": df.dtypes.astype(str).values,
            "Missing Values": df.isnull().sum().values,
            "Unique Values": [df[col].nunique() for col in df.columns],
        }
    )
    metrics = get_dataframe_metrics(df)

    return DatasetSummary(
        row_count=metrics["row_count"],
        column_count=metrics["column_count"],
        memory_mb=metrics["memory_mb"],
        missing_values=metrics["missing_values"],
        duplicate_rows=metrics["duplicate_rows"],
        numeric_count=len(numeric_cols),
        categorical_count=len(categorical_cols),
        column_info=column_info,
    )


def format_file_size(size_bytes: int) -> str:
    """Format byte count as a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.1f} MB"
    return f"{size_bytes / (1024 ** 3):.1f} GB"


def get_dataframe_preview(df: pd.DataFrame, rows: int = 10) -> pd.DataFrame:
    """Return the first N rows of a DataFrame."""
    return df.head(rows)
