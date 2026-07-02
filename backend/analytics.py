"""Analytics engine for InsightFlow AI."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class AnalyticsSummary:
    row_count: int = 0
    column_count: int = 0
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    missing_values: dict[str, int] = field(default_factory=dict)


class AnalyticsEngine:
    """Computes statistical summaries and insights from datasets."""

    def summarize(self, df: pd.DataFrame) -> AnalyticsSummary:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        missing = df.isnull().sum().to_dict()

        return AnalyticsSummary(
            row_count=len(df),
            column_count=len(df.columns),
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            missing_values=missing,
        )
