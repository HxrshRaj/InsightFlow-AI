"""Analytics engine for InsightFlow AI."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


@dataclass
class AnalyticsSummary:
    row_count: int = 0
    column_count: int = 0
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    missing_values: dict[str, int] = field(default_factory=dict)


def _make_empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
    return fig


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda df: (df.shape, df.dtypes.astype(str).to_dict(), df.head(5).to_dict())})
def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return a descriptive summary for numeric and categorical columns."""
    if df is None or df.empty:
        return pd.DataFrame()

    summary = df.describe(include="all").T
    if summary.empty:
        return pd.DataFrame()

    return summary.reset_index().rename(columns={"index": "Column"})


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda df: (df.shape, df.dtypes.astype(str).to_dict(), df.head(5).to_dict())})
def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return the correlation matrix for numeric columns."""
    numeric_df = numeric_columns(df)
    if not numeric_df:
        return pd.DataFrame()
    return df[numeric_df].corr(numeric_only=True)


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda df: (df.shape, df.dtypes.astype(str).to_dict(), df.head(5).to_dict())})
def numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric column names."""
    if df is None:
        return []
    return df.select_dtypes(include="number").columns.tolist()


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda df: (df.shape, df.dtypes.astype(str).to_dict(), df.head(5).to_dict())})
def categorical_columns(df: pd.DataFrame) -> list[str]:
    """Return categorical-style column names."""
    if df is None:
        return []
    return df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda df: (df.shape, df.dtypes.astype(str).to_dict(), df.head(5).to_dict())})
def top_categories(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return the top 10 category counts for a selected column."""
    if df is None or column not in df.columns:
        return pd.DataFrame(columns=[column, "Count"])

    counts = df[column].dropna().value_counts().head(10).reset_index()
    if counts.empty:
        return pd.DataFrame(columns=[column, "Count"])

    counts.columns = [column, "Count"]
    return counts


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda df: (df.shape, df.dtypes.astype(str).to_dict(), df.head(5).to_dict())})
def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Detect outliers for each numeric column using the IQR method."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Column", "Outlier Count", "Outlier %"])

    rows: list[dict[str, object]] = []
    for column in numeric_columns(df):
        series = pd.to_numeric(df[column], errors="coerce")
        if series.dropna().empty:
            rows.append({"Column": column, "Outlier Count": 0, "Outlier %": 0.0})
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            rows.append({"Column": column, "Outlier Count": 0, "Outlier %": 0.0})
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        count = int(outliers.count())
        percent = round((count / len(series)) * 100, 1) if len(series) else 0.0
        rows.append({"Column": column, "Outlier Count": count, "Outlier %": percent})

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda df: (df.shape, df.dtypes.astype(str).to_dict(), df.head(5).to_dict())})
def create_histogram(df: pd.DataFrame, column: str) -> go.Figure:
    """Create a histogram for a selected numeric column."""
    if df is None or df.empty or column not in df.columns or column not in numeric_columns(df):
        return _make_empty_figure("No numeric data available for this histogram.")

    figure = px.histogram(df, x=column, nbins=20, template="plotly_dark")
    figure.update_layout(margin=dict(l=20, r=20, t=40, b=20), title=f"Histogram: {column}")
    return figure


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda df: (df.shape, df.dtypes.astype(str).to_dict(), df.head(5).to_dict())})
def create_boxplot(df: pd.DataFrame, column: str) -> go.Figure:
    """Create a box plot for a selected numeric column."""
    if df is None or df.empty or column not in df.columns or column not in numeric_columns(df):
        return _make_empty_figure("No numeric data available for this box plot.")

    figure = px.box(df, y=column, template="plotly_dark")
    figure.update_layout(margin=dict(l=20, r=20, t=40, b=20), title=f"Box Plot: {column}")
    return figure


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda df: (df.shape, df.dtypes.astype(str).to_dict(), df.head(5).to_dict())})
def create_scatter(df: pd.DataFrame, x: str, y: str) -> go.Figure:
    """Create a scatter plot for two selected numeric columns."""
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        return _make_empty_figure("Select two numeric columns for the scatter plot.")

    numeric_cols = numeric_columns(df)
    if x not in numeric_cols or y not in numeric_cols:
        return _make_empty_figure("Select two numeric columns for the scatter plot.")

    figure = px.scatter(df, x=x, y=y, template="plotly_dark")
    figure.update_layout(margin=dict(l=20, r=20, t=40, b=20), title=f"Scatter: {x} vs {y}")
    return figure


@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda df: (df.shape, df.dtypes.astype(str).to_dict(), df.head(5).to_dict())})
def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create a correlation heatmap for all numeric columns."""
    numeric_cols = numeric_columns(df)
    if df is None or df.empty or len(numeric_cols) < 2:
        return _make_empty_figure("At least two numeric columns are required for a correlation heatmap.")

    corr = correlation_matrix(df)
    figure = px.imshow(
        corr,
        template="plotly_dark",
        color_continuous_scale="Viridis",
        text_auto=True,
        aspect="auto",
    )
    figure.update_layout(margin=dict(l=20, r=20, t=40, b=20), title="Correlation Heatmap")
    return figure


class AnalyticsEngine:
    """Computes statistical summaries and insights from datasets."""

    def summarize(self, df: pd.DataFrame) -> AnalyticsSummary:
        numeric_cols = numeric_columns(df)
        categorical_cols = categorical_columns(df)
        missing = df.isnull().sum().to_dict() if df is not None else {}

        return AnalyticsSummary(
            row_count=len(df) if df is not None else 0,
            column_count=len(df.columns) if df is not None else 0,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            missing_values=missing,
        )
