"""AI-powered insights module for InsightFlow AI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd
from dotenv import load_dotenv
from groq import Groq
import streamlit as st

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.3
GROQ_MAX_TOKENS = 700


@dataclass
class InsightResponse:
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)


def create_dataset_context(df: pd.DataFrame | None) -> str:
    """Create a concise business-oriented context string for the AI prompt."""
    if df is None or df.empty:
        return "The dataset is empty."

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
    missing_values = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    summary_stats = df.describe(include="all").T.reset_index().to_string(index=False)
    top_categories: list[str] = []
    for column in categorical_columns[:5]:
        counts = df[column].dropna().value_counts().head(3)
        if not counts.empty:
            formatted_counts = ", ".join(f"{label} ({count})" for label, count in counts.items())
            top_categories.append(f"{column}: {formatted_counts}")

    correlation_highlights: list[str] = []
    if len(numeric_columns) >= 2:
        correlation_matrix = df[numeric_columns].corr(numeric_only=True)
        correlation_pairs = [
            (column, correlation_matrix[column].drop(column).abs().sort_values(ascending=False).iloc[0])
            for column in correlation_matrix.columns
            if correlation_matrix[column].drop(column).abs().max() > 0.3
        ]
        for column, value in correlation_pairs[:3]:
            correlation_highlights.append(
                f"{column} shows a strong relationship with another numeric feature (correlation strength: {value:.2f})."
            )

    context_sections = [
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        f"Missing values: {missing_values}",
        f"Duplicates: {duplicates}",
        f"Numeric columns: {', '.join(numeric_columns) if numeric_columns else 'None'}",
        f"Categorical columns: {', '.join(categorical_columns) if categorical_columns else 'None'}",
        "Summary statistics:",
        summary_stats,
    ]

    if top_categories:
        context_sections.append("Top categories:")
        context_sections.extend(top_categories)

    if correlation_highlights:
        context_sections.append("Correlation highlights:")
        context_sections.extend(correlation_highlights)

    return "\n".join(context_sections)


def build_prompt(summary: str) -> str:
    """Build a concise prompt for the business-insight generation model."""
    return (
        "You are an expert business analyst. Based on the dataset context below, "
        "write a concise professional business report with only these sections and no markdown tables:\n\n"
        "# Executive Summary\n"
        "# Key Business Insights\n"
        "# Potential Risks\n"
        "# Recommendations\n\n"
        f"Dataset context:\n{summary}"
    )


@st.cache_data(show_spinner=False)
def generate_ai_insights(summary: str) -> str:
    """Generate AI business insights from a dataset summary using the Groq API."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "No Groq API key was found. Add GROQ_API_KEY to your .env file to enable AI insights."

    try:
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a senior business analyst."},
                {"role": "user", "content": build_prompt(summary)},
            ],
            model=GROQ_MODEL,
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as exc:  # pragma: no cover - runtime-dependent network/API issues
        return f"AI insights could not be generated. Please try again later. Error: {exc}"


class AIInsights:
    """Generates natural-language insights from analytical results."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def generate_insights(self, context: str) -> InsightResponse:
        """Generate a structured insight response from the AI report text."""
        report = generate_ai_insights(context)
        sections = [segment.strip() for segment in report.split("#") if segment.strip()]
        recommendations: list[str] = []
        anomalies: list[str] = []

        for section in sections:
            if section.lower().startswith("key business insights"):
                recommendations = [line.strip("-• ") for line in section.splitlines() if line.strip()]
            elif section.lower().startswith("potential risks"):
                anomalies = [line.strip("-• ") for line in section.splitlines() if line.strip()]

        return InsightResponse(summary=report, recommendations=recommendations, anomalies=anomalies)
