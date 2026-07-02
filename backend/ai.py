"""AI-powered insights module for InsightFlow AI."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InsightResponse:
    summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)


class AIInsights:
    """Generates natural-language insights from analytical results."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def generate_insights(self, context: str) -> InsightResponse:
        return InsightResponse(
            summary="AI insights will be available once analytics are configured.",
            recommendations=[],
            anomalies=[],
        )
