"""Report generation module for InsightFlow AI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class ReportMetadata:
    title: str
    generated_at: str
    row_count: int
    column_count: int


class ReportGenerator:
    """Builds exportable reports from analysis results."""

    def build_metadata(self, title: str, row_count: int, column_count: int) -> ReportMetadata:
        return ReportMetadata(
            title=title,
            generated_at=datetime.now(timezone.utc).isoformat(),
            row_count=row_count,
            column_count=column_count,
        )

    def to_markdown(self, metadata: ReportMetadata, sections: dict[str, str]) -> str:
        lines = [
            f"# {metadata.title}",
            "",
            f"**Generated:** {metadata.generated_at}",
            f"**Rows:** {metadata.row_count} | **Columns:** {metadata.column_count}",
            "",
        ]
        for heading, body in sections.items():
            lines.extend([f"## {heading}", "", body, ""])
        return "\n".join(lines)
