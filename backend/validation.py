"""Data validation module for InsightFlow AI."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DataValidator:
    """Validates uploaded datasets before processing."""

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []

        if df.empty:
            errors.append("Dataset is empty.")

        if df.columns.duplicated().any():
            errors.append("Dataset contains duplicate column names.")

        if df.isnull().all().any():
            warnings.append("One or more columns contain only null values.")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )
