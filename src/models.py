"""Structured drift report dataclasses with JSON and Markdown serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Optional

from src.metrics import Severity


@dataclass
class NumericDriftResult:
    """Feature-level statistical drift metrics for a numeric column."""

    column: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    js_divergence: float
    ref_mean: Optional[float]
    cur_mean: Optional[float]
    ref_std: Optional[float]
    cur_std: Optional[float]
    status: Severity


@dataclass
class CategoricalDriftResult:
    """Feature-level distribution shift and unseen category report."""

    column: str
    js_divergence: float
    new_categories: list[str]
    unseen_rate: float
    ref_frequencies: dict[str, float]
    cur_frequencies: dict[str, float]
    status: Severity


@dataclass
class PredictionDriftResult:
    """Drift metrics comparing reference vs. current model predictions/probabilities."""

    column: str
    kind: str  # "probability" | "class" | "continuous"
    psi: float
    ks_statistic: float
    ks_pvalue: float
    js_divergence: float
    new_classes: list[str]
    ref_mean: Optional[float]
    cur_mean: Optional[float]
    status: Severity


@dataclass
class DataQualityColumnResult:
    """Null-rate delta and out-of-range diagnostics for a shared column."""

    column: str
    is_numeric: bool
    ref_null_rate: float
    cur_null_rate: float
    null_rate_delta: float
    ref_min: Optional[float]
    ref_max: Optional[float]
    below_min_count: int
    above_max_count: int
    out_of_range_count: int
    out_of_range_rate: float
    status: Severity


@dataclass
class SchemaCheckResult:
    """Schema validation summary comparing reference and current columns."""

    missing_columns: list[str]
    extra_columns: list[str]
    shared_columns: list[str]
    dtype_mismatches: dict[str, tuple[str, str]]
    status: Severity


@dataclass
class DriftReport:
    """Top-level batch observability report serializable to JSON and Markdown."""

    reference_rows: int
    current_rows: int
    overall_status: Severity
    schema: SchemaCheckResult
    numeric_features: list[NumericDriftResult] = field(default_factory=list)
    categorical_features: list[CategoricalDriftResult] = field(default_factory=list)
    predictions: list[PredictionDriftResult] = field(default_factory=list)
    data_quality: list[DataQualityColumnResult] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the full report into a plain JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the drift report to formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Render a structured Markdown summary report with tables."""
        lines: list[str] = [
            "# Drift Tracker Observability Report",
            "",
            f"- **Overall Status:** `{self.overall_status}`",
            f"- **Reference Rows:** {self.reference_rows} | **Current Rows:** {self.current_rows}",
            f"- **Schema Status:** `{self.schema.status}` "
            f"(missing: {len(self.schema.missing_columns)}, extra: {len(self.schema.extra_columns)})",
        ]

        if self.schema.missing_columns:
            lines.append(f"- **Missing Columns:** {', '.join(self.schema.missing_columns)}")
        if self.schema.extra_columns:
            lines.append(f"- **Extra Columns:** {', '.join(self.schema.extra_columns)}")

        if self.numeric_features:
            lines.extend(
                [
                    "",
                    "## Numeric Feature Drift",
                    "",
                    "| Feature | PSI | KS Stat | KS p-value | JS Div | Ref Mean | Cur Mean | Status |",
                    "|---|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for item in self.numeric_features:
                ref_m = f"{item.ref_mean:.4f}" if item.ref_mean is not None else "n/a"
                cur_m = f"{item.cur_mean:.4f}" if item.cur_mean is not None else "n/a"
                lines.append(
                    f"| `{item.column}` | {item.psi:.4f} | {item.ks_statistic:.4f} | "
                    f"{item.ks_pvalue:.4g} | {item.js_divergence:.4f} | {ref_m} | {cur_m} | **{item.status}** |"
                )

        if self.categorical_features:
            lines.extend(
                [
                    "",
                    "## Categorical Feature Drift",
                    "",
                    "| Feature | JS Div | Unseen Categories | Unseen Rate | Status |",
                    "|---|---:|---|---:|---|",
                ]
            )
            for item in self.categorical_features:
                unseen = ", ".join(item.new_categories) if item.new_categories else "none"
                lines.append(
                    f"| `{item.column}` | {item.js_divergence:.4f} | {unseen} | "
                    f"{item.unseen_rate:.2%} | **{item.status}** |"
                )

        if self.predictions:
            lines.extend(
                [
                    "",
                    "## Prediction Drift",
                    "",
                    "| Output | Kind | PSI | KS Stat | KS p-value | JS Div | Unseen Classes | Status |",
                    "|---|---|---:|---:|---:|---:|---|---|",
                ]
            )
            for item in self.predictions:
                unseen = ", ".join(item.new_classes) if item.new_classes else "none"
                lines.append(
                    f"| `{item.column}` | {item.kind} | {item.psi:.4f} | {item.ks_statistic:.4f} | "
                    f"{item.ks_pvalue:.4g} | {item.js_divergence:.4f} | {unseen} | **{item.status}** |"
                )

        if self.data_quality:
            lines.extend(
                [
                    "",
                    "## Data Quality",
                    "",
                    "| Column | Ref Null | Cur Null | Null Delta | Out-of-Range | OOR Rate | Status |",
                    "|---|---:|---:|---:|---:|---:|---|",
                ]
            )
            for item in self.data_quality:
                lines.append(
                    f"| `{item.column}` | {item.ref_null_rate:.2%} | {item.cur_null_rate:.2%} | "
                    f"{item.null_rate_delta:+.2%} | {item.out_of_range_count} | "
                    f"{item.out_of_range_rate:.2%} | **{item.status}** |"
                )

        if self.alerts or self.warnings:
            lines.extend(["", "## Findings"])
            for msg in self.alerts:
                lines.append(f"- **[ALERT]** {msg}")
            for msg in self.warnings:
                lines.append(f"- **[WARN]** {msg}")

        return "\n".join(lines)
