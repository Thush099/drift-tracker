"""Batch ML observability pipeline for feature, prediction, and data quality drift."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config import DriftTrackerConfig
from src.metrics import (
    Severity,
    classify_js,
    classify_psi,
    compute_discrete_psi,
    compute_js_categorical,
    compute_js_numeric,
    compute_ks_test,
    compute_psi,
    max_severity,
)
from src.models import (
    CategoricalDriftResult,
    DataQualityColumnResult,
    DriftReport,
    NumericDriftResult,
    PredictionDriftResult,
    SchemaCheckResult,
)


def _is_numeric_column(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
    )


class DriftTracker:
    """Compare a reference DataFrame against a current production DataFrame."""

    def __init__(
        self,
        config: Optional[DriftTrackerConfig] = None,
        prediction_col: Optional[str] = "prediction",
        proba_col: Optional[str] = "pred_proba",
        target_col: Optional[str] = "target",
    ) -> None:
        self.config = config or DriftTrackerConfig()
        self.prediction_col = prediction_col
        self.proba_col = proba_col
        self.target_col = target_col

    def _validate_schema(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
    ) -> SchemaCheckResult:
        ref_cols = list(reference.columns)
        cur_cols = list(current.columns)
        ref_set = set(ref_cols)
        cur_set = set(cur_cols)

        missing = [c for c in ref_cols if c not in cur_set]
        extra = [c for c in cur_cols if c not in ref_set]
        shared = [c for c in ref_cols if c in cur_set]

        dtype_mismatches: dict[str, tuple[str, str]] = {}
        for col in shared:
            ref_num = _is_numeric_column(reference[col])
            cur_num = _is_numeric_column(current[col])
            if ref_num != cur_num:
                dtype_mismatches[col] = (str(reference[col].dtype), str(current[col].dtype))

        status: Severity = "OK"
        if missing or dtype_mismatches:
            status = "ALERT"
        elif extra:
            status = "WARN"

        return SchemaCheckResult(
            missing_columns=missing,
            extra_columns=extra,
            shared_columns=shared,
            dtype_mismatches=dtype_mismatches,
            status=status,
        )

    def _analyze_data_quality(
        self,
        col: str,
        ref_s: pd.Series,
        cur_s: pd.Series,
        is_numeric: bool,
    ) -> DataQualityColumnResult:
        cfg = self.config
        ref_null = float(ref_s.isna().mean()) if len(ref_s) else 0.0
        cur_null = float(cur_s.isna().mean()) if len(cur_s) else 0.0
        null_delta = cur_null - ref_null

        if abs(null_delta) >= cfg.null_delta_alert or cur_null == 1.0:
            null_status: Severity = "ALERT"
        elif abs(null_delta) >= cfg.null_delta_warn:
            null_status = "WARN"
        else:
            null_status = "OK"

        ref_min: Optional[float] = None
        ref_max: Optional[float] = None
        below_min = 0
        above_max = 0
        oor_count = 0
        oor_rate = 0.0
        oor_status: Severity = "OK"

        if is_numeric:
            ref_num = pd.to_numeric(ref_s, errors="coerce").dropna()
            cur_num = pd.to_numeric(cur_s, errors="coerce").dropna()
            if not ref_num.empty:
                ref_min = float(ref_num.min())
                ref_max = float(ref_num.max())
                if not cur_num.empty:
                    below_min = int((cur_num < ref_min).sum())
                    above_max = int((cur_num > ref_max).sum())
                    oor_count = below_min + above_max
                    oor_rate = float(oor_count / len(cur_num))
                    if oor_rate >= cfg.out_of_range_alert:
                        oor_status = "ALERT"
                    elif oor_rate >= cfg.out_of_range_warn:
                        oor_status = "WARN"

        return DataQualityColumnResult(
            column=col,
            is_numeric=is_numeric,
            ref_null_rate=ref_null,
            cur_null_rate=cur_null,
            null_rate_delta=null_delta,
            ref_min=ref_min,
            ref_max=ref_max,
            below_min_count=below_min,
            above_max_count=above_max,
            out_of_range_count=oor_count,
            out_of_range_rate=oor_rate,
            status=max_severity(null_status, oor_status),
        )

    def _analyze_numeric_feature(
        self,
        col: str,
        ref_s: pd.Series,
        cur_s: pd.Series,
    ) -> NumericDriftResult:
        cfg = self.config
        ref_vals = pd.to_numeric(ref_s, errors="coerce").to_numpy(dtype=np.float64)
        cur_vals = pd.to_numeric(cur_s, errors="coerce").to_numpy(dtype=np.float64)

        psi = compute_psi(ref_vals, cur_vals, n_bins=cfg.n_bins, eps=cfg.psi_eps)
        ks_stat, ks_pval = compute_ks_test(ref_vals, cur_vals)
        js_div = compute_js_numeric(ref_vals, cur_vals, n_bins=cfg.n_bins, eps=cfg.js_eps)

        ref_fin = ref_vals[np.isfinite(ref_vals)]
        cur_fin = cur_vals[np.isfinite(cur_vals)]
        ref_mean = float(np.mean(ref_fin)) if ref_fin.size else None
        cur_mean = float(np.mean(cur_fin)) if cur_fin.size else None
        ref_std = float(np.std(ref_fin)) if ref_fin.size else None
        cur_std = float(np.std(cur_fin)) if cur_fin.size else None

        status = max_severity(
            classify_psi(psi, cfg.psi_warn, cfg.psi_alert),
            classify_js(js_div, cfg.js_warn, cfg.js_alert),
        )
        return NumericDriftResult(
            column=col,
            psi=psi,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            js_divergence=js_div,
            ref_mean=ref_mean,
            cur_mean=cur_mean,
            ref_std=ref_std,
            cur_std=cur_std,
            status=status,
        )

    def _analyze_categorical_feature(
        self,
        col: str,
        ref_s: pd.Series,
        cur_s: pd.Series,
    ) -> CategoricalDriftResult:
        cfg = self.config
        js_div, new_cats, ref_freqs, cur_freqs = compute_js_categorical(
            ref_s,
            cur_s,
            eps=cfg.js_eps,
        )
        cur_non_null = cur_s.dropna().astype(str)
        if not cur_non_null.empty and new_cats:
            unseen_rate = float(cur_non_null.isin(new_cats).mean())
        else:
            unseen_rate = 0.0

        status = classify_js(js_div, cfg.js_warn, cfg.js_alert)
        if new_cats:
            status = max_severity(status, "ALERT" if unseen_rate >= 0.10 else "WARN")

        return CategoricalDriftResult(
            column=col,
            js_divergence=js_div,
            new_categories=new_cats,
            unseen_rate=unseen_rate,
            ref_frequencies=ref_freqs,
            cur_frequencies=cur_freqs,
            status=status,
        )

    def _analyze_prediction_column(
        self,
        col: str,
        ref_s: pd.Series,
        cur_s: pd.Series,
        force_probability: bool = False,
    ) -> PredictionDriftResult:
        cfg = self.config
        is_num = _is_numeric_column(ref_s) and _is_numeric_column(cur_s)

        if force_probability or (is_num and pd.api.types.is_float_dtype(ref_s) and ref_s.dropna().nunique() > 10):
            kind = "probability" if force_probability else "continuous"
            ref_vals = pd.to_numeric(ref_s, errors="coerce").to_numpy(dtype=np.float64)
            cur_vals = pd.to_numeric(cur_s, errors="coerce").to_numpy(dtype=np.float64)
            psi = compute_psi(ref_vals, cur_vals, n_bins=cfg.n_bins, eps=cfg.psi_eps)
            ks_stat, ks_pval = compute_ks_test(ref_vals, cur_vals)
            js_div = compute_js_numeric(ref_vals, cur_vals, n_bins=cfg.n_bins, eps=cfg.js_eps)
            ref_fin = ref_vals[np.isfinite(ref_vals)]
            cur_fin = cur_vals[np.isfinite(cur_vals)]
            status = max_severity(
                classify_psi(psi, cfg.psi_warn, cfg.psi_alert),
                classify_js(js_div, cfg.js_warn, cfg.js_alert),
            )
            return PredictionDriftResult(
                column=col,
                kind=kind,
                psi=psi,
                ks_statistic=ks_stat,
                ks_pvalue=ks_pval,
                js_divergence=js_div,
                new_classes=[],
                ref_mean=float(np.mean(ref_fin)) if ref_fin.size else None,
                cur_mean=float(np.mean(cur_fin)) if cur_fin.size else None,
                status=status,
            )

        # Discrete class outputs
        psi = compute_discrete_psi(ref_s, cur_s, eps=cfg.psi_eps)
        js_div, new_classes, _, _ = compute_js_categorical(ref_s, cur_s, eps=cfg.js_eps)
        if is_num:
            ref_vals = pd.to_numeric(ref_s, errors="coerce").to_numpy(dtype=np.float64)
            cur_vals = pd.to_numeric(cur_s, errors="coerce").to_numpy(dtype=np.float64)
            ks_stat, ks_pval = compute_ks_test(ref_vals, cur_vals)
            ref_fin = ref_vals[np.isfinite(ref_vals)]
            cur_fin = cur_vals[np.isfinite(cur_vals)]
            ref_mean = float(np.mean(ref_fin)) if ref_fin.size else None
            cur_mean = float(np.mean(cur_fin)) if cur_fin.size else None
        else:
            ks_stat, ks_pval = 0.0, 1.0
            ref_mean, cur_mean = None, None

        status = max_severity(
            classify_psi(psi, cfg.psi_warn, cfg.psi_alert),
            classify_js(js_div, cfg.js_warn, cfg.js_alert),
            "WARN" if new_classes else "OK",
        )
        return PredictionDriftResult(
            column=col,
            kind="class",
            psi=psi,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            js_divergence=js_div,
            new_classes=new_classes,
            ref_mean=ref_mean,
            cur_mean=cur_mean,
            status=status,
        )

    def compare(self, reference: pd.DataFrame, current: pd.DataFrame) -> DriftReport:
        """Run full schema, data quality, feature, and prediction drift checks."""
        if reference.empty:
            raise ValueError("Reference DataFrame must not be empty")
        if current.empty:
            raise ValueError("Current DataFrame must not be empty")

        schema = self._validate_schema(reference, current)

        special_cols = {
            c for c in (self.prediction_col, self.proba_col, self.target_col) if c is not None
        }

        numeric_results: list[NumericDriftResult] = []
        categorical_results: list[CategoricalDriftResult] = []
        prediction_results: list[PredictionDriftResult] = []
        quality_results: list[DataQualityColumnResult] = []

        for col in schema.shared_columns:
            ref_s = reference[col]
            cur_s = current[col]
            is_num_ref = _is_numeric_column(ref_s)
            is_num_both = is_num_ref and _is_numeric_column(cur_s)

            quality_results.append(self._analyze_data_quality(col, ref_s, cur_s, is_num_both))

            if col in schema.dtype_mismatches:
                continue

            if col == self.proba_col:
                prediction_results.append(
                    self._analyze_prediction_column(col, ref_s, cur_s, force_probability=is_num_both)
                )
            elif col == self.prediction_col:
                prediction_results.append(
                    self._analyze_prediction_column(col, ref_s, cur_s, force_probability=False)
                )
            elif col in special_cols:
                continue
            elif is_num_both:
                numeric_results.append(self._analyze_numeric_feature(col, ref_s, cur_s))
            else:
                categorical_results.append(self._analyze_categorical_feature(col, ref_s, cur_s))

        alerts: list[str] = []
        warnings: list[str] = []

        if schema.missing_columns:
            alerts.append(f"Missing columns in current batch: {schema.missing_columns}")
        if schema.dtype_mismatches:
            alerts.append(f"Column dtype mismatches: {schema.dtype_mismatches}")
        if schema.extra_columns:
            warnings.append(f"Extra unexpected columns in current batch: {schema.extra_columns}")

        for num in numeric_results:
            msg = (
                f"Numeric feature '{num.column}': PSI={num.psi:.4f}, "
                f"KS={num.ks_statistic:.4f} (p={num.ks_pvalue:.3g}), JS={num.js_divergence:.4f}"
            )
            if num.status == "ALERT":
                alerts.append(msg)
            elif num.status == "WARN":
                warnings.append(msg)

        for cat in categorical_results:
            extra_info = f", unseen={cat.new_categories}" if cat.new_categories else ""
            msg = f"Categorical feature '{cat.column}': JS={cat.js_divergence:.4f}{extra_info}"
            if cat.status == "ALERT":
                alerts.append(msg)
            elif cat.status == "WARN":
                warnings.append(msg)

        for pred in prediction_results:
            msg = (
                f"Prediction output '{pred.column}' ({pred.kind}): PSI={pred.psi:.4f}, "
                f"KS={pred.ks_statistic:.4f} (p={pred.ks_pvalue:.3g}), JS={pred.js_divergence:.4f}"
            )
            if pred.status == "ALERT":
                alerts.append(msg)
            elif pred.status == "WARN":
                warnings.append(msg)

        for dq in quality_results:
            if dq.status == "OK":
                continue
            msg = (
                f"Data quality '{dq.column}': null_delta={dq.null_rate_delta:+.2%}, "
                f"out_of_range={dq.out_of_range_count} ({dq.out_of_range_rate:.2%})"
            )
            if dq.status == "ALERT":
                alerts.append(msg)
            else:
                warnings.append(msg)

        all_statuses = [
            schema.status,
            *(r.status for r in numeric_results),
            *(r.status for r in categorical_results),
            *(r.status for r in prediction_results),
            *(r.status for r in quality_results),
        ]
        overall_status = max_severity(*all_statuses)

        return DriftReport(
            reference_rows=len(reference),
            current_rows=len(current),
            overall_status=overall_status,
            schema=schema,
            numeric_features=numeric_results,
            categorical_features=categorical_results,
            predictions=prediction_results,
            data_quality=quality_results,
            alerts=alerts,
            warnings=warnings,
        )


# Backwards-compatible alias
Pipeline = DriftTracker
