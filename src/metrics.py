"""Statistical drift and divergence metrics for numeric and categorical distributions."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

Severity = Literal["OK", "WARN", "ALERT"]
_SEVERITY_RANK: dict[str, int] = {"OK": 0, "WARN": 1, "ALERT": 2}


def max_severity(*statuses: str) -> Severity:
    """Return the highest severity level among the provided statuses."""
    best = "OK"
    best_rank = 0
    for status in statuses:
        rank = _SEVERITY_RANK.get(status, 0)
        if rank > best_rank:
            best = status
            best_rank = rank
    return best  # type: ignore[return-value]


def classify_psi(
    psi: float,
    warn_threshold: float = 0.1,
    alert_threshold: float = 0.25,
) -> Severity:
    """Classify PSI using standard thresholds (<0.1 OK, 0.1-0.25 WARN, >=0.25 ALERT)."""
    if psi >= alert_threshold:
        return "ALERT"
    if psi >= warn_threshold:
        return "WARN"
    return "OK"


def classify_js(
    js_divergence: float,
    warn_threshold: float = 0.05,
    alert_threshold: float = 0.15,
) -> Severity:
    """Classify Jensen-Shannon divergence (base 2, bounded in [0, 1])."""
    if js_divergence >= alert_threshold:
        return "ALERT"
    if js_divergence >= warn_threshold:
        return "WARN"
    return "OK"


def _to_finite_array(values: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def _validate_n_bins(n_bins: int) -> None:
    if isinstance(n_bins, bool) or not isinstance(n_bins, Integral) or n_bins < 2:
        raise ValueError(f"n_bins must be an integer >= 2, got {n_bins}")


def _validate_eps(eps: float) -> None:
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps}")


def _build_quantile_edges(reference: np.ndarray, n_bins: int = 10) -> np.ndarray:
    _validate_n_bins(n_bins)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 2:
        center = float(edges[0])
        return np.array([-np.inf, center - 1e-9, center + 1e-9, np.inf], dtype=np.float64)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _smoothed_proportions(counts: np.ndarray, eps: float) -> np.ndarray:
    total = float(np.sum(counts))
    if total <= 0.0:
        probs = np.full(len(counts), 1.0 / len(counts), dtype=np.float64)
    else:
        probs = counts.astype(np.float64) / total
    smoothed = probs + eps
    return smoothed / np.sum(smoothed)


def compute_psi(
    reference: Sequence[float] | np.ndarray | pd.Series,
    current: Sequence[float] | np.ndarray | pd.Series,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """Compute Population Stability Index (PSI) with quantile binning and smoothing."""
    _validate_n_bins(n_bins)
    _validate_eps(eps)
    ref_clean = _to_finite_array(reference)
    cur_clean = _to_finite_array(current)
    if ref_clean.size == 0 and cur_clean.size == 0:
        return 0.0
    if ref_clean.size == 0 or cur_clean.size == 0:
        return 1.0

    edges = _build_quantile_edges(ref_clean, n_bins=n_bins)
    ref_counts, _ = np.histogram(ref_clean, bins=edges)
    cur_counts, _ = np.histogram(cur_clean, bins=edges)

    p_ref = _smoothed_proportions(ref_counts, eps=eps)
    p_cur = _smoothed_proportions(cur_counts, eps=eps)
    psi = float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))
    return max(0.0, psi)


def compute_discrete_psi(
    reference: Sequence[Any] | np.ndarray | pd.Series,
    current: Sequence[Any] | np.ndarray | pd.Series,
    eps: float = 1e-6,
) -> float:
    """Compute PSI across discrete classes or categories with smoothing."""
    _validate_eps(eps)
    ref_s = pd.Series(reference).dropna().astype(str)
    cur_s = pd.Series(current).dropna().astype(str)
    if ref_s.empty and cur_s.empty:
        return 0.0
    if ref_s.empty or cur_s.empty:
        return 1.0

    categories = sorted(set(ref_s.unique()) | set(cur_s.unique()))
    ref_vc = ref_s.value_counts()
    cur_vc = cur_s.value_counts()
    ref_counts = np.array([ref_vc.get(c, 0) for c in categories], dtype=np.float64)
    cur_counts = np.array([cur_vc.get(c, 0) for c in categories], dtype=np.float64)

    p_ref = _smoothed_proportions(ref_counts, eps=eps)
    p_cur = _smoothed_proportions(cur_counts, eps=eps)
    psi = float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))
    return max(0.0, psi)


def compute_ks_test(
    reference: Sequence[float] | np.ndarray | pd.Series,
    current: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[float, float]:
    """Compute the two-sample Kolmogorov-Smirnov test (statistic, p-value)."""
    ref_clean = _to_finite_array(reference)
    cur_clean = _to_finite_array(current)
    if ref_clean.size == 0 and cur_clean.size == 0:
        return 0.0, 1.0
    if ref_clean.size == 0 or cur_clean.size == 0:
        return 1.0, 0.0
    res = ks_2samp(ref_clean, cur_clean)
    return float(res.statistic), float(res.pvalue)


def compute_js_numeric(
    reference: Sequence[float] | np.ndarray | pd.Series,
    current: Sequence[float] | np.ndarray | pd.Series,
    n_bins: int = 10,
    eps: float = 1e-8,
) -> float:
    """Compute base-2 Jensen-Shannon divergence in [0, 1] for numeric samples."""
    _validate_n_bins(n_bins)
    _validate_eps(eps)
    ref_clean = _to_finite_array(reference)
    cur_clean = _to_finite_array(current)
    if ref_clean.size == 0 and cur_clean.size == 0:
        return 0.0
    if ref_clean.size == 0 or cur_clean.size == 0:
        return 1.0

    edges = _build_quantile_edges(ref_clean, n_bins=n_bins)
    ref_counts, _ = np.histogram(ref_clean, bins=edges)
    cur_counts, _ = np.histogram(cur_clean, bins=edges)

    p_ref = _smoothed_proportions(ref_counts, eps=eps)
    p_cur = _smoothed_proportions(cur_counts, eps=eps)
    if np.allclose(p_ref, p_cur, atol=1e-15, rtol=0.0):
        return 0.0
    js_dist = float(jensenshannon(p_ref, p_cur, base=2.0))
    return float(np.clip(js_dist ** 2, 0.0, 1.0))


def compute_js_categorical(
    reference: Sequence[Any] | np.ndarray | pd.Series,
    current: Sequence[Any] | np.ndarray | pd.Series,
    eps: float = 1e-8,
) -> tuple[float, list[str], dict[str, float], dict[str, float]]:
    """Compute base-2 Jensen-Shannon divergence and unseen categories for categorical data."""
    _validate_eps(eps)
    ref_s = pd.Series(reference).dropna().astype(str)
    cur_s = pd.Series(current).dropna().astype(str)

    if ref_s.empty and cur_s.empty:
        return 0.0, [], {}, {}

    ref_cats = set(ref_s.unique())
    cur_cats = set(cur_s.unique())
    new_categories = sorted(cur_cats - ref_cats)

    if ref_s.empty or cur_s.empty:
        return 1.0, new_categories, {}, {}

    all_cats = sorted(ref_cats | cur_cats)
    ref_vc = ref_s.value_counts(normalize=True)
    cur_vc = cur_s.value_counts(normalize=True)
    ref_freqs = {cat: float(ref_vc.get(cat, 0.0)) for cat in all_cats}
    cur_freqs = {cat: float(cur_vc.get(cat, 0.0)) for cat in all_cats}

    p_ref = _smoothed_proportions(
        np.array([ref_s.value_counts().get(c, 0) for c in all_cats], dtype=np.float64),
        eps=eps,
    )
    p_cur = _smoothed_proportions(
        np.array([cur_s.value_counts().get(c, 0) for c in all_cats], dtype=np.float64),
        eps=eps,
    )
    if np.allclose(p_ref, p_cur, atol=1e-15, rtol=0.0):
        js_div = 0.0
    else:
        js_dist = float(jensenshannon(p_ref, p_cur, base=2.0))
        js_div = float(np.clip(js_dist ** 2, 0.0, 1.0))

    return js_div, new_categories, ref_freqs, cur_freqs
