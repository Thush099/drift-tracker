"""Unit tests for PSI, KS, and Jensen-Shannon drift metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics import (
    classify_js,
    classify_psi,
    compute_discrete_psi,
    compute_js_categorical,
    compute_js_numeric,
    compute_ks_test,
    compute_psi,
    max_severity,
    _build_quantile_edges,
    _smoothed_proportions,
    _to_finite_array,
)


class TestNumericMetrics:
    def test_identical_distributions_zero_drift(self) -> None:
        rng = np.random.default_rng(42)
        ref = rng.normal(50.0, 10.0, size=1000)

        psi = compute_psi(ref, ref)
        ks_stat, ks_pval = compute_ks_test(ref, ref)
        js = compute_js_numeric(ref, ref)

        assert psi == pytest.approx(0.0, abs=1e-12)
        assert ks_stat == pytest.approx(0.0, abs=1e-12)
        assert ks_pval == pytest.approx(1.0, abs=1e-12)
        assert js == pytest.approx(0.0, abs=1e-12)
        assert classify_psi(psi) == "OK"
        assert classify_js(js) == "OK"

    def test_moderate_and_severe_shifts_trigger_warn_and_alert(self) -> None:
        rng = np.random.default_rng(7)
        ref = rng.normal(0.0, 1.0, size=2000)

        # Moderate shift -> PSI in [0.1, 0.25) -> WARN
        moderate = rng.normal(0.38, 1.0, size=2000)
        psi_mod = compute_psi(ref, moderate)
        assert 0.1 <= psi_mod < 0.25
        assert classify_psi(psi_mod) == "WARN"

        # Severe shift -> PSI >= 0.25 -> ALERT
        severe = rng.normal(1.2, 1.0, size=2000)
        psi_sev = compute_psi(ref, severe)
        ks_stat, ks_pval = compute_ks_test(ref, severe)
        js_sev = compute_js_numeric(ref, severe)

        assert psi_sev >= 0.25
        assert classify_psi(psi_sev) == "ALERT"
        assert ks_stat > 0.3
        assert ks_pval < 1e-10
        assert js_sev > 0.05

    def test_constant_feature_and_empty_edge_cases(self) -> None:
        const_ref = np.ones(100) * 5.0
        assert compute_psi(const_ref, const_ref) == pytest.approx(0.0, abs=1e-12)
        assert compute_js_numeric(const_ref, const_ref) == pytest.approx(0.0, abs=1e-12)

        shifted_const = np.ones(100) * 9.0
        assert compute_psi(const_ref, shifted_const) > 0.25

        assert compute_psi(np.array([np.nan, np.nan]), np.array([np.nan])) == 0.0
        assert compute_ks_test(np.array([np.nan]), np.array([1.0, 2.0])) == (1.0, 0.0)


class TestCategoricalMetrics:
    def test_identical_categories_zero_js(self) -> None:
        ref = ["A"] * 50 + ["B"] * 30 + ["C"] * 20
        js, new_cats, ref_f, cur_f = compute_js_categorical(ref, ref)

        assert js == pytest.approx(0.0, abs=1e-12)
        assert new_cats == []
        assert ref_f == cur_f

    def test_shifted_and_unseen_categories(self) -> None:
        ref = ["Full-time"] * 70 + ["Part-time"] * 20 + ["Self-employed"] * 10
        cur = ["Full-time"] * 20 + ["Part-time"] * 20 + ["Self-employed"] * 45 + ["Contractor"] * 15

        js, new_cats, _, _ = compute_js_categorical(ref, cur)
        psi = compute_discrete_psi(ref, cur)

        assert js > 0.15
        assert classify_js(js) == "ALERT"
        assert new_cats == ["Contractor"]
        assert psi > 0.25


@pytest.mark.parametrize("eps", [0.0, -1.0, float("nan"), float("inf")])
def test_distribution_metrics_reject_invalid_smoothing(eps: float) -> None:
    values = [0.0, 1.0, 2.0]
    calls = (
        lambda: compute_psi(values, values, eps=eps),
        lambda: compute_discrete_psi(values, values, eps=eps),
        lambda: compute_js_numeric(values, values, eps=eps),
        lambda: compute_js_categorical(values, values, eps=eps),
    )

    for call in calls:
        with pytest.raises(ValueError, match="eps must be finite and positive"):
            call()


@pytest.mark.parametrize("n_bins", [1, 2.5, True])
def test_numeric_distribution_metrics_reject_invalid_bin_counts(n_bins: int) -> None:
    values = [0.0, 1.0, 2.0]

    with pytest.raises(ValueError, match="n_bins must be an integer >= 2"):
        compute_psi(values, values, n_bins=n_bins)
    with pytest.raises(ValueError, match="n_bins must be an integer >= 2"):
        compute_js_numeric(values, values, n_bins=n_bins)


class TestSeverityAndHistogramHelpers:
    def test_max_severity_ranks_alert_above_warn_above_ok(self) -> None:
        assert max_severity("OK", "WARN", "ALERT") == "ALERT"
        assert max_severity("WARN", "OK") == "WARN"
        assert max_severity("OK", "OK") == "OK"

    def test_max_severity_ignores_unknown_status_strings(self) -> None:
        assert max_severity("NOT_A_STATUS", "OK") == "OK"
        assert max_severity("NOT_A_STATUS", "ALERT") == "ALERT"

    def test_to_finite_array_drops_nan_and_inf(self) -> None:
        arr = _to_finite_array([1.0, float("nan"), 2.0, float("inf"), -3.0])
        assert arr.tolist() == [1.0, 2.0, -3.0]

    def test_build_quantile_edges_rejects_tiny_bin_count(self) -> None:
        with pytest.raises(ValueError, match="n_bins"):
            _build_quantile_edges(np.arange(10.0), n_bins=1)

    def test_build_quantile_edges_are_unique_and_sorted(self) -> None:
        reference = np.zeros(100)
        edges = _build_quantile_edges(reference, n_bins=10)
        assert len(edges) >= 2
        assert np.all(np.diff(edges) > 0)

    def test_smoothed_proportions_uniform_when_no_mass(self) -> None:
        probs = _smoothed_proportions(np.zeros(4), eps=0.0)
        assert probs == pytest.approx(np.full(4, 0.25))

    def test_smoothed_proportions_sum_to_one(self) -> None:
        probs = _smoothed_proportions(np.array([3.0, 1.0, 0.0]), eps=1e-6)
        assert probs.sum() == pytest.approx(1.0)
        assert np.all(probs > 0.0)
