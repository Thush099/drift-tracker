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
