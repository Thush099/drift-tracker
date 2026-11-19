"""Integration and CLI tests for DriftTracker pipeline and reporting."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main as cli_main
from config import DriftTrackerConfig
from src.pipeline import DriftTracker
from src.utils import generate_demo_datasets


class TestDriftTrackerPipeline:
    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"psi_warn": 0.3, "psi_alert": 0.2}, "psi_warn must be less than psi_alert"),
            ({"ks_alpha": 1.0}, "ks_alpha must be in (0, 1)"),
            ({"null_delta_alert": 1.1}, "null_delta_alert must be in [0, 1]"),
            ({"n_bins": 1}, "n_bins must be >= 2"),
            ({"psi_eps": 0.0}, "psi_eps must be finite and positive"),
        ],
    )
    def test_invalid_config_thresholds_fail_fast(
        self, overrides: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=re.escape(message)):
            DriftTrackerConfig(**overrides)  # type: ignore[arg-type]

    def test_fully_null_current_column_is_alert_even_if_reference_is_also_null(self) -> None:
        ref = pd.DataFrame({"unusable": [np.nan] * 5})
        cur = pd.DataFrame({"unusable": [np.nan] * 5})

        report = DriftTracker().compare(ref, cur)

        assert report.overall_status == "ALERT"
        quality = report.data_quality[0]
        assert quality.column == "unusable"
        assert quality.cur_null_rate == 1.0
        assert quality.status == "ALERT"

    def test_identical_reference_and_current_all_ok(self) -> None:
        ref_df, _ = generate_demo_datasets(n_ref=500, n_cur=500, seed=42)
        tracker = DriftTracker()
        report = tracker.compare(ref_df, ref_df.copy())

        assert report.overall_status == "OK"
        assert report.schema.status == "OK"
        assert report.schema.missing_columns == []
        assert report.schema.extra_columns == []
        assert report.alerts == []
        assert report.warnings == []

        for num in report.numeric_features:
            assert num.status == "OK"
            assert num.psi == pytest.approx(0.0, abs=1e-12)
            assert num.ks_statistic == pytest.approx(0.0, abs=1e-12)
            assert num.ks_pvalue == pytest.approx(1.0, abs=1e-12)
            assert num.js_divergence == pytest.approx(0.0, abs=1e-12)

        for cat in report.categorical_features:
            assert cat.status == "OK"
            assert cat.js_divergence == pytest.approx(0.0, abs=1e-12)
            assert cat.new_categories == []

        for pred in report.predictions:
            assert pred.status == "OK"
            assert pred.psi == pytest.approx(0.0, abs=1e-12)

        for dq in report.data_quality:
            assert dq.status == "OK"
            assert dq.null_rate_delta == pytest.approx(0.0, abs=1e-12)
            assert dq.out_of_range_count == 0

    def test_shifted_features_unseen_categories_missing_cols_and_nulls_flagged(self) -> None:
        rng = np.random.default_rng(10)
        n = 600
        ref = pd.DataFrame(
            {
                "stable_num": rng.normal(0.0, 1.0, size=n),
                "moderate_num": rng.normal(10.0, 2.0, size=n),
                "severe_num": rng.normal(100.0, 10.0, size=n),
                "segment": rng.choice(["A", "B", "C"], size=n, p=[0.6, 0.3, 0.1]),
                "dropped_col": rng.uniform(0.0, 1.0, size=n),
                "pred_proba": np.clip(rng.beta(2.0, 5.0, size=n), 0.01, 0.99),
                "prediction": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
            }
        )

        cur = pd.DataFrame(
            {
                "stable_num": rng.normal(0.0, 1.0, size=n),
                "moderate_num": rng.normal(10.8, 2.0, size=n),
                "severe_num": rng.normal(125.0, 12.0, size=n),
                "segment": rng.choice(["A", "B", "C", "UNSEEN"], size=n, p=[0.2, 0.2, 0.4, 0.2]),
                "extra_col": np.ones(n),
                "pred_proba": np.clip(rng.beta(5.0, 2.0, size=n), 0.01, 0.99),
                "prediction": rng.choice([0, 1], size=n, p=[0.25, 0.75]),
            }
        )
        # Elevate null rate in stable_num by 15% and add out-of-range spikes to moderate_num
        cur.loc[:89, "stable_num"] = np.nan
        cur.loc[:49, "moderate_num"] = 999.0

        report = DriftTracker().compare(ref, cur)

        assert report.overall_status == "ALERT"
        assert report.schema.status == "ALERT"
        assert report.schema.missing_columns == ["dropped_col"]
        assert report.schema.extra_columns == ["extra_col"]

        num_by_col = {r.column: r for r in report.numeric_features}
        assert num_by_col["stable_num"].status == "OK"
        assert num_by_col["moderate_num"].status in ("WARN", "ALERT")
        assert num_by_col["severe_num"].status == "ALERT"
        assert num_by_col["severe_num"].psi >= 0.25

        cat_by_col = {r.column: r for r in report.categorical_features}
        assert cat_by_col["segment"].status == "ALERT"
        assert cat_by_col["segment"].new_categories == ["UNSEEN"]
        assert cat_by_col["segment"].unseen_rate > 0.1

        pred_by_col = {r.column: r for r in report.predictions}
        assert pred_by_col["pred_proba"].status == "ALERT"
        assert pred_by_col["prediction"].status == "ALERT"

        dq_by_col = {r.column: r for r in report.data_quality}
        assert dq_by_col["stable_num"].null_rate_delta == pytest.approx(0.15, abs=1e-6)
        assert dq_by_col["stable_num"].status == "ALERT"
        assert dq_by_col["moderate_num"].out_of_range_count >= 50
        assert dq_by_col["moderate_num"].status == "ALERT"

    def test_json_and_markdown_serialization(self) -> None:
        ref_df, cur_df = generate_demo_datasets(n_ref=300, n_cur=300, seed=42)
        report = DriftTracker().compare(ref_df, cur_df)

        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["overall_status"] == report.overall_status
        assert len(parsed["numeric_features"]) == 5
        assert len(parsed["categorical_features"]) == 2
        assert len(parsed["predictions"]) == 2

        md = report.to_markdown()
        assert "# Drift Tracker Observability Report" in md
        assert "## Numeric Feature Drift" in md
        assert "## Categorical Feature Drift" in md
        assert "## Prediction Drift" in md
        assert "## Data Quality" in md


class TestCLI:
    def test_cli_default_and_output_json(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        out_json = tmp_path / "drift_report.json"
        exit_code = cli_main.main(["--output-json", str(out_json)])
        assert exit_code == 0

        stdout = capsys.readouterr().out
        assert "# Drift Tracker Observability Report" in stdout
        assert out_json.exists()
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["overall_status"] in ("WARN", "ALERT")
        assert payload["reference_rows"] == 1000
        assert payload["current_rows"] == 1000

    def test_cli_subprocess_smoke(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "main.py"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Drift Tracker Observability Report" in proc.stdout
        assert "credit_score" in proc.stdout
        assert "pred_proba" in proc.stdout


class TestFailOnExitCode:
    @staticmethod
    def _report(status: str):
        from src.models import DriftReport, SchemaCheckResult

        return DriftReport(
            reference_rows=10,
            current_rows=10,
            overall_status=status,
            schema=SchemaCheckResult(
                missing_columns=[], extra_columns=[], shared_columns=[], dtype_mismatches={}, status="OK"
            ),
        )

    def test_fail_on_none_always_succeeds(self):
        from main import exit_code_for

        assert exit_code_for(self._report("ALERT"), "none") == 0

    def test_fail_on_alert_gates_on_alert(self):
        from main import exit_code_for

        assert exit_code_for(self._report("ALERT"), "alert") == 1
        assert exit_code_for(self._report("WARN"), "alert") == 0
        assert exit_code_for(self._report("OK"), "alert") == 0

    def test_fail_on_warn_gates_on_warn_and_alert(self):
        from main import exit_code_for

        assert exit_code_for(self._report("WARN"), "warn") == 1
        assert exit_code_for(self._report("ALERT"), "warn") == 1
        assert exit_code_for(self._report("OK"), "warn") == 0

    def test_fail_on_rejects_unknown_level(self):
        import pytest

        from main import exit_code_for

        with pytest.raises(ValueError, match="fail_on must be one of"):
            exit_code_for(self._report("OK"), "bogus")

    def test_cli_default_still_exits_zero(self, capsys):
        import main as cli_main

        code = cli_main.main(["--demo"])
        assert code == 0

    def test_cli_fail_on_alert_fails_on_drifted_demo(self, capsys):
        import main as cli_main

        code = cli_main.main(["--demo", "--fail-on", "alert"])
        assert code == 1
