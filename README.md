# Drift Tracker

Batch ML observability tool for monitoring numeric & categorical feature drift, prediction drift, schema integrity, and data quality between a reference dataset and a production batch.

## Problem Statement

Silent covariate shifts, schema changes, unseen categorical values, and missing-value spikes degrade production ML models long before ground-truth labels arrive. `drift-tracker` compares a reference baseline against a current batch and generates a structured `DriftReport` (JSON and Markdown) with actionable severity tiers (`OK`, `WARN`, `ALERT`).

## Metrics & Thresholds

| Check Category | Metric / Rule | `OK` | `WARN` | `ALERT` |
|---|---|---|---|---|
| **Numeric Feature Drift** | Population Stability Index (`compute_psi`, 10 quantile bins + smoothing) | `< 0.10` | `0.10 – 0.25` | `>= 0.25` |
| | Two-sample Kolmogorov–Smirnov (`ks_2samp` statistic & p-value) | Reported alongside PSI | — | — |
| | Jensen–Shannon Divergence (base 2, `[0, 1]`) | `< 0.05` | `0.05 – 0.15` | `>= 0.15` |
| **Categorical Feature Drift** | Jensen–Shannon Divergence (base 2, `[0, 1]`) | `< 0.05` | `0.05 – 0.15` | `>= 0.15` |
| | Unseen Categories (`new_categories`) | None | Unseen rate `< 10%` | Unseen rate `>= 10%` |
| **Prediction Drift** | Probability score PSI & KS test / Class PSI & JS divergence | `< 0.10` PSI | `0.10 – 0.25` PSI / unseen class | `>= 0.25` PSI |
| **Data Quality** | Null-rate delta (`cur_null - ref_null`) | `< 2%` | `2% – 5%` | `>= 5%` or current batch is 100% null |
| | Out-of-range rate (vs. reference `[min, max]`) | `< 1%` | `1% – 5%` | `>= 5%` |
| **Schema Validation** | Missing, extra, or dtype-mismatched columns | None | Extra columns | Missing / dtype mismatch |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run out-of-the-box against the bundled `data/reference.csv` and `data/production_drifted.csv` datasets:

```bash
python main.py
```

Compare custom CSV files and export structured JSON / Markdown reports:

```bash
python main.py \
  --reference path/to/reference.csv \
  --current path/to/current.csv \
  --prediction-col prediction \
  --proba-col pred_proba \
  --output-json outputs/drift_report.json \
  --output-markdown outputs/drift_report.md
```

Use `--demo` to generate fresh synthetic reference & drifted batches in memory.

### Sample Output

Output from `python main.py`:

```markdown
# Drift Tracker Observability Report

- **Overall Status:** `ALERT`
- **Reference Rows:** 1000 | **Current Rows:** 1000
- **Schema Status:** `OK` (missing: 0, extra: 0)

## Numeric Feature Drift

| Feature | PSI | KS Stat | KS p-value | JS Div | Ref Mean | Cur Mean | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| `age` | 0.0166 | 0.0390 | 0.4326 | 0.0030 | 38.8369 | 39.1721 | **OK** |
| `income` | 0.3783 | 0.2735 | 4.934e-32 | 0.0662 | 73537.2390 | 62421.3624 | **ALERT** |
| `credit_score` | 1.3250 | 0.4790 | 3.179e-104 | 0.2163 | 706.6110 | 640.6560 | **ALERT** |
| `debt_ratio` | 0.5613 | 0.3030 | 6.548e-41 | 0.0960 | 0.2893 | 0.4191 | **ALERT** |
| `account_tenure` | 0.0307 | 0.0570 | 0.07761 | 0.0055 | 5.9422 | 6.2407 | **OK** |

## Categorical Feature Drift

| Feature | JS Div | Unseen Categories | Unseen Rate | Status |
|---|---:|---|---:|---|
| `region` | 0.0795 | APAC | 11.90% | **ALERT** |
| `employment_type` | 0.0814 | Contractor | 9.50% | **WARN** |

## Prediction Drift

| Output | Kind | PSI | KS Stat | KS p-value | JS Div | Unseen Classes | Status |
|---|---|---:|---:|---:|---:|---|---|
| `pred_proba` | probability | 1.3317 | 0.5010 | 1.837e-114 | 0.2152 | none | **ALERT** |
| `prediction` | class | 0.8319 | 0.4400 | 2.091e-87 | 0.1447 | none | **ALERT** |

## Data Quality

| Column | Ref Null | Cur Null | Null Delta | Out-of-Range | OOR Rate | Status |
|---|---:|---:|---:|---:|---:|---|
| `age` | 0.00% | 0.00% | +0.00% | 2 | 0.20% | **OK** |
| `income` | 0.00% | 8.10% | +8.10% | 16 | 1.74% | **ALERT** |
| `credit_score` | 0.00% | 0.00% | +0.00% | 87 | 8.70% | **ALERT** |
| `debt_ratio` | 0.00% | 0.00% | +0.00% | 42 | 4.20% | **WARN** |
| `account_tenure` | 0.00% | 0.00% | +0.00% | 1 | 0.10% | **OK** |
| `region` | 0.00% | 0.00% | +0.00% | 0 | 0.00% | **OK** |
| `employment_type` | 0.00% | 0.00% | +0.00% | 0 | 0.00% | **OK** |
| `pred_proba` | 0.00% | 0.00% | +0.00% | 10 | 1.00% | **WARN** |
| `prediction` | 0.00% | 0.00% | +0.00% | 0 | 0.00% | **OK** |
```

## Running Tests

```bash
pytest -v
```

## License

MIT — see [LICENSE](LICENSE).
