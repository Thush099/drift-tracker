"""Dataset generation and I/O helpers for batch drift tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder


def _fit_reference_scorer(df_ref: pd.DataFrame, seed: int = 42) -> SkPipeline:
    """Train a lightweight classifier on reference features to score batches."""
    rng = np.random.default_rng(seed)
    # Latent risk score driven by credit_score, debt_ratio, income, and employment_type.
    emp_bonus = np.where(df_ref["employment_type"] == "Full-time", 0.35, -0.25)
    logits = (
        -0.015 * (df_ref["credit_score"] - 680.0)
        + 3.2 * (df_ref["debt_ratio"] - 0.35)
        - 0.018 * ((df_ref["income"] / 1000.0) - 70.0)
        + emp_bonus
        + rng.normal(0.0, 0.35, size=len(df_ref))
    )
    y = (logits > 0.0).astype(int)

    num_cols = ["age", "income", "credit_score", "debt_ratio", "account_tenure"]
    cat_cols = ["region", "employment_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                cat_cols,
            ),
        ]
    )
    clf = SkPipeline(
        steps=[
            ("prep", preprocessor),
            ("rf", RandomForestClassifier(n_estimators=60, max_depth=6, random_state=seed)),
        ]
    )
    clf.fit(df_ref[num_cols + cat_cols], y)
    return clf


def generate_demo_datasets(
    n_ref: int = 1000,
    n_cur: int = 1000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate realistic reference and drifted production batches with model outputs."""
    rng = np.random.default_rng(seed)

    ref = pd.DataFrame(
        {
            "age": np.clip(rng.normal(39.0, 10.0, size=n_ref), 20.0, 72.0).round(1),
            "income": np.clip(rng.normal(75000.0, 18000.0, size=n_ref), 22000.0, 165000.0).round(0),
            "credit_score": np.clip(rng.normal(705.0, 48.0, size=n_ref), 520.0, 840.0).round(0),
            "debt_ratio": np.clip(rng.beta(2.5, 6.0, size=n_ref), 0.02, 0.85).round(4),
            "account_tenure": np.clip(rng.gamma(4.0, 1.5, size=n_ref), 0.5, 20.0).round(1),
            "region": rng.choice(
                ["North", "South", "East", "West"],
                size=n_ref,
                p=[0.30, 0.25, 0.25, 0.20],
            ),
            "employment_type": rng.choice(
                ["Full-time", "Part-time", "Self-employed"],
                size=n_ref,
                p=[0.68, 0.18, 0.14],
            ),
        }
    )

    scorer = _fit_reference_scorer(ref, seed=seed)
    ref_proba = scorer.predict_proba(ref)[:, 1]
    ref["pred_proba"] = np.round(ref_proba, 4)
    ref["prediction"] = (ref_proba >= 0.5).astype(int)

    cur = pd.DataFrame(
        {
            # age stays stable (OK)
            "age": np.clip(rng.normal(39.2, 10.1, size=n_cur), 20.0, 72.0).round(1),
            # income shifts moderately + elevated nulls (WARN/ALERT)
            "income": np.clip(rng.normal(63000.0, 21000.0, size=n_cur), 18000.0, 165000.0).round(0),
            # credit_score shifts strongly downward (ALERT)
            "credit_score": np.clip(rng.normal(640.0, 58.0, size=n_cur), 450.0, 830.0).round(0),
            # debt_ratio shifts moderately upward + out-of-range spikes
            "debt_ratio": np.clip(rng.beta(3.4, 5.2, size=n_cur), 0.02, 0.96).round(4),
            # account_tenure stays mostly stable (OK)
            "account_tenure": np.clip(rng.gamma(4.1, 1.48, size=n_cur), 0.5, 20.0).round(1),
            # region has shifted weights + unseen category "APAC"
            "region": rng.choice(
                ["North", "South", "East", "West", "APAC"],
                size=n_cur,
                p=[0.20, 0.32, 0.20, 0.18, 0.10],
            ),
            # employment_type shifts toward Self-employed + unseen "Contractor"
            "employment_type": rng.choice(
                ["Full-time", "Part-time", "Self-employed", "Contractor"],
                size=n_cur,
                p=[0.42, 0.22, 0.26, 0.10],
            ),
        }
    )

    # Score before injecting raw data-quality nulls so model outputs stay realistic,
    # then inject 8% missingness in income and 4% out-of-range values in debt_ratio.
    cur_proba = scorer.predict_proba(cur)[:, 1]
    cur["pred_proba"] = np.round(cur_proba, 4)
    cur["prediction"] = (cur_proba >= 0.5).astype(int)

    null_mask = rng.random(n_cur) < 0.08
    cur.loc[null_mask, "income"] = np.nan

    oor_indices = rng.choice(n_cur, size=max(1, int(0.03 * n_cur)), replace=False)
    cur.loc[oor_indices, "debt_ratio"] = np.round(rng.uniform(0.90, 1.15, size=len(oor_indices)), 4)

    return ref, cur


def load_csv_pair(reference_path: str | Path, current_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load reference and current CSV files into DataFrames."""
    for label, raw in (("reference", reference_path), ("current", current_path)):
        path = Path(raw)
        if not path.is_file():
            raise FileNotFoundError(f"{label} CSV file not found: {path}")
    ref_df = pd.read_csv(reference_path)
    cur_df = pd.read_csv(current_path)
    return ref_df, cur_df
