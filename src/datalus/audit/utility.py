"""Utility audit: TSTR/TRTR evaluation using gradient-boosting classifiers."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class UtilityEvaluator:
    """TSTR/TRTR utility audit using gradient boosting classifiers."""

    def __init__(
        self,
        real_df: pl.DataFrame,
        synthetic_df: pl.DataFrame,
        schema_metadata: dict[str, Any],
        target_column: str,
        random_state: int = 42,
    ) -> None:
        """Store real and synthetic frames with their target column."""

        self.real_df = real_df
        self.synthetic_df = synthetic_df
        self.schema_metadata = schema_metadata
        self.target_column = target_column
        self.random_state = random_state

    def run_audit(self, approval_threshold: float = 0.90) -> dict[str, Any]:
        """Run TRTR and TSTR evaluations and return the utility verdict."""

        real_pd = self.real_df.to_pandas()
        synth_pd = self.synthetic_df.to_pandas()
        categorical = [
            column
            for column, meta in self.schema_metadata.items()
            if column != self.target_column
            and column in real_pd.columns
            and (
                "CATEGORICAL" in meta.get("inferred_topology", "")
                or meta.get("inferred_topology") == "BOOLEAN"
            )
        ]
        for frame in [real_pd, synth_pd]:
            for column in categorical:
                frame[column] = frame[column].astype("category")
        x_real = real_pd.drop(columns=[self.target_column])
        y_real = real_pd[self.target_column]
        x_synth = synth_pd.drop(columns=[self.target_column])
        y_synth = synth_pd[self.target_column]
        x_train, x_test, y_train, y_test = train_test_split(
            x_real,
            y_real,
            test_size=0.2,
            random_state=self.random_state,
            stratify=y_real if y_real.nunique() > 1 else None,
        )
        trtr = self._fit_eval(x_train, y_train, x_test, y_test)
        tstr = self._fit_eval(x_synth, y_synth, x_test, y_test)
        mle_auc = safe_ratio(tstr["roc_auc"], trtr["roc_auc"])
        mle_f1 = safe_ratio(tstr["f1_score"], trtr["f1_score"])
        return {
            "audit_type": "Utility",
            "utility": {
                "trtr_auc": trtr["roc_auc"],
                "trtr_f1": trtr["f1_score"],
                "tstr_auc": tstr["roc_auc"],
                "tstr_f1": tstr["f1_score"],
                "mle_ratio_auc": mle_auc,
                "mle_ratio_f1": mle_f1,
                "utility_verdict": ("APPROVED" if mle_auc >= approval_threshold else "REJECTED"),
            },
        }

    def _fit_eval(self, x_train, y_train, x_test, y_test) -> dict[str, float]:
        """Fit a classifier and return AUC and F1 on the test split."""

        model = build_tabular_classifier(random_state=self.random_state)
        model.fit(x_train, y_train)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(x_test)[:, 1]
        else:  # pragma: no cover - all configured models expose predict_proba
            scores = model.decision_function(x_test)
        preds = (scores >= 0.5).astype(int)
        return {
            "roc_auc": (float(roc_auc_score(y_test, scores)) if len(np.unique(y_test)) > 1 else 0.5),
            "f1_score": float(f1_score(y_test, preds, zero_division=0)),
        }


def build_tabular_classifier(random_state: int = 42):
    """Use LightGBM when available; otherwise use a sklearn fallback for CI."""

    try:
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
    except ImportError:
        numeric = Pipeline([("scale", StandardScaler())])
        categorical = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
        preprocessor = ColumnTransformer(
            [
                ("num", numeric, _numeric_selector),
                ("cat", categorical, _categorical_selector),
            ]
        )
        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", GradientBoostingClassifier(random_state=random_state)),
            ]
        )


def _numeric_selector(frame):
    """Return the numeric column names of a pandas frame."""

    return list(frame.select_dtypes(include=[np.number]).columns)


def _categorical_selector(frame):
    """Return the non-numeric column names of a pandas frame."""

    return list(frame.select_dtypes(exclude=[np.number]).columns)


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return the ratio, or 0.0 when the denominator is effectively zero."""

    return float(numerator / denominator) if abs(denominator) > 1e-12 else 0.0
