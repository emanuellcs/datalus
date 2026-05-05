"""Autonomous Audit Orchestrator metrics for privacy and utility."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PrivacyThresholds:
    """Default privacy thresholds from the README/OAA protocol."""

    memorization_ratio: float = 0.01
    mia_roc_auc: float = 0.55
    dcr_percentile: float = 1.0


@dataclass(slots=True)
class ShadowMIAConfig:
    """Configuration for black-box shadow membership inference."""

    n_shadow_models: int = 4
    shadow_train_fraction: float = 0.5
    synthetic_multiplier: float = 1.0
    n_neighbors: int = 5
    random_state: int = 42


class PrivacyEvaluator:
    """Compute DCR and black-box shadow-model MIA metrics."""

    def __init__(
        self,
        real_train: pl.DataFrame,
        synthetic: pl.DataFrame,
        schema_metadata: dict[str, Any],
        real_holdout: pl.DataFrame | None = None,
    ) -> None:
        self.real_train = real_train
        self.synthetic = synthetic
        self.real_holdout = real_holdout
        self.schema_metadata = schema_metadata
        self.columns = [
            column
            for column, meta in schema_metadata.items()
            if meta.get("encoding_strategy") != "DROP"
            and meta.get("retained", True)
            and not meta.get("is_target", False)
            and column in real_train.columns
            and column in synthetic.columns
        ]
        self.real_train = self.real_train.select(self.columns)
        self.synthetic = self.synthetic.select(self.columns)
        if self.real_holdout is not None:
            self.real_holdout = self.real_holdout.select(self.columns)
        self.preprocessor = self._build_preprocessor()

    def _build_preprocessor(self) -> ColumnTransformer:
        num_cols = []
        cat_cols = []
        for column in self.columns:
            topology = self.schema_metadata[column].get("inferred_topology", "")
            if "NUMERICAL" in topology:
                num_cols.append(column)
            else:
                cat_cols.append(column)
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    cat_cols,
                ),
            ],
            remainder="drop",
        )

    def _project(
        self, fit_frame: pl.DataFrame, *frames: pl.DataFrame
    ) -> list[np.ndarray]:
        fit_pd = fit_frame.to_pandas()
        self.preprocessor.fit(fit_pd)
        return [
            self.preprocessor.transform(frame.to_pandas()).astype(np.float32)
            for frame in frames
        ]

    def compute_dcr(
        self, thresholds: PrivacyThresholds | None = None
    ) -> dict[str, Any]:
        threshold_cfg = thresholds or PrivacyThresholds()
        x_real, x_synth = self._project(
            self.real_train, self.real_train, self.synthetic
        )
        nn = NearestNeighbors(n_neighbors=2, algorithm="auto", n_jobs=-1).fit(x_real)
        synth_distances, _ = nn.kneighbors(x_synth, n_neighbors=1)
        real_distances, _ = nn.kneighbors(x_real, n_neighbors=2)
        baseline_threshold = float(
            np.percentile(real_distances[:, 1], threshold_cfg.dcr_percentile)
        )
        distances = synth_distances[:, 0]
        memorization_ratio = float(np.mean(distances < baseline_threshold))
        return {
            "dcr_median": float(np.median(distances)),
            "dcr_p5": float(np.percentile(distances, 5)),
            "dcr_alert_threshold": baseline_threshold,
            "prop_below_alert_threshold": memorization_ratio,
            "privacy_verdict_dcr": (
                "APPROVED"
                if memorization_ratio < threshold_cfg.memorization_ratio
                else "REJECTED"
            ),
        }

    def shadow_membership_inference(
        self,
        synthetic_provider: (
            Callable[[pl.DataFrame, int, int], pl.DataFrame] | None
        ) = None,
        config: ShadowMIAConfig | None = None,
        thresholds: PrivacyThresholds | None = None,
    ) -> dict[str, Any]:
        """Run a Shokri-style shadow attack adapted to generated samples."""

        cfg = config or ShadowMIAConfig()
        threshold_cfg = thresholds or PrivacyThresholds()
        rng = np.random.default_rng(cfg.random_state)
        shadow_features: list[np.ndarray] = []
        shadow_labels: list[np.ndarray] = []
        real_pd = self.real_train.to_pandas()
        for idx in range(cfg.n_shadow_models):
            member_idx, nonmember_idx = train_test_split(
                np.arange(len(real_pd)),
                train_size=cfg.shadow_train_fraction,
                random_state=cfg.random_state + idx,
                shuffle=True,
            )
            members = pl.from_pandas(real_pd.iloc[member_idx].reset_index(drop=True))
            nonmembers = pl.from_pandas(
                real_pd.iloc[nonmember_idx].reset_index(drop=True)
            )
            n_synth = max(1, int(len(members) * cfg.synthetic_multiplier))
            shadow_synth = (
                synthetic_provider(members, n_synth, cfg.random_state + idx)
                if synthetic_provider is not None
                else _bootstrap_shadow_generator(members, n_synth, rng)
            )
            features_m = self._attack_features(members, shadow_synth, cfg)
            features_n = self._attack_features(nonmembers, shadow_synth, cfg)
            shadow_features.extend([features_m, features_n])
            shadow_labels.extend([np.ones(len(features_m)), np.zeros(len(features_n))])

        x_attack = np.vstack(shadow_features)
        y_attack = np.concatenate(shadow_labels)
        attack_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=cfg.random_state,
        )
        attack_model.fit(x_attack, y_attack)

        if self.real_holdout is None:
            member_idx, nonmember_idx = train_test_split(
                np.arange(len(self.real_train)),
                train_size=0.5,
                random_state=cfg.random_state + 10_000,
                shuffle=True,
            )
            target_members = self.real_train[member_idx]
            target_nonmembers = self.real_train[nonmember_idx]
        else:
            target_members = self.real_train
            target_nonmembers = self.real_holdout
        member_features = self._attack_features(target_members, self.synthetic, cfg)
        nonmember_features = self._attack_features(
            target_nonmembers, self.synthetic, cfg
        )
        x_eval = np.vstack([member_features, nonmember_features])
        y_eval = np.concatenate(
            [np.ones(len(member_features)), np.zeros(len(nonmember_features))]
        )
        scores = attack_model.predict_proba(x_eval)[:, 1]
        auc = float(roc_auc_score(y_eval, scores))
        return {
            "mia_roc_auc": auc,
            "mia_average_precision": float(average_precision_score(y_eval, scores)),
            "privacy_verdict_mia": (
                "APPROVED" if auc < threshold_cfg.mia_roc_auc else "REJECTED"
            ),
            "shadow_models": cfg.n_shadow_models,
        }

    def _attack_features(
        self,
        candidates: pl.DataFrame,
        generated: pl.DataFrame,
        config: ShadowMIAConfig,
    ) -> np.ndarray:
        fit_frame = pl.concat(
            [candidates.select(self.columns), generated.select(self.columns)],
            how="vertical_relaxed",
        )
        x_candidates, x_generated = self._project(
            fit_frame, candidates.select(self.columns), generated.select(self.columns)
        )
        k = min(config.n_neighbors, len(x_generated))
        nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1).fit(
            x_generated
        )
        distances, _ = nn.kneighbors(x_candidates)
        min_d = distances[:, 0]
        mean_k = distances.mean(axis=1)
        std_k = distances.std(axis=1)
        ratio = min_d / np.maximum(mean_k, 1e-8)
        return np.stack([min_d, mean_k, std_k, ratio], axis=1).astype(np.float32)

    def run_audit(
        self,
        synthetic_provider: (
            Callable[[pl.DataFrame, int, int], pl.DataFrame] | None
        ) = None,
        thresholds: PrivacyThresholds | None = None,
    ) -> dict[str, Any]:
        dcr = self.compute_dcr(thresholds)
        mia = self.shadow_membership_inference(
            synthetic_provider, thresholds=thresholds
        )
        approved = (
            dcr["privacy_verdict_dcr"] == "APPROVED"
            and mia["privacy_verdict_mia"] == "APPROVED"
        )
        return {
            "audit_type": "Privacy",
            "privacy": {
                **dcr,
                **mia,
                "privacy_verdict": "APPROVED" if approved else "REJECTED",
            },
        }


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
        self.real_df = real_df
        self.synthetic_df = synthetic_df
        self.schema_metadata = schema_metadata
        self.target_column = target_column
        self.random_state = random_state

    def run_audit(self, approval_threshold: float = 0.90) -> dict[str, Any]:
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
                "utility_verdict": (
                    "APPROVED" if mle_auc >= approval_threshold else "REJECTED"
                ),
            },
        }

    def _fit_eval(self, x_train, y_train, x_test, y_test) -> dict[str, float]:
        model = build_tabular_classifier(random_state=self.random_state)
        model.fit(x_train, y_train)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(x_test)[:, 1]
        else:  # pragma: no cover - all configured models expose predict_proba
            scores = model.decision_function(x_test)
        preds = (scores >= 0.5).astype(int)
        return {
            "roc_auc": (
                float(roc_auc_score(y_test, scores))
                if len(np.unique(y_test)) > 1
                else 0.5
            ),
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
    except Exception:
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
    return list(frame.select_dtypes(include=[np.number]).columns)


def _categorical_selector(frame):
    return list(frame.select_dtypes(exclude=[np.number]).columns)


def _bootstrap_shadow_generator(
    frame: pl.DataFrame, n_records: int, rng: np.random.Generator
) -> pl.DataFrame:
    """Small fallback shadow generator used only when no trained generator is supplied."""

    indices = rng.integers(0, len(frame), size=n_records)
    sample = frame[indices]
    mutated: dict[str, Any] = {}
    for column in sample.columns:
        series = sample.get_column(column)
        if series.dtype.is_numeric():
            arr = series.cast(pl.Float64).to_numpy()
            scale = np.nanstd(arr) * 0.01
            mutated[column] = arr + rng.normal(
                0.0, scale if scale > 0 else 1e-6, size=len(arr)
            )
        else:
            mutated[column] = series.to_list()
    return pl.DataFrame(mutated)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > 1e-12 else 0.0


def write_audit_report(path: str | Path, report: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
