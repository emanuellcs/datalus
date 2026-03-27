"""
DATALUS Autonomous Audit Orchestrator (OAA)
Layer 5: Machine Learning Efficacy (Train-on-Synthetic, Test-on-Real)
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import polars as pl

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("OAAUtility")


class UtilityEvaluator:
    """
    Executes the Machine Learning Efficacy (MLE) audit using the TSTR protocol
    (Train on Synthetic, Test on Real) to guarantee data utility for policymakers.
    """
    def __init__(
        self,
        real_df: pl.DataFrame,
        synth_df: pl.DataFrame,
        schema_metadata: Dict[str, Any],
        target_column: str,
        task_type: str = "binary"
    ):
        self.real_df = real_df
        self.synth_df = synth_df
        self.schema_metadata = schema_metadata
        self.target_column = target_column
        self.task_type = task_type
        
        # Identify categorical features for native LightGBM handling
        self.categorical_features = [
            col for col, meta in self.schema_metadata.items()
            if "CATEGORICAL" in meta["inferred_topology"] or meta["inferred_topology"] == "BOOLEAN"
        ]
        if self.target_column in self.categorical_features:
            self.categorical_features.remove(self.target_column)

        logger.info(f"Initialized Utility Evaluator. Target: '{self.target_column}' | Categorical Features: {len(self.categorical_features)}")
        self._prepare_datasets()

    def _prepare_datasets(self) -> None:
        """
        Splits the real data into Train and Test sets. 
        The synthetic data acts as a replacement for the real Train set.
        """
        # Convert Polars to Pandas for LightGBM compatibility
        real_pd = self.real_df.to_pandas()
        synth_pd = self.synth_df.to_pandas()
        
        # Ensure categorical columns are properly typed as 'category' in Pandas
        for col in self.categorical_features:
            real_pd[col] = real_pd[col].astype("category")
            synth_pd[col] = synth_pd[col].astype("category")

        # Split Real Data (TRTR Baseline)
        X_real = real_pd.drop(columns=[self.target_column])
        y_real = real_pd[self.target_column]
        
        self.X_real_train, self.X_real_test, self.y_real_train, self.y_real_test = train_test_split(
            X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
        )
        
        # Prepare Synthetic Data (TSTR Challenger)
        self.X_synth_train = synth_pd.drop(columns=[self.target_column])
        self.y_synth_train = synth_pd[self.target_column]
        
        logger.debug(f"TRTR Baseline Shape: {self.X_real_train.shape}")
        logger.debug(f"TSTR Challenger Shape: {self.X_synth_train.shape}")
        logger.debug(f"Holdout Test Shape: {self.X_real_test.shape}")

    def _train_and_evaluate(self, X_train, y_train, X_test, y_test, run_name: str) -> Dict[str, float]:
        """
        Trains a LightGBM classifier and evaluates it on the holdout set.
        Returns key metrics: ROC-AUC, F1-Score, and Average Precision (PR-AUC).
        """
        logger.info(f"Training LightGBM model for: {run_name}")
        
        # LightGBM Configuration for Tabular Data
        params = {
            "objective": "binary" if self.task_type == "binary" else "multiclass",
            "metric": "auc" if self.task_type == "binary" else "multi_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 8,
            "feature_fraction": 0.8,
            "verbose": -1,
            "seed": 42,
            "n_jobs": -1
        }
        
        # Create LightGBM Datasets
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=self.categorical_features)
        
        # Train Model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200
        )
        
        # Inference on Real Holdout Test Set
        y_pred_proba = model.predict(X_test)
        y_pred_class = (y_pred_proba > 0.5).astype(int) if self.task_type == "binary" else np.argmax(y_pred_proba, axis=1)
        
        # Calculate Metrics
        metrics = {}
        if self.task_type == "binary":
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_pred_proba))
            metrics["f1_score"] = float(f1_score(y_test, y_pred_class))
            metrics["pr_auc"] = float(average_precision_score(y_test, y_pred_proba))
        else:
            # For multiclass, adjust metric calculation accordingly
            metrics["f1_score_macro"] = float(f1_score(y_test, y_pred_class, average="macro"))
            
        logger.info(f"[{run_name}] ROC-AUC: {metrics.get('roc_auc', 0):.4f} | F1: {metrics.get('f1_score', 0):.4f}")
        return metrics

    def run_audit(self, approval_threshold: float = 0.90) -> Dict[str, Any]:
        """
        Orchestrates the TRTR and TSTR evaluation pipelines and compares the results.
        Calculates the MLE-Ratio (Machine Learning Efficacy Ratio).
        """
        logger.info("Initiating Machine Learning Efficacy (MLE) Audit...")
        
        # 1. Baseline: Train on Real, Test on Real (TRTR)
        trtr_metrics = self._train_and_evaluate(
            self.X_real_train, self.y_real_train, 
            self.X_real_test, self.y_real_test, 
            run_name="TRTR_Baseline"
        )
        
        # 2. Challenger: Train on Synthetic, Test on Real (TSTR)
        tstr_metrics = self._train_and_evaluate(
            self.X_synth_train, self.y_synth_train, 
            self.X_real_test, self.y_real_test, 
            run_name="TSTR_Synthetic"
        )
        
        # 3. Calculate MLE-Ratio
        # Formula: AUC(Synthetic) / AUC(Real)
        mle_ratio_auc = tstr_metrics["roc_auc"] / trtr_metrics["roc_auc"]
        mle_ratio_f1 = tstr_metrics["f1_score"] / trtr_metrics["f1_score"]
        
        logger.info(f"MLE-Ratio (ROC-AUC): {mle_ratio_auc:.4f}")
        logger.info(f"MLE-Ratio (F1-Score): {mle_ratio_f1:.4f}")
        
        passed_audit = mle_ratio_auc >= approval_threshold
        
        report = {
            "audit_type": "Machine Learning Efficacy (MLE)",
            "approval_threshold": approval_threshold,
            "passed_utility_audit": passed_audit,
            "metrics": {
                "TRTR_Baseline": trtr_metrics,
                "TSTR_Synthetic": tstr_metrics,
                "MLE_Ratio_AUC": float(mle_ratio_auc),
                "MLE_Ratio_F1": float(mle_ratio_f1)
            }
        }
        
        return report

if __name__ == "__main__":
    # Smoke Test
    import json
    
    # Mock Schema
    schema = {
        "AGE": {"inferred_topology": "NUMERICAL"},
        "GENDER": {"inferred_topology": "CATEGORICAL"},
        "MORTE": {"inferred_topology": "BOOLEAN"}
    }
    
    # Mock Real Data
    real_mock = pl.DataFrame({
        "AGE": np.random.normal(50, 15, 2000),
        "GENDER": np.random.choice(["M", "F"], 2000),
        "MORTE": np.random.choice([0, 1], 2000, p=[0.8, 0.2])
    })
    
    # Mock Synthetic Data (Adding slight noise to simulate generation)
    synth_mock = pl.DataFrame({
        "AGE": np.random.normal(50.5, 14.5, 2000),
        "GENDER": np.random.choice(["M", "F"], 2000, p=[0.5, 0.5]),
        "MORTE": np.random.choice([0, 1], 2000, p=[0.78, 0.22])
    })
    
    auditor = UtilityEvaluator(
        real_df=real_mock, 
        synth_df=synth_mock, 
        schema_metadata=schema, 
        target_column="MORTE"
    )
    
    final_report = auditor.run_audit()
    print("\n--- OAA UTILITY AUDIT REPORT ---")
    print(json.dumps(final_report, indent=4))