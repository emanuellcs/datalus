"""
DATALUS Autonomous Audit Orchestrator (OAA)
Layer 5: Privacy Metrics (DCR and Membership Inference Attack)
"""

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import polars as pl

from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("OAAPrivacy")


class PrivacyEvaluator:
    """
    Executes a rigorous privacy audit on synthetic datasets to ensure compliance 
    with data protection regulations (e.g., LGPD).
    """
    def __init__(
        self, 
        real_df: pl.DataFrame, 
        synth_df: pl.DataFrame, 
        schema_metadata: Dict[str, Any]
    ):
        self.real_df = real_df
        self.synth_df = synth_df
        self.schema_metadata = schema_metadata
        
        # Ensure identical column ordering
        self.columns = list(self.schema_metadata.keys())
        self.real_df = self.real_df.select(self.columns)
        self.synth_df = self.synth_df.select(self.columns)
        
        logger.info(f"Initialized Privacy Evaluator with {len(self.real_df)} real and {len(self.synth_df)} synthetic records.")
        self._build_preprocessor()
        self._transform_data()

    def _build_preprocessor(self) -> None:
        """
        Builds a scikit-learn ColumnTransformer to map all heterogeneous data 
        into a unified, normalized Euclidean space for accurate distance calculations.
        """
        num_cols = []
        cat_cols = []
        
        for col, meta in self.schema_metadata.items():
            if meta["encoding_strategy"] == "DROP" or meta.get("is_target", False):
                continue
            if "NUMERICAL" in meta["inferred_topology"]:
                num_cols.append(col)
            else:
                cat_cols.append(col)
                
        # StandardScaler for continuous data, OneHotEncoder for categorical distances
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
            ],
            remainder="drop"
        )
        logger.debug(f"Preprocessor built: {len(num_cols)} numerical, {len(cat_cols)} categorical.")

    def _transform_data(self) -> None:
        """Applies the transformations to bring data into the evaluation manifold."""
        logger.info("Projecting data into normalized evaluation space...")
        # Convert Polars to Pandas/NumPy for Scikit-Learn compatibility
        real_pd = self.real_df.to_pandas()
        synth_pd = self.synth_df.to_pandas()
        
        # Fit entirely on real data to define the spatial bounds
        self.X_real = self.preprocessor.fit_transform(real_pd)
        self.X_synth = self.preprocessor.transform(synth_pd)
        logger.info(f"Projection complete. Feature manifold dimension: {self.X_real.shape[1]}")

    def compute_dcr(self, alert_threshold_percentile: float = 5.0) -> Dict[str, float]:
        """
        Calculates the Distance to Closest Record (DCR).
        A high DCR proves the model is generating novel samples, not memorizing training data.
        Utilizes KD-Tree for O(N log N) spatial querying.
        """
        logger.info("Computing Distance to Closest Record (DCR)...")
        
        # 1. Fit spatial tree on REAL data
        nn_engine = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', n_jobs=-1)
        nn_engine.fit(self.X_real)
        
        # 2. Query distances from SYNTHETIC data to the closest REAL data points
        distances, _ = nn_engine.kneighbors(self.X_synth)
        distances = distances.flatten()
        
        # Calculate intra-dataset distance to define a dynamic threshold
        # (What is the normal distance between two completely different real people?)
        sample_real = self.X_real[np.random.choice(self.X_real.shape[0], min(10000, self.X_real.shape[0]), replace=False)]
        intra_distances, _ = nn_engine.kneighbors(sample_real, n_neighbors=2)
        intra_distances = intra_distances[:, 1] # 0 is self, 1 is the closest other person
        
        baseline_threshold = np.percentile(intra_distances, alert_threshold_percentile)
        
        # Metrics compilation
        median_dcr = float(np.median(distances))
        p5_dcr = float(np.percentile(distances, 5))
        memorization_ratio = float(np.mean(distances < baseline_threshold))
        
        logger.info(f"DCR Median: {median_dcr:.4f} | Memorization Ratio: {memorization_ratio:.2%}")
        
        return {
            "dcr_median": median_dcr,
            "dcr_5th_percentile": p5_dcr,
            "baseline_threshold": float(baseline_threshold),
            "memorization_risk_ratio": memorization_ratio,
            "passed_dcr_audit": memorization_ratio < 0.05
        }

    def simulate_mia_attack(self) -> Dict[str, float]:
        """
        Simulates a Black-Box Membership Inference Attack (MIA).
        Trains a shadow classifier to distinguish between Real and Synthetic data.
        If ROC-AUC is near 0.5, the synthetic data is indistinguishable (High Privacy).
        """
        logger.info("Simulating Black-Box Membership Inference Attack (MIA)...")
        
        # Create adversarial dataset: Real = 1, Synthetic = 0
        y_real = np.ones(self.X_real.shape[0])
        y_synth = np.zeros(self.X_synth.shape[0])
        
        X_attack = np.vstack([self.X_real, self.X_synth])
        y_attack = np.concatenate([y_real, y_synth])
        
        # Split into shadow training and evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_attack, y_attack, test_size=0.3, random_state=42, stratify=y_attack
        )
        
        # Train shadow adversary (Random Forest provides strong non-linear boundaries)
        adversary = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
        adversary.fit(X_train, y_train)
        
        # Predict probabilities
        y_pred_proba = adversary.predict_proba(X_test)[:, 1]
        
        # Calculate ROC-AUC
        mia_roc_auc = float(roc_auc_score(y_test, y_pred_proba))
        
        logger.info(f"MIA Adversary ROC-AUC: {mia_roc_auc:.4f} (Target is ~0.5)")
        
        # In a perfect privacy scenario, the classifier guesses randomly (AUC = 0.5).
        # We allow a margin up to 0.65 for acceptable privacy.
        return {
            "mia_roc_auc": mia_roc_auc,
            "passed_mia_audit": mia_roc_auc < 0.65
        }

    def run_audit(self) -> Dict[str, Any]:
        """Orchestrates the full privacy evaluation and returns a formal audit report."""
        dcr_metrics = self.compute_dcr()
        mia_metrics = self.simulate_mia_attack()
        
        is_compliant = dcr_metrics["passed_dcr_audit"] and mia_metrics["passed_mia_audit"]
        
        report = {
            "audit_type": "Privacy & Memorization",
            "lgpd_compliant": is_compliant,
            "metrics": {
                **dcr_metrics,
                **mia_metrics
            }
        }
        return report

if __name__ == "__main__":
    # Smoke Test: Mocking the OAA execution
    import json
    
    # Generate mock DataFrames
    schema = {
        "AGE": {"inferred_topology": "NUMERICAL", "encoding_strategy": "QT"},
        "GENDER": {"inferred_topology": "CATEGORICAL", "encoding_strategy": "EMB"},
    }
    
    real_mock = pl.DataFrame({
        "AGE": np.random.normal(40, 10, 1000),
        "GENDER": np.random.choice(["M", "F"], 1000)
    })
    
    # Synthetic data (slightly shifted to simulate generation)
    synth_mock = pl.DataFrame({
        "AGE": np.random.normal(40.5, 9.5, 1000),
        "GENDER": np.random.choice(["M", "F"], 1000, p=[0.52, 0.48])
    })
    
    auditor = PrivacyEvaluator(real_mock, synth_mock, schema)
    final_report = auditor.run_audit()
    
    print("\n--- OAA PRIVACY AUDIT REPORT ---")
    print(json.dumps(final_report, indent=4))