import polars as pl

from datalus.application.audit import PrivacyEvaluator
from datalus.domain.schemas import PrivacyThresholds, ShadowMIAConfig


def test_privacy_audit_report_has_dcr_and_shadow_mia():
    real = pl.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 70, 80, 90],
            "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
        }
    )
    synth = pl.DataFrame(
        {
            "age": [21, 31, 41, 51, 61, 71, 81, 91],
            "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
        }
    )
    schema = {
        "age": {
            "inferred_topology": "NUMERICAL_CONTINUOUS",
            "encoding_strategy": "QUANTILE_TRANSFORM",
        },
        "sex": {
            "inferred_topology": "CATEGORICAL_LOW_CARDINALITY",
            "encoding_strategy": "CONTINUOUS_EMBEDDING",
        },
    }
    report = PrivacyEvaluator(real, synth, schema).run_audit(
        thresholds=PrivacyThresholds(memorization_ratio=1.0, mia_roc_auc=1.0)
    )
    assert "dcr_median" in report["privacy"]
    assert "mia_roc_auc" in report["privacy"]


def test_shadow_mia_ci_lite_uses_bounded_fast_configuration():
    real = pl.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 70, 80, 90],
            "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
        }
    )
    synth = pl.DataFrame(
        {
            "age": [21, 31, 41, 51, 61, 71, 81, 91],
            "sex": ["M", "F", "M", "F", "M", "F", "M", "F"],
        }
    )
    schema = {
        "age": {
            "inferred_topology": "NUMERICAL_CONTINUOUS",
            "encoding_strategy": "QUANTILE_TRANSFORM",
        },
        "sex": {
            "inferred_topology": "CATEGORICAL_LOW_CARDINALITY",
            "encoding_strategy": "CONTINUOUS_EMBEDDING",
        },
    }
    result = PrivacyEvaluator(real, synth, schema).shadow_membership_inference(
        config=ShadowMIAConfig(mode="ci_lite", n_shadow_models=4, max_rows=8),
        thresholds=PrivacyThresholds(mia_roc_auc=1.0),
    )
    assert result["mia_mode"] == "ci_lite"
    assert result["shadow_models"] == 2
