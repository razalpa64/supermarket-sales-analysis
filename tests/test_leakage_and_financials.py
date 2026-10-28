"""
Unit tests for leakage detection and financial arithmetic sanity.
"""

import pandas as pd
from src.config import LEAKAGE_COLUMNS, TARGET_COLUMN
from src.feature_engineering import TargetLeakageAuditor, SupermarketFeaturePipeline
from src.data_cleaning import clean_and_prepare_data


def test_leakage_auditor_flags_tautological_columns():
    df = pd.DataFrame({
        "Total": [100.0, 200.0],
        "cogs": [95.23, 190.47],
        "Tax 5%": [4.76, 9.52],
        "Rating": [7.0, 8.5],
        "Hour": [12, 14],
    })

    audit_result = TargetLeakageAuditor.audit_features(df, ["cogs", "Tax 5%", "Rating", "Hour"])
    assert audit_result["has_leakage"] is True
    assert "cogs" in audit_result["flagged_columns"]
    assert "Tax 5%" in audit_result["flagged_columns"]
    assert "Rating" in audit_result["clean_columns"]
    assert "Hour" in audit_result["clean_columns"]


def test_scenario_a_pipeline_excludes_leakage_columns():
    clean_df, _ = clean_and_prepare_data(save_outputs=False)
    pipe = SupermarketFeaturePipeline(clean_df)
    scen_a = pipe.prepare_scenario_a()

    features_used = scen_a["numeric_features"] + scen_a["categorical_features"]
    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in features_used


def test_scenario_b_pipeline_excludes_scan_level_and_leakage_features():
    clean_df, _ = clean_and_prepare_data(save_outputs=False)
    pipe = SupermarketFeaturePipeline(clean_df)
    scen_b = pipe.prepare_scenario_b()

    features_used = scen_b["numeric_features"] + scen_b["categorical_features"]
    assert "Unit price" not in features_used
    assert "Quantity" not in features_used
    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in features_used

