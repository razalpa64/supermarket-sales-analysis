"""
Unit tests for model training, metrics computation, and artifact loading.
"""

import numpy as np
import pandas as pd
import pytest
import joblib

from src.model_training import compute_regression_metrics, train_scenario_models
from src.config import MODELS_DIR
from src.data_cleaning import clean_and_prepare_data
from src.feature_engineering import SupermarketFeaturePipeline
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor


def test_compute_regression_metrics_exact_match():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["R2"] == 1.0
    assert metrics["MAPE_Pct"] == 0.0


def test_compute_regression_metrics_known_error():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 190.0])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["MAE"] == 10.0
    assert metrics["RMSE"] == 10.0


def test_train_scenario_models_execution():
    clean_df, _ = clean_and_prepare_data(save_outputs=False)
    pipe = SupermarketFeaturePipeline(clean_df)
    scen_a = pipe.prepare_scenario_a(test_size=0.2)

    candidate_models = {
        "Baseline": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=1.0),
    }

    comp_df, fitted_pipes, best_name = train_scenario_models(scen_a, candidate_models)
    assert len(comp_df) == 2
    assert "Val_RMSE" in comp_df.columns
    assert "Val_R2" in comp_df.columns
    assert best_name in candidate_models


def test_saved_model_artifact_inference():
    model_file = MODELS_DIR / "scenario_a_best_model.joblib"
    if not model_file.exists():
        pytest.skip("Model artifact not yet saved.")

    loaded_model = joblib.load(model_file)
    test_sample = pd.DataFrame([{
        "Unit price": 50.0,
        "Quantity": 4,
        "Branch": "A",
        "City": "Yangon",
        "Customer type": "Member",
        "Gender": "Female",
        "Product line": "Electronic accessories",
        "Payment": "Ewallet",
    }])

    pred = float(loaded_model.predict(test_sample)[0])
    # Exact: 50 * 4 * 1.05 = 210.0
    # Extra Trees should predict close to 210
    assert 180.0 <= pred <= 240.0

