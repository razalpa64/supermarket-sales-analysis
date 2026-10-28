"""
Model Training and Cross-Algorithm Benchmarking Module.
Trains, evaluates, and persists candidate machine learning pipelines
for transaction value estimation and context-based prediction.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    MODELS_DIR,
    REPORTS_DIR,
    CLEANED_DATA_CSV,
    RANDOM_STATE,
)
from src.data_loader import load_dataset
from src.data_cleaning import clean_and_prepare_data
from src.feature_engineering import SupermarketFeaturePipeline


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes MAE, RMSE, and R2 with zero-division safeguards.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # MAPE calculation with epsilon
    epsilon = 1e-6
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon)))) * 100

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE_Pct": round(mape, 2),
    }


def train_scenario_models(
    scenario_dict: Dict[str, Any],
    models_to_test: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], str]:
    """
    Fits and evaluates a dictionary of model architectures on a prepared scenario.
    """
    X_train = scenario_dict["X_train"]
    X_test = scenario_dict["X_test"]
    y_train = scenario_dict["y_train"]
    y_test = scenario_dict["y_test"]
    preprocessor = scenario_dict["preprocessor"]

    results = []
    fitted_pipelines = {}

    for name, regressor in models_to_test.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ])

        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        train_preds = pipeline.predict(X_train)
        test_preds = pipeline.predict(X_test)

        train_metrics = compute_regression_metrics(y_train.values, train_preds)
        test_metrics = compute_regression_metrics(y_test.values, test_preds)

        results.append({
            "Algorithm": name,
            "Train_MAE": train_metrics["MAE"],
            "Train_RMSE": train_metrics["RMSE"],
            "Train_R2": train_metrics["R2"],
            "Val_MAE": test_metrics["MAE"],
            "Val_RMSE": test_metrics["RMSE"],
            "Val_R2": test_metrics["R2"],
            "Val_MAPE_Pct": test_metrics["MAPE_Pct"],
            "Train_Samples": len(X_train),
            "Val_Samples": len(X_test),
        })

    comparison_df = pd.DataFrame(results).sort_values("Val_RMSE").reset_index(drop=True)
    best_model_name = comparison_df.iloc[0]["Algorithm"]
    return comparison_df, fitted_pipelines, best_model_name


def train_supermarket_models(df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Executes complete model training for both Scenario A and Scenario B,
    persisting best artifacts and benchmark reports.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if df is None:
        if not CLEANED_DATA_CSV.exists():
            df, _ = clean_and_prepare_data()
        else:
            df = load_dataset(CLEANED_DATA_CSV)

    feature_pipeline = SupermarketFeaturePipeline(df)

    # Candidate regressors for comparison
    candidate_models = {
        "Historical Mean Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression (OLS)": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=RANDOM_STATE
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.05, random_state=RANDOM_STATE
        ),
        "Extra Trees Regressor": ExtraTreesRegressor(
            n_estimators=100, max_depth=12, random_state=RANDOM_STATE
        ),
    }

    print("\n--- Training Scenario A Models (Transaction Value Estimation) ---")
    scen_a = feature_pipeline.prepare_scenario_a()
    comp_a, pipes_a, best_a_name = train_scenario_models(scen_a, candidate_models)
    comp_a.to_csv(REPORTS_DIR / "scenario_a_model_comparison.csv", index=False)
    joblib.dump(pipes_a[best_a_name], MODELS_DIR / "scenario_a_best_model.joblib")
    joblib.dump(pipes_a["Ridge Regression"], MODELS_DIR / "scenario_a_ridge_model.joblib")
    joblib.dump(pipes_a["Random Forest Regressor"], MODELS_DIR / "scenario_a_rf_model.joblib")
    print(f"[SUCCESS] Scenario A Best Model: '{best_a_name}' (RMSE: {comp_a.iloc[0]['Val_RMSE']})")

    print("\n--- Training Scenario B Models (Pre-Scan Operational Context) ---")
    scen_b = feature_pipeline.prepare_scenario_b()
    comp_b, pipes_b, best_b_name = train_scenario_models(scen_b, candidate_models)
    comp_b.to_csv(REPORTS_DIR / "scenario_b_model_comparison.csv", index=False)
    joblib.dump(pipes_b[best_b_name], MODELS_DIR / "scenario_b_best_model.joblib")
    print(f"[SUCCESS] Scenario B Best Model: '{best_b_name}' (RMSE: {comp_b.iloc[0]['Val_RMSE']})")

    return {
        "scenario_a_comparison": comp_a,
        "scenario_b_comparison": comp_b,
        "best_scenario_a_model": best_a_name,
        "best_scenario_b_model": best_b_name,
    }


if __name__ == "__main__":
    train_supermarket_models()

