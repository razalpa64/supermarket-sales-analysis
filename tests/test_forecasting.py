"""
Unit and integration tests for time-series feature engineering and multi-day forecasting.
"""

import pandas as pd
import numpy as np
import pytest

from src.forecasting import SupermarketSalesForecaster
from src.data_cleaning import clean_and_prepare_data


def test_time_series_feature_engineering_strictly_causal():
    _, daily_df = clean_and_prepare_data(save_outputs=False)
    forecaster = SupermarketSalesForecaster(daily_df)
    ts_df, feature_cols = forecaster.engineer_time_series_features()

    assert len(ts_df) > 0
    # Lags must exist
    for lag in [1, 2, 3, 7, 14]:
        assert f"Lag_{lag}" in feature_cols

    # Check that Lag_1 equals previous row's Daily_Revenue
    for i in range(1, len(ts_df)):
        assert ts_df.iloc[i]["Lag_1"] == daily_df[daily_df["Date"] == ts_df.iloc[i]["Date"] - pd.Timedelta(days=1)]["Daily_Revenue"].values[0]


def test_forecasting_benchmark_and_evaluation_7days():
    _, daily_df = clean_and_prepare_data(save_outputs=False)
    forecaster = SupermarketSalesForecaster(daily_df)
    results = forecaster.run_benchmark_and_evaluation(test_horizon_days=7)

    assert "metrics" in results
    assert "best_model_name" in results
    assert "future_forecast_df" in results

    metrics_df = results["metrics"]
    assert len(metrics_df) >= 4
    assert "RMSE" in metrics_df.columns
    assert "MAE" in metrics_df.columns

    forecast_df = results["future_forecast_df"]
    assert len(forecast_df) == 7
    assert (forecast_df["Forecasted_Daily_Revenue"] > 0).all()
    assert (forecast_df["Lower_95_CI"] <= forecast_df["Forecasted_Daily_Revenue"]).all()
    assert (forecast_df["Upper_95_CI"] >= forecast_df["Forecasted_Daily_Revenue"]).all()


@pytest.mark.parametrize("horizon", [1, 3, 7, 14])
def test_forecasting_supports_1_to_14_day_horizons(horizon):
    _, daily_df = clean_and_prepare_data(save_outputs=False)
    forecaster = SupermarketSalesForecaster(daily_df)
    results = forecaster.run_benchmark_and_evaluation(test_horizon_days=horizon)
    forecast_df = results["future_forecast_df"]
    assert len(forecast_df) == horizon

