"""
Future Sales Forecasting Module for Supermarket Sales System.
Aggregates transactions into daily revenue time-series, engineers causal autoregressive lag features,
performs chronological out-of-sample train/test evaluation, benchmarks baselines against ML models,
and produces recursive multi-step forecasts with 95% confidence intervals across 1–14 day horizons.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from src.config import (
    DAILY_SALES_FILE,
    CLEANED_DATA_CSV,
    FIGURES_DIR,
    REPORTS_DIR,
    PREDICTIONS_DIR,
    DEFAULT_FORECAST_HORIZON,
    RANDOM_STATE,
    COLOR_PRIMARY_ACCENT,
    COLOR_PRIMARY_TEXT,
    COLOR_SECONDARY_TEXT,
    COLOR_BORDER,
    CURRENCY_SYMBOL,
)
from src.data_cleaning import clean_and_prepare_data


class SupermarketSalesForecaster:
    """
    Time-series forecaster for aggregate supermarket daily sales.
    """

    def __init__(self, daily_df: Optional[pd.DataFrame] = None):
        if daily_df is None:
            if not DAILY_SALES_FILE.exists():
                _, daily_df = clean_and_prepare_data()
            else:
                daily_df = pd.read_csv(DAILY_SALES_FILE)

        self.df = daily_df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.df = self.df.sort_values("Date").reset_index(drop=True)

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    def engineer_time_series_features(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Engineers autoregressive lag features, rolling statistics, and calendar attributes.
        Strictly prevents look-ahead leakage.
        """
        ts_df = self.df.copy()

        # Lag features (causal only)
        for lag in [1, 2, 3, 7, 14]:
            ts_df[f"Lag_{lag}"] = ts_df["Daily_Revenue"].shift(lag)

        # Rolling statistics strictly from past days
        ts_df["Rolling_Mean_7"] = ts_df["Daily_Revenue"].shift(1).rolling(window=7, min_periods=1).mean()
        ts_df["Rolling_Std_7"] = ts_df["Daily_Revenue"].shift(1).rolling(window=7, min_periods=1).std().fillna(0)

        # Calendar features
        ts_df["DayOfWeek"] = ts_df["Date"].dt.dayofweek
        ts_df["DayOfMonth"] = ts_df["Date"].dt.day
        ts_df["IsWeekend"] = ts_df["DayOfWeek"].isin([5, 6]).astype(int)

        feature_cols = [
            "Lag_1", "Lag_2", "Lag_3", "Lag_7", "Lag_14",
            "Rolling_Mean_7", "Rolling_Std_7",
            "DayOfWeek", "DayOfMonth", "IsWeekend",
        ]

        # Drop rows where initial lags are NaN
        ts_clean = ts_df.dropna(subset=feature_cols).reset_index(drop=True)
        return ts_clean, feature_cols

    def evaluate_forecast_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Computes MAE, RMSE, R2, and MAPE for time-series forecasts.
        """
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else 0.0

        epsilon = 1e-6
        mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon)))) * 100

        return {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
            "MAPE_Pct": round(mape, 2),
        }

    def run_benchmark_and_evaluation(
        self,
        test_horizon_days: int = DEFAULT_FORECAST_HORIZON,
    ) -> Dict[str, Any]:
        """
        Splits data chronologically:
        - Test set: Last `test_horizon_days` days
        - Train set: All preceding days
        Evaluates Naive, Seasonal Naive, Moving Average, AR Ridge, Random Forest, Gradient Boosting,
        and Exponential Smoothing.
        """
        test_horizon_days = max(1, min(14, int(test_horizon_days)))
        ts_df, feature_cols = self.engineer_time_series_features()
        n = len(ts_df)

        if n < test_horizon_days * 2 + 14:
            raise ValueError(f"Insufficient history ({n} days) for {test_horizon_days}-day chronological evaluation.")

        split_test = n - test_horizon_days
        train_df = ts_df.iloc[:split_test].copy()
        test_df = ts_df.iloc[split_test:].copy()

        X_train, y_train = train_df[feature_cols], train_df["Daily_Revenue"]
        X_test, y_test = test_df[feature_cols], test_df["Daily_Revenue"]

        predictions = {}

        # 1. Naive Baseline (Last observed value before test window)
        last_train_val = train_df["Daily_Revenue"].iloc[-1]
        predictions["Naive Baseline (Last Value)"] = np.full(shape=test_horizon_days, fill_value=last_train_val)

        # 2. 7-Day Moving Average Baseline
        predictions["7-Day Moving Average"] = test_df["Rolling_Mean_7"].values

        # 3. Seasonal Naive (7-Day Lag)
        predictions["Seasonal Naive (7-Day Lag)"] = test_df["Lag_7"].values

        # 4. Autoregressive Ridge Regression
        ridge = Ridge(alpha=10.0, random_state=RANDOM_STATE)
        ridge.fit(X_train, y_train)
        predictions["Autoregressive Ridge"] = ridge.predict(X_test)

        # 5. Random Forest Forecaster
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
        rf.fit(X_train, y_train)
        predictions["Random Forest Forecaster"] = rf.predict(X_test)

        # 6. Gradient Boosting Regressor
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE)
        gb.fit(X_train, y_train)
        predictions["Gradient Boosting Forecaster"] = gb.predict(X_test)

        # 7. Exponential Smoothing (Holt-Winters)
        if HAS_STATSMODELS:
            try:
                hw = ExponentialSmoothing(
                    train_df["Daily_Revenue"].values,
                    seasonal="add",
                    seasonal_periods=7,
                    trend="add",
                    damped_trend=True,
                ).fit()
                hw_preds = hw.forecast(test_horizon_days)
                if len(hw_preds) == test_horizon_days and not np.isnan(hw_preds).any():
                    predictions["Holt-Winters Exp Smoothing"] = hw_preds
            except Exception:
                try:
                    hw_simple = ExponentialSmoothing(train_df["Daily_Revenue"].values).fit()
                    hw_preds = hw_simple.forecast(test_horizon_days)
                    if len(hw_preds) == test_horizon_days and not np.isnan(hw_preds).any():
                        predictions["Simple Exp Smoothing"] = hw_preds
                except Exception:
                    pass

        # Evaluate all candidate models
        results_records = []
        for model_name, preds in predictions.items():
            metrics = self.evaluate_forecast_metrics(y_test.values, preds)
            results_records.append({
                "Model": model_name,
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "MAPE_Pct": metrics["MAPE_Pct"],
                "Test_Obs": test_horizon_days,
            })

        metrics_df = pd.DataFrame(results_records).sort_values("RMSE").reset_index(drop=True)
        metrics_df.to_csv(REPORTS_DIR / "forecasting_model_comparison.csv", index=False)

        best_model_name = metrics_df.iloc[0]["Model"]
        best_test_preds = predictions[best_model_name]

        # Model instance to use for future unseen recursive forecast
        model_obj = rf if "Random Forest" in best_model_name else (gb if "Gradient Boosting" in best_model_name else ridge)

        # Generate Future Forecast (beyond the available dataset)
        future_forecast, future_dates, lower_bound, upper_bound = self.generate_future_forecast(
            ts_df, feature_cols, model_obj, horizon_days=test_horizon_days
        )

        forecast_export = pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d") for d in future_dates],
            "Forecasted_Daily_Revenue": np.round(future_forecast, 2),
            "Lower_95_CI": np.round(lower_bound, 2),
            "Upper_95_CI": np.round(upper_bound, 2),
            "Model_Used": best_model_name,
            "Horizon_Days": test_horizon_days,
        })
        forecast_export.to_csv(PREDICTIONS_DIR / f"daily_sales_forecast_{test_horizon_days}days.csv", index=False)
        forecast_export.to_csv(PREDICTIONS_DIR / "daily_sales_forecast_7days.csv", index=False)

        # Matplotlib visualization export
        fig_path = self._plot_forecast_results(
            train_df=train_df,
            test_df=test_df,
            predictions=predictions,
            best_model_name=best_model_name,
            future_dates=future_dates,
            future_forecast=future_forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            horizon_days=test_horizon_days,
        )

        return {
            "metrics": metrics_df,
            "best_model_name": best_model_name,
            "best_test_predictions": best_test_preds,
            "all_predictions": predictions,
            "future_forecast_df": forecast_export,
            "figure_path": fig_path,
            "test_actuals": y_test.values,
            "test_dates": test_df["Date"].values,
            "train_dates": train_df["Date"].values,
            "train_actuals": train_df["Daily_Revenue"].values,
            "future_dates": future_dates,
            "future_forecast": future_forecast,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "ts_df": ts_df,
            "feature_cols": feature_cols,
        }

    def generate_future_forecast(
        self,
        ts_df: pd.DataFrame,
        feature_cols: List[str],
        model: Any,
        horizon_days: int = 7,
    ) -> Tuple[np.ndarray, List[pd.Timestamp], np.ndarray, np.ndarray]:
        """
        Multi-step recursive future forecasting for next horizon_days into the unseen future.
        """
        horizon_days = max(1, min(14, int(horizon_days)))
        history_rev = list(ts_df["Daily_Revenue"].values)
        last_date = ts_df["Date"].iloc[-1]
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, horizon_days + 1)]
        future_preds = []

        for f_date in future_dates:
            lag_1 = history_rev[-1]
            lag_2 = history_rev[-2] if len(history_rev) >= 2 else lag_1
            lag_3 = history_rev[-3] if len(history_rev) >= 3 else lag_1
            lag_7 = history_rev[-7] if len(history_rev) >= 7 else lag_1
            lag_14 = history_rev[-14] if len(history_rev) >= 14 else lag_1
            roll_7 = np.mean(history_rev[-7:])
            std_7 = np.std(history_rev[-7:])
            dow = f_date.dayofweek
            dom = f_date.day
            is_wknd = 1 if dow in [5, 6] else 0

            feat_row = pd.DataFrame([{
                "Lag_1": lag_1, "Lag_2": lag_2, "Lag_3": lag_3,
                "Lag_7": lag_7, "Lag_14": lag_14,
                "Rolling_Mean_7": roll_7, "Rolling_Std_7": std_7,
                "DayOfWeek": dow, "DayOfMonth": dom, "IsWeekend": is_wknd,
            }])[feature_cols]

            pred_val = float(model.predict(feat_row)[0])
            pred_val = max(100.0, pred_val)
            future_preds.append(pred_val)
            history_rev.append(pred_val)

        future_preds_arr = np.array(future_preds)
        std_resid = float(np.std(ts_df["Daily_Revenue"] - ts_df["Rolling_Mean_7"]))
        lower_bound = np.maximum(0.0, future_preds_arr - 1.96 * std_resid)
        upper_bound = future_preds_arr + 1.96 * std_resid

        return future_preds_arr, future_dates, lower_bound, upper_bound

    def _plot_forecast_results(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        predictions: Dict[str, np.ndarray],
        best_model_name: str,
        future_dates: List[pd.Timestamp],
        future_forecast: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        horizon_days: int = 7,
    ) -> Path:
        """
        Plots historical training actuals, held-out test actuals vs forecasts, and future projection.
        """
        fig, ax = plt.subplots(figsize=(15, 6))

        # Recent 30 days of training
        recent_train = train_df.iloc[-30:]
        ax.plot(recent_train["Date"], recent_train["Daily_Revenue"], color="#94A3B8", linewidth=1.5, label="Historical Training Actuals")

        # Held-out test actuals
        ax.plot(test_df["Date"], test_df["Daily_Revenue"], color="#172033", marker="o", linewidth=2.2, label="Held-Out Test Actuals")

        # Best model on test set
        best_preds = predictions.get(best_model_name, None)
        if best_preds is not None:
            ax.plot(test_df["Date"], best_preds, color="#2563EB", linestyle="--", linewidth=2.2, label=f"Test Prediction ({best_model_name})")

        # Future projection
        ax.plot(future_dates, future_forecast, color="#059669", marker="s", linewidth=2.5, label=f"Future {horizon_days}-Day Forecast ({best_model_name})")
        ax.fill_between(future_dates, lower_bound, upper_bound, color="#10B981", alpha=0.18, label="95% Confidence Band")

        ax.axvline(test_df["Date"].iloc[0], color="#CBD5E1", linestyle="--", alpha=0.8)
        ax.axvline(future_dates[0], color="#CBD5E1", linestyle=":", alpha=0.8)

        ax.set_title(f"Supermarket Daily Sales Forecasting: Historical Test & {horizon_days}-Day Future Horizon", fontweight="bold", fontsize=12, color=COLOR_PRIMARY_TEXT)
        ax.set_xlabel("Date", color=COLOR_PRIMARY_TEXT)
        ax.set_ylabel(f"Daily Revenue ({CURRENCY_SYMBOL})", color=COLOR_PRIMARY_TEXT)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
        plt.tight_layout()

        fig_path = FIGURES_DIR / f"09_future_sales_forecasting_{horizon_days}days.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        return fig_path


def run_future_sales_forecasting(horizon_days: int = DEFAULT_FORECAST_HORIZON) -> Dict[str, Any]:
    """
    Convenience function to run future sales forecasting pipeline.
    """
    forecaster = SupermarketSalesForecaster()
    return forecaster.run_benchmark_and_evaluation(test_horizon_days=horizon_days)


if __name__ == "__main__":
    run_future_sales_forecasting()
