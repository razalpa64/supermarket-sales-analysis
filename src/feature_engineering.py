"""
Feature Engineering and Preprocessing Pipeline Module.
Implements scikit-learn transformers, target leakage auditing,
and causal time-series lag feature builders.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.config import (
    TARGET_COLUMN,
    LEAKAGE_COLUMNS,
    SCENARIO_A_NUMERIC_FEATURES,
    SCENARIO_A_CATEGORICAL_FEATURES,
    SCENARIO_B_NUMERIC_FEATURES,
    SCENARIO_B_CATEGORICAL_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)


class TargetLeakageAuditor:
    """
    Detects and flags target leakage variables that trivialize regression models.
    """

    @staticmethod
    def audit_features(df: pd.DataFrame, candidate_cols: List[str]) -> Dict[str, Any]:
        flagged = []
        clean = []
        for col in candidate_cols:
            if col in LEAKAGE_COLUMNS or col == TARGET_COLUMN:
                flagged.append(col)
            else:
                clean.append(col)

        return {
            "has_leakage": len(flagged) > 0,
            "flagged_columns": flagged,
            "clean_columns": clean,
            "explanation": (
                "Columns such as 'cogs', 'Tax 5%', and 'gross income' are exact linear multiples "
                "or fractions of 'Total' (e.g. cogs = Total / 1.05). Including them yields an "
                "artificial R² = 1.000 without learning real operational relationships."
            ),
        }


class SupermarketFeaturePipeline:
    """
    Constructs scikit-learn preprocessing pipelines for Scenario A and Scenario B.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def get_preprocessor(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
    ) -> ColumnTransformer:
        """
        Builds a standard scikit-learn ColumnTransformer.
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ]
        )
        return preprocessor

    def prepare_scenario_a(
        self,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> Dict[str, Any]:
        """
        Prepares Scenario A: Transaction Value Estimation using Unit Price and Quantity.
        Features: Unit price, Quantity, plus categorical transaction context.
        """
        num_cols = [c for c in SCENARIO_A_NUMERIC_FEATURES if c in self.df.columns]
        cat_cols = [c for c in SCENARIO_A_CATEGORICAL_FEATURES if c in self.df.columns]

        X = self.df[num_cols + cat_cols]
        y = self.df[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        preprocessor = self.get_preprocessor(num_cols, cat_cols)

        return {
            "scenario": "Scenario A: Transaction Value Estimation",
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "numeric_features": num_cols,
            "categorical_features": cat_cols,
            "preprocessor": preprocessor,
        }

    def prepare_scenario_b(
        self,
        test_size: float = TEST_SIZE,
        random_state: int = RANDOM_STATE,
    ) -> Dict[str, Any]:
        """
        Prepares Scenario B: Context-Only Transaction Spend Prediction.
        Features: Customer type, Gender, Department, Payment, Hour, DayOfWeek, Rating (NO item scan).
        """
        num_cols = [c for c in SCENARIO_B_NUMERIC_FEATURES if c in self.df.columns]
        cat_cols = [c for c in SCENARIO_B_CATEGORICAL_FEATURES if c in self.df.columns]

        X = self.df[num_cols + cat_cols]
        y = self.df[TARGET_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        preprocessor = self.get_preprocessor(num_cols, cat_cols)

        return {
            "scenario": "Scenario B: Pre-Scan Context Prediction",
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "numeric_features": num_cols,
            "categorical_features": cat_cols,
            "preprocessor": preprocessor,
        }

    @staticmethod
    def build_time_series_features(daily_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Builds causal lag and calendar features on daily aggregated sales.
        Strictly prevents look-ahead leakage.
        """
        ts = daily_df.copy().sort_values("Date").reset_index(drop=True)

        for lag in [1, 2, 3, 7, 14]:
            ts[f"Lag_{lag}"] = ts["Daily_Revenue"].shift(lag)

        # Rolling statistics strictly from past days (shift 1)
        ts["Rolling_Mean_7"] = ts["Daily_Revenue"].shift(1).rolling(window=7, min_periods=1).mean()
        ts["Rolling_Std_7"] = ts["Daily_Revenue"].shift(1).rolling(window=7, min_periods=1).std().fillna(0)

        ts["DayOfWeek"] = ts["Date"].dt.dayofweek
        ts["DayOfMonth"] = ts["Date"].dt.day
        ts["IsWeekend"] = ts["DayOfWeek"].isin([5, 6]).astype(int)

        feature_cols = [
            "Lag_1", "Lag_2", "Lag_3", "Lag_7", "Lag_14",
            "Rolling_Mean_7", "Rolling_Std_7",
            "DayOfWeek", "DayOfMonth", "IsWeekend",
        ]

        ts_clean = ts.dropna(subset=feature_cols).reset_index(drop=True)
        return ts_clean, feature_cols

