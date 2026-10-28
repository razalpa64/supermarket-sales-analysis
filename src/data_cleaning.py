"""
Data Cleaning and Preparation Module for Supermarket Sales System.
Validates schemas, parses temporal features, audits financial calculations,
and generates contiguous daily aggregated time-series data.
"""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_FILE,
    CLEANED_DATA_CSV,
    CLEANED_DATA_FILE,
    DAILY_SALES_FILE,
    PROCESSED_DATA_DIR,
)
from src.data_loader import SupermarketDataLoader


def clean_and_prepare_data(
    input_file: Optional[Path] = None,
    save_outputs: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans raw transaction data, derives temporal attributes, verifies financial
    consistency, and computes aggregate daily sales.

    Returns:
        (cleaned_df, daily_df): tuple of cleaned transactions and daily aggregate dataframe.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    loader = SupermarketDataLoader()
    raw_df = loader.load_from_csv(input_file or RAW_DATA_FILE)

    df = raw_df.copy()

    # 1. Deduplication and null handling
    df = df.drop_duplicates()
    df = df.dropna(subset=["Total", "Unit price", "Quantity", "Date"])

    # 2. Date parsing and calendar feature engineering
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["DayName"] = df["Date"].dt.day_name()
    df["Month"] = df["Date"].dt.month
    df["DayOfMonth"] = df["Date"].dt.day
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

    # 3. Time parsing and temporal binning
    if "Time" in df.columns:
        parsed_time = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.time
        df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.hour
        df["Minute"] = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce").dt.minute
    else:
        df["Hour"] = 12
        df["Minute"] = 0

    # 4. Standardize text categories
    categorical_cols = ["Branch", "City", "Customer type", "Gender", "Product line", "Payment"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 5. Financial verification & consistency check
    # cogs = Unit price * Quantity
    # Tax 5% = cogs * 0.05
    # Total = cogs + Tax 5% = 1.05 * cogs
    expected_cogs = df["Unit price"] * df["Quantity"]
    expected_tax = expected_cogs * 0.05
    expected_total = expected_cogs + expected_tax

    # Audit discrepancy (should be near machine precision)
    max_cogs_diff = np.max(np.abs(df["cogs"] - expected_cogs)) if "cogs" in df.columns else 0.0
    max_total_diff = np.max(np.abs(df["Total"] - expected_total))

    if max_total_diff > 0.01:
        print(f"[WARN] Total amount discrepancy detected (max diff: {max_total_diff:.4f}). Re-aligning with actual rules.")
        df["Total"] = np.round(expected_total, 4)

    # 6. Daily aggregate time-series generation
    daily_grouped = (
        df.groupby("Date")
        .agg(
            Daily_Revenue=("Total", "sum"),
            Transaction_Count=("Invoice ID", "count") if "Invoice ID" in df.columns else ("Total", "count"),
            Total_Quantity=("Quantity", "sum"),
            Average_Transaction_Value=("Total", "mean"),
            Average_Rating=("Rating", "mean") if "Rating" in df.columns else ("Total", "count"),
        )
        .reset_index()
    )

    # Ensure full calendar continuity without missing dates
    min_date = daily_grouped["Date"].min()
    max_date = daily_grouped["Date"].max()
    full_date_range = pd.date_range(start=min_date, end=max_date, freq="D")
    daily_df = (
        daily_grouped.set_index("Date")
        .reindex(full_date_range)
        .rename_axis("Date")
        .reset_index()
    )

    # Fill any missing days with 0 revenue / transactions
    daily_df["Daily_Revenue"] = daily_df["Daily_Revenue"].fillna(0.0)
    daily_df["Transaction_Count"] = daily_df["Transaction_Count"].fillna(0).astype(int)
    daily_df["Total_Quantity"] = daily_df["Total_Quantity"].fillna(0).astype(int)
    daily_df["Average_Transaction_Value"] = daily_df["Average_Transaction_Value"].fillna(0.0)
    if "Average_Rating" in daily_df.columns:
        daily_df["Average_Rating"] = daily_df["Average_Rating"].fillna(daily_df["Average_Rating"].mean())

    daily_df["DayOfWeek"] = daily_df["Date"].dt.dayofweek
    daily_df["DayName"] = daily_df["Date"].dt.day_name()
    daily_df["Month"] = daily_df["Date"].dt.month
    daily_df["IsWeekend"] = daily_df["DayOfWeek"].isin([5, 6]).astype(int)

    # 7. Save outputs
    if save_outputs:
        df.to_csv(CLEANED_DATA_CSV, index=False)
        try:
            df.to_parquet(CLEANED_DATA_FILE, index=False)
        except Exception as e:
            print(f"[NOTE] Parquet save skipped ({e}); CSV saved successfully.")

        daily_df.to_csv(DAILY_SALES_FILE, index=False)
        print(f"[SUCCESS] Cleaned data saved: {len(df)} transactions -> {CLEANED_DATA_CSV}")
        print(f"[SUCCESS] Daily aggregated sales saved: {len(daily_df)} contiguous days -> {DAILY_SALES_FILE}")

    return df, daily_df


if __name__ == "__main__":
    clean_and_prepare_data()

