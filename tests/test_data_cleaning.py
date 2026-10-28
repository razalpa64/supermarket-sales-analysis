"""
Unit tests for data cleaning, temporal parsing, and financial verification.
"""

import pandas as pd
import numpy as np
import pytest

from src.data_cleaning import clean_and_prepare_data
from src.config import RAW_DATA_FILE


def test_clean_and_prepare_data_outputs():
    clean_df, daily_df = clean_and_prepare_data(RAW_DATA_FILE, save_outputs=False)

    assert len(clean_df) == 1000
    assert len(daily_df) == 89  # 89 calendar days between 2019-01-01 and 2019-03-30

    # Temporal feature existence
    for col in ["DayOfWeek", "DayName", "Month", "Hour", "Minute", "IsWeekend"]:
        assert col in clean_df.columns

    # Daily aggregation fields
    for col in ["Date", "Daily_Revenue", "Transaction_Count", "Total_Quantity", "DayOfWeek"]:
        assert col in daily_df.columns


def test_financial_calculations_consistency():
    clean_df, _ = clean_and_prepare_data(RAW_DATA_FILE, save_outputs=False)

    # Total must equal Unit price * Quantity * 1.05 within rounding
    expected_totals = clean_df["Unit price"] * clean_df["Quantity"] * 1.05
    diffs = np.abs(clean_df["Total"] - expected_totals)
    assert np.max(diffs) < 0.01

    # Tax 5% must equal cogs * 0.05
    if "Tax 5%" in clean_df.columns and "cogs" in clean_df.columns:
        expected_tax = clean_df["cogs"] * 0.05
        tax_diffs = np.abs(clean_df["Tax 5%"] - expected_tax)
        assert np.max(tax_diffs) < 0.01


def test_daily_date_continuity():
    _, daily_df = clean_and_prepare_data(RAW_DATA_FILE, save_outputs=False)

    # Verify no missing dates in range
    daily_df["Date"] = pd.to_datetime(daily_df["Date"])
    date_diffs = daily_df["Date"].diff().dropna()
    assert (date_diffs == pd.Timedelta(days=1)).all()

