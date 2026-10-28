"""
Unit and integration tests for data loading and schema validation.
"""

import io
import pytest
import pandas as pd
from pathlib import Path

from src.data_loader import SupermarketDataLoader, DataValidationError
from src.config import RAW_DATA_FILE


def test_discover_raw_files():
    loader = SupermarketDataLoader()
    discovered = loader.discover_raw_files()
    assert "csv" in discovered
    assert "excel" in discovered
    assert "sqlite" in discovered
    assert len(discovered["csv"]) > 0


def test_load_from_csv():
    loader = SupermarketDataLoader()
    df = loader.load_from_csv(RAW_DATA_FILE)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1000
    assert "Total" in df.columns
    assert "Unit price" in df.columns
    assert "Quantity" in df.columns


def test_validate_schema_valid_df():
    loader = SupermarketDataLoader()
    df = loader.load_from_csv(RAW_DATA_FILE)
    is_valid, report = loader.validate_schema(df)
    assert is_valid is True
    assert report["num_rows"] == 1000
    assert len(report["missing_critical_columns"]) == 0


def test_validate_schema_missing_critical_column():
    loader = SupermarketDataLoader()
    df_broken = pd.DataFrame({
        "Branch": ["A"],
        "City": ["Yangon"],
        # Missing 'Total', 'Unit price', 'Quantity', etc.
    })
    is_valid, report = loader.validate_schema(df_broken)
    assert is_valid is False
    assert "Total" in report["missing_critical_columns"]


def test_load_from_buffer_valid_csv():
    loader = SupermarketDataLoader()
    csv_content = (
        "Invoice ID,Branch,City,Customer type,Gender,Product line,Unit price,Quantity,Tax 5%,Total,Date,Time,Payment,cogs,gross margin percentage,gross income,Rating\n"
        "101-11-1001,A,Yangon,Member,Female,Health and beauty,50.0,2,5.0,105.0,1/5/2019,13:08,Cash,100.0,4.7619,5.0,9.1\n"
    )
    buffer = io.BytesIO(csv_content.encode("utf-8"))
    df = loader.load_from_buffer(buffer, "test.csv")
    assert len(df) == 1
    assert df.iloc[0]["Total"] == 105.0


def test_load_from_buffer_invalid_csv_raises():
    loader = SupermarketDataLoader()
    csv_broken = "SomeColumn,AnotherColumn\n1,2\n"
    buffer = io.BytesIO(csv_broken.encode("utf-8"))
    with pytest.raises(DataValidationError):
        loader.load_from_buffer(buffer, "broken.csv")


def test_inspect_dataset():
    loader = SupermarketDataLoader()
    df = loader.load_from_csv(RAW_DATA_FILE)
    info = loader.inspect_dataset(df)
    assert info["shape"] == (1000, 17)
    assert info["duplicate_count"] == 0
    assert "Total" in info["columns"]

