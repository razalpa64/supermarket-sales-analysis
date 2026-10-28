"""
Data Loader Module for Supermarket Sales System.
Supports loading transactions from CSV, Excel, SQLite databases, and user uploads.
Performs automatic schema discovery, validation, and defensive data verification.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
import io
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    RAW_DATA_FILE,
    DATABASE_PATH,
    EXPECTED_RAW_COLUMNS,
)


class DataValidationError(Exception):
    """Custom exception raised when uploaded or loaded dataset fails validation."""
    pass


class SupermarketDataLoader:
    """
    Universal data loader supporting CSV, Excel, SQLite tables, and uploaded file objects.
    """

    def __init__(self, raw_data_dir: Optional[Path] = None):
        self.raw_data_dir = raw_data_dir or RAW_DATA_DIR

    def discover_raw_files(self) -> Dict[str, list]:
        """
        Discovers all CSV, Excel, and SQLite files present in the raw data directory.
        """
        csv_files = list(self.raw_data_dir.glob("*.csv"))
        excel_files = list(self.raw_data_dir.glob("*.xlsx")) + list(self.raw_data_dir.glob("*.xls"))
        db_files = list(self.raw_data_dir.glob("*.db")) + list(self.raw_data_dir.glob("*.sqlite"))

        return {
            "csv": csv_files,
            "excel": excel_files,
            "sqlite": db_files,
        }

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Validates that a dataframe satisfies the supermarket sales schema.
        Returns (is_valid, validation_report).
        """
        missing_columns = [col for col in EXPECTED_RAW_COLUMNS if col not in df.columns]
        extra_columns = [col for col in df.columns if col not in EXPECTED_RAW_COLUMNS]

        null_summary = df.isnull().sum().to_dict()
        total_nulls = int(sum(null_summary.values()))
        num_rows = len(df)
        num_cols = len(df.columns)
        duplicate_count = int(df.duplicated().sum())

        # Check essential columns for modeling
        critical_cols = ["Total", "Unit price", "Quantity", "Date", "Branch", "Product line"]
        missing_critical = [c for c in critical_cols if c not in df.columns]

        is_valid = len(missing_critical) == 0 and num_rows > 0

        report = {
            "is_valid": is_valid,
            "num_rows": num_rows,
            "num_cols": num_cols,
            "missing_expected_columns": missing_columns,
            "missing_critical_columns": missing_critical,
            "extra_columns": extra_columns,
            "total_null_values": total_nulls,
            "duplicate_rows": duplicate_count,
            "columns_present": list(df.columns),
        }
        return is_valid, report

    def load_from_csv(self, filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Loads dataset from a CSV file.
        """
        path = Path(filepath) if filepath else RAW_DATA_FILE
        if not path.exists():
            discovered = self.discover_raw_files().get("csv", [])
            if discovered:
                path = discovered[0]
                print(f"[INFO] Configured path not found. Auto-discovered CSV: {path}")
            else:
                raise FileNotFoundError(f"No CSV dataset found at {path} or in {self.raw_data_dir}")

        df = pd.read_csv(path)
        print(f"[SUCCESS] Loaded {len(df)} rows and {len(df.columns)} columns from CSV: {path.name}")
        return df

    def load_from_excel(self, filepath: Union[str, Path], sheet_name: Union[str, int] = 0) -> pd.DataFrame:
        """
        Loads dataset from an Excel spreadsheet.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Excel file not found at: {path}")
        df = pd.read_excel(path, sheet_name=sheet_name)
        print(f"[SUCCESS] Loaded {len(df)} rows and {len(df.columns)} columns from Excel: {path.name}")
        return df

    def load_from_sqlite(
        self,
        table_name: str = "transactions",
        db_path: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """
        Loads dataset from an SQLite database table.
        """
        import sqlite3

        path = Path(db_path) if db_path else DATABASE_PATH
        if not path.exists():
            raise FileNotFoundError(f"SQLite database not found at: {path}")

        conn = sqlite3.connect(str(path))
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        print(f"[SUCCESS] Loaded {len(df)} rows from SQLite table '{table_name}' in {path.name}")
        return df

    def load_from_buffer(self, file_buffer: Union[io.BytesIO, Any], filename: str) -> pd.DataFrame:
        """
        Loads dataset from an in-memory buffer (such as Streamlit UploadedFile).
        """
        fname = filename.lower()
        if fname.endswith(".csv"):
            df = pd.read_csv(file_buffer)
        elif fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file_buffer)
        else:
            raise ValueError(f"Unsupported file format for '{filename}'. Please upload a CSV or Excel file.")

        is_valid, report = self.validate_schema(df)
        if not is_valid:
            missing = ", ".join(report["missing_critical_columns"])
            raise DataValidationError(f"Uploaded file '{filename}' is missing critical columns: {missing}")

        return df

    def inspect_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Provides detailed structural and statistical summary of loaded dataframe.
        """
        summary = {
            "shape": df.shape,
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
            "duplicate_count": int(df.duplicated().sum()),
            "sample_head": df.head(3).to_dict(orient="records"),
        }
        return summary


def load_dataset(source_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Convenience function to load the primary supermarket sales dataset.
    """
    loader = SupermarketDataLoader()
    if source_path is None:
        return loader.load_from_csv()
    path = Path(source_path)
    if path.suffix.lower() == ".csv":
        return loader.load_from_csv(path)
    elif path.suffix.lower() in [".xlsx", ".xls"]:
        return loader.load_from_excel(path)
    elif path.suffix.lower() in [".db", ".sqlite"]:
        return loader.load_from_sqlite(db_path=path)
    else:
        return loader.load_from_csv(path)
