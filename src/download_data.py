"""
Dataset Acquisition and Verification Module.
Handles verification of local Kaggle supermarket dataset, schema validation,
and graceful offline fallback.
"""

from pathlib import Path
import pandas as pd

from src.config import (
    RAW_DATA_DIR,
    RAW_DATA_FILE,
    EXPECTED_RAW_COLUMNS,
)


def verify_dataset_schema(filepath: Path) -> bool:
    """
    Checks that the dataset file exists, is non-empty, and includes expected columns.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file does not exist at: {filepath}")

    df = pd.read_csv(filepath, nrows=5)
    missing = [c for c in EXPECTED_RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset at {filepath} is missing expected columns: {missing}")

    print(f"[SUCCESS] Verified dataset schema at: {filepath} ({len(df.columns)} columns present)")
    return True


def download_from_kaggle(destination_dir: Path = RAW_DATA_DIR) -> Path:
    """
    Verifies that supermarket_sales.csv is available.
    If the file exists locally, verifies it. If not, attempts to download via Kaggle API.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    target_file = destination_dir / "supermarket_sales.csv"

    if target_file.exists():
        print(f"[INFO] Dataset already present locally at: {target_file}")
        verify_dataset_schema(target_file)
        return target_file

    # Attempt Kaggle download if credentials exist
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("[INFO] Kaggle API authenticated. Downloading dataset 'aungpyaeap/supermarket-sales'...")
        api.dataset_download_files("aungpyaeap/supermarket-sales", path=str(destination_dir), unzip=True)
        print(f"[SUCCESS] Downloaded and extracted dataset to: {destination_dir}")
        verify_dataset_schema(target_file)
        return target_file
    except Exception as e:
        if target_file.exists():
            return target_file
        raise RuntimeError(
            f"Kaggle download failed ({e}). Please ensure '{target_file.name}' is placed in '{destination_dir}'."
        ) from e


if __name__ == "__main__":
    download_from_kaggle()

