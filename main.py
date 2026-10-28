"""
Main Execution Entry Point for Supermarket Sales Analysis & Prediction System.
Runs the complete end-to-end data science and machine learning pipeline:
1. Dataset Acquisition & Verification
2. Data Loading & Cleaning
3. Exploratory Data Analysis & Visualization Generation
4. Supervised Regression Machine Learning (Scenario A & B)
5. Time-Series Future Sales Forecasting
6. Model Evaluation, Benchmark Diagnostics & Reporting
7. Business Insights & Operational Recommendations
8. SQLite Database Migration & Analytical Queries
"""

import sys
import time
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.download_data import download_from_kaggle, verify_dataset_schema
from src.data_cleaning import clean_and_prepare_data
from src.eda import run_exploratory_analysis
from src.model_training import train_supermarket_models
from src.forecasting import run_future_sales_forecasting
from src.evaluation import run_evaluation_suite
from src.business_insights import run_business_insights
from src.database import setup_and_seed_database


def run_full_pipeline(
    run_download: bool = True,
    run_clean: bool = True,
    run_eda_step: bool = True,
    run_train: bool = True,
    run_forecast: bool = True,
    run_eval: bool = True,
    run_insights_step: bool = True,
    run_db: bool = True,
):
    """
    Executes each pipeline step sequentially and tracks elapsed runtime.
    """
    total_start_time = time.time()
    print("=" * 80)
    print("  SUPERMARKET SALES ANALYSIS & MACHINE LEARNING PREDICTION SYSTEM  ")
    print("=" * 80)

    # 1. Dataset Acquisition
    if run_download:
        print("\n>>> STEP 1: Dataset Acquisition & Verification")
        try:
            raw_csv = download_from_kaggle()
            verify_dataset_schema(raw_csv)
        except Exception as e:
            print(f"[NOTE] Automated download step note: {e}")
            from src.config import RAW_DATA_FILE
            if RAW_DATA_FILE.exists():
                print(f"[INFO] Proceeding with existing dataset at: {RAW_DATA_FILE}")
            else:
                raise

    # 2. Data Cleaning & Validation
    if run_clean:
        print("\n>>> STEP 2: Data Cleaning, Parsing & Validation")
        cleaned_df, daily_df = clean_and_prepare_data()

    # 3. Exploratory Data Analysis
    if run_eda_step:
        print("\n>>> STEP 3: Exploratory Data Analysis & Chart Generation")
        run_exploratory_analysis()

    # 4. Supervised Regression Training (Scenario A & B)
    if run_train:
        print("\n>>> STEP 4: Transaction-Level Machine Learning (Scenario A & B)")
        train_supermarket_models()

    # 5. Future Sales Forecasting
    if run_forecast:
        print("\n>>> STEP 5: Aggregate Daily Future Sales Forecasting")
        run_future_sales_forecasting()

    # 6. Evaluation, Diagnostics & Professor Benchmark Report
    if run_eval:
        print("\n>>> STEP 6: Cross-Algorithm Model Evaluation & Diagnostics")
        run_evaluation_suite()

    # 7. Business Insights & Recommendations
    if run_insights_step:
        print("\n>>> STEP 7: Empirical Business Insights & Strategy")
        run_business_insights()

    # 8. SQLite Database Migration
    if run_db:
        print("\n>>> STEP 8: SQLite Database Migration & Analytical Queries")
        setup_and_seed_database()

    elapsed = round(time.time() - total_start_time, 2)
    print("\n" + "=" * 80)
    print(f" [SUCCESS] Complete Pipeline Finished Successfully in {elapsed} seconds!")
    print(" Outputs Generated:")
    print("   - Figures:      outputs/figures/ (13 publication-grade plots)")
    print("   - Reports:      outputs/reports/ (Evaluation, Comparisons & Insights)")
    print("   - Models:       models/ (Trained Joblib pipelines)")
    print("   - Predictions:  outputs/predictions/ (7-Day Daily Forecast CSV)")
    print("   - Database:     supermarket_sales.db (SQLite database)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Run Supermarket Sales ML & Analytics Pipeline")
    parser.add_argument("--all", action="store_true", default=True, help="Run complete end-to-end pipeline")
    parser.add_argument("--clean-only", action="store_true", help="Run only cleaning step")
    parser.add_argument("--eda-only", action="store_true", help="Run only EDA step")
    parser.add_argument("--train-only", action="store_true", help="Run only model training")
    parser.add_argument("--forecast-only", action="store_true", help="Run only forecasting")
    parser.add_argument("--db-only", action="store_true", help="Run only database setup")

    args = parser.parse_args()

    if args.clean_only:
        clean_and_prepare_data()
    elif args.eda_only:
        run_exploratory_analysis()
    elif args.train_only:
        train_supermarket_models()
    elif args.forecast_only:
        run_future_sales_forecasting()
    elif args.db_only:
        setup_and_seed_database()
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
