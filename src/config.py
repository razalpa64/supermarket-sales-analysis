"""
Configuration Module for Supermarket Sales Analysis and ML Prediction System.
Defines portable paths, enterprise design color constants, random seeds, and feature sets.
"""

from pathlib import Path

# Base Paths (Portable using pathlib)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Primary Dataset Configuration
RAW_DATA_FILE = RAW_DATA_DIR / "supermarket_sales.csv"
CLEANED_DATA_FILE = PROCESSED_DATA_DIR / "cleaned_supermarket_sales.parquet"
CLEANED_DATA_CSV = PROCESSED_DATA_DIR / "cleaned_supermarket_sales.csv"
DAILY_SALES_FILE = PROCESSED_DATA_DIR / "daily_aggregated_sales.csv"

# Models and Outputs
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# Database Configuration
DATABASE_PATH = BASE_DIR / "supermarket_sales.db"

# Ensure runtime directories exist
for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    PREDICTIONS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# ENTERPRISE DESIGN SYSTEM PALETTE (Clean White Theme)
# -----------------------------------------------------------------------------
COLOR_MAIN_BG = "#F7F9FC"
COLOR_CARD_SURFACE = "#FFFFFF"
COLOR_PRIMARY_TEXT = "#172033"
COLOR_SECONDARY_TEXT = "#64748B"
COLOR_PRIMARY_ACCENT = "#2563EB"
COLOR_ACCENT_HOVER = "#1D4ED8"
COLOR_BORDER = "#E2E8F0"
COLOR_POSITIVE = "#15803D"
COLOR_WARNING = "#B45309"
COLOR_ERROR = "#B91C1C"

# Restrained chart palette for Plotly / Matplotlib
CHART_COLOR_PALETTE = [
    "#2563EB",  # Primary Blue
    "#0EA5E9",  # Sky Blue
    "#6366F1",  # Indigo
    "#3B82F6",  # Light Blue
    "#64748B",  # Slate
    "#0D9488",  # Teal
    "#F59E0B",  # Amber
    "#10B981",  # Emerald
]

# -----------------------------------------------------------------------------
# DATASET AND FINANCIAL RULES
# -----------------------------------------------------------------------------
# Default currency representation: Preserve original dataset native units.
# The source dataset uses standard numerical currency amounts (commonly interpreted as USD in Kaggle benchmarks).
CURRENCY_LABEL = "Currency Units ($)"
CURRENCY_SYMBOL = "$"

# Target variable
TARGET_COLUMN = "Total"
IDENTIFIER_COLUMNS = ["Invoice ID"]

# Expected Raw Dataset Columns
EXPECTED_RAW_COLUMNS = [
    "Invoice ID",
    "Branch",
    "City",
    "Customer type",
    "Gender",
    "Product line",
    "Unit price",
    "Quantity",
    "Tax 5%",
    "Total",
    "Date",
    "Time",
    "Payment",
    "cogs",
    "gross margin percentage",
    "gross income",
    "Rating",
]

# LEAKAGE COLUMNS:
# cogs = Unit price * Quantity = Total / 1.05
# Tax 5% = 0.05 * cogs = Total * (5/105)
# gross income = Tax 5%
# gross margin percentage = 4.761905% (constant)
LEAKAGE_COLUMNS = [
    "Tax 5%",
    "cogs",
    "gross income",
    "gross margin percentage",
]

# Scenario A: Estimation features (Unit price & Quantity used to estimate/calculate transaction value)
SCENARIO_A_NUMERIC_FEATURES = ["Unit price", "Quantity"]
SCENARIO_A_CATEGORICAL_FEATURES = ["Branch", "City", "Customer type", "Gender", "Product line", "Payment"]

# Scenario B: Context-only features (predict transaction value WITHOUT item-level scan)
SCENARIO_B_CATEGORICAL_FEATURES = ["Branch", "Customer type", "Gender", "Product line", "Payment"]
SCENARIO_B_NUMERIC_FEATURES = ["Hour", "DayOfWeek", "Month", "Rating"]

# Reproducibility & Splits
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.15

# Time Series Settings
DEFAULT_FORECAST_HORIZON = 7
MIN_FORECAST_HORIZON = 1
MAX_FORECAST_HORIZON = 14
TIME_SERIES_DATE_COLUMN = "Date"
