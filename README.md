# Supermarket Sales Prediction System Using Machine Learning

A complete, production-style, enterprise-grade academic data science and machine learning system built with **Python**, **Scikit-learn**, **Pandas**, **Plotly**, and **Streamlit**.

The platform is designed to answer critical retail analytics and machine learning questions:
1. **Transaction Value Estimation**: How supervised regressors approximate multiplicative scan-level pricing surfaces ($Total = 1.05 \times Unit\;Price \times Quantity$) compared to deterministic accounting rules.
2. **Context-Only Predictability**: Why pre-scan customer demographics (gender, loyalty tier) cannot reliably predict transaction totals before items are scanned ($R^2 \approx 0.0$).
3. **Aggregate Daily Sales Forecasting**: How time-series models (Autoregressive Ridge, Random Forest, Gradient Boosting, Holt-Winters Exponential Smoothing) forecast forward daily store revenue over 1–14 day horizons using chronological out-of-sample validation.
4. **Target Leakage Auditing**: A formal mathematical proof demonstrating why beginner Kaggle notebooks report an artificial $R^2 = 1.000$ when including derived accounting identity fields (`cogs`, `Tax 5%`, `gross income`).

---

## 🎨 Enterprise Design System & UI Specifications

The web dashboard is built as a **clean, modern, white enterprise analytics platform** adhering strictly to the required controlled color system:

| UI Element | Color Hex Code | Usage |
| :--- | :--- | :--- |
| **Main Background** | `#F7F9FC` | Global viewport background |
| **Primary Surfaces & Cards** | `#FFFFFF` | Card containers, charts, input panels |
| **Primary Typography** | `#172033` | Page titles, KPI numbers, primary labels |
| **Secondary Typography** | `#64748B` | Subheadings, metadata, axis labels |
| **Primary Accent** | `#2563EB` | Active buttons, primary series, brand marks |
| **Accent Hover** | `#1D4ED8` | Button hover and interactive focus |
| **Borders & Dividers** | `#E2E8F0` | Subtle card borders, data table gridlines |
| **Positive Status** | `#15803D` | Validated models, positive trends |
| **Warning Status** | `#B45309` | Limitations alerts, seasonal caveats |
| **Error Status** | `#B91C1C` | Target leakage indicators, malformed data |

- **Typography**: Modern, highly readable Inter typeface, paired with JetBrains Mono tabular figures for alignment.
- **Charts**: Coordinated Plotly charts utilizing restrained enterprise blues, slate, and teal without excessive gradients, neon colors, or decorative glassmorphism.
- **Responsive Layout**: Designed for seamless scaling across Desktop ($1440\text{px}$), Laptop ($1024\text{px}$), Tablet ($768\text{px}$), and Mobile ($360\text{px} - 390\text{px}$) viewports.

---

## 💵 Currency Correctness & Financial Integrity Rules

- **Native Unit Preservation**: In strict accordance with the project guidelines, numerical amounts are preserved in their original native dataset units (`$ / Currency Units`).
- **No Forced Currency Multiplications**: Amounts are **not** multiplied by arbitrary conversion rates (e.g. 83.50 to INR).
- **Exact Financial Formula Audit**:
  $$\text{cogs} = \text{Unit Price} \times \text{Quantity}$$
  $$\text{Tax 5\%} = 0.05 \times \text{cogs} = 0.05 \times (\text{Unit Price} \times \text{Quantity})$$
  $$\text{Total} = \text{cogs} + \text{Tax 5\%} = 1.05 \times \text{Unit Price} \times \text{Quantity}$$
  $$\text{gross margin percentage} = 4.761905\% \quad \left(\frac{0.05}{1.05} \times 100\right)$$

---

## 🏗️ System Architecture & Directory Structure

```
supermarket_sales_prediction/
├── app.py                             # Interactive Streamlit enterprise dashboard (6 sections)
├── main.py                            # Standalone end-to-end execution pipeline (CLI)
├── requirements.txt                   # Production Python package dependencies
├── environment.yml                    # Conda environment definition
├── README.md                          # Comprehensive technical documentation
├── supermarket_sales.db               # SQLite database with relational schemas
├── data/
│   ├── raw/
│   │   └── supermarket_sales.csv      # Raw Kaggle supermarket sales dataset (1,000 records)
│   └── processed/
│       ├── cleaned_supermarket_sales.csv     # Cleaned, parsed, and validated transactions
│       ├── cleaned_supermarket_sales.parquet # High-performance columnar parquet dataset
│       └── daily_aggregated_sales.csv       # Contiguous 89-day aggregate daily sales
├── src/
│   ├── __init__.py
│   ├── config.py                      # Global paths, color palette, feature definitions, seeds
│   ├── download_data.py               # Kaggle acquisition handler & schema verification
│   ├── data_loader.py                 # Multi-format data loader (CSV/Excel/SQLite/Buffer)
│   ├── data_cleaning.py               # Parsing, temporal binning, financial consistency audit
│   ├── feature_engineering.py         # Leakage auditor, Scikit-learn pipelines, causal lag features
│   ├── model_training.py              # Candidate model training, benchmarks, joblib artifact saving
│   ├── forecasting.py                 # Dynamic 1–14 day aggregate time-series forecasting engine
│   ├── evaluation.py                  # Evaluation suite, diagnostic charts, leakage proof, executive report
│   ├── business_insights.py           # Empirical descriptive insights engine with causal disclaimers
│   ├── eda.py                         # Statistical summaries & 8 publication-grade visualization plots
│   └── database.py                    # Safe parameterized SQLite operations & seeded tables
├── models/
│   ├── scenario_a_best_model.joblib   # Persisted Extra Trees transaction estimation pipeline
│   ├── scenario_a_rf_model.joblib     # Persisted Random Forest estimation pipeline
│   ├── scenario_a_ridge_model.joblib  # Persisted Ridge Regression estimation pipeline
│   └── scenario_b_best_model.joblib   # Persisted context-only baseline pipeline
├── outputs/
│   ├── figures/                       # Publication-grade PNG charts (13 figures)
│   ├── reports/                       # Executive Markdown and CSV benchmark reports
│   └── predictions/                   # Multi-day forward forecast CSV projections
└── tests/
    ├── __init__.py
    ├── test_data_loader.py            # Data loading & schema validation tests
    ├── test_data_cleaning.py          # Temporal parsing & financial consistency tests
    ├── test_leakage_and_financials.py # Leakage detection & target isolation tests
    ├── test_models.py                 # Scikit-learn pipeline & metrics tests
    ├── test_forecasting.py            # Chronological splitting & 1-14 day forecast tests
    ├── test_database.py               # SQLite migration & parameterized query tests
    └── test_insights.py               # Dynamic insights & empty-state tests
```

---

## 🔬 Machine Learning Methodology

### 1. Scenario A: Transaction Value Estimation
- **Input Features**: `Unit price`, `Quantity`, `Branch`, `City`, `Customer type`, `Gender`, `Product line`, `Payment`.
- **Target**: `Total`.
- **Key Scientific Finding**: Because $Total = 1.05 \times Unit\;price \times Quantity$ is a multiplicative interaction, standard Linear Regression without interaction terms underfits ($R^2 \approx 0.8988$, $RMSE = \$80.46$).
- Non-linear tree ensembles (**Extra Trees Regressor** and **Gradient Boosting**) partition the numerical space to closely approximate the multiplicative bilinear surface, achieving validation **$R^2 > 0.999$** and **$RMSE < \$5.00$**.
- **Academic Note**: In the dashboard, we clearly explain that this represents transaction calculation / valuation rather than future forecasting.

### 2. Scenario B: Pre-Scan Context Prediction (Negative Result)
- **Input Features**: `Branch`, `Customer type`, `Gender`, `Product line`, `Payment`, `Hour`, `DayOfWeek`, `Month`, `Rating`.
- **Target**: `Total` (without scanning item quantity or price).
- **Key Scientific Finding**: Customer demographic and time features alone achieve validation **$R^2 \approx 0.0$**. This provides an honest, empirical negative result: supermarkets cannot reliably predict an individual shopper's receipt spend before items are scanned.

### 3. Aggregate Daily Sales Forecasting (1–14 Day Horizon)
- **Dataset Transformation**: 1,000 transactions are aggregated into 89 contiguous daily totals ($Daily\_Revenue = \sum Total$).
- **Causal Feature Engineering**:
  - Autoregressive lags: $Lag_1, Lag_2, Lag_3, Lag_7, Lag_{14}$.
  - Rolling statistics strictly from past periods: $Rolling\_Mean_7, Rolling\_Std_7$ (computed via `shift(1)` to eliminate look-ahead bias).
  - Calendar attributes: $DayOfWeek, DayOfMonth, IsWeekend$.
- **Validation Methodology**: Strict chronological out-of-sample train/test splitting (no random shuffling).
- **Candidate Models**:
  - Naive Persistence (Last observed daily value)
  - 7-Day Moving Average Baseline
  - Seasonal Naive (7-day seasonal lag)
  - Autoregressive Ridge Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Holt-Winters Exponential Smoothing (additive trend + 7-day seasonality)
- **Recursive Multi-Step Projections**: Provides future point forecasts and 95% confidence intervals across user-selected 1–14 day horizons.

### 4. Mathematical Proof of Target Leakage
Many naive Kaggle notebooks report an artificial $R^2 = 1.000$ by including derived columns:
- $cogs = Unit\;price \times Quantity = \frac{Total}{1.05}$
- $Tax\;5\% = 0.05 \times cogs = Total \times \frac{5}{105}$
- $gross\;income = Tax\;5\%$

Our system includes an automated **Target Leakage Auditor** and a diagnostic experiment (`outputs/figures/13_leakage_demonstration.png`) proving that models trained on these variables learn a trivial identity equation ($y = 1.05 x$) rather than actual retail behavior.

---

## 📊 Application Navigation & Modules

The Streamlit web interface (`app.py`) provides 6 structured modules:

1. **Overview Dashboard**:
   - Executive KPIs (Total Revenue, Transactions, Average Ticket, Units Sold, Customer Rating, Top Branch).
   - Global reactive filters (Branches, Categories, Date Range) with immediate recalculation.
   - Interactive Plotly visualizations (Revenue Velocity with 7-Day Moving Average, Department Share, Branch Comparisons, Payment Distribution).
2. **Supermarket Sales Prediction System**:
   - Primary predictive application page.
   - **Mode 1**: Transaction Value Estimation with side-by-side comparison of mathematical ground truth vs. Extra Trees ML estimate.
   - **Mode 2**: Multi-Day Future Sales Forecasting with dynamic 1–14 day slider, model selector, interactive confidence bands, projected revenue table, and CSV export.
3. **Model Evaluation**:
   - Benchmark comparison tables across classical and modern algorithms.
   - Actual vs. Predicted scatter plot with $y = x$ reference line.
   - Residual diagnostics and error distribution.
   - Target leakage mathematical demonstration.
4. **Exploratory Data Analysis (EDA)**:
   - Spend distributions, quantity histograms, customer rating boxplots, hourly traffic velocity, and day-of-week performance.
5. **Business Insights**:
   - Empirical findings calculated dynamically from the filtered dataset.
   - Clear distinction between descriptive associations and causal claims.
6. **Data & Model Information**:
   - Dataset dimensions, column schemas, data types, missing value summaries, duplicate checks.
   - Interactive CSV/Excel dataset upload with defensive schema validation.
   - Safe, parameterized SQLite database viewer with pre-defined analytical queries (arbitrary user SQL disabled for security).

---

## ⚡ Quickstart & Installation

### Option 1: Using Pip & Virtual Environment

1. **Clone or Extract the Project**:
   ```bash
   cd supermarket_sales_prediction
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using Conda

```bash
conda env create -f environment.yml
conda activate supermarket_sales_env
```

---

## 🚀 Execution Instructions

### 1. Execute Complete End-to-End CLI Pipeline
To run the full data acquisition, cleaning, exploratory analysis, model training, time-series forecasting, evaluation report generation, and SQLite database seeding:

```bash
python main.py
```

*CLI Options Supported*:
- `python main.py --all` : Complete end-to-end execution (default)
- `python main.py --clean-only` : Run data cleaning and preparation
- `python main.py --eda-only` : Generate all 8 exploratory figures
- `python main.py --train-only` : Train and persist machine learning models
- `python main.py --forecast-only` : Run aggregate time-series forecasting
- `python main.py --db-only` : Seed SQLite database tables

### 2. Launch the Interactive Enterprise Dashboard
To launch the Streamlit web application:

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Run Automated Quality Assurance Test Suite
To execute all 26 automated unit and integration tests:

```bash
pytest -v
```

---

## 🧪 Automated Test Suite Summary

The test suite in `tests/` covers:
- `test_data_loader.py`: CSV loading, file discovery, valid schema checks, malformed data rejection (`DataValidationError`).
- `test_data_cleaning.py`: Temporal parsing, financial equation verification ($Total = 1.05 \times Price \times Qty$), calendar continuity.
- `test_leakage_and_financials.py`: Target leakage detection, column isolation for Scenario A and Scenario B.
- `test_models.py`: Regression metric computations, train/val splits, joblib artifact inference.
- `test_forecasting.py`: Causal lag engineering, chronological splitting, 1–14 day horizon support, 95% confidence intervals.
- `test_database.py`: SQLite table lifecycle, data migrations, parameterized analytical queries.
- `test_insights.py`: Dynamic insight extraction, empty dataset handling.

---

## ⚠️ Transparent Limitations & Disclosures

1. **Dataset Coverage & Seasonality**: The Kaggle supermarket dataset spans **89 contiguous calendar days** (January 1, 2019 to March 30, 2019). While robust for 1–14 day operational planning, it cannot extrapolate annual holiday seasonality (e.g., Black Friday, Christmas, summer peaks) without multi-year records.
2. **Deterministic vs. ML Estimation**: In Mode 1, machine learning regression approximates a known mathematical formula ($1.05 \times P \times Q$). In commercial POS systems, direct arithmetic is preferred for final billing accuracy, while ML benchmarking highlights algorithm characteristics.
3. **Statistical Uncertainty**: Time-series forecast bands represent 95% empirical prediction intervals; they serve as decision-support guides rather than guaranteed financial performance.
