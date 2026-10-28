"""
Supermarket Sales Prediction System Using Machine Learning.
Production-style Enterprise Analytics and Machine Learning Web Application.
Built with Streamlit, Plotly, Scikit-learn, and Pandas.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import io
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CLEANED_DATA_CSV,
    DAILY_SALES_FILE,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    PREDICTIONS_DIR,
    DATABASE_PATH,
    COLOR_MAIN_BG,
    COLOR_CARD_SURFACE,
    COLOR_PRIMARY_TEXT,
    COLOR_SECONDARY_TEXT,
    COLOR_PRIMARY_ACCENT,
    COLOR_ACCENT_HOVER,
    COLOR_BORDER,
    COLOR_POSITIVE,
    COLOR_WARNING,
    COLOR_ERROR,
    CHART_COLOR_PALETTE,
    CURRENCY_SYMBOL,
    CURRENCY_LABEL,
    EXPECTED_RAW_COLUMNS,
    RANDOM_STATE,
)
from src.data_loader import SupermarketDataLoader, DataValidationError
from src.data_cleaning import clean_and_prepare_data
from src.forecasting import SupermarketSalesForecaster
from src.business_insights import SupermarketInsightsEngine
from src.database import SupermarketDatabase

# -----------------------------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Supermarket Sales Prediction System",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# ENTERPRISE DESIGN SYSTEM (CLEAN WHITE THEME CSS)
# -----------------------------------------------------------------------------
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Background and Typography */
    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: {COLOR_MAIN_BG} !important;
        color: {COLOR_PRIMARY_TEXT} !important;
    }}

    /* Top Streamlit Header Overrides (Remove Dark Bar) */
    header[data-testid="stHeader"] {{
        background-color: #FFFFFF !important;
        border-bottom: 1px solid {COLOR_BORDER} !important;
    }}
    header[data-testid="stHeader"] * {{
        color: {COLOR_PRIMARY_TEXT} !important;
    }}
    div[data-testid="stDecoration"] {{
        display: none !important;
    }}

    /* Main Container Padding */
    .block-container {{
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1380px !important;
    }}

    /* Clean Enterprise Card */
    .ent-card {{
        background-color: {COLOR_CARD_SURFACE};
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }}

    /* KPI Metric Tile */
    .kpi-tile {{
        background-color: #FFFFFF;
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        padding: 1.05rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        height: 100%;
        min-height: 105px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .kpi-label {{
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {COLOR_SECONDARY_TEXT};
        margin-bottom: 0.25rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .kpi-value {{
        font-size: 1.55rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: {COLOR_PRIMARY_TEXT};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        line-height: 1.2;
    }}
    .kpi-sub {{
        font-size: 0.75rem;
        color: {COLOR_SECONDARY_TEXT};
        font-weight: 500;
        margin-top: 0.35rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    /* Page Header */
    .page-tag {{
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {COLOR_PRIMARY_ACCENT};
        background-color: #EFF6FF;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        border: 1px solid #DBEAFE;
        margin-bottom: 0.4rem;
    }}
    .page-title {{
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        color: {COLOR_PRIMARY_TEXT};
        margin-bottom: 0.25rem;
    }}
    .page-subtitle {{
        font-size: 0.92rem;
        color: {COLOR_SECONDARY_TEXT};
        margin-bottom: 1.4rem;
        line-height: 1.5;
    }}

    /* Custom Button Overrides */
    div.stButton > button:first-child {{
        background-color: {COLOR_PRIMARY_ACCENT} !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border: 1px solid {COLOR_PRIMARY_ACCENT} !important;
        border-radius: 6px !important;
        padding: 0.55rem 1.25rem !important;
        box-shadow: 0 1px 2px rgba(37, 99, 235, 0.15) !important;
        transition: all 0.15s ease !important;
    }}
    div.stButton > button:first-child:hover {{
        background-color: {COLOR_ACCENT_HOVER} !important;
        border-color: {COLOR_ACCENT_HOVER} !important;
        color: #FFFFFF !important;
    }}

    /* Sidebar Clean Styling */
    section[data-testid="stSidebar"] {{
        background-color: #FFFFFF !important;
        border-right: 1px solid {COLOR_BORDER} !important;
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {{
        color: {COLOR_PRIMARY_TEXT};
    }}
    section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] span {{
        color: #334155 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        color: {COLOR_PRIMARY_TEXT} !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }}
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"] {{
        color: {COLOR_PRIMARY_ACCENT} !important;
        font-weight: 600 !important;
    }}

    /* Input Controls, Multiselect & Selectbox */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    .stTextInput input,
    .stNumberInput input,
    .stDateInput input {{
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 6px !important;
        color: {COLOR_PRIMARY_TEXT} !important;
    }}
    div[data-baseweb="select"] svg {{
        fill: #64748B !important;
    }}

    /* Multiselect Tag Chips (Soft Enterprise Blue Pills) */
    span[data-baseweb="tag"] {{
        background-color: #EFF6FF !important;
        border: 1px solid #BFDBFE !important;
        border-radius: 4px !important;
        padding: 0.1rem 0.35rem !important;
    }}
    span[data-baseweb="tag"] span {{
        color: #1D4ED8 !important;
        font-weight: 500 !important;
        font-size: 0.78rem !important;
    }}
    span[data-baseweb="tag"] svg {{
        fill: #1D4ED8 !important;
    }}

    /* Badge indicators */
    .badge-pill {{
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .badge-positive {{
        background-color: #DCFCE7;
        color: {COLOR_POSITIVE};
        border: 1px solid #BBF7D0;
    }}
    .badge-warning {{
        background-color: #FEF3C7;
        color: {COLOR_WARNING};
        border: 1px solid #FDE68A;
    }}
    .badge-error {{
        background-color: #FEE2E2;
        color: {COLOR_ERROR};
        border: 1px solid #FECACA;
    }}
    .badge-neutral {{
        background-color: #F1F5F9;
        color: {COLOR_SECONDARY_TEXT};
        border: 1px solid {COLOR_BORDER};
    }}

    /* Disclaimers & Alert Boxes */
    .alert-box {{
        background-color: #F8FAFC;
        border-left: 4px solid {COLOR_PRIMARY_ACCENT};
        padding: 0.85rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.84rem;
        color: {COLOR_SECONDARY_TEXT};
        line-height: 1.5;
        margin-bottom: 1rem;
    }}
    .alert-warning-box {{
        background-color: #FFFBEB;
        border-left: 4px solid {COLOR_WARNING};
        padding: 0.85rem 1rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.84rem;
        color: #92400E;
        line-height: 1.5;
        margin-bottom: 1rem;
    }}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# DATA CACHING & INITIALIZATION
# -----------------------------------------------------------------------------
@st.cache_data
def load_application_data():
    """
    Loads and caches transaction dataset and daily aggregated time-series.
    """
    if not CLEANED_DATA_CSV.exists() or not DAILY_SALES_FILE.exists():
        clean_df, daily_df = clean_and_prepare_data()
    else:
        clean_df = pd.read_csv(CLEANED_DATA_CSV)
        clean_df["Date"] = pd.to_datetime(clean_df["Date"])
        daily_df = pd.read_csv(DAILY_SALES_FILE)
        daily_df["Date"] = pd.to_datetime(daily_df["Date"])
    return clean_df, daily_df


@st.cache_resource
def load_trained_pipelines():
    """
    Loads persisted Scikit-learn models.
    """
    model_a_file = MODELS_DIR / "scenario_a_best_model.joblib"
    model_b_file = MODELS_DIR / "scenario_b_best_model.joblib"

    model_a = joblib.load(model_a_file) if model_a_file.exists() else None
    model_b = joblib.load(model_b_file) if model_b_file.exists() else None
    return model_a, model_b


# Load primary datasets
df_raw, daily_raw = load_application_data()
model_scenario_a, model_scenario_b = load_trained_pipelines()


# -----------------------------------------------------------------------------
# PLOTLY ENTERPRISE THEME HELPER
# -----------------------------------------------------------------------------
def format_plotly_figure(
    fig: go.Figure,
    title: str = "",
    height: int = 370,
    show_legend: bool = True,
) -> go.Figure:
    """
    Applies the consistent white enterprise theme to any Plotly figure,
    ensuring zero collision between title and legend.
    """
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family="Inter, -apple-system, sans-serif", size=13, color=COLOR_PRIMARY_TEXT),
            x=0.01,
            y=0.98,
        ),
        paper_bgcolor=COLOR_CARD_SURFACE,
        plot_bgcolor=COLOR_CARD_SURFACE,
        font=dict(family="Inter, -apple-system, sans-serif", size=11, color=COLOR_SECONDARY_TEXT),
        margin=dict(
            l=40,
            r=30,
            t=55 if title else 25,
            b=55 if show_legend else 30,
        ),
        height=height,
        xaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            tickfont=dict(size=10, color=COLOR_SECONDARY_TEXT),
            title_font=dict(size=11, color=COLOR_PRIMARY_TEXT),
            linecolor=COLOR_BORDER,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            zeroline=False,
            tickfont=dict(size=10, color=COLOR_SECONDARY_TEXT),
            title_font=dict(size=11, color=COLOR_PRIMARY_TEXT),
            linecolor=COLOR_BORDER,
        ),
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color=COLOR_PRIMARY_TEXT),
        ) if show_legend else None,
    )
    return fig


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & GLOBAL FILTERS
# -----------------------------------------------------------------------------
st.sidebar.markdown(f"""
<div style="padding: 0.2rem 0 1.0rem 0; border-bottom: 1px solid {COLOR_BORDER}; margin-bottom: 1.0rem;">
    <div style="font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; color: {COLOR_PRIMARY_ACCENT}; text-transform: uppercase;">
        Retail Analytics & ML
    </div>
    <div style="font-size: 1.15rem; font-weight: 700; color: {COLOR_PRIMARY_TEXT}; letter-spacing: -0.02em;">
        Supermarket Sales OS
    </div>
    <div style="font-size: 0.75rem; color: {COLOR_SECONDARY_TEXT}; margin-top: 0.15rem;">
        Enterprise Academic Platform
    </div>
</div>
""", unsafe_allow_html=True)

nav_section = st.sidebar.radio(
    "Navigation Menu",
    [
        "Overview Dashboard",
        "Supermarket Sales Prediction System",
        "Model Evaluation",
        "Exploratory Data Analysis",
        "Business Insights",
        "Data & Model Information",
    ],
    index=0,
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"<div style='font-size:0.78rem; font-weight:600; color:{COLOR_PRIMARY_TEXT}; margin-bottom:0.4rem;'>DATASET FILTERS</div>", unsafe_allow_html=True)

# Branch filter
available_branches = sorted(df_raw["Branch"].unique())
selected_branches = st.sidebar.multiselect(
    "Select Branches:",
    options=available_branches,
    default=available_branches,
    format_func=lambda x: f"Branch {x} ({'Yangon' if x=='A' else 'Mandalay' if x=='B' else 'Naypyitaw'})",
)

# Product line filter
available_categories = sorted(df_raw["Product line"].unique())
selected_categories = st.sidebar.multiselect(
    "Select Product Categories:",
    options=available_categories,
    default=available_categories,
)

# Date filter
min_dt = df_raw["Date"].min().date()
max_dt = df_raw["Date"].max().date()
selected_date_range = st.sidebar.date_input(
    "Select Date Range:",
    value=(min_dt, max_dt),
    min_value=min_dt,
    max_value=max_dt,
)

# Filter dataset based on sidebar controls
if isinstance(selected_date_range, (tuple, list)) and len(selected_date_range) == 2:
    start_d, end_d = selected_date_range
else:
    start_d, end_d = min_dt, max_dt

mask = (
    df_raw["Branch"].isin(selected_branches) &
    df_raw["Product line"].isin(selected_categories) &
    (df_raw["Date"].dt.date >= start_d) &
    (df_raw["Date"].dt.date <= end_d)
)
filtered_df = df_raw[mask].copy()

# Filtered daily dataset
daily_mask = (
    (daily_raw["Date"].dt.date >= start_d) &
    (daily_raw["Date"].dt.date <= end_d)
)
filtered_daily = daily_raw[daily_mask].copy()

# Sidebar footer
st.sidebar.markdown(f"""
<div style="margin-top: 2rem; padding: 0.8rem 0; border-top: 1px solid {COLOR_BORDER}; font-size: 0.72rem; color: {COLOR_SECONDARY_TEXT};">
    <div><b>Active Dataset</b>: {len(filtered_df):,} of {len(df_raw):,} records</div>
    <div><b>Currency</b>: Native Units ({CURRENCY_SYMBOL})</div>
    <div style="margin-top: 0.4rem; font-family: 'JetBrains Mono', monospace;">v2.5.0-Enterprise</div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# SECTION A: OVERVIEW DASHBOARD
# ==============================================================================
if nav_section == "Overview Dashboard":
    st.markdown('<div class="page-tag">Executive Summary</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Supermarket Sales Overview Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-subtitle">Real-time operational KPIs, revenue velocity, and merchandise performance calculated from the loaded dataset. Filtered period: <b>{start_d}</b> to <b>{end_d}</b>.</div>',
        unsafe_allow_html=True
    )

    if filtered_df.empty:
        st.warning("⚠️ No records match the active filter criteria. Please broaden your branch, category, or date selections in the sidebar.")
    else:
        # Genuine KPIs calculated from filtered dataset
        total_rev = float(filtered_df["Total"].sum())
        total_txns = int(len(filtered_df))
        avg_txn_val = float(filtered_df["Total"].mean()) if total_txns > 0 else 0.0
        total_units = int(filtered_df["Quantity"].sum()) if "Quantity" in filtered_df.columns else 0
        avg_rating = float(filtered_df["Rating"].mean()) if "Rating" in filtered_df.columns else 0.0

        # Best performing branch
        branch_rev = filtered_df.groupby("Branch")["Total"].sum()
        best_branch = branch_rev.idxmax() if not branch_rev.empty else "N/A"
        best_branch_rev = float(branch_rev.max()) if not branch_rev.empty else 0.0

        # Row 1: Primary Revenue & Order Metrics (3 Columns)
        k1, k2, k3 = st.columns(3)

        with k1:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Total Revenue</div>
                <div class="kpi-value">{CURRENCY_SYMBOL}{total_rev:,.2f}</div>
                <div class="kpi-sub">Filtered Period Gross Turnover</div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            daily_avg_orders = total_txns / max(1, (end_d - start_d).days + 1)
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Total Transactions</div>
                <div class="kpi-value">{total_txns:,} Orders</div>
                <div class="kpi-sub">{daily_avg_orders:.1f} Orders / Day Average</div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Average Basket Value</div>
                <div class="kpi-value">{CURRENCY_SYMBOL}{avg_txn_val:,.2f}</div>
                <div class="kpi-sub">Average Revenue Per Order</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

        # Row 2: Units Sold, Customer Survey & Store Ranking (3 Columns)
        k4, k5, k6 = st.columns(3)

        with k4:
            units_per_order = total_units / max(1, total_txns)
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Total Units Sold</div>
                <div class="kpi-value">{total_units:,} Units</div>
                <div class="kpi-sub">{units_per_order:.1f} Units Per Basket Average</div>
            </div>
            """, unsafe_allow_html=True)

        with k5:
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Customer Satisfaction Rating</div>
                <div class="kpi-value">{avg_rating:.2f} <span style="font-size:0.92rem; color:{COLOR_SECONDARY_TEXT}; font-weight:500;">/ 10</span></div>
                <div class="kpi-sub">Empirical Survey Average</div>
            </div>
            """, unsafe_allow_html=True)

        with k6:
            branch_city_name = 'Yangon' if best_branch=='A' else ('Mandalay' if best_branch=='B' else 'Naypyitaw')
            branch_share = (best_branch_rev / max(1, total_rev)) * 100
            st.markdown(f"""
            <div class="kpi-tile">
                <div class="kpi-label">Leading Store Branch</div>
                <div class="kpi-value">Branch {best_branch}</div>
                <div class="kpi-sub">{branch_city_name} — {CURRENCY_SYMBOL}{best_branch_rev:,.0f} ({branch_share:.1f}% Share)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        # Row 2: Charts (Revenue Trend & Sales by Product Category)
        col_c1, col_c2 = st.columns([1.6, 1.4])

        with col_c1:
            # Daily revenue trend from filtered daily sales
            daily_filtered = filtered_df.groupby("Date")["Total"].sum().reset_index().sort_values("Date")
            daily_filtered["MA7"] = daily_filtered["Total"].rolling(7, min_periods=1).mean()

            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(
                x=daily_filtered["Date"],
                y=daily_filtered["Total"],
                mode="lines",
                name="Daily Sales",
                line=dict(color="#94A3B8", width=1.5),
            ))
            fig_trend.add_trace(go.Scatter(
                x=daily_filtered["Date"],
                y=daily_filtered["MA7"],
                mode="lines",
                name="7-Day Moving Avg",
                line=dict(color=COLOR_PRIMARY_ACCENT, width=2.5),
            ))
            format_plotly_figure(fig_trend, title=f"Daily Revenue Velocity with 7-Day Moving Average ({CURRENCY_SYMBOL})", height=360, show_legend=True)
            st.plotly_chart(fig_trend, use_container_width=True)

        with col_c2:
            prod_summary = (
                filtered_df.groupby("Product line")["Total"]
                .sum()
                .reset_index()
                .sort_values("Total", ascending=True)
            )
            fig_prod = px.bar(
                prod_summary,
                x="Total",
                y="Product line",
                orientation="h",
                color_discrete_sequence=[COLOR_PRIMARY_ACCENT],
                labels={"Total": f"Revenue ({CURRENCY_SYMBOL})", "Product line": ""},
            )
            format_plotly_figure(fig_prod, title=f"Revenue by Product Category ({CURRENCY_SYMBOL})", height=360, show_legend=False)
            st.plotly_chart(fig_prod, use_container_width=True)

        # Row 3: Branch Performance & Payment Distribution
        col_c3, col_c4 = st.columns(2)

        with col_c3:
            branch_city = filtered_df.groupby(["Branch", "City"])["Total"].sum().reset_index()
            fig_branch = px.bar(
                branch_city,
                x="Branch",
                y="Total",
                color="City",
                barmode="group",
                color_discrete_sequence=[COLOR_PRIMARY_ACCENT, "#0EA5E9", "#6366F1"],
                labels={"Total": f"Revenue ({CURRENCY_SYMBOL})"},
            )
            format_plotly_figure(fig_branch, title=f"Branch Revenue Comparison by City ({CURRENCY_SYMBOL})", height=340, show_legend=True)
            st.plotly_chart(fig_branch, use_container_width=True)

        with col_c4:
            pay_dist = filtered_df.groupby("Payment")["Total"].agg(["sum", "count"]).reset_index()
            fig_pay = px.pie(
                pay_dist,
                values="sum",
                names="Payment",
                hole=0.55,
                color_discrete_sequence=CHART_COLOR_PALETTE[:3],
            )
            fig_pay.update_traces(textposition="inside", textinfo="percent+label")
            format_plotly_figure(fig_pay, title="Revenue Share by Payment Method", height=340, show_legend=True)
            st.plotly_chart(fig_pay, use_container_width=True)

        # Empirical Highlights Card
        st.markdown(f"""
        <div class="ent-card" style="margin-top: 0.5rem;">
            <div style="font-size:0.85rem; font-weight:700; color:{COLOR_PRIMARY_TEXT}; margin-bottom:0.4rem;">
                📌 Executive Observations (Calculated From Filtered Data):
            </div>
            <ul style="font-size:0.84rem; color:{COLOR_SECONDARY_TEXT}; line-height:1.6; margin-bottom:0;">
                <li>The highest turnover department within current filters is <b>{prod_summary.iloc[-1]['Product line']}</b> generating <b>{CURRENCY_SYMBOL}{prod_summary.iloc[-1]['Total']:,.2f}</b>.</li>
                <li>Branch <b>{best_branch}</b> represents the leading revenue driver with <b>{CURRENCY_SYMBOL}{best_branch_rev:,.2f}</b>.</li>
                <li>Customer ratings across current selection average <b>{avg_rating:.2f} / 10</b> across <b>{total_txns:,}</b> transactions.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# SECTION B: SUPERMARKET SALES PREDICTION SYSTEM (PRIMARY PREDICTION FEATURE)
# ==============================================================================
elif nav_section == "Supermarket Sales Prediction System":
    st.markdown('<div class="page-tag">Core Predictive Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Supermarket Sales Prediction System Using Machine Learning</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-subtitle">Production-grade predictive workflows combining deterministic retail accounting rules with supervised machine learning and chronological aggregate time-series forecasting.</div>',
        unsafe_allow_html=True
    )

    pred_mode = st.radio(
        "Select Prediction Mode:",
        [
            "Mode 1: Transaction Value Estimation (Scan-Level Valuation)",
            "Mode 2: Future Sales Forecasting (Multi-Day Aggregate Time-Series)",
        ],
        index=0,
        horizontal=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MODE 1: TRANSACTION VALUE ESTIMATION
    # -------------------------------------------------------------------------
    if "Mode 1" in pred_mode:
        st.markdown(f"""
        <div class="alert-box">
            <b>Understanding Mode 1</b>: When unit price and quantity are known at checkout, total receipt revenue is mathematically determined:
            <br>
            <code>COGS = Unit Price × Quantity</code>, &nbsp;
            <code>Tax (5%) = 0.05 × COGS</code>, &nbsp;
            <code>Total = COGS + Tax = 1.05 × Unit Price × Quantity</code>.
            <br>
            <i>Academic Note</i>: This is deterministic transaction calculation/estimation rather than future forecasting. 
            Below, we compare the exact mathematical formula with our trained Scikit-learn regression models to benchmark how machine learning approximates the bilinear interaction surface.
        </div>
        """, unsafe_allow_html=True)

        col_input, col_output = st.columns([1.1, 1.4])

        with col_input:
            st.markdown(f"""
            <div class="ent-card">
                <div style="font-size:0.85rem; font-weight:700; color:{COLOR_PRIMARY_TEXT}; margin-bottom:0.8rem;">
                    1. Input Basket Parameters
                </div>
            """, unsafe_allow_html=True)

            in_unit_price = st.number_input(
                f"Unit Price ({CURRENCY_SYMBOL}):",
                min_value=1.0,
                max_value=200.0,
                value=55.0,
                step=1.0,
                help="Catalog unit price of the item",
            )
            in_qty = st.number_input(
                "Quantity Purchased:",
                min_value=1,
                max_value=50,
                value=5,
                step=1,
                help="Number of units in basket",
            )
            in_prod_line = st.selectbox(
                "Product Category:",
                sorted(df_raw["Product line"].unique()),
                index=0,
            )
            in_branch = st.selectbox(
                "Store Branch:",
                ["A", "B", "C"],
                format_func=lambda x: f"Branch {x} ({'Yangon' if x=='A' else 'Mandalay' if x=='B' else 'Naypyitaw'})",
                index=0,
            )
            in_city = "Yangon" if in_branch == "A" else ("Mandalay" if in_branch == "B" else "Naypyitaw")

            in_cust_type = st.selectbox("Customer Membership:", ["Member", "Normal"], index=0)
            in_gender = st.selectbox("Customer Gender:", ["Female", "Male"], index=0)
            in_payment = st.selectbox("Payment Channel:", ["Ewallet", "Cash", "Credit card"], index=0)

            run_estimation_btn = st.button("Calculate & Estimate Transaction Value", use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

        with col_output:
            # Deterministic Mathematical Calculation
            cogs_calc = in_unit_price * in_qty
            tax_calc = cogs_calc * 0.05
            math_total = cogs_calc + tax_calc

            # Machine Learning Estimation (Using Scikit-learn Pipeline)
            input_df = pd.DataFrame([{
                "Unit price": float(in_unit_price),
                "Quantity": int(in_qty),
                "Branch": in_branch,
                "City": in_city,
                "Customer type": in_cust_type,
                "Gender": in_gender,
                "Product line": in_prod_line,
                "Payment": in_payment,
            }])

            ml_pred = None
            if model_scenario_a is not None:
                try:
                    ml_pred = float(model_scenario_a.predict(input_df)[0])
                except Exception as e:
                    st.error(f"Inference error: {e}")

            st.markdown(f"""
            <div class="ent-card">
                <div style="font-size:0.85rem; font-weight:700; color:{COLOR_PRIMARY_TEXT}; margin-bottom:0.8rem;">
                    2. Valuation & Estimation Results
                </div>
            """, unsafe_allow_html=True)

            res1, res2 = st.columns(2)

            with res1:
                st.markdown(f"""
                <div style="padding: 1rem; background-color: #F8FAFC; border: 1px solid {COLOR_BORDER}; border-radius: 6px;">
                    <div style="font-size:0.72rem; font-weight:700; color:{COLOR_SECONDARY_TEXT}; text-transform:uppercase;">
                        Mathematical Ground Truth
                    </div>
                    <div style="font-size:1.8rem; font-weight:700; color:{COLOR_PRIMARY_TEXT}; margin:0.3rem 0;" class="num-mono">
                        {CURRENCY_SYMBOL}{math_total:,.2f}
                    </div>
                    <div style="font-size:0.75rem; color:{COLOR_SECONDARY_TEXT};">
                        Deterministic retail rule: <code>1.05 × P × Q</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with res2:
                if ml_pred is not None:
                    error_diff = abs(ml_pred - math_total)
                    st.markdown(f"""
                    <div style="padding: 1rem; background-color: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 6px;">
                        <div style="font-size:0.72rem; font-weight:700; color:{COLOR_PRIMARY_ACCENT}; text-transform:uppercase;">
                            ML Model Regressor Estimate
                        </div>
                        <div style="font-size:1.8rem; font-weight:700; color:{COLOR_PRIMARY_ACCENT}; margin:0.3rem 0;" class="num-mono">
                            {CURRENCY_SYMBOL}{ml_pred:,.2f}
                        </div>
                        <div style="font-size:0.75rem; color:{COLOR_SECONDARY_TEXT};">
                            Extra Trees Regressor (Residual: ±{CURRENCY_SYMBOL}{error_diff:.2f})
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Detailed Breakdown Table
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.82rem; font-weight:600; color:#172033; margin-bottom:0.4rem;'>FINANCIAL & ESTIMATION BREAKDOWN</div>", unsafe_allow_html=True)

            breakdown_data = [
                {"Item / Component": "Unit Price", "Value": f"{CURRENCY_SYMBOL}{in_unit_price:,.2f}", "Methodology": "Input Variable"},
                {"Item / Component": "Quantity Purchased", "Value": f"{in_qty} units", "Methodology": "Input Variable"},
                {"Item / Component": "Cost of Goods Sold (cogs)", "Value": f"{CURRENCY_SYMBOL}{cogs_calc:,.2f}", "Methodology": "Direct Arithmetic (P × Q)"},
                {"Item / Component": "Applicable Tax (5%)", "Value": f"{CURRENCY_SYMBOL}{tax_calc:,.2f}", "Methodology": "Statutory Rate (0.05 × cogs)"},
                {"Item / Component": "Exact Transaction Total", "Value": f"{CURRENCY_SYMBOL}{math_total:,.2f}", "Methodology": "Exact Accounting Formula"},
            ]
            if ml_pred is not None:
                breakdown_data.append({
                    "Item / Component": "Machine Learning Prediction",
                    "Value": f"{CURRENCY_SYMBOL}{ml_pred:,.2f}",
                    "Methodology": "Trained Scikit-learn Pipeline (Extra Trees)",
                })
                breakdown_data.append({
                    "Item / Component": "Estimation Discrepancy",
                    "Value": f"{CURRENCY_SYMBOL}{abs(ml_pred - math_total):,.4f} ({abs(ml_pred - math_total) / math_total * 100:.2f}%)",
                    "Methodology": "Approximation Residual |Actual - Predicted|",
                })

            st.dataframe(pd.DataFrame(breakdown_data), use_container_width=True, hide_index=True)

            st.markdown(f"""
            <div class="alert-box" style="margin-top:0.8rem;">
                <b>Critical Scientific Transparency</b>:
                Notice how the machine learning model produces a prediction very close to the true total, but with a small non-zero residual. 
                This demonstrates why in commercial supermarket environments, scanning known items at checkout uses deterministic arithmetic formulas. 
                Machine learning regression on transaction items serves as an algorithm benchmarking experiment, not a replacement for basic arithmetic.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # MODE 2: FUTURE SALES FORECASTING
    # -------------------------------------------------------------------------
    elif "Mode 2" in pred_mode:
        st.markdown(f"""
        <div class="alert-box">
            <b>Understanding Mode 2</b>: Store managers require aggregate forward-looking daily sales forecasts to schedule cashier staffing, logistics, and inventory orders.
            This module evaluates candidate forecasting models using <b>chronological out-of-sample validation</b> (no look-ahead leakage) and generates multi-day recursive projections with 95% confidence intervals.
        </div>
        """, unsafe_allow_html=True)

        fc_controls_c1, fc_controls_c2, fc_controls_c3 = st.columns([1, 1, 1])

        with fc_controls_c1:
            horizon_selected = st.slider(
                "Select Forecast Horizon (Days):",
                min_value=1,
                max_value=14,
                value=7,
                step=1,
                help="Forecast period into the unseen future (1 to 14 days supported)",
            )

        with fc_controls_c2:
            model_selected = st.selectbox(
                "Select Forecasting Algorithm:",
                [
                    "Autoregressive Ridge Regression",
                    "Random Forest Forecaster",
                    "Gradient Boosting Forecaster",
                    "7-Day Moving Average Baseline",
                    "Seasonal Naive (7-Day Lag)",
                    "Naive Baseline (Last Value)",
                ],
                index=0,
            )

        with fc_controls_c3:
            st.markdown(f"""
            <div style="padding-top: 1.6rem;">
                <span class="badge-pill badge-positive">Chronological Validation</span>
                <span class="badge-pill badge-neutral">89 Days History</span>
            </div>
            """, unsafe_allow_html=True)

        # Run time series forecaster
        try:
            forecaster = SupermarketSalesForecaster(daily_raw)
            fc_results = forecaster.run_benchmark_and_evaluation(test_horizon_days=horizon_selected)

            # Historical slice for plotting (last 35 days)
            hist_plot_slice = daily_raw.iloc[-35:].copy()

            # Future forecast dataframe
            future_df = fc_results["future_forecast_df"].copy()

            # Interactive Plotly Chart
            fig_fc_plotly = go.Figure()

            # 1. Historical Actuals
            fig_fc_plotly.add_trace(go.Scatter(
                x=hist_plot_slice["Date"],
                y=hist_plot_slice["Daily_Revenue"],
                mode="lines+markers",
                name="Historical Actuals",
                line=dict(color="#64748B", width=1.8),
                marker=dict(size=4, color="#64748B"),
            ))

            # 2. Out-of-sample test actuals vs prediction
            test_dates = [pd.to_datetime(d) for d in fc_results["test_dates"]]
            fig_fc_plotly.add_trace(go.Scatter(
                x=test_dates,
                y=fc_results["test_actuals"],
                mode="lines+markers",
                name="Held-Out Test Actuals",
                line=dict(color=COLOR_PRIMARY_TEXT, width=2.2),
                marker=dict(size=6, color=COLOR_PRIMARY_TEXT),
            ))

            if model_selected in fc_results["all_predictions"]:
                test_preds = fc_results["all_predictions"][model_selected]
                fig_fc_plotly.add_trace(go.Scatter(
                    x=test_dates,
                    y=test_preds,
                    mode="lines",
                    name=f"Test Prediction ({model_selected})",
                    line=dict(color=COLOR_PRIMARY_ACCENT, width=2.0, dash="dash"),
                ))

            # 3. Future Forecast Projection
            fut_dates = [pd.to_datetime(d) for d in future_df["Date"]]
            fig_fc_plotly.add_trace(go.Scatter(
                x=fut_dates,
                y=future_df["Forecasted_Daily_Revenue"],
                mode="lines+markers",
                name=f"Future {horizon_selected}-Day Forecast",
                line=dict(color="#10B981", width=2.5),
                marker=dict(size=6, color="#10B981", symbol="square"),
            ))

            # 4. 95% Confidence Interval Band
            fig_fc_plotly.add_trace(go.Scatter(
                x=fut_dates,
                y=future_df["Upper_95_CI"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig_fc_plotly.add_trace(go.Scatter(
                x=fut_dates,
                y=future_df["Lower_95_CI"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(16, 185, 129, 0.12)",
                name="95% Confidence Band",
            ))

            format_plotly_figure(
                fig_fc_plotly,
                title=f"Historical Sales, Held-Out Test Evaluation, and Future {horizon_selected}-Day Forecast ({CURRENCY_SYMBOL})",
                height=420,
                show_legend=True,
            )
            st.plotly_chart(fig_fc_plotly, use_container_width=True)

            # Split Forecast Table & Metrics
            t_col1, t_col2 = st.columns([1.3, 1.0])

            with t_col1:
                st.markdown(f"<div style='font-size:0.85rem; font-weight:700; color:{COLOR_PRIMARY_TEXT}; margin-bottom:0.5rem;'>PROJECTED REVENUE TABLE ({CURRENCY_SYMBOL})</div>", unsafe_allow_html=True)
                table_display = future_df.copy()
                table_display.columns = ["Date", f"Projected Daily Sales ({CURRENCY_SYMBOL})", f"95% Min ({CURRENCY_SYMBOL})", f"95% Max ({CURRENCY_SYMBOL})", "Model", "Horizon (Days)"]
                st.dataframe(table_display, use_container_width=True, hide_index=True)

                # CSV Download Button
                csv_buffer = future_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"📥 Download {horizon_selected}-Day Forecast as CSV",
                    data=csv_buffer,
                    file_name=f"supermarket_sales_forecast_{horizon_selected}days.csv",
                    mime="text/csv",
                )

            with t_col2:
                st.markdown(f"<div style='font-size:0.85rem; font-weight:700; color:{COLOR_PRIMARY_TEXT}; margin-bottom:0.5rem;'>OUT-OF-SAMPLE BENCHMARK ({horizon_selected}-DAY TEST)</div>", unsafe_allow_html=True)
                st.dataframe(fc_results["metrics"], use_container_width=True, hide_index=True)

                st.markdown(f"""
                <div class="alert-box" style="margin-top:0.6rem;">
                    <b>Model Validation Period</b>: Chronologically evaluated on the last <b>{horizon_selected}</b> held-out test days.
                    The top model (lowest RMSE) is automatically benchmarked against naive last-value persistence and seasonal lag baselines.
                </div>
                """, unsafe_allow_html=True)

            # Limitations Box
            st.markdown(f"""
            <div class="alert-warning-box">
                <b>⚠️ Transparent Academic Disclosures & Limitations</b>:
                <ul style="margin-top:0.3rem; margin-bottom:0; padding-left:1.2rem;">
                    <li><b>Sample Size Horizon</b>: The Kaggle supermarket dataset covers 89 calendar days (Jan 1, 2019 – Mar 30, 2019). While sufficient for short-term 1–14 day operational forecasting, it cannot reliably establish multi-year trends or major annual holiday seasonality.</li>
                    <li><b>Confidence Intervals</b>: Projections represent statistical estimates with 95% uncertainty bands; they are not financial guarantees.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        except Exception as err:
            st.error(f"Forecasting error: {err}")


# ==============================================================================
# SECTION C: MODEL EVALUATION
# ==============================================================================
elif nav_section == "Model Evaluation":
    st.markdown('<div class="page-tag">Rigorous Machine Learning Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Model Evaluation, Diagnostics & Leakage Audit</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-subtitle">Benchmarking classical Kaggle algorithms against modern tree ensembles, residual diagnostics, and mathematical proof of target leakage.</div>',
        unsafe_allow_html=True
    )

    tab_benchmark, tab_residuals, tab_leakage = st.tabs([
        "Cross-Algorithm Benchmark",
        "Residual Diagnostics & Fit",
        "Target Leakage Mathematical Proof",
    ])

    with tab_benchmark:
        rep_a_file = REPORTS_DIR / "scenario_a_model_comparison.csv"
        rep_b_file = REPORTS_DIR / "scenario_b_model_comparison.csv"

        if rep_a_file.exists():
            df_comp_a = pd.read_csv(rep_a_file)
            st.markdown(f"<div style='font-size:0.9rem; font-weight:700; color:{COLOR_PRIMARY_TEXT}; margin-bottom:0.4rem;'>Scenario A: Transaction Value Estimation (Unit Price & Quantity Available)</div>", unsafe_allow_html=True)
            st.dataframe(df_comp_a, use_container_width=True, hide_index=True)

            fig_comp = px.bar(
                df_comp_a,
                x="Algorithm",
                y="Val_RMSE",
                color="Algorithm",
                color_discrete_sequence=CHART_COLOR_PALETTE[:len(df_comp_a)],
                labels={"Val_RMSE": f"Validation RMSE ({CURRENCY_SYMBOL}) - Lower is Better"},
            )
            format_plotly_figure(fig_comp, title=f"Validation RMSE Comparison Across Candidate Algorithms ({CURRENCY_SYMBOL})", height=340, show_legend=False)
            st.plotly_chart(fig_comp, use_container_width=True)

        if rep_b_file.exists():
            df_comp_b = pd.read_csv(rep_b_file)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.9rem; font-weight:700; color:{COLOR_PRIMARY_TEXT}; margin-bottom:0.4rem;'>Scenario B: Pre-Scan Context Prediction (Demographics & Time Only)</div>", unsafe_allow_html=True)
            st.dataframe(df_comp_b, use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div class="ent-card" style="margin-top: 1rem;">
            <div style="font-size:0.85rem; font-weight:700; color:{COLOR_PRIMARY_TEXT}; margin-bottom:0.4rem;">
                Core Machine Learning Insights for Reviewers & Viva:
            </div>
            <ul style="font-size:0.84rem; color:{COLOR_SECONDARY_TEXT}; line-height: 1.6; margin-bottom: 0;">
                <li><b>Linear Regression Underfitting</b>: Standard OLS achieves higher RMSE because $Total = 1.05 \\times Price \\times Quantity$ is a bilinear multiplicative relationship that linear models cannot natively represent without interaction features.</li>
                <li><b>Ensemble Expressiveness</b>: <b>Extra Trees Regressor</b> and <b>Gradient Boosting</b> partition feature space to fit the non-linear interaction surface, achieving validation $RMSE < \\$5.00$ and $R^2 > 0.999$.</li>
                <li><b>Scenario B Negative Result</b>: Pre-scan customer demographics alone yield $R^2 \\approx 0.0$, honestly proving that retail customer demographics do not predict cart spending before items are selected.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with tab_residuals:
        st.markdown(f"<div style='font-size:0.9rem; font-weight:700; color:{COLOR_PRIMARY_TEXT}; margin-bottom:0.6rem;'>Actual vs. Predicted and Residual Analysis</div>", unsafe_allow_html=True)

        img_avp = FIGURES_DIR / "11_actual_vs_predicted.png"
        img_res = FIGURES_DIR / "12_residual_diagnostics.png"

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if img_avp.exists():
                st.image(str(img_avp), caption="Figure 11: Actual vs. Predicted Values (Scenario A - Extra Trees)", use_container_width=True)
            else:
                st.info("Run `python main.py` to generate Figure 11.")
        with col_r2:
            if img_res.exists():
                st.image(str(img_res), caption="Figure 12: Residual Diagnostics & Error Distribution", use_container_width=True)
            else:
                st.info("Run `python main.py` to generate Figure 12.")

    with tab_leakage:
        img_leak = FIGURES_DIR / "13_leakage_demonstration.png"
        if img_leak.exists():
            st.image(str(img_leak), caption="Figure 13: Target Leakage Tautology Demonstration", use_container_width=True)

        st.markdown(f"""
        <div class="alert-box" style="margin-top:1rem;">
            <b>Why Many Beginner Kaggle Notebooks Claim $R^2 = 1.0000$</b>:
            <br>
            The Kaggle supermarket dataset includes fields:
            <br>
            • <code>cogs</code> = Cost of Goods Sold = <code>Unit price × Quantity = Total / 1.05</code>
            <br>
            • <code>Tax 5%</code> = <code>0.05 × cogs = Total × (5 / 105)</code>
            <br>
            • <code>gross income</code> = identical to <code>Tax 5%</code>
            <br>
            When a model includes <code>cogs</code> or <code>Tax 5%</code> to predict <code>Total</code>, it merely learns an exact mathematical identity tautology ($y = 1.05 x$). 
            Our system strictly segregates and audits these leakage features to maintain scientific integrity.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# SECTION D: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
elif nav_section == "Exploratory Data Analysis":
    st.markdown('<div class="page-tag">Descriptive Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Exploratory Data Analysis & Empirical Distributions</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-subtitle">Multivariate statistical distributions, spending patterns, customer demographics, and operational temporal velocity.</div>',
        unsafe_allow_html=True
    )

    if filtered_df.empty:
        st.warning("No records match the active filter criteria.")
    else:
        # Row 1: Spend Distribution & Quantity Distribution
        e_col1, e_col2 = st.columns(2)

        with e_col1:
            fig_hist = px.histogram(
                filtered_df,
                x="Total",
                nbins=28,
                color_discrete_sequence=[COLOR_PRIMARY_ACCENT],
                labels={"Total": f"Transaction Spend ({CURRENCY_SYMBOL})"},
            )
            format_plotly_figure(fig_hist, title=f"Distribution of Transaction Total Spend ({CURRENCY_SYMBOL})", height=340, show_legend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

        with e_col2:
            fig_qty = px.histogram(
                filtered_df,
                x="Quantity",
                nbins=10,
                color_discrete_sequence=["#0EA5E9"],
                labels={"Quantity": "Cart Units Purchased"},
            )
            format_plotly_figure(fig_qty, title="Distribution of Cart Units Sold Per Transaction", height=340, show_legend=False)
            st.plotly_chart(fig_qty, use_container_width=True)

        # Row 2: Customer Rating by Branch & Spending by Customer Type
        e_col3, e_col4 = st.columns(2)

        with e_col3:
            fig_box = px.box(
                filtered_df,
                x="Branch",
                y="Rating",
                color="Branch",
                color_discrete_sequence=[COLOR_PRIMARY_ACCENT, "#0EA5E9", "#6366F1"],
            )
            format_plotly_figure(fig_box, title="Customer Survey Rating by Store Branch (1-10 Scale)", height=340, show_legend=False)
            st.plotly_chart(fig_box, use_container_width=True)

        with e_col4:
            fig_cust_box = px.box(
                filtered_df,
                x="Customer type",
                y="Total",
                color="Gender",
                color_discrete_sequence=[COLOR_PRIMARY_ACCENT, "#F59E0B"],
                labels={"Total": f"Transaction Total ({CURRENCY_SYMBOL})"},
            )
            format_plotly_figure(fig_cust_box, title=f"Spending Distribution by Loyalty Tier & Gender ({CURRENCY_SYMBOL})", height=340, show_legend=True)
            st.plotly_chart(fig_cust_box, use_container_width=True)

        # Row 3: Hourly Traffic & Day of Week
        e_col5, e_col6 = st.columns(2)

        with e_col5:
            hourly_txns = filtered_df.groupby("Hour")["Total"].agg(["count", "sum"]).reset_index()
            fig_hour = px.bar(
                hourly_txns,
                x="Hour",
                y="count",
                color_discrete_sequence=[COLOR_PRIMARY_ACCENT],
                labels={"Hour": "Hour of Day (24-Hour)", "count": "Transactions"},
            )
            format_plotly_figure(fig_hour, title="Customer Checkout Velocity by Hour of Day", height=340, show_legend=False)
            st.plotly_chart(fig_hour, use_container_width=True)

        with e_col6:
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_summary = (
                filtered_df.groupby("DayName")["Total"]
                .agg(["sum", "count"])
                .reindex(day_order)
                .dropna()
                .reset_index()
            )
            fig_dow = px.bar(
                dow_summary,
                x="DayName",
                y="sum",
                color_discrete_sequence=["#6366F1"],
                labels={"DayName": "Day of Week", "sum": f"Turnover ({CURRENCY_SYMBOL})"},
            )
            format_plotly_figure(fig_dow, title=f"Total Revenue by Day of the Week ({CURRENCY_SYMBOL})", height=340, show_legend=False)
            st.plotly_chart(fig_dow, use_container_width=True)


# ==============================================================================
# SECTION E: BUSINESS INSIGHTS
# ==============================================================================
elif nav_section == "Business Insights":
    st.markdown('<div class="page-tag">Strategic Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Empirical Supermarket Business Insights</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-subtitle">Data-driven findings calculated directly from the active filtered transaction dataset.</div>',
        unsafe_allow_html=True
    )

    if filtered_df.empty:
        st.warning("No records match the active filter criteria.")
    else:
        insights_engine = SupermarketInsightsEngine(filtered_df)
        all_insights = insights_engine.extract_all_insights()

        m = all_insights.get("merchandise", {})
        g = all_insights.get("geography", {})
        p = all_insights.get("payments", {})
        t = all_insights.get("temporal", {})
        c = all_insights.get("customers", {})

        bi_col1, bi_col2 = st.columns(2)

        with bi_col1:
            st.markdown(f"""
            <div class="ent-card">
                <div style="font-size:0.85rem; font-weight:700; color:{COLOR_PRIMARY_TEXT}; margin-bottom:0.6rem;">
                    📦 Merchandise & Department Performance
                </div>
                <div style="font-size:0.84rem; color:{COLOR_SECONDARY_TEXT}; line-height:1.6;">
                    • <b>Top Grossing Department</b>: <b>{m.get('top_revenue_category', 'N/A')}</b> leads with <b>{CURRENCY_SYMBOL}{m.get('top_revenue_value', 0):,.2f}</b> ({m.get('top_revenue_share_pct', 0)}% share).<br>
                    • <b>Lowest Revenue Department</b>: <b>{m.get('lowest_revenue_category', 'N/A')}</b> with <b>{CURRENCY_SYMBOL}{m.get('lowest_revenue_value', 0):,.2f}</b>.<br>
                    • <b>Highest Average Basket</b>: <b>{m.get('highest_avg_ticket_category', 'N/A')}</b> orders averaged <b>{CURRENCY_SYMBOL}{m.get('highest_avg_ticket_value', 0):,.2f}</b> per checkout.
                </div>
            </div>

            <div class="ent-card">
                <div style="font-size:0.85rem; font-weight:700; color:{COLOR_PRIMARY_TEXT}; margin-bottom:0.6rem;">
                    💳 Payment Channels & Digital Preferences
                </div>
                <div style="font-size:0.84rem; color:{COLOR_SECONDARY_TEXT}; line-height:1.6;">
                    • <b>Most Common Method</b>: <b>{p.get('most_frequent_method', 'N/A')}</b> represents <b>{p.get('transaction_share_pct', 0)}%</b> of orders ({p.get('transaction_count', 0):,} transactions).<br>
                    • <b>Highest Spend Channel</b>: <b>{p.get('highest_spend_method', 'N/A')}</b> exhibits the highest ticket value at <b>{CURRENCY_SYMBOL}{p.get('highest_spend_avg', 0):,.2f}</b>.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with bi_col2:
            st.markdown(f"""
            <div class="ent-card">
                <div style="font-size:0.85rem; font-weight:700; color:{COLOR_PRIMARY_TEXT}; margin-bottom:0.6rem;">
                    🏬 Store Geography & Branch Dynamics
                </div>
                <div style="font-size:0.84rem; color:{COLOR_SECONDARY_TEXT}; line-height:1.6;">
                    • <b>Leading Store</b>: <b>{g.get('leading_branch', 'N/A')}</b> accounts for <b>{CURRENCY_SYMBOL}{g.get('leading_branch_revenue', 0):,.2f}</b> ({g.get('leading_branch_share_pct', 0)}% share).<br>
                    • <b>Branch Parity</b>: Performance is balanced across locations with a spread of <b>{g.get('branch_spread_pct', 0)}%</b> between top and bottom stores.
                </div>
            </div>

            <div class="ent-card">
                <div style="font-size:0.85rem; font-weight:700; color:{COLOR_PRIMARY_TEXT}; margin-bottom:0.6rem;">
                    ⏰ Shopping Rhythm & Loyalty Engagement
                </div>
                <div style="font-size:0.84rem; color:{COLOR_SECONDARY_TEXT}; line-height:1.6;">
                    • <b>Peak Shopping Hour</b>: <b>{t.get('peak_shopping_hour', 'N/A')}</b> ({t.get('peak_hour_transactions', 0)} transactions).<br>
                    • <b>Top Sales Day</b>: <b>{t.get('peak_sales_day', 'N/A')}</b> generated <b>{CURRENCY_SYMBOL}{t.get('peak_day_revenue', 0):,.2f}</b>.<br>
                    • <b>Weekend Premium</b>: Weekend baskets average <b>{CURRENCY_SYMBOL}{t.get('weekend_avg_basket', 0):,.2f}</b> vs. <b>{CURRENCY_SYMBOL}{t.get('weekday_avg_basket', 0):,.2f}</b> on weekdays ({t.get('weekend_basket_premium_pct', 0)}% premium).<br>
                    • <b>Loyalty Contribution</b>: Members generate <b>{c.get('member_revenue_share_pct', 0)}%</b> of total turnover.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Causal Disclaimer
        st.markdown(f"""
        <div class="alert-box" style="margin-top:0.8rem;">
            <b>Scientific Methodology Disclaimer (Descriptive Association vs. Causal Attribution)</b>:
            <br>
            All findings presented above are strictly descriptive correlations observed in historical supermarket transactions. 
            For example, while transactions completed with digital e-wallets exhibit higher ticket values, this does <i>not</i> imply that forcing shoppers to use e-wallets causes them to purchase more. 
            Supermarket management must avoid committing capital expenditures based on descriptive associations without conducting controlled A/B testing.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# SECTION F: DATA AND MODEL INFORMATION & CSV UPLOAD
# ==============================================================================
elif nav_section == "Data & Model Information":
    st.markdown('<div class="page-tag">System Metadata & Inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Dataset Schema, Validation & Database Audit</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-subtitle">Inspect underlying data structures, audit schemas, upload custom CSVs, and examine relational tables.</div>',
        unsafe_allow_html=True
    )

    tab_data_info, tab_upload, tab_sql = st.tabs([
        "Dataset Metadata & Schema",
        "Upload Custom CSV Dataset",
        "Relational SQLite Database",
    ])

    with tab_data_info:
        loader = SupermarketDataLoader()
        inspection = loader.inspect_dataset(df_raw)

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Total Rows", f"{inspection['shape'][0]:,}")
        with m_col2:
            st.metric("Total Columns", f"{inspection['shape'][1]}")
        with m_col3:
            st.metric("Duplicate Rows", f"{inspection['duplicate_count']}")
        with m_col4:
            st.metric("Memory Footprint", f"{inspection['memory_usage_mb']} MB")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.85rem; font-weight:700; color:{COLOR_PRIMARY_TEXT}; margin-bottom:0.4rem;'>COLUMN SCHEMA & DATA TYPES</div>", unsafe_allow_html=True)

        schema_table = pd.DataFrame({
            "Column Name": list(df_raw.columns),
            "Data Type": [str(t) for t in df_raw.dtypes.values],
            "Non-Null Count": [df_raw[c].count() for c in df_raw.columns],
            "Missing Values": [df_raw[c].isnull().sum() for c in df_raw.columns],
            "Sample Value": [str(df_raw[c].iloc[0]) for c in df_raw.columns],
        })
        st.dataframe(schema_table, use_container_width=True, hide_index=True)

    with tab_upload:
        st.markdown(f"<div style='font-size:0.85rem; font-weight:700; color:{COLOR_PRIMARY_TEXT}; margin-bottom:0.4rem;'>Upload New Supermarket Transaction CSV</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file:",
            type=["csv", "xlsx", "xls"],
            help="Upload a file conforming to the Kaggle Supermarket Sales schema",
        )

        if uploaded_file is not None:
            try:
                loader = SupermarketDataLoader()
                uploaded_df = loader.load_from_buffer(uploaded_file, uploaded_file.name)
                is_valid, report = loader.validate_schema(uploaded_df)

                if is_valid:
                    st.success(f"✅ Schema validation passed! File '{uploaded_file.name}' contains {len(uploaded_df):,} valid transactions.")
                    st.dataframe(uploaded_df.head(5), use_container_width=True)
                else:
                    st.warning(f"⚠️ Validation warnings: Missing expected columns: {report['missing_expected_columns']}")
            except DataValidationError as e:
                st.error(f"❌ Validation Error: {e}")
            except Exception as e:
                st.error(f"❌ Error parsing file: {e}")

    with tab_sql:
        st.markdown(f"<div style='font-size:0.85rem; font-weight:700; color:{COLOR_PRIMARY_TEXT}; margin-bottom:0.4rem;'>Predefined Safe Analytical Database Queries</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='font-size:0.84rem; color:{COLOR_SECONDARY_TEXT}; margin-bottom:0.8rem;'>Select from validated, parameterized analytical queries executed directly on <code>supermarket_sales.db</code>. Arbitrary user SQL is disabled for application security.</div>",
            unsafe_allow_html=True
        )

        preset_queries = {
            "Top 5 Product Lines by Revenue": "SELECT product_line, ROUND(SUM(total), 2) AS total_revenue, SUM(quantity) AS units_sold, COUNT(*) AS orders FROM transactions GROUP BY product_line ORDER BY total_revenue DESC LIMIT 5;",
            "Branch Performance Comparison": "SELECT branch, city, ROUND(SUM(total), 2) AS total_revenue, COUNT(*) AS transactions, ROUND(AVG(total), 2) AS avg_ticket FROM transactions GROUP BY branch, city ORDER BY total_revenue DESC;",
            "Hourly Transaction Velocity": "SELECT hour, COUNT(*) AS transactions, ROUND(SUM(total), 2) AS hourly_revenue FROM transactions WHERE hour IS NOT NULL GROUP BY hour ORDER BY hour ASC;",
            "Model Metrics Audit Table": "SELECT scenario, algorithm, mae, rmse, r2, mape_pct, created_at FROM model_metrics ORDER BY id DESC LIMIT 15;",
        }

        selected_query_label = st.selectbox("Select Predefined Query:", list(preset_queries.keys()))
        selected_sql = preset_queries[selected_query_label]

        st.code(selected_sql, language="sql")

        if st.button("Execute Safe Query"):
            try:
                db_conn = SupermarketDatabase()
                query_result = db_conn.query_to_dataframe(selected_sql)
                st.success(f"Execution successful: {len(query_result)} record(s) returned.")
                st.dataframe(query_result, use_container_width=True)
            except Exception as err:
                st.error(f"Database query error: {err}")
