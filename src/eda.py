"""
Exploratory Data Analysis (EDA) Module for Supermarket Sales System.
Calculates summary statistics, performance metrics, product and branch rankings,
customer behavior, and exports publication-grade visualization artifacts.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from src.config import (
        CLEANED_DATA_CSV,
        FIGURES_DIR,
        REPORTS_DIR,
    )
except ImportError:
    from config import (
        CLEANED_DATA_CSV,
        FIGURES_DIR,
        REPORTS_DIR,
    )

# Visual style configuration
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["figure.titlesize"] = 13
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9


class SupermarketEDA:
    """
    Performs comprehensive statistical, temporal, and category analytics.
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        if df is None:
            if not CLEANED_DATA_CSV.exists():
                raise FileNotFoundError(f"Cleaned data not found at {CLEANED_DATA_CSV}")
            self.df = pd.read_csv(CLEANED_DATA_CSV)
            if "Date" in self.df.columns:
                self.df["Date"] = pd.to_datetime(self.df["Date"])
        else:
            self.df = df.copy()
            if "Date" in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df["Date"]):
                self.df["Date"] = pd.to_datetime(self.df["Date"])

        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def calculate_overall_kpis(self) -> Dict[str, Any]:
        """
        Computes core business KPIs across the entire dataset.
        """
        total_revenue = float(self.df["Total"].sum())
        total_transactions = int(len(self.df))
        total_units = int(self.df["Quantity"].sum()) if "Quantity" in self.df.columns else 0
        avg_order_value = float(self.df["Total"].mean())
        avg_units_per_order = float(self.df["Quantity"].mean()) if "Quantity" in self.df.columns else 0.0
        total_cogs = float(self.df["cogs"].sum()) if "cogs" in self.df.columns else 0.0
        total_gross_income = float(self.df["gross income"].sum()) if "gross income" in self.df.columns else 0.0
        avg_rating = float(self.df["Rating"].mean()) if "Rating" in self.df.columns else 0.0

        return {
            "total_revenue": round(total_revenue, 2),
            "total_transactions": total_transactions,
            "total_units_sold": total_units,
            "average_order_value": round(avg_order_value, 2),
            "average_units_per_order": round(avg_units_per_order, 2),
            "total_cogs": round(total_cogs, 2),
            "total_gross_income": round(total_gross_income, 2),
            "gross_profit_margin_pct": round((total_gross_income / total_revenue) * 100, 2) if total_revenue > 0 else 0.0,
            "average_rating": round(avg_rating, 2),
        }

    def analyze_product_performance(self) -> pd.DataFrame:
        """
        Ranks product lines by revenue, units sold, and average ticket size.
        """
        prod_summary = (
            self.df.groupby("Product line")
            .agg(
                Revenue=("Total", "sum"),
                Transactions=("Total", "count"),
                Units_Sold=("Quantity", "sum"),
                Avg_Ticket=("Total", "mean"),
                Avg_Unit_Price=("Unit price", "mean"),
                Avg_Rating=("Rating", "mean"),
            )
            .reset_index()
        )
        total_rev = prod_summary["Revenue"].sum()
        prod_summary["Revenue_Share_Pct"] = (prod_summary["Revenue"] / total_rev) * 100
        prod_summary = prod_summary.sort_values(by="Revenue", ascending=False).reset_index(drop=True)
        return prod_summary

    def analyze_branch_performance(self) -> pd.DataFrame:
        """
        Compares store branches and cities on revenue and customer volume.
        """
        branch_summary = (
            self.df.groupby(["Branch", "City"])
            .agg(
                Revenue=("Total", "sum"),
                Transactions=("Total", "count"),
                Units_Sold=("Quantity", "sum"),
                Avg_Ticket=("Total", "mean"),
                Avg_Rating=("Rating", "mean"),
            )
            .reset_index()
        )
        total_rev = branch_summary["Revenue"].sum()
        branch_summary["Revenue_Share_Pct"] = (branch_summary["Revenue"] / total_rev) * 100
        branch_summary = branch_summary.sort_values(by="Revenue", ascending=False).reset_index(drop=True)
        return branch_summary

    def analyze_customer_and_payment(self) -> Dict[str, pd.DataFrame]:
        """
        Analyzes customer types, gender demographics, and payment preferences.
        """
        cust_summary = (
            self.df.groupby("Customer type")
            .agg(
                Revenue=("Total", "sum"),
                Transactions=("Total", "count"),
                Avg_Ticket=("Total", "mean"),
                Avg_Rating=("Rating", "mean"),
            )
            .reset_index()
        )

        gender_summary = (
            self.df.groupby("Gender")
            .agg(
                Revenue=("Total", "sum"),
                Transactions=("Total", "count"),
                Avg_Ticket=("Total", "mean"),
            )
            .reset_index()
        )

        payment_summary = (
            self.df.groupby("Payment")
            .agg(
                Revenue=("Total", "sum"),
                Transactions=("Total", "count"),
                Avg_Ticket=("Total", "mean"),
            )
            .reset_index()
        )

        return {
            "customer_type": cust_summary,
            "gender": gender_summary,
            "payment": payment_summary,
        }

    def generate_all_figures(self) -> Dict[str, Path]:
        """
        Generates and saves all 8 analytical figures to outputs/figures/.
        """
        figures_saved = {}

        # 1. Overall Revenue Distribution
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(self.df["Total"], kde=True, color="#2b5c8f", bins=30, ax=ax[0])
        ax[0].set_title("Distribution of Transaction Total Sales", fontweight="bold")
        ax[0].set_xlabel("Total Transaction Value ($)")
        ax[0].set_ylabel("Frequency")
        ax[0].axvline(self.df["Total"].mean(), color="red", linestyle="--", label=f"Mean: ${self.df['Total'].mean():.2f}")
        ax[0].axvline(self.df["Total"].median(), color="green", linestyle=":", label=f"Median: ${self.df['Total'].median():.2f}")
        ax[0].legend()

        sns.boxplot(y=self.df["Total"], color="#6baed6", ax=ax[1])
        ax[1].set_title("Boxplot of Transaction Values (Outlier Check)", fontweight="bold")
        ax[1].set_ylabel("Total Transaction Value ($)")
        plt.tight_layout()
        p1 = FIGURES_DIR / "01_overall_revenue_distribution.png"
        fig.savefig(p1, dpi=300)
        plt.close(fig)
        figures_saved["revenue_dist"] = p1

        # 2. Product Line Performance
        prod_df = self.analyze_product_performance()
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        sns.barplot(data=prod_df, x="Revenue", y="Product line", hue="Product line", palette="Blues_r", legend=False, ax=ax[0])
        ax[0].set_title("Total Revenue by Product Line", fontweight="bold")
        ax[0].set_xlabel("Revenue ($)")

        sns.barplot(data=prod_df, x="Avg_Ticket", y="Product line", hue="Product line", palette="Purples_r", legend=False, ax=ax[1])
        ax[1].set_title("Average Ticket Value by Product Line", fontweight="bold")
        ax[1].set_xlabel("Average Ticket ($)")
        plt.tight_layout()
        p2 = FIGURES_DIR / "02_product_line_performance.png"
        fig.savefig(p2, dpi=300)
        plt.close(fig)
        figures_saved["product_performance"] = p2

        # 3. Branch and City Comparison
        branch_df = self.analyze_branch_performance()
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        sns.barplot(data=branch_df, x="Branch", y="Revenue", hue="City", palette="Set2", ax=ax[0])
        ax[0].set_title("Total Revenue by Store Branch & City", fontweight="bold")
        ax[0].set_ylabel("Total Revenue ($)")

        # Product breakdown by branch
        branch_prod = self.df.groupby(["Branch", "Product line"])["Total"].sum().unstack()
        branch_prod.plot(kind="bar", stacked=True, colormap="tab20", ax=ax[1])
        ax[1].set_title("Revenue Composition by Branch", fontweight="bold")
        ax[1].set_ylabel("Total Revenue ($)")
        ax[1].legend(title="Product Line", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        p3 = FIGURES_DIR / "03_branch_and_city_analysis.png"
        fig.savefig(p3, dpi=300)
        plt.close(fig)
        figures_saved["branch_analysis"] = p3

        # 4. Customer Demographics and Payment Breakdown
        cust_dict = self.analyze_customer_and_payment()
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        # Payment method
        payment_df = cust_dict["payment"]
        ax[0].pie(
            payment_df["Revenue"],
            labels=payment_df["Payment"],
            autopct="%1.1f%%",
            colors=["#66c2a5", "#fc8d62", "#8da0cb"],
            startangle=140,
        )
        ax[0].set_title("Revenue Share by Payment Method", fontweight="bold")

        # Customer type vs gender
        sns.countplot(data=self.df, x="Customer type", hue="Gender", palette="pastel", ax=ax[1])
        ax[1].set_title("Transaction Count by Customer Type & Gender", fontweight="bold")

        # Average spending by customer type
        sns.boxplot(data=self.df, x="Customer type", y="Total", hue="Gender", palette="Set3", ax=ax[2])
        ax[2].set_title("Spending Distribution by Customer & Gender", fontweight="bold")
        plt.tight_layout()
        p4 = FIGURES_DIR / "04_customer_and_payment_breakdown.png"
        fig.savefig(p4, dpi=300)
        plt.close(fig)
        figures_saved["customer_payment"] = p4

        # 5. Hourly Activity and Day-of-Week Patterns
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        hourly = self.df.groupby("Hour")["Total"].agg(["count", "sum"]).reset_index()
        sns.barplot(data=hourly, x="Hour", y="count", color="#3182bd", ax=ax[0])
        ax[0].set_title("Transaction Volume by Hour of Day", fontweight="bold")
        ax[0].set_xlabel("Hour (24-Hour Format)")
        ax[0].set_ylabel("Number of Transactions")

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_df = self.df.groupby("DayName")["Total"].agg(["sum", "count"]).reindex(day_order).reset_index()
        sns.barplot(data=dow_df, x="DayName", y="sum", hue="DayName", palette="Blues_d", legend=False, ax=ax[1])
        ax[1].set_title("Total Revenue by Day of the Week", fontweight="bold")
        ax[1].set_xlabel("Day of Week")
        ax[1].set_ylabel("Revenue ($)")
        ax[1].tick_params(axis="x", rotation=30)
        plt.tight_layout()
        p5 = FIGURES_DIR / "05_hourly_and_dayofweek_patterns.png"
        fig.savefig(p5, dpi=300)
        plt.close(fig)
        figures_saved["temporal_patterns"] = p5

        # 6. Daily Sales Trend & 7-Day Moving Average
        daily_sales = self.df.groupby("Date")["Total"].sum().reset_index().sort_values("Date")
        daily_sales["MA7"] = daily_sales["Total"].rolling(window=7, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(daily_sales["Date"], daily_sales["Total"], marker="o", markersize=3, color="#9ecae1", label="Daily Revenue")
        ax.plot(daily_sales["Date"], daily_sales["MA7"], color="#08519c", linewidth=2.5, label="7-Day Moving Average")
        ax.set_title("Daily Supermarket Revenue Trend with 7-Day Rolling Average (Jan - Mar 2019)", fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Revenue ($)")
        ax.legend()
        plt.tight_layout()
        p6 = FIGURES_DIR / "06_daily_sales_trend_moving_avg.png"
        fig.savefig(p6, dpi=300)
        plt.close(fig)
        figures_saved["daily_trend"] = p6

        # 7. Correlation Matrix of Numerical Attributes
        num_cols = ["Unit price", "Quantity", "Tax 5%", "Total", "cogs", "gross income", "Rating"]
        available_num_cols = [c for c in num_cols if c in self.df.columns]
        corr = self.df[available_num_cols].corr()

        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="vlag", vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Pearson Correlation"})
        ax.set_title("Correlation Heatmap: Highlight of Target Leakage Tautology", fontweight="bold")
        plt.tight_layout()
        p7 = FIGURES_DIR / "07_correlation_matrix.png"
        fig.savefig(p7, dpi=300)
        plt.close(fig)
        figures_saved["correlation_matrix"] = p7

        # 8. Customer Ratings by Branch
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=self.df, x="Branch", y="Rating", hue="Branch", palette="Set1", legend=False, ax=ax)
        ax.set_title("Customer Rating Distribution by Branch", fontweight="bold")
        ax.set_ylabel("Rating (Scale 1-10)")
        plt.tight_layout()
        p8 = FIGURES_DIR / "08_rating_distribution_by_branch.png"
        fig.savefig(p8, dpi=300)
        plt.close(fig)
        figures_saved["ratings_by_branch"] = p8

        print(f"[SUCCESS] Generated and saved {len(figures_saved)} figures to {FIGURES_DIR}")
        return figures_saved


def run_exploratory_analysis() -> Dict[str, Any]:
    """
    Executes full EDA suite and produces summaries and figures.
    """
    eda = SupermarketEDA()
    kpis = eda.calculate_overall_kpis()
    prod_summary = eda.analyze_product_performance()
    branch_summary = eda.analyze_branch_performance()
    figs = eda.generate_all_figures()

    print("\n================== KEY PERFORMANCE INDICATORS ==================")
    for k, v in kpis.items():
        print(f"{k.replace('_', ' ').title():<30}: {v}")
    print("================================================================")

    return {
        "kpis": kpis,
        "product_performance": prod_summary,
        "branch_performance": branch_summary,
        "figures": figs,
    }


if __name__ == "__main__":
    run_exploratory_analysis()
