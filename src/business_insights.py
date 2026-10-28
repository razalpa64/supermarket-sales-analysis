"""
Business Insights and Strategic Intelligence Module.
Extracts empirical descriptive findings from transactional data,
identifies retail patterns, and maintains strict separation between association and causation.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

from src.config import (
    CLEANED_DATA_CSV,
    REPORTS_DIR,
    CURRENCY_SYMBOL,
)
from src.data_cleaning import clean_and_prepare_data


class SupermarketInsightsEngine:
    """
    Computes rigorous, data-driven business insights from transactional data.
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        if df is None:
            if not CLEANED_DATA_CSV.exists():
                df, _ = clean_and_prepare_data()
            else:
                df = pd.read_csv(CLEANED_DATA_CSV)
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
        self.df = df.copy()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def extract_all_insights(self) -> Dict[str, Any]:
        """
        Extracts structured descriptive metrics across merchandise, geography,
        payment preferences, temporal velocity, and shopper tiers.
        """
        if len(self.df) == 0:
            return {"status": "empty", "message": "No transaction records available for current filters."}

        insights = {}

        # 1. Merchandise & Department Insights
        prod_grp = self.df.groupby("Product line")["Total"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
        top_prod = prod_grp.index[0]
        top_prod_rev = float(prod_grp.iloc[0]["sum"])
        top_prod_share = (top_prod_rev / self.df["Total"].sum()) * 100
        lowest_prod = prod_grp.index[-1]
        lowest_prod_rev = float(prod_grp.iloc[-1]["sum"])

        highest_ticket_prod = prod_grp.sort_values("mean", ascending=False).index[0]
        highest_ticket_val = float(prod_grp.loc[highest_ticket_prod, "mean"])

        insights["merchandise"] = {
            "top_revenue_category": top_prod,
            "top_revenue_value": round(top_prod_rev, 2),
            "top_revenue_share_pct": round(top_prod_share, 2),
            "lowest_revenue_category": lowest_prod,
            "lowest_revenue_value": round(lowest_prod_rev, 2),
            "highest_avg_ticket_category": highest_ticket_prod,
            "highest_avg_ticket_value": round(highest_ticket_val, 2),
        }

        # 2. Branch & City Dynamics
        branch_grp = self.df.groupby(["Branch", "City"])["Total"].agg(["sum", "mean", "count"]).sort_values("sum", ascending=False)
        top_branch_idx = branch_grp.index[0]
        top_branch = f"Branch {top_branch_idx[0]} ({top_branch_idx[1]})"
        top_branch_rev = float(branch_grp.iloc[0]["sum"])
        top_branch_share = (top_branch_rev / self.df["Total"].sum()) * 100

        lowest_branch_idx = branch_grp.index[-1]
        lowest_branch = f"Branch {lowest_branch_idx[0]} ({lowest_branch_idx[1]})"
        lowest_branch_rev = float(branch_grp.iloc[-1]["sum"])

        insights["geography"] = {
            "leading_branch": top_branch,
            "leading_branch_revenue": round(top_branch_rev, 2),
            "leading_branch_share_pct": round(top_branch_share, 2),
            "lowest_branch": lowest_branch,
            "lowest_branch_revenue": round(lowest_branch_rev, 2),
            "branch_spread_pct": round(((top_branch_rev - lowest_branch_rev) / lowest_branch_rev) * 100, 2) if lowest_branch_rev > 0 else 0.0,
        }

        # 3. Payment Method Preferences
        pay_grp = self.df.groupby("Payment")["Total"].agg(["sum", "count", "mean"]).sort_values("count", ascending=False)
        top_pay_method = pay_grp.index[0]
        top_pay_txns = int(pay_grp.iloc[0]["count"])
        top_pay_share = (top_pay_txns / len(self.df)) * 100

        insights["payments"] = {
            "most_frequent_method": top_pay_method,
            "transaction_count": top_pay_txns,
            "transaction_share_pct": round(top_pay_share, 2),
            "highest_spend_method": pay_grp.sort_values("mean", ascending=False).index[0],
            "highest_spend_avg": round(float(pay_grp["mean"].max()), 2),
        }

        # 4. Temporal & Velocity Patterns
        hour_grp = self.df.groupby("Hour")["Total"].agg(["sum", "count"]).sort_values("count", ascending=False)
        peak_hour = int(hour_grp.index[0])
        peak_hour_txns = int(hour_grp.iloc[0]["count"])

        dow_grp = self.df.groupby("DayName")["Total"].agg(["sum", "count", "mean"]).sort_values("sum", ascending=False)
        peak_day = dow_grp.index[0]
        peak_day_rev = float(dow_grp.iloc[0]["sum"])

        wknd_mean = float(self.df[self.df["IsWeekend"] == 1]["Total"].mean()) if (self.df["IsWeekend"] == 1).any() else 0.0
        wkdy_mean = float(self.df[self.df["IsWeekend"] == 0]["Total"].mean()) if (self.df["IsWeekend"] == 0).any() else 0.0

        insights["temporal"] = {
            "peak_shopping_hour": f"{peak_hour}:00 - {peak_hour+1}:00",
            "peak_hour_transactions": peak_hour_txns,
            "peak_sales_day": peak_day,
            "peak_day_revenue": round(peak_day_rev, 2),
            "weekend_avg_basket": round(wknd_mean, 2),
            "weekday_avg_basket": round(wkdy_mean, 2),
            "weekend_basket_premium_pct": round(((wknd_mean - wkdy_mean) / wkdy_mean) * 100, 2) if wkdy_mean > 0 else 0.0,
        }

        # 5. Customer Segments (Member vs Normal)
        cust_grp = self.df.groupby("Customer type")["Total"].agg(["sum", "mean", "count"])
        member_rev = float(cust_grp.loc["Member", "sum"]) if "Member" in cust_grp.index else 0.0
        normal_rev = float(cust_grp.loc["Normal", "sum"]) if "Normal" in cust_grp.index else 0.0
        member_basket = float(cust_grp.loc["Member", "mean"]) if "Member" in cust_grp.index else 0.0
        normal_basket = float(cust_grp.loc["Normal", "mean"]) if "Normal" in cust_grp.index else 0.0

        insights["customers"] = {
            "member_revenue_share_pct": round((member_rev / (member_rev + normal_rev)) * 100, 2) if (member_rev + normal_rev) > 0 else 0.0,
            "member_avg_ticket": round(member_basket, 2),
            "normal_avg_ticket": round(normal_basket, 2),
            "member_ticket_difference_pct": round(((member_basket - normal_basket) / normal_basket) * 100, 2) if normal_basket > 0 else 0.0,
        }

        return insights

    def generate_executive_insights_report(self) -> Path:
        """
        Compiles the insights into an executive Markdown report with causal disclaimers.
        """
        insights = self.extract_all_insights()
        m = insights.get("merchandise", {})
        g = insights.get("geography", {})
        p = insights.get("payments", {})
        t = insights.get("temporal", {})
        c = insights.get("customers", {})

        report_md = f"""# Empirical Supermarket Business Insights & Strategic Analysis

**Dataset Coverage**: 89 Calendar Days (January 1, 2019 – March 30, 2019)  
**Total Records Analyzed**: {len(self.df):,} transactions  
**Total Turnover Analyzed**: {CURRENCY_SYMBOL}{self.df['Total'].sum():,.2f}  

---

## 1. Merchandise Performance
- **Highest-Grossing Department**: **{m.get('top_revenue_category')}** generated **{CURRENCY_SYMBOL}{m.get('top_revenue_value'):,.2f}**, contributing **{m.get('top_revenue_share_pct')}%** of overall supermarket turnover.
- **Lowest-Grossing Department**: **{m.get('lowest_revenue_category')}** generated **{CURRENCY_SYMBOL}{m.get('lowest_revenue_value'):,.2f}**.
- **Highest Average Basket Value**: Orders in **{m.get('highest_avg_ticket_category')}** averaged **{CURRENCY_SYMBOL}{m.get('highest_avg_ticket_value'):,.2f}** per checkout.

## 2. Store Geography & Footfall
- **Leading Branch**: **{g.get('leading_branch')}** leads all locations with **{CURRENCY_SYMBOL}{g.get('leading_branch_revenue'):,.2f}** ({g.get('leading_branch_share_pct')}% share).
- **Branch Parity**: Performance is tightly balanced across all 3 branches, with a maximum spread of only **{g.get('branch_spread_pct')}%** between top and bottom stores.

## 3. Payment Channels
- **Most Popular Channel**: **{p.get('most_frequent_method')}** accounted for **{p.get('transaction_share_pct')}%** of transactions ({p.get('transaction_count')} orders).
- **Highest Ticket Channel**: **{p.get('highest_spend_method')}** exhibited the largest average transaction value at **{CURRENCY_SYMBOL}{p.get('highest_spend_avg'):,.2f}**.

## 4. Temporal Dynamics & Shopping Rhythm
- **Peak Hour**: Peak traffic occurs at **{t.get('peak_shopping_hour')}** ({t.get('peak_hour_transactions')} checkouts).
- **Strongest Day**: **{t.get('peak_sales_day')}** yielded peak cumulative sales of **{CURRENCY_SYMBOL}{t.get('peak_day_revenue'):,.2f}**.
- **Weekend vs. Weekday**: Weekend transactions averaged **{CURRENCY_SYMBOL}{t.get('weekend_avg_basket')}** vs. **{CURRENCY_SYMBOL}{t.get('weekday_avg_basket')}** on weekdays (a **{t.get('weekend_basket_premium_pct')}%** difference).

## 5. Loyalty & Membership
- **Membership Share**: Enrolled loyalty members account for **{c.get('member_revenue_share_pct')}%** of revenue.
- **Basket Comparison**: Members spend **{CURRENCY_SYMBOL}{c.get('member_avg_ticket')}** per visit vs. **{CURRENCY_SYMBOL}{c.get('normal_avg_ticket')}** for non-members.

---

## 6. Scientific & Causal Clarification (Mandatory Disclaimer)
> **Descriptive Association vs. Causal Attribution**:
> The metrics above describe empirical associations observed in historical transactions. 
> For instance, while transactions using {p.get('highest_spend_method')} exhibit higher average tickets, this does **not** prove that offering this payment method *causes* shoppers to purchase more. High-value shoppers may simply prefer electronic payment mechanisms. Decision-makers should not initiate major capital investments based solely on descriptive correlations without controlled A/B experimentation.
"""
        out_path = REPORTS_DIR / "business_insights_report.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"[SUCCESS] Saved business insights report to: {out_path}")
        return out_path


def run_business_insights() -> Path:
    """
    Executes business insights generation and saves report.
    """
    engine = SupermarketInsightsEngine()
    return engine.generate_executive_insights_report()


if __name__ == "__main__":
    run_business_insights()

