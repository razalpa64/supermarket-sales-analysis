"""
Unit tests for business insights engine and causal disclaimers.
"""

from src.business_insights import SupermarketInsightsEngine
from src.data_cleaning import clean_and_prepare_data


def test_business_insights_generation():
    clean_df, _ = clean_and_prepare_data(save_outputs=False)
    engine = SupermarketInsightsEngine(clean_df)
    insights = engine.extract_all_insights()

    assert "merchandise" in insights
    assert "geography" in insights
    assert "payments" in insights
    assert "temporal" in insights
    assert "customers" in insights

    assert "top_revenue_category" in insights["merchandise"]
    assert "leading_branch" in insights["geography"]
    assert "most_frequent_method" in insights["payments"]
    assert "peak_shopping_hour" in insights["temporal"]


def test_empty_dataframe_insights_handling():
    empty_df = clean_and_prepare_data(save_outputs=False)[0].iloc[0:0]
    engine = SupermarketInsightsEngine(empty_df)
    insights = engine.extract_all_insights()
    assert insights.get("status") == "empty"

