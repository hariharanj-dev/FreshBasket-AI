"""
Tests for FreshBasket AI - Business Insights Engine & Endpoints
===============================================================
Validates statistical trend detection, forecast intelligence, inventory risk correlation,
edge cases (empty, zeros, partial weeks), and REST API endpoints.
"""

import pytest
import pandas as pd
from fastapi.testclient import TestClient
from api.main import app, get_data
from src.business.insight_engine import (
    detect_demand_trends,
    generate_forecast_insights,
    generate_inventory_risk_insights,
    synthesize_business_insights,
)
from src.business.inventory_planner import compute_replenishment_plan

client = TestClient(app)


def test_detect_demand_trends_statistical_integrity():
    data = get_data()
    df_h = data["historical"]
    res = detect_demand_trends(df_h)

    assert "summary" in res
    assert "comparison_period" in res
    assert "methodology" in res
    assert res["summary"]["total"] == 33  # 33 product families on aggregate

    # Sum of categories must equal total
    s = res["summary"]
    assert s["rising"] + s["declining"] + s["stable"] + s["insufficient"] == s["total"]

    # Verify each trend item has required statistical properties
    for t in res["trends"]:
        assert "family" in t
        assert t["trend_class"] in ["Rising", "Declining", "Stable", "Insufficient data"]
        assert "pct_change" in t
        assert "slope" in t
        assert "p_value" in t
        assert "is_significant" in t
        assert isinstance(t["is_significant"], bool)
        # If marked significant, p_value must be < 0.05
        if t["is_significant"]:
            assert t["p_value"] < 0.05


def test_detect_demand_trends_with_filters():
    data = get_data()
    df_h = data["historical"]

    # Store filter
    res_store = detect_demand_trends(df_h, store_nbr=1)
    assert res_store["summary"]["total"] > 0
    assert all(t["store_nbr"] == 1 for t in res_store["trends"])

    # Family filter
    res_fam = detect_demand_trends(df_h, family="BEVERAGES")
    assert res_fam["summary"]["total"] == 54  # 54 stores for BEVERAGES
    assert all(t["family"] == "BEVERAGES" for t in res_fam["trends"])


def test_detect_demand_trends_empty_and_insufficient():
    # Empty DataFrame
    empty_res = detect_demand_trends(pd.DataFrame())
    assert empty_res["summary"]["total"] == 0

    # Near-zero dummy series
    dummy = pd.DataFrame({
        "week_start_date": pd.date_range("2017-05-01", periods=12, freq="W-MON"),
        "store_nbr": [1] * 12,
        "family": ["DORMANT_CAT"] * 12,
        "weekly_sales": [0.0] * 12,
        "is_partial_week": [False] * 12,
    })
    res_dummy = detect_demand_trends(dummy, store_nbr=1, family="DORMANT_CAT")
    assert res_dummy["summary"]["insufficient"] == 1
    assert res_dummy["trends"][0]["trend_class"] == "Insufficient data"


def test_forecast_insights_generation():
    data = get_data()
    df_fc = data["forecast"]
    df_h = data["historical"]

    res = generate_forecast_insights(df_fc, df_h, model="naive_forecast")
    assert len(res["top_demand_families"]) == 10
    assert len(res["horizon_uncertainty"]) == 13
    assert len(res["top_demand_stores"]) == 10
    assert len(res["model_divergence"]) > 0

    # Verify uncertainty metrics across weeks 1-13
    for w in res["horizon_uncertainty"]:
        assert 1 <= w["horizon_week"] <= 13
        assert w["uncertainty_level"] in ["Low", "Moderate", "High"]
        assert w["uncertainty_cv_pct"] >= 0.0


def test_inventory_risk_insights_generation():
    data = get_data()
    plan = compute_replenishment_plan(data["inventory"], data["forecast"], model_col="naive_forecast", supplier_df=data["suppliers"])

    res_demo = generate_inventory_risk_insights(plan, data_source_mode="DEMO")
    assert res_demo["is_demo_data"] is True
    assert "DEMO" in res_demo["demo_warning"]
    assert res_demo["risk_counts"]["HIGH"] == 354
    assert res_demo["risk_counts"]["MEDIUM"] == 368
    assert res_demo["below_reorder_point_count"] > 0
    assert len(res_demo["critical_vulnerable_items"]) > 0

    res_custom = generate_inventory_risk_insights(plan, data_source_mode="CUSTOM")
    assert res_custom["is_demo_data"] is False
    assert "USER-UPLOADED" in res_custom["demo_warning"]


def test_synthesize_business_insights_contract():
    data = get_data()
    plan = compute_replenishment_plan(data["inventory"], data["forecast"], model_col="naive_forecast", supplier_df=data["suppliers"])

    cards = synthesize_business_insights(data["historical"], data["forecast"], plan)
    assert len(cards) >= 4

    required_keys = [
        "id", "category", "priority", "title", "conclusion",
        "supporting_data", "comparison_period", "recommended_action", "limitations"
    ]

    for card in cards:
        for k in required_keys:
            assert k in card, f"Missing key {k} in insight card {card.get('id')}"
        assert card["priority"] in ["high", "medium", "low", "info"]
        assert len(card["title"]) > 5
        assert len(card["conclusion"]) > 10
        assert isinstance(card["supporting_data"], dict)
        assert len(card["recommended_action"]) > 10
        assert len(card["limitations"]) > 10


def test_api_insights_overview_endpoint():
    response = client.get("/api/insights/overview")
    assert response.status_code == 200
    data = response.json()
    assert "insight_cards" in data
    assert len(data["insight_cards"]) >= 4
    assert "trend_summary" in data
    assert "forecast_highlights" in data
    assert "inventory_risk_highlights" in data


def test_api_insights_demand_trends_endpoint():
    response = client.get("/api/insights/demand-trends?window_weeks=12")
    assert response.status_code == 200
    data = response.json()
    assert "trends" in data
    assert len(data["trends"]) == 33
    assert "methodology" in data


def test_api_insights_product_opportunities_endpoint():
    response = client.get("/api/insights/product-opportunities?model=naive_forecast")
    assert response.status_code == 200
    data = response.json()
    assert "top_demand_families" in data
    assert "model_divergence" in data
    assert "horizon_uncertainty" in data


def test_api_insights_inventory_risks_endpoint():
    response = client.get("/api/insights/inventory-risks")
    assert response.status_code == 200
    data = response.json()
    assert data["risk_counts"]["HIGH"] == 354
    assert "critical_vulnerable_items" in data
    assert len(data["critical_vulnerable_items"]) > 0
