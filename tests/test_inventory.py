"""FreshBasket AI - Phase 8: Inventory Planning & Stockout Risk Unit Tests.

Covers:
1. Expected Lead-Time Demand calculation (proportional < 7 days, multi-week aggregation, >91-day horizon warning)
2. Stockout Risk Assessment (HIGH, MEDIUM, LOW, UNKNOWN) with transparent explanations
3. Replenishment Recommendations & discrete integer ceiling rounding
4. Data validation (missing columns, duplicates, negative stocks, invalid lead times)
5. Edge cases: zero stock, zero safety stock, empty filters, and missing forecasts
6. Integration with demo datasets and dashboard loaders
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.business.inventory_planner import (
    calculate_lead_time_demand,
    assess_stockout_risk,
    compute_replenishment_plan,
    validate_inventory_dataframe,
    generate_inventory_business_summary,
)
from src.dashboard.data_loader import (
    load_inventory_data,
    load_supplier_data,
    load_forecast_data,
)
from src.dashboard.charts import (
    create_stockout_risk_pie,
    create_highest_risk_items_bar,
    create_demand_vs_stock_scatter,
)


# -----------------------------------------------------------------------------
# 1. Expected Lead-Time Demand Tests
# -----------------------------------------------------------------------------

def test_lead_time_demand_short_proportional():
    """Confirms lead time <= 7 days uses proportional daily rate from week 1 forecast."""
    # 13 weeks of forecast demand = [700, 700, 700, ...]
    forecast = [700.0] * 13

    # 1 day lead time: (700 / 7) * 1 = 100
    demand_1d, warning_1d = calculate_lead_time_demand(forecast, lead_time_days=1)
    assert math.isclose(demand_1d, 100.0, rel_tol=1e-5)
    assert warning_1d is None

    # 3 days lead time: (700 / 7) * 3 = 300
    demand_3d, warning_3d = calculate_lead_time_demand(forecast, lead_time_days=3)
    assert math.isclose(demand_3d, 300.0, rel_tol=1e-5)
    assert warning_3d is None

    # 7 days lead time: exactly 1 week = 700
    demand_7d, warning_7d = calculate_lead_time_demand(forecast, lead_time_days=7)
    assert math.isclose(demand_7d, 700.0, rel_tol=1e-5)
    assert warning_7d is None


def test_lead_time_demand_multi_week_aggregation():
    """Confirms lead times spanning multiple weeks sum full weeks + partial week remainder."""
    # Week 1: 700, Week 2: 1400, Week 3: 2100...
    forecast = [700.0, 1400.0, 2100.0] + [500.0] * 10

    # 10 days = 1 full week (700) + 3 days of Week 2 (3/7 * 1400 = 600) -> 1300.0
    demand_10d, warning_10d = calculate_lead_time_demand(forecast, lead_time_days=10)
    assert math.isclose(demand_10d, 1300.0, rel_tol=1e-5)
    assert warning_10d is None

    # 14 days = 2 full weeks (700 + 1400 = 2100)
    demand_14d, warning_14d = calculate_lead_time_demand(forecast, lead_time_days=14)
    assert math.isclose(demand_14d, 2100.0, rel_tol=1e-5)
    assert warning_14d is None


def test_lead_time_demand_horizon_overflow_warning():
    """Confirms lead times > 91 days (13 weeks) return a warning and cap demand at 13-week sum."""
    forecast = [100.0] * 13  # total 13-week demand = 1300
    demand_over, warning_over = calculate_lead_time_demand(forecast, lead_time_days=100)

    assert math.isclose(demand_over, 1300.0, rel_tol=1e-5)
    assert warning_over is not None
    assert "exceeds available 13-week forecast horizon" in warning_over


def test_lead_time_demand_zero_or_empty():
    """Confirms zero lead time or empty forecast returns 0 demand without error."""
    demand, warning = calculate_lead_time_demand([], lead_time_days=7)
    assert demand == 0.0
    assert warning is None

    demand_zero_days, warning_zero_days = calculate_lead_time_demand([100.0] * 13, lead_time_days=0)
    assert demand_zero_days == 0.0
    assert warning_zero_days is None


# -----------------------------------------------------------------------------
# 2. Stockout Risk Assessment Tests
# -----------------------------------------------------------------------------

def test_assess_stockout_risk_tiers():
    """Confirms deterministic classification for HIGH, MEDIUM, LOW, and UNKNOWN."""
    # HIGH: inventory_position < expected_lt_demand
    tier, reason = assess_stockout_risk(
        inventory_position=200.0,
        expected_lt_demand=500.0,
        safety_stock=100.0,
        reorder_point=600.0,
    )
    assert tier == "HIGH"
    assert "below expected lead-time demand" in reason

    # MEDIUM: inventory_position covers lt_demand but < reorder_point
    tier, reason = assess_stockout_risk(
        inventory_position=550.0,
        expected_lt_demand=500.0,
        safety_stock=100.0,
        reorder_point=600.0,
    )
    assert tier == "MEDIUM"
    assert "penetrates safety buffer" in reason

    # LOW: inventory_position >= reorder_point
    tier, reason = assess_stockout_risk(
        inventory_position=650.0,
        expected_lt_demand=500.0,
        safety_stock=100.0,
        reorder_point=600.0,
    )
    assert tier == "LOW"
    assert "at or above reorder point" in reason


def test_assess_stockout_risk_unknown_guards():
    """Confirms invalid negative values and NaNs trigger UNKNOWN risk level."""
    # Negative stock
    tier, reason = assess_stockout_risk(-50.0, 100.0, 50.0, 150.0)
    assert tier == "UNKNOWN"
    assert "Invalid negative" in reason

    # NaN input
    tier, reason = assess_stockout_risk(float("nan"), 100.0, 50.0, 150.0)
    assert tier == "UNKNOWN"
    assert "Missing or incomplete" in reason


# -----------------------------------------------------------------------------
# 3. Replenishment Recommendations & Rounding Tests
# -----------------------------------------------------------------------------

def test_replenishment_discrete_rounding():
    """Confirms suggested replenishment quantities are rounded up to whole units."""
    # Simulated df
    df_inv = pd.DataFrame([{
        "store_nbr": 1,
        "family": "GROCERY I",
        "current_stock": 100.0,
        "safety_stock": 25.0,
        "supplier_lead_time_days": 7,
    }])
    # Forecast week 1 = 120.4 units -> ROP = 120.4 + 25 = 145.4
    # Raw deficit = 145.4 - 100.0 = 45.4 -> math.ceil(45.4) = 46 units
    df_fc = pd.DataFrame([
        {
            "store_nbr": 1,
            "family": "GROCERY I",
            "forecast_horizon_week": w,
            "lightgbm_forecast": 120.4,
        }
        for w in range(1, 14)
    ])

    plan = compute_replenishment_plan(df_inv, df_fc, model_col="lightgbm_forecast")
    assert len(plan) == 1
    assert plan.iloc[0]["suggested_replenishment_qty"] == 46
    assert isinstance(plan.iloc[0]["suggested_replenishment_qty"], (int, np.integer))


def test_replenishment_zero_when_surplus():
    """Confirms suggested replenishment quantity is 0 when inventory exceeds ROP."""
    df_inv = pd.DataFrame([{
        "store_nbr": 1,
        "family": "BEVERAGES",
        "current_stock": 1000.0,
        "safety_stock": 100.0,
        "supplier_lead_time_days": 7,
    }])
    df_fc = pd.DataFrame([
        {
            "store_nbr": 1,
            "family": "BEVERAGES",
            "forecast_horizon_week": w,
            "lightgbm_forecast": 200.0,
        }
        for w in range(1, 14)
    ])

    plan = compute_replenishment_plan(df_inv, df_fc, model_col="lightgbm_forecast")
    assert plan.iloc[0]["suggested_replenishment_qty"] == 0
    assert plan.iloc[0]["stockout_risk"] == "LOW"


# -----------------------------------------------------------------------------
# 4. Inventory Dataframe Validation Tests
# -----------------------------------------------------------------------------

def test_validate_inventory_dataframe_valid():
    """Confirms valid dataframe passes validation clean without errors."""
    valid_df = pd.DataFrame({
        "store_nbr": [1, 2],
        "family": ["BREAD", "DAIRY"],
        "current_stock": [150.0, 300.0],
        "safety_stock": [30.0, 50.0],
        "supplier_lead_time_days": [7, 10],
    })
    is_valid, errors, clean_df = validate_inventory_dataframe(valid_df)
    assert is_valid
    assert len(errors) == 0
    assert len(clean_df) == 2


def test_validate_inventory_dataframe_missing_columns():
    """Confirms missing mandatory columns are reported clearly."""
    bad_df = pd.DataFrame({
        "store_nbr": [1],
        "family": ["BREAD"],
        # Missing current_stock and safety_stock
    })
    is_valid, errors, _ = validate_inventory_dataframe(bad_df)
    assert not is_valid
    assert any("Missing mandatory columns" in e for e in errors)


def test_validate_inventory_dataframe_duplicates():
    """Confirms duplicate store-family keys are detected."""
    dup_df = pd.DataFrame({
        "store_nbr": [1, 1],
        "family": ["BREAD", "BREAD"],
        "current_stock": [100.0, 200.0],
        "safety_stock": [20.0, 20.0],
    })
    is_valid, errors, _ = validate_inventory_dataframe(dup_df)
    assert not is_valid
    assert any("duplicate" in e.lower() for e in errors)


def test_validate_inventory_dataframe_negative_values():
    """Confirms negative current stock or safety stock fails validation."""
    neg_df = pd.DataFrame({
        "store_nbr": [1],
        "family": ["BREAD"],
        "current_stock": [-10.0],
        "safety_stock": [20.0],
    })
    is_valid, errors, _ = validate_inventory_dataframe(neg_df)
    assert not is_valid
    assert any("Negative values found in 'current_stock'" in e for e in errors)


def test_validate_inventory_dataframe_missing_lead_time_fallback():
    """Confirms missing supplier_lead_time_days defaults to 7 days."""
    df_no_lt = pd.DataFrame({
        "store_nbr": [1],
        "family": ["BREAD"],
        "current_stock": [100.0],
        "safety_stock": [20.0],
    })
    is_valid, errors, clean_df = validate_inventory_dataframe(df_no_lt)
    assert is_valid
    assert clean_df.iloc[0]["supplier_lead_time_days"] == 7


# -----------------------------------------------------------------------------
# 5. Integration with Verified Files & Loaders
# -----------------------------------------------------------------------------

def test_inventory_files_and_loaders():
    """Confirms inventory and supplier files exist and load via data_loader."""
    df_inv = load_inventory_data()
    df_sup = load_supplier_data()

    assert not df_inv.empty
    assert len(df_inv) == 1782  # 54 stores * 33 families
    assert set(["store_nbr", "family", "current_stock", "safety_stock", "supplier_lead_time_days"]).issubset(df_inv.columns)

    assert not df_sup.empty
    assert len(df_sup) == 1782
    assert set(["store_nbr", "family", "supplier_lead_time_days", "supplier_name"]).issubset(df_sup.columns)


def test_full_replenishment_pipeline():
    """Runs full replenishment computation with live demo inventory and forecast datasets."""
    df_inv = load_inventory_data()
    df_sup = load_supplier_data()
    df_fc = load_forecast_data()

    plan = compute_replenishment_plan(
        df_inventory=df_inv,
        df_forecast=df_fc,
        model_col="naive_forecast",
        supplier_df=df_sup,
    )

    assert len(plan) == 1782
    assert "stockout_risk" in plan.columns
    assert "suggested_replenishment_qty" in plan.columns
    assert "explanation" in plan.columns

    # Verify risk categories present
    risk_tiers = set(plan["stockout_risk"].unique())
    assert "HIGH" in risk_tiers
    assert "MEDIUM" in risk_tiers
    assert "LOW" in risk_tiers

    # Verify no NaN values in calculated numeric metrics
    assert not plan["expected_lt_demand"].isnull().any()
    assert not plan["reorder_point"].isnull().any()
    assert not plan["suggested_replenishment_qty"].isnull().any()


# -----------------------------------------------------------------------------
# 6. Edge Cases & Resilience Tests
# -----------------------------------------------------------------------------

def test_zero_stock_and_zero_safety_stock():
    """Tests extreme edge cases: 0 stock and 0 safety buffer."""
    df_inv = pd.DataFrame([{
        "store_nbr": 1,
        "family": "ZERO_TEST",
        "current_stock": 0.0,
        "safety_stock": 0.0,
        "supplier_lead_time_days": 7,
    }])
    df_fc = pd.DataFrame([
        {
            "store_nbr": 1,
            "family": "ZERO_TEST",
            "forecast_horizon_week": w,
            "lightgbm_forecast": 50.0,
        }
        for w in range(1, 14)
    ])

    plan = compute_replenishment_plan(df_inv, df_fc, model_col="lightgbm_forecast")
    # Expected demand = 50.0, ROP = 50.0 + 0 = 50.0
    # Suggested order = ceil(50.0 - 0) = 50
    # Risk = HIGH (0 < 50)
    assert plan.iloc[0]["expected_lt_demand"] == 50.0
    assert plan.iloc[0]["reorder_point"] == 50.0
    assert plan.iloc[0]["suggested_replenishment_qty"] == 50
    assert plan.iloc[0]["stockout_risk"] == "HIGH"


def test_empty_filters_and_summary():
    """Confirms empty plan gracefully generates empty summaries and charts without crashes."""
    empty_df = pd.DataFrame(columns=[
        "store_nbr", "family", "stockout_risk", "current_stock",
        "expected_lt_demand", "safety_stock", "reorder_point",
        "suggested_replenishment_qty", "supplier_lead_time_days"
    ])

    summary = generate_inventory_business_summary(empty_df)
    assert len(summary) == 1
    assert "No inventory records match" in summary[0]

    # Chart generators handle empty dataframes safely
    fig_pie = create_stockout_risk_pie(empty_df)
    assert fig_pie is not None

    fig_bar = create_highest_risk_items_bar(empty_df)
    assert fig_bar is not None

    fig_scatter = create_demand_vs_stock_scatter(empty_df)
    assert fig_scatter is not None


def test_replenishment_risk_counts_reconciliation():
    """Verifies that High (354) + Medium (368) + Low (1,060) reconcile exactly to 1,782 items."""
    df_inv = load_inventory_data()
    df_sup = load_supplier_data()
    df_fc = load_forecast_data()

    plan = compute_replenishment_plan(df_inv, df_fc, model_col="naive_forecast", supplier_df=df_sup)

    high_cnt = (plan["stockout_risk"] == "HIGH").sum()
    med_cnt = (plan["stockout_risk"] == "MEDIUM").sum()
    low_cnt = (plan["stockout_risk"] == "LOW").sum()
    unknown_cnt = (plan["stockout_risk"] == "UNKNOWN").sum()

    assert high_cnt == 354
    assert med_cnt == 368
    assert low_cnt == 1060
    assert unknown_cnt == 0
    assert (high_cnt + med_cnt + low_cnt + unknown_cnt) == len(plan) == 1782

    # Check baseline totals
    assert plan["suggested_replenishment_qty"].sum() == 932315
    assert round(plan["expected_lt_demand"].sum()) == 3482969


def test_missing_and_negative_inventory_values_rejected():
    """Confirms missing values are NOT silently zeroed, and negative stocks/lead times are rejected as UNKNOWN."""
    # Data with missing values and negative values
    test_df = pd.DataFrame([
        {
            "store_nbr": 1,
            "family": "MISSING_STOCK",
            "current_stock": np.nan,  # Missing
            "safety_stock": 50.0,
            "supplier_lead_time_days": 7,
        },
        {
            "store_nbr": 1,
            "family": "NEGATIVE_STOCK",
            "current_stock": -100.0,  # Negative
            "safety_stock": 50.0,
            "supplier_lead_time_days": 7,
        },
        {
            "store_nbr": 1,
            "family": "INVALID_LEAD_TIME",
            "current_stock": 200.0,
            "safety_stock": 50.0,
            "supplier_lead_time_days": 0,  # Invalid
        },
    ])

    df_fc = pd.DataFrame([
        {
            "store_nbr": 1,
            "family": fam,
            "forecast_horizon_week": w,
            "naive_forecast": 100.0,
        }
        for fam in ["MISSING_STOCK", "NEGATIVE_STOCK", "INVALID_LEAD_TIME"]
        for w in range(1, 14)
    ])

    plan = compute_replenishment_plan(test_df, df_fc, model_col="naive_forecast")
    assert len(plan) == 3

    # All 3 invalid rows must be flagged as UNKNOWN with 0 suggested order
    for _, row in plan.iterrows():
        assert row["stockout_risk"] == "UNKNOWN"
        assert row["suggested_replenishment_qty"] == 0
        assert "rejected" in row["explanation"].lower() or "cannot be negative" in row["explanation"].lower()


def test_inventory_filter_reactivity():
    """Confirms filtering by store or family recalculates aggregated totals consistently."""
    df_inv = load_inventory_data()
    df_sup = load_supplier_data()
    df_fc = load_forecast_data()

    plan = compute_replenishment_plan(df_inv, df_fc, model_col="naive_forecast", supplier_df=df_sup)

    # Filter to Store 1 only (33 families)
    store_1_plan = plan[plan["store_nbr"] == 1]
    assert len(store_1_plan) == 33
    risk_sum = (store_1_plan["stockout_risk"] == "HIGH").sum() + \
               (store_1_plan["stockout_risk"] == "MEDIUM").sum() + \
               (store_1_plan["stockout_risk"] == "LOW").sum()
    assert risk_sum == 33
    assert store_1_plan["suggested_replenishment_qty"].sum() < plan["suggested_replenishment_qty"].sum()

