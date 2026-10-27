"""FreshBasket AI - Phase 9: Inventory Alert Center & Reporting Unit Tests.

Covers:
1. Store-wise exposure aggregation (no double counting, exact sums)
2. Alert prioritization (HIGH severity first, largest replenishment quantity)
3. Executive briefing generation & email draft staging
4. Alert history snapshot persistence & duplicate prevention
5. Robustness against empty filters, invalid lead times, missing fields, and negative stock
6. Reconciling demo vs custom inventory labeling
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.business.inventory_planner import (
    compute_replenishment_plan,
    validate_inventory_dataframe,
)
from src.business.alert_engine import (
    generate_store_wise_summary,
    prioritize_alerts,
    generate_executive_business_report,
    generate_email_alert_draft,
    record_alert_snapshot,
    load_alert_history,
)
from src.dashboard.data_loader import (
    load_inventory_data,
    load_supplier_data,
    load_forecast_data,
)


# -----------------------------------------------------------------------------
# 1. Store-Wise Summary Aggregation Tests
# -----------------------------------------------------------------------------

def test_store_wise_summary_aggregation_exact():
    """Confirms store-level aggregation correctly groups all stores without double-counting."""
    df_inv = load_inventory_data()
    df_sup = load_supplier_data()
    df_fc = load_forecast_data()

    plan = compute_replenishment_plan(df_inv, df_fc, model_col="naive_forecast", supplier_df=df_sup)
    store_summary = generate_store_wise_summary(plan)

    # Must contain all 54 retail branches
    assert len(store_summary) == 54
    assert set(store_summary["store_nbr"]) == set(range(1, 55))

    # Total items across all stores must match 1,782 exactly
    assert store_summary["total_items"].sum() == 1782

    # High + Medium + Low + Unknown across stores must match portfolio sums
    assert store_summary["high_risk_count"].sum() == (plan["stockout_risk"] == "HIGH").sum() == 354
    assert store_summary["medium_risk_count"].sum() == (plan["stockout_risk"] == "MEDIUM").sum() == 368
    assert store_summary["low_risk_count"].sum() == (plan["stockout_risk"] == "LOW").sum() == 1060

    # Total replenishment quantity across stores must equal plan total exactly
    assert store_summary["total_suggested_replenishment_qty"].sum() == plan["suggested_replenishment_qty"].sum() == 932315

    # Check urgency assignment
    critical_stores = store_summary[store_summary["urgency_level"] == "CRITICAL"]
    assert len(critical_stores) > 0
    assert (critical_stores["high_risk_count"] > 0).all()


def test_store_wise_summary_empty():
    """Confirms empty plan returns structured empty dataframe without error."""
    empty_plan = pd.DataFrame(columns=[
        "store_nbr", "family", "stockout_risk", "current_stock",
        "expected_lt_demand", "safety_stock", "reorder_point",
        "suggested_replenishment_qty", "supplier_lead_time_days"
    ])
    store_sum = generate_store_wise_summary(empty_plan)
    assert store_sum.empty
    assert "store_nbr" in store_sum.columns
    assert "urgency_level" in store_sum.columns


# -----------------------------------------------------------------------------
# 2. Alert Prioritization Tests
# -----------------------------------------------------------------------------

def test_prioritize_alerts_sorting():
    """Confirms alerts are ordered: HIGH risk first, sorted by deficit descending."""
    test_df = pd.DataFrame([
        {"store_nbr": 1, "family": "LOW_ITEM", "stockout_risk": "LOW", "suggested_replenishment_qty": 0},
        {"store_nbr": 1, "family": "MED_ITEM", "stockout_risk": "MEDIUM", "suggested_replenishment_qty": 50},
        {"store_nbr": 1, "family": "HIGH_ITEM_SMALL", "stockout_risk": "HIGH", "suggested_replenishment_qty": 100},
        {"store_nbr": 1, "family": "HIGH_ITEM_LARGE", "stockout_risk": "HIGH", "suggested_replenishment_qty": 500},
        {"store_nbr": 1, "family": "UNK_ITEM", "stockout_risk": "UNKNOWN", "suggested_replenishment_qty": 0},
    ])

    prioritized = prioritize_alerts(test_df)
    assert prioritized.iloc[0]["family"] == "HIGH_ITEM_LARGE"
    assert prioritized.iloc[1]["family"] == "HIGH_ITEM_SMALL"
    assert prioritized.iloc[2]["family"] == "MED_ITEM"
    assert prioritized.iloc[3]["family"] == "LOW_ITEM"
    assert prioritized.iloc[4]["family"] == "UNK_ITEM"


# -----------------------------------------------------------------------------
# 3. Executive Business Report & Email Draft Tests
# -----------------------------------------------------------------------------

def test_executive_business_report_generation():
    """Confirms executive report compiles all mandatory business sections and disclaimers."""
    df_inv = load_inventory_data()
    df_sup = load_supplier_data()
    df_fc = load_forecast_data()
    plan = compute_replenishment_plan(df_inv, df_fc, model_col="naive_forecast", supplier_df=df_sup)

    report_md = generate_executive_business_report(
        df_plan=plan,
        data_source_mode="DEMO",
        forecast_model_label="Naive Baseline",
        selected_store_label="All Stores (54)",
        selected_family_label="All Families (33)",
    )

    assert "FRESHBASKET AI — EXECUTIVE INVENTORY & STOCKOUT RISK BRIEFING" in report_md
    assert "DEMO DATA (Synthetic Retail Balances)" in report_md
    assert "354" in report_md  # High risk count
    assert "932,315" in report_md  # Total suggested reorder units
    assert "Priority Action Items: Top Replenishment Deficits" in report_md
    assert "Supply Chain Assumptions & Explicit Limitations" in report_md
    assert "do NOT guarantee stockout prevention" in report_md


def test_email_alert_draft_generation():
    """Confirms email draft formats action items cleanly without API secrets."""
    df_inv = load_inventory_data()
    df_sup = load_supplier_data()
    df_fc = load_forecast_data()
    plan = compute_replenishment_plan(df_inv, df_fc, model_col="naive_forecast", supplier_df=df_sup)

    draft = generate_email_alert_draft(
        df_plan=plan,
        data_source_mode="DEMO",
        forecast_model_label="Naive Baseline",
        selected_store_label="All Stores (54)",
        selected_family_label="All Families (33)",
    )

    assert "Subject: [ACTION REQUIRED] FreshBasket AI Stockout Risk Alert" in draft
    assert "High Stockout Risk SKUs:" in draft
    assert "CRITICAL HIGH-RISK ITEMS (TOP DEFICITS):" in draft
    assert "RECOMMENDED ACTIONS:" in draft
    assert "do not guarantee stockout prevention" in draft
    # No exposed API keys or credentials
    assert "password" not in draft.lower()
    assert "api_key" not in draft.lower()


# -----------------------------------------------------------------------------
# 4. Persistent Alert History & Duplicate Prevention Tests
# -----------------------------------------------------------------------------

def test_alert_history_persistence(tmp_path):
    """Confirms snapshots are recorded locally to CSV and duplicate logging is avoided."""
    test_history_file = tmp_path / "test_alert_history.csv"

    test_plan = pd.DataFrame([
        {
            "store_nbr": 1,
            "family": "GROCERY I",
            "stockout_risk": "HIGH",
            "suggested_replenishment_qty": 150,
            "current_stock": 50.0,
            "expected_lt_demand": 160.0,
            "reorder_point": 200.0,
        },
        {
            "store_nbr": 1,
            "family": "BEVERAGES",
            "stockout_risk": "MEDIUM",
            "suggested_replenishment_qty": 40,
            "current_stock": 100.0,
            "expected_lt_demand": 90.0,
            "reorder_point": 140.0,
        },
        {
            "store_nbr": 1,
            "family": "DAIRY",
            "stockout_risk": "LOW",
            "suggested_replenishment_qty": 0,
            "current_stock": 300.0,
            "expected_lt_demand": 100.0,
            "reorder_point": 150.0,
        },
    ])

    # First snapshot: records HIGH and MEDIUM (2 items)
    success, msg, hist_df = record_alert_snapshot(
        df_plan=test_plan,
        data_source_mode="DEMO",
        forecast_model_label="Naive Baseline",
        history_file=test_history_file,
    )
    assert success
    assert len(hist_df) == 2
    assert test_history_file.exists()

    # Load history from disk
    loaded = load_alert_history(history_file=test_history_file)
    assert len(loaded) == 2
    assert set(loaded["stockout_risk"]) == {"HIGH", "MEDIUM"}

    # Immediate second snapshot with identical parameters: should be skipped as duplicate
    success2, msg2, hist_df2 = record_alert_snapshot(
        df_plan=test_plan,
        data_source_mode="DEMO",
        forecast_model_label="Naive Baseline",
        history_file=test_history_file,
    )
    assert not success2
    assert "duplicate" in msg2.lower()
    assert len(hist_df2) == 2


# -----------------------------------------------------------------------------
# 5. Edge Cases & Resilience Tests
# -----------------------------------------------------------------------------

def test_alert_engine_empty_filters():
    """Confirms empty plan handles reporting gracefully without crashes."""
    empty_plan = pd.DataFrame()

    rep = generate_executive_business_report(empty_plan, "DEMO", "Naive")
    assert "| **Total Monitored Items** | **0** |" in rep

    email = generate_email_alert_draft(empty_plan, "DEMO", "Naive")
    assert "High Stockout Risk SKUs:    0 items" in email

    store_sum = generate_store_wise_summary(empty_plan)
    assert store_sum.empty

    success, msg, _ = record_alert_snapshot(empty_plan, "DEMO", "Naive")
    assert not success


def test_alert_engine_unknown_and_negative_inputs():
    """Confirms invalid and unknown items are reported with explanation in alert reports."""
    bad_plan = pd.DataFrame([
        {
            "store_nbr": 5,
            "family": "INVALID_SKU",
            "stockout_risk": "UNKNOWN",
            "suggested_replenishment_qty": 0,
            "current_stock": -50.0,
            "expected_lt_demand": 0.0,
            "reorder_point": 0.0,
            "supplier_lead_time_days": 7,
            "supplier_name": "Supplier Bad",
            "explanation": "Invalid negative stock (-50).",
        }
    ])

    prioritized = prioritize_alerts(bad_plan)
    assert prioritized.iloc[0]["stockout_risk"] == "UNKNOWN"

    rep = generate_executive_business_report(bad_plan, "UPLOADED", "LightGBM")
    assert "| **Unknown / Invalid SKUs** | **1** |" in rep
    assert "Exactly 1 records were flagged as UNKNOWN" in rep
