"""
FreshBasket AI - What-If Scenario Simulator Engine
===================================================
Module providing deterministic scenario simulation on demand forecasts,
available inventory stock, and supplier lead times:

1. Demand scaling (-50% to +100%) applied across 13-week forward forecasts
2. Current stock scaling (-75% to +100%) with non-negativity constraint
3. Supplier lead-time adjustment (-7 to +30 days) with non-negativity constraint
4. Full recalculation of Lead-Time Demand, ROP, Suggested Replenishment, and Stockout Risk
5. Comparison metrics (units and percentage deltas) and summary narrative
"""

import math
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

from src.business.inventory_planner import (
    calculate_lead_time_demand,
    assess_stockout_risk,
)


def run_inventory_simulation(
    df_inventory: pd.DataFrame,
    df_forecast: pd.DataFrame,
    supplier_df: Optional[pd.DataFrame] = None,
    demand_change_pct: float = 0.0,
    stock_change_pct: float = 0.0,
    lead_time_adjustment_days: int = 0,
    store_nbr: Optional[int] = None,
    family: Optional[str] = None,
    model_col: str = "naive_forecast",
) -> Dict[str, Any]:
    """
    Executes a deterministic what-if scenario simulation without modifying source datasets.

    Parameters
    ----------
    df_inventory : pd.DataFrame
        Baseline inventory records (store_nbr, family, current_stock, safety_stock, supplier_lead_time_days).
    df_forecast : pd.DataFrame
        13-week forward predictions.
    supplier_df : Optional[pd.DataFrame]
        Optional supplier metadata.
    demand_change_pct : float
        Percentage change in demand (-50.0 to +100.0).
    stock_change_pct : float
        Percentage change in available stock (-75.0 to +100.0).
    lead_time_adjustment_days : int
        Days added or subtracted from supplier lead time (-7 to +30).
    store_nbr : Optional[int]
        Specific store filter (1-54), or None for all stores.
    family : Optional[str]
        Specific product family filter, or None for all families.
    model_col : str
        Forecast model column to use (default: "naive_forecast").

    Returns
    -------
    Dict[str, Any]
        Structured comparison between original and simulated inventory plans.
    """
    # 1. Parameter Validation
    if demand_change_pct < -50.0 or demand_change_pct > 100.0:
        raise ValueError(f"demand_change_pct must be between -50% and +100%, got {demand_change_pct}%")
    if stock_change_pct < -75.0 or stock_change_pct > 100.0:
        raise ValueError(f"stock_change_pct must be between -75% and +100%, got {stock_change_pct}%")
    if lead_time_adjustment_days < -7 or lead_time_adjustment_days > 30:
        raise ValueError(f"lead_time_adjustment_days must be between -7 and +30 days, got {lead_time_adjustment_days}")

    if df_inventory.empty or df_forecast.empty:
        raise ValueError("Inventory or forecast data is missing or empty.")

    # 2. Filter working copies
    inv_work = df_inventory.copy()
    if store_nbr is not None:
        inv_work = inv_work[inv_work["store_nbr"] == int(store_nbr)]
    if family is not None:
        inv_work = inv_work[inv_work["family"] == str(family)]

    if inv_work.empty:
        raise ValueError(
            f"No inventory records found for store_nbr={store_nbr}, family='{family}'."
        )

    # Filter forecasts
    fc_work = df_forecast.copy()
    if store_nbr is not None:
        fc_work = fc_work[fc_work["store_nbr"] == int(store_nbr)]
    if family is not None:
        fc_work = fc_work[fc_work["family"] == str(family)]

    # Validate model column
    if model_col not in fc_work.columns:
        model_col = "naive_forecast" if "naive_forecast" in fc_work.columns else fc_work.columns[-1]

    # Map forecast series by (store_nbr, family) -> list of 13 weekly values
    fc_sorted = fc_work.sort_values(["store_nbr", "family", "forecast_horizon_week"])
    forecast_map = (
        fc_sorted.groupby(["store_nbr", "family"])[model_col]
        .apply(list)
        .to_dict()
    )

    demand_multiplier = 1.0 + (demand_change_pct / 100.0)
    stock_multiplier = 1.0 + (stock_change_pct / 100.0)

    item_results = []

    orig_total_fc = 0.0
    sim_total_fc = 0.0
    orig_total_stock = 0.0
    sim_total_stock = 0.0
    orig_total_order = 0
    sim_total_order = 0

    orig_risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}
    sim_risk_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "UNKNOWN": 0}

    total_lt_orig_days = 0
    total_lt_sim_days = 0
    valid_items_count = 0

    # 3. Item-level Simulation
    for _, row in inv_work.iterrows():
        s_id = int(row["store_nbr"])
        fam = str(row["family"])
        raw_stock = row.get("current_stock")
        raw_safety = row.get("safety_stock")
        raw_lt = row.get("supplier_lead_time_days")

        if pd.isna(raw_stock) or pd.isna(raw_safety):
            continue

        orig_stock = float(raw_stock)
        safety_stock = float(raw_safety)
        orig_lt = int(raw_lt) if (raw_lt is not None and not pd.isna(raw_lt)) else 7

        # Apply simulation transformations with non-negativity constraints
        sim_stock = max(0.0, orig_stock * stock_multiplier)
        sim_lt = max(0, orig_lt + lead_time_adjustment_days)

        fc_series = forecast_map.get((s_id, fam), [])
        if not fc_series:
            continue

        valid_items_count += 1
        orig_series_sum = sum(fc_series)
        orig_total_fc += orig_series_sum
        orig_total_stock += orig_stock
        total_lt_orig_days += orig_lt
        total_lt_sim_days += sim_lt

        # Simulated demand series
        sim_series = [max(0.0, float(v) * demand_multiplier) for v in fc_series]
        sim_series_sum = sum(sim_series)
        sim_total_fc += sim_series_sum
        sim_total_stock += sim_stock

        # Original calculations
        orig_lt_demand, _ = calculate_lead_time_demand(fc_series, orig_lt)
        orig_rop = orig_lt_demand + safety_stock
        orig_suggested = math.ceil(max(0.0, orig_rop - orig_stock))
        orig_risk, orig_exp = assess_stockout_risk(orig_stock, orig_lt_demand, safety_stock, orig_rop)
        orig_total_order += orig_suggested
        orig_risk_counts[orig_risk] = orig_risk_counts.get(orig_risk, 0) + 1

        # Simulated calculations
        sim_lt_demand, sim_warning = calculate_lead_time_demand(sim_series, sim_lt)
        sim_rop = sim_lt_demand + safety_stock
        sim_suggested = math.ceil(max(0.0, sim_rop - sim_stock))
        sim_risk, sim_exp = assess_stockout_risk(sim_stock, sim_lt_demand, safety_stock, sim_rop)
        sim_total_order += sim_suggested
        sim_risk_counts[sim_risk] = sim_risk_counts.get(sim_risk, 0) + 1

        item_results.append({
            "store_nbr": s_id,
            "family": fam,
            "orig_demand": round(orig_series_sum, 1),
            "sim_demand": round(sim_series_sum, 1),
            "orig_stock": round(orig_stock, 1),
            "sim_stock": round(sim_stock, 1),
            "orig_lead_time": orig_lt,
            "adjusted_lead_time": sim_lt,
            "orig_suggested_order": orig_suggested,
            "sim_suggested_order": sim_suggested,
            "order_diff": sim_suggested - orig_suggested,
            "orig_risk": orig_risk,
            "sim_risk": sim_risk,
            "risk_escalated": (orig_risk != "HIGH" and sim_risk == "HIGH"),
        })

    if valid_items_count == 0:
        raise ValueError("No valid item records with forecast demand could be evaluated.")

    # 4. Averages and Differences
    avg_orig_lt = round(total_lt_orig_days / valid_items_count, 1)
    avg_sim_lt = round(total_lt_sim_days / valid_items_count, 1)

    demand_diff_units = sim_total_fc - orig_total_fc
    demand_diff_pct = (demand_diff_units / orig_total_fc * 100.0) if orig_total_fc > 0 else 0.0

    stock_diff_units = sim_total_stock - orig_total_stock
    stock_diff_pct = (stock_diff_units / orig_total_stock * 100.0) if orig_total_stock > 0 else 0.0

    order_diff_units = sim_total_order - orig_total_order
    order_diff_pct = (order_diff_units / orig_total_order * 100.0) if orig_total_order > 0 else 0.0

    high_risk_diff = sim_risk_counts["HIGH"] - orig_risk_counts["HIGH"]

    # Primary risk label for single vs aggregate
    if valid_items_count == 1:
        orig_primary_risk = item_results[0]["orig_risk"]
        sim_primary_risk = item_results[0]["sim_risk"]
    else:
        orig_primary_risk = "HIGH" if orig_risk_counts["HIGH"] > 0 else ("MEDIUM" if orig_risk_counts["MEDIUM"] > 0 else "LOW")
        sim_primary_risk = "HIGH" if sim_risk_counts["HIGH"] > 0 else ("MEDIUM" if sim_risk_counts["MEDIUM"] > 0 else "LOW")

    # 5. Summary Highlight Construction
    highlights = []
    if abs(demand_diff_units) > 0:
        d_dir = "increase" if demand_diff_units > 0 else "decrease"
        highlights.append(f"Demand shift of {demand_change_pct:+.1f}% creates a {abs(demand_diff_units):,.0f} unit ({d_dir}) in 13-week forecast.")
    if abs(stock_diff_units) > 0:
        s_dir = "expands" if stock_diff_units > 0 else "reduces"
        highlights.append(f"Stock adjustment of {stock_change_pct:+.1f}% {s_dir} available inventory by {abs(stock_diff_units):,.0f} units.")
    if lead_time_adjustment_days != 0:
        lt_dir = "elongation" if lead_time_adjustment_days > 0 else "shortening"
        highlights.append(f"Supplier lead-time {lt_dir} of {lead_time_adjustment_days:+d} days alters replenishment horizon.")

    if order_diff_units > 0:
        highlights.append(f"Replenishment orders increase by +{order_diff_units:,.0f} units (+{order_diff_pct:.1f}%).")
    elif order_diff_units < 0:
        highlights.append(f"Replenishment orders decrease by {order_diff_units:,.0f} units ({order_diff_pct:.1f}%).")
    else:
        highlights.append("Suggested replenishment orders remain unchanged.")

    if high_risk_diff > 0:
        highlights.append(f"⚠️ {high_risk_diff} additional items escalate to HIGH stockout risk.")
    elif high_risk_diff < 0:
        highlights.append(f"✅ High-risk stockouts decrease by {abs(high_risk_diff)} items.")

    summary_highlight = " ".join(highlights) if highlights else "Baseline scenario evaluated. Simulated plan matches original plan exactly."

    # 6. Weekly Aggregated Trajectory
    # Group fc_work across all items by forecast_horizon_week
    weekly_orig = fc_work.groupby("forecast_horizon_week")[model_col].sum()
    weekly_trajectory = []
    for week_num, orig_w_demand in weekly_orig.items():
        sim_w_demand = max(0.0, float(orig_w_demand) * demand_multiplier)
        weekly_trajectory.append({
            "week": int(week_num),
            "original_demand": round(float(orig_w_demand), 1),
            "simulated_demand": round(float(sim_w_demand), 1),
            "difference": round(sim_w_demand - float(orig_w_demand), 1),
        })

    # Sort item breakdown by largest order difference
    item_results.sort(key=lambda x: abs(x["order_diff"]), reverse=True)

    return {
        "success": True,
        "parameters": {
            "demand_change_pct": demand_change_pct,
            "stock_change_pct": stock_change_pct,
            "lead_time_adjustment_days": lead_time_adjustment_days,
            "store_nbr": store_nbr,
            "family": family,
            "model": model_col,
        },
        "metrics": {
            "original_forecast_demand": round(orig_total_fc, 1),
            "simulated_forecast_demand": round(sim_total_fc, 1),
            "demand_diff_units": round(demand_diff_units, 1),
            "demand_diff_pct": round(demand_diff_pct, 2),

            "original_available_stock": round(orig_total_stock, 1),
            "simulated_available_stock": round(sim_total_stock, 1),
            "stock_diff_units": round(stock_diff_units, 1),
            "stock_diff_pct": round(stock_diff_pct, 2),

            "original_lead_time": avg_orig_lt,
            "adjusted_lead_time": avg_sim_lt,
            "lead_time_adjustment_days": lead_time_adjustment_days,

            "original_suggested_order_qty": int(orig_total_order),
            "simulated_suggested_order_qty": int(sim_total_order),
            "order_qty_diff_units": int(order_diff_units),
            "order_qty_diff_pct": round(order_diff_pct, 2),

            "original_stockout_risk": orig_primary_risk,
            "simulated_stockout_risk": sim_primary_risk,
            "original_risk_counts": orig_risk_counts,
            "simulated_risk_counts": sim_risk_counts,
            "high_risk_diff": high_risk_diff,

            "items_evaluated": valid_items_count,
        },
        "summary_highlight": summary_highlight,
        "weekly_trajectory": weekly_trajectory,
        "top_affected_items": item_results[:50],  # Top 50 items by order impact
        "assumptions": [
            "Demand percentage scaling is uniformly applied across all 13 weekly forecast steps.",
            "Simulated stock cannot drop below 0 units (non-negativity constraint enforced).",
            "Adjusted supplier lead time cannot drop below 0 days.",
            "Safety stock thresholds are held constant according to active FMCG policy.",
            "Suggested order quantities use deterministic ceiling discrete rounding: ceil(max(0, ROP - Stock)).",
            "Scenario calculations are non-destructive and do not alter source CSV or model files.",
        ],
    }
