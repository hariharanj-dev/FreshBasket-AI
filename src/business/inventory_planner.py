"""
FreshBasket AI - Inventory Planning & Stockout Risk Calculation Engine
======================================================================
Module providing deterministic, explainable replenishment heuristics:
1. Expected Lead-Time Demand calculation across 13-week multi-step forecasts
2. Reorder Point (ROP) & Suggested Replenishment Quantity calculation
3. Stockout Risk Classification (HIGH, MEDIUM, LOW, UNKNOWN) with transparent audit reasons
4. Uploaded inventory data validation and sanity checking
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
INVENTORY_FILE = WORKSPACE_ROOT / "data" / "inventory" / "inventory.csv"
SUPPLIER_FILE = WORKSPACE_ROOT / "data" / "inventory" / "supplier_lead_times.csv"


def calculate_lead_time_demand(
    forecast_series: List[float],
    lead_time_days: int,
) -> Tuple[float, Optional[str]]:
    """
    Calculates expected demand over a supplier replenishment lead time.
    
    Parameters
    ----------
    forecast_series : List[float]
        Chronological list of 13 weekly forecast demand values (weeks t+1 to t+13).
    lead_time_days : int
        Replenishment lead time in days.

    Returns
    -------
    Tuple[float, Optional[str]]
        (expected_demand, horizon_warning_message)
    """
    if lead_time_days <= 0 or not forecast_series:
        return 0.0, None

    max_horizon_days = len(forecast_series) * 7
    warning_msg = None

    if lead_time_days > max_horizon_days:
        # Lead time exceeds available 13-week horizon (91 days)
        warning_msg = (
            f"Supplier lead time ({lead_time_days} days) exceeds available 13-week "
            f"forecast horizon ({max_horizon_days} days). Demand capped at total 13-week horizon."
        )
        total_demand = sum(forecast_series)
        return float(total_demand), warning_msg

    if lead_time_days <= 7:
        # Lead time shorter than or equal to one week (proportional approximation)
        # Assumes daily sales are uniformly distributed across the week
        daily_rate = forecast_series[0] / 7.0
        expected_demand = daily_rate * lead_time_days
        return float(expected_demand), None

    # Lead time spans multiple weeks
    full_weeks = lead_time_days // 7
    remaining_days = lead_time_days % 7

    demand = sum(forecast_series[:full_weeks])

    if remaining_days > 0 and full_weeks < len(forecast_series):
        # Add proportional partial week from the subsequent forecast week
        daily_rate_next = forecast_series[full_weeks] / 7.0
        demand += daily_rate_next * remaining_days

    return float(demand), None


def assess_stockout_risk(
    inventory_position: float,
    expected_lt_demand: float,
    safety_stock: float,
    reorder_point: float,
) -> Tuple[str, str]:
    """
    Determines stockout risk level and provides a transparent plain-language explanation.

    Risk Tiers:
    - HIGH: Inventory position is below expected lead-time demand.
    - MEDIUM: Inventory position covers lead-time demand but breaches safety stock buffer.
    - LOW: Inventory position is at or above the reorder point.
    - UNKNOWN: Required values are missing, NaN, or invalid negative numbers.
    """
    # Validation guard for missing or invalid values
    if any(v is None or math.isnan(v) for v in [inventory_position, expected_lt_demand, safety_stock, reorder_point]):
        return "UNKNOWN", "Missing or incomplete inventory/forecast parameters."

    if inventory_position < 0 or expected_lt_demand < 0 or safety_stock < 0:
        return "UNKNOWN", "Invalid negative parameters detected in stock or safety calculation."

    if inventory_position < expected_lt_demand:
        deficit = expected_lt_demand - inventory_position
        return (
            "HIGH",
            f"Stock ({inventory_position:,.0f}) is below expected lead-time demand ({expected_lt_demand:,.0f}). "
            f"Projected stockout deficit of {deficit:,.0f} units before shipment arrives.",
        )

    if inventory_position < reorder_point:
        safety_deficit = reorder_point - inventory_position
        return (
            "MEDIUM",
            f"Stock ({inventory_position:,.0f}) covers lead-time demand ({expected_lt_demand:,.0f}) "
            f"but penetrates safety buffer by {safety_deficit:,.0f} units. Reorder recommended.",
        )

    buffer_excess = inventory_position - reorder_point
    return (
        "LOW",
        f"Stock ({inventory_position:,.0f}) is at or above reorder point ({reorder_point:,.0f}). "
        f"Safety buffer has {buffer_excess:,.0f} units of surplus capacity.",
    )


def compute_replenishment_plan(
    df_inventory: pd.DataFrame,
    df_forecast: pd.DataFrame,
    model_col: str = "lightgbm_forecast",
    supplier_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Computes baseline replenishment recommendations for all store-family items.

    Assumptions & Formulas:
    -----------------------
    1. Inventory Position = Current Stock (committed demand and incoming shipments are excluded).
    2. Expected Lead-Time Demand = Sum of forecast demand over the supplier lead time.
    3. Reorder Point (ROP) = Expected Lead-Time Demand + Safety Stock.
    4. Suggested Replenishment Quantity = ceil(max(0, ROP - Inventory Position)).
    5. Baseline Heuristic: Does not claim to be a cost-optimized (s, S) or (R, Q) stochastic policy.
    """
    if df_inventory.empty or df_forecast.empty:
        return pd.DataFrame()

    # Merge supplier lead times if supplier_df provided and not already in inventory
    inv_work = df_inventory.copy()
    if supplier_df is not None and "supplier_name" in supplier_df.columns:
        if "supplier_name" not in inv_work.columns:
            inv_work = inv_work.merge(
                supplier_df[["store_nbr", "family", "supplier_name"]],
                on=["store_nbr", "family"],
                how="left",
            )

    # Index forecasts by (store_nbr, family) -> list of 13 weekly forecast values
    fc_sorted = df_forecast.sort_values(["store_nbr", "family", "forecast_horizon_week"])
    forecast_map = (
        fc_sorted.groupby(["store_nbr", "family"])[model_col]
        .apply(list)
        .to_dict()
    )

    results = []

    for _, row in inv_work.iterrows():
        store_nbr = int(row["store_nbr"])
        family = str(row["family"])
        raw_stock = row.get("current_stock")
        raw_safety = row.get("safety_stock")
        raw_lt = row.get("supplier_lead_time_days")
        supplier_name = str(row.get("supplier_name", "Standard Supplier"))
        last_updated = str(row.get("last_updated", "N/A"))

        # Guard 1: Missing values (do NOT silently convert to zero)
        if raw_stock is None or pd.isna(raw_stock) or raw_safety is None or pd.isna(raw_safety):
            results.append({
                "store_nbr": store_nbr,
                "family": family,
                "current_stock": np.nan,
                "safety_stock": np.nan,
                "supplier_lead_time_days": int(raw_lt) if (raw_lt is not None and not pd.isna(raw_lt) and raw_lt > 0) else 7,
                "supplier_name": supplier_name,
                "last_updated": last_updated,
                "expected_lt_demand": 0.0,
                "reorder_point": 0.0,
                "suggested_replenishment_qty": 0,
                "stockout_risk": "UNKNOWN",
                "explanation": "Missing required inventory values (current_stock or safety_stock). Calculation rejected.",
                "horizon_warning": None,
            })
            continue

        current_stock = float(raw_stock)
        safety_stock = float(raw_safety)
        lead_time = int(raw_lt) if (raw_lt is not None and not pd.isna(raw_lt)) else 7

        # Guard 2: Negative stock or safety buffer or non-positive lead time
        if current_stock < 0 or safety_stock < 0 or lead_time <= 0:
            invalid_reasons = []
            if current_stock < 0:
                invalid_reasons.append(f"negative current stock ({current_stock:,.0f})")
            if safety_stock < 0:
                invalid_reasons.append(f"negative safety stock ({safety_stock:,.0f})")
            if lead_time <= 0:
                invalid_reasons.append(f"non-positive lead time ({lead_time} days)")
            
            results.append({
                "store_nbr": store_nbr,
                "family": family,
                "current_stock": current_stock,
                "safety_stock": safety_stock,
                "supplier_lead_time_days": lead_time,
                "supplier_name": supplier_name,
                "last_updated": last_updated,
                "expected_lt_demand": 0.0,
                "reorder_point": 0.0,
                "suggested_replenishment_qty": 0,
                "stockout_risk": "UNKNOWN",
                "explanation": f"Invalid parameter: {', '.join(invalid_reasons)}. Inventory and lead times cannot be negative.",
                "horizon_warning": None,
            })
            continue

        # Retrieve 13-week forecast series
        forecast_series = forecast_map.get((store_nbr, family), [])

        if not forecast_series:
            # Missing forecast
            results.append({
                "store_nbr": store_nbr,
                "family": family,
                "current_stock": current_stock,
                "safety_stock": safety_stock,
                "supplier_lead_time_days": lead_time,
                "supplier_name": supplier_name,
                "last_updated": last_updated,
                "expected_lt_demand": 0.0,
                "reorder_point": safety_stock,
                "suggested_replenishment_qty": 0,
                "stockout_risk": "UNKNOWN",
                "explanation": "No forecast demand found for this store-family series.",
                "horizon_warning": None,
            })
            continue

        # 1. Lead time demand
        lt_demand, warning = calculate_lead_time_demand(forecast_series, lead_time)

        # 2. Inventory position (baseline: current_stock)
        inv_position = current_stock

        # 3. Reorder point
        reorder_point = lt_demand + safety_stock

        # 4. Suggested replenishment quantity (rounded up to discrete units)
        raw_qty = max(0.0, reorder_point - inv_position)
        suggested_qty = math.ceil(raw_qty)

        # 5. Stockout risk level
        risk_level, explanation = assess_stockout_risk(
            inventory_position=inv_position,
            expected_lt_demand=lt_demand,
            safety_stock=safety_stock,
            reorder_point=reorder_point,
        )

        results.append({
            "store_nbr": store_nbr,
            "family": family,
            "current_stock": current_stock,
            "safety_stock": safety_stock,
            "supplier_lead_time_days": lead_time,
            "supplier_name": supplier_name,
            "last_updated": last_updated,
            "expected_lt_demand": round(lt_demand, 1),
            "reorder_point": round(reorder_point, 1),
            "suggested_replenishment_qty": suggested_qty,
            "stockout_risk": risk_level,
            "explanation": explanation,
            "horizon_warning": warning,
        })

    return pd.DataFrame(results)


def validate_inventory_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str], pd.DataFrame]:
    """
    Validates uploaded inventory CSV dataset against mandatory schema and business rules.

    Checks:
    1. Mandatory columns: store_nbr, family, current_stock, safety_stock.
    2. Optional columns: supplier_lead_time_days, last_updated.
    3. Numerical validity: store_nbr > 0, current_stock >= 0, safety_stock >= 0, lead_time >= 1.
    4. Business key uniqueness: No duplicate (store_nbr, family) pairs.
    5. Missing / NaN values in required fields.
    """
    errors = []
    if df.empty:
        return False, ["Uploaded inventory CSV is empty."], df

    required_cols = ["store_nbr", "family", "current_stock", "safety_stock"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing mandatory columns: {', '.join(missing_cols)}")
        return False, errors, df

    clean_df = df.copy()

    # Check for duplicate keys
    dups = clean_df.duplicated(subset=["store_nbr", "family"]).sum()
    if dups > 0:
        errors.append(f"Found {dups} duplicate (store_nbr, family) combinations.")

    # Check nulls in required columns
    for col in required_cols:
        null_count = clean_df[col].isnull().sum()
        if null_count > 0:
            errors.append(f"Column '{col}' contains {null_count} null or missing values.")

    # Check numeric types and non-negativity
    try:
        clean_df["store_nbr"] = clean_df["store_nbr"].astype(int)
        clean_df["current_stock"] = clean_df["current_stock"].astype(float)
        clean_df["safety_stock"] = clean_df["safety_stock"].astype(float)
        
        if "supplier_lead_time_days" in clean_df.columns:
            clean_df["supplier_lead_time_days"] = clean_df["supplier_lead_time_days"].fillna(7).astype(int)
        else:
            clean_df["supplier_lead_time_days"] = 7
    except Exception as e:
        errors.append(f"Data type conversion error: {e}")
        return False, errors, clean_df

    # Range checks
    if (clean_df["store_nbr"] <= 0).any():
        errors.append("Column 'store_nbr' must contain positive integers.")

    if (clean_df["current_stock"] < 0).any():
        errors.append("Negative values found in 'current_stock'. Inventory cannot be negative.")

    if (clean_df["safety_stock"] < 0).any():
        errors.append("Negative values found in 'safety_stock'. Safety buffer cannot be negative.")

    if (clean_df["supplier_lead_time_days"] <= 0).any():
        errors.append("Lead time 'supplier_lead_time_days' must be at least 1 day.")

    is_valid = len(errors) == 0
    return is_valid, errors, clean_df


def generate_inventory_business_summary(df_plan: pd.DataFrame) -> List[str]:
    """
    Generates rule-based executive insights for the selected replenishment scope.
    """
    if df_plan.empty:
        return ["No inventory records match the current filter selection."]

    total_skus = len(df_plan)
    high_risk_count = int((df_plan["stockout_risk"] == "HIGH").sum())
    med_risk_count = int((df_plan["stockout_risk"] == "MEDIUM").sum())
    low_risk_count = int((df_plan["stockout_risk"] == "LOW").sum())
    
    total_suggested_order = float(df_plan["suggested_replenishment_qty"].sum())
    total_lead_time_demand = float(df_plan["expected_lt_demand"].sum())
    total_stock_on_hand = float(df_plan["current_stock"].sum())

    insights = []

    # High-risk summary
    high_pct = (high_risk_count / total_skus) * 100 if total_skus > 0 else 0
    if high_risk_count > 0:
        top_high_items = (
            df_plan[df_plan["stockout_risk"] == "HIGH"]
            .sort_values("suggested_replenishment_qty", ascending=False)
            .head(3)
        )
        top_cats = ", ".join(top_high_items["family"].unique().tolist()[:2])
        insights.append(
            f"🚨 **High Stockout Risk:** {high_risk_count} items ({high_pct:.1f}% of selection) "
            f"have stock below supplier lead-time demand. Immediate reorders required, "
            f"particularly in {top_cats}."
        )
    else:
        insights.append(
            "✅ **Zero Immediate Stockout Risk:** All selected items currently have enough stock "
            "to cover their expected replenishment lead times."
        )

    # Reorder point summary
    if med_risk_count > 0:
        insights.append(
            f"⚠️ **Safety Buffer Breaches:** {med_risk_count} items cover immediate lead-time demand "
            f"but have breached their safety stock threshold. Replenish soon to restore resiliency."
        )

    # Order volume summary
    insights.append(
        f"📦 **Suggested Replenishment Volume:** A total baseline order of "
        f"**{total_suggested_order:,.0f} units** is recommended across the selected items "
        f"to satisfy reorder points (compared to on-hand stock of {total_stock_on_hand:,.0f} units)."
    )

    # Disclaimer note
    insights.append(
        "ℹ️ **Operational Policy Note:** Recommendations represent baseline heuristics under static lead-time "
        "assumptions (excluding in-transit orders) and should be validated against supplier order minimums."
    )

    return insights
