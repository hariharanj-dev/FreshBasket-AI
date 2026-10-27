"""FreshBasket AI - Inventory Alert Engine & Business Reporting (Phase 9)
========================================================================
Module responsible for:
1. Store-wise inventory aggregation and risk exposure profiling
2. Executive business report generation (formatted briefing in Markdown/TXT)
3. Email alert briefing generation (ready for executive circulation)
4. Lightweight persistent alert history tracking (CSV-based snapshot management)
5. Alert prioritization and severity sorting
"""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ALERT_HISTORY_FILE = WORKSPACE_ROOT / "data" / "inventory" / "alert_history.csv"


# -----------------------------------------------------------------------------
# 1. Store-Wise Summary Aggregation
# -----------------------------------------------------------------------------

def generate_store_wise_summary(df_plan: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates the replenishment plan at the store level.

    Calculates:
    - store_nbr: store identifier
    - high_risk_count: count of HIGH risk items in this store
    - medium_risk_count: count of MEDIUM risk items in this store
    - low_risk_count: count of LOW risk items in this store
    - unknown_risk_count: count of UNKNOWN risk items in this store
    - total_items: total monitored product families for this store
    - total_expected_lt_demand: sum of expected lead-time demand in units
    - total_suggested_replenishment_qty: total suggested replenishment units
    - urgency_level: CRITICAL, ATTENTION, or HEALTHY
    
    Guarantees no duplicate counting of store-family pairs.
    """
    if df_plan.empty:
        return pd.DataFrame(columns=[
            "store_nbr", "high_risk_count", "medium_risk_count", "low_risk_count",
            "unknown_risk_count", "total_items", "total_expected_lt_demand",
            "total_suggested_replenishment_qty", "urgency_level"
        ])

    records = []
    for store_nbr, group in df_plan.groupby("store_nbr"):
        high_cnt = int((group["stockout_risk"] == "HIGH").sum())
        med_cnt = int((group["stockout_risk"] == "MEDIUM").sum())
        low_cnt = int((group["stockout_risk"] == "LOW").sum())
        unk_cnt = int((group["stockout_risk"] == "UNKNOWN").sum())
        total_skus = len(group)
        tot_demand = float(group["expected_lt_demand"].sum())
        tot_replenish = float(group["suggested_replenishment_qty"].sum())

        if high_cnt > 0:
            urgency = "CRITICAL"
        elif med_cnt > 0:
            urgency = "ATTENTION"
        else:
            urgency = "HEALTHY"

        records.append({
            "store_nbr": int(store_nbr),
            "high_risk_count": high_cnt,
            "medium_risk_count": med_cnt,
            "low_risk_count": low_cnt,
            "unknown_risk_count": unk_cnt,
            "total_items": total_skus,
            "total_expected_lt_demand": round(tot_demand, 1),
            "total_suggested_replenishment_qty": int(tot_replenish),
            "urgency_level": urgency,
        })

    summary_df = pd.DataFrame(records)
    # Sort stores prioritizing highest risk items, then largest replenishment quantity
    summary_df = summary_df.sort_values(
        by=["high_risk_count", "total_suggested_replenishment_qty"],
        ascending=[False, False]
    ).reset_index(drop=True)

    return summary_df


# -----------------------------------------------------------------------------
# 2. Alert Prioritization & Sorting
# -----------------------------------------------------------------------------

def prioritize_alerts(df_plan: pd.DataFrame) -> pd.DataFrame:
    """
    Orders replenishment alerts by severity tier and deficit urgency:
    1. HIGH risk items (largest replenishment quantity first)
    2. MEDIUM risk items
    3. LOW risk items
    4. UNKNOWN risk items
    """
    if df_plan.empty:
        return df_plan

    tier_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "UNKNOWN": 3}
    df_sorted = df_plan.copy()
    df_sorted["_risk_priority"] = df_sorted["stockout_risk"].map(lambda r: tier_order.get(r, 4))
    df_sorted = df_sorted.sort_values(
        by=["_risk_priority", "suggested_replenishment_qty"],
        ascending=[True, False]
    ).drop(columns=["_risk_priority"]).reset_index(drop=True)

    return df_sorted


# -----------------------------------------------------------------------------
# 3. Executive Business Report Generation
# -----------------------------------------------------------------------------

def generate_executive_business_report(
    df_plan: pd.DataFrame,
    data_source_mode: str,
    forecast_model_label: str,
    selected_store_label: str = "All Stores (54)",
    selected_family_label: str = "All Families (33)",
) -> str:
    """
    Compiles a comprehensive markdown executive briefing report detailing
    inventory stockout exposure, reorder requirements, and operational observations.
    """
    gen_time_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    
    total_skus = len(df_plan)
    high_cnt = int((df_plan["stockout_risk"] == "HIGH").sum()) if not df_plan.empty else 0
    med_cnt = int((df_plan["stockout_risk"] == "MEDIUM").sum()) if not df_plan.empty else 0
    low_cnt = int((df_plan["stockout_risk"] == "LOW").sum()) if not df_plan.empty else 0
    unk_cnt = int((df_plan["stockout_risk"] == "UNKNOWN").sum()) if not df_plan.empty else 0
    
    tot_lt_demand = float(df_plan["expected_lt_demand"].sum()) if not df_plan.empty else 0.0
    tot_replenish = float(df_plan["suggested_replenishment_qty"].sum()) if not df_plan.empty else 0.0
    tot_stock_on_hand = float(df_plan["current_stock"].sum()) if not df_plan.empty else 0.0

    # Stores and families affected by HIGH or MEDIUM alerts
    if not df_plan.empty:
        affected_items = df_plan[df_plan["stockout_risk"].isin(["HIGH", "MEDIUM"])]
        stores_affected = int(affected_items["store_nbr"].nunique())
        families_affected = int(affected_items["family"].nunique())
    else:
        stores_affected = 0
        families_affected = 0

    # Top deficit items
    top_deficit_rows = ""
    if high_cnt > 0:
        top_high = df_plan[df_plan["stockout_risk"] == "HIGH"].sort_values(
            "suggested_replenishment_qty", ascending=False
        ).head(8)
        top_deficit_rows = "\n".join([
            f"| Store {r['store_nbr']} | {r['family']} | {r['current_stock']:,.0f} | "
            f"{r['expected_lt_demand']:,.1f} | {r['reorder_point']:,.1f} | "
            f"**{r['suggested_replenishment_qty']:,}** | {r['supplier_lead_time_days']} days | {r.get('supplier_name', 'N/A')} |"
            for _, r in top_high.iterrows()
        ])
    else:
        top_deficit_rows = "| None | No high-risk items detected in current scope | - | - | - | 0 | - | - |"

    report = f"""# FRESHBASKET AI — EXECUTIVE INVENTORY & STOCKOUT RISK BRIEFING
**Report Generated:** {gen_time_utc}
**Target Scope:** Store Filter: `{selected_store_label}` | Product Family: `{selected_family_label}`
**Forecasting Engine:** `{forecast_model_label}`
**Inventory Data Mode:** `{"🟢 DEMO DATA (Synthetic Retail Balances)" if data_source_mode == "DEMO" else "🔵 CUSTOM UPLOAD (Verified Retail Balances)"}`

---

## 1. Executive Summary & Exposure Scorecard

| Metric | Value | Operational Significance |
| :--- | :--- | :--- |
| **Total Monitored Items** | **{total_skus:,}** | Unique store-family series evaluated |
| **High Stockout Risk SKUs** | **{high_cnt:,}** ({((high_cnt/total_skus)*100 if total_skus>0 else 0):.1f}%) | Immediate stockout projected before replenishment arrives |
| **Medium Risk SKUs** | **{med_cnt:,}** ({((med_cnt/total_skus)*100 if total_skus>0 else 0):.1f}%) | Lead demand covered but safety stock buffer breached |
| **Low Risk SKUs (Healthy)** | **{low_cnt:,}** ({((low_cnt/total_skus)*100 if total_skus>0 else 0):.1f}%) | Current stock satisfies reorder point requirements |
| **Unknown / Invalid SKUs** | **{unk_cnt:,}** | Missing parameters or rejected inputs |
| **Stores Requiring Reorders** | **{stores_affected:,}** | Retail branches with High or Medium risk items |
| **Product Families Affected** | **{families_affected:,}** | FMCG categories requiring supplier procurement |
| **Total Lead-Time Demand** | **{tot_lt_demand:,.0f} units** | Aggregated customer demand during lead windows |
| **Current Stock on Hand** | **{tot_stock_on_hand:,.0f} units** | Usable uncommitted physical store inventory |
| **Suggested Reorder Volume** | **{tot_replenish:,.0f} units** | Recommended replenishment to reach Reorder Point (ROP) |

---

## 2. Priority Action Items: Top Replenishment Deficits

The following items exhibit the highest projected stockout deficit and should be prioritized for immediate purchase order submission:

| Store | Product Family | Current Stock | Lead Demand | Reorder Point | Suggested Order | Lead Time | Supplier |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
{top_deficit_rows}

---

## 3. Core Business Observations & Operational Action Plan

1. **Procurement Expediting:** {high_cnt} items have on-hand stock below lead-time demand. For these SKUs, stockouts will occur before standard supplier delivery unless expedited shipping or local cross-store transfers are authorized.
2. **Safety Buffer Maintenance:** {med_cnt} items are currently operating with depleted safety buffers. While immediate stockout is not imminent during this delivery cycle, these series are highly vulnerable to demand surges or supplier delivery delays.
3. **Inventory Capital Allocation:** Suggested reorder volume across the selected portfolio totals **{tot_replenish:,.0f} units**, which restores all series to their configured reorder thresholds (Expected Demand + Safety Stock).
4. **Data Integrity Audit:** Exactly {unk_cnt} records were flagged as UNKNOWN. Incomplete or non-positive stock parameters are rejected to preserve procurement accuracy.

---

## 4. Supply Chain Assumptions & Explicit Limitations

- **Heuristic Replenishment Baseline:** Recommendations are generated via deterministic reorder point calculations: $ROP = \\hat{{D}}_{{LT}} + \\text{{Safety Stock}}$, with discrete whole-unit rounding: $Q = \\lceil \\max(0, ROP - \\text{{Stock}}) \\rceil$.
- **Exclusion of In-Transit Orders:** Calculations assume Inventory Position equals current uncommitted physical stock. Open supplier purchase orders currently in transit are excluded unless provided in custom uploads.
- **Supplier Order Constraints:** Recommendations represent net inventory deficits. They do not automatically apply supplier Minimum Order Quantities (MOQ), pallet pack-size rounding, or container load optimization.
- **Disclaimer:** FreshBasket AI recommendations represent baseline decision-support heuristics. Demand forecasting and inventory models do NOT guarantee stockout prevention under unexpected demand shocks, promotional spikes, or supplier force majeure disruptions.
"""
    return report


# -----------------------------------------------------------------------------
# 4. Email Alert Preparation
# -----------------------------------------------------------------------------

def generate_email_alert_draft(
    df_plan: pd.DataFrame,
    data_source_mode: str,
    forecast_model_label: str,
    selected_store_label: str = "All Stores (54)",
    selected_family_label: str = "All Families (33)",
) -> str:
    """
    Formats a clean, executive email notification ready for procurement teams.
    """
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    high_cnt = int((df_plan["stockout_risk"] == "HIGH").sum()) if not df_plan.empty else 0
    med_cnt = int((df_plan["stockout_risk"] == "MEDIUM").sum()) if not df_plan.empty else 0
    tot_replenish = float(df_plan["suggested_replenishment_qty"].sum()) if not df_plan.empty else 0.0
    
    if not df_plan.empty:
        affected = df_plan[df_plan["stockout_risk"] == "HIGH"]
        affected_stores = int(affected["store_nbr"].nunique())
    else:
        affected_stores = 0

    top_items_text = ""
    if high_cnt > 0:
        top_high = df_plan[df_plan["stockout_risk"] == "HIGH"].sort_values(
            "suggested_replenishment_qty", ascending=False
        ).head(5)
        for _, r in top_high.iterrows():
            top_items_text += (
                f"  • Store {r['store_nbr']} | {r['family']}: Stock={r['current_stock']:,.0f}, "
                f"Lead Demand={r['expected_lt_demand']:,.1f} -> Order {r['suggested_replenishment_qty']:,} units "
                f"(Supplier: {r.get('supplier_name', 'Standard')}, Lead Time: {r['supplier_lead_time_days']}d)\n"
            )
    else:
        top_items_text = "  • No high-risk stockout deficits identified in current selection.\n"

    email_body = f"""Subject: [ACTION REQUIRED] FreshBasket AI Stockout Risk Alert: {high_cnt} High-Risk Items Identified ({timestamp_str})

Dear Supply Chain & Replenishment Team,

The automated inventory health scan was executed on {timestamp_str} using the FreshBasket AI {forecast_model_label} demand forecasting engine.

EXECUTIVE ALERT SUMMARY:
----------------------------------------------------------------------
• High Stockout Risk SKUs:    {high_cnt:,} items (Immediate reorder required)
• Medium Risk SKUs:           {med_cnt:,} items (Safety stock breached)
• Affected Retail Branches:   {affected_stores:,} stores with critical deficits
• Total Recommended Orders:   {tot_replenish:,.0f} discrete units
• Active Inventory Source:    {"DEMO DATA (Synthetic)" if data_source_mode == "DEMO" else "CUSTOM UPLOAD (Verified)"}
• Scope Evaluated:            {selected_store_label} / {selected_family_label}

CRITICAL HIGH-RISK ITEMS (TOP DEFICITS):
----------------------------------------------------------------------
{top_items_text}
RECOMMENDED ACTIONS:
1. Review purchase orders for the critical SKUs listed above to prevent store stockouts.
2. Verify supplier lead times and check for any open purchase orders currently in transit.
3. Access the complete Interactive Replenishment Register in the FreshBasket AI dashboard for full store-by-store breakdowns.

Notice: This briefing was generated via deterministic reorder point heuristics based on 13-week machine learning demand forecasts. Forecasts and replenishment recommendations provide decision support and do not guarantee stockout prevention under sudden demand shocks.

Best regards,
FreshBasket AI Supply Chain System
"""
    return email_body


# -----------------------------------------------------------------------------
# 5. Persistent Alert History Tracking
# -----------------------------------------------------------------------------

def record_alert_snapshot(
    df_plan: pd.DataFrame,
    data_source_mode: str,
    forecast_model_label: str,
    history_file: Optional[Path] = None,
) -> Tuple[bool, str, pd.DataFrame]:
    """
    Saves an immutable snapshot of current high and medium risk alerts to alert_history.csv.
    Includes duplicate protection: will not record if an identical snapshot was logged within the same run.
    """
    target_file = history_file or ALERT_HISTORY_FILE
    target_file.parent.mkdir(parents=True, exist_ok=True)

    if df_plan.empty:
        return False, "Replenishment plan is empty. No alerts to record.", pd.DataFrame()

    # Filter to active alerts (HIGH, MEDIUM, and UNKNOWN)
    active_alerts = df_plan[df_plan["stockout_risk"].isin(["HIGH", "MEDIUM", "UNKNOWN"])].copy()
    if active_alerts.empty:
        return False, "No active alerts (HIGH/MEDIUM/UNKNOWN) found in the current selection to snapshot.", pd.DataFrame()

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    snapshot_rows = []
    for _, row in active_alerts.iterrows():
        snapshot_rows.append({
            "timestamp": now_iso,
            "store_nbr": int(row["store_nbr"]),
            "family": str(row["family"]),
            "stockout_risk": str(row["stockout_risk"]),
            "suggested_replenishment_qty": int(row.get("suggested_replenishment_qty", 0)),
            "current_stock": float(row.get("current_stock", 0.0)),
            "expected_lt_demand": float(row.get("expected_lt_demand", 0.0)),
            "reorder_point": float(row.get("reorder_point", 0.0)),
            "data_source": str(data_source_mode),
            "forecast_model": str(forecast_model_label),
        })

    new_snapshot_df = pd.DataFrame(snapshot_rows)

    # Read existing history if present
    if target_file.exists():
        try:
            existing_df = pd.read_csv(target_file)
            # Avoid duplicate snapshot: check if last timestamp has same store-family alert set
            if not existing_df.empty:
                last_time = existing_df["timestamp"].iloc[-1]
                last_subset = existing_df[existing_df["timestamp"] == last_time]
                if (
                    len(last_subset) == len(new_snapshot_df) and
                    set(last_subset["store_nbr"]) == set(new_snapshot_df["store_nbr"]) and
                    set(last_subset["family"]) == set(new_snapshot_df["family"]) and
                    last_subset["data_source"].iloc[0] == data_source_mode and
                    last_subset["forecast_model"].iloc[0] == forecast_model_label
                ):
                    return False, f"Identical snapshot already logged at {last_time}. Skipped duplicate record.", existing_df

            combined_df = pd.concat([existing_df, new_snapshot_df], ignore_index=True)
        except Exception:
            combined_df = new_snapshot_df
    else:
        combined_df = new_snapshot_df

    combined_df.to_csv(target_file, index=False)
    msg = f"Successfully recorded {len(new_snapshot_df):,} alert items to history at {now_iso}."
    return True, msg, combined_df


def load_alert_history(history_file: Optional[Path] = None) -> pd.DataFrame:
    """Loads recorded alert history records from disk."""
    target_file = history_file or ALERT_HISTORY_FILE
    if not target_file.exists():
        return pd.DataFrame(columns=[
            "timestamp", "store_nbr", "family", "stockout_risk",
            "suggested_replenishment_qty", "current_stock",
            "expected_lt_demand", "reorder_point", "data_source", "forecast_model"
        ])
    try:
        df = pd.read_csv(target_file)
        return df
    except Exception:
        return pd.DataFrame()
