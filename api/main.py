"""FreshBasket AI - FastAPI Production Backend
==============================================
High-performance REST API serving FMCG weekly demand forecasts,
evaluation metrics, store and product performance, inventory planning heuristics,
and automated stockout risk alerts for the React frontend.
"""

from datetime import datetime
import io
import math
from pathlib import Path
import sys
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, PlainTextResponse

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.data_loader import (
    load_all_dashboard_data,
    filter_historical_sales,
    filter_forecast_data,
    compute_kpis,
)
from src.business.inventory_planner import (
    compute_replenishment_plan,
    validate_inventory_dataframe,
    generate_inventory_business_summary,
)
from src.business.alert_engine import (
    generate_store_wise_summary,
    prioritize_alerts,
    generate_executive_business_report,
    generate_email_alert_draft,
    record_alert_snapshot,
    load_alert_history,
)
from pydantic import BaseModel, Field
from src.business.insight_engine import (
    detect_demand_trends,
    generate_forecast_insights,
    generate_inventory_risk_insights,
    synthesize_business_insights,
)
from src.business.assistant_service import answer_business_question
from src.business.scenario_simulator import run_inventory_simulation

# -----------------------------------------------------------------------------
# FastAPI Application Configuration & CORS
# -----------------------------------------------------------------------------
app = FastAPI(
    title="FreshBasket AI API",
    description="FMCG Demand Forecasting and Inventory Intelligence REST API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for local React development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Data Layer In-Memory Cache
# -----------------------------------------------------------------------------
DATASETS: Dict[str, pd.DataFrame] = {}
CUSTOM_INVENTORY: Optional[pd.DataFrame] = None
DATA_SOURCE_MODE: str = "DEMO"

MODEL_MAP = {
    "naive": "naive_forecast",
    "naive_forecast": "naive_forecast",
    "Naive Baseline": "naive_forecast",
    "lightgbm": "lightgbm_forecast",
    "lightgbm_forecast": "lightgbm_forecast",
    "LightGBM (Recursive)": "lightgbm_forecast",
    "seasonal_naive": "seasonal_naive_forecast",
    "seasonal_naive_forecast": "seasonal_naive_forecast",
    "Seasonal Naive (Lag 52)": "seasonal_naive_forecast",
}

PERF_MODEL_MAP = {
    "naive": "Naive",
    "naive_forecast": "Naive",
    "Naive Baseline": "Naive",
    "lightgbm": "LightGBM",
    "lightgbm_forecast": "LightGBM",
    "LightGBM (Recursive)": "LightGBM",
    "seasonal_naive": "Seasonal Naive",
    "seasonal_naive_forecast": "Seasonal Naive",
    "Seasonal Naive (Lag 52)": "Seasonal Naive",
}

MODEL_LABELS = {
    "naive_forecast": "Naive Baseline",
    "lightgbm_forecast": "LightGBM (Recursive)",
    "seasonal_naive_forecast": "Seasonal Naive (Lag 52)",
}


def get_data() -> Dict[str, pd.DataFrame]:
    """Loads and caches datasets into memory."""
    global DATASETS
    if not DATASETS:
        DATASETS = load_all_dashboard_data()
    return DATASETS


def resolve_model(model_param: Optional[str]) -> str:
    """Resolves arbitrary model name/param to dataframe column."""
    if not model_param:
        return "naive_forecast"
    clean = model_param.strip().lower()
    for key, val in MODEL_MAP.items():
        if key.lower() == clean:
            return val
    return "naive_forecast"


def resolve_perf_model(model_param: Optional[str]) -> str:
    """Resolves arbitrary model name/param to performance table model name."""
    if not model_param:
        return "Naive"
    clean = model_param.strip().lower()
    for key, val in PERF_MODEL_MAP.items():
        if key.lower() == clean:
            return val
    return "Naive"


def get_active_inventory() -> pd.DataFrame:
    """Returns active inventory DataFrame (custom uploaded if present, else baseline demo)."""
    global CUSTOM_INVENTORY, DATA_SOURCE_MODE
    if CUSTOM_INVENTORY is not None and not CUSTOM_INVENTORY.empty:
        return CUSTOM_INVENTORY
    return get_data()["inventory"]


# -----------------------------------------------------------------------------
# 1. Health & Filter Metadata Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/health")
def get_health():
    """System health check and diagnostic status."""
    data = get_data()
    return {
        "status": "healthy",
        "service": "FreshBasket AI API",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "data_mode": DATA_SOURCE_MODE,
        "datasets": {k: len(v) for k, v in data.items()},
    }


@app.get("/api/filters")
def get_filter_options():
    """Returns unique store numbers, store details, product families, and models."""
    data = get_data()
    df_hist = data["historical"]
    df_store = data["store_performance"]

    # Unique stores with metadata
    stores = []
    store_meta_map = df_store.drop_duplicates(subset=["store_nbr"]).set_index("store_nbr")
    for s in sorted(df_hist["store_nbr"].unique()):
        store_int = int(s)
        meta = store_meta_map.loc[store_int] if store_int in store_meta_map.index else None
        stores.append({
            "store_nbr": store_int,
            "city": str(meta["city"]) if meta is not None and "city" in meta else f"Store {store_int}",
            "state": str(meta["state"]) if meta is not None and "state" in meta else "Ecuador",
            "type": str(meta["type"]) if meta is not None and "type" in meta else "D",
            "cluster": int(meta["cluster"]) if meta is not None and "cluster" in meta else 1,
        })

    families = sorted(df_hist["family"].unique().tolist())
    min_date = df_hist["week_start_date"].min().strftime("%Y-%m-%d")
    max_date = df_hist["week_start_date"].max().strftime("%Y-%m-%d")

    return {
        "stores": stores,
        "families": families,
        "models": [
            {"id": "naive_forecast", "label": "Naive Baseline", "wape": 11.61, "is_best": True},
            {"id": "lightgbm_forecast", "label": "LightGBM (Recursive)", "wape": 12.27, "is_best": False},
            {"id": "seasonal_naive_forecast", "label": "Seasonal Naive (Lag 52)", "wape": 16.17, "is_best": False},
        ],
        "date_range": {
            "min_date": min_date,
            "max_date": max_date,
            "last_complete_week": "2017-08-13",
            "forecast_origin": "2017-08-13",
            "forecast_start": "2017-08-14",
            "forecast_end": "2017-11-12",
        },
    }


# -----------------------------------------------------------------------------
# 2. Executive Overview Endpoint
# -----------------------------------------------------------------------------

@app.get("/api/overview")
def get_overview(
    model: str = Query("naive_forecast"),
    store_nbr: Optional[int] = Query(None),
    family: Optional[str] = Query(None),
):
    """Aggregates executive KPIs, historical trajectory, forward forecast, and leaderboard."""
    data = get_data()
    model_col = resolve_model(model)
    df_hist = data["historical"]
    df_fc = data["forecast"]
    df_perf = data["model_performance"]

    selected_stores = [store_nbr] if store_nbr is not None else None
    selected_families = [family] if family is not None else None

    # Filtered subsets
    df_h_filtered = filter_historical_sales(
        df_hist,
        selected_stores=selected_stores,
        selected_families=selected_families,
        exclude_partial_weeks=True,
    )
    df_f_filtered = filter_forecast_data(
        df_fc,
        selected_stores=selected_stores,
        selected_families=selected_families,
    )

    kpis = compute_kpis(df_h_filtered, df_f_filtered, model_col=model_col)

    # 1. Historical Weekly Aggregated Trend (last 52 complete weeks for crisp chart display)
    hist_agg = (
        df_h_filtered.groupby("week_start_date")["weekly_sales"]
        .sum()
        .reset_index()
        .sort_values("week_start_date")
    )
    hist_agg["date_str"] = hist_agg["week_start_date"].dt.strftime("%Y-%m-%d")
    recent_hist = hist_agg.tail(52)

    # 2. Forward 13-week predictions
    fc_agg = (
        df_f_filtered.groupby(["week_start_date", "forecast_horizon_week"])[
            ["naive_forecast", "lightgbm_forecast", "seasonal_naive_forecast"]
        ]
        .sum()
        .reset_index()
        .sort_values("forecast_horizon_week")
    )
    fc_agg["date_str"] = fc_agg["week_start_date"].dt.strftime("%Y-%m-%d")

    # Combine for combined trajectory timeline
    timeline = []
    for _, r in recent_hist.iterrows():
        timeline.append({
            "date": r["date_str"],
            "actual": round(float(r["weekly_sales"]), 1),
            "naive": None,
            "lightgbm": None,
            "seasonal_naive": None,
            "type": "Historical Actual",
        })

    for _, r in fc_agg.iterrows():
        timeline.append({
            "date": r["date_str"],
            "actual": None,
            "naive": round(float(r["naive_forecast"]), 1),
            "lightgbm": round(float(r["lightgbm_forecast"]), 1),
            "seasonal_naive": round(float(r["seasonal_naive_forecast"]), 1),
            "type": "Forecast",
            "horizon_week": int(r["forecast_horizon_week"]),
        })

    # 3. Model performance comparison
    global_perf = df_perf[df_perf["granularity"] == "global"].sort_values("wape").to_dict(orient="records")

    # 4. Top 10 product families by volume
    top_families = (
        df_h_filtered.groupby("family")["weekly_sales"]
        .sum()
        .reset_index()
        .sort_values("weekly_sales", ascending=False)
        .head(10)
    )
    top_families["weekly_sales"] = top_families["weekly_sales"].round(1)

    return {
        "kpis": kpis,
        "selected_model": {
            "id": model_col,
            "label": MODEL_LABELS.get(model_col, "Naive Baseline"),
        },
        "best_model": {
            "name": "Naive Baseline",
            "wape": 11.61,
            "mae": 328.7,
            "rmse": 1052.2,
        },
        "timeline": timeline,
        "model_performance": global_perf,
        "top_families": top_families.to_dict(orient="records"),
    }


# -----------------------------------------------------------------------------
# 3. Sales Analysis Endpoint
# -----------------------------------------------------------------------------

@app.get("/api/sales")
def get_sales(
    store_nbr: Optional[int] = Query(None),
    family: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    exclude_partial_weeks: bool = Query(True),
):
    """Detailed historical weekly sales series, moving averages, category shares, and rankings."""
    data = get_data()
    df_hist = data["historical"]

    selected_stores = [store_nbr] if store_nbr is not None else None
    selected_families = [family] if family is not None else None
    s_date = pd.to_datetime(start_date) if start_date else None
    e_date = pd.to_datetime(end_date) if end_date else None

    filtered = filter_historical_sales(
        df_hist,
        selected_stores=selected_stores,
        selected_families=selected_families,
        start_date=s_date,
        end_date=e_date,
        exclude_partial_weeks=exclude_partial_weeks,
    )

    if filtered.empty:
        return {
            "time_series": [],
            "top_stores": [],
            "top_families": [],
            "monthly_summary": [],
            "records": [],
            "summary": {"total_sales": 0, "weeks_count": 0, "avg_weekly_sales": 0},
        }

    # Weekly Aggregated Series with 4-week Rolling Moving Average
    weekly = (
        filtered.groupby("week_start_date")
        .agg(
            weekly_sales=("weekly_sales", "sum"),
            onpromotion_items=("onpromotion_sum", "sum"),
            transactions=("weekly_transactions", "sum"),
        )
        .reset_index()
        .sort_values("week_start_date")
    )
    weekly["moving_avg_4wk"] = weekly["weekly_sales"].rolling(window=4, min_periods=1).mean().round(1)
    weekly["date"] = weekly["week_start_date"].dt.strftime("%Y-%m-%d")
    weekly["weekly_sales"] = weekly["weekly_sales"].round(1)

    # Top stores
    top_stores = (
        filtered.groupby("store_nbr")["weekly_sales"]
        .sum()
        .reset_index()
        .sort_values("weekly_sales", ascending=False)
        .head(15)
    )
    top_stores["store_label"] = [f"Store {s}" for s in top_stores["store_nbr"]]
    top_stores["weekly_sales"] = top_stores["weekly_sales"].round(1)

    # Top product families
    top_fams = (
        filtered.groupby("family")["weekly_sales"]
        .sum()
        .reset_index()
        .sort_values("weekly_sales", ascending=False)
        .head(15)
    )
    top_fams["weekly_sales"] = top_fams["weekly_sales"].round(1)

    # Monthly aggregation
    filtered["year_month"] = filtered["week_start_date"].dt.strftime("%Y-%m")
    monthly = (
        filtered.groupby("year_month")["weekly_sales"]
        .sum()
        .reset_index()
        .sort_values("year_month")
    )
    monthly["weekly_sales"] = monthly["weekly_sales"].round(1)

    # Summary
    tot_sales = float(filtered["weekly_sales"].sum())
    w_count = int(filtered["week_start_date"].nunique())
    avg_w_sales = tot_sales / w_count if w_count > 0 else 0

    return {
        "summary": {
            "total_sales": round(tot_sales, 1),
            "weeks_count": w_count,
            "avg_weekly_sales": round(avg_w_sales, 1),
            "store_filter": store_nbr or "All",
            "family_filter": family or "All",
        },
        "time_series": weekly[["date", "weekly_sales", "moving_avg_4wk", "onpromotion_items", "transactions"]].to_dict(orient="records"),
        "top_stores": top_stores.to_dict(orient="records"),
        "top_families": top_fams.to_dict(orient="records"),
        "monthly_summary": monthly.tail(24).to_dict(orient="records"),
    }


# -----------------------------------------------------------------------------
# 4. Forecast Endpoint
# -----------------------------------------------------------------------------

@app.get("/api/forecasts")
def get_forecasts(
    model: str = Query("naive_forecast"),
    store_nbr: Optional[int] = Query(None),
    family: Optional[str] = Query(None),
):
    """Multi-step 13-week forward predictions, comparisons, and tabular forecast."""
    data = get_data()
    model_col = resolve_model(model)
    df_fc = data["forecast"]

    selected_stores = [store_nbr] if store_nbr is not None else None
    selected_families = [family] if family is not None else None

    filtered = filter_forecast_data(
        df_fc,
        selected_stores=selected_stores,
        selected_families=selected_families,
    )

    if filtered.empty:
        return {
            "metadata": {"origin": "2017-08-13", "horizon_weeks": 13, "model": model_col},
            "weekly_forecast": [],
            "models_comparison": [],
            "records": [],
            "total_forecast_units": 0,
        }

    # Aggregate by horizon week
    weekly_agg = (
        filtered.groupby(["forecast_horizon_week", "week_start_date", "week_end_date"])[
            ["naive_forecast", "lightgbm_forecast", "seasonal_naive_forecast"]
        ]
        .sum()
        .reset_index()
        .sort_values("forecast_horizon_week")
    )

    weekly_agg["start_date"] = weekly_agg["week_start_date"].dt.strftime("%Y-%m-%d")
    weekly_agg["end_date"] = weekly_agg["week_end_date"].dt.strftime("%Y-%m-%d")

    records = []
    for _, r in weekly_agg.iterrows():
        records.append({
            "horizon_week": int(r["forecast_horizon_week"]),
            "week_start_date": r["start_date"],
            "week_end_date": r["end_date"],
            "naive_forecast": round(float(r["naive_forecast"]), 1),
            "lightgbm_forecast": round(float(r["lightgbm_forecast"]), 1),
            "seasonal_naive_forecast": round(float(r["seasonal_naive_forecast"]), 1),
            "selected_forecast": round(float(r[model_col]), 1),
        })

    tot_units = float(weekly_agg[model_col].sum())

    return {
        "metadata": {
            "forecast_origin": "2017-08-13",
            "horizon_weeks": 13,
            "horizon_start": "2017-08-14",
            "horizon_end": "2017-11-12",
            "selected_model": model_col,
            "selected_model_label": MODEL_LABELS.get(model_col, "Naive Baseline"),
        },
        "total_forecast_units": round(tot_units, 1),
        "weekly_forecast": records,
    }


# -----------------------------------------------------------------------------
# 5. Model Evaluation Endpoint
# -----------------------------------------------------------------------------

@app.get("/api/evaluation")
def get_evaluation():
    """Global evaluation matrix, horizon decay curves, and RMSLE discrepancy audit."""
    data = get_data()
    df_perf = data["model_performance"]

    # Global Performance rows
    global_perf = df_perf[df_perf["granularity"] == "global"].sort_values("wape").copy()
    global_perf["rank"] = [1, 2, 3]
    global_perf["model_label"] = [MODEL_LABELS.get(m, m) for m in global_perf["model"]]

    # Horizon Decay curve (weeks 1 to 13)
    horizon_df = df_perf[df_perf["granularity"] == "horizon_week"].copy()
    horizon_df["horizon_week"] = horizon_df["forecast_horizon_week"].astype(int)
    horizon_df = horizon_df.sort_values("horizon_week")

    decay_records = []
    for hw, group in horizon_df.groupby("horizon_week"):
        row = {"horizon_week": int(hw)}
        for _, g in group.iterrows():
            m = str(g["model"])
            row[f"{m}_wape"] = round(float(g["wape"]), 2)
            row[f"{m}_mae"] = round(float(g["mae"]), 1)
            row[f"{m}_rmse"] = round(float(g["rmse"]), 1)
        decay_records.append(row)

    return {
        "global_matrix": global_perf.to_dict(orient="records"),
        "horizon_decay": decay_records,
        "rmsle_audit": {
            "explanation": (
                "LightGBM's global RMSLE is slightly higher than Naive because of 1,858 zero-sales "
                "actual observations in the holdout test period. On zero actuals, predicting a continuous "
                "small expectation (~30 units) incurs heavy log penalties ((ln(31)-ln(1))^2 = 11.8 points). "
                "However, on core commercial volume (>1,000 units, representing >85% of company revenue), "
                "LightGBM achieves RMSLE 0.2318 versus Naive 0.2583 (10.3% more accurate)."
            ),
            "commercial_volume_rmsle": {
                "lightgbm": 0.2318,
                "naive": 0.2583,
                "lgbm_advantage_pct": 10.3,
            },
        },
    }


# -----------------------------------------------------------------------------
# 6. Product Family Insights Endpoint
# -----------------------------------------------------------------------------

@app.get("/api/products")
def get_products(model: str = Query("naive_forecast")):
    """Product-family sales rankings, forecast accuracy by category, and LGBM wins."""
    data = get_data()
    perf_model = resolve_perf_model(model)
    df_fam_perf = data["family_performance"]
    df_hist = data["historical"]

    # Filter to selected model or comparison
    model_rows = df_fam_perf[df_fam_perf["model"] == perf_model].copy()
    if model_rows.empty:
        model_rows = df_fam_perf.copy()

    # Calculate categories where LightGBM beat Naive
    lgbm_rows = df_fam_perf[df_fam_perf["model"] == "LightGBM"].set_index("family")
    naive_rows = df_fam_perf[df_fam_perf["model"] == "Naive"].set_index("family")

    lgbm_leads = []
    for fam in lgbm_rows.index:
        if fam in naive_rows.index:
            l_wape = float(lgbm_rows.loc[fam, "wape"])
            n_wape = float(naive_rows.loc[fam, "wape"])
            if l_wape < n_wape:
                lgbm_leads.append({
                    "family": fam,
                    "lightgbm_wape": round(l_wape, 2),
                    "naive_wape": round(n_wape, 2),
                    "improvement_pts": round(n_wape - l_wape, 2),
                })
    lgbm_leads = sorted(lgbm_leads, key=lambda x: x["improvement_pts"], reverse=True)

    # Family historical sales volume
    fam_vol = (
        df_hist[df_hist["is_partial_week"] == False]
        .groupby("family")["weekly_sales"]
        .sum()
        .reset_index()
        .sort_values("weekly_sales", ascending=False)
    )
    fam_vol["weekly_sales"] = fam_vol["weekly_sales"].round(1)

    return {
        "family_accuracy": df_fam_perf.to_dict(orient="records"),
        "family_volumes": fam_vol.to_dict(orient="records"),
        "categories_where_lgbm_leads": lgbm_leads,
    }


# -----------------------------------------------------------------------------
# 7. Store Performance Endpoint
# -----------------------------------------------------------------------------

@app.get("/api/stores")
def get_stores(model: str = Query("naive_forecast"), store_nbr: Optional[int] = Query(None)):
    """Store rankings, store format types (A-E), operational clusters, and accuracy."""
    data = get_data()
    perf_model = resolve_perf_model(model)
    df_store_perf = data["store_performance"]

    model_store_perf = df_store_perf[df_store_perf["model"] == perf_model].copy()
    if store_nbr is not None:
        model_store_perf = model_store_perf[model_store_perf["store_nbr"] == store_nbr]

    model_store_perf = model_store_perf.sort_values("wape")

    # Format distribution
    type_summary = (
        model_store_perf.groupby("type")
        .agg(store_count=("store_nbr", "count"), avg_wape=("wape", "mean"))
        .reset_index()
        .sort_values("store_count", ascending=False)
    )
    type_summary["avg_wape"] = type_summary["avg_wape"].round(2)

    return {
        "store_rankings": model_store_perf.to_dict(orient="records"),
        "type_summary": type_summary.to_dict(orient="records"),
    }


# -----------------------------------------------------------------------------
# 8. Inventory Planning & Replenishment Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/inventory/summary")
def get_inventory_summary(
    model: str = Query("naive_forecast"),
    store_nbr: Optional[int] = Query(None),
    family: Optional[str] = Query(None),
):
    """Scorecard metrics, risk breakdown, and rule-based executive insights."""
    data = get_data()
    model_col = resolve_model(model)
    active_inv = get_active_inventory()
    df_fc = data["forecast"]
    df_sup = data["suppliers"]

    plan = compute_replenishment_plan(active_inv, df_fc, model_col=model_col, supplier_df=df_sup)

    if store_nbr is not None:
        plan = plan[plan["store_nbr"] == store_nbr]
    if family is not None:
        plan = plan[plan["family"] == family]

    total_items = len(plan)
    high_cnt = int((plan["stockout_risk"] == "HIGH").sum()) if not plan.empty else 0
    med_cnt = int((plan["stockout_risk"] == "MEDIUM").sum()) if not plan.empty else 0
    low_cnt = int((plan["stockout_risk"] == "LOW").sum()) if not plan.empty else 0
    unk_cnt = int((plan["stockout_risk"] == "UNKNOWN").sum()) if not plan.empty else 0

    tot_lt_demand = float(plan["expected_lt_demand"].sum()) if not plan.empty else 0.0
    tot_replenish = float(plan["suggested_replenishment_qty"].sum()) if not plan.empty else 0.0
    tot_stock = float(plan["current_stock"].sum()) if not plan.empty else 0.0
    tot_safety = float(plan["safety_stock"].sum()) if not plan.empty else 0.0

    insights = generate_inventory_business_summary(plan)

    return {
        "data_source_mode": DATA_SOURCE_MODE,
        "is_demo_data": DATA_SOURCE_MODE == "DEMO",
        "total_items": total_items,
        "high_risk_count": high_cnt,
        "medium_risk_count": med_cnt,
        "low_risk_count": low_cnt,
        "unknown_risk_count": unk_cnt,
        "total_expected_lt_demand": round(tot_lt_demand, 1),
        "total_current_stock": round(tot_stock, 1),
        "total_safety_stock": round(tot_safety, 1),
        "total_suggested_replenishment_qty": int(tot_replenish),
        "business_insights": insights,
    }


@app.get("/api/inventory/recommendations")
def get_inventory_recommendations(
    model: str = Query("naive_forecast"),
    store_nbr: Optional[int] = Query(None),
    family: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
    limit: int = Query(500, le=2000),
    offset: int = Query(0),
):
    """Detailed replenishment recommendation records with pagination and filtering."""
    data = get_data()
    model_col = resolve_model(model)
    active_inv = get_active_inventory()
    df_fc = data["forecast"]
    df_sup = data["suppliers"]

    plan = compute_replenishment_plan(active_inv, df_fc, model_col=model_col, supplier_df=df_sup)

    if store_nbr is not None:
        plan = plan[plan["store_nbr"] == store_nbr]
    if family is not None:
        plan = plan[plan["family"] == family]
    if risk_level and risk_level.upper() in ["HIGH", "MEDIUM", "LOW", "UNKNOWN"]:
        plan = plan[plan["stockout_risk"] == risk_level.upper()]

    plan_sorted = prioritize_alerts(plan)
    total_count = len(plan_sorted)
    paginated = plan_sorted.iloc[offset : offset + limit]

    return {
        "total_count": total_count,
        "offset": offset,
        "limit": limit,
        "records": paginated.to_dict(orient="records"),
    }


@app.post("/api/inventory/upload")
async def upload_inventory_csv(file: UploadFile = File(...)):
    """Uploads custom store inventory CSV, validates schemas and values, and updates active mode."""
    global CUSTOM_INVENTORY, DATA_SOURCE_MODE
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a valid CSV document.",
        )

    try:
        content = await file.read()
        df_upload = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV file: {str(e)}",
        )

    is_valid, errors, validated_df = validate_inventory_dataframe(df_upload)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "Inventory CSV validation failed", "errors": errors},
        )

    CUSTOM_INVENTORY = validated_df
    DATA_SOURCE_MODE = "CUSTOM"

    return {
        "status": "success",
        "message": f"Successfully validated and loaded {len(validated_df):,} inventory records.",
        "record_count": len(validated_df),
        "data_source_mode": DATA_SOURCE_MODE,
    }


@app.post("/api/inventory/reset")
def reset_inventory():
    """Resets inventory to baseline demo balances."""
    global CUSTOM_INVENTORY, DATA_SOURCE_MODE
    CUSTOM_INVENTORY = None
    DATA_SOURCE_MODE = "DEMO"
    return {
        "status": "success",
        "message": "Reset to baseline demo inventory successfully.",
        "data_source_mode": DATA_SOURCE_MODE,
    }


# -----------------------------------------------------------------------------
# 9. Alerts & Store Exposure Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/inventory/alerts")
def get_inventory_alerts(
    model: str = Query("naive_forecast"),
    store_nbr: Optional[int] = Query(None),
    family: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
):
    """Alert triage center: prioritized alerts, store-level exposure rollup, affected counts."""
    data = get_data()
    model_col = resolve_model(model)
    active_inv = get_active_inventory()
    df_fc = data["forecast"]
    df_sup = data["suppliers"]

    plan = compute_replenishment_plan(active_inv, df_fc, model_col=model_col, supplier_df=df_sup)

    # Store rollup computed on all stores within store/family filter (before risk filter)
    scoped_plan = plan.copy()
    if store_nbr is not None:
        scoped_plan = scoped_plan[scoped_plan["store_nbr"] == store_nbr]
    if family is not None:
        scoped_plan = scoped_plan[scoped_plan["family"] == family]

    store_summary = generate_store_wise_summary(scoped_plan)

    # Filtered alert items
    filtered_alerts = scoped_plan.copy()
    if risk_level and risk_level.upper() in ["HIGH", "MEDIUM", "LOW", "UNKNOWN"]:
        filtered_alerts = filtered_alerts[filtered_alerts["stockout_risk"] == risk_level.upper()]

    prioritized = prioritize_alerts(filtered_alerts)

    affected_items = scoped_plan[scoped_plan["stockout_risk"].isin(["HIGH", "MEDIUM"])]
    stores_affected = int(affected_items["store_nbr"].nunique())
    families_affected = int(affected_items["family"].nunique())

    return {
        "data_source_mode": DATA_SOURCE_MODE,
        "is_demo_data": DATA_SOURCE_MODE == "DEMO",
        "stores_affected": stores_affected,
        "families_affected": families_affected,
        "total_alerts": len(prioritized),
        "high_risk_count": int((scoped_plan["stockout_risk"] == "HIGH").sum()),
        "medium_risk_count": int((scoped_plan["stockout_risk"] == "MEDIUM").sum()),
        "low_risk_count": int((scoped_plan["stockout_risk"] == "LOW").sum()),
        "prioritized_alerts": prioritized.head(300).to_dict(orient="records"),
        "store_summary": store_summary.to_dict(orient="records"),
    }


# -----------------------------------------------------------------------------
# 10. Reports & Downloads Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/reports/replenishment")
def download_replenishment_csv(
    model: str = Query("naive_forecast"),
    store_nbr: Optional[int] = Query(None),
    family: Optional[str] = Query(None),
    risk_level: Optional[str] = Query(None),
):
    """Streams generated replenishment recommendations CSV."""
    data = get_data()
    model_col = resolve_model(model)
    active_inv = get_active_inventory()
    df_fc = data["forecast"]
    df_sup = data["suppliers"]

    plan = compute_replenishment_plan(active_inv, df_fc, model_col=model_col, supplier_df=df_sup)

    if store_nbr is not None:
        plan = plan[plan["store_nbr"] == store_nbr]
    if family is not None:
        plan = plan[plan["family"] == family]
    if risk_level and risk_level.upper() in ["HIGH", "MEDIUM", "LOW", "UNKNOWN"]:
        plan = plan[plan["stockout_risk"] == risk_level.upper()]

    plan_sorted = prioritize_alerts(plan)
    csv_bytes = plan_sorted.to_csv(index=False).encode("utf-8")
    filename = f"freshbasket_replenishment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/api/reports/executive")
def get_executive_report(
    model: str = Query("naive_forecast"),
    store_nbr: Optional[int] = Query(None),
    family: Optional[str] = Query(None),
):
    """Compiles and returns real executive briefing markdown document."""
    data = get_data()
    model_col = resolve_model(model)
    active_inv = get_active_inventory()
    df_fc = data["forecast"]
    df_sup = data["suppliers"]

    plan = compute_replenishment_plan(active_inv, df_fc, model_col=model_col, supplier_df=df_sup)
    if store_nbr is not None:
        plan = plan[plan["store_nbr"] == store_nbr]
    if family is not None:
        plan = plan[plan["family"] == family]

    report_text = generate_executive_business_report(
        df_plan=plan,
        data_source_mode=DATA_SOURCE_MODE,
        forecast_model_label=MODEL_LABELS.get(model_col, "Naive Baseline"),
        selected_store_label=f"Store {store_nbr}" if store_nbr is not None else "All Stores (54)",
        selected_family_label=family if family is not None else "All Families (33)",
    )

    return PlainTextResponse(content=report_text, media_type="text/markdown")


@app.get("/api/reports/email")
def get_email_draft(
    model: str = Query("naive_forecast"),
    store_nbr: Optional[int] = Query(None),
    family: Optional[str] = Query(None),
):
    """Generates staging email draft text."""
    data = get_data()
    model_col = resolve_model(model)
    active_inv = get_active_inventory()
    df_fc = data["forecast"]
    df_sup = data["suppliers"]

    plan = compute_replenishment_plan(active_inv, df_fc, model_col=model_col, supplier_df=df_sup)
    if store_nbr is not None:
        plan = plan[plan["store_nbr"] == store_nbr]
    if family is not None:
        plan = plan[plan["family"] == family]

    email_text = generate_email_alert_draft(
        df_plan=plan,
        data_source_mode=DATA_SOURCE_MODE,
        forecast_model_label=MODEL_LABELS.get(model_col, "Naive Baseline"),
        selected_store_label=f"Store {store_nbr}" if store_nbr is not None else "All Stores (54)",
        selected_family_label=family if family is not None else "All Families (33)",
    )

    return PlainTextResponse(content=email_text, media_type="text/plain")


@app.post("/api/inventory/alerts/snapshot")
def capture_alert_snapshot(model: str = Query("naive_forecast")):
    """Captures and appends current alert state to persistent CSV audit log."""
    data = get_data()
    model_col = resolve_model(model)
    active_inv = get_active_inventory()
    df_fc = data["forecast"]
    df_sup = data["suppliers"]

    plan = compute_replenishment_plan(active_inv, df_fc, model_col=model_col, supplier_df=df_sup)
    success, msg, hist_df = record_alert_snapshot(
        df_plan=plan,
        data_source_mode=DATA_SOURCE_MODE,
        forecast_model_label=MODEL_LABELS.get(model_col, "Naive Baseline"),
    )
    return {"status": "success" if success else "info", "message": msg, "total_records": len(hist_df)}


@app.get("/api/inventory/alerts/history")
def get_alert_history():
    """Retrieves logged historical alert snapshots."""
    history_df = load_alert_history()
    return {
        "total_records": len(history_df),
        "history": history_df.tail(300).to_dict(orient="records"),
    }


# -----------------------------------------------------------------------------
# 11. AI Business Insights Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/insights/overview")
def get_insights_overview(
    model: str = Query("naive_forecast"),
    store_nbr: Optional[int] = Query(None),
    family: Optional[str] = Query(None),
):
    """
    Synthesizes executive business insights across demand trends, forecast shifts,
    model divergence, and inventory replenishment exposures.
    """
    data = get_data()
    model_col = resolve_model(model)
    active_inv = get_active_inventory()
    df_hist = data["historical"]
    df_fc = data["forecast"]
    df_sup = data["suppliers"]

    plan = compute_replenishment_plan(active_inv, df_fc, model_col=model_col, supplier_df=df_sup)

    # Filter plan if parameters present
    scoped_plan = plan.copy()
    if store_nbr is not None:
        scoped_plan = scoped_plan[scoped_plan["store_nbr"] == store_nbr]
    if family is not None:
        scoped_plan = scoped_plan[scoped_plan["family"] == family]

    insights = synthesize_business_insights(
        df_historical=df_hist,
        df_forecast=df_fc,
        df_plan=scoped_plan,
        store_nbr=store_nbr,
        family=family,
        model=model_col,
        data_source_mode=DATA_SOURCE_MODE,
    )

    trends_data = detect_demand_trends(df_hist, store_nbr=store_nbr, family=family)
    fc_data = generate_forecast_insights(df_fc, df_hist, store_nbr=store_nbr, family=family, model=model_col)
    inv_data = generate_inventory_risk_insights(scoped_plan, data_source_mode=DATA_SOURCE_MODE)

    return {
        "data_source_mode": DATA_SOURCE_MODE,
        "is_demo_data": DATA_SOURCE_MODE == "DEMO",
        "scope": {
            "store_nbr": store_nbr,
            "family": family,
            "model": model_col,
            "model_label": MODEL_LABELS.get(model_col, "Naive Baseline"),
        },
        "insight_cards": insights,
        "trend_summary": trends_data["summary"],
        "comparison_period": trends_data["comparison_period"],
        "forecast_highlights": {
            "top_family": fc_data["top_demand_families"][0]["family"] if fc_data["top_demand_families"] else None,
            "top_family_units": fc_data["top_demand_families"][0]["selected_units"] if fc_data["top_demand_families"] else 0,
            "highest_divergence_family": fc_data["model_divergence"][0]["family"] if fc_data["model_divergence"] else None,
            "highest_divergence_pct": fc_data["model_divergence"][0]["difference_pct"] if fc_data["model_divergence"] else 0,
        },
        "inventory_risk_highlights": {
            "high_risk_count": inv_data["risk_counts"]["HIGH"],
            "medium_risk_count": inv_data["risk_counts"]["MEDIUM"],
            "below_reorder_point_count": inv_data["below_reorder_point_count"],
        },
    }


@app.get("/api/insights/demand-trends")
def get_insights_demand_trends(
    store_nbr: Optional[int] = Query(None),
    family: Optional[str] = Query(None),
    window_weeks: int = Query(12, ge=4, le=52),
):
    """
    Detailed historical demand trends with OLS regression slope, p-value, % change,
    and rigorous statistical significance classification.
    """
    data = get_data()
    df_hist = data["historical"]
    return detect_demand_trends(df_hist, store_nbr=store_nbr, family=family, window_weeks=window_weeks)


@app.get("/api/insights/product-opportunities")
def get_insights_product_opportunities(
    model: str = Query("naive_forecast"),
    store_nbr: Optional[int] = Query(None),
    family: Optional[str] = Query(None),
):
    """
    Forward forecast intelligence: projected volume leaders, trajectory shifts,
    high demand stores, model divergence (Naive vs LightGBM), and horizon uncertainty.
    """
    data = get_data()
    model_col = resolve_model(model)
    df_hist = data["historical"]
    df_fc = data["forecast"]
    return generate_forecast_insights(df_fc, df_hist, store_nbr=store_nbr, family=family, model=model_col)


@app.get("/api/insights/inventory-risks")
def get_insights_inventory_risks(
    model: str = Query("naive_forecast"),
    store_nbr: Optional[int] = Query(None),
    family: Optional[str] = Query(None),
):
    """
    Strategic inventory risk evaluation: high risk SKUs, items below reorder points,
    lead-time exposure, and review recommendations.
    """
    data = get_data()
    model_col = resolve_model(model)
    active_inv = get_active_inventory()
    df_fc = data["forecast"]
    df_sup = data["suppliers"]

    plan = compute_replenishment_plan(active_inv, df_fc, model_col=model_col, supplier_df=df_sup)
    if store_nbr is not None:
        plan = plan[plan["store_nbr"] == store_nbr]
    if family is not None:
        plan = plan[plan["family"] == family]

    return generate_inventory_risk_insights(plan, data_source_mode=DATA_SOURCE_MODE)


# -----------------------------------------------------------------------------
# 12. Business Query Assistant ("Ask FreshBasket AI")
# -----------------------------------------------------------------------------

class AssistantQueryRequest(BaseModel):
    question: str
    store_nbr: Optional[int] = None
    family: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    model: Optional[str] = None


@app.post("/api/assistant/query")
def query_assistant(payload: AssistantQueryRequest):
    """
    Business Query Assistant endpoint.
    Answers natural language questions regarding demand forecasts, model evaluations,
    historical trends, and inventory risks using verified project datasets.
    """
    data = get_data()
    return answer_business_question(
        question=payload.question,
        data=data,
        store_nbr=payload.store_nbr,
        family=payload.family,
        model=payload.model,
        data_source_mode=DATA_SOURCE_MODE,
    )


# -----------------------------------------------------------------------------
# 13. What-If Scenario Simulator
# -----------------------------------------------------------------------------

class ScenarioSimulateRequest(BaseModel):
    demand_change_pct: float = Field(0.0, ge=-50.0, le=100.0, description="Demand change percentage (-50% to +100%)")
    stock_change_pct: float = Field(0.0, ge=-75.0, le=100.0, description="Current stock change percentage (-75% to +100%)")
    lead_time_adjustment_days: int = Field(0, ge=-7, le=30, description="Supplier lead-time adjustment days (-7 to +30)")
    store_nbr: Optional[int] = Field(None, description="Optional store identifier (1-54)")
    family: Optional[str] = Field(None, description="Optional product family name")
    model: Optional[str] = Field("naive_forecast", description="Forecast model to simulate against")


@app.post("/api/scenarios/simulate")
def simulate_scenario(payload: ScenarioSimulateRequest):
    """
    What-If Scenario Simulation endpoint.
    Simulates operational impacts of demand shocks, inventory stock deviations,
    and supplier lead-time changes without altering underlying project datasets.
    """
    data = get_data()
    active_inv = get_active_inventory()
    df_fc = data["forecast"]
    df_sup = data.get("suppliers")

    # Validate store_nbr if provided
    if payload.store_nbr is not None:
        if payload.store_nbr < 1 or payload.store_nbr > 54:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid store_nbr '{payload.store_nbr}'. Store numbers must be between 1 and 54.",
            )

    # Validate family if provided
    if payload.family is not None and payload.family.strip():
        valid_families = set(df_fc["family"].unique())
        if payload.family not in valid_families:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown product family '{payload.family}'. Must be one of the 33 valid categories.",
            )

    model_col = resolve_model(payload.model)

    try:
        results = run_inventory_simulation(
            df_inventory=active_inv,
            df_forecast=df_fc,
            supplier_df=df_sup,
            demand_change_pct=payload.demand_change_pct,
            stock_change_pct=payload.stock_change_pct,
            lead_time_adjustment_days=payload.lead_time_adjustment_days,
            store_nbr=payload.store_nbr,
            family=payload.family,
            model_col=model_col,
        )
        return results
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Simulation error: {str(e)}",
        )


