"""
FreshBasket AI - Dashboard Data Loader & Transformer
====================================================
Modular data loading, schema validation, and filtering helpers for the Streamlit dashboard.
Uses Streamlit caching (@st.cache_data) for responsive sub-second page rendering.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import streamlit as st

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DIR = WORKSPACE_ROOT / "data" / "dashboard"

HISTORICAL_FILE = DASHBOARD_DIR / "historical_sales.csv"
FORECAST_FILE = DASHBOARD_DIR / "forecast_data.csv"
MODEL_PERF_FILE = DASHBOARD_DIR / "model_performance.csv"
FAMILY_PERF_FILE = DASHBOARD_DIR / "family_performance.csv"
STORE_PERF_FILE = DASHBOARD_DIR / "store_performance.csv"
INVENTORY_FILE = WORKSPACE_ROOT / "data" / "inventory" / "inventory.csv"
SUPPLIER_FILE = WORKSPACE_ROOT / "data" / "inventory" / "supplier_lead_times.csv"



def verify_dashboard_files_exist() -> Tuple[bool, List[str]]:
    """Checks whether all 5 required dashboard CSVs exist on disk."""
    required_files = [
        ("Historical Sales", HISTORICAL_FILE),
        ("Forecast Data", FORECAST_FILE),
        ("Model Performance", MODEL_PERF_FILE),
        ("Family Performance", FAMILY_PERF_FILE),
        ("Store Performance", STORE_PERF_FILE),
    ]
    missing = [name for name, path in required_files if not path.exists()]
    return (len(missing) == 0, missing)


@st.cache_data(show_spinner=True)
def load_historical_sales() -> pd.DataFrame:
    """Loads historical weekly sales data with memory-optimized dtypes."""
    if not HISTORICAL_FILE.exists():
        raise FileNotFoundError(f"Missing required dataset: {HISTORICAL_FILE}")
    
    dtypes = {
        "store_nbr": "int16",
        "family": "category",
        "city": "category",
        "state": "category",
        "type": "category",
        "cluster": "int16",
        "weekly_sales": "float32",
        "onpromotion_sum": "int32",
        "has_promotion": "bool",
        "weekly_transactions": "int32",
        "oil_price_avg": "float32",
        "is_holiday_week": "bool",
        "is_partial_week": "bool",
        "store_status": "category",
        "is_training_eligible": "bool",
        "is_forecast": "bool",
    }
    df = pd.read_csv(HISTORICAL_FILE, dtype=dtypes)
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    df["week_end_date"] = pd.to_datetime(df["week_end_date"])
    return df


@st.cache_data(show_spinner=True)
def load_forecast_data() -> pd.DataFrame:
    """Loads 13-week production forecast data with all candidate model outputs."""
    if not FORECAST_FILE.exists():
        raise FileNotFoundError(f"Missing required dataset: {FORECAST_FILE}")
    
    dtypes = {
        "forecast_horizon_week": "int8",
        "store_nbr": "int16",
        "family": "category",
        "city": "category",
        "state": "category",
        "type": "category",
        "cluster": "int16",
        "lightgbm_forecast": "float32",
        "naive_forecast": "float32",
        "seasonal_naive_forecast": "float32",
        "is_forecast": "bool",
    }
    df = pd.read_csv(FORECAST_FILE, dtype=dtypes)
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    df["week_end_date"] = pd.to_datetime(df["week_end_date"])
    df["forecast_origin"] = pd.to_datetime(df["forecast_origin"])
    return df


@st.cache_data(show_spinner=False)
def load_model_performance() -> pd.DataFrame:
    """Loads overall and horizon-specific model evaluation metrics."""
    if not MODEL_PERF_FILE.exists():
        raise FileNotFoundError(f"Missing required dataset: {MODEL_PERF_FILE}")
    df = pd.read_csv(MODEL_PERF_FILE)
    return df


@st.cache_data(show_spinner=False)
def load_family_performance() -> pd.DataFrame:
    """Loads product-family sliced metrics for all 33 categories."""
    if not FAMILY_PERF_FILE.exists():
        raise FileNotFoundError(f"Missing required dataset: {FAMILY_PERF_FILE}")
    df = pd.read_csv(FAMILY_PERF_FILE)
    return df


@st.cache_data(show_spinner=False)
def load_store_performance() -> pd.DataFrame:
    """Loads store-sliced metrics across all 54 retail branches."""
    if not STORE_PERF_FILE.exists():
        raise FileNotFoundError(f"Missing required dataset: {STORE_PERF_FILE}")
    df = pd.read_csv(STORE_PERF_FILE)
    return df


@st.cache_data(show_spinner=False)
def load_inventory_data() -> pd.DataFrame:
    """Loads baseline demo inventory balances."""
    if not INVENTORY_FILE.exists():
        # Fall back to generating demo data if missing
        from src.data.generate_demo_inventory import generate_demo_inventory_data
        df_inv, _ = generate_demo_inventory_data()
        return df_inv
    return pd.read_csv(INVENTORY_FILE)


@st.cache_data(show_spinner=False)
def load_supplier_data() -> pd.DataFrame:
    """Loads supplier lead-time records."""
    if not SUPPLIER_FILE.exists():
        from src.data.generate_demo_inventory import generate_demo_inventory_data
        _, df_sup = generate_demo_inventory_data()
        return df_sup
    return pd.read_csv(SUPPLIER_FILE)


def load_all_dashboard_data() -> Dict[str, pd.DataFrame]:
    """Loads and caches all dashboard and inventory datasets concurrently."""
    return {
        "historical": load_historical_sales(),
        "forecast": load_forecast_data(),
        "model_performance": load_model_performance(),
        "family_performance": load_family_performance(),
        "store_performance": load_store_performance(),
        "inventory": load_inventory_data(),
        "suppliers": load_supplier_data(),
    }



def filter_historical_sales(
    df: pd.DataFrame,
    selected_stores: Optional[List[int]] = None,
    selected_families: Optional[List[str]] = None,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    exclude_partial_weeks: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Filters historical weekly sales by store list, product family list, and date interval.
    By default, excludes incomplete boundary weeks (e.g. 2017-08-14 with only 2 observed days)
    to prevent artificial visual volume cliffs and misleading partial-week representations.
    Safe against empty selections and extra keyword arguments.
    """
    if df.empty:
        return df

    filtered = df

    if exclude_partial_weeks and "is_partial_week" in filtered.columns:
        filtered = filtered[filtered["is_partial_week"] == False]

    if selected_stores and len(selected_stores) > 0 and "All" not in selected_stores:
        filtered = filtered[filtered["store_nbr"].isin(selected_stores)]

    if selected_families and len(selected_families) > 0 and "All" not in selected_families:
        filtered = filtered[filtered["family"].isin(selected_families)]

    if start_date is not None:
        filtered = filtered[filtered["week_start_date"] >= pd.to_datetime(start_date)]

    if end_date is not None:
        filtered = filtered[filtered["week_start_date"] <= pd.to_datetime(end_date)]

    return filtered



def filter_forecast_data(
    df: pd.DataFrame,
    selected_stores: Optional[List[int]] = None,
    selected_families: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filters forecast data by store and family selections."""
    if df.empty:
        return df

    filtered = df

    if selected_stores and len(selected_stores) > 0 and "All" not in selected_stores:
        filtered = filtered[filtered["store_nbr"].isin(selected_stores)]

    if selected_families and len(selected_families) > 0 and "All" not in selected_families:
        filtered = filtered[filtered["family"].isin(selected_families)]

    return filtered


def get_model_column_mapping() -> Dict[str, str]:
    """Maps user-facing model selection labels to column names in forecast_data.csv."""
    return {
        "Naive Baseline": "naive_forecast",
        "LightGBM (Recursive)": "lightgbm_forecast",
        "Seasonal Naive (Lag 52)": "seasonal_naive_forecast",
    }


def compute_kpis(
    df_hist: pd.DataFrame,
    df_forecast: pd.DataFrame,
    model_col: str = "naive_forecast",
) -> Dict[str, Any]:
    """Computes high-level executive KPIs safe against zero division and empty sets."""
    total_hist_sales = float(df_hist["weekly_sales"].sum()) if not df_hist.empty else 0.0
    total_forecast_sales = float(df_forecast[model_col].sum()) if (not df_forecast.empty and model_col in df_forecast.columns) else 0.0
    
    unique_stores = int(df_hist["store_nbr"].nunique()) if not df_hist.empty else 0
    unique_families = int(df_hist["family"].nunique()) if not df_hist.empty else 0
    
    total_weeks_hist = int(df_hist["week_start_date"].nunique()) if not df_hist.empty else 0
    total_weeks_forecast = int(df_forecast["week_start_date"].nunique()) if not df_forecast.empty else 0
    
    avg_weekly_sales = (total_hist_sales / total_weeks_hist) if total_weeks_hist > 0 else 0.0

    return {
        "total_historical_sales": total_hist_sales,
        "historical_sales_sum": total_hist_sales,
        "total_forecast_sales": total_forecast_sales,
        "forecast_sales_sum": total_forecast_sales,
        "unique_stores": unique_stores,
        "store_count": unique_stores,
        "unique_families": unique_families,
        "family_count": unique_families,
        "total_weeks_historical": total_weeks_hist,
        "total_weeks_forecast": total_weeks_forecast,
        "avg_weekly_sales": avg_weekly_sales,
    }
