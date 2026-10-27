"""
FreshBasket AI - Dashboard Data Preparation Module
==================================================
Step 3: Creates clean, dashboard-ready datasets from verified project outputs:
- data/dashboard/historical_sales.csv
- data/dashboard/forecast_data.csv
- data/dashboard/model_performance.csv
- data/dashboard/family_performance.csv
- data/dashboard/store_performance.csv

Key Design Principles:
1. Separation of Concerns: Reads existing verified outputs; does not duplicate ML logic.
2. Historical vs. Future Segregation: Historical actuals and future forecasts are
   kept distinct so dashboards never confuse predictions with actual sales.
3. Rich Context: Joins store and product family attributes (city, state, type, cluster)
   to enable multi-dimensional slicing in Streamlit.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Paths
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = WORKSPACE_ROOT / "data" / "processed" / "weekly_features.csv"
FORECAST_PATH = WORKSPACE_ROOT / "data" / "forecasts" / "forecast_13_weeks.csv"
BASELINE_PATH = WORKSPACE_ROOT / "data" / "forecasts" / "baseline_forecasts.csv"
VAL_PREDS_PATH = WORKSPACE_ROOT / "data" / "forecasts" / "validation_predictions.csv"
METRICS_PATH = WORKSPACE_ROOT / "data" / "forecasts" / "model_metrics.csv"
DASHBOARD_DIR = WORKSPACE_ROOT / "data" / "dashboard"


def prepare_historical_sales(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts clean historical sales records with store and calendar attributes.
    """
    cols = [
        "week_start_date",
        "week_end_date",
        "year",
        "month",
        "quarter",
        "week_of_year",
        "store_nbr",
        "family",
        "city",
        "state",
        "type",
        "cluster",
        "weekly_sales",
        "onpromotion_sum",
        "has_promotion",
        "weekly_transactions",
        "oil_price_avg",
        "is_holiday_week",
        "is_partial_week",
        "store_status",
        "is_training_eligible",
    ]
    df_hist = df_features[cols].copy()
    df_hist["is_forecast"] = False
    return df_hist.sort_values(["week_start_date", "store_nbr", "family"]).reset_index(drop=True)


def prepare_forecast_data(
    df_features: pd.DataFrame,
    df_lgb: pd.DataFrame,
    df_base: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combines 13-week forward forecasts from LightGBM, Naive, and Seasonal Naive,
    annotated with store attributes and is_forecast = True.
    """
    # Extract store metadata mapping
    meta_cols = ["store_nbr", "family", "city", "state", "type", "cluster"]
    store_meta = df_features[meta_cols].drop_duplicates()

    # Merge LightGBM forecasts with baseline forecasts
    merged_fc = pd.merge(
        df_lgb.rename(columns={"predicted_sales": "lightgbm_forecast"}),
        df_base[[
            "forecast_origin", "week_start_date", "week_end_date",
            "store_nbr", "family", "forecast_horizon_week",
            "naive_prediction", "seasonal_naive_prediction"
        ]].rename(columns={
            "naive_prediction": "naive_forecast",
            "seasonal_naive_prediction": "seasonal_naive_forecast",
        }),
        on=["forecast_origin", "week_start_date", "week_end_date", "store_nbr", "family", "forecast_horizon_week"],
        how="inner",
    )

    # Attach store metadata
    merged_fc = pd.merge(merged_fc, store_meta, on=["store_nbr", "family"], how="left")

    merged_fc["is_forecast"] = True

    cols_order = [
        "forecast_origin",
        "week_start_date",
        "week_end_date",
        "forecast_horizon_week",
        "store_nbr",
        "family",
        "city",
        "state",
        "type",
        "cluster",
        "lightgbm_forecast",
        "naive_forecast",
        "seasonal_naive_forecast",
        "is_forecast",
    ]
    return merged_fc[cols_order].sort_values(["forecast_horizon_week", "store_nbr", "family"]).reset_index(drop=True)


def prepare_model_performance(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Filters global and horizon-decay performance metrics for high-level dashboard KPIs.
    """
    mask = df_metrics["granularity"].isin(["global", "horizon_week"])
    sub = df_metrics[mask].copy()
    return sub.sort_values(["granularity", "forecast_horizon_week", "model"]).reset_index(drop=True)


def prepare_family_performance(
    df_metrics: pd.DataFrame,
    df_val_preds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepares product-family performance table, enriched with validation sales volume.
    """
    fam_metrics = df_metrics[df_metrics["granularity"] == "family"].copy().rename(columns={"segment": "family"})

    # Compute validation sales volume per family
    fam_volumes = df_val_preds.groupby("family")["actual_sales"].sum().reset_index().rename(columns={"actual_sales": "actual_sales_volume"})

    merged = pd.merge(fam_metrics, fam_volumes, on="family", how="left")
    cols = [
        "family",
        "model",
        "observations",
        "actual_sales_volume",
        "mae",
        "rmse",
        "wape",
        "rmsle",
    ]
    return merged[cols].sort_values(["actual_sales_volume", "family", "model"], ascending=[False, True, True]).reset_index(drop=True)


def prepare_store_performance(
    df_metrics: pd.DataFrame,
    df_val_preds: pd.DataFrame,
    df_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepares store-level performance table enriched with store metadata and validation volume.
    """
    store_metrics = df_metrics[df_metrics["granularity"] == "store_nbr"].copy()
    # Extract store integer from segment "Store X"
    store_metrics["store_nbr"] = store_metrics["segment"].str.replace("Store ", "").astype(int)

    # Store metadata
    store_meta = df_features[["store_nbr", "city", "state", "type", "cluster"]].drop_duplicates()

    # Store validation volumes
    store_volumes = df_val_preds.groupby("store_nbr")["actual_sales"].sum().reset_index().rename(columns={"actual_sales": "actual_sales_volume"})

    merged = pd.merge(store_metrics, store_meta, on="store_nbr", how="left")
    merged = pd.merge(merged, store_volumes, on="store_nbr", how="left")

    cols = [
        "store_nbr",
        "city",
        "state",
        "type",
        "cluster",
        "model",
        "observations",
        "actual_sales_volume",
        "mae",
        "rmse",
        "wape",
        "rmsle",
    ]
    return merged[cols].sort_values(["actual_sales_volume", "store_nbr", "model"], ascending=[False, True, True]).reset_index(drop=True)


def main():
    print("=" * 70)
    print(" FreshBasket AI - Dashboard Data Preparation")
    print("=" * 70)

    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[*] Reading verified source datasets...")
    df_features = pd.read_csv(FEATURES_PATH)
    df_lgb = pd.read_csv(FORECAST_PATH)
    df_base = pd.read_csv(BASELINE_PATH)
    df_val_preds = pd.read_csv(VAL_PREDS_PATH)
    df_metrics = pd.read_csv(METRICS_PATH)

    # 1. Historical sales
    print("[*] Generating historical_sales.csv...")
    df_hist = prepare_historical_sales(df_features)
    hist_out = DASHBOARD_DIR / "historical_sales.csv"
    df_hist.to_csv(hist_out, index=False)
    print(f"  [+] Saved {hist_out} ({len(df_hist):,} rows)")

    # 2. Forecast data
    print("[*] Generating forecast_data.csv...")
    df_fc = prepare_forecast_data(df_features, df_lgb, df_base)
    fc_out = DASHBOARD_DIR / "forecast_data.csv"
    df_fc.to_csv(fc_out, index=False)
    print(f"  [+] Saved {fc_out} ({len(df_fc):,} rows)")

    # 3. Model performance
    print("[*] Generating model_performance.csv...")
    df_perf = prepare_model_performance(df_metrics)
    perf_out = DASHBOARD_DIR / "model_performance.csv"
    df_perf.to_csv(perf_out, index=False)
    print(f"  [+] Saved {perf_out} ({len(df_perf):,} rows)")

    # 4. Family performance
    print("[*] Generating family_performance.csv...")
    df_fam_perf = prepare_family_performance(df_metrics, df_val_preds)
    fam_out = DASHBOARD_DIR / "family_performance.csv"
    df_fam_perf.to_csv(fam_out, index=False)
    print(f"  [+] Saved {fam_out} ({len(df_fam_perf):,} rows)")

    # 5. Store performance
    print("[*] Generating store_performance.csv...")
    df_store_perf = prepare_store_performance(df_metrics, df_val_preds, df_features)
    store_out = DASHBOARD_DIR / "store_performance.csv"
    df_store_perf.to_csv(store_out, index=False)
    print(f"  [+] Saved {store_out} ({len(df_store_perf):,} rows)")

    print("=" * 70)
    print("[+] All dashboard datasets successfully prepared in data/dashboard/")
    print("=" * 70)


if __name__ == "__main__":
    main()
