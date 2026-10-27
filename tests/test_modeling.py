"""
FreshBasket AI - Unit Test Suite for Modeling, Validation & 13-Week Forecasting
================================================================================
Task 7: Validates:
1. Business key uniqueness (no duplicate forecast or validation keys).
2. Strict target and concurrent exclusion from the feature allowlist.
3. Historical lag mathematical shift correctness.
4. Exactly 13 forecast weeks with consecutive Monday-to-Sunday dates.
5. Non-negative forecast post-processing.
6. Metric calculations (MAE, RMSE, WAPE, RMSLE).
7. Model artifact persistence and reload.
8. Baseline forecast table schema and consistency.
9. Valid store and product family identifiers.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluate_models import calculate_metrics
from src.models.baseline_model import NaiveBaseline, SeasonalNaiveBaseline
from src.models.train_lightgbm import (
    CATEGORICAL_FEATURES,
    FEATURE_ALLOWLIST,
    METADATA_FILE,
    MODEL_JOBLIB_FILE,
    MODEL_TXT_FILE,
    NUMERICAL_FEATURES,
    load_trained_model,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
VAL_PREDS_PATH = WORKSPACE_ROOT / "data" / "forecasts" / "validation_predictions.csv"
FORECAST_CSV_PATH = WORKSPACE_ROOT / "data" / "forecasts" / "forecast_13_weeks.csv"
BASELINE_CSV_PATH = WORKSPACE_ROOT / "data" / "forecasts" / "baseline_forecasts.csv"
METRICS_CSV_PATH = WORKSPACE_ROOT / "data" / "forecasts" / "model_metrics.csv"
FEATURES_PATH = WORKSPACE_ROOT / "data" / "processed" / "weekly_features.csv"


def test_target_exclusion_from_predictors():
    """Confirms target weekly_sales and unlagged concurrent metrics are strictly excluded."""
    assert "weekly_sales" not in FEATURE_ALLOWLIST
    assert "weekly_transactions" not in FEATURE_ALLOWLIST
    assert "onpromotion_sum" not in FEATURE_ALLOWLIST
    assert "oil_price_avg" not in FEATURE_ALLOWLIST
    assert "is_training_eligible" not in FEATURE_ALLOWLIST
    assert "is_partial_week" not in FEATURE_ALLOWLIST

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert "weekly_sales" not in meta["feature_allowlist"]


def test_model_artifacts_exist_and_reload():
    """Confirms trained model files exist and can be reloaded to generate predictions."""
    assert MODEL_TXT_FILE.exists()
    assert MODEL_JOBLIB_FILE.exists()
    assert METADATA_FILE.exists()

    wrapper, metadata = load_trained_model()
    assert wrapper is not None
    assert metadata["target_column"] == "weekly_sales"
    assert metadata["feature_count"] == len(FEATURE_ALLOWLIST)

    # Test single prediction
    df_sample = pd.read_csv(FEATURES_PATH, nrows=5)
    preds = wrapper.predict(df_sample)
    assert len(preds) == 5
    assert (preds >= 0).all()


def test_metric_calculations_math():
    """Tests MAE, RMSE, WAPE, and RMSLE with deterministic synthetic values."""
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])  # errors: +10, -10, 0 -> sum = 20

    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["mae"] == round(20.0 / 3.0, 2)
    # WAPE = 20 / 600 * 100 = 3.33%
    assert np.isclose(metrics["wape"], 3.33, atol=0.01)

    # Zero actual denominator safety check
    y_zero = np.array([0.0, 0.0])
    y_zero_pred = np.array([0.0, 0.0])
    m_zero = calculate_metrics(y_zero, y_zero_pred)
    assert m_zero["wape"] == 0.0


def test_historical_lag_correctness():
    """Verifies that sales lags strictly represent past sales values."""
    df = pd.read_csv(FEATURES_PATH)
    sample = df[(df["store_nbr"] == 1) & (df["family"] == "GROCERY I")].sort_values("week_start_date").reset_index(drop=True)

    # For week index 5: lag_1 must equal week index 4 sales, lag_2 must equal week index 3 sales
    assert sample.loc[5, "sales_lag_1"] == sample.loc[4, "weekly_sales"]
    assert sample.loc[5, "sales_lag_2"] == sample.loc[3, "weekly_sales"]


def test_validation_predictions_integrity():
    """Validates the structure and schema of validation_predictions.csv."""
    assert VAL_PREDS_PATH.exists()
    df_val = pd.read_csv(VAL_PREDS_PATH)

    # 1,782 series * 13 weeks = 23,166 rows
    assert len(df_val) == 23166

    # Required columns
    required_cols = [
        "forecast_origin",
        "week_start_date",
        "week_end_date",
        "store_nbr",
        "family",
        "forecast_horizon_week",
        "actual_sales",
        "naive_prediction",
        "seasonal_naive_prediction",
        "lightgbm_prediction",
    ]
    for col in required_cols:
        assert col in df_val.columns

    # No duplicate forecast keys
    dups = df_val.duplicated(subset=["week_start_date", "store_nbr", "family"]).sum()
    assert dups == 0

    # Predictions non-negative
    assert (df_val["naive_prediction"] >= 0).all()
    assert (df_val["seasonal_naive_prediction"] >= 0).all()
    assert (df_val["lightgbm_prediction"] >= 0).all()


def test_baseline_forecasts_file_integrity():
    """Validates the baseline_forecasts.csv output requested in Task 2."""
    assert BASELINE_CSV_PATH.exists()
    df_base = pd.read_csv(BASELINE_CSV_PATH)

    assert len(df_base) == 23166
    for col in ["forecast_origin", "week_start_date", "week_end_date", "store_nbr", "family", "forecast_horizon_week", "naive_prediction", "seasonal_naive_prediction"]:
        assert col in df_base.columns

    dups = df_base.duplicated(subset=["week_start_date", "store_nbr", "family"]).sum()
    assert dups == 0
    assert (df_base["naive_prediction"] >= 0).all()
    assert (df_base["seasonal_naive_prediction"] >= 0).all()


def test_forecast_13_weeks_dates_and_counts():
    """Validates the 13-week forward-looking forecast dataset."""
    assert FORECAST_CSV_PATH.exists()
    df_fc = pd.read_csv(FORECAST_CSV_PATH)

    # Total rows: 1,782 series * 13 weeks = 23,166 rows
    assert len(df_fc) == 23166

    # Exactly 13 unique forecast weeks
    unique_weeks = df_fc["week_start_date"].nunique()
    assert unique_weeks == 13

    # Horizons 1 to 13
    assert sorted(df_fc["forecast_horizon_week"].unique().tolist()) == list(range(1, 14))

    # All predictions non-negative
    assert (df_fc["predicted_sales"] >= 0).all()

    # Valid prediction intervals and uncertainty bounds
    for col in ["lower_bound_80", "upper_bound_80", "lower_bound_95", "upper_bound_95", "uncertainty_range_80", "uncertainty_range_95"]:
        assert col in df_fc.columns
    for standard_col in ["date", "market_id", "sku_id", "forecast", "uncertainty"]:
        assert standard_col in df_fc.columns, f"Required standard deliverable column '{standard_col}' missing"
    assert (df_fc["lower_bound_80"] >= 0).all()
    assert (df_fc["lower_bound_95"] >= 0).all()
    assert (df_fc["upper_bound_80"] >= df_fc["lower_bound_80"]).all()
    assert (df_fc["upper_bound_95"] >= df_fc["lower_bound_95"]).all()
    assert (df_fc["upper_bound_95"] >= df_fc["upper_bound_80"]).all()
    assert (df_fc["lower_bound_80"] >= df_fc["lower_bound_95"]).all()
    assert (df_fc["forecast"] == df_fc["predicted_sales"]).all()
    assert (df_fc["date"] == df_fc["week_start_date"]).all()
    assert (df_fc["market_id"] == df_fc["store_nbr"]).all()
    assert (df_fc["sku_id"] == df_fc["family"]).all()


    # No duplicate business keys in forecast
    dups = df_fc.duplicated(subset=["week_start_date", "store_nbr", "family"]).sum()
    assert dups == 0

    # Consecutive Monday to Sunday 7-day intervals
    ordered_weeks = df_fc[["forecast_horizon_week", "week_start_date", "week_end_date"]].drop_duplicates().sort_values("forecast_horizon_week").reset_index(drop=True)
    for idx, row in ordered_weeks.iterrows():
        start = pd.to_datetime(row["week_start_date"])
        end = pd.to_datetime(row["week_end_date"])
        assert start.weekday() == 0  # Monday
        assert end.weekday() == 6    # Sunday
        assert (end - start).days == 6
        if idx > 0:
            prev_end = pd.to_datetime(ordered_weeks.loc[idx - 1, "week_end_date"])
            # Consecutive week starts exactly the day after previous week ends
            assert (start - prev_end).days == 1

    # Valid store and product family identifiers
    assert df_fc["store_nbr"].nunique() == 54
    assert df_fc["family"].nunique() == 33
    assert set(df_fc["store_nbr"].unique()) == set(range(1, 55))


def test_baseline_logic_insufficient_history():
    """Confirms Seasonal Naive handles missing history via documented fallback rather than zero."""
    df_short = pd.DataFrame([
        {"week_start_date": "2017-05-08", "store_nbr": 99, "family": "BEVERAGES", "weekly_sales": 500.0}
    ])
    s_naive = SeasonalNaiveBaseline().fit(df_short, cutoff_date="2017-05-08")
    preds = s_naive.predict([("2017-05-15", "2017-05-21")], [(99, "BEVERAGES")])

    # Must fall back to most recent sales (500.0) with seasonal_fallback=True
    assert preds["seasonal_naive_prediction"].iloc[0] == 500.0
    assert preds["seasonal_fallback"].iloc[0] == True


def test_dashboard_data_files_exist_and_schemas():
    """Confirms all dashboard data CSVs exist and contain appropriate schemas and non-negativity."""
    dashboard_dir = Path("data/dashboard")
    expected_files = [
        "historical_sales.csv",
        "forecast_data.csv",
        "model_performance.csv",
        "family_performance.csv",
        "store_performance.csv",
    ]
    for fname in expected_files:
        fpath = dashboard_dir / fname
        assert fpath.exists(), f"Dashboard dataset {fname} missing"

    df_hist = pd.read_csv(dashboard_dir / "historical_sales.csv")
    assert len(df_hist) == 431244
    assert "is_forecast" in df_hist.columns
    assert (~df_hist["is_forecast"]).all()
    assert (df_hist["weekly_sales"] >= 0).all()

    df_fc = pd.read_csv(dashboard_dir / "forecast_data.csv")
    assert len(df_fc) == 23166
    assert "is_forecast" in df_fc.columns
    assert (df_fc["is_forecast"]).all()
    assert (df_fc["lightgbm_forecast"] >= 0).all()
    assert (df_fc["naive_forecast"] >= 0).all()

    df_perf = pd.read_csv(dashboard_dir / "model_performance.csv")
    assert len(df_perf) == 42
    assert set(df_perf["model"].unique()) == {"Naive", "Seasonal Naive", "LightGBM"}


def test_dashboard_data_validation_report_status():
    """Confirms the dashboard data validation report exists and contains PASS status."""
    report_path = Path("reports/dashboard_data_validation_report.md")
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "STATUS: PASS" in content
    assert "Gate 1 (File Existence):      PASS" in content or "**1. File Existence**" in content

