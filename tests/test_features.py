"""
Unit and integration tests for FreshBasket AI weekly feature engineering.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BASE_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR


def test_weekly_features_file_exists():
    """Verify weekly_features.csv exists."""
    features_path = PROCESSED_DATA_DIR / "weekly_features.csv"
    assert features_path.is_file(), f"weekly_features.csv not found at {features_path}"


def test_weekly_features_dimensions():
    """Verify weekly features dataset has 431,244 rows."""
    features_path = PROCESSED_DATA_DIR / "weekly_features.csv"
    row_count = sum(1 for _ in open(features_path, "r", encoding="utf-8")) - 1
    assert row_count == 431244


def test_weekly_features_keys_uniqueness():
    """Verify zero duplicate business keys (week_start_date, store_nbr, family)."""
    features_path = PROCESSED_DATA_DIR / "weekly_features.csv"
    df = pd.read_csv(features_path, usecols=["week_start_date", "store_nbr", "family"])
    assert df.duplicated().sum() == 0


def test_leakage_safety_shift_verification():
    """Verify rolling statistics strictly exclude the target week sales."""
    features_path = PROCESSED_DATA_DIR / "weekly_features.csv"
    cols = ["week_start_date", "store_nbr", "family", "weekly_sales", "sales_lag_1", "sales_rolling_mean_4"]
    sample = pd.read_csv(features_path, usecols=cols)
    s1 = sample[(sample["store_nbr"] == 1) & (sample["family"] == "AUTOMOTIVE")].sort_values("week_start_date").reset_index(drop=True)

    # Week index 4 (5th observation)
    y_target = s1.loc[4, "weekly_sales"]
    lag_1 = s1.loc[4, "sales_lag_1"]
    roll_4 = s1.loc[4, "sales_rolling_mean_4"]

    # Historical sales from weeks 0, 1, 2, 3
    hist_sales = s1.loc[0:3, "weekly_sales"]
    expected_roll_4 = hist_sales.mean()

    assert lag_1 == s1.loc[3, "weekly_sales"]
    assert abs(roll_4 - expected_roll_4) < 1e-3
    assert roll_4 != y_target or expected_roll_4 == y_target


def test_calendar_features_presence():
    """Verify all calendar features exist and are populated."""
    features_path = PROCESSED_DATA_DIR / "weekly_features.csv"
    cols = ["year", "month", "quarter", "week_of_year", "week_of_month", "month_start_flag", "month_end_flag", "quincena_flag", "days_in_week_observed", "is_partial_week"]
    df = pd.read_csv(features_path, usecols=cols, nrows=100)
    for c in cols:
        assert c in df.columns
        assert df[c].isnull().sum() == 0


def test_operational_eligibility_flag():
    """Verify is_training_eligible flag logic."""
    features_path = PROCESSED_DATA_DIR / "weekly_features.csv"
    cols = ["store_status", "is_partial_week", "is_training_eligible"]
    df = pd.read_csv(features_path, usecols=cols)

    expected_eligible = (df["store_status"] == "operational") & (~df["is_partial_week"])
    assert (df["is_training_eligible"] == expected_eligible).all()


def test_target_sales_column_preserved():
    """Verify target weekly_sales is preserved and total sales equals 1.073B units."""
    features_path = PROCESSED_DATA_DIR / "weekly_features.csv"
    sales_sum = pd.read_csv(features_path, usecols=["weekly_sales"])["weekly_sales"].sum()
    diff = abs(sales_sum - 1073644952.2030685)
    assert diff < 1e-2, f"Target sales mismatch! Diff: {diff}"
