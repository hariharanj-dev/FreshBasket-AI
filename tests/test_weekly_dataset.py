"""
Unit and integration tests for FreshBasket AI weekly dataset aggregation.
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


def test_weekly_dataset_file_exists():
    """Verify weekly_sales.csv exists."""
    weekly_path = PROCESSED_DATA_DIR / "weekly_sales.csv"
    assert weekly_path.is_file(), f"weekly_sales.csv not found at {weekly_path}"


def test_weekly_dataset_dimensions():
    """Verify weekly dataset has exact 431,244 rows (242 weeks x 54 stores x 33 families)."""
    weekly_path = PROCESSED_DATA_DIR / "weekly_sales.csv"
    df = pd.read_csv(weekly_path, usecols=["week_start_date", "store_nbr", "family"])
    assert len(df) == 431244
    assert df["week_start_date"].nunique() == 242
    assert df["store_nbr"].nunique() == 54
    assert df["family"].nunique() == 33


def test_weekly_business_keys_uniqueness():
    """Verify zero duplicate (week_start_date, store_nbr, family) combinations."""
    weekly_path = PROCESSED_DATA_DIR / "weekly_sales.csv"
    df = pd.read_csv(weekly_path, usecols=["week_start_date", "store_nbr", "family"])
    assert df.duplicated().sum() == 0


def test_weekly_sales_reconciliation():
    """Verify total weekly sales reconciles with daily sales within tight float precision."""
    weekly_path = PROCESSED_DATA_DIR / "weekly_sales.csv"
    daily_path = PROCESSED_DATA_DIR / "cleaned_sales.csv"

    weekly_sales_sum = pd.read_csv(weekly_path, usecols=["weekly_sales"])["weekly_sales"].sum()
    daily_sales_sum = pd.read_csv(daily_path, usecols=["sales"])["sales"].sum()

    diff = abs(weekly_sales_sum - daily_sales_sum)
    assert diff < 1e-2, f"Sales mismatch! Diff: {diff}"


def test_boundary_weeks_and_partial_flags():
    """Verify partial week flags: Week 1 has 6 days, Week 242 has 2 days, others have 7."""
    weekly_path = PROCESSED_DATA_DIR / "weekly_sales.csv"
    df = pd.read_csv(weekly_path, usecols=["week_start_date", "days_observed", "is_partial_week"])

    week_days = df.groupby("week_start_date")["days_observed"].first()
    assert week_days.iloc[0] == 6  # Week 1: 2012-12-31 (Jan 1 to Jan 6)
    assert week_days.iloc[-1] == 2  # Week 242: 2017-08-14 (Aug 14 to Aug 15)
    assert (week_days.iloc[1:-1] == 7).all()  # 240 full interior weeks


def test_store_status_categories():
    """Verify operational status values and pre-opening classification."""
    weekly_path = PROCESSED_DATA_DIR / "weekly_sales.csv"
    df = pd.read_csv(weekly_path, usecols=["store_status", "store_nbr", "week_start_date"])

    valid_statuses = {"operational", "pre_opening", "temporary_closure"}
    assert set(df["store_status"].unique()).issubset(valid_statuses)

    # Store 52 opened in April 2017, must be pre_opening in 2013-2016
    st52_2013 = df[(df["store_nbr"] == 52) & (df["week_start_date"] < "2017-04-01")]
    assert (st52_2013["store_status"] == "pre_opening").all()
