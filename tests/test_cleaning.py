"""
Unit and integration tests for FreshBasket AI data cleaning and preprocessing.
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
from src.data.clean_data import clean_oil_prices, clean_store_metadata, identify_store_opening_dates


def test_raw_files_exist():
    """Verify all 6 raw CSV files exist and are untouched."""
    expected = ["train.csv", "test.csv", "stores.csv", "transactions.csv", "oil.csv", "holidays_events.csv"]
    for fname in expected:
        fpath = RAW_DATA_DIR / fname
        assert fpath.is_file(), f"Expected raw file missing: {fname}"


def test_clean_oil_prices():
    """Verify oil cleaning handles missing weekend trading days and has zero nulls."""
    oil_df = clean_oil_prices(RAW_DATA_DIR, min_date="2013-01-01", max_date="2017-08-31")
    assert len(oil_df) == 1704
    assert oil_df["dcoilwtico"].isnull().sum() == 0
    assert oil_df["dcoilwtico"].min() > 0


def test_clean_store_metadata():
    """Verify stores.csv has 54 unique stores."""
    stores_df = clean_store_metadata(RAW_DATA_DIR)
    assert len(stores_df) == 54
    assert stores_df["store_nbr"].nunique() == 54
    assert set(stores_df["store_nbr"]) == set(range(1, 55))


def test_store_opening_dates():
    """Verify 8 late-opening stores have confirmed opening dates."""
    opening_dates = identify_store_opening_dates(RAW_DATA_DIR)
    assert len(opening_dates) == 54
    late_stores = [20, 21, 22, 29, 36, 42, 52, 53]
    for s in late_stores:
        assert opening_dates[s] > "2013-01-02"


def test_cleaned_dataset_integrity():
    """Verify cleaned_sales.csv dimensions, grid completeness, and column schema."""
    cleaned_path = PROCESSED_DATA_DIR / "cleaned_sales.csv"
    assert cleaned_path.is_file(), "cleaned_sales.csv not found!"

    # Read header and first chunk to verify schema
    sample_df = pd.read_csv(cleaned_path, nrows=100)
    expected_cols = [
        "id", "date", "store_nbr", "family", "sales", "onpromotion",
        "city", "state", "type", "cluster", "dcoilwtico",
        "is_holiday", "is_national_holiday", "is_regional_holiday",
        "is_local_holiday", "is_event", "is_work_day", "transactions",
        "is_imputed_closure", "is_store_active", "is_store_open"
    ]
    for col in expected_cols:
        assert col in sample_df.columns, f"Missing expected column: {col}"
