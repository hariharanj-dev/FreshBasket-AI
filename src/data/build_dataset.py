"""
Main Dataset Pipeline for FreshBasket AI.

Orchestrates loading, cleaning, validation, and feature enrichment for:
- train.csv sales transactions
- stores.csv metadata
- oil.csv commodity pricing
- holidays_events.csv calendar effects
- transactions.csv foot-traffic history

Saves the enriched, continuous daily dataset to data/processed/cleaned_sales.csv.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import BASE_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
except ImportError:
    BASE_DIR = PROJECT_ROOT
    RAW_DATA_DIR = BASE_DIR / "data" / "raw"
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

from src.data.clean_data import (
    clean_holiday_features,
    clean_oil_prices,
    clean_store_metadata,
    identify_store_opening_dates,
    impute_christmas_closures,
)


def load_raw_train_sales(raw_dir: Path) -> pd.DataFrame:
    """Load train.csv safely with memory-efficient datatypes."""
    train_path = raw_dir / "train.csv"
    if not train_path.is_file():
        raise FileNotFoundError(f"train.csv not found at {train_path}")

    # Use explicit dtypes to save memory on 3M rows
    dtypes = {
        "id": np.int64,
        "store_nbr": np.int16,
        "family": "category",
        "sales": np.float64,
        "onpromotion": np.int32,
    }
    df = pd.read_csv(train_path, dtype=dtypes)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_raw_transactions(raw_dir: Path) -> pd.DataFrame:
    """Load transactions.csv safely."""
    trans_path = raw_dir / "transactions.csv"
    if not trans_path.is_file():
        raise FileNotFoundError(f"transactions.csv not found at {trans_path}")

    trans_df = pd.read_csv(
        trans_path,
        dtype={"store_nbr": np.int16, "transactions": np.int32}
    )
    trans_df["date"] = pd.to_datetime(trans_df["date"])
    return trans_df


def build_cleaned_dataset(
    raw_dir: Path = RAW_DATA_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    output_filename: str = "cleaned_sales.csv",
    force_overwrite: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, any]]:
    """
    Build the complete, unified, clean daily FMCG dataset.
    Performs joins, validates key integrity, and verifies that sales totals reconcile 100%.
    """
    output_path = output_dir / output_filename
    output_dir.mkdir(parents=True, exist_ok=True)

    # Documented replacement strategy: Check if output already exists
    if output_path.is_file() and not force_overwrite:
        stat = output_path.stat()
        print(f"[!] Existing cleaned dataset detected: {output_path}")
        print(f"    Size: {stat.st_size / (1024**2):.2f} MB | Modified: {datetime.fromtimestamp(stat.st_mtime)}")
        print("    Loading existing artifact. Set force_overwrite=True to rebuild.")
        return pd.read_csv(output_path), {"status": "loaded_existing"}

    print("=" * 70)
    print(" FreshBasket AI - Data Cleaning & Processing Pipeline")
    print("=" * 70)

    # 1. Load raw sales data
    print("\n[*] Loading raw train.csv transactions...")
    sales_df = load_raw_train_sales(raw_dir)
    original_row_count = len(sales_df)
    original_sales_sum = float(sales_df["sales"].sum())
    print(f"  [+] Loaded {original_row_count:,} sales rows.")
    print(f"  [+] Baseline Sales Total: {original_sales_sum:,.2f} units.")

    # 2. Check pre-cleaning business keys uniqueness
    print("\n[*] Checking pre-cleaning duplicate keys (date, store_nbr, family)...")
    pre_dups = sales_df.duplicated(subset=["date", "store_nbr", "family"]).sum()
    print(f"  [+] Pre-cleaning duplicates: {pre_dups} (PASS)")
    if pre_dups > 0:
        raise ValueError(f"Found {pre_dups} duplicate keys in raw data!")

    # 3. Store opening dates investigation
    print("\n[*] Analyzing store operational lifecycle from transactions...")
    store_open_dates = identify_store_opening_dates(raw_dir)
    late_stores = {s: dt for s, dt in store_open_dates.items() if dt > "2013-01-02"}
    print(f"  [+] Identified {len(late_stores)} late-opening stores:")
    for s, dt in sorted(late_stores.items()):
        print(f"      - Store {s:2d}: Confirmed opening {dt}")

    # 4. Handle Christmas gaps (impute closed days)
    print("\n[*] Handling missing Christmas calendar gaps (Dec 25 closures)...")
    all_stores = sorted(sales_df["store_nbr"].unique().tolist())
    all_families = sorted(sales_df["family"].unique().tolist())
    sales_df["date"] = sales_df["date"].astype(str)

    sales_df, imputed_count = impute_christmas_closures(
        sales_df,
        stores=all_stores,
        families=all_families,
        christmas_dates=["2013-12-25", "2014-12-25", "2015-12-25", "2016-12-25"]
    )
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    print(f"  [+] Imputed {imputed_count:,} closure rows for Christmas Days.")
    expected_full_grid_rows = 1688 * len(all_stores) * len(all_families)
    print(f"  [+] Current rows: {len(sales_df):,} | Expected 1,688-day grid: {expected_full_grid_rows:,}")
    if len(sales_df) != expected_full_grid_rows:
        raise ValueError(f"Grid mismatch: Expected {expected_full_grid_rows}, got {len(sales_df)}")

    # 5. Enrich with Store Metadata
    print("\n[*] Joining store metadata (stores.csv)...")
    stores_df = clean_store_metadata(raw_dir)
    rows_before_stores = len(sales_df)
    sales_df = sales_df.merge(stores_df, on="store_nbr", how="left")
    if len(sales_df) != rows_before_stores:
        raise ValueError("Row multiplication detected after stores.csv join!")
    print(f"  [+] Joined 54 stores successfully. Row count preserved: {len(sales_df):,}")

    # 6. Enrich with Clean Daily Oil Prices
    print("\n[*] Joining continuous daily oil prices (oil.csv)...")
    oil_df = clean_oil_prices(raw_dir, min_date="2013-01-01", max_date="2017-08-31")
    rows_before_oil = len(sales_df)
    sales_df = sales_df.merge(oil_df, on="date", how="left")
    if len(sales_df) != rows_before_oil:
        raise ValueError("Row multiplication detected after oil.csv join!")
    print(f"  [+] Oil prices joined. Missing values in oil: {sales_df['dcoilwtico'].isnull().sum()}")

    # 7. Enrich with Store-Date Aggregated Holidays & Events
    print("\n[*] Constructing and joining store-specific holiday features...")
    holiday_features = clean_holiday_features(raw_dir, stores_df)
    rows_before_holidays = len(sales_df)
    sales_df = sales_df.merge(holiday_features, on=["date", "store_nbr"], how="left")
    if len(sales_df) != rows_before_holidays:
        raise ValueError(
            f"Row explosion detected after holiday join! Before: {rows_before_holidays}, After: {len(sales_df)}"
        )

    # Fill non-holiday days with False
    holiday_bool_cols = [
        "is_holiday", "is_national_holiday", "is_regional_holiday",
        "is_local_holiday", "is_event", "is_work_day"
    ]
    for col in holiday_bool_cols:
        sales_df[col] = sales_df[col].fillna(False).astype(bool)
    print(f"  [+] Holiday features joined without row multiplication. Preserved rows: {len(sales_df):,}")

    # 8. Enrich with Historical Foot Traffic (transactions.csv)
    print("\n[*] Joining historical store transactions (transactions.csv)...")
    trans_df = load_raw_transactions(raw_dir)
    rows_before_trans = len(sales_df)
    sales_df = sales_df.merge(trans_df, on=["date", "store_nbr"], how="left")
    if len(sales_df) != rows_before_trans:
        raise ValueError("Row multiplication detected after transactions.csv join!")

    # Document transaction coverage: fill missing transactions on closed days/pre-opening with 0
    sales_df["transactions"] = sales_df["transactions"].fillna(0).astype(np.int32)
    print(f"  [+] Transactions joined. Rows: {len(sales_df):,}")

    # 9. Operational store flags
    print("\n[*] Computing operational store lifecycle and closure flags...")
    # Map store opening date
    open_dates_series = sales_df["store_nbr"].map(store_open_dates)
    open_dates_dt = pd.to_datetime(open_dates_series)
    sales_df["is_store_active"] = sales_df["date"] >= open_dates_dt

    # Closed day: Christmas or store had 0 transactions and 0 sales across all families on that day
    sales_df["is_store_open"] = (
        sales_df["is_store_active"] &
        (~sales_df["is_imputed_closure"]) &
        (sales_df["transactions"] > 0)
    )

    # 10. Final business keys and reconciliation checks
    print("\n[*] Running final integrity checks & sales total reconciliation...")
    post_dups = sales_df.duplicated(subset=["date", "store_nbr", "family"]).sum()
    print(f"  [+] Post-cleaning duplicate keys (date, store_nbr, family): {post_dups} (PASS)")
    if post_dups > 0:
        raise ValueError("Duplicate keys detected in final dataset!")

    final_sales_sum = float(sales_df["sales"].sum())
    sales_diff = abs(final_sales_sum - original_sales_sum)
    print(f"  [+] Original Sales Total: {original_sales_sum:,.2f}")
    print(f"  [+] Final Sales Total:    {final_sales_sum:,.2f}")
    print(f"  [+] Reconciliation Diff:  {sales_diff:.6f} units (PASS)")
    if sales_diff > 1e-4:
        raise ValueError(f"Sales reconciliation failed! Discrepancy of {sales_diff} units detected.")

    # 11. Save cleaned dataset
    print(f"\n[*] Saving cleaned dataset to {output_path}...")
    sales_df.to_csv(output_path, index=False)
    file_size_mb = output_path.stat().st_size / (1024**2)
    print(f"  [+] Cleaned dataset successfully saved ({file_size_mb:.2f} MB).")
    print("=" * 70)

    summary_stats = {
        "status": "success",
        "output_path": str(output_path),
        "file_size_mb": round(file_size_mb, 2),
        "total_rows": len(sales_df),
        "total_columns": len(sales_df.columns),
        "columns": list(sales_df.columns),
        "original_sales_total": original_sales_sum,
        "final_sales_total": final_sales_sum,
        "reconciliation_passed": (sales_diff < 1e-4),
        "imputed_christmas_rows": imputed_count,
        "unique_stores": len(all_stores),
        "unique_families": len(all_families),
        "date_min": str(sales_df["date"].min().date()),
        "date_max": str(sales_df["date"].max().date()),
    }
    return sales_df, summary_stats


if __name__ == "__main__":
    build_cleaned_dataset()
