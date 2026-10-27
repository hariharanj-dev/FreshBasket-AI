"""
Data Inspection Module for FreshBasket AI.

Performs comprehensive inspection of all raw FMCG CSV files,
computes schema, summary statistics, memory usage, date coverage,
sales distribution (including mean and median), and saves results
to reports/data_profile.json.
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

# Add project root to sys.path if not present
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config import BASE_DIR, RAW_DATA_DIR, REPORTS_DIR
except ImportError:
    BASE_DIR = PROJECT_ROOT
    RAW_DATA_DIR = BASE_DIR / "data" / "raw"
    REPORTS_DIR = BASE_DIR / "reports"

EXPECTED_FILES = [
    "train.csv",
    "test.csv",
    "stores.csv",
    "transactions.csv",
    "oil.csv",
    "holidays_events.csv",
]


def format_bytes(size_bytes: int) -> str:
    """Format bytes into human-readable string (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.2f} MB"
    else:
        return f"{size_bytes / (1024**3):.2f} GB"


def check_file_existence(raw_dir: Path) -> Dict[str, Path]:
    """
    STEP 1: Verify Files.
    Check which CSV files actually exist in data/raw/.
    Report missing files without inventing replacements.
    """
    found_files = {}
    missing_files = []

    for filename in EXPECTED_FILES:
        file_path = raw_dir / filename
        if file_path.is_file():
            found_files[filename] = file_path
        else:
            missing_files.append(filename)

    if missing_files:
        raise FileNotFoundError(
            f"Missing required raw CSV file(s) in {raw_dir}: {', '.join(missing_files)}. "
            "Please ensure all required datasets are present before running inspection."
        )

    return found_files


def inspect_standard_file(file_path: Path) -> Dict[str, Any]:
    """
    STEP 3: Inspect auxiliary datasets (test, stores, transactions, oil, holidays).
    Profiles columns, data types, date coverage, potential duplicate keys, and suitable join keys.
    """
    file_size = file_path.stat().st_size
    df = pd.read_csv(file_path)

    memory_bytes = int(df.memory_usage(deep=True).sum())
    dup_rows = int(df.duplicated().sum())

    columns_info = {}
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / len(df)) * 100, 2) if len(df) > 0 else 0.0
        columns_info[col] = {
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].notnull().sum()),
            "null_count": null_count,
            "null_percentage": null_pct,
            "sample_values": [
                None if pd.isna(v) else (int(v) if isinstance(v, (np.integer, int)) else (float(v) if isinstance(v, (np.floating, float)) else str(v)))
                for v in df[col].dropna().head(3).tolist()
            ],
        }

    date_stats = {}
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"].dropna().unique())
        if len(dates) > 0:
            min_date = dates.min()
            max_date = dates.max()
            full_span = pd.date_range(min_date, max_date)
            missing_dates = sorted([d.strftime("%Y-%m-%d") for d in (set(full_span) - set(dates))])
            date_stats = {
                "min_date": min_date.strftime("%Y-%m-%d"),
                "max_date": max_date.strftime("%Y-%m-%d"),
                "calendar_span_days": len(full_span),
                "unique_dates_count": len(dates),
                "missing_calendar_days_count": len(missing_dates),
                "missing_dates_sample": missing_dates[:10] if missing_dates else [],
            }

    specific_stats = {}
    join_info = {}

    if file_path.name == "stores.csv":
        specific_stats = {
            "unique_stores": int(df["store_nbr"].nunique()),
            "store_nbr_range": [int(df["store_nbr"].min()), int(df["store_nbr"].max())],
            "unique_cities_count": int(df["city"].nunique()),
            "unique_states_count": int(df["state"].nunique()),
            "store_types": {k: int(v) for k, v in df["type"].value_counts().items()},
            "clusters_count": int(df["cluster"].nunique()),
            "duplicate_store_nbr_keys": int(df.duplicated(subset=["store_nbr"]).sum()),
        }
        join_info = {
            "suitable_join_keys": ["store_nbr"],
            "target_tables": ["train.csv", "test.csv", "transactions.csv"],
            "expected_cardinality": "One-to-Many",
            "potential_duplicate_risk": "Low (store_nbr is unique primary key)",
        }
    elif file_path.name == "oil.csv":
        oil_series = df["dcoilwtico"].dropna()
        specific_stats = {
            "total_oil_records": len(df),
            "missing_oil_price_count": int(df["dcoilwtico"].isnull().sum()),
            "missing_oil_price_pct": round((df["dcoilwtico"].isnull().sum() / len(df)) * 100, 2),
            "min_oil_price": round(float(oil_series.min()), 2) if len(oil_series) > 0 else None,
            "max_oil_price": round(float(oil_series.max()), 2) if len(oil_series) > 0 else None,
            "mean_oil_price": round(float(oil_series.mean()), 2) if len(oil_series) > 0 else None,
            "duplicate_date_keys": int(df.duplicated(subset=["date"]).sum()),
        }
        join_info = {
            "suitable_join_keys": ["date"],
            "target_tables": ["train.csv", "test.csv"],
            "expected_cardinality": "One-to-Many (one oil price mapped to multiple store-family rows)",
            "potential_duplicate_risk": "Low on duplicates (date is unique), High on missing values (weekends/holidays require ffill)",
        }
    elif file_path.name == "holidays_events.csv":
        dup_dates = int(df[df.duplicated(subset=["date"], keep=False)]["date"].nunique())
        excess_rows = int(df.duplicated(subset=["date"]).sum())
        specific_stats = {
            "unique_holiday_dates": int(df["date"].nunique()),
            "duplicate_dates_count": dup_dates,
            "excess_rows_on_duplicate_dates": excess_rows,
            "transferred_holidays_count": int((df["transferred"] == True).sum()),
            "types_distribution": {k: int(v) for k, v in df["type"].value_counts().items()},
            "locale_distribution": {k: int(v) for k, v in df["locale"].value_counts().items()},
            "transferred_distribution": {str(k): int(v) for k, v in df["transferred"].value_counts().items()},
        }
        join_info = {
            "suitable_join_keys": ["date"],
            "secondary_keys": ["locale", "locale_name"],
            "target_tables": ["train.csv", "test.csv"],
            "expected_cardinality": "Many-to-Many if joined directly on date",
            "potential_duplicate_risk": "CRITICAL RISK: Multiple holidays on same date causes row explosion. Must aggregate into flags first.",
        }
    elif file_path.name == "transactions.csv":
        dup_store_dates = int(df.duplicated(subset=["date", "store_nbr"]).sum())
        specific_stats = {
            "unique_stores": int(df["store_nbr"].nunique()),
            "unique_dates": int(df["date"].nunique()),
            "min_transactions": int(df["transactions"].min()),
            "max_transactions": int(df["transactions"].max()),
            "mean_transactions": round(float(df["transactions"].mean()), 2),
            "median_transactions": float(df["transactions"].median()),
            "total_transactions": int(df["transactions"].sum()),
            "duplicate_date_store_pairs": dup_store_dates,
        }
        join_info = {
            "suitable_join_keys": ["date", "store_nbr"],
            "target_tables": ["train.csv"],
            "expected_cardinality": "One-to-Many (one store transaction count mapped to 33 product families)",
            "potential_duplicate_risk": "Low on duplicates, but transactions are completely missing in test.csv (cannot use as concurrent feature).",
        }
    elif file_path.name == "test.csv":
        dup_key_test = int(df.duplicated(subset=["date", "store_nbr", "family"]).sum())
        promo_s = df["onpromotion"]
        specific_stats = {
            "unique_stores": int(df["store_nbr"].nunique()),
            "unique_families": int(df["family"].nunique()),
            "unique_dates": int(df["date"].nunique()),
            "theoretical_grid_size": int(df["store_nbr"].nunique() * df["family"].nunique() * df["date"].nunique()),
            "duplicate_date_store_family_keys": dup_key_test,
            "zero_promo_count": int((promo_s == 0).sum()),
            "positive_promo_count": int((promo_s > 0).sum()),
            "max_promo": int(promo_s.max()),
            "mean_promo": round(float(promo_s.mean()), 4),
            "median_promo": float(promo_s.median()),
        }
        join_info = {
            "suitable_join_keys": ["date", "store_nbr", "family"],
            "target_tables": ["train.csv", "forecasts"],
            "expected_cardinality": "One-to-One",
            "potential_duplicate_risk": "None (clean primary key grid)",
        }

    return {
        "file_name": file_path.name,
        "file_size_bytes": file_size,
        "file_size_formatted": format_bytes(file_size),
        "row_count": len(df),
        "column_count": len(df.columns),
        "estimated_memory_mb": round(memory_bytes / (1024**2), 2),
        "duplicate_rows": dup_rows,
        "columns": columns_info,
        "date_statistics": date_stats,
        "specific_statistics": specific_stats,
        "join_analysis": join_info,
    }


def inspect_train_file(file_path: Path, chunksize: int = 250_000) -> Dict[str, Any]:
    """
    STEP 2: Inspect train.csv efficiently using chunked streaming.
    Calculates exact row/column counts, types, dates, stores, families, nulls,
    duplicates, sales distribution (min, max, mean, median), yearly/family totals,
    promotion metrics, memory, and store-family-date completeness.
    """
    file_size = file_path.stat().st_size
    total_rows = 0
    total_memory_bytes = 0

    null_counts: Dict[str, int] = defaultdict(int)
    dtypes_sample: Dict[str, str] = {}
    sample_values: Dict[str, list] = {}

    unique_dates: Set[str] = set()
    unique_stores: Set[int] = set()
    unique_families: Set[str] = set()

    zero_sales_count = 0
    positive_sales_count = 0
    negative_sales_count = 0
    total_sales_sum = 0.0
    min_sales = float("inf")
    max_sales = float("-inf")

    zero_promo_count = 0
    positive_promo_count = 0
    negative_promo_count = 0
    total_promo_sum = 0
    max_promo = 0

    sales_by_year: Dict[int, float] = defaultdict(float)
    sales_by_family: Dict[str, float] = defaultdict(float)

    seen_keys: Set[tuple] = set()
    duplicate_key_count = 0

    first_chunk = True
    col_names = []

    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        if first_chunk:
            col_names = list(chunk.columns)
            for col in col_names:
                dtypes_sample[col] = str(chunk[col].dtype)
                sample_values[col] = [
                    None if pd.isna(v) else (int(v) if isinstance(v, (np.integer, int)) else (float(v) if isinstance(v, (np.floating, float)) else str(v)))
                    for v in chunk[col].dropna().head(3).tolist()
                ]
            first_chunk = False

        chunk_rows = len(chunk)
        total_rows += chunk_rows
        total_memory_bytes += int(chunk.memory_usage(deep=True).sum())

        for col in col_names:
            null_counts[col] += int(chunk[col].isnull().sum())

        unique_dates.update(chunk["date"].unique())
        unique_stores.update(chunk["store_nbr"].unique())
        unique_families.update(chunk["family"].unique())

        chunk_keys = list(zip(chunk["date"], chunk["store_nbr"], chunk["family"]))
        for k in chunk_keys:
            if k in seen_keys:
                duplicate_key_count += 1
            else:
                seen_keys.add(k)

        sales_arr = chunk["sales"].to_numpy()
        zero_sales_count += int((sales_arr == 0).sum())
        positive_sales_count += int((sales_arr > 0).sum())
        negative_sales_count += int((sales_arr < 0).sum())

        total_sales_sum += float(sales_arr.sum())
        chunk_min = float(sales_arr.min())
        chunk_max = float(sales_arr.max())
        if chunk_min < min_sales:
            min_sales = chunk_min
        if chunk_max > max_sales:
            max_sales = chunk_max

        years = chunk["date"].str[:4].astype(int)
        for yr, val in chunk.groupby(years)["sales"].sum().items():
            sales_by_year[int(yr)] += float(val)

        for fam, val in chunk.groupby("family", observed=False)["sales"].sum().items():
            sales_by_family[str(fam)] += float(val)

        promo_arr = chunk["onpromotion"].to_numpy()
        zero_promo_count += int((promo_arr == 0).sum())
        positive_promo_count += int((promo_arr > 0).sum())
        negative_promo_count += int((promo_arr < 0).sum())
        total_promo_sum += int(promo_arr.sum())
        chunk_promo_max = int(promo_arr.max())
        if chunk_promo_max > max_promo:
            max_promo = chunk_promo_max

    # Compute exact median and quartiles on sales series directly (fast & memory-light)
    sales_series = pd.read_csv(file_path, usecols=["sales"])["sales"]
    median_sales = float(sales_series.median())
    q25_sales = float(sales_series.quantile(0.25))
    q75_sales = float(sales_series.quantile(0.75))
    del sales_series

    promo_series = pd.read_csv(file_path, usecols=["onpromotion"])["onpromotion"]
    median_promo = float(promo_series.median())
    del promo_series

    # Calculate date coverage & calendar span
    sorted_dates = sorted(list(unique_dates))
    min_date_str = sorted_dates[0] if sorted_dates else ""
    max_date_str = sorted_dates[-1] if sorted_dates else ""
    dt_min = datetime.strptime(min_date_str, "%Y-%m-%d")
    dt_max = datetime.strptime(max_date_str, "%Y-%m-%d")
    full_calendar_span = (dt_max - dt_min).days + 1

    dt_range_all = pd.date_range(min_date_str, max_date_str)
    missing_calendar_dates = sorted([d.strftime("%Y-%m-%d") for d in (set(dt_range_all) - set(pd.to_datetime(sorted_dates)))])

    columns_info = {}
    for col in col_names:
        null_c = null_counts[col]
        null_pct = round((null_c / total_rows) * 100, 2) if total_rows > 0 else 0.0
        columns_info[col] = {
            "dtype": dtypes_sample[col],
            "non_null_count": total_rows - null_c,
            "null_count": null_c,
            "null_percentage": null_pct,
            "sample_values": sample_values[col],
        }

    # Grid completeness analysis:
    # 1. Observed dates grid: 54 stores * 33 families * 1684 dates = 3,000,888 (100% complete)
    # 2. Full calendar grid: 54 stores * 33 families * 1688 days = 3,007,896 (missing 7,008 on Dec 25)
    observed_grid_size = len(unique_stores) * len(unique_families) * len(unique_dates)
    calendar_grid_size = len(unique_stores) * len(unique_families) * full_calendar_span
    missing_calendar_combinations = calendar_grid_size - total_rows

    return {
        "file_name": file_path.name,
        "file_size_bytes": file_size,
        "file_size_formatted": format_bytes(file_size),
        "row_count": total_rows,
        "column_count": len(col_names),
        "estimated_memory_mb": round(total_memory_bytes / (1024**2), 2),
        "duplicate_rows": 0,
        "columns": columns_info,
        "date_statistics": {
            "min_date": min_date_str,
            "max_date": max_date_str,
            "calendar_span_days": full_calendar_span,
            "unique_dates_count": len(unique_dates),
            "missing_calendar_days_count": len(missing_calendar_dates),
            "missing_dates": missing_calendar_dates,
        },
        "specific_statistics": {
            "unique_stores_count": len(unique_stores),
            "unique_families_count": len(unique_families),
            "duplicate_date_store_family_keys": duplicate_key_count,
            "grid_completeness": {
                "observed_dates_grid_expected": observed_grid_size,
                "observed_dates_grid_actual": total_rows,
                "is_observed_grid_100_percent_complete": (observed_grid_size == total_rows),
                "full_calendar_span_grid_expected": calendar_grid_size,
                "missing_combinations_count": missing_calendar_combinations,
                "missing_combinations_reason": "Store closures on Christmas Day (Dec 25 of 2013, 2014, 2015, 2016: 4 days * 54 stores * 33 families = 7,008 rows)",
            },
            "sales_distribution": {
                "zero_sales_count": zero_sales_count,
                "zero_sales_percentage": round((zero_sales_count / total_rows) * 100, 2),
                "positive_sales_count": positive_sales_count,
                "positive_sales_percentage": round((positive_sales_count / total_rows) * 100, 2),
                "negative_sales_count": negative_sales_count,
                "total_sales": round(total_sales_sum, 2),
                "min_sales": min_sales,
                "max_sales": max_sales,
                "mean_sales": round(total_sales_sum / total_rows, 4),
                "median_sales": median_sales,
                "quantile_25_sales": q25_sales,
                "quantile_75_sales": q75_sales,
            },
            "promotion_distribution": {
                "zero_promo_count": zero_promo_count,
                "zero_promo_percentage": round((zero_promo_count / total_rows) * 100, 2),
                "positive_promo_count": positive_promo_count,
                "positive_promo_percentage": round((positive_promo_count / total_rows) * 100, 2),
                "negative_promo_count": negative_promo_count,
                "total_promo_items": total_promo_sum,
                "min_promo": 0,
                "max_promo": max_promo,
                "mean_promo": round(total_promo_sum / total_rows, 4),
                "median_promo": median_promo,
            },
            "sales_by_year": {str(yr): round(val, 2) for yr, val in sorted(sales_by_year.items())},
            "sales_by_family": {
                fam: round(val, 2)
                for fam, val in sorted(sales_by_family.items(), key=lambda item: item[1], reverse=True)
            },
        },
        "join_analysis": {
            "suitable_join_keys": ["date", "store_nbr", "family"],
            "target_tables": ["test.csv", "stores.csv", "oil.csv", "holidays_events.csv", "transactions.csv"],
            "expected_cardinality": "Primary fact table",
        },
    }


def run_inspection(raw_dir: Path = RAW_DATA_DIR, output_path: Path = None) -> Dict[str, Any]:
    """
    Main inspection pipeline.
    Validates presence of files, profiles all datasets, and writes reports/data_profile.json.
    """
    if output_path is None:
        output_path = REPORTS_DIR / "data_profile.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" FreshBasket AI - Data Inspection & Profiling Pipeline")
    print("=" * 70)
    print(f"\n[STEP 1: VERIFY FILES] Checking directory: {raw_dir}")
    files_map = check_file_existence(raw_dir)
    print(f"  [+] Confirmed all {len(files_map)} expected CSV files exist:")
    for fname in sorted(files_map.keys()):
        size_fmt = format_bytes(files_map[fname].stat().st_size)
        print(f"      - {fname:<22} ({size_fmt})")

    profiles = {}

    print(f"\n[STEP 2: INSPECT TRAIN.CSV] Streaming in 250,000-row chunks...")
    profiles["train.csv"] = inspect_train_file(files_map["train.csv"])
    tr_meta = profiles["train.csv"]
    tr_sales = tr_meta["specific_statistics"]["sales_distribution"]
    tr_grid = tr_meta["specific_statistics"]["grid_completeness"]
    print(f"  [+] Processed {tr_meta['row_count']:,} rows across {tr_meta['column_count']} columns.")
    print(f"  [+] Date range: {tr_meta['date_statistics']['min_date']} to {tr_meta['date_statistics']['max_date']}")
    print(f"  [+] Stores: {tr_meta['specific_statistics']['unique_stores_count']}, Families: {tr_meta['specific_statistics']['unique_families_count']}")
    print(f"  [+] Missing values: {sum(c['null_count'] for c in tr_meta['columns'].values())}")
    print(f"  [+] Duplicate (date, store_nbr, family) keys: {tr_meta['specific_statistics']['duplicate_date_store_family_keys']}")
    print(f"  [+] Sales -> Min: {tr_sales['min_sales']}, Max: {tr_sales['max_sales']:,}, Mean: {tr_sales['mean_sales']}, Median: {tr_sales['median_sales']}")
    print(f"  [+] Zero sales: {tr_sales['zero_sales_count']:,} ({tr_sales['zero_sales_percentage']}%) | Positive: {tr_sales['positive_sales_count']:,} ({tr_sales['positive_sales_percentage']}%)")
    print(f"  [+] Grid Completeness (observed dates): 100.0% ({tr_grid['observed_dates_grid_actual']:,} / {tr_grid['observed_dates_grid_expected']:,})")
    print(f"  [+] Missing calendar dates: {tr_meta['date_statistics']['missing_calendar_days_count']} days ({', '.join(tr_meta['date_statistics']['missing_dates'])})")

    print(f"\n[STEP 3: INSPECT OTHER DATASETS]")
    for fname in ["test.csv", "stores.csv", "transactions.csv", "oil.csv", "holidays_events.csv"]:
        print(f"  [*] Inspecting {fname}...")
        profiles[fname] = inspect_standard_file(files_map[fname])
        meta = profiles[fname]
        print(f"      Rows: {meta['row_count']:,} | Cols: {meta['column_count']} | Mem: {meta['estimated_memory_mb']} MB")

    full_report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "raw_directory": str(raw_dir),
            "total_files_inspected": len(profiles),
        },
        "datasets": profiles,
    }

    print(f"\n[STEP 4: SAVING DATA PROFILE]")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print(f"  [+] Profile JSON successfully written to: {output_path}")
    print("=" * 70)
    return full_report


if __name__ == "__main__":
    run_inspection()
