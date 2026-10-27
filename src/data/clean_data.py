"""
Data Cleaning and Transformation Functions for FreshBasket AI.

Contains modular functions to clean, normalize, and construct features from:
- Oil commodity prices (calendar expansion & ffill)
- Store metadata (categorical mapping)
- Holiday and calendar events (transferred resolution & store-date level aggregation)
- Store operational lifecycle (opening dates & closure flags)
- Missing calendar dates (Christmas closure imputation)
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def clean_oil_prices(raw_dir: Path, min_date: str = "2013-01-01", max_date: str = "2017-08-31") -> pd.DataFrame:
    """
    Construct continuous daily oil prices via calendar expansion and forward-fill.

    Business Rationale:
    Crude oil (WTI) is an economic driver for Ecuador's economy, influencing consumer
    spending. However, commodity markets are closed on weekends and public holidays.
    Forward-filling propagates the last traded price over inactive days.
    """
    oil_path = raw_dir / "oil.csv"
    if not oil_path.is_file():
        raise FileNotFoundError(f"oil.csv not found at {oil_path}")

    oil_df = pd.read_csv(oil_path)
    oil_df["date"] = pd.to_datetime(oil_df["date"])

    # Expand to complete uninterrupted calendar range
    full_calendar = pd.DataFrame({"date": pd.date_range(min_date, max_date)})
    oil_clean = full_calendar.merge(oil_df, on="date", how="left")

    # Forward fill weekend gaps, then backward fill Jan 1 (market closed on New Year)
    oil_clean["dcoilwtico"] = oil_clean["dcoilwtico"].ffill().bfill()
    oil_clean["dcoilwtico"] = oil_clean["dcoilwtico"].astype(np.float32)

    return oil_clean


def clean_store_metadata(raw_dir: Path) -> pd.DataFrame:
    """
    Load and validate store metadata.
    Ensures store_nbr is unique and categoricals are properly typed.
    """
    stores_path = raw_dir / "stores.csv"
    if not stores_path.is_file():
        raise FileNotFoundError(f"stores.csv not found at {stores_path}")

    stores_df = pd.read_csv(stores_path)

    # Validate uniqueness of primary key
    if stores_df["store_nbr"].duplicated().any():
        raise ValueError("Duplicate store_nbr detected in stores.csv")

    stores_df["store_nbr"] = stores_df["store_nbr"].astype(np.int16)
    stores_df["city"] = stores_df["city"].astype(str)
    stores_df["state"] = stores_df["state"].astype(str)
    stores_df["type"] = stores_df["type"].astype(str)
    stores_df["cluster"] = stores_df["cluster"].astype(np.int8)

    return stores_df


def identify_store_opening_dates(raw_dir: Path) -> Dict[int, str]:
    """
    Determine confirmed store opening dates.

    Business Rule:
    A store opening date is confirmed when BOTH transactions > 0 and positive sales
    appear. In this dataset, for all 8 late-opening stores (20, 21, 22, 29, 36, 42, 52, 53),
    their first recorded transaction and first sales date match exactly.
    """
    trans_path = raw_dir / "transactions.csv"
    trans_df = pd.read_csv(trans_path)
    first_trans = trans_df.groupby("store_nbr")["date"].min().to_dict()

    return {int(store): str(dt) for store, dt in first_trans.items()}


def clean_holiday_features(raw_dir: Path, stores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process holidays_events.csv and aggregate to (date, store_nbr) level.

    Critical Data-Quality Handling:
    1. Transferred Holidays:
       - If `type == 'Holiday'` and `transferred == True`, the official holiday was shifted
         to another date. On the original date, it was a regular work day (not a holiday off).
       - If `type == 'Transfer'`, this is the celebrated holiday compensation day.
    2. Work Day:
       - If `type == 'Work Day'`, a weekend day was made into a working day (recovery day).
    3. Multi-event date consolidation:
       - 31 dates have multiple holiday entries.
       - Instead of a naive merge (which creates 53,460 duplicate rows), we resolve geographic
         scope (National, Regional matching store state, Local matching store city) and
         aggregate into clean boolean flags per (date, store_nbr).
    """
    holidays_path = raw_dir / "holidays_events.csv"
    if not holidays_path.is_file():
        raise FileNotFoundError(f"holidays_events.csv not found at {holidays_path}")

    holidays_df = pd.read_csv(holidays_path)
    holidays_df["date"] = pd.to_datetime(holidays_df["date"])

    # Separate active holidays from non-holiday work days
    # Transferred=True means celebration moved elsewhere; don't count original as celebrated holiday
    active_holidays = holidays_df[
        (holidays_df["transferred"] == False) & (holidays_df["type"] != "Work Day")
    ].copy()

    events_df = holidays_df[holidays_df["type"] == "Event"].copy()
    work_days_df = holidays_df[holidays_df["type"] == "Work Day"].copy()

    # Pre-build lookup sets for fast matching
    # National holidays apply to all stores
    nat_dates = set(active_holidays[active_holidays["locale"] == "National"]["date"])
    event_nat_dates = set(events_df[events_df["locale"] == "National"]["date"])
    work_day_dates = set(work_days_df["date"])

    # Regional holidays apply to stores in that state
    reg_lookup: Dict[str, Set[pd.Timestamp]] = {}
    for _, row in active_holidays[active_holidays["locale"] == "Regional"].iterrows():
        reg_lookup.setdefault(row["locale_name"], set()).add(row["date"])

    # Local holidays apply to stores in that city
    loc_lookup: Dict[str, Set[pd.Timestamp]] = {}
    for _, row in active_holidays[active_holidays["locale"] == "Local"].iterrows():
        loc_lookup.setdefault(row["locale_name"], set()).add(row["date"])

    # Create cartesian product of unique dates in holidays range with all stores
    all_dates = pd.date_range("2013-01-01", "2017-08-31")
    store_info = stores_df[["store_nbr", "city", "state"]].drop_duplicates()

    records = []
    for dt in all_dates:
        is_nat = dt in nat_dates
        is_ev_nat = dt in event_nat_dates
        is_wd = dt in work_day_dates

        # If date has any potential holiday or event activity
        for _, srow in store_info.iterrows():
            st_nbr = srow["store_nbr"]
            city = srow["city"]
            state = srow["state"]

            is_reg = dt in reg_lookup.get(state, set())
            is_loc = dt in loc_lookup.get(city, set())
            is_any_hol = is_nat or is_reg or is_loc

            # Only append if there is at least one non-zero holiday/event attribute
            # to keep the feature matrix clean
            if is_any_hol or is_ev_nat or is_wd:
                records.append({
                    "date": dt,
                    "store_nbr": st_nbr,
                    "is_holiday": True,
                    "is_national_holiday": is_nat,
                    "is_regional_holiday": is_reg,
                    "is_local_holiday": is_loc,
                    "is_event": is_ev_nat,
                    "is_work_day": is_wd,
                })

    holiday_features = pd.DataFrame(records)
    if len(holiday_features) > 0:
        # Groupby in case multiple local/regional events hit the same store-date
        holiday_features = holiday_features.groupby(["date", "store_nbr"], as_index=False).max()
    else:
        holiday_features = pd.DataFrame(columns=[
            "date", "store_nbr", "is_holiday", "is_national_holiday",
            "is_regional_holiday", "is_local_holiday", "is_event", "is_work_day"
        ])

    return holiday_features


def impute_christmas_closures(
    train_df: pd.DataFrame,
    stores: List[int],
    families: List[str],
    christmas_dates: List[str] = ("2013-12-25", "2014-12-25", "2015-12-25", "2016-12-25"),
) -> Tuple[pd.DataFrame, int]:
    """
    Impute nationwide Christmas closures into the sales DataFrame.

    Business Rationale:
    On December 25th of each year, all stores are closed nationwide.
    The raw data omitted these 4 dates entirely.
    To maintain daily continuous time series for lag generation (e.g. 7-day, 14-day lags),
    we generate rows for all store-family combinations on Dec 25 with:
      - sales = 0.0
      - onpromotion = 0
      - is_closed_day = True
      - is_imputed_closure = True

    Returns:
      (reindexed_df, imputed_row_count)
    """
    imputed_rows = []
    # Use negative IDs for imputed rows to guarantee no collision with raw IDs (0 to 3,000,887)
    # and test IDs (3,000,888 to 3,029,399)
    next_synthetic_id = -1

    for c_date in christmas_dates:
        for s_nbr in stores:
            for fam in families:
                imputed_rows.append({
                    "id": next_synthetic_id,
                    "date": c_date,
                    "store_nbr": s_nbr,
                    "family": fam,
                    "sales": 0.0,
                    "onpromotion": 0,
                    "is_imputed_closure": True,
                })
                next_synthetic_id -= 1

    imputed_df = pd.DataFrame(imputed_rows)
    train_df["is_imputed_closure"] = False

    combined_df = pd.concat([train_df, imputed_df], ignore_index=True)
    return combined_df, len(imputed_rows)
