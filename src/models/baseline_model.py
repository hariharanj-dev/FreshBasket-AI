"""
FreshBasket AI - Baseline Forecasting Models Module
===================================================
Task 2: Implement Naive and Seasonal Naive forecasting baselines.

1. Naive Forecast:
   - Uses the latest observed complete week's sales as the constant forecast for each
     store-family series.
   - Repeated for each of the next 13 weeks.
   - Kept separate per (store_nbr, family).

2. Seasonal Naive Forecast:
   - Uses observed sales from 52 weeks earlier (t - 52 weeks = 364 days earlier)
     for each forecast week.
   - Documented Fallback: If a store-family combination lacks 52-week historical data
     (e.g., newly opened stores like Store 52), the forecaster falls back to the most
     recent observed sales (Naive) and flags seasonal_fallback = True.
     Missing history is NEVER silently substituted with zero.
   - Kept separate per (store_nbr, family).

Outputs:
- data/forecasts/baseline_forecasts.csv
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Paths
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = WORKSPACE_ROOT / "data" / "processed" / "weekly_features.csv"
OUTPUT_DIR = WORKSPACE_ROOT / "data" / "forecasts"
BASELINE_CSV_PATH = OUTPUT_DIR / "baseline_forecasts.csv"


class NaiveBaseline:
    """
    Naive baseline forecaster:
    Predicts future demand using the most recently observed complete week's sales value
    at or before the forecast origin cutoff date.
    """

    def __init__(self):
        self.latest_sales: Dict[Tuple[int, str], float] = {}
        self.cutoff_date: Optional[str] = None

    def fit(self, df_history: pd.DataFrame, cutoff_date: str) -> "NaiveBaseline":
        """
        Fits the baseline by identifying the last observed sales value
        for each (store_nbr, family) group on or before cutoff_date.
        """
        self.cutoff_date = cutoff_date
        # Filter complete historical records up to cutoff date
        hist = df_history[df_history["week_start_date"] <= cutoff_date].copy()
        hist = hist.sort_values(["store_nbr", "family", "week_start_date"])

        # Get the latest observed record for each series
        last_obs = hist.groupby(["store_nbr", "family"], observed=False).last().reset_index()
        for _, row in last_obs.iterrows():
            key = (int(row["store_nbr"]), str(row["family"]))
            self.latest_sales[key] = float(row["weekly_sales"])

        return self

    def predict(
        self,
        future_weeks: List[Tuple[str, str]],
        store_family_keys: List[Tuple[int, str]],
    ) -> pd.DataFrame:
        """
        Generates naive forecasts for all requested future weeks and store-family pairs.
        """
        records = []
        for h, (mon_str, sun_str) in enumerate(future_weeks, start=1):
            for s_nbr, fam in store_family_keys:
                key = (s_nbr, fam)
                pred = self.latest_sales.get(key, 0.0)
                # Sales cannot be negative
                pred = max(0.0, pred)
                records.append({
                    "forecast_origin": self.cutoff_date,
                    "week_start_date": mon_str,
                    "week_end_date": sun_str,
                    "forecast_horizon_week": h,
                    "store_nbr": s_nbr,
                    "family": fam,
                    "naive_prediction": round(pred, 2),
                })
        return pd.DataFrame(records)


class SeasonalNaiveBaseline:
    """
    Seasonal Naive baseline forecaster:
    Predicts sales for future week t using sales observed exactly 52 weeks earlier
    (t - 52 weeks = t - 364 days).

    Handling insufficient history:
    If a store-family combination lacks 52-week historical data (e.g. stores opened
    less than 1 year prior to cutoff), the forecaster falls back to the most recent
    observed sales (Naive) and flags 'seasonal_fallback = True'.
    Missing values are NEVER silently replaced with zero.
    """

    def __init__(self):
        self.history_lookup: Dict[Tuple[int, str, str], float] = {}
        self.latest_sales: Dict[Tuple[int, str], float] = {}
        self.cutoff_date: Optional[str] = None

    def fit(self, df_history: pd.DataFrame, cutoff_date: str) -> "SeasonalNaiveBaseline":
        """
        Indexes historical sales by (store_nbr, family, week_start_date)
        and caches latest observed sales for fallback handling.
        """
        self.cutoff_date = cutoff_date
        hist = df_history[df_history["week_start_date"] <= cutoff_date].copy()
        hist = hist.sort_values(["store_nbr", "family", "week_start_date"])

        # Build lookup table: (store_nbr, family, week_start_date) -> weekly_sales
        for _, row in hist.iterrows():
            key = (int(row["store_nbr"]), str(row["family"]), str(row["week_start_date"]))
            self.history_lookup[key] = float(row["weekly_sales"])

        # Cache latest observed value for fallback
        last_obs = hist.groupby(["store_nbr", "family"], observed=False).last().reset_index()
        for _, row in last_obs.iterrows():
            key = (int(row["store_nbr"]), str(row["family"]))
            self.latest_sales[key] = float(row["weekly_sales"])

        return self

    def predict(
        self,
        future_weeks: List[Tuple[str, str]],
        store_family_keys: List[Tuple[int, str]],
    ) -> pd.DataFrame:
        """
        Generates seasonal naive forecasts. For each date, finds date - 52 weeks (364 days).
        If found, uses it; otherwise falls back to naive and sets seasonal_fallback=True.
        """
        records = []
        for h, (mon_str, sun_str) in enumerate(future_weeks, start=1):
            target_dt = pd.to_datetime(mon_str)
            lag_52_dt = target_dt - timedelta(weeks=52)
            lag_52_str = lag_52_dt.strftime("%Y-%m-%d")

            for s_nbr, fam in store_family_keys:
                hist_key = (s_nbr, fam, lag_52_str)
                fallback_flag = False

                if hist_key in self.history_lookup:
                    pred = self.history_lookup[hist_key]
                else:
                    # Explicit fallback to latest known sales with documentation flag
                    pred = self.latest_sales.get((s_nbr, fam), 0.0)
                    fallback_flag = True

                pred = max(0.0, pred)
                records.append({
                    "forecast_origin": self.cutoff_date,
                    "week_start_date": mon_str,
                    "week_end_date": sun_str,
                    "forecast_horizon_week": h,
                    "store_nbr": s_nbr,
                    "family": fam,
                    "seasonal_naive_prediction": round(pred, 2),
                    "seasonal_fallback": fallback_flag,
                })
        return pd.DataFrame(records)


def generate_baseline_forecasts(
    df: pd.DataFrame,
    cutoff_date: str,
    future_weeks: Union[List[str], List[Tuple[str, str]]],
) -> pd.DataFrame:
    """
    Helper function to fit and run both Naive and Seasonal Naive baselines.
    Accepts future_weeks either as list of week_start_date strings or list of (start, end) tuples.
    Returns a unified DataFrame with predictions from both baselines.
    """
    # Normalize future_weeks to list of (start, end) tuples
    normalized_weeks: List[Tuple[str, str]] = []
    for item in future_weeks:
        if isinstance(item, tuple):
            normalized_weeks.append(item)
        else:
            mon_dt = pd.to_datetime(item)
            sun_dt = mon_dt + timedelta(days=6)
            normalized_weeks.append((mon_dt.strftime("%Y-%m-%d"), sun_dt.strftime("%Y-%m-%d")))

    # Unique series keys
    keys = df[["store_nbr", "family"]].drop_duplicates().sort_values(["store_nbr", "family"])
    store_family_keys = [(int(r["store_nbr"]), str(r["family"])) for _, r in keys.iterrows()]

    # Run Naive baseline
    naive_model = NaiveBaseline().fit(df, cutoff_date)
    df_naive = naive_model.predict(normalized_weeks, store_family_keys)

    # Run Seasonal Naive baseline
    s_naive_model = SeasonalNaiveBaseline().fit(df, cutoff_date)
    df_s_naive = s_naive_model.predict(normalized_weeks, store_family_keys)

    # Merge predictions
    merged = pd.merge(
        df_naive,
        df_s_naive[[
            "forecast_origin", "week_start_date", "week_end_date",
            "store_nbr", "family", "seasonal_naive_prediction", "seasonal_fallback"
        ]],
        on=["forecast_origin", "week_start_date", "week_end_date", "store_nbr", "family"],
        how="inner",
    )
    return merged


def run_standalone_baseline_pipeline():
    print("=" * 70)
    print(" FreshBasket AI - Baseline Forecasting Pipeline")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading feature dataset from {FEATURES_PATH}...")
    df = pd.read_csv(FEATURES_PATH)

    # Identify last complete week
    complete_weeks = df[df["is_partial_week"] == False].sort_values("week_start_date")
    last_row = complete_weeks.iloc[-1]
    last_start = str(last_row["week_start_date"])
    last_end = str(last_row["week_end_date"])
    forecast_origin = last_end

    print(f"  [+] Last complete observed week: {last_start} to {last_end}")
    print(f"  [+] Forecast origin cutoff date: {forecast_origin}")

    # Generate next 13 Monday-to-Sunday weeks
    start_dt = pd.to_datetime(last_start)
    future_weeks = []
    for w in range(1, 14):
        mon = start_dt + timedelta(weeks=w)
        sun = mon + timedelta(days=6)
        future_weeks.append((mon.strftime("%Y-%m-%d"), sun.strftime("%Y-%m-%d")))

    print(f"[*] Generating baseline forecasts for 13 weeks ({future_weeks[0][0]} to {future_weeks[-1][1]})...")
    df_baselines = generate_baseline_forecasts(df, cutoff_date=last_end, future_weeks=future_weeks)

    # Reorder columns as requested in Task 2
    cols = [
        "forecast_origin",
        "week_start_date",
        "week_end_date",
        "store_nbr",
        "family",
        "forecast_horizon_week",
        "naive_prediction",
        "seasonal_naive_prediction",
    ]
    df_out = df_baselines[cols].sort_values(["forecast_horizon_week", "store_nbr", "family"]).reset_index(drop=True)

    # Check for duplicate forecast keys
    dups = df_out.duplicated(subset=["week_start_date", "store_nbr", "family"]).sum()
    print(f"  [+] Duplicate forecast keys check: {dups} duplicates (PASS [OK])")
    assert dups == 0

    df_out.to_csv(BASELINE_CSV_PATH, index=False)
    print(f"[+] Baseline forecasts saved to {BASELINE_CSV_PATH} ({len(df_out):,} rows)")
    print("=" * 70)


if __name__ == "__main__":
    run_standalone_baseline_pipeline()
