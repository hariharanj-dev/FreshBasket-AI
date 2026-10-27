"""
FreshBasket AI - 13-Week Future Demand Forecasting Pipeline
===========================================================
Task 7: Generate forward-looking weekly demand forecasts for every store and product family
over the next 13 Monday-to-Sunday calendar weeks.

Operating Rules:
1. Origin: Identifies the last complete observed 7-day week (2017-08-07 to 2017-08-13).
   Cutoff date / forecast origin: 2017-08-13.
2. Horizon: Generates exactly 13 future Monday-to-Sunday weeks (2017-08-14 through 2017-11-12).
3. Deterministic Future Calendar Features: Computes month, quarter, week_of_year,
   quincena_flag (15th payday and month-end cycles).
4. Public Holiday Lookups: Checks known statutory Ecuadorian holidays (e.g., Primer Grito de
   Independencia in August, Independencia de Guayaquil in October, Dia de Difuntos in November).
5. Recursive Autoregressive Lags & Rolling Statistics: Each future week h uses dynamically
   recalculated lag and rolling metrics based on past actuals plus previously predicted weeks.
   Future actuals are NEVER accessed.
6. Information Horizon Barriers: Macroeconomic oil prices and customer transactions are held
   at the latest known levels at cutoff, avoiding lookahead assumptions.
7. Post-Processing: Ensures non-negative demand (sales cannot be negative).
8. Output: Saves data/forecasts/forecast_13_weeks.csv (23,166 rows: 1,782 series * 13 weeks).
"""

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from src.models.train_lightgbm import (
    CATEGORICAL_FEATURES,
    FEATURE_ALLOWLIST,
    NUMERICAL_FEATURES,
    load_trained_model,
)

# Paths
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = WORKSPACE_ROOT / "data" / "processed" / "weekly_features.csv"
RAW_HOLIDAYS_PATH = WORKSPACE_ROOT / "data" / "raw" / "holidays_events.csv"
OUTPUT_DIR = WORKSPACE_ROOT / "data" / "forecasts"
FORECAST_CSV_PATH = OUTPUT_DIR / "forecast_13_weeks.csv"
VAL_PREDS_PATH = OUTPUT_DIR / "validation_predictions.csv"


def add_calibrated_prediction_intervals(
    df_forecast: pd.DataFrame,
    val_preds_path: Path = VAL_PREDS_PATH,
) -> pd.DataFrame:
    """
    Computes calibrated empirical prediction intervals (80% and 95% confidence bounds)
    using the historical out-of-time validation residuals across product families and horizons.
    Non-negativity constraint is strictly enforced on lower bounds.
    """
    sigma_fh: Dict[Tuple[str, int], float] = {}
    fam_fallback: Dict[str, float] = {}
    global_sigma: float = 100.0

    if val_preds_path.exists():
        try:
            vp = pd.read_csv(val_preds_path)
            if "actual_sales" in vp.columns and "lightgbm_prediction" in vp.columns:
                residuals = vp["actual_sales"].astype(float) - vp["lightgbm_prediction"].astype(float)
                vp["_res"] = residuals
                global_sigma = float(np.sqrt(np.mean(residuals**2)))
                fam_grouped = vp.groupby("family", observed=False)["_res"].apply(
                    lambda x: float(np.sqrt(np.mean(x**2)))
                )
                fam_fallback = fam_grouped.to_dict()
                fh_grouped = vp.groupby(["family", "forecast_horizon_week"], observed=False)["_res"].apply(
                    lambda x: float(np.sqrt(np.mean(x**2)))
                )
                sigma_fh = fh_grouped.to_dict()
        except Exception as e:
            print(f"  [!] Note: Fallback prediction intervals used ({e})")

    lower_80, upper_80 = [], []
    lower_95, upper_95 = [], []
    unc_80, unc_95 = [], []

    for _, row in df_forecast.iterrows():
        y_hat = float(row["predicted_sales"])
        fam = str(row["family"])
        h = int(row["forecast_horizon_week"])
        sig = sigma_fh.get((fam, h), fam_fallback.get(fam, global_sigma))

        # 80% CI (z = 1.282), 95% CI (z = 1.960)
        l80 = max(0.0, round(y_hat - 1.282 * sig, 2))
        u80 = max(y_hat, round(y_hat + 1.282 * sig, 2))
        l95 = max(0.0, round(y_hat - 1.960 * sig, 2))
        u95 = max(y_hat, round(y_hat + 1.960 * sig, 2))

        lower_80.append(l80)
        upper_80.append(u80)
        lower_95.append(l95)
        upper_95.append(u95)
        unc_80.append(round(u80 - l80, 2))
        unc_95.append(round(u95 - l95, 2))

    df_forecast["lower_bound_80"] = lower_80
    df_forecast["upper_bound_80"] = upper_80
    df_forecast["lower_bound_95"] = lower_95
    df_forecast["upper_bound_95"] = upper_95
    df_forecast["uncertainty_range_80"] = unc_80
    df_forecast["uncertainty_range_95"] = unc_95

    # Standardized evaluation aliases (date, market_id, sku_id, forecast, uncertainty)
    df_forecast["date"] = df_forecast["week_start_date"]
    df_forecast["market_id"] = df_forecast["store_nbr"]
    df_forecast["sku_id"] = df_forecast["family"]
    df_forecast["forecast"] = df_forecast["predicted_sales"]
    df_forecast["uncertainty"] = df_forecast["uncertainty_range_95"]
    return df_forecast



def get_last_complete_week(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Finds the last complete 7-day observed week in the dataset.
    """
    complete_weeks = df[df["is_partial_week"] == False].sort_values("week_start_date")
    last_row = complete_weeks.iloc[-1]
    return str(last_row["week_start_date"]), str(last_row["week_end_date"])


def generate_future_calendar_weeks(
    last_week_start: str,
    n_weeks: int = 13,
) -> List[Tuple[str, str]]:
    """
    Generates the next n_weeks of Monday-to-Sunday calendar date intervals.
    """
    start_dt = pd.to_datetime(last_week_start)
    weeks = []
    for w in range(1, n_weeks + 1):
        mon = start_dt + timedelta(weeks=w)
        sun = mon + timedelta(days=6)
        weeks.append((mon.strftime("%Y-%m-%d"), sun.strftime("%Y-%m-%d")))
    return weeks


def load_future_holidays() -> Dict[str, int]:
    """
    Loads statutory holidays from raw holidays_events.csv for future dates, if available.
    """
    holiday_dates = set()
    if RAW_HOLIDAYS_PATH.exists():
        try:
            h_df = pd.read_csv(RAW_HOLIDAYS_PATH)
            h_df = h_df[h_df["transferred"] == False]
            for d in h_df["date"].unique():
                holiday_dates.add(str(d))
        except Exception:
            pass
    return holiday_dates


def generate_13_week_forecast() -> pd.DataFrame:
    print("=" * 70)
    print(" FreshBasket AI - 13-Week Future Demand Forecasting Pipeline")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading feature dataset from {FEATURES_PATH}...")
    df = pd.read_csv(FEATURES_PATH)

    # 1. Identify last complete training week
    last_start, last_end = get_last_complete_week(df)
    forecast_origin = last_end  # Forecast origin is the Sunday close of the last complete week
    print(f"  [+] Last complete observed week: {last_start} to {last_end}")
    print(f"  [+] Forecast origin cutoff date: {forecast_origin}")

    # 2. Generate 13 future forecast weeks
    future_weeks = generate_future_calendar_weeks(last_start, n_weeks=13)
    print(f"[*] Generating {len(future_weeks)} forecast weeks from {future_weeks[0][0]} to {future_weeks[-1][1]}...")

    # 3. Load trained model & metadata
    print("[*] Loading trained forecasting model...")
    wrapper, metadata = load_trained_model()
    print(f"  [+] Loaded model engine: {wrapper.engine}")

    # 4. Load store hierarchy metadata and statutory holidays
    store_meta = df[["store_nbr", "family", "city", "state", "type", "cluster"]].drop_duplicates().set_index(["store_nbr", "family"]).to_dict("index")
    statutory_holidays = load_future_holidays()

    # 5. Build historical sales buffer up to cutoff week
    hist = df[df["week_start_date"] <= last_start].sort_values(["store_nbr", "family", "week_start_date"])

    sales_buffers: Dict[Tuple[int, str], List[float]] = {}
    for (s_nbr, fam), group in hist.groupby(["store_nbr", "family"], observed=False):
        sales_buffers[(int(s_nbr), str(fam))] = group["weekly_sales"].astype(float).tolist()

    # Latest exogenous indicators at cutoff
    last_rows = hist.groupby(["store_nbr", "family"], observed=False).last().reset_index()
    last_promo = {(int(r["store_nbr"]), str(r["family"])): float(r.get("onpromotion_rolling_mean_4", 0.0)) for _, r in last_rows.iterrows()}
    last_trans = {(int(r["store_nbr"]), str(r["family"])): float(r.get("transactions_lag_1", 0.0)) for _, r in last_rows.iterrows()}
    last_oil = float(hist["oil_price_avg"].dropna().iloc[-1]) if "oil_price_avg" in hist.columns else 45.0

    forecast_records = []

    # 6. Step-by-step recursive 13-week rollout
    for h, (mon_str, sun_str) in enumerate(future_weeks, start=1):
        mon_dt = pd.to_datetime(mon_str)
        sun_dt = pd.to_datetime(sun_str)

        # Calendar features
        month = mon_dt.month
        quarter = (month - 1) // 3 + 1
        week_of_year = int(mon_dt.strftime("%W")) + 1
        week_of_month = (mon_dt.day - 1) // 7 + 1
        m_start_flag = int(mon_dt.day <= 7 and mon_dt.month != (mon_dt - timedelta(days=7)).month)
        m_end_flag = int(sun_dt.month != (sun_dt + timedelta(days=1)).month)
        quincena_flag = int((mon_dt.day <= 15 and sun_dt.day >= 15) or m_end_flag)

        # Check holidays in week
        week_days = [(mon_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]
        holidays_count = sum(1 for d in week_days if d in statutory_holidays)
        is_holiday_week = int(holidays_count > 0)

        step_features = []
        series_keys = []

        for (s_nbr, fam), buffer in sales_buffers.items():
            meta = store_meta.get((s_nbr, fam), {})

            n_buf = len(buffer)
            l1 = buffer[-1] if n_buf >= 1 else 0.0
            l2 = buffer[-2] if n_buf >= 2 else l1
            l4 = buffer[-4] if n_buf >= 4 else l2
            l8 = buffer[-8] if n_buf >= 8 else l4
            l13 = buffer[-13] if n_buf >= 13 else l8
            l26 = buffer[-26] if n_buf >= 26 else l13
            l52 = buffer[-52] if n_buf >= 52 else l26

            w4 = buffer[-4:] if n_buf >= 4 else buffer
            w8 = buffer[-8:] if n_buf >= 8 else buffer
            w13 = buffer[-13:] if n_buf >= 13 else buffer
            w26 = buffer[-26:] if n_buf >= 26 else buffer

            rm4 = float(np.mean(w4))
            rm8 = float(np.mean(w8))
            rm13 = float(np.mean(w13))
            rm26 = float(np.mean(w26))

            rs4 = float(np.std(w4, ddof=1)) if len(w4) > 1 else 0.0
            rs13 = float(np.std(w13, ddof=1)) if len(w13) > 1 else 0.0
            rs26 = float(np.std(w26, ddof=1)) if len(w26) > 1 else 0.0

            alpha = 2.0 / 14.0
            ewm13 = float(pd.Series(w13).ewm(alpha=alpha, adjust=False).mean().iloc[-1])

            record = {
                "store_nbr": s_nbr,
                "family": fam,
                "city": meta.get("city", "Quito"),
                "state": meta.get("state", "Pichincha"),
                "type": meta.get("type", "A"),
                "cluster": meta.get("cluster", 1),
                "month": month,
                "quarter": quarter,
                "week_of_year": week_of_year,
                "week_of_month": week_of_month,
                "month_start_flag": m_start_flag,
                "month_end_flag": m_end_flag,
                "quincena_flag": quincena_flag,
                "sales_lag_1": l1,
                "sales_lag_2": l2,
                "sales_lag_4": l4,
                "sales_lag_8": l8,
                "sales_lag_13": l13,
                "sales_lag_26": l26,
                "sales_lag_52": l52,
                "sales_rolling_mean_4": rm4,
                "sales_rolling_mean_8": rm8,
                "sales_rolling_mean_13": rm13,
                "sales_rolling_mean_26": rm26,
                "sales_rolling_std_4": rs4,
                "sales_rolling_std_13": rs13,
                "sales_rolling_std_26": rs26,
                "sales_ewm_mean_13": ewm13,
                "onpromotion_lag_1": last_promo.get((s_nbr, fam), 0.0),
                "onpromotion_lag_2": last_promo.get((s_nbr, fam), 0.0),
                "onpromotion_rolling_mean_4": last_promo.get((s_nbr, fam), 0.0),
                "transactions_lag_1": last_trans.get((s_nbr, fam), 0.0),
                "oil_price_lag_1": last_oil,
                "oil_price_weekly_change": 0.0,
                "is_holiday_week": is_holiday_week,
                "holidays_in_week": holidays_count,
            }
            step_features.append(record)
            series_keys.append((s_nbr, fam))

        df_step = pd.DataFrame(step_features)
        preds = wrapper.predict(df_step)

        # Store predictions and update recursive buffer
        for idx, pred_val in enumerate(preds):
            s_nbr, fam = series_keys[idx]
            # Post-processing: non-negative sales
            clean_pred = max(0.0, round(float(pred_val), 2))
            sales_buffers[(s_nbr, fam)].append(clean_pred)

            forecast_records.append({
                "forecast_origin": forecast_origin,
                "week_start_date": mon_str,
                "week_end_date": sun_str,
                "forecast_horizon_week": h,
                "store_nbr": s_nbr,
                "family": fam,
                "predicted_sales": clean_pred,
            })

    df_forecast = pd.DataFrame(forecast_records)
    print("[*] Calculating calibrated prediction intervals (80% & 95% confidence bounds)...")
    df_forecast = add_calibrated_prediction_intervals(df_forecast)
    df_forecast.to_csv(FORECAST_CSV_PATH, index=False)
    print(f"[+] 13-Week Forecast saved to {FORECAST_CSV_PATH} ({len(df_forecast):,} rows)")
    print(f"  [+] Predicted sales range: [{df_forecast['predicted_sales'].min():.2f}, {df_forecast['predicted_sales'].max():.2f}]")
    print(f"  [+] Total 13-week forecasted volume: {df_forecast['predicted_sales'].sum():,.2f} units")
    print(f"  [+] Uncertainty fields generated: lower_bound_80, upper_bound_80, lower_bound_95, upper_bound_95")
    print("=" * 70)
    return df_forecast


if __name__ == "__main__":
    generate_13_week_forecast()
