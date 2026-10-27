# FreshBasket AI - Weekly Dataset Validation & Aggregation Report 📊📅

**Generated On:** 2026-10-24 18:52:26  
**Source Dataset:** Cleaned Daily Sales (`data/processed/cleaned_sales.csv`)  
**Output Dataset:** [`data/processed/weekly_sales.csv`](file:///c:/Users/user/Desktop/freshbasket-ai/data/processed/weekly_sales.csv)  
**Forecasting Scope:** 13-Week FMCG Demand Forecasting Horizon  

---

## 1. Executive Summary

The daily sales dataset was aggregated into an analysis-ready weekly dataset using **Monday-to-Sunday calendar weeks** (`W-SUN`). The weekly series encompasses **431,244 observations** spanning **242 continuous calendar weeks** across all **54 stores** and **33 product families** ($242 \times 54 \times 33 = 431,244$).

### Integrity Validation Highlights
- **100.0% Sales Volume Reconciliation:**  
  $$\text{Total Daily Sales} = 1,073,644,952.20 \quad \Longleftrightarrow \quad \text{Total Weekly Sales} = 1,073,644,952.20$$  
  *Exact match to 6 decimal places ($0.000000$ unit variance).*
- **Zero Duplicate Keys:** `0` duplicate combinations of `(week_start_date, store_nbr, family)`.
- **Zero Fabricated Observations:** All data points derive strictly from observed and confirmed daily transactions.
- **Operational State Categorization:** Every weekly record explicitly documents whether the store was *operational*, *not yet opened (pre-opening)*, or *temporarily closed*.

---

## 2. Comparison: Daily vs. Weekly Dataset

| Metric | Daily Dataset (`cleaned_sales.csv`) | Weekly Dataset (`weekly_sales.csv`) | Compression / Transformation |
| :--- | :--- | :--- | :--- |
| **Observation Unit** | `(date, store_nbr, family)` | `(week_start_date, store_nbr, family)` | Monday-to-Sunday aggregation |
| **Row Count** | 3,008,016 rows | **431,244 rows** | **85.66% dimensionality reduction** |
| **Time Steps** | 1,688 daily intervals | **242 calendar weeks** | Forecasting granularity |
| **Date Horizon** | `2013-01-01` to `2017-08-15` | `2012-12-31` to `2017-08-14` (week starts) | Monday reference anchors |
| **Cumulative Sales** | 1,073,644,952.20 units | **1,073,644,952.20 units** | Reconciled ($0.00$ variance) ✅ |
| **Unique Time Series** | 1,782 series/day | **1,782 series/week** | Preserved ($54 \text{ stores} \times 33 \text{ families}$) |

---

## 3. Boundary Weeks & Incomplete Week Analysis

The daily training dataset begins on Tuesday `2013-01-01` and ends on Tuesday `2017-08-15`. Under standard Monday-to-Sunday calendar weeks, the boundary weeks contain fewer than 7 days. These are explicitly tagged with `is_partial_week = True` and `days_observed`:

| Boundary Week | Monday (`week_start_date`) | Sunday (`week_end_date`) | Days Observed | Days Missing in Week | Reason & Handling |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Initial Week (Week 1)** | `2012-12-31` | `2013-01-06` | **6 days** | 1 day (Monday 2012-12-31) | Dataset begins on Tuesday Jan 1, 2013. Sales represent 6 trading days. |
| **Full Interior Weeks** | Weeks 2 to 241 | Weeks 2 to 241 | **7 days** | 0 days | **240 full consecutive 7-day calendar weeks.** |
| **Final Week (Week 242)** | `2017-08-14` | `2017-08-20` | **2 days** | 5 days (Aug 16 to Aug 20) | Training data ends Tuesday Aug 15. The remaining 5 days reside in `test.csv`. |

> [!IMPORTANT]
> **Forecasting Alignment Note:** When evaluating 13-week forecast models, Week 242 (with only 2 days observed) must not be treated as a full 7-day sales week without daily-rate scaling or proper time-series truncation to Week 241.

---

## 4. Promotion Aggregation & Statistical Meaning

In raw retail logs, `onpromotion` represents the **number of unique product items** within a product family that were under active price promotion on a given day. Summing and averaging across the 7-day week yields distinct promotional metrics:

| Field Name | Data Type | Statistical Meaning | FMCG Business Interpretation |
| :--- | :---: | :--- | :--- |
| **`onpromotion_sum`** | `int64` | Total promotional item-days | Cumulative promotion exposure across the week. |
| **`onpromotion_mean`** | `float64` | Average daily promoted items | Baseline promotional intensity. |
| **`onpromotion_max`** | `int32` | Peak daily promoted items | Maximum promotional breadth on any single day of the week. |
| **`promo_days_count`** | `int8` | Number of days with promotion | Promotion duration (0 to 7 days). |
| **`has_promotion`** | `bool` | Any promotion active in week | Binary flag indicating whether the category was promoted at all. |

*Crucial rule: These metrics represent promotional breadth and intensity, NOT unit sales volume.*

---

## 5. Store Operational Lifecycle & Closure Distribution

Rather than converting every zero-sales observation to an active zero-demand day, the pipeline separates operational history into three distinct statuses:

| Status (`store_status`) | Row Count | % of Dataset | Explanation & Modeling Recommendation |
| :--- | :--- | :---: | :--- |
| **`operational`** | **399,578** | **92.66%** | Store was open and trading. Zero sales in this state represent genuine zero consumer demand. |
| **`pre_opening`** | **31,515** | **7.31%** | Occurs before the store's confirmed launch date (e.g. Store 52 opened in April 2017). **Exclude from baseline demand training.** |
| **`temporary_closure`** | **151** | **0.03%** | Store was officially launched, but closed all days of that week (e.g. remodeling or local disaster). |

---

## 6. Output Dataset Schema (`weekly_sales.csv`)

```text
Columns (24 total):
├── Identifiers & Time:
│   ├── week_start_date (str, YYYY-MM-DD Monday)
│   ├── week_end_date (str, YYYY-MM-DD Sunday)
│   ├── year (int32)
│   ├── week_number (int32, 1-53)
│   ├── store_nbr (int16, 1-54)
│   └── family (str, 33 categories)
├── Target:
│   └── weekly_sales (float64, sum of sales units)
├── Promotional Features:
│   ├── onpromotion_sum (int64)
│   ├── onpromotion_mean (float64)
│   ├── onpromotion_max (int32)
│   ├── promo_days_count (int8, 0-7)
│   └── has_promotion (bool)
├── Operational Context:
│   ├── days_observed (int8, 2, 6, or 7)
│   ├── days_store_open (int8, 0-7)
│   ├── days_store_closed (int8, 0-7)
│   ├── is_partial_week (bool)
│   └── store_status (str: operational, pre_opening, temporary_closure)
└── External & Metadata Drivers:
    ├── weekly_transactions (int32, foot traffic)
    ├── oil_price_avg (float64, WTI crude weekly mean)
    ├── holidays_in_week (int8, count of holidays)
    ├── city (str)
    ├── state (str)
    ├── type (str, store format A-E)
    └── cluster (int8, 1-17)
```

---

## 7. Recommendations for Phase 4: 13-Week Forecasting Features

1. **Horizon Definition:**
   - 13 weekly steps ($t+1, t+2, \dots, t+13$) align directly with retail quarterly supply chain planning.
2. **Weekly Lagged Features:**
   - Safe weekly lags: $\text{Lag}_{1}, \text{Lag}_{2}, \text{Lag}_{4}, \text{Lag}_{13}, \text{Lag}_{52}$ (capturing 1-week, 2-week, 1-month, 1-quarter, and 1-year annual seasonality).
   - Rolling weekly statistics: 4-week, 13-week, and 26-week rolling mean and standard deviation of `weekly_sales`.
3. **Filtering Rules:**
   - Filter out `store_status == 'pre_opening'` rows when calculating store demand baselines.
   - For evaluation, align test predictions with full 7-day equivalent weeks.
