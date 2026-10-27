# FreshBasket AI - Data Quality & Inspection Report 📊🛒

**Generated On:** 2026-10-24 18:05:09  
**Project:** FreshBasket AI - FMCG Demand Forecasting (13-Week Horizon)  
**Dataset Source:** Corporación Favorita Grocery Time-Series Datasets (`data/raw/`)  

---

## 1. Executive Summary

A comprehensive, automated data quality inspection was conducted on all 6 CSV files in `data/raw/`. The training dataset (`train.csv`) comprises **3,000,888 transactions** spanning **54 stores** and **33 product families** over **1,684 active days** (from `2013-01-01` to `2017-08-15`).

### Key Highlights
- **100% Primary Grid Completeness:** Every observed date contains records for all 54 stores and 33 product families without a single missing cell ($54 \times 33 = 1,782$ series/day).
- **Zero Primary Key Duplicates:** 0 duplicate rows across all tables and 0 duplicate combinations of `(date, store_nbr, family)`.
- **Zero Negative Sales:** All unit sales are $\ge 0$.
- **Zero Missing Values in Core Tables:** `train.csv`, `test.csv`, `stores.csv`, and `transactions.csv` have 0 missing values.
- **Identified Data Quirks:**
  1. **Holiday Row Explosion:** `holidays_events.csv` has 31 duplicate dates (multiple events on the same day). A direct join causes an explosion of **53,460 redundant rows (+1.78%)**.
  2. **Missing Oil Trading Days:** Oil prices are missing on weekends and trading holidays (43 explicit NaNs + 466 calendar dates).
  3. **Late-Opening Stores:** 8 stores opened after 2013-01-01, producing **220,572 artificial zero-sales records** before opening.
  4. **Calendar Gaps:** Exactly **4 calendar days** are missing in the 4.5-year span, corresponding to Christmas Day (Dec 25) when all stores were closed.
  5. **No Transactions in Test Set:** `transactions.csv` cannot be used as a concurrent predictor for the 13-week forecasting horizon.

---

## 2. Dataset Inventory & High-Level Metrics

| Dataset | File Size | Row Count | Column Count | Memory Usage | Date Coverage | Suitable Join Keys |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`train.csv`** | 116.16 MB | 3,000,888 | 6 | 196.78 MB | 2013-01-01 to 2017-08-15 | `date, store_nbr, family` |
| **`test.csv`** | 998.31 KB | 28,512 | 5 | 1.65 MB | 2017-08-16 to 2017-08-31 | `date, store_nbr, family` |
| **`stores.csv`** | 1.35 KB | 54 | 5 | 0.0 MB | Static metadata | `store_nbr` |
| **`transactions.csv`** | 1.48 MB | 83,488 | 3 | 2.71 MB | 2013-01-01 to 2017-08-15 | `date, store_nbr` |
| **`oil.csv`** | 20.10 KB | 1,218 | 2 | 0.03 MB | 2013-01-01 to 2017-08-31 | `date` |
| **`holidays_events.csv`** | 21.79 KB | 350 | 6 | 0.03 MB | 2012-03-02 to 2017-12-26 | `date` (requires flag aggregation) |

---

## 3. Training Dataset Deep-Dive (`train.csv`)

### Schema & Column-Level Summary
| Column | Inferred Dtype | Non-Null Count | Missing Count | Null % | Sample Values |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `id` | `int64` | 3,000,888 | 0 | 0.0% | `0, 1, 2` |
| `date` | `str` | 3,000,888 | 0 | 0.0% | `2013-01-01, 2013-01-01, 2013-01-01` |
| `store_nbr` | `int64` | 3,000,888 | 0 | 0.0% | `1, 1, 1` |
| `family` | `str` | 3,000,888 | 0 | 0.0% | `AUTOMOTIVE, BABY CARE, BEAUTY` |
| `sales` | `float64` | 3,000,888 | 0 | 0.0% | `0.0, 0.0, 0.0` |
| `onpromotion` | `int64` | 3,000,888 | 0 | 0.0% | `0, 0, 0` |

### Statistical Profile of Sales
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Total Rows** | 3,000,888 | Total recorded store-family-date sales observations |
| **Total Sales Volume** | 1,073,644,952.20 | Cumulative units sold across all stores and product families |
| **Minimum Sales** | 0.0 | Lowest recorded sales (0.0 units) |
| **Maximum Sales** | 124,717.00 | Peak sales day (grocery bulk demand / event) |
| **Mean Sales** | 357.7757 | Average daily sales per store-family combination |
| **Median Sales (P50)** | **11.0** | True central tendency (resistant to high volume outliers) |
| **25th Percentile (P25)** | 0.0 | 25% of all daily family sales are zero |
| **75th Percentile (P75)** | 195.85 | Top quartile threshold for fast-moving items |
| **Zero Sales Count** | 939,130 (31.3%) | Expected in FMCG for slow-moving categories or pre-opening stores |
| **Positive Sales Count** | 2,061,758 (68.7%) | Active commercial transactions |
| **Negative Sales Count** | 0 (0.0%) | No erroneous negative return records present |

### Statistical Profile of Promotions
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Zero Promo Count** | 2,389,559 (79.63%) | Everyday low price (EDLP) base demand observations |
| **Positive Promo Count** | 611,329 (20.37%) | Active promotional discount observations |
| **Total Items on Promo** | 7,810,622 | Cumulative count of product items under active promotion |
| **Max Items on Promo** | 741 items | Highest promotional event depth |
| **Mean Promo Count** | 2.6028 | Average promotional intensity |
| **Median Promo Count** | 0.0 | Typical non-promoted baseline day |

### Grid Completeness & Calendar Continuity
- **Unique Stores:** 54
- **Unique Product Families:** 33
- **Unique Daily Time Series:** $54 \times 33 = 1,782$ series.
- **Observed Dates:** 1,684 days. Expected grid = $1,782 \times 1,684 = 3,000,888$ rows (**100.0% Complete** ✅).
- **Calendar Span:** 1,688 days (`2013-01-01` to `2017-08-15`). Full span grid = $1,782 \times 1,688 = 3,007,896$ rows.
- **Missing Combinations:** Exactly 7,008 combinations (4 missing days x 1,782 series), caused by nationwide Christmas Day store closures:
  - `2013-12-25`
  - `2014-12-25`
  - `2015-12-25`
  - `2016-12-25`

---

## 4. Join Key Compatibility & Row Multiplication Risks

| Join Target | Key Columns | Cardinality | Null Impact | Row Explosion Impact | Validation Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`stores.csv`** | `store_nbr` | Many-to-1 | 0 missing | 0 excess rows | **Safe to join directly** ✅ |
| **`oil.csv`** | `date` | Many-to-1 | 466 dates missing in train + 43 explicit NaNs | 0 excess rows | **Requires calendar reindexing & `ffill()`** ⚠️ |
| **`holidays_events.csv`** | `date` | Many-to-Many | 0 nulls | **+53,460 excess rows (+1.78%)** | **CRITICAL: Must aggregate into binary flags first** 🚨 |
| **`transactions.csv`** | `date, store_nbr` | Many-to-1 | 7,448 store-date pairs missing | 0 excess rows | **Lagged features only; absent in test set** ⚠️ |

---

## 5. Sales & Business Breakdown

### Annual Sales Trend
| Year | Total Sales (Units) | YoY Growth | Business Context |
| :--- | :--- | :--- | :--- |
| **2013** | 140,419,013.92 | Baseline | Baseline operations (store 25 was only store open on Jan 1) |
| **2014** | 209,474,246.30 | +49.18% | Store network expansion and supply chain scaling |
| **2015** | 240,880,100.65 | +14.99% | Sustained organic demand growth |
| **2016** | 288,654,522.95 | +19.83% | Peak volume period (emergency relief surge following April 2016 earthquake) |
| **2017** | 194,217,068.37 | N/A (Partial) | Covers Jan 1 to Aug 15 (7.5 months) |

### Top 10 Product Families by Volume
| Rank | Product Family | Total Sales | % of Total Volume | Category Type |
| :---: | :--- | :--- | :--- | :--- |
| 1 | **GROCERY I** | 343,462,734.89 | 31.99% | Food & Beverage |
| 2 | **BEVERAGES** | 216,954,486.00 | 20.21% | Food & Beverage |
| 3 | **PRODUCE** | 122,704,684.68 | 11.43% | Food & Beverage |
| 4 | **CLEANING** | 97,521,289.00 | 9.08% | Household & Care |
| 5 | **DAIRY** | 64,487,709.00 | 6.01% | Food & Beverage |
| 6 | **BREAD/BAKERY** | 42,133,945.58 | 3.92% | Food & Beverage |
| 7 | **POULTRY** | 31,876,004.47 | 2.97% | Food & Beverage |
| 8 | **MEATS** | 31,086,468.40 | 2.90% | Food & Beverage |
| 9 | **PERSONAL CARE** | 24,592,051.00 | 2.29% | Household & Care |
| 10 | **DELI** | 24,110,322.47 | 2.25% | Food & Beverage |

---

## 6. Identified Data Quality Issues & Mitigation Strategies

### 1. Row Multiplication from Holiday Joins (Severity: High 🚨)
- **Issue:** 31 dates contain multiple holiday records (e.g. 2014-06-25 has 4 entries). A standard left join increases row count from **3,000,888 to 3,054,348 (+53,460 rows)**.
- **Mitigation:**
  1. Filter out transferred holidays (`transferred == True`).
  2. Map store city/state to create 1-to-1 indicators per `(date, store_nbr)`:
     - `is_holiday` (boolean)
     - `is_national_holiday` (boolean)
     - `is_local_holiday` (boolean, matches store city)
     - `is_event` (earthquake, world cup)

### 2. Missing Oil Market Prices (Severity: Medium ⚠️)
- **Issue:** Oil trading is inactive on weekends and public holidays, leaving 466 dates without entries.
- **Mitigation:** Create continuous daily calendar, then forward-fill (`ffill()`) oil prices to propagate Friday close prices through the weekend.

### 3. Late-Opening Stores & Synthetic Zeros (Severity: Medium ⚠️)
- **Issue:** 8 stores opened after `2013-01-01` (Store 36 in May 2013, Store 53 in May 2014, Store 20 in Feb 2015, Store 29 in Mar 2015, Store 21 in Jul 2015, Store 42 in Aug 2015, Store 22 in Oct 2015, Store 52 in Apr 2017). They generate **220,572 artificial zero-sales rows**.
- **Mitigation:** When calculating historical demand baselines or time-series lags, mask or filter observations prior to each store's first operational date.

### 4. Christmas Day Calendar Gaps (Severity: Low ℹ️)
- **Issue:** Dec 25 is missing in 2013, 2014, 2015, and 2016.
- **Mitigation:** Reindex series to continuous daily calendar before constructing lag features, filling Dec 25 sales with `0.0`.

### 5. Absence of Transactions in Future Horizon (Severity: High 🚨)
- **Issue:** `transactions.csv` terminates on `2017-08-15` and is absent in `test.csv`.
- **Mitigation:** Never use concurrent daily transactions as a model feature for 13-week forecasting. Only use rolling lagged transaction statistics ($t - 14$ or older) or model demand from store/calendar/pricing drivers.
