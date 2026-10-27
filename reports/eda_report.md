# FreshBasket AI - Exploratory Data Analysis & Business Intelligence Report 📈🛒

**Generated On:** 2026-10-24 18:29:11  
**Source Dataset:** Cleaned Unified Dataset (`data/processed/cleaned_sales.csv`)  
**Scope:** 54 Stores | 33 Product Families | Continuous 1,688-Day Horizon (2013-01-01 to 2017-08-15)  

---

## 1. Executive Summary

This report documents the exploratory analysis of **3,008,016 daily sales observations** prepared for the 13-week FMCG demand forecasting engine. All data quality issues identified during initial inspection have been resolved:
- **100% Calendar Continuity:** 4 Christmas closures were imputed with zero sales, completing the $1,782$-series grid across all 1,688 calendar days.
- **Zero Row Duplication:** Store-specific holiday aggregations prevented the 53,460 redundant rows caused by naive holiday joins.
- **Continuous Economic Drivers:** Daily crude oil prices (WTI) have been forward-filled across weekends and holidays ($0$ missing values).
- **Exact Reconciliation:** Pre-cleaning total sales volume matches post-cleaning volume to 6 decimal places ($1,073,644,952.20$ units).

---

## 2. Key Business & Demand Insights

### Demand Velocity & Seasonality
- **Daily Sales Run-Rate:** Average daily system volume is **636,046 units**, fluctuating between **0 units** (Jan 1 store closures) and **1,463,084 units** (peak relief surges and pre-holiday shopping).
- **Weekly Run-Rate:** Average weekly volume is **4,436,549 units**. Strong 7-day cyclicality is observed, with retail volumes peaking heavily on Saturdays and Sundays.
- **Annual Growth Trends:**
  - **2013 to 2014:** +49.18% growth driven by network expansion.
  - **2014 to 2015:** +14.99% organic volume growth.
  - **2015 to 2016:** +19.83% volume surge, driven in part by emergency relief supplies following the April 16, 2016 Ecuador Earthquake.
- **Holiday & Month-End Spikes:** Significant surges occur in December (+35% to +50% over baseline) and on the 15th and last day of each month, aligning with public/private bi-weekly salary disbursement dates in Ecuador (*quincena* effect).

### Promotional Impact
- **Observed Volume Lift:** Products on active promotion sold an average of **1136.47 units/day**, compared to **176.0 units/day** on non-promoted days — an observed difference of **+545.74%**.
- **Promotion Depth:** Demand increases non-linearly with promotion depth. However, promotional features must be modeled with product category interactions, as impulse categories (beverages, snacks) respond substantially stronger than staple goods.

### Store-Level Performance Disparities
- **Top Store:** **Store 44** generated the highest total volume (**62,087,553 units**), benefiting from a high-density urban catchment area and Large Supermarket format.
- **Lowest Volume Store:** **Store 52** generated **2,696,170 units** (Store 52 opened late in April 2017, having only 4 months of operating history).
- **Format Differentiation:** Superstore types A and D consistently outperform smaller type C stores in throughput, though type C stores exhibit lower demand coefficient of variation.

### Product Family Concentration & Volatility
- **Pareto Principle:** The top 5 product families (`GROCERY I`, `BEVERAGES`, `PRODUCE`, `CLEANING`, `DAIRY`) account for **78.72%** of total unit throughput.
- **Least Volatile Category:** `CLEANING` exhibits highly predictable, staple demand patterns with low coefficient of variation.
- **Most Volatile Category:** `BOOKS` experiences high demand intermittency and sharp seasonal swings.

---

## 3. Visual Catalog of Analysis Charts

All generated high-resolution charts are saved in `reports/figures/`:

| Figure | Chart Description | Key Insight Observed |
| :---: | :--- | :--- |
| **01** | [`01_daily_sales_trend.png`](figures/01_daily_sales_trend.png) | Long-term growth trajectory and the April 2016 earthquake relief spike. |
| **02** | [`02_weekly_sales_trend.png`](figures/02_weekly_sales_trend.png) | 13-week rolling demand trend illustrating underlying retail seasonality. |
| **03** | [`03_monthly_sales_trend.png`](figures/03_monthly_sales_trend.png) | Month-over-month seasonality highlighting December holiday peaks. |
| **04** | [`04_sales_by_product_family.png`](figures/04_sales_by_product_family.png) | Unit distribution across all 33 FMCG product families. |
| **05** | [`05_top10_families.png`](figures/05_top10_families.png) | Top 10 categories representing over 85% of total system volume. |
| **06** | [`06_sales_by_store.png`](figures/06_sales_by_store.png) | Revenue and unit throughput ranking across all 54 retail store locations. |
| **07** | [`07_sales_dist_zero_freq.png`](figures/07_sales_dist_zero_freq.png) | Zero-demand vs positive sales distribution (log-normal positive sales). |
| **08** | [`08_promotion_impact.png`](figures/08_promotion_impact.png) | Comparative unit sales on promotional vs non-promotional trading days. |
| **09** | [`09_holiday_sales_patterns.png`](figures/09_holiday_sales_patterns.png) | Demand variation on national and local public holidays vs normal days. |
| **10** | [`10_store_sales_variability.png`](figures/10_store_sales_variability.png) | Demand coefficient of variation (CV) ranked across stores. |
| **11** | [`11_family_sales_variability.png`](figures/11_family_sales_variability.png) | Intermittency and volatility profile across product categories. |
| **12** | [`12_data_quality_missing_matrix.png`](figures/12_data_quality_missing_matrix.png) | Cleaned grid verification showing 0 nulls and 1,782 continuous daily series. |

---

## 4. Analytical Caveats & Limitations

1. **Correlation vs Causation:**
   - Higher average sales during promotional days reflects both promotional stimulation and promotional scheduling during anticipated peak seasons (e.g. holiday discounts). Causation should not be claimed without uplift modeling or counterfactual controls.
2. **Exogenous Earthquake Anomaly (April–May 2016):**
   - A 7.8 magnitude earthquake struck Ecuador on April 16, 2016. High demand for water, canned goods, and cleaning supplies in subsequent weeks represents an emergency humanitarian response rather than sustainable organic retail growth. Models must avoid over-indexing on this isolated shock.
3. **Store Ramp-Up Horizon:**
   - Newly opened stores (Stores 20, 21, 22, 29, 36, 42, 52, 53) show lower cumulative volumes because they did not trade during earlier years. The `is_store_active` flag should be used to condition training windows.
4. **Transactions in Forecast Horizon:**
   - In-store transaction counts are an output of consumer foot-traffic and are unobserved for future horizons ($t + 1$ to $t + 13$ weeks). Therefore, same-day transactions must never be used as concurrent predictive features.

---

## 5. Recommendations for Phase 4: Feature Engineering

1. **Calendar & Payday Features:**
   - Day of week (1–7), Day of month (1–31), Month of year (1–12), Week of year (1–52).
   - Bi-weekly salary indicators (*quincena* flags on 15th and end of month).
2. **Horizon-Safe Lagged Demand Signals:**
   - For direct 13-week forecasting: Lags must start at $\ge 14$ days or $\ge 91$ days depending on direct vs recursive strategy.
   - 7-day, 14-day, 28-day, and 91-day rolling statistics (mean, std, min, max, exponentially weighted moving averages).
3. **Promotion Features:**
   - Current promotion status, rolling promotional intensity (number of items on promotion in past 7 and 14 days).
4. **Target Encoding & Hierarchical Embeddings:**
   - Store cluster, store type, and city mean historical volume embeddings.
   - Product family hierarchical groupings (Staple vs Fresh vs Household).
5. **Loss Functions & Evaluation Metrics:**
   - Heavy right-skew and zero prevalence dictate evaluating models on **WAPE (Weighted Absolute Percentage Error)**, **MAE**, and **RMSLE** rather than standard MSE.
