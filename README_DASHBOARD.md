# FreshBasket AI — FMCG Demand Forecasting & BI Dashboard 🛒📊

Welcome to the **FreshBasket AI Streamlit Business Intelligence Dashboard**. This application provides grocery executives, replenishment managers, and demand planners with end-to-end visibility into historical sales, multi-step 13-week forward forecasts, store-level performance, and audited model accuracy metrics.

---

## 🌟 Key Dashboard Features

The dashboard is structured into six comprehensive analytical pages:

1. **🏢 Executive Overview:**
   - Real-time KPI metric cards (Total Historical Sales, 13-Week Forecasted Units, Active Store Count, FMCG Product Families, Leading Global Model).
   - Clear visual separation of the historical analysis period (`2013-01-01` to `2017-08-13`), holdout evaluation origin (`2017-05-14`), and forward forecast horizon (`2017-08-14` to `2017-11-12`).
   - Combined historical demand trajectory and multi-model forecast projections.
   - Global model WAPE leaderboard and top-10 product family distribution.

2. **📈 Sales Analysis:**
   - Interactive weekly sales time series with 4-week moving average trendlines.
   - Top store volume rankings and category share breakdowns.
   - Monthly sales aggregation matrix.
   - Dynamic sidebar filtering by store, product family, and calendar date range.
   - Direct CSV data export of filtered historical views.

3. **🔮 13-Week Forecast:**
   - Forward-looking 13-week procurement projections across 54 stores and 33 product families ($23,166 \text{ series-weeks}$).
   - Multi-model comparison across **LightGBM (Recursive)**, **Naive Baseline**, and **Seasonal Naive (Lag 52)**.
   - Detailed tabular view of weekly predictions with formatting.
   - Comprehensive documentation of the forecast origin (`2017-08-13`) and full 7-day future weeks.
   - Exportable forecast datasets in CSV format.

4. **🧺 Product Family Insights:**
   - Accuracy comparisons across all 33 FMCG categories.
   - Identification of categories where LightGBM machine learning significantly outperforms persistence baselines (e.g., Personal Care $+5.81\%$, Cleaning $+3.14\%$, Deli $+1.74\%$).
   - Interactive category deep-dive with error metrics (MAE, RMSE, WAPE, RMSLE).

5. **🏬 Store Performance:**
   - Store network ranking across 54 retail branches.
   - Store metadata integration (City, State, Store Type A-E, and Operational Cluster 1-17).
   - Store format type volume distribution.
   - Granular forecast precision matrix per branch.

6. **🔬 Model Evaluation & Audit:**
   - Completely transparent, audited evaluation metrics from the 13-week holdout simulation.
   - Week-by-week horizon error decay curve (Weeks 1 to 13).
   - Honest representation: **Naive Baseline holds the lowest global WAPE at 11.61%**, while **LightGBM achieves 12.27%** and **Seasonal Naive achieves 16.17%**.
   - Technical investigation into the **RMSLE Discrepancy** (explaining how intermittent zero-sales penalize continuous gradient boosted regressors while LightGBM leads by $10.3\%$ on core commercial volume $>1,000$ units).

7. **📦 Inventory Planning & Stockout Risk Analysis (Phase 8):**
   - Integrates 13-week forecast trajectories with supplier replenishment lead times and safety stock buffers.
   - Baseline deterministic reorder heuristics: $\text{Reorder Point (ROP)} = \text{Lead-Time Demand} + \text{Safety Stock}$.
   - Suggested replenishment quantity: $\lceil \max(0, \text{ROP} - \text{Inventory Position}) \rceil$ in whole integer units.
   - Stockout risk classification: **HIGH** (stock < lead demand), **MEDIUM** (safety stock breached), **LOW** (sufficient stock), **UNKNOWN** (invalid/missing).
   - Interactive Plotly diagnostics: Risk tier donut chart, top deficit items bar chart, and demand vs. stock scatter plot with 1:1 parity line.
   - CSV data upload feature with automated 5-point validation and downloadable replenishment recommendation register.

8. **🚨 Inventory Alerts & Business Reports (Phase 9):**
   - Dedicated alert triage center displaying high-, medium-, and low-risk counts, suggested replenishment units, and affected store and family counts.
   - Store-wise exposure aggregation: Rollup of stockout risk and order requirements per branch ($54 \text{ stores}$) with zero double-counting.
   - Downloadable CSV alert packages: Instant export buttons for High-Risk SKUs, Medium-Risk SKUs, All Filtered SKUs, and Store Summaries.
   - Executive briefing report generator: Compiles ready-to-share executive briefings with exposure metrics, top deficit tables, observations, and explicit limitations.
   - Email staging preparation: Formats email drafts ready for procurement circulation without requiring automated mailing credentials or exposing API secrets.
   - Lightweight persistent alert history: Logs snapshot audits into [`data/inventory/alert_history.csv`](file:///c:/Users/user/Desktop/freshbasket-ai/data/inventory/alert_history.csv) on demand with duplicate prevention.

---

## 🏗️ Architecture & Dashboard Data Layer

The dashboard operates on a clean, validated data layer in [`data/dashboard/`](file:///c:/Users/user/Desktop/freshbasket-ai/data/dashboard/) rather than re-computing heavy machine learning models on the fly:

- [`data/dashboard/historical_sales.csv`](file:///c:/Users/user/Desktop/freshbasket-ai/data/dashboard/historical_sales.csv): 431,244 records ($242 \text{ weeks} \times 54 \text{ stores} \times 33 \text{ families}$) with actual sales, promotional flags, transaction volume, and oil prices (`is_forecast=False`).
- [`data/dashboard/forecast_data.csv`](file:///c:/Users/user/Desktop/freshbasket-ai/data/dashboard/forecast_data.csv): 23,166 records ($13 \text{ future weeks} \times 54 \text{ stores} \times 33 \text{ families}$) containing predictions for LightGBM, Naive, and Seasonal Naive (`is_forecast=True`).
- [`data/dashboard/model_performance.csv`](file:///c:/Users/user/Desktop/freshbasket-ai/data/dashboard/model_performance.csv): Global and week-by-week horizon accuracy metrics.
- [`data/dashboard/family_performance.csv`](file:///c:/Users/user/Desktop/freshbasket-ai/data/dashboard/family_performance.csv): Category-sliced error metrics for all 33 families.
- [`data/dashboard/store_performance.csv`](file:///c:/Users/user/Desktop/freshbasket-ai/data/dashboard/store_performance.csv): Store-sliced error metrics for all 54 retail locations.
- [`data/inventory/inventory.csv`](file:///c:/Users/user/Desktop/freshbasket-ai/data/inventory/inventory.csv): 1,782 records ($54 \text{ stores} \times 33 \text{ families}$) with current stock balances, safety stock buffers, and supplier lead times.
- [`data/inventory/supplier_lead_times.csv`](file:///c:/Users/user/Desktop/freshbasket-ai/data/inventory/supplier_lead_times.csv): 1,782 supplier metadata and replenishment lead time records.

---

## 🚀 Quickstart: Running the Application

### 1. Environment Setup

Ensure your Python virtual environment is activated:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

If dependencies need to be installed or updated:

```powershell
pip install -r requirements.txt
```

### 2. Verify Data Layer (Pre-Flight Audit)

You can run the automated 8-gate validation audit at any time:

```powershell
python -m src.data.validate_dashboard_data
# or
python -m src.evaluation.validate_dashboard_data
```

Both commands will verify schema integrity, non-negativity, and key uniqueness, generating [`reports/dashboard_data_validation_report.md`](file:///c:/Users/user/Desktop/freshbasket-ai/reports/dashboard_data_validation_report.md) with **STATUS: PASS**.

### 3. Launch the Streamlit Dashboard

Run the following command from the project root (`freshbasket-ai`):

```powershell
streamlit run app.py
```

The application will launch automatically in your default web browser at:
`http://localhost:8501`

---

## 🛡️ Validation & Reliability Safeguards

- **Partial-Week Boundary Handling:** The historical dataset ends on August 15, 2017 (only 2 observed days in that final calendar week). In this dashboard:
  - Weekly velocity charts and trajectory trends display complete 7-day trading weeks (ending on the last complete observed week: Monday `2017-08-07` to Sunday `2017-08-13`).
  - This eliminates misleading visual cliff drops and prevents date overlap.
  - The 13-week forward forecast begins cleanly on Monday `2017-08-14` projecting full 7-day demand across the entire 13-week horizon.
  - Raw 2-day historical observations are fully preserved in `data/dashboard/historical_sales.csv` (`is_partial_week = True`) and inspectable in the Sales Analysis audit expander.
- **Performance Caching:** Utilizes `@st.cache_data` to ensure sub-second interaction speed when navigating pages and applying filters.
- **Strict Data Segregation:** Historical sales actuals and future predictions are separated via the `is_forecast` boolean flag; predictions are never presented as observed reality.
- **Zero-Division Protection:** Aggregations and WAPE calculations handle zero-actual segments safely.
- **Non-Negativity Guarantee:** Predictions are bounded at zero ($\max(0, \hat{y})$) to ensure validity for retail supply chain planning.

