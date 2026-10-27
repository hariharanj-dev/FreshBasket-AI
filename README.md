# FreshBasket AI 🛒📊
### Enterprise FMCG Demand Forecasting & Inventory Planning System
*A modern, full-stack machine learning platform predicting weekly retail demand across 1,782 series, quantifying uncertainty, and driving automated inventory replenishment.*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.14-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19.3+-61DAFB.svg)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8.3+-646CFF.svg)](https://vitejs.dev/)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4+-38B2AC.svg)](https://tailwindcss.com/)
[![Naive Baseline](https://img.shields.io/badge/Model-Naive%20Baseline%20(11.61%25%20WAPE)-brightgreen.svg)](#3--audited-multi-model-benchmark)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM%20GBDT%20(12.27%25%20WAPE)-blueviolet.svg)](https://lightgbm.readthedocs.io/)
[![Seasonal Naive](https://img.shields.io/badge/Model-Seasonal%20Naive%20Lag--52%20(16.17%25%20WAPE)-orange.svg)](#3--audited-multi-model-benchmark)
[![Tests Passing](https://img.shields.io/badge/Tests-112%2F112%20Passing-success.svg)](file:///c:/Users/user/Desktop/freshbasket-ai/tests/)

---

## 🎯 Project Overview

**FreshBasket AI** is a production-grade demand forecasting and inventory intelligence web application built for the FMCG (Fast-Moving Consumer Goods) retail sector. It predicts weekly unit sales across **1,782 time series** ($54 \text{ retail stores} \times 33 \text{ product categories}$) for a **13-week forward procurement window** (a complete 90-day retail quarter).

The system pairs machine learning models with an automated **inventory replenishment engine ($ROP = \hat{D}_{LT} + SS$)** and an **interactive what-if scenario simulator**, delivered through a modern **React 19** frontend and **FastAPI** backend.

---

## 🚀 How to Run the Application

### 1. Start the FastAPI Backend (Terminal 1)
```powershell
cd c:\Users\user\Desktop\freshbasket-ai
.\.venv\Scripts\activate
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```
- **Backend API:** [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Start the React Frontend (Terminal 2)
```powershell
cd c:\Users\user\Desktop\freshbasket-ai\frontend
npm run dev -- --host 127.0.0.1 --port 5173
```
- **Live React Web App:** [http://localhost:5173](http://localhost:5173)

---

## ⭐ Key Highlights & Features

### 1. 🛡️ Leakage-Safe Time-Series Engineering
- **Weekly Aggregation:** Aggregated 3.0M daily records into 242 continuous trading weeks ($431,244 \text{ rows}$).
- **56 Supervised Features:** Strictly shifted (t-1) rolling means (4, 8, 13, 26W), rolling stds, lags (1, 2, 4, 8, 13, 26, 52), and Ecuadorian *quincena* payday flags (15th and month-end salary surges).
- **Out-of-Time Holdout:** Evaluated on a strict 13-week chronological holdout window (`2017-05-15` to `2017-08-13`). Zero random splitting.

### 2. 🔮 13-Week Forecasts with Prediction Intervals
- **23,166 Forward Predictions:** Full forward quarter rollout across 54 stores and 33 categories.
- **Calibrated Prediction Intervals:** Empirical residual error bounds providing **80% and 95% confidence intervals** with non-negativity constraint ($\max(0, \hat{y} \pm z \cdot \sigma)$).
- **Output:** Stored in [`data/forecasts/forecast_13_weeks.csv`](file:///c:/Users/user/Desktop/freshbasket-ai/data/forecasts/forecast_13_weeks.csv).

### 3. 📊 Audited Multi-Model Benchmark
Transparently benchmarks 3 distinct approaches across 23,166 holdout observations:

| Model Architecture | 13-Week WAPE | MAE (Units) | RMSE (Units) | RMSLE | Key Commercial Strength |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Naive Baseline ($y_{t+h} = y_t$)** | **11.61%** | **390.49** | **1,306.86** | **0.5606** | Lowest global aggregate error on intermittent tail items. |
| **LightGBM (Recursive GBDT)** | **12.27%** | 412.74 | 1,441.24 | 1.1363 | **Outperforms Naive by +10.3%** on commercial volume ($>1,000 \text{ units/wk}$). |
| **Seasonal Naive (Lag 52)** | **16.17%** | 544.02 | 2,260.38 | 1.0870 | Trailing persistence benchmark. |

> **Category Wins:** LightGBM dominates high-margin staples over Naive: **Personal Care (+5.81% lead)**, **Cleaning (+3.14% lead)**, **Home Care (+2.16% lead)**, and **Deli (+1.74% lead)**.

### 4. 🧠 Explainability & Demand Driver Attribution
- Quantified driver contribution: **Historical Sales Velocity (54.6%)**, **Calendar & Quincena Paydays (16.3%)**, **Store Hierarchy (11.1%)**, **Promotions (9.9%)**, and **Macro/Fuel (8.1%)**.
- Detailed in [`reports/explainability_report.md`](file:///c:/Users/user/Desktop/freshbasket-ai/reports/explainability_report.md).

### 5. 🔬 Feature Ablation Study
- Measured empirical accuracy degradation when omitting feature families:
  - **Lags Only (No Rolling Stats):** **+31.5% error spike** (WAPE climbs from 12.27% to 16.13%).
  - **No Calendar / Paydays:** **+15.7% error spike** (misses bi-weekly payday surges).
  - **No Promotions:** **+12.1% error spike** (blinds model to trade discounts).
- Detailed in [`reports/ablation_study_report.md`](file:///c:/Users/user/Desktop/freshbasket-ai/reports/ablation_study_report.md).

### 6. 📦 Inventory Planning & Reorder Point Heuristics
- **Lead-Time Demand ($\hat{D}_{LT}$):** Cumulative forecast over vendor lead time.
- **Reorder Point ($ROP$):** $ROP = \hat{D}_{LT} + \text{Safety Stock}$.
- **Suggested Order Units ($Q$):** $Q = \lceil \max(0, ROP - \text{Current Stock}) \rceil$ in whole discrete units.
- **Stockout Risk Tiers:** 🔴 **HIGH** (Stock $< \hat{D}_{LT}$), 🟡 **MEDIUM** (Stock $< ROP$), 🟢 **LOW** (Healthy).

### 7. ⚡ What-If Scenario Simulator
- Interactive supply chain stress testing in the React UI:
  - Commercial demand elasticity scaling: $-50\%$ to $+100\%$.
  - Available stock shrinkage: $-75\%$ to $+100\%$.
  - Vendor lead-time delays: $-7$ to $+30$ days.
- Recomputes ROP, purchase order units, and risk tiers in real time.

### 8. 💻 Enterprise React 19 Dashboard
- Modern 11-page web interface built with **React 19**, **Vite**, **Tailwind CSS**, and **Recharts**:
  - Executive Overview & KPI cards
  - Historical Sales & 4-Week Moving Averages
  - 13-Week Forecast Browser & Tabular Exports
  - Statistical OLS Trend Detection & Anomaly Spotting
  - Store Network (54 branches) & Format Breakdown (Types A–E)
  - Model Evaluation & Horizon Error Decay
  - Inventory Replenishment & Custom CSV Upload
  - Stockout Alert Triage Center & Briefing Generator
  - What-If Scenario Simulator
  - Natural-Language Replenishment Query Assistant

---

## 🧪 Automated Testing (112 Tests Passing)

All pipelines, algorithms, and endpoints are validated via Pytest:
```powershell
pytest
```
```text
======================= 112 passed, 1 warning in 16.83s =======================
```
- ✅ **15 FastAPI REST Endpoints** (`tests/test_api.py`)
- ✅ **10 Modeling & 13-Week Predictions** (`tests/test_modeling.py`)
- ✅ **20 Inventory ROP & Lead-Time Demand** (`tests/test_inventory.py`)
- ✅ **10 Scenario Simulator Tests** (`tests/test_scenarios.py`)
- ✅ **10 Statistical Demand Insights** (`tests/test_insights.py`)
- ✅ **8 Stockout Alert Engine Tests** (`tests/test_alerts.py`)
- ✅ **14 Replenishment Assistant Tests** (`tests/test_assistant.py`)
- ✅ **12 Data Cleaning & Features Tests** (`tests/test_cleaning.py`, `tests/test_features.py`)
- ✅ **3 Explainability & Ablation Tests** (`tests/test_explainability.py`, `tests/test_ablation.py`)

---

## 📁 Project Structure

```text
freshbasket-ai/
├── api/                                # FastAPI Backend REST Service (24 Endpoints)
│   └── main.py                         # REST engine serving forecasts, metrics & inventory
├── frontend/                           # Enterprise React 19 + Vite Application
│   ├── src/
│   │   ├── App.jsx                     # Route definitions (11 analytical pages)
│   │   ├── components/                 # Reusable UI widgets (Header, Sidebar, DataTable, KpiCard)
│   │   ├── context/FilterContext.jsx   # Global store, category, model, and date filter state
│   │   ├── services/api.js             # Axios API client
│   │   └── pages/                      # 11 Dedicated Analytical Dashboard Pages
│   ├── package.json                    # React 19, Recharts, Tailwind CSS dependencies
│   └── vite.config.js                  # Vite server & build config
├── data/                               # Data Storage Layer
│   ├── raw/                            # Immutable raw CSVs (sales, stores, oil, holidays)
│   ├── processed/                      # Cleaned daily (3.0M rows) & weekly features (56 cols)
│   ├── forecasts/                      # 13-Week forecast with 80% & 95% prediction intervals
│   └── inventory/                      # Store inventory positions & supplier lead times
├── models/trained/                     # Serialized LightGBM models, configs & metadata
├── src/                                # Core Python Data Science & Business Logic
│   ├── data/                           # Data cleaning, validation, and weekly aggregation
│   ├── features/                       # 56-feature engineering pipeline (lags, rolling stats)
│   ├── models/                         # LightGBM training & 13-week recursive forecaster
│   ├── evaluation/                     # Multi-model evaluation & ablation study
│   ├── explainability/                 # Demand driver attribution & importance analysis
│   └── business/                       # Inventory ROP heuristics, alerts & scenario simulator
├── reports/                            # Comprehensive Markdown & JSON Audit Reports
│   ├── model_evaluation_report.md      # Out-of-time holdout benchmark
│   ├── explainability_report.md        # Driver attribution & case studies
│   ├── ablation_study_report.md        # Feature ablation experiments
│   └── presentation_deck.md            # 14-slide executive presentation & defense guide
├── tests/                              # Automated Pytest Suite (112 Tests)
└── README.md                           # Master Project Documentation
```

---

## 📑 Core Reviewer Deliverables

- **13-Week Forecast File (with Uncertainty):** [`data/forecasts/forecast_13_weeks.csv`](file:///c:/Users/user/Desktop/freshbasket-ai/data/forecasts/forecast_13_weeks.csv)
- **Model Evaluation Audit:** [`reports/model_evaluation_report.md`](file:///c:/Users/user/Desktop/freshbasket-ai/reports/model_evaluation_report.md)
- **Demand Driver Attribution:** [`reports/explainability_report.md`](file:///c:/Users/user/Desktop/freshbasket-ai/reports/explainability_report.md)
- **Feature Ablation Study:** [`reports/ablation_study_report.md`](file:///c:/Users/user/Desktop/freshbasket-ai/reports/ablation_study_report.md)
- **10–15 Min Presentation Deck:** [`reports/presentation_deck.md`](file:///c:/Users/user/Desktop/freshbasket-ai/reports/presentation_deck.md)
- **Data Quality & EDA Reports:** [`reports/data_quality_report.md`](file:///c:/Users/user/Desktop/freshbasket-ai/reports/data_quality_report.md) and [`reports/eda_report.md`](file:///c:/Users/user/Desktop/freshbasket-ai/reports/eda_report.md)
