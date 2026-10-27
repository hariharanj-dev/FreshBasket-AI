# FreshBasket AI — Modern React Frontend & FastAPI Backend

A production-grade, enterprise FMCG Demand Forecasting and Inventory Intelligence web application built with **React**, **Vite**, **Tailwind CSS**, and **FastAPI**.

---

## 🚀 Architecture Overview

- **Frontend**: React 19, Vite, React Router v7, Recharts, Axios, Lucide React, Tailwind CSS 3.4
- **Backend**: FastAPI, Uvicorn, Pandas, NumPy, Scikit-learn, LightGBM
- **Forecasting Models**: Naive Baseline (Verified Rank #1, 11.61% WAPE), LightGBM Recursive (12.27% WAPE), Seasonal Naive Lag-52 (16.17% WAPE)
- **Inventory Engine**: Deterministic Lead-Time Demand Aggregation, Safety Stock, Reorder Point ($ROP = \hat{D}_{LT} + SS$), and Discrete Replenishment Units ($Q = \lceil \max(0, ROP - \text{Stock}) \rceil$)

---

## 🛠️ Quick Start

### 1. Start the FastAPI Backend

From the project root:

```bash
# Activate your Python virtual environment
.\.venv\Scripts\activate

# Start the FastAPI server (runs on http://localhost:8000)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The FastAPI interactive documentation is available at:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### 2. Start the React Frontend

Open a new terminal and navigate to `frontend/`:

```bash
cd frontend

# Install dependencies (if not already installed)
npm install

# Start Vite development server (runs on http://localhost:5173)
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) in your web browser.

### 3. Build for Production

```bash
cd frontend
npm run build
npm run preview
```

---

## 📡 REST API Endpoints Specification

| Method | Endpoint | Description | Query Parameters |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | System health check, dataset lengths, and inventory data mode | None |
| `GET` | `/api/filters` | 54 stores with metadata, 33 product families, models, date bounds | None |
| `GET` | `/api/overview` | Executive KPIs, actual vs forecast timeline, model leaderboard, top families | `model`, `store_nbr`, `family` |
| `GET` | `/api/sales` | Historical weekly series with 4-week MA, top stores, top families, monthly aggregations | `store_nbr`, `family`, `start_date`, `end_date`, `exclude_partial_weeks` |
| `GET` | `/api/forecasts` | 13-week forward predictions across all models, cumulative units, horizon matrix | `model`, `store_nbr`, `family` |
| `GET` | `/api/evaluation` | Global metrics (WAPE, MAE, RMSE, RMSLE), horizon 1-13 error decay, RMSLE zero-sales audit | None |
| `GET` | `/api/products` | 33 product family rankings, category forecast accuracy, LightGBM wins | `model` |
| `GET` | `/api/stores` | 54 retail store rankings, store format types (A–E), operational clusters | `model`, `store_nbr` |
| `GET` | `/api/inventory/summary` | Risk scorecard (High/Medium/Low), lead-time demand, suggested orders, rule-based heuristics | `model`, `store_nbr`, `family` |
| `GET` | `/api/inventory/recommendations` | Item-level replenishment recommendations with risk tiers and pagination | `model`, `store_nbr`, `family`, `risk_level`, `limit`, `offset` |
| `POST` | `/api/inventory/upload` | Upload custom retail inventory CSV with schema and positive-value validation | Multipart file (`file`) |
| `POST` | `/api/inventory/reset` | Reset inventory to baseline demo synthetic balances | None |
| `GET` | `/api/inventory/alerts` | Alert triage feed, 54-store risk exposure rollup, affected store/family counts | `model`, `store_nbr`, `family`, `risk_level` |
| `POST` | `/api/inventory/alerts/snapshot` | Append current alert exposure to persistent audit log (`alert_history.csv`) | `model` |
| `GET` | `/api/inventory/alerts/history` | Retrieve persistent alert audit log history | None |
| `GET` | `/api/reports/replenishment` | Stream generated replenishment order plan as downloadable CSV file | `model`, `store_nbr`, `family`, `risk_level` |
| `GET` | `/api/reports/executive` | Compile and stream real executive briefing markdown document | `model`, `store_nbr`, `family` |
| `GET` | `/api/reports/email` | Generate staging email communication draft for suppliers/buyers | `model`, `store_nbr`, `family` |
| `GET` | `/api/insights/overview` | Executive AI business recommendations with conclusion, supporting data, actions & limitations | `model`, `store_nbr`, `family` |
| `GET` | `/api/insights/demand-trends` | Statistical demand trend classifier (Rising, Declining, Stable) via OLS regression & t-test | `store_nbr`, `family`, `window_weeks` |
| `GET` | `/api/insights/product-opportunities` | Forward forecast intelligence, volume concentration, model divergence, horizon uncertainty | `model`, `store_nbr`, `family` |
| `GET` | `/api/insights/inventory-risks` | Strategic inventory risk evaluation: high risk SKUs, items below ROP, lead-time exposure | `model`, `store_nbr`, `family` |
| `POST` | `/api/assistant/query` | "Ask FreshBasket AI" natural language business query assistant with 10 supported query domains | Body: `question`, `store_nbr?`, `family?`, `date_range?`, `model?` |
| `POST` | `/api/scenarios/simulate` | What-If Scenario Simulator: demand shocks (-50% to +100%), stock changes (-75% to +100%), lead-time adjustments (-7 to +30 days) | Body: `demand_change_pct`, `stock_change_pct`, `lead_time_adjustment_days`, `store_nbr?`, `family?`, `model?` |

---

## ⚙️ Environment Variables

Create `.env` inside `frontend/` (optional for local development, defaults to port 8000):

```env
VITE_API_URL=http://localhost:8000
```

---

## 📊 Application Pages (11 Core Modules)

1. **Executive Overview (`/`)**: High-level KPIs, dual-series trajectory chart (historical actual vs 13-week predictions), model leaderboard, top 10 FMCG families.
2. **AI Assistant (`/assistant`)**: "Ask FreshBasket AI" natural language business query assistant answering questions on demand leaders, model benchmarks, high-risk inventory items, replenishment units, demand trajectories, and comparisons strictly using verified project datasets.
3. **What-If Simulator (`/simulator`)**: Interactive operational scenario simulator allowing users to test demand shocks (-50% to +100%), inventory deficit/surplus (-75% to +100%), and supplier lead-time changes (-7 to +30 days) side-by-side with original plans, complete with risk migration charts and session history.
4. **Sales Analysis (`/sales`)**: Time series with 4-week moving average, store volume rankings, monthly aggregations, and paginated weekly sales table.
5. **13-Week Forecast (`/forecast`)**: Pure forward predictions from origin `2017-08-13` comparing Naive, LightGBM, and Seasonal Naive, with tabular forecasts and CSV export.
6. **AI Business Insights (`/insights`)**: Documented statistical demand trends (Rising/Declining/Stable via OLS slope & Student's t-test), forecast opportunities, model divergence analysis, and actionable inventory risk recommendations.
7. **Product Insights (`/products`)**: FMCG category volume rankings, 33-family accuracy matrix, and deep dive into categories where LightGBM leads.
8. **Store Performance (`/stores`)**: 54-store ranking leaderboard, format distribution (Types A–E), and accuracy benchmarks across Ecuador provinces.
9. **Model Evaluation (`/evaluation`)**: Audited WAPE, MAE, RMSE, RMSLE comparison, week 1–13 horizon decay curve, and zero-sales RMSLE audit.
10. **Inventory Planning (`/inventory`)**: Stockout risk scorecard, custom CSV upload with validation, interactive risk distribution donut, and replenishment recommendations.
11. **Inventory Alerts (`/alerts`)**: Priority stockout alerts feed, 54-store exposure rollup, real executive briefing (.md), staging email draft, and snapshot audit logger.

---

## 🔍 Model Truths & Verification

- **Rank #1 Best Model**: Naive Baseline with **11.61% Global WAPE** (MAE 328.7, RMSE 1,052.2).
- **Rank #2 Runner-Up**: LightGBM Recursive with **12.27% Global WAPE** (MAE 347.3, RMSE 1,085.6).
- **Rank #3 Baseline**: Seasonal Naive (Lag 52) with **16.17% Global WAPE** (MAE 457.6, RMSE 1,463.3).
- **Forecast Horizon**: Exactly 13 weeks (`2017-08-14` to `2017-11-12`).
- **Historical Data Boundary**: Last complete week is `2017-08-07` to `2017-08-13`. The partial boundary week of `2017-08-14` (only 2 observed days) is excluded to avoid misleading drops.

---

## 🔧 Troubleshooting

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **"Failed to load data" / API Offline** | FastAPI server is not running on port 8000 | Run `uvicorn api.main:app --port 8000` in the Python `.venv` environment. |
| **CORS Network Error** | Frontend origin blocked | FastAPI backend has CORS enabled with `allow_origins=["*"]`. Ensure backend is listening on `0.0.0.0` or `127.0.0.1`. |
| **CSV Upload Validation Error (422)** | Missing required columns or negative stock values | Ensure CSV has `store_nbr`, `family`, `current_stock`, and non-negative numbers. |
| **Streamlit App Conflicts** | Streamlit running on ports 8501/8502 | Streamlit and FastAPI use separate ports (8501 vs 8000) and do not conflict. |
