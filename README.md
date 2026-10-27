# FreshBasket AI 🛒📊

FreshBasket AI is a professional FMCG (Fast-Moving Consumer Goods) demand forecasting and business intelligence dashboard. It is designed to predict weekly sales for a 13-week forecast horizon using real-world store, product, and transaction CSV datasets.

---

## 🎯 Project Objective

To empower retail category managers, store planners, and supply chain analysts with:
- **13-Week Demand Forecasting:** Accurate weekly unit and revenue predictions using machine learning (LightGBM & Scikit-learn).
- **Business Intelligence & Insights:** Intuitive Streamlit and Plotly dashboards to track trends, promotional impacts, and inventory risks.
- **Scenario & What-If Analysis:** Simulating price adjustments, promotional campaigns, and supply disruptions.
- **Explainability:** Clear, interpretable insights showing the drivers behind forecast numbers.

---

## 🛠️ Technology Stack

- **Core & Data Processing:** Python, Pandas, NumPy
- **Machine Learning & Modeling:** Scikit-learn, LightGBM, Joblib
- **Interactive UI & Visualization:** Streamlit, Plotly
- **Code Quality & Testing:** Pytest

---

## 📁 Project Structure

```text
freshbasket-ai/
├── data/
│   ├── raw/             # Original immutable CSV datasets
│   ├── processed/       # Cleaned, merged, and feature-ready tables
│   └── forecasts/       # Output forecast results and predictions
├── src/
│   ├── data/            # Ingestion, validation, and cleaning scripts
│   ├── features/        # Feature engineering (lags, rolling stats, calendar)
│   ├── models/          # Model training, inference, and serialization
│   ├── evaluation/      # Metrics (WAPE, RMSE, MAE, bias analysis)
│   ├── explainability/  # Feature importance and prediction drivers
│   ├── scenarios/       # Scenario & what-if simulation logic
│   └── business/        # Business KPIs, safety stock, and inventory logic
├── dashboard/
│   └── pages/           # Multi-page Streamlit dashboard modules
├── notebooks/           # Exploratory data analysis (EDA) and prototyping
├── reports/
│   └── figures/         # Generated charts, plots, and visual summaries
├── tests/               # Unit and integration test suites
├── models/
│   └── trained/         # Saved model artifacts (.joblib, .pkl)
├── scripts/             # CLI runners for data pipelines and batch jobs
├── config.py            # Centralized project configuration & paths
├── requirements.txt     # Python project dependencies
└── README.md            # Project documentation
```

---

## 🚀 Planned Development Phases

1. **Phase 1: Project Setup & Structure** (Current)
   - Initialize clean, modular project directory structure.
   - Configure dependencies, paths, and environment settings.

2. **Phase 2: Data Ingestion & Exploratory Analysis**
   - Ingest raw FMCG CSV datasets (sales, products, stores, promotions).
   - Perform data health checks, handle missing values, and validate time-series continuity.

3. **Phase 3: Feature Engineering**
   - Build time-series lag features, rolling statistics, calendar indicators, and promotional tags.
   - Prevent data leakage with strict temporal splitting.

4. **Phase 4: Model Training & Evaluation**
   - Benchmark baseline models against LightGBM regressors.
   - Evaluate performance across 13-week forecast horizons using WAPE, MAE, and RMSE.

5. **Phase 5: Business Intelligence & Explainability**
   - Compute key FMCG metrics (reorder points, safety stock, revenue impact).
   - Generate feature attribution and decision drivers.

6. **Phase 6: Interactive Dashboard**
   - Build multi-page dashboard with Plotly charts for interactive forecasting and scenario simulation.

7. **Phase 7: Testing & Documentation**
   - Add automated Pytest test coverage across data pipelines, features, and model modules.

---

## 🚀 How to Run the Project

### Option 1: Modern Web App (FastAPI Backend + React Frontend)

**Terminal 1 — Backend (FastAPI):**
```powershell
cd c:\Users\user\Desktop\freshbasket-ai
.\.venv\Scripts\activate
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```
- **Backend API:** [http://localhost:8000](http://localhost:8000)
- **API Docs (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)

**Terminal 2 — Frontend (React + Vite):**
```powershell
cd c:\Users\user\Desktop\freshbasket-ai\frontend
npm run dev -- --host 127.0.0.1 --port 5173
```
- **Web App Dashboard:** [http://localhost:5173](http://localhost:5173)

---

### Option 2: Streamlit BI Dashboard (Alternative)

```powershell
cd c:\Users\user\Desktop\freshbasket-ai
.\.venv\Scripts\activate
streamlit run app.py
```
- **Streamlit URL:** [http://localhost:8501](http://localhost:8501)
