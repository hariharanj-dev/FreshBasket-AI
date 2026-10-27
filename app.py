"""
FreshBasket AI - Executive Demand Forecasting & FMCG Business Intelligence
===========================================================================
Interactive Streamlit application providing end-to-end visibility into
historical sales, machine learning forecasts, store network analytics,
and audited model evaluation metrics.
"""

from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import importlib
import src.dashboard.data_loader
import src.dashboard.charts
import src.business.inventory_planner
import src.business.alert_engine

# Reload submodules to ensure hot-reload sync with active Streamlit runtime
importlib.reload(src.dashboard.data_loader)
importlib.reload(src.dashboard.charts)
importlib.reload(src.business.inventory_planner)
importlib.reload(src.business.alert_engine)

from src.dashboard.data_loader import (
    compute_kpis,
    filter_forecast_data,
    filter_historical_sales,
    get_model_column_mapping,
    load_all_dashboard_data,
    verify_dashboard_files_exist,
)
from src.dashboard.charts import (
    MODEL_COLORS,
    PRIMARY_GREEN,
    RISK_COLORS,
    create_family_sales_bar_chart,
    create_family_wape_comparison_chart,
    create_forecast_trajectory_chart,
    create_horizon_decay_chart,
    create_model_comparison_bar,
    create_store_ranking_chart,
    create_weekly_historical_sales_chart,
    create_stockout_risk_pie,
    create_highest_risk_items_bar,
    create_demand_vs_stock_scatter,
    render_kpi_card,
)
from src.business.inventory_planner import (
    compute_replenishment_plan,
    validate_inventory_dataframe,
    generate_inventory_business_summary,
)
from src.business.alert_engine import (
    generate_store_wise_summary,
    prioritize_alerts,
    generate_executive_business_report,
    generate_email_alert_draft,
    record_alert_snapshot,
    load_alert_history,
)


# -----------------------------------------------------------------------------
# Streamlit Application Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="FreshBasket AI - FMCG Demand Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Retail Green Theme & Clean Cards)
st.markdown(
    """
    <style>
    /* Global Styling */
    .main {
        background-color: #F8FAFC;
    }
    h1, h2, h3, h4 {
        color: #0F172A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #059669;
    }
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    /* Buttons & Accents */
    .stButton>button {
        background-color: #059669;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #047857;
        color: white;
    }
    /* Dataframe table borders */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    # -------------------------------------------------------------------------
    # Pre-Flight File & Schema Validation
    # -------------------------------------------------------------------------
    files_ok, missing_files = verify_dashboard_files_exist()
    if not files_ok:
        st.error(f"⚠️ Dashboard data files missing in `data/dashboard/`: {', '.join(missing_files)}")
        st.info("Please execute `python -m src.data.prepare_dashboard_data` to generate verified dashboard datasets.")
        return

    # Load verified datasets (Cached)
    try:
        datasets = load_all_dashboard_data()
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")
        return

    df_hist = datasets["historical"]
    df_fc = datasets["forecast"]
    df_model_perf = datasets["model_performance"]
    df_fam_perf = datasets["family_performance"]
    df_store_perf = datasets["store_performance"]
    df_inv = datasets["inventory"]
    df_suppliers = datasets["suppliers"]

    # Initialize Session State for Custom Inventory Uploads
    if "custom_inventory_df" not in st.session_state:
        st.session_state.custom_inventory_df = None
    if "data_source_mode" not in st.session_state:
        st.session_state.data_source_mode = "DEMO"

    # -------------------------------------------------------------------------
    # Sidebar: Branding & Global Filters
    # -------------------------------------------------------------------------
    st.sidebar.markdown(
        """
        <div style="padding: 10px 0 20px 0; border-bottom: 1px solid #E2E8F0; margin-bottom: 15px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 2rem;">🛒</span>
                <div>
                    <h2 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #059669; line-height: 1.1;">FreshBasket AI</h2>
                    <span style="font-size: 0.75rem; color: #64748B; font-weight: 500;">FMCG Demand Forecasting</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation Menu",
        [
            "🏢 Executive Overview",
            "📈 Sales Analysis",
            "🔮 13-Week Forecast",
            "🧺 Product Family Insights",
            "🏬 Store Performance",
            "🔬 Model Evaluation & Audit",
            "📦 Inventory Planning",
            "🚨 Inventory Alerts",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Analytics Filters")

    # Store Filter
    store_options = ["All Stores (54)"] + sorted(df_hist["store_nbr"].unique().tolist())
    selected_store_raw = st.sidebar.selectbox("Select Store", store_options, index=0)
    selected_stores = None if "All" in str(selected_store_raw) else [int(selected_store_raw)]

    # Product Family Filter
    all_families = sorted(df_hist["family"].unique().tolist())
    family_options = ["All Families (33)"] + all_families
    selected_family_raw = st.sidebar.selectbox("Select Product Family", family_options, index=0)
    selected_families = None if "All" in str(selected_family_raw) else [selected_family_raw]

    # Date Range Filter (Historical)
    min_date = df_hist["week_start_date"].min().to_pydatetime()
    max_date = df_hist["week_start_date"].max().to_pydatetime()
    selected_date_range = st.sidebar.date_input(
        "Historical Date Range",
        value=(datetime(2016, 1, 1), max_date),
        min_value=min_date,
        max_value=max_date,
    )

    start_date = selected_date_range[0] if len(selected_date_range) > 0 else min_date
    end_date = selected_date_range[1] if len(selected_date_range) > 1 else max_date

    # Model Selector (For Forecast View)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Production Model")
    model_labels = ["Naive Baseline", "LightGBM (Recursive)", "Seasonal Naive (Lag 52)"]
    selected_model_label = st.sidebar.selectbox(
        "Forecast Model",
        model_labels,
        index=0,  # Default to best WAPE model (Naive)
        help="Select the forecasting engine. Note: In holdout backtesting, Naive holds the lowest global WAPE (11.61%).",
    )
    model_col_map = get_model_column_mapping()
    selected_model_col = model_col_map[selected_model_label]

    st.sidebar.markdown(
        """
        <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 25px; line-height: 1.4;">
            <b>Operational Timeline:</b><br>
            • Backtest Holdout: 2017-05-15 to 2017-08-13<br>
            • Forecast Origin: 2017-08-13<br>
            • Future Plan: 2017-08-14 to 2017-11-12
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Apply global filters to working copies (excluding incomplete weeks by default for clean velocity trends)
    df_hist_filtered = filter_historical_sales(
        df_hist,
        selected_stores=selected_stores,
        selected_families=selected_families,
        start_date=pd.to_datetime(start_date),
        end_date=pd.to_datetime(end_date),
        exclude_partial_weeks=True,
    )

    df_fc_filtered = filter_forecast_data(
        df_fc,
        selected_stores=selected_stores,
        selected_families=selected_families,
    )

    # -------------------------------------------------------------------------
    # PAGE 1: EXECUTIVE OVERVIEW
    # -------------------------------------------------------------------------
    if page == "🏢 Executive Overview":
        st.title("🏢 Executive Demand Forecasting Overview")
        st.markdown(
            "Executive business intelligence tracking historical FMCG demand velocity and "
            "the 13-week forward procurement plan across 54 Ecuadorian retail branches."
        )

        # Clear Historical vs Forecast Boundary Callout
        st.markdown(
            """
            <div style="background-color: #F0FDF4; border: 1px solid #86EFAC; border-left: 5px solid #059669; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 800; color: #065F46; font-size: 0.95rem;">📅 Rigorous Historical-Versus-Forecast Boundary</span>
                    <span style="background-color: #DEF7EC; color: #03543F; font-size: 0.72rem; padding: 3px 8px; border-radius: 9999px; font-weight: 700;">AUDIT COMPLIANT</span>
                </div>
                <div style="color: #047857; font-size: 0.88rem; line-height: 1.5;">
                    • <b>Last Complete Historical Week:</b> Monday <code>2017-08-07</code> to Sunday <code>2017-08-13</code> (Operational Week 240)<br>
                    • <b>Production Forecast Origin:</b> Sunday close <code>2017-08-13</code><br>
                    • <b>13-Week Future Horizon:</b> Monday <code>2017-08-14</code> to Sunday <code>2017-11-12</code> (Horizon Weeks 1 to 13)<br>
                    <i>*Note on August 14, 2017 Boundary:* The raw historical dataset ends on August 15, 2017 (only 2 observed days). In this dashboard, historical velocity trends display complete 7-day weeks to prevent artificial visual volume cliffs, while the forward forecast projects full 7-day commercial demand starting August 14, 2017 with zero date overlap.</i>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Executive KPI Cards
        kpis = compute_kpis(df_hist_filtered, df_fc_filtered, model_col=selected_model_col)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(
                render_kpi_card(
                    "Historical Sales",
                    f"{kpis['total_historical_sales']:,.0f}",
                    subtitle=f"{kpis['total_weeks_historical']} complete 7-day weeks",
                    badge_text="Actuals",
                    badge_type="info",
                ),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                render_kpi_card(
                    "13-Week Forecast",
                    f"{kpis['total_forecast_sales']:,.0f}",
                    subtitle=f"Model: {selected_model_label}",
                    badge_text="13 Wk Plan",
                    badge_type="success",
                ),
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                render_kpi_card(
                    "Store Network",
                    f"{kpis['unique_stores']} / 54",
                    subtitle="Retail branches covered",
                    badge_text="Network",
                    badge_type="neutral",
                ),
                unsafe_allow_html=True,
            )
        with col4:
            st.markdown(
                render_kpi_card(
                    "Product Families",
                    f"{kpis['unique_families']} / 33",
                    subtitle="FMCG categories",
                    badge_text="Categories",
                    badge_type="neutral",
                ),
                unsafe_allow_html=True,
            )
        with col5:
            st.markdown(
                render_kpi_card(
                    "Verified Best Model",
                    "Naive (11.61%)",
                    subtitle="LightGBM: 12.27% WAPE",
                    badge_text="Benchmark",
                    badge_type="warning",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Verified Business Insights Card
        st.markdown(
            """
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 16px 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="font-size: 1.1rem;">💡</span>
                    <h3 style="margin: 0; font-size: 1rem; font-weight: 700; color: #1E293B;">Executive Business Insights (Based on Audited Backtest Data)</h3>
                </div>
                <div style="font-size: 0.86rem; color: #475569; line-height: 1.6;">
                    <b>1. Empirical Benchmark Leader:</b> The <b>Naive Baseline holds the lowest global WAPE at 11.61%</b> across all 23,166 holdout evaluations, proving that stable FMCG demand persistence is hard to beat in routine weeks.<br>
                    <b>2. Strategic Role of LightGBM (12.27% WAPE):</b> While trailing slightly overall (+0.66%), LightGBM excels during <b>volatile month-end turning points</b> (winning Weeks 3, 4, 8, 12) and outperforms across <b>9 key commercial categories</b> (Personal Care +5.81%, Cleaning +3.14%, Deli +1.74%, Home Care +2.16%).<br>
                    <b>3. Core Commercial Accuracy:</b> On high-volume series (weekly sales > 1,000 units, representing >85% of company revenue), <b>LightGBM is 10.3% more accurate in RMSLE than Naive (0.2318 vs. 0.2583)</b>.<br>
                    <b>4. Network Forward Demand:</b> Aggregate 13-week procurement demand is projected at <b>70.01M units (Naive)</b> to <b>72.93M units (LightGBM)</b>, providing procurement teams with high-confidence planning bounds.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Combined Historical & Forecast Trajectory Chart
        # Extract recent complete weeks of historical actuals as reference
        recent_hist = df_hist_filtered[df_hist_filtered["week_start_date"] >= "2017-04-01"]
        fig_trajectory = create_forecast_trajectory_chart(
            df_fc_filtered,
            df_hist_tail=recent_hist,
            active_models=[selected_model_col],
        )
        st.plotly_chart(fig_trajectory, width="stretch")

        # Bottom Row: Category Distribution & Model Comparison
        row2_col1, row2_col2 = st.columns([1, 1])
        with row2_col1:
            fig_fam = create_family_sales_bar_chart(df_hist_filtered, top_n=10)
            st.plotly_chart(fig_fam, width="stretch")

        with row2_col2:
            fig_model = create_model_comparison_bar(df_model_perf)
            st.plotly_chart(fig_model, width="stretch")

    # -------------------------------------------------------------------------
    # PAGE 2: SALES ANALYSIS
    # -------------------------------------------------------------------------
    elif page == "📈 Sales Analysis":
        st.title("📈 Historical Sales Deep-Dive & Trends")
        st.markdown(
            "Granular historical demand analytics across store formats, geographic clusters, "
            "and product categories. Filter dynamically using the sidebar controls."
        )

        if df_hist_filtered.empty:
            st.warning("No sales records found for the selected filter combination.")
            return

        # Main Weekly Sales Time Series Chart (Complete 7-day weeks)
        fig_weekly = create_weekly_historical_sales_chart(df_hist_filtered)
        st.plotly_chart(fig_weekly, width="stretch")

        # Partial-Week Audit & Data Transparency Expander
        with st.expander("🔍 Partial-Week Data Transparency & Boundary Audit"):
            st.markdown(
                """
                **Why are partial weeks excluded from weekly trend charts by default?**
                - The raw training dataset contains two calendar boundary weeks that do not represent complete 7-day trading:
                  1. `2012-12-31`: Monday only (1 day).
                  2. `2017-08-14`: Monday and Tuesday only (2 days, 1,440,490 units total) before the dataset ends on August 15, 2017.
                - Displaying a 2-day record on a 7-day weekly chart creates an artificial, misleading drop of ~70%.
                - The raw records are **completely preserved** in `data/dashboard/historical_sales.csv` with the `is_partial_week = True` flag.
                """
            )
            raw_partial = df_hist[df_hist["is_partial_week"] == True]
            st.dataframe(
                raw_partial.groupby(["week_start_date", "week_end_date"])
                .agg(
                    total_sales=("weekly_sales", "sum"),
                    series_count=("weekly_sales", "count"),
                    avg_sales_per_series=("weekly_sales", "mean"),
                )
                .reset_index(),
                width="stretch",
            )

        # Mid Row: Store Ranking & Category Share
        m_col1, m_col2 = st.columns([1, 1])
        with m_col1:
            fig_store_rank = create_store_ranking_chart(df_hist_filtered, top_n=10)
            st.plotly_chart(fig_store_rank, width="stretch")

        with m_col2:
            # Monthly summary aggregation
            df_monthly = (
                df_hist_filtered.groupby(["year", "month"])
                .agg(
                    total_sales=("weekly_sales", "sum"),
                    avg_weekly_sales=("weekly_sales", "mean"),
                    promoted_items=("onpromotion_sum", "sum"),
                )
                .reset_index()
                .sort_values(["year", "month"], ascending=False)
            )
            df_monthly["period"] = df_monthly["year"].astype(str) + "-" + df_monthly["month"].astype(str).str.zfill(2)
            
            fig_month = px.bar(
                df_monthly.head(18),
                x="period",
                y="total_sales",
                title="<b>Recent Monthly FMCG Volume Trajectory</b>",
                labels={"period": "Year-Month", "total_sales": "Total Sales (Units)"},
                color_discrete_sequence=[PRIMARY_GREEN],
            )
            fig_month.update_layout(
                plot_bgcolor="#FFFFFF",
                paper_bgcolor="#FFFFFF",
                xaxis=dict(tickangle=-45),
            )
            st.plotly_chart(fig_month, width="stretch")


        # Download Filtered Historical Sales
        st.markdown("### 📥 Export Filtered Data")
        csv_hist = df_hist_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered Historical Sales as CSV",
            data=csv_hist,
            file_name=f"freshbasket_filtered_sales_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    # -------------------------------------------------------------------------
    # PAGE 3: 13-WEEK FORECAST
    # -------------------------------------------------------------------------
    elif page == "🔮 13-Week Forecast":
        st.title("🔮 13-Week Forward Demand Forecast Plan")
        st.markdown(
            "Multi-step forward procurement projections for the next 13 complete business weeks. "
            "Predictions are generated recursively using the audited production models."
        )

        # Prominent Forecast Horizon Notice
        st.markdown(
            """
            <div style="background-color: #ECFDF5; border: 1px solid #6EE7B7; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px;">
                <span style="font-weight: 800; color: #065F46; font-size: 0.95rem;">🛡️ PREDICTED DEMAND FORECAST — NOT HISTORICAL ACTUALS</span>
                <p style="color: #047857; font-size: 0.88rem; margin: 6px 0 0 0; line-height: 1.5;">
                    • <b>Forecast Origin Date:</b> <code>2017-08-13</code> (Sunday close of last complete observed week)<br>
                    • <b>13-Week Forecast Horizon:</b> <code>2017-08-14</code> to <code>2017-11-12</code> (13 consecutive Monday-to-Sunday weeks)<br>
                    • <b>Coverage:</b> 54 Retail Branches $\\times$ 33 FMCG Families = 1,782 series $\\times$ 13 weeks = 23,166 predictions<br>
                    <i>*Full Week Planning Notice:* The raw historical dataset ended on August 15, 2017. The forecast below represents the **full 7-day expected commercial demand** for inventory planning across all 13 future weeks.</i>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # In-Page Interactive Slicers for Forecast
        st.markdown("### 🎛️ Forecast View Controls")
        fc_c1, fc_c2, fc_c3 = st.columns(3)
        with fc_c1:
            page_store_options = ["All Stores (54)"] + sorted(df_fc["store_nbr"].unique().tolist())
            p_store = st.selectbox("Store Selection", page_store_options, index=0 if selected_stores is None else page_store_options.index(selected_stores[0]))
            page_stores = None if "All" in str(p_store) else [int(p_store)]

        with fc_c2:
            page_fam_options = ["All Families (33)"] + sorted(df_fc["family"].unique().tolist())
            p_fam = st.selectbox("Product Family Selection", page_fam_options, index=0 if selected_families is None else page_fam_options.index(selected_families[0]))
            page_fams = None if "All" in str(p_fam) else [p_fam]

        with fc_c3:
            p_model = st.selectbox("Primary Model View", ["Naive Baseline", "LightGBM (Recursive)", "Seasonal Naive (Lag 52)", "All Models Combined"], index=0)

        # Slice forecast data for this view
        df_fc_page_view = filter_forecast_data(df_fc, selected_stores=page_stores, selected_families=page_fams)
        df_hist_page_view = filter_historical_sales(df_hist, selected_stores=page_stores, selected_families=page_fams, exclude_partial_weeks=True)

        # Multi-model Comparison Trajectory (using clean historical complete tail)
        active_chart_models = ["lightgbm_forecast", "naive_forecast", "seasonal_naive_forecast"] if p_model == "All Models Combined" else [model_col_map.get(p_model, "naive_forecast")]
        fig_fc = create_forecast_trajectory_chart(
            df_fc_page_view,
            df_hist_tail=df_hist_page_view[df_hist_page_view["week_start_date"] >= "2017-05-01"],
            active_models=active_chart_models,
        )
        st.plotly_chart(fig_fc, width="stretch")

        # Complete 13-Week Forecast Table
        st.markdown("### 📋 Complete 13-Week Forecast Planning Table")
        
        # Summary aggregation by week
        weekly_fc_table = (
            df_fc_page_view.groupby(["forecast_horizon_week", "week_start_date", "week_end_date"])[
                ["lightgbm_forecast", "naive_forecast", "seasonal_naive_forecast"]
            ]
            .sum()
            .reset_index()
            .sort_values("forecast_horizon_week")
        )

        weekly_fc_table["forecast_horizon_week"] = "Week " + weekly_fc_table["forecast_horizon_week"].astype(str)
        weekly_fc_table["week_start_date"] = weekly_fc_table["week_start_date"].dt.strftime("%Y-%m-%d")
        weekly_fc_table["week_end_date"] = weekly_fc_table["week_end_date"].dt.strftime("%Y-%m-%d")
        weekly_fc_table["days_count"] = 7
        weekly_fc_table["data_type"] = "Forecast"

        weekly_fc_table = weekly_fc_table.rename(
            columns={
                "forecast_horizon_week": "Horizon Week",
                "week_start_date": "Week Start (Mon)",
                "week_end_date": "Week End (Sun)",
                "days_count": "Forecast Days",
                "lightgbm_forecast": "LightGBM Forecast (Units)",
                "naive_forecast": "Naive Forecast (Units)",
                "seasonal_naive_forecast": "Seasonal Naive (Units)",
                "data_type": "Data Type",
            }
        )

        st.dataframe(
            weekly_fc_table[[
                "Horizon Week", "Week Start (Mon)", "Week End (Sun)", "Forecast Days",
                "Naive Forecast (Units)", "LightGBM Forecast (Units)", "Seasonal Naive (Units)", "Data Type"
            ]].style.format(
                {
                    "Naive Forecast (Units)": "{:,.0f}",
                    "LightGBM Forecast (Units)": "{:,.0f}",
                    "Seasonal Naive (Units)": "{:,.0f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        # Download Forecast CSV
        st.markdown("### 📥 Download 13-Week Forecast Dataset")
        csv_fc = df_fc_page_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Filtered 13-Week Forecast (CSV)",
            data=csv_fc,
            file_name=f"freshbasket_forecast_13_weeks_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    # -------------------------------------------------------------------------
    # PAGE 4: PRODUCT FAMILY INSIGHTS
    # -------------------------------------------------------------------------
    elif page == "🧺 Product Family Insights":
        st.title("🧺 Product Family Demand & Accuracy Insights")
        st.markdown(
            "Detailed accuracy breakdown across all 33 FMCG product families. "
            "Identify categories where LightGBM machine learning outperforms the baseline persistence models."
        )

        # Grouped WAPE comparison across top families
        fig_fam_wape = create_family_wape_comparison_chart(df_fam_perf, top_n=15)
        st.plotly_chart(fig_fam_wape, width="stretch")

        # Category Deep-Dive Selector
        st.markdown("### 🔍 Category Deep-Dive")
        col_sel, _ = st.columns([1, 2])
        with col_sel:
            chosen_family = st.selectbox(
                "Select Category to Inspect",
                sorted(df_fam_perf["family"].unique()),
                index=0,
            )

        fam_subset = df_fam_perf[df_fam_perf["family"] == chosen_family].copy()
        
        # Display performance table for chosen category
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.markdown(f"**Model Performance for {chosen_family}**")
            disp_fam_table = fam_subset[["model", "wape", "mae", "rmse", "rmsle", "actual_sales_volume"]].copy()
            disp_fam_table = disp_fam_table.rename(
                columns={
                    "model": "Model",
                    "wape": "WAPE (%)",
                    "mae": "MAE (Units)",
                    "rmse": "RMSE",
                    "rmsle": "RMSLE",
                    "actual_sales_volume": "Holdout Sales Volume",
                }
            )
            st.dataframe(
                disp_fam_table.style.format(
                    {
                        "WAPE (%)": "{:.2f}%",
                        "MAE (Units)": "{:,.1f}",
                        "RMSE": "{:,.1f}",
                        "RMSLE": "{:.4f}",
                        "Holdout Sales Volume": "{:,.0f}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )


        with col_t2:
            # Highlight LightGBM vs Naive advantage
            naive_wape = fam_subset.loc[fam_subset["model"] == "Naive", "wape"].values
            lgb_wape = fam_subset.loc[fam_subset["model"] == "LightGBM", "wape"].values

            if len(naive_wape) > 0 and len(lgb_wape) > 0:
                diff = naive_wape[0] - lgb_wape[0]
                if diff > 0:
                    st.success(
                        f"✅ **LightGBM Advantage:** LightGBM achieves **{lgb_wape[0]:.2f}% WAPE**, "
                        f"beating Naive (**{naive_wape[0]:.2f}%**) by **+{diff:.2f}% percentage points** on {chosen_family}."
                    )
                else:
                    st.info(
                        f"ℹ️ **Baseline Lead:** Naive baseline holds **{naive_wape[0]:.2f}% WAPE**, "
                        f"leading LightGBM (**{lgb_wape[0]:.2f}%**) by **{abs(diff):.2f}% percentage points** on {chosen_family}."
                    )

            st.markdown(
                """
                > **FMCG Insight:** Categories with high promotional elasticity or seasonal variation 
                > (e.g. *Personal Care, Cleaning, Deli, Bread/Bakery*) benefit strongly from LightGBM's 
                > nonlinear autoregressive tree splits.
                """
            )

    # -------------------------------------------------------------------------
    # PAGE 5: STORE PERFORMANCE
    # -------------------------------------------------------------------------
    elif page == "🏬 Store Performance":
        st.title("🏬 Store Network Demand & Operational Performance")
        st.markdown(
            "Evaluation of store-level forecast precision across 54 stores, categorized by "
            "city, state, store type (A-E), and operational clusters (1-17)."
        )

        s_col1, s_col2 = st.columns([1, 1])
        with s_col1:
            fig_store = create_store_ranking_chart(df_hist_filtered, top_n=12)
            st.plotly_chart(fig_store, width="stretch")

        with s_col2:
            # Store Type Aggregation
            st.markdown("**Sales Distribution by Store Format Type**")
            type_agg = (
                df_hist_filtered.groupby("type")["weekly_sales"]
                .sum()
                .reset_index()
                .sort_values("weekly_sales", ascending=False)
            )
            fig_type = px.pie(
                type_agg,
                names="type",
                values="weekly_sales",
                color_discrete_sequence=px.colors.qualitative.Prism,
                hole=0.4,
            )
            fig_type.update_layout(margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_type, width="stretch")

        # Store Details Table
        st.markdown("### 📊 Store Forecast Precision Table")
        disp_store_perf = (
            df_store_perf[df_store_perf["model"] == ("Naive" if "Naive" in selected_model_label else "LightGBM")]
            .sort_values("actual_sales_volume", ascending=False)
            .copy()
        )
        disp_store_perf = disp_store_perf.rename(
            columns={
                "store_nbr": "Store #",
                "city": "City",
                "state": "State",
                "type": "Type",
                "cluster": "Cluster",
                "actual_sales_volume": "Holdout Volume",
                "wape": "WAPE (%)",
                "mae": "MAE",
                "rmse": "RMSE",
                "rmsle": "RMSLE",
            }
        )

        st.dataframe(
            disp_store_perf[["Store #", "City", "State", "Type", "Cluster", "Holdout Volume", "WAPE (%)", "MAE", "RMSE"]].style.format(
                {
                    "Holdout Volume": "{:,.0f}",
                    "WAPE (%)": "{:.2f}%",
                    "MAE": "{:,.1f}",
                    "RMSE": "{:,.1f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    # -------------------------------------------------------------------------
    # PAGE 6: MODEL EVALUATION & AUDIT
    # -------------------------------------------------------------------------
    elif page == "🔬 Model Evaluation & Audit":
        st.title("🔬 Model Evaluation Audit & Horizon Error Decay")
        st.markdown(
            "Rigorous, audit-verified backtest results across the 13-week evaluation holdout "
            "(`2017-05-15` to `2017-08-13`). Denominators and metrics are calculated without lookahead bias."
        )

        # Honest Global Leaderboard Callout
        st.markdown(
            """
            <div style="background-color: #EFF6FF; border: 1px solid #BFDBFE; border-left: 5px solid #2563EB; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px;">
                <span style="font-weight: 800; color: #1E40AF; font-size: 1rem;">🏆 Verified Backtest Benchmark Findings:</span>
                <p style="color: #1E3A8A; font-size: 0.9rem; margin: 6px 0 0 0; line-height: 1.5;">
                    • <b>Global WAPE Winner:</b> The <b>Naive Baseline holds the best overall score at 11.61% WAPE</b>.<br>
                    • <b>Machine Learning Performance:</b> <b>LightGBM achieves 12.27% WAPE</b> (+0.66% gap to Naive, outperforming Seasonal Naive by 3.90%).<br>
                    • <b>Seasonal Baseline:</b> <b>Seasonal Naive trails at 16.17% WAPE</b> due to ongoing store network expansion.<br>
                    • <i>Empirical Reality:</i> These metrics describe this specific 13-week historical holdout period and demonstrate why baseline models are critical benchmarks in retail supply chain forecasting. LightGBM is not declared the overall winner because Naive empirically holds the lower global WAPE.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Global Metric Table (All 4 metrics: MAE, RMSE, WAPE, RMSLE)
        st.markdown("### 📊 Global 13-Week Performance Matrix (All Evaluated Models)")
        global_rows = df_model_perf[df_model_perf["granularity"] == "global"].copy()
        
        # Add rank column based on WAPE
        global_rows = global_rows.sort_values("wape")
        global_rows["Rank"] = [1, 2, 3]

        global_rows = global_rows.rename(
            columns={
                "model": "Model",
                "observations": "Evaluated Series-Weeks",
                "mae": "MAE (Units)",
                "rmse": "RMSE (Units)",
                "wape": "WAPE (%)",
                "rmsle": "RMSLE",
            }
        )
        st.dataframe(
            global_rows[["Rank", "Model", "Evaluated Series-Weeks", "MAE (Units)", "RMSE (Units)", "WAPE (%)", "RMSLE"]].style.format(
                {
                    "Rank": "#{:d}",
                    "Evaluated Series-Weeks": "{:,}",
                    "MAE (Units)": "{:,.2f}",
                    "RMSE (Units)": "{:,.2f}",
                    "WAPE (%)": "{:.2f}%",
                    "RMSLE": "{:.4f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        # Horizon Decay Chart
        fig_decay = create_horizon_decay_chart(df_model_perf)
        st.plotly_chart(fig_decay, width="stretch")


        # Technical Deep-Dive: The RMSLE Discrepancy
        st.markdown("---")
        st.markdown("### 🔬 Technical Audit: Why LightGBM's RMSLE is Higher than Naive's")

        t_col1, t_col2 = st.columns([1, 1])
        with t_col1:
            st.markdown(
                """
                **The Mathematical Mechanism:**
                $$\\text{RMSLE} = \\sqrt{\\frac{1}{N} \\sum_{i=1}^N \\left(\\ln(\\hat{y}_i + 1) - \\ln(y_i + 1)\\right)^2}$$
                
                - **Intermittent Zero Actuals ($y_i = 0$):** In the 23,166 holdout observations, **1,858 instances have 0 sales** (due to store renovations or dormant product codes).
                - **Naive Behavior:** For previously dormant series, Naive predicts $\\hat{y} = 0$, giving $\\ln(1) - \\ln(1) = 0$ (zero penalty).
                - **LightGBM Behavior:** LightGBM outputs small positive continuous expectations (e.g. $\\hat{y} \\approx 30$ units). On actual 0, this yields:
                  $$(\\ln(31) - \\ln(1))^2 = (3.43)^2 \\approx 11.8 \\text{ penalty points!}$$
                """
            )
        with t_col2:
            st.markdown(
                """
                **Core Commercial Volume ($y_i > 1,000$ units):**
                
                - When evaluating genuine high-volume FMCG sales (**8,076 observations, >85% of company revenue**):
                  - **LightGBM RMSLE:** **0.2318** ✅
                  - **Naive RMSLE:** **0.2583** ❌
                - On commercial store volume, **LightGBM is 10.3% more accurate in RMSLE than Naive**.
                - **Conclusion:** The higher global RMSLE is driven by log-penalty sensitivity on intermittent zero-sales tail observations, not poor forecasting on primary store volume.
                """
            )

    # -------------------------------------------------------------------------
    # PAGE 7: INVENTORY PLANNING & STOCKOUT RISK ANALYSIS
    # -------------------------------------------------------------------------
    elif page == "📦 Inventory Planning":
        st.title("📦 Inventory Planning & Stockout Risk Analysis")
        st.markdown(
            "Deterministic replenishment heuristics integrating multi-step machine learning "
            "demand forecasts with supplier lead times and safety stock buffers."
        )

        # Upload / Data Source Controls
        with st.expander("📂 Inventory Data Source & Custom CSV Upload", expanded=False):
            col_up1, col_up2 = st.columns([2, 1])
            with col_up1:
                uploaded_file = st.file_uploader(
                    "Upload Actual Retail Inventory CSV",
                    type=["csv"],
                    help="Upload a CSV with store_nbr, family, current_stock, safety_stock, and optional supplier_lead_time_days.",
                )
                if uploaded_file is not None:
                    try:
                        upload_df = pd.read_csv(uploaded_file)
                        is_valid, validation_errors, validated_df = validate_inventory_dataframe(upload_df)
                        if is_valid:
                            st.session_state.custom_inventory_df = validated_df
                            st.session_state.data_source_mode = "CUSTOM"
                            st.success(f"✅ Successfully validated and loaded {len(validated_df):,} inventory records!")
                        else:
                            st.error(f"❌ Upload validation failed ({len(validation_errors)} issues):")
                            for err in validation_errors[:5]:
                                st.write(f"- {err}")
                            if len(validation_errors) > 5:
                                st.write(f"- ...and {len(validation_errors) - 5} more errors.")
                    except Exception as e:
                        st.error(f"Failed to read uploaded file: {e}")

            with col_up2:
                st.markdown("**Active Data Mode:**")
                if st.session_state.data_source_mode == "CUSTOM" and st.session_state.custom_inventory_df is not None:
                    st.success("🔵 **CUSTOM DATA** (User Upload)")
                    if st.button("🔄 Reset to Demo Balances"):
                        st.session_state.custom_inventory_df = None
                        st.session_state.data_source_mode = "DEMO"
                        st.rerun()
                else:
                    st.info("🟢 **DEMO DATA** (Synthetic Baseline)")
                
                st.caption(
                    "Expected Columns: `store_nbr`, `family`, `current_stock`, `safety_stock`, "
                    "`supplier_lead_time_days` (optional), `last_updated` (optional)."
                )

        # Select working inventory dataset
        active_inventory = (
            st.session_state.custom_inventory_df
            if (st.session_state.data_source_mode == "CUSTOM" and st.session_state.custom_inventory_df is not None)
            else df_inv
        )

        # Transparency Notice Banner
        if st.session_state.data_source_mode == "DEMO":
            st.warning(
                "⚠️ **DEMO INVENTORY BALANCES:** Currently evaluating simulated stock and lead times for development. "
                "Calculations assume Inventory Position = Current Stock (excluding committed reservations "
                "and in-transit purchase orders). Do not present demo balances as real retailer inventory."
            )
        else:
            st.info(
                "ℹ️ **CUSTOM INVENTORY BALANCES:** Evaluating verified user inventory. "
                "Inventory Position = Current Stock on Hand."
            )

        # Compute replenishment plan using selected forecast model
        df_plan_all = compute_replenishment_plan(
            active_inventory,
            df_fc,
            model_col=selected_model_col,
            supplier_df=df_suppliers,
        )

        # Check for horizon warnings
        horizon_warnings = df_plan_all[df_plan_all["horizon_warning"].notnull()]
        if not horizon_warnings.empty:
            st.warning(
                f"⚠️ **Lead-Time Exceeds Horizon Warning:** {len(horizon_warnings)} items have supplier "
                f"lead times longer than 91 days (13 weeks). Their expected lead-time demand is capped "
                f"at the 13-week total rather than extrapolating speculative demand."
            )

        # In-Page Risk Level Filter
        filter_col1, filter_col2 = st.columns([1, 1])
        with filter_col1:
            risk_filter = st.selectbox(
                "Filter by Stockout Risk Level",
                ["All Risk Levels", "HIGH (Immediate Stockout Risk)", "MEDIUM (Safety Stock Breached)", "LOW (Sufficient Stock)"],
                index=0,
            )
        with filter_col2:
            st.markdown(
                f"<div style='padding-top: 28px; font-size: 0.85rem; color: #64748B;'>"
                f"Forecasting Engine: <b>{selected_model_label}</b> | Scope: "
                f"{selected_store_raw} & {selected_family_raw}</div>",
                unsafe_allow_html=True,
            )

        # Filter the plan by sidebar store/family and in-page risk filter
        df_plan = df_plan_all.copy()
        if selected_stores is not None:
            df_plan = df_plan[df_plan["store_nbr"].isin(selected_stores)]
        if selected_families is not None:
            df_plan = df_plan[df_plan["family"].isin(selected_families)]
        if "HIGH" in risk_filter:
            df_plan = df_plan[df_plan["stockout_risk"] == "HIGH"]
        elif "MEDIUM" in risk_filter:
            df_plan = df_plan[df_plan["stockout_risk"] == "MEDIUM"]
        elif "LOW" in risk_filter:
            df_plan = df_plan[df_plan["stockout_risk"] == "LOW"]

        # KPI Summary Metrics
        st.markdown("### 📊 Inventory & Risk Scorecard")
        k1, k2, k3, k4, k5, k6 = st.columns(6)
        
        total_items = len(df_plan)
        high_risk_cnt = int((df_plan["stockout_risk"] == "HIGH").sum()) if not df_plan.empty else 0
        med_risk_cnt = int((df_plan["stockout_risk"] == "MEDIUM").sum()) if not df_plan.empty else 0
        low_risk_cnt = int((df_plan["stockout_risk"] == "LOW").sum()) if not df_plan.empty else 0
        tot_lt_demand = float(df_plan["expected_lt_demand"].sum()) if not df_plan.empty else 0.0
        tot_replenish = float(df_plan["suggested_replenishment_qty"].sum()) if not df_plan.empty else 0.0

        with k1:
            st.metric(
                label="Monitored Items",
                value=f"{total_items:,}",
                help="Total active store-family combinations evaluated under the selected filters.",
                border=True,
            )
            st.caption("Active store-family pairs")
        with k2:
            st.metric(
                label="High-Risk Items",
                value=f"{high_risk_cnt:,}",
                help="Critical: Stock on hand is below expected lead-time demand (stockout deficit expected before delivery).",
                border=True,
            )
            st.caption("Stock < lead-time demand")
        with k3:
            st.metric(
                label="Medium-Risk Items",
                value=f"{med_risk_cnt:,}",
                help="Warning: Stock covers lead-time demand but has penetrated safety buffer (stock < reorder point).",
                border=True,
            )
            st.caption("Stock < reorder point")
        with k4:
            st.metric(
                label="Low-Risk Items",
                value=f"{low_risk_cnt:,}",
                help="Healthy: Stock is at or above the reorder point with sufficient safety buffer capacity.",
                border=True,
            )
            st.caption("Stock ≥ reorder point")
        with k5:
            st.metric(
                label="Total Lead-Time Demand",
                value=f"{tot_lt_demand:,.0f}",
                help="Aggregate customer demand forecasted across supplier lead times for the selected scope.",
                border=True,
            )
            st.caption("Forecasted lead-time units")
        with k6:
            st.metric(
                label="Suggested Replenishment Units",
                value=f"{tot_replenish:,.0f}",
                help="Total suggested replenishment units required: ceil(max(0, Reorder Point - Inventory Position)).",
                border=True,
            )
            st.caption("Net reorder units to ROP")

        st.markdown("---")

        # Visualizations (Plotly)
        st.markdown("### 📈 Visual Risk & Replenishment Diagnostics")
        vcol1, vcol2 = st.columns([1, 1])
        with vcol1:
            fig_risk_pie = create_stockout_risk_pie(df_plan)
            st.plotly_chart(fig_risk_pie, width="stretch")
        with vcol2:
            fig_highest_risk = create_highest_risk_items_bar(df_plan, top_n=10)
            st.plotly_chart(fig_highest_risk, width="stretch")

        fig_scatter = create_demand_vs_stock_scatter(df_plan)
        st.plotly_chart(fig_scatter, width="stretch")

        # Rule-Based Business Summary
        st.markdown("---")
        st.markdown("### 💡 Executive Replenishment Insights & Action Rules")
        insights = generate_inventory_business_summary(df_plan)
        for ins in insights:
            st.markdown(f"- {ins}")

        # Interactive Table & Download
        st.markdown("---")
        st.markdown("### 📋 Interactive Replenishment Recommendation Register")

        if df_plan.empty:
            st.info("No items match the selected filter criteria.")
        else:
            display_cols = [
                "store_nbr",
                "family",
                "stockout_risk",
                "current_stock",
                "expected_lt_demand",
                "safety_stock",
                "reorder_point",
                "suggested_replenishment_qty",
                "supplier_lead_time_days",
                "supplier_name",
                "explanation",
            ]
            
            table_df = df_plan[display_cols].rename(
                columns={
                    "store_nbr": "Store",
                    "family": "Product Family",
                    "stockout_risk": "Stockout Risk",
                    "current_stock": "Current Stock",
                    "expected_lt_demand": "Expected LT Demand",
                    "safety_stock": "Safety Stock",
                    "reorder_point": "Reorder Point",
                    "suggested_replenishment_qty": "Suggested Order (Units)",
                    "supplier_lead_time_days": "Lead Time (Days)",
                    "supplier_name": "Supplier",
                    "explanation": "Calculation Rationale",
                }
            )

            st.dataframe(
                table_df.style.format(
                    {
                        "Current Stock": "{:,.0f}",
                        "Expected LT Demand": "{:,.1f}",
                        "Safety Stock": "{:,.0f}",
                        "Reorder Point": "{:,.1f}",
                        "Suggested Order (Units)": "{:,.0f}",
                        "Lead Time (Days)": "{:d}",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

            # CSV Download
            csv_export = df_plan.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Replenishment Recommendations (CSV)",
                data=csv_export,
                file_name=f"freshbasket_replenishment_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

    # -------------------------------------------------------------------------
    # PAGE 8: INVENTORY ALERT CENTER & BUSINESS REPORTS (PHASE 9)
    # -------------------------------------------------------------------------
    elif page == "🚨 Inventory Alerts":
        st.title("🚨 Inventory Alert Center & Business Reporting")
        st.markdown(
            "Priority stockout risk triage, store-level exposure rollups, executive briefings, "
            "and email staging powered by multi-step demand forecasting."
        )

        # Working Inventory selection
        active_inventory = (
            st.session_state.custom_inventory_df
            if (st.session_state.data_source_mode == "CUSTOM" and st.session_state.custom_inventory_df is not None)
            else df_inv
        )

        # Active Data Mode Notice Banner
        if st.session_state.data_source_mode == "DEMO":
            st.warning(
                "⚠️ **DEMO INVENTORY BALANCES:** Alerts are currently evaluating synthetic baseline inventory. "
                "Calculations assume Inventory Position = Current Stock (excluding committed reservations "
                "and in-transit purchase orders). Do not use demo balances for production reorders."
            )
        else:
            st.info(
                "ℹ️ **CUSTOM INVENTORY BALANCES:** Evaluating verified user inventory. "
                "Inventory Position = Current Stock on Hand."
            )

        # Compute complete baseline replenishment plan using selected forecast model
        df_plan_all = compute_replenishment_plan(
            active_inventory,
            df_fc,
            model_col=selected_model_col,
            supplier_df=df_suppliers,
        )

        # In-Page Risk Level Filter
        filter_col1, filter_col2 = st.columns([1, 1])
        with filter_col1:
            alert_risk_filter = st.selectbox(
                "Filter Alerts by Risk Level",
                [
                    "All Risk Levels (HIGH + MEDIUM + LOW)",
                    "HIGH Risk Only (Critical Stockouts)",
                    "MEDIUM Risk Only (Safety Stock Breached)",
                    "LOW Risk Only (Sufficient Stock)",
                    "UNKNOWN (Invalid / Missing Data)",
                ],
                index=0,
            )
        with filter_col2:
            st.markdown(
                f"<div style='padding-top: 28px; font-size: 0.85rem; color: #64748B;'>"
                f"Forecasting Engine: <b>{selected_model_label}</b> | Scope: "
                f"{selected_store_raw} & {selected_family_raw}</div>",
                unsafe_allow_html=True,
            )

        # Apply sidebar store/family filters
        df_plan = df_plan_all.copy()
        if selected_stores is not None:
            df_plan = df_plan[df_plan["store_nbr"].isin(selected_stores)]
        if selected_families is not None:
            df_plan = df_plan[df_plan["family"].isin(selected_families)]

        # Calculate counts BEFORE risk filter for complete scope scorecard
        high_cnt = int((df_plan["stockout_risk"] == "HIGH").sum())
        med_cnt = int((df_plan["stockout_risk"] == "MEDIUM").sum())
        low_cnt = int((df_plan["stockout_risk"] == "LOW").sum())
        unk_cnt = int((df_plan["stockout_risk"] == "UNKNOWN").sum())

        # Determine affected stores & families with HIGH or MEDIUM risk in current scope
        affected_items = df_plan[df_plan["stockout_risk"].isin(["HIGH", "MEDIUM"])]
        stores_affected = int(affected_items["store_nbr"].nunique()) if not affected_items.empty else 0
        families_affected = int(affected_items["family"].nunique()) if not affected_items.empty else 0

        # Apply risk filter to working plan
        if "HIGH" in alert_risk_filter and "MEDIUM" not in alert_risk_filter:
            df_plan = df_plan[df_plan["stockout_risk"] == "HIGH"]
        elif "MEDIUM" in alert_risk_filter and "HIGH" not in alert_risk_filter:
            df_plan = df_plan[df_plan["stockout_risk"] == "MEDIUM"]
        elif "LOW" in alert_risk_filter and "HIGH" not in alert_risk_filter:
            df_plan = df_plan[df_plan["stockout_risk"] == "LOW"]
        elif "UNKNOWN" in alert_risk_filter:
            df_plan = df_plan[df_plan["stockout_risk"] == "UNKNOWN"]

        # Prioritize alerts (HIGH first, then MEDIUM, sorted by replenishment quantity descending)
        df_prioritized = prioritize_alerts(df_plan)
        tot_replenish = float(df_prioritized["suggested_replenishment_qty"].sum()) if not df_prioritized.empty else 0.0

        # ---------------------------------------------------------------------
        # KPI Summary Cards (Native Streamlit components)
        # ---------------------------------------------------------------------
        st.markdown("### 📊 Inventory Alert Scorecard")
        a1, a2, a3, a4, a5, a6 = st.columns(6)
        with a1:
            st.metric(
                label="High-Risk Items",
                value=f"{high_cnt:,}",
                help="Stock < expected lead-time demand: Immediate stockout deficit projected.",
                border=True,
            )
            st.caption("Immediate stockout danger")
        with a2:
            st.metric(
                label="Medium-Risk Items",
                value=f"{med_cnt:,}",
                help="Stock covers lead-time demand but has penetrated safety buffer (stock < ROP).",
                border=True,
            )
            st.caption("Safety buffer breached")
        with a3:
            st.metric(
                label="Low-Risk Items",
                value=f"{low_cnt:,}",
                help="Stock is at or above reorder point with sufficient safety buffer.",
                border=True,
            )
            st.caption("Healthy inventory buffer")
        with a4:
            st.metric(
                label="Suggested Replenishment",
                value=f"{tot_replenish:,.0f}",
                help="Total suggested replenishment units required: ceil(max(0, ROP - Inventory Position)).",
                border=True,
            )
            st.caption("Net reorder units")
        with a5:
            st.metric(
                label="Stores Affected",
                value=f"{stores_affected:,}",
                help="Number of retail locations with at least one HIGH or MEDIUM risk SKU.",
                border=True,
            )
            st.caption("Branches needing reorders")
        with a6:
            st.metric(
                label="Product Families Affected",
                value=f"{families_affected:,}",
                help="Number of FMCG categories with at least one HIGH or MEDIUM risk SKU.",
                border=True,
            )
            st.caption("Categories needing reorders")

        st.markdown("---")

        # ---------------------------------------------------------------------
        # Downloadable Alert Reports Buttons (FEATURE 3)
        # ---------------------------------------------------------------------
        st.markdown("### 📥 Instant Alert Reports & Exports")
        dcol1, dcol2, dcol3 = st.columns(3)

        date_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # High-risk items CSV
        high_risk_export = df_plan_all[df_plan_all["stockout_risk"] == "HIGH"]
        if selected_stores is not None:
            high_risk_export = high_risk_export[high_risk_export["store_nbr"].isin(selected_stores)]
        if selected_families is not None:
            high_risk_export = high_risk_export[high_risk_export["family"].isin(selected_families)]
        csv_high = high_risk_export.to_csv(index=False).encode("utf-8")

        # Medium-risk items CSV
        med_risk_export = df_plan_all[df_plan_all["stockout_risk"] == "MEDIUM"]
        if selected_stores is not None:
            med_risk_export = med_risk_export[med_risk_export["store_nbr"].isin(selected_stores)]
        if selected_families is not None:
            med_risk_export = med_risk_export[med_risk_export["family"].isin(selected_families)]
        csv_med = med_risk_export.to_csv(index=False).encode("utf-8")

        # All items CSV
        csv_all = df_prioritized.to_csv(index=False).encode("utf-8")

        with dcol1:
            st.download_button(
                label=f"🚨 Download High-Risk Items ({len(high_risk_export):,} SKUs)",
                data=csv_high,
                file_name=f"freshbasket_high_risk_alerts_{date_tag}.csv",
                mime="text/csv",
                width="stretch",
            )
        with dcol2:
            st.download_button(
                label=f"⚠️ Download Medium-Risk Items ({len(med_risk_export):,} SKUs)",
                data=csv_med,
                file_name=f"freshbasket_medium_risk_alerts_{date_tag}.csv",
                mime="text/csv",
                width="stretch",
            )
        with dcol3:
            st.download_button(
                label=f"📋 Download All Filtered ({len(df_prioritized):,} SKUs)",
                data=csv_all,
                file_name=f"freshbasket_all_replenishment_alerts_{date_tag}.csv",
                mime="text/csv",
                width="stretch",
            )

        st.markdown("---")

        # ---------------------------------------------------------------------
        # Tabbed Interactive Views
        # ---------------------------------------------------------------------
        tab_alerts, tab_stores, tab_exec, tab_email, tab_history = st.tabs([
            "📋 Prioritized Alert Register",
            "🏬 Store-Wise Exposure Report",
            "📄 Executive Briefing Report",
            "✉️ Email Preparation Draft",
            "📜 Alert History & Audit Trail",
        ])

        with tab_alerts:
            st.markdown("#### 📋 Prioritized Stockout Risk Register")
            st.caption("Sorted prioritizing high-risk items and largest suggested replenishment deficits.")
            
            if df_prioritized.empty:
                st.info("No items match the selected filter criteria.")
            else:
                display_cols = [
                    "store_nbr", "family", "stockout_risk", "current_stock",
                    "expected_lt_demand", "safety_stock", "reorder_point",
                    "suggested_replenishment_qty", "supplier_lead_time_days",
                    "supplier_name", "explanation",
                ]
                table_disp = df_prioritized[display_cols].rename(
                    columns={
                        "store_nbr": "Store",
                        "family": "Product Family",
                        "stockout_risk": "Risk Level",
                        "current_stock": "Current Stock",
                        "expected_lt_demand": "Expected LT Demand",
                        "safety_stock": "Safety Stock",
                        "reorder_point": "Reorder Point",
                        "suggested_replenishment_qty": "Suggested Order (Units)",
                        "supplier_lead_time_days": "Lead Time (Days)",
                        "supplier_name": "Supplier",
                        "explanation": "Triage Explanation",
                    }
                )
                st.dataframe(
                    table_disp.style.format({
                        "Current Stock": "{:,.0f}",
                        "Expected LT Demand": "{:,.1f}",
                        "Safety Stock": "{:,.0f}",
                        "Reorder Point": "{:,.1f}",
                        "Suggested Order (Units)": "{:,.0f}",
                        "Lead Time (Days)": "{:d}",
                    }),
                    width="stretch",
                    hide_index=True,
                )

        with tab_stores:
            st.markdown("#### 🏬 Store-Level Stockout Exposure Rollup")
            st.caption("Aggregated risk metrics per retail branch (no double-counting of store-family records).")
            
            store_source_df = df_plan_all.copy()
            if selected_stores is not None:
                store_source_df = store_source_df[store_source_df["store_nbr"].isin(selected_stores)]
            if selected_families is not None:
                store_source_df = store_source_df[store_source_df["family"].isin(selected_families)]

            store_summary_df = generate_store_wise_summary(store_source_df)

            if store_summary_df.empty:
                st.info("No store data matches the current filters.")
            else:
                store_disp = store_summary_df.rename(
                    columns={
                        "store_nbr": "Store Number",
                        "urgency_level": "Store Urgency",
                        "high_risk_count": "High-Risk Items",
                        "medium_risk_count": "Medium-Risk Items",
                        "low_risk_count": "Low-Risk Items",
                        "total_items": "Monitored SKUs",
                        "total_expected_lt_demand": "Total LT Demand (Units)",
                        "total_suggested_replenishment_qty": "Total Suggested Order (Units)",
                    }
                )
                st.dataframe(
                    store_disp.style.format({
                        "High-Risk Items": "{:,}",
                        "Medium-Risk Items": "{:,}",
                        "Low-Risk Items": "{:,}",
                        "Monitored SKUs": "{:,}",
                        "Total LT Demand (Units)": "{:,.1f}",
                        "Total Suggested Order (Units)": "{:,.0f}",
                    }),
                    width="stretch",
                    hide_index=True,
                )

                csv_store = store_summary_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Store-Wise Summary (CSV)",
                    data=csv_store,
                    file_name=f"freshbasket_store_summary_{date_tag}.csv",
                    mime="text/csv",
                )

        with tab_exec:
            st.markdown("#### 📄 Executive Business Briefing Report")
            exec_briefing_md = generate_executive_business_report(
                df_plan=df_prioritized,
                data_source_mode=st.session_state.data_source_mode,
                forecast_model_label=selected_model_label,
                selected_store_label=selected_store_raw,
                selected_family_label=selected_family_raw,
            )
            st.markdown(exec_briefing_md)

            st.download_button(
                label="📥 Download Executive Briefing (Markdown / TXT)",
                data=exec_briefing_md.encode("utf-8"),
                file_name=f"freshbasket_executive_briefing_{date_tag}.md",
                mime="text/markdown",
            )

        with tab_email:
            st.markdown("#### ✉️ Email Alert Briefing Preparation")
            st.info(
                "ℹ️ **Operational Note:** Prepare an email draft to circulate among replenishment teams. "
                "FreshBasket AI does not automatically dispatch emails and requires no credentials or API keys."
            )
            email_draft = generate_email_alert_draft(
                df_plan=df_prioritized,
                data_source_mode=st.session_state.data_source_mode,
                forecast_model_label=selected_model_label,
                selected_store_label=selected_store_raw,
                selected_family_label=selected_family_raw,
            )
            st.text_area("Email Draft (Copy & Paste)", value=email_draft, height=350)
            st.download_button(
                label="📥 Download Email Text Draft",
                data=email_draft.encode("utf-8"),
                file_name=f"freshbasket_email_alert_{date_tag}.txt",
                mime="text/plain",
            )

        with tab_history:
            st.markdown("#### 📜 Persistent Alert History & Audit Trail")
            st.caption(
                "Lightweight snapshot management stored in `data/inventory/alert_history.csv`. "
                "Alerts are evaluated at dashboard launch/refresh, not continuously running as a background daemon."
            )

            hcol1, hcol2 = st.columns([1, 2])
            with hcol1:
                if st.button("📸 Capture Alert Snapshot Now"):
                    success, msg, hist_df = record_alert_snapshot(
                        df_plan=df_prioritized,
                        data_source_mode=st.session_state.data_source_mode,
                        forecast_model_label=selected_model_label,
                    )
                    if success:
                        st.success(msg)
                    else:
                        st.info(msg)

            history_df = load_alert_history()
            if history_df.empty:
                st.info("No alert snapshots recorded yet. Click 'Capture Alert Snapshot Now' to log the current alert state.")
            else:
                st.write(f"**Total Historical Alert Records Logged:** {len(history_df):,} entries across {history_df['timestamp'].nunique():,} snapshots")
                st.dataframe(
                    history_df.tail(200).rename(
                        columns={
                            "timestamp": "Timestamp (UTC)",
                            "store_nbr": "Store",
                            "family": "Product Family",
                            "stockout_risk": "Risk Level",
                            "suggested_replenishment_qty": "Suggested Order",
                            "current_stock": "Current Stock",
                            "expected_lt_demand": "Lead Demand",
                            "reorder_point": "Reorder Point",
                            "data_source": "Data Source",
                            "forecast_model": "Model",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

                csv_hist = history_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Full Alert History (CSV)",
                    data=csv_hist,
                    file_name=f"freshbasket_alert_history_{date_tag}.csv",
                    mime="text/csv",
                )



if __name__ == "__main__":
    main()
