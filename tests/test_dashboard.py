"""FreshBasket AI - Dashboard Unit Tests.

Validates the dashboard data loading, caching helpers, filter functions,
KPI computations, and chart generators.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.dashboard.data_loader import (
    compute_kpis,
    filter_forecast_data,
    filter_historical_sales,
    load_all_dashboard_data,
    verify_dashboard_files_exist,
)
from src.dashboard.charts import (
    create_family_sales_bar_chart,
    create_family_wape_comparison_chart,
    create_forecast_trajectory_chart,
    create_horizon_decay_chart,
    create_model_comparison_bar,
    create_store_ranking_chart,
    create_weekly_historical_sales_chart,
    render_kpi_card,
)


def test_dashboard_files_presence():
    """Confirms all 5 verified dashboard datasets are present."""
    all_ok, missing = verify_dashboard_files_exist()
    assert all_ok, f"Missing dashboard files: {missing}"
    assert len(missing) == 0


def test_dashboard_data_loader():
    """Confirms all datasets load with expected record counts and columns."""
    datasets = load_all_dashboard_data()
    assert "historical" in datasets
    assert "forecast" in datasets
    assert "model_performance" in datasets
    assert "family_performance" in datasets
    assert "store_performance" in datasets

    assert len(datasets["historical"]) == 431244
    assert len(datasets["forecast"]) == 23166
    assert len(datasets["model_performance"]) == 42
    assert len(datasets["family_performance"]) == 99
    assert len(datasets["store_performance"]) == 162


def test_filtering_and_kpis():
    """Confirms filtering functions and KPI computation are robust and safe against empty sets."""
    datasets = load_all_dashboard_data()
    df_hist = datasets["historical"]
    df_fc = datasets["forecast"]

    # Filter by specific store and family
    filtered_hist = filter_historical_sales(
        df_hist,
        selected_stores=[1],
        selected_families=["BEVERAGES"],
        start_date=pd.to_datetime("2016-01-01"),
        end_date=pd.to_datetime("2017-01-01"),
    )
    assert not filtered_hist.empty
    assert (filtered_hist["store_nbr"] == 1).all()
    assert (filtered_hist["family"] == "BEVERAGES").all()

    # KPI Calculation
    kpis = compute_kpis(filtered_hist, df_fc, model_col="naive_forecast")
    assert kpis["total_historical_sales"] > 0
    assert kpis["unique_stores"] == 1
    assert kpis["unique_families"] == 1

    # Empty filter safety
    empty_kpis = compute_kpis(pd.DataFrame(), pd.DataFrame())
    assert empty_kpis["total_historical_sales"] == 0.0
    assert empty_kpis["unique_stores"] == 0


def test_chart_generation():
    """Confirms that all chart helper functions return valid Plotly figures without exceptions."""
    datasets = load_all_dashboard_data()
    df_hist = datasets["historical"]
    df_fc = datasets["forecast"]
    df_model_perf = datasets["model_performance"]
    df_fam_perf = datasets["family_performance"]

    # Sample slice for fast test
    sample_hist = df_hist[df_hist["week_start_date"] >= "2017-06-01"]
    sample_fc = df_fc.head(100)

    fig_hist = create_weekly_historical_sales_chart(sample_hist)
    assert fig_hist is not None

    fig_fc = create_forecast_trajectory_chart(sample_fc, df_hist_tail=sample_hist)
    assert fig_fc is not None

    fig_fam = create_family_sales_bar_chart(sample_hist, top_n=5)
    assert fig_fam is not None

    fig_model = create_model_comparison_bar(df_model_perf)
    assert fig_model is not None

    fig_decay = create_horizon_decay_chart(df_model_perf)
    assert fig_decay is not None

    fig_store = create_store_ranking_chart(sample_hist, top_n=5)
    assert fig_store is not None

    fig_fam_wape = create_family_wape_comparison_chart(df_fam_perf, top_n=5)
    assert fig_fam_wape is not None

    # Card HTML
    html_card = render_kpi_card("Test", "123", "subtitle", "badge", "success")
    assert "123" in html_card
