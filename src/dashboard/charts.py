"""
FreshBasket AI - Dashboard Plotly Visualizations & UI Components
================================================================
Reusable, beautifully styled Plotly charts and KPI cards adhering to
retail FMCG design guidelines (primary emerald green accents #059669, #10B981).
"""

from typing import Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Color Palette
PRIMARY_GREEN = "#059669"
ACCENT_MINT = "#10B981"
LIGHT_BG_GREEN = "#ECFDF5"
SLATE_DARK = "#1E293B"
SLATE_MUTED = "#64748B"
BORDER_GRAY = "#E2E8F0"
BLUE_ACCENT = "#2563EB"
AMBER_ACCENT = "#D97706"
ROSE_ACCENT = "#E11D48"
PURPLE_ACCENT = "#7C3AED"

MODEL_COLORS = {
    "Naive": "#2563EB",                # Solid Corporate Blue
    "Naive Baseline": "#2563EB",
    "LightGBM": "#059669",             # FreshBasket Emerald Green
    "LightGBM (Recursive)": "#059669",
    "Seasonal Naive": "#D97706",       # Amber Orange
    "Seasonal Naive (Lag 52)": "#D97706",
}


def render_kpi_card(
    title: str,
    value: str,
    subtitle: str = "",
    badge_text: str = "",
    badge_type: str = "success",  # "success", "info", "warning", "neutral", "danger"
    color: Optional[str] = None,
    **kwargs,
) -> str:
    """Returns styled HTML for an executive KPI metric card."""
    badge_colors = {
        "success": ("#DEF7EC", "#03543F"),
        "info": ("#E1EFFE", "#1E429F"),
        "warning": ("#FEF08A", "#713F12"),
        "neutral": ("#F3F4F6", "#374151"),
        "danger": ("#FDE8E8", "#9B1C1C"),
    }
    bg_col, text_col = badge_colors.get(badge_type, badge_colors["neutral"])

    badge_html = (
        f'<span style="background-color: {bg_col}; color: {text_col}; font-size: 0.72rem; '
        f'padding: 3px 8px; border-radius: 9999px; font-weight: 600; text-transform: uppercase;">'
        f'{badge_text}</span>'
        if badge_text
        else ""
    )

    val_color = color if color else SLATE_DARK

    return (
        f'<div style="background: #FFFFFF; border: 1px solid {BORDER_GRAY}; border-radius: 12px; '
        f'padding: 16px 18px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); '
        f'display: flex; flex-direction: column; justify-content: space-between; height: 100%;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">'
        f'<span style="font-size: 0.80rem; font-weight: 600; color: {SLATE_MUTED}; text-transform: uppercase; letter-spacing: 0.05em;">{title}</span>'
        f'{badge_html}'
        f'</div>'
        f'<div style="font-size: 1.75rem; font-weight: 700; color: {val_color}; margin-bottom: 4px; line-height: 1.2;">{value}</div>'
        f'<div style="font-size: 0.78rem; color: {SLATE_MUTED}; line-height: 1.3;">{subtitle}</div>'
        f'</div>'
    )


def create_weekly_historical_sales_chart(df: pd.DataFrame) -> go.Figure:
    """Creates an aggregated weekly sales line chart with moving average for complete 7-day weeks."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No sales data for selected filters", showarrow=False, font=dict(size=14))
        return fig

    # Strictly filter for complete 7-day weeks to prevent misleading volume cliffs
    if "is_partial_week" in df.columns:
        df_complete = df[df["is_partial_week"] == False]
    else:
        df_complete = df

    if df_complete.empty:
        fig = go.Figure()
        fig.add_annotation(text="No complete-week sales data available", showarrow=False, font=dict(size=14))
        return fig

    # Aggregate weekly total sales across selected filters
    weekly = (
        df_complete.groupby("week_start_date")["weekly_sales"]
        .sum()
        .reset_index()
        .sort_values("week_start_date")
    )
    weekly["rolling_4w"] = weekly["weekly_sales"].rolling(4, min_periods=1).mean()

    fig = go.Figure()

    # Actual Sales Area / Line
    fig.add_trace(
        go.Scatter(
            x=weekly["week_start_date"],
            y=weekly["weekly_sales"],
            mode="lines",
            name="Weekly Actual Sales (7-Day Complete)",
            line=dict(color=PRIMARY_GREEN, width=2),
            fill="tozeroy",
            fillcolor="rgba(5, 150, 105, 0.08)",
            hovertemplate="<b>Week:</b> %{x|%Y-%m-%d}<br><b>Actual Sales:</b> %{y:,.0f} units (7-day complete)<extra></extra>",
        )
    )

    # 4-Week Moving Average
    fig.add_trace(
        go.Scatter(
            x=weekly["week_start_date"],
            y=weekly["rolling_4w"],
            mode="lines",
            name="4-Week Moving Avg",
            line=dict(color=SLATE_MUTED, width=1.5, dash="dash"),
            hovertemplate="<b>4W Moving Avg:</b> %{y:,.0f} units<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Historical Weekly FMCG Sales Trajectory (Complete 7-Day Weeks)</b>",
        title_font=dict(size=16, color=SLATE_DARK),
        xaxis=dict(
            title="Week Start Date (Monday)",
            showgrid=True,
            gridcolor=BORDER_GRAY,
            linecolor=BORDER_GRAY,
        ),
        yaxis=dict(
            title="Total Weekly Sales (Units)",
            showgrid=True,
            gridcolor=BORDER_GRAY,
            tickformat=",.0f",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=40, r=30, t=60, b=40),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        hovermode="x unified",
    )
    return fig


def create_forecast_trajectory_chart(
    df_forecast: pd.DataFrame,
    df_hist_tail: Optional[pd.DataFrame] = None,
    active_models: Optional[List[str]] = None,
) -> go.Figure:
    """
    Renders 13-week forward forecast trajectory with a distinct historical-vs-forecast boundary.
    Strictly filters out partial weeks (e.g. 2017-08-14 2-day record) to prevent artificial volume cliffs
    or overlapping actual-forecast points.
    """
    if active_models is None:
        active_models = ["lightgbm_forecast", "naive_forecast", "seasonal_naive_forecast"]

    fig = go.Figure()

    last_hist_date = None
    last_hist_sales = None

    # If historical tail provided, filter for complete weeks strictly before forecast start
    if df_hist_tail is not None and not df_hist_tail.empty:
        # Guarantee no partial weeks and strictly before forecast week 1 (2017-08-14)
        hist_valid = df_hist_tail.copy()
        if "is_partial_week" in hist_valid.columns:
            hist_valid = hist_valid[hist_valid["is_partial_week"] == False]
        hist_valid = hist_valid[hist_valid["week_start_date"] < "2017-08-14"]

        if not hist_valid.empty:
            hist_agg = (
                hist_valid.groupby("week_start_date")["weekly_sales"]
                .sum()
                .reset_index()
                .sort_values("week_start_date")
            )
            fig.add_trace(
                go.Scatter(
                    x=hist_agg["week_start_date"],
                    y=hist_agg["weekly_sales"],
                    mode="lines+markers",
                    name="Historical Actuals (7-Day Weeks)",
                    line=dict(color=SLATE_DARK, width=2.5),
                    marker=dict(size=6, color=SLATE_DARK),
                    hovertemplate="<b>Actual Week:</b> %{x|%Y-%m-%d}<br><b>Actual Sales:</b> %{y:,.0f} units (7-day complete)<extra></extra>",
                )
            )
            last_hist_date = hist_agg["week_start_date"].iloc[-1]
            last_hist_sales = hist_agg["weekly_sales"].iloc[-1]

    # Forecast Lines
    model_labels = {
        "lightgbm_forecast": ("LightGBM (Recursive)", MODEL_COLORS["LightGBM"], "solid"),
        "naive_forecast": ("Naive Baseline", MODEL_COLORS["Naive"], "dash"),
        "seasonal_naive_forecast": ("Seasonal Naive (Lag 52)", MODEL_COLORS["Seasonal Naive"], "dot"),
    }

    if not df_forecast.empty:
        fc_agg = (
            df_forecast.groupby(["week_start_date", "forecast_horizon_week"])[
                [m for m in active_models if m in df_forecast.columns]
            ]
            .sum()
            .reset_index()
            .sort_values("week_start_date")
        )

        for col in active_models:
            if col in fc_agg.columns:
                label, color, dash_style = model_labels.get(col, (col, "#6B7280", "solid"))
                
                # Plot forward forecast trajectory
                fig.add_trace(
                    go.Scatter(
                        x=fc_agg["week_start_date"],
                        y=fc_agg[col],
                        mode="lines+markers",
                        name=f"Predicted Forecast: {label}",
                        line=dict(color=color, width=2.5, dash=dash_style),
                        marker=dict(size=6, color=color),
                        hovertemplate=f"<b>Horizon W%{{customdata}}:</b> %{{x|%Y-%m-%d}}<br><b>Forecast ({label}):</b> %{{y:,.0f}} units<extra></extra>",
                        customdata=fc_agg["forecast_horizon_week"],
                    )
                )

                # Add smooth dotted transition line from last historical actual to week 1 forecast
                if last_hist_date is not None and not fc_agg.empty:
                    bridge_x = [last_hist_date, fc_agg["week_start_date"].iloc[0]]
                    bridge_y = [last_hist_sales, fc_agg[col].iloc[0]]
                    fig.add_trace(
                        go.Scatter(
                            x=bridge_x,
                            y=bridge_y,
                            mode="lines",
                            showlegend=False,
                            line=dict(color=color, width=1.5, dash="dot"),
                            hoverinfo="skip",
                        )
                    )

        # Highlight Future Forecast Horizon Region
        fig.add_vrect(
            x0="2017-08-13",
            x1="2017-11-13",
            fillcolor="rgba(16, 185, 129, 0.04)",
            layer="below",
            line_width=0,
            annotation_text="13-Week Future Forecast Horizon (2017-08-14 to 2017-11-12)",
            annotation_position="top left",
            annotation_font=dict(color=PRIMARY_GREEN, size=11, weight="bold"),
        )

        # Add vertical line for Forecast Origin (2017-08-13)
        origin_date = "2017-08-13"
        fig.add_vline(
            x=origin_date,
            line_width=2,
            line_dash="dash",
            line_color=ROSE_ACCENT,
            annotation_text="Forecast Origin (2017-08-13)",
            annotation_position="top right",
            annotation_font=dict(color=ROSE_ACCENT, size=11, weight="bold"),
        )

    fig.update_layout(
        title="<b>13-Week FMCG Demand Forecast vs. Historical Actuals (Zero Boundary Overlap)</b>",
        title_font=dict(size=16, color=SLATE_DARK),
        xaxis=dict(
            title="Week Start Date (Monday)",
            showgrid=True,
            gridcolor=BORDER_GRAY,
            linecolor=BORDER_GRAY,
        ),
        yaxis=dict(
            title="Weekly Sales Volume (Units)",
            showgrid=True,
            gridcolor=BORDER_GRAY,
            tickformat=",.0f",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=40, r=30, t=60, b=40),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        hovermode="x unified",
    )
    return fig



def create_family_sales_bar_chart(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Renders horizontal bar chart of top product families by sales volume."""
    if df.empty:
        return go.Figure()

    fam_agg = (
        df.groupby("family")["weekly_sales"]
        .sum()
        .reset_index()
        .sort_values("weekly_sales", ascending=True)
    )
    if top_n and len(fam_agg) > top_n:
        fam_agg = fam_agg.tail(top_n)

    fig = go.Figure(
        go.Bar(
            x=fam_agg["weekly_sales"],
            y=fam_agg["family"],
            orientation="h",
            marker=dict(color=PRIMARY_GREEN),
            hovertemplate="<b>%{y}:</b> %{x:,.0f} units<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"<b>Top {min(top_n, len(fam_agg))} Product Families by Total Volume</b>",
        title_font=dict(size=15, color=SLATE_DARK),
        xaxis=dict(
            title="Total Sales (Units)",
            showgrid=True,
            gridcolor=BORDER_GRAY,
            tickformat=",.0f",
        ),
        yaxis=dict(title="", automargin=True),
        margin=dict(l=30, r=20, t=50, b=30),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    return fig


def create_model_comparison_bar(df_perf: pd.DataFrame) -> go.Figure:
    """Renders comparative bar chart for global model WAPE (%)."""
    global_perf = df_perf[df_perf["granularity"] == "global"].copy()
    if global_perf.empty:
        return go.Figure()

    colors = [MODEL_COLORS.get(m, "#6B7280") for m in global_perf["model"]]

    fig = go.Figure(
        go.Bar(
            x=global_perf["model"],
            y=global_perf["wape"],
            marker_color=colors,
            text=[f"{v:.2f}%" for v in global_perf["wape"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Global WAPE: %{y:.2f}%<br>MAE: %{customdata[0]:.1f}<br>RMSE: %{customdata[1]:.1f}<extra></extra>",
            customdata=global_perf[["mae", "rmse"]].values,
        )
    )

    fig.update_layout(
        title="<b>Global 13-Week Holdout WAPE (%) — Lower is Better</b>",
        title_font=dict(size=15, color=SLATE_DARK),
        xaxis=dict(title=""),
        yaxis=dict(
            title="WAPE (%)",
            range=[0, max(global_perf["wape"]) * 1.25],
            showgrid=True,
            gridcolor=BORDER_GRAY,
        ),
        margin=dict(l=30, r=20, t=50, b=30),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    return fig


def create_horizon_decay_chart(df_perf: pd.DataFrame) -> go.Figure:
    """Renders week-by-week horizon error decay curve for all models."""
    horizon_df = df_perf[df_perf["granularity"] == "horizon_week"].copy()
    if horizon_df.empty:
        return go.Figure()

    horizon_df["week_num"] = horizon_df["forecast_horizon_week"].astype(int)
    horizon_df = horizon_df.sort_values("week_num")

    fig = go.Figure()

    for model_name, group in horizon_df.groupby("model"):
        color = MODEL_COLORS.get(model_name, "#6B7280")
        fig.add_trace(
            go.Scatter(
                x=group["week_num"],
                y=group["wape"],
                mode="lines+markers",
                name=model_name,
                line=dict(color=color, width=2.5),
                marker=dict(size=7, color=color),
                hovertemplate=f"<b>{model_name}</b><br>Week %{{x}}<br>WAPE: %{{y:.2f}}%<extra></extra>",
            )
        )

    fig.update_layout(
        title="<b>Multi-Step Forecast Horizon Decay (Weeks 1 to 13)</b>",
        title_font=dict(size=15, color=SLATE_DARK),
        xaxis=dict(
            title="Forecast Horizon (Week Number t+1 to t+13)",
            tickmode="linear",
            tick0=1,
            dtick=1,
            showgrid=True,
            gridcolor=BORDER_GRAY,
        ),
        yaxis=dict(
            title="WAPE (%)",
            showgrid=True,
            gridcolor=BORDER_GRAY,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=30, r=20, t=50, b=40),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        hovermode="x unified",
    )
    return fig


def create_store_ranking_chart(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Renders top stores ranked by historical volume."""
    if df.empty:
        return go.Figure()

    store_agg = (
        df.groupby(["store_nbr", "city", "type"])["weekly_sales"]
        .sum()
        .reset_index()
        .sort_values("weekly_sales", ascending=True)
    )
    if top_n and len(store_agg) > top_n:
        store_agg = store_agg.tail(top_n)

    store_labels = [f"Store {s} ({c}, Type {t})" for s, c, t in zip(store_agg["store_nbr"], store_agg["city"], store_agg["type"])]

    fig = go.Figure(
        go.Bar(
            x=store_agg["weekly_sales"],
            y=store_labels,
            orientation="h",
            marker=dict(color=BLUE_ACCENT),
            hovertemplate="<b>%{y}</b><br>Total Sales: %{x:,.0f} units<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"<b>Top {min(top_n, len(store_agg))} Retail Stores by Total Volume</b>",
        title_font=dict(size=15, color=SLATE_DARK),
        xaxis=dict(title="Total Sales (Units)", showgrid=True, gridcolor=BORDER_GRAY, tickformat=",.0f"),
        yaxis=dict(title="", automargin=True),
        margin=dict(l=30, r=20, t=50, b=30),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    return fig


def create_family_wape_comparison_chart(df_fam_perf: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Grouped bar chart comparing model WAPE across product families."""
    if df_fam_perf.empty:
        return go.Figure()

    # Get top N families by sales volume
    top_families = (
        df_fam_perf.groupby("family")["actual_sales_volume"]
        .first()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    sub_df = df_fam_perf[df_fam_perf["family"].isin(top_families)].copy()

    fig = px.bar(
        sub_df,
        x="family",
        y="wape",
        color="model",
        barmode="group",
        color_discrete_map=MODEL_COLORS,
        title=f"<b>Model WAPE (%) Comparison Across Top {top_n} FMCG Families</b>",
        labels={"family": "Product Family", "wape": "WAPE (%)", "model": "Model"},
    )

    fig.update_layout(
        title_font=dict(size=15, color=SLATE_DARK),
        xaxis=dict(tickangle=-35, showgrid=False),
        yaxis=dict(title="WAPE (%)", showgrid=True, gridcolor=BORDER_GRAY),
        margin=dict(l=30, r=20, t=50, b=60),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


RISK_COLORS = {
    "HIGH": "#E11D48",
    "MEDIUM": "#D97706",
    "LOW": "#059669",
    "UNKNOWN": "#64748B",
}


def create_stockout_risk_pie(df_plan: pd.DataFrame) -> go.Figure:
    """Renders donut chart for stockout risk distribution (HIGH, MEDIUM, LOW)."""
    if df_plan.empty:
        return go.Figure()

    risk_counts = (
        df_plan["stockout_risk"]
        .value_counts()
        .reindex(["HIGH", "MEDIUM", "LOW", "UNKNOWN"], fill_value=0)
        .reset_index()
    )
    risk_counts.columns = ["Risk Tier", "Item Count"]
    risk_counts = risk_counts[risk_counts["Item Count"] > 0]

    colors = [RISK_COLORS.get(r, "#64748B") for r in risk_counts["Risk Tier"]]

    fig = go.Figure(
        go.Pie(
            labels=risk_counts["Risk Tier"],
            values=risk_counts["Item Count"],
            hole=0.45,
            marker=dict(colors=colors),
            textinfo="label+percent",
            hovertemplate="<b>%{label} Risk</b><br>Count: %{value:,} items<br>Share: %{percent}<extra></extra>",
        )
    )

    fig.update_layout(
        title="<b>Stockout Risk Tier Distribution</b>",
        title_font=dict(size=15, color=SLATE_DARK),
        margin=dict(l=20, r=20, t=45, b=20),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
    )
    return fig


def create_highest_risk_items_bar(df_plan: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart showing items with largest suggested replenishment deficit."""
    if df_plan.empty:
        return go.Figure()

    high_items = df_plan[df_plan["suggested_replenishment_qty"] > 0].copy()
    if high_items.empty:
        fig = go.Figure()
        fig.add_annotation(text="No items require replenishment reorders", showarrow=False)
        return fig

    top_items = (
        high_items.sort_values("suggested_replenishment_qty", ascending=True)
        .tail(top_n)
    )

    labels = [f"Store {s}: {f}" for s, f in zip(top_items["store_nbr"], top_items["family"])]
    bar_colors = [RISK_COLORS.get(r, "#64748B") for r in top_items["stockout_risk"]]

    fig = go.Figure(
        go.Bar(
            x=top_items["suggested_replenishment_qty"],
            y=labels,
            orientation="h",
            marker=dict(color=bar_colors),
            hovertemplate="<b>%{y}</b><br>Suggested Order: %{x:,.0f} units<br>Risk: %{customdata[0]}<br>Stock: %{customdata[1]:,.0f}<br>ROP: %{customdata[2]:,.0f}<extra></extra>",
            customdata=top_items[["stockout_risk", "current_stock", "reorder_point"]].values,
        )
    )

    fig.update_layout(
        title=f"<b>Top {len(top_items)} Items with Largest Replenishment Deficit</b>",
        title_font=dict(size=15, color=SLATE_DARK),
        xaxis=dict(title="Suggested Reorder Quantity (Units)", showgrid=True, gridcolor=BORDER_GRAY),
        yaxis=dict(title="", automargin=True),
        margin=dict(l=30, r=20, t=50, b=30),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
    )
    return fig


def create_demand_vs_stock_scatter(df_plan: pd.DataFrame) -> go.Figure:
    """Scatter comparison of Expected Lead-Time Demand vs. Current Stock on Hand."""
    if df_plan.empty:
        return go.Figure()

    # Limit to reasonable display sample if large to preserve 60fps responsiveness
    plot_df = df_plan.copy()
    if len(plot_df) > 800:
        plot_df = plot_df.sample(800, random_state=42)

    color_discrete = {
        "HIGH": RISK_COLORS["HIGH"],
        "MEDIUM": RISK_COLORS["MEDIUM"],
        "LOW": RISK_COLORS["LOW"],
        "UNKNOWN": RISK_COLORS["UNKNOWN"],
    }

    fig = px.scatter(
        plot_df,
        x="expected_lt_demand",
        y="current_stock",
        color="stockout_risk",
        color_discrete_map=color_discrete,
        hover_data=["store_nbr", "family", "reorder_point", "suggested_replenishment_qty"],
        labels={
            "expected_lt_demand": "Expected Lead-Time Demand (Units)",
            "current_stock": "Current Stock on Hand (Units)",
            "stockout_risk": "Stockout Risk",
        },
        title="<b>Expected Demand vs. Current Stock (Items Below 1:1 Line are at Risk)</b>",
    )

    # Add 45-degree parity reference line
    max_val = max(plot_df["expected_lt_demand"].max(), plot_df["current_stock"].max()) * 1.05
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="1:1 Parity Line (Stock = Demand)",
            line=dict(color="#94A3B8", dash="dash", width=1.5),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title_font=dict(size=15, color=SLATE_DARK),
        xaxis=dict(showgrid=True, gridcolor=BORDER_GRAY),
        yaxis=dict(showgrid=True, gridcolor=BORDER_GRAY),
        margin=dict(l=30, r=20, t=50, b=40),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig

