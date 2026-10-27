"""
Unit tests for Demand Driver Attribution & Explainability Module
"""
import json
from pathlib import Path
import pytest
from src.explainability.explain_model import run_explainability_analysis, categorize_feature

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
EXPLAINABILITY_REPORT = WORKSPACE_ROOT / "reports" / "explainability_report.md"
ATTRIBUTION_JSON = WORKSPACE_ROOT / "reports" / "feature_attribution.json"


def test_feature_categorization():
    assert categorize_feature("onpromotion_lag_1") == "Promotions & Marketing"
    assert categorize_feature("sales_rolling_mean_4") == "Historical Sales Velocity & Lags"
    assert categorize_feature("quincena_flag") == "Calendar, Holidays & Quincena Paydays"
    assert categorize_feature("store_nbr") == "Store Hierarchy & Product Category"
    assert categorize_feature("oil_price_lag_1") == "Macroeconomic & Traffic Drivers"


def test_explainability_run_and_artifacts():
    result = run_explainability_analysis()

    assert "feature_group_importance_share" in result
    assert "top_15_global_features" in result
    assert len(result["top_15_global_features"]) == 15
    assert len(result["case_studies"]) > 0

    assert EXPLAINABILITY_REPORT.exists()
    assert ATTRIBUTION_JSON.exists()

    with open(ATTRIBUTION_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "feature_group_importance_share" in data
