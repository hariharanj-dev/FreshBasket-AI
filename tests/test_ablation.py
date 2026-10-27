"""
Unit tests for Feature Ablation Study Module
"""
import json
from pathlib import Path
import pytest
from src.evaluation.ablation_study import run_ablation_study

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
ABLATION_REPORT = WORKSPACE_ROOT / "reports" / "ablation_study_report.md"
ABLATION_JSON = WORKSPACE_ROOT / "reports" / "ablation_metrics.json"


def test_ablation_study_execution_and_metrics():
    result = run_ablation_study()

    assert "ablations" in result
    assert len(result["ablations"]) == 5

    # Check baseline exists
    benchmark = next(a for a in result["ablations"] if a["config_id"] == "M0_FULL_MODEL")
    assert benchmark["wape"] > 0
    assert benchmark["wape_delta_pp"] == 0.0

    # Check degradation for missing rolling statistics
    lags_only = next(a for a in result["ablations"] if a["config_id"] == "M4_LAGS_ONLY")
    assert lags_only["wape"] > benchmark["wape"]
    assert lags_only["wape_delta_pp"] > 0

    assert ABLATION_REPORT.exists()
    assert ABLATION_JSON.exists()

    with open(ABLATION_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data["ablations"]) == 5
