"""
FreshBasket AI - What-If Scenario Simulator Unit & API Tests
============================================================
Audits the simulation engine and FastAPI /api/scenarios/simulate endpoint:
1. Baseline neutrality (0%, 0%, 0 days matches original plan)
2. +20% demand shock scales forecast demand by +20%
3. -15% demand contraction scales forecast demand by -15%
4. -25% stock shock scales available stock by -25%
5. Lead-time elongation increases lead-time demand, ROP, and orders
6. Non-negativity constraint enforcement (stock >= 0, lead time >= 0)
7. Invalid store numbers and invalid product families validation
8. Dataset preservation audit (files remain byte-for-byte identical)
9. FastAPI API client integration tests
"""

import hashlib
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app, get_data, get_active_inventory
from src.business.scenario_simulator import run_inventory_simulation

client = TestClient(app)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def file_checksum(filepath: Path) -> str:
    """Calculates SHA256 checksum of a file to verify zero modifications."""
    if not filepath.exists():
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_scenario_baseline_neutrality():
    """Test 1: 0% demand, 0% stock, 0 days lead time must match original plan."""
    data = get_data()
    active_inv = get_active_inventory()
    df_fc = data["forecast"]
    df_sup = data.get("suppliers")

    res = run_inventory_simulation(
        df_inventory=active_inv,
        df_forecast=df_fc,
        supplier_df=df_sup,
        demand_change_pct=0.0,
        stock_change_pct=0.0,
        lead_time_adjustment_days=0,
    )

    metrics = res["metrics"]
    assert metrics["original_forecast_demand"] == pytest.approx(metrics["simulated_forecast_demand"], rel=1e-3)
    assert metrics["original_available_stock"] == pytest.approx(metrics["simulated_available_stock"], rel=1e-3)
    assert metrics["original_suggested_order_qty"] == metrics["simulated_suggested_order_qty"]
    assert metrics["original_lead_time"] == metrics["adjusted_lead_time"]
    assert metrics["demand_diff_units"] == pytest.approx(0.0, abs=1e-2)
    assert metrics["order_qty_diff_units"] == 0
    assert metrics["original_risk_counts"] == metrics["simulated_risk_counts"]


def test_scenario_demand_increase_20pct():
    """Test 2: +20% demand change should increase simulated demand by exactly 20%."""
    data = get_data()
    active_inv = get_active_inventory()
    df_fc = data["forecast"]

    res = run_inventory_simulation(
        df_inventory=active_inv,
        df_forecast=df_fc,
        demand_change_pct=20.0,
        stock_change_pct=0.0,
        lead_time_adjustment_days=0,
    )

    metrics = res["metrics"]
    expected_sim = metrics["original_forecast_demand"] * 1.20
    assert metrics["simulated_forecast_demand"] == pytest.approx(expected_sim, rel=1e-3)
    assert metrics["demand_diff_pct"] == pytest.approx(20.0, abs=0.05)
    # Order quantity should increase or stay same
    assert metrics["simulated_suggested_order_qty"] >= metrics["original_suggested_order_qty"]


def test_scenario_demand_decrease_15pct():
    """Test 3: -15% demand change should decrease simulated demand by exactly 15%."""
    data = get_data()
    active_inv = get_active_inventory()
    df_fc = data["forecast"]

    res = run_inventory_simulation(
        df_inventory=active_inv,
        df_forecast=df_fc,
        demand_change_pct=-15.0,
        stock_change_pct=0.0,
        lead_time_adjustment_days=0,
    )

    metrics = res["metrics"]
    expected_sim = metrics["original_forecast_demand"] * 0.85
    assert metrics["simulated_forecast_demand"] == pytest.approx(expected_sim, rel=1e-3)
    assert metrics["demand_diff_pct"] == pytest.approx(-15.0, abs=0.05)
    # Order quantity should decrease or stay same
    assert metrics["simulated_suggested_order_qty"] <= metrics["original_suggested_order_qty"]


def test_scenario_stock_decrease_25pct():
    """Test 4: -25% stock change should decrease simulated stock by exactly 25%."""
    data = get_data()
    active_inv = get_active_inventory()
    df_fc = data["forecast"]

    res = run_inventory_simulation(
        df_inventory=active_inv,
        df_forecast=df_fc,
        demand_change_pct=0.0,
        stock_change_pct=-25.0,
        lead_time_adjustment_days=0,
    )

    metrics = res["metrics"]
    expected_sim_stock = metrics["original_available_stock"] * 0.75
    assert metrics["simulated_available_stock"] == pytest.approx(expected_sim_stock, rel=1e-3)
    assert metrics["stock_diff_pct"] == pytest.approx(-25.0, abs=0.05)
    # Lower stock must increase replenishment order requirements and escalate risk
    assert metrics["simulated_suggested_order_qty"] >= metrics["original_suggested_order_qty"]
    assert metrics["simulated_risk_counts"]["HIGH"] >= metrics["original_risk_counts"]["HIGH"]


def test_scenario_lead_time_increase():
    """Test 5: Increasing lead time should recalculate lead-time demand, ROP, and orders."""
    data = get_data()
    active_inv = get_active_inventory()
    df_fc = data["forecast"]

    res = run_inventory_simulation(
        df_inventory=active_inv,
        df_forecast=df_fc,
        demand_change_pct=0.0,
        stock_change_pct=0.0,
        lead_time_adjustment_days=14,  # +2 weeks lead time
    )

    metrics = res["metrics"]
    assert metrics["adjusted_lead_time"] == pytest.approx(metrics["original_lead_time"] + 14.0, abs=0.1)
    # Lead time demand increases -> orders increase
    assert metrics["simulated_suggested_order_qty"] > metrics["original_suggested_order_qty"]
    assert metrics["high_risk_diff"] >= 0


def test_scenario_non_negativity_constraints():
    """Test 6: Floor constraints guarantee stock and lead time never become negative."""
    data = get_data()
    active_inv = get_active_inventory()
    df_fc = data["forecast"]

    res = run_inventory_simulation(
        df_inventory=active_inv,
        df_forecast=df_fc,
        demand_change_pct=-50.0,
        stock_change_pct=-75.0,
        lead_time_adjustment_days=-7,  # Max allowed reduction
    )

    metrics = res["metrics"]
    assert metrics["simulated_available_stock"] >= 0.0
    assert metrics["adjusted_lead_time"] >= 0.0


def test_scenario_with_filters():
    """Test 7: Filtering by store and family correctly scopes simulation."""
    data = get_data()
    active_inv = get_active_inventory()
    df_fc = data["forecast"]

    res = run_inventory_simulation(
        df_inventory=active_inv,
        df_forecast=df_fc,
        store_nbr=44,
        family="GROCERY I",
        demand_change_pct=10.0,
        stock_change_pct=-10.0,
        lead_time_adjustment_days=3,
    )

    metrics = res["metrics"]
    assert metrics["items_evaluated"] == 1
    assert len(res["top_affected_items"]) == 1
    assert res["top_affected_items"][0]["store_nbr"] == 44
    assert res["top_affected_items"][0]["family"] == "GROCERY I"


def test_scenario_source_files_preserved():
    """Test 8: Confirm simulation does NOT alter forecast or inventory source files."""
    inv_path = PROJECT_ROOT / "data" / "inventory" / "inventory.csv"
    fc_path = PROJECT_ROOT / "data" / "dashboard" / "forecast_data.csv"

    checksum_inv_before = file_checksum(inv_path)
    checksum_fc_before = file_checksum(fc_path)

    data = get_data()
    active_inv = get_active_inventory()
    df_fc = data["forecast"]

    # Run heavy simulation
    _ = run_inventory_simulation(
        df_inventory=active_inv,
        df_forecast=df_fc,
        demand_change_pct=50.0,
        stock_change_pct=-50.0,
        lead_time_adjustment_days=10,
    )

    checksum_inv_after = file_checksum(inv_path)
    checksum_fc_after = file_checksum(fc_path)

    assert checksum_inv_before == checksum_inv_after, "inventory.csv was modified by simulation!"
    assert checksum_fc_before == checksum_fc_after, "forecast_data.csv was modified by simulation!"


def test_api_scenario_simulate_endpoint():
    """Test 9: FastAPI POST /api/scenarios/simulate endpoint returns correct schema."""
    payload = {
        "demand_change_pct": 15.0,
        "stock_change_pct": -10.0,
        "lead_time_adjustment_days": 5,
        "store_nbr": 1,
        "family": "BEVERAGES",
        "model": "naive_forecast",
    }
    response = client.post("/api/scenarios/simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "metrics" in data
    assert "summary_highlight" in data
    assert "weekly_trajectory" in data
    assert "top_affected_items" in data
    assert "assumptions" in data
    assert data["metrics"]["items_evaluated"] == 1


def test_api_scenario_simulate_invalid_inputs():
    """Test 10: Endpoint rejects invalid parameters with 400 or 422."""
    # Invalid store number (> 54)
    res_bad_store = client.post("/api/scenarios/simulate", json={"store_nbr": 999})
    assert res_bad_store.status_code == 400
    assert "Invalid store_nbr" in res_bad_store.json()["detail"]

    # Invalid product family
    res_bad_family = client.post("/api/scenarios/simulate", json={"family": "SPACESHIPS"})
    assert res_bad_family.status_code == 400
    assert "Unknown product family" in res_bad_family.json()["detail"]

    # Out-of-bounds demand percentage (> 100%)
    res_bad_demand = client.post("/api/scenarios/simulate", json={"demand_change_pct": 250.0})
    assert res_bad_demand.status_code == 422

    # Out-of-bounds stock percentage (< -75%)
    res_bad_stock = client.post("/api/scenarios/simulate", json={"stock_change_pct": -90.0})
    assert res_bad_stock.status_code == 422

    # Out-of-bounds lead time adjustment (> 30 days)
    res_bad_lt = client.post("/api/scenarios/simulate", json={"lead_time_adjustment_days": 45})
    assert res_bad_lt.status_code == 422
