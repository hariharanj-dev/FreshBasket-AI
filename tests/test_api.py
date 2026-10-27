"""Tests for FreshBasket AI FastAPI Endpoints
============================================
Validates response schema, status codes, data integrity, and business logic
across all REST API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "FreshBasket AI API"
    assert "datasets" in data


def test_filters_endpoint():
    response = client.get("/api/filters")
    assert response.status_code == 200
    data = response.json()
    assert len(data["stores"]) == 54
    assert len(data["families"]) == 33
    assert len(data["models"]) == 3
    # Check best model identification
    best_model = next(m for m in data["models"] if m["is_best"])
    assert best_model["id"] == "naive_forecast"
    assert best_model["wape"] == 11.61


def test_overview_endpoint():
    response = client.get("/api/overview")
    assert response.status_code == 200
    data = response.json()
    assert "kpis" in data
    assert "timeline" in data
    assert "model_performance" in data
    assert "top_families" in data
    assert data["best_model"]["name"] == "Naive Baseline"
    assert data["best_model"]["wape"] == 11.61


def test_sales_endpoint():
    response = client.get("/api/sales")
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "time_series" in data
    assert len(data["time_series"]) > 0
    assert "moving_avg_4wk" in data["time_series"][0]


def test_sales_with_filters():
    response = client.get("/api/sales?store_nbr=1&family=BEVERAGES")
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["store_filter"] == 1
    assert data["summary"]["family_filter"] == "BEVERAGES"
    assert len(data["time_series"]) > 0


def test_forecasts_endpoint():
    response = client.get("/api/forecasts?model=naive_forecast")
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"]["horizon_weeks"] == 13
    assert len(data["weekly_forecast"]) == 13
    assert data["total_forecast_units"] > 0


def test_evaluation_endpoint():
    response = client.get("/api/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert len(data["global_matrix"]) == 3
    # Verify Naive is lowest WAPE
    sorted_by_wape = sorted(data["global_matrix"], key=lambda x: x["wape"])
    assert sorted_by_wape[0]["model"].lower() == "naive"
    assert sorted_by_wape[0]["wape"] == 11.61
    assert "horizon_decay" in data
    assert len(data["horizon_decay"]) == 13
    assert "rmsle_audit" in data


def test_products_endpoint():
    response = client.get("/api/products")
    assert response.status_code == 200
    data = response.json()
    assert len(data["family_accuracy"]) > 0
    assert len(data["family_volumes"]) == 33
    assert "categories_where_lgbm_leads" in data


def test_stores_endpoint():
    response = client.get("/api/stores")
    assert response.status_code == 200
    data = response.json()
    assert len(data["store_rankings"]) == 54
    assert len(data["type_summary"]) > 0


def test_inventory_summary_endpoint():
    response = client.get("/api/inventory/summary")
    assert response.status_code == 200
    data = response.json()
    assert data["total_items"] == 1782
    assert data["high_risk_count"] == 354
    assert data["medium_risk_count"] == 368
    assert data["low_risk_count"] == 1060
    assert data["unknown_risk_count"] == 0
    assert data["total_suggested_replenishment_qty"] == 932315
    assert "business_insights" in data


def test_inventory_recommendations_endpoint():
    response = client.get("/api/inventory/recommendations?risk_level=HIGH&limit=10")
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 354
    assert len(data["records"]) == 10
    assert all(r["stockout_risk"] == "HIGH" for r in data["records"])


def test_inventory_alerts_endpoint():
    response = client.get("/api/inventory/alerts")
    assert response.status_code == 200
    data = response.json()
    assert data["high_risk_count"] == 354
    assert data["medium_risk_count"] == 368
    assert data["stores_affected"] == 54
    assert len(data["store_summary"]) == 54


def test_download_replenishment_csv():
    response = client.get("/api/reports/replenishment")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    assert b"store_nbr,family,current_stock" in response.content or b"store_nbr" in response.content


def test_executive_report_endpoint():
    response = client.get("/api/reports/executive")
    assert response.status_code == 200
    assert "text/markdown" in response.headers["content-type"]
    text = response.text
    assert "EXECUTIVE INVENTORY" in text


def test_upload_invalid_csv():
    # Test uploading non-csv file or invalid data
    files = {"file": ("test.txt", b"not a csv", "text/plain")}
    response = client.post("/api/inventory/upload", files=files)
    assert response.status_code == 400
