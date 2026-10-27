"""
Tests for FreshBasket AI - Business Query Assistant ("Ask FreshBasket AI")
==========================================================================
Validates question intent classification, verified response accuracy, filter handling,
edge cases (unsupported, empty, invalid stores/families), and REST API integration.
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app, get_data
from src.business.assistant_service import (
    classify_question_intent,
    extract_entities_from_text,
    answer_business_question,
)

client = TestClient(app)


def test_intent_classification():
    assert classify_question_intent("What are the highest demand product families?")[0] == "HIGHEST_DEMAND_FAMILIES"
    assert classify_question_intent("What are the lowest demand categories?")[0] == "LOWEST_DEMAND_FAMILIES"
    assert classify_question_intent("Which stores have the highest forecast demand?")[0] == "STORE_FORECAST_COMPARISON"
    assert classify_question_intent("What is the best forecasting model by WAPE?")[0] == "BEST_MODEL_BY_WAPE"
    assert classify_question_intent("Compare models using MAE, RMSE, WAPE, and RMSLE")[0] == "MODEL_COMPARISON_ALL_METRICS"
    assert classify_question_intent("Which items are at high risk of stockout?")[0] == "HIGH_RISK_INVENTORY"
    assert classify_question_intent("What are the suggested replenishment quantities?")[0] == "SUGGESTED_REPLENISHMENT"
    assert classify_question_intent("Which categories have rising or declining demand trends?")[0] == "DEMAND_TRENDS"
    assert classify_question_intent("What is the total forecast demand over the next 13 weeks?")[0] == "TOTAL_FORECAST_DEMAND"
    assert classify_question_intent("How do historical sales compare over time?")[0] == "HISTORICAL_SALES_COMPARISON"
    assert classify_question_intent("What is the stock price of Apple next year?")[0] == "UNSUPPORTED_QUESTION"


def test_entity_extraction():
    valid_fams = ["BEVERAGES", "GROCERY I", "CLEANING", "DAIRY"]
    ent1 = extract_entities_from_text("Show me sales for Store 44 in BEVERAGES using LightGBM", valid_fams)
    assert ent1["store_nbr"] == 44
    assert ent1["family"] == "BEVERAGES"
    assert ent1["model"] == "lightgbm_forecast"

    ent2 = extract_entities_from_text("Store 99 is out of range", valid_fams)
    assert "store_nbr" not in ent2  # Store 99 rejected


def test_answer_best_model_by_wape():
    data = get_data()
    res = answer_business_question("What is the best forecasting model by WAPE?", data)
    assert res["success"] is True
    assert res["intent"] == "BEST_MODEL_BY_WAPE"
    assert "Naive Baseline" in res["answer"]
    assert "11.61%" in res["answer"]
    assert "LightGBM" in res["answer"]
    assert "12.27%" in res["answer"]


def test_answer_model_comparison_metrics():
    data = get_data()
    res = answer_business_question("Compare models using MAE, RMSE, WAPE, and RMSLE", data)
    assert res["success"] is True
    assert res["intent"] == "MODEL_COMPARISON_ALL_METRICS"
    assert "RMSLE" in res["answer"]
    assert "supporting_data" in res


def test_answer_high_risk_inventory():
    data = get_data()
    res = answer_business_question("Which items are at high risk of stockout?", data)
    assert res["success"] is True
    assert res["intent"] == "HIGH_RISK_INVENTORY"
    assert res["supporting_data"]["high_risk_count"] == 354
    assert "354 items classified as HIGH RISK" in res["answer"]


def test_answer_suggested_replenishment():
    data = get_data()
    res = answer_business_question("What are the total suggested replenishment quantities?", data)
    assert res["success"] is True
    assert res["intent"] == "SUGGESTED_REPLENISHMENT"
    assert res["supporting_data"]["total_suggested_replenishment_units"] == 932315
    assert "932,315 units" in res["answer"]


def test_answer_demand_trends():
    data = get_data()
    res = answer_business_question("Which product families have rising demand trends?", data)
    assert res["success"] is True
    assert res["intent"] == "DEMAND_TRENDS"
    assert "SCHOOL AND OFFICE SUPPLIES" in res["answer"]


def test_answer_total_forecast_demand():
    data = get_data()
    res = answer_business_question("What is the total forecast demand over the next 13 weeks?", data)
    assert res["success"] is True
    assert res["intent"] == "TOTAL_FORECAST_DEMAND"
    assert res["supporting_data"]["horizon_weeks"] == 13
    assert res["supporting_data"]["total_units"] > 0


def test_answer_empty_question():
    data = get_data()
    res = answer_business_question("", data)
    assert res["success"] is False
    assert res["intent"] == "EMPTY_QUESTION"


def test_answer_invalid_store():
    data = get_data()
    res = answer_business_question("What is the forecast for Store 99?", data, store_nbr=99)
    assert res["success"] is False
    assert res["intent"] == "INVALID_STORE_IDENTIFIER"
    assert "Store number 99 is invalid" in res["answer"]


def test_answer_invalid_family():
    data = get_data()
    res = answer_business_question("What is the forecast for AIRPLANES?", data, family="AIRPLANES")
    assert res["success"] is False
    assert res["intent"] == "INVALID_PRODUCT_FAMILY"
    assert "AIRPLANES" in res["answer"]


def test_answer_unsupported_question():
    data = get_data()
    res = answer_business_question("What will the weather be in Paris next month?", data)
    assert res["success"] is False
    assert res["intent"] == "UNSUPPORTED_QUESTION"
    assert "I cannot answer that specific question" in res["answer"]
    assert "Ask FreshBasket AI can answer verified questions on:" in res["answer"]


def test_api_assistant_query_endpoint():
    payload = {
        "question": "What are the highest demand product families?",
        "model": "naive_forecast"
    }
    response = client.post("/api/assistant/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["intent"] == "HIGHEST_DEMAND_FAMILIES"
    assert "GROCERY I" in data["answer"]
    assert "supporting_data" in data
    assert "limitations" in data


def test_api_assistant_query_with_filters():
    payload = {
        "question": "Which items are at risk?",
        "store_nbr": 1,
        "family": "BEVERAGES"
    }
    response = client.post("/api/assistant/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["applied_filters"]["store_nbr"] == 1
    assert data["applied_filters"]["family"] == "BEVERAGES"
