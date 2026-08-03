from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

@pytest.fixture
def client() -> TestClient:
                                                           
    with TestClient(app) as test_client:
        yield test_client

def test_root_health_check(client: TestClient) -> None:
                                                                     
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert "models" in data
    assert "article" in data["models"]
    assert "headline" in data["models"]

def test_predict_classical_headline_routing(client: TestClient) -> None:
                                                     
    res = client.post(
        "/predict",
        json={
            "text": "European leaders meet in Brussels to discuss trade.",
            "model_type": "classical",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["model_tier"] == "headline"
    assert data["model_type"] == "classical"
    assert data["label_name"] in {"real", "fake", "uncertain"}
    assert isinstance(data.get("flagged_phrases"), list)

def test_predict_classical_article_routing(client: TestClient) -> None:
                                                   
    long_text = "This is a very long text to simulate an article body. " * 10
    res = client.post(
        "/predict",
        json={
            "text": long_text,
            "model_type": "classical",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["model_tier"] == "article"
    assert data["model_type"] == "classical"
    assert data["label_name"] in {"real", "fake", "uncertain"}

def test_predict_deep_learning_routing(client: TestClient) -> None:
                                                                                     
    res = client.post(
        "/predict",
        json={
            "text": "European leaders meet in Brussels to discuss trade.",
            "model_type": "deep_learning",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["model_tier"] == "deep_learning"
    assert data["model_type"] == "deep_learning"
    assert data["label_name"] in {"real", "fake", "uncertain"}
    assert isinstance(data.get("flagged_phrases"), list)

def test_predict_transformer_routing(client: TestClient) -> None:
                                                                                              
    res = client.post(
        "/predict",
        json={
            "text": "European leaders meet in Brussels to discuss trade.",
            "model_type": "transformer",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["model_tier"] == "transformer"
    assert data["model_type"] == "transformer"
    assert data["label_name"] in {"real", "fake", "uncertain"}
    assert "accuracy" in data["model_metadata"]["metrics"]
    assert isinstance(data.get("flagged_phrases"), list)
