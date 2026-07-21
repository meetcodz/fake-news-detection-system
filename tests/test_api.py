"""Unit tests for the FastAPI API and routing layers."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    """Test client instance with loaded model lifespans."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_health_check(client: TestClient) -> None:
    """Verify that root returns 200 and indicates model readiness."""
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert "models" in data
    assert "article" in data["models"]
    assert "headline" in data["models"]


def test_predict_classical_headline_routing(client: TestClient) -> None:
    """Short input should route to headline model."""
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


def test_predict_classical_article_routing(client: TestClient) -> None:
    """Long input should route to article model."""
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
    """Explicit deep learning model selection should route to deep learning model."""
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
