import requests
import pytest

BASE_URL = "http://127.0.0.1:8000"


def _must_reach(url: str):
    try:
        requests.get(url, timeout=1.0)
        return True
    except requests.RequestException:
        pytest.skip(f"Server not reachable at {BASE_URL}; start uvicorn and retry")


def test_root_integration():
    _must_reach(BASE_URL)
    resp = requests.get(f"{BASE_URL}/")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("message") == "Integrated Analytics API"
    assert "text" in data.get("domains", [])
    assert "text" in data.get("active_domains", [])


def test_health_integration():
    _must_reach(BASE_URL)
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "healthy"
    assert "timestamp" in data


def test_analyze_positive_integration():
    _must_reach(BASE_URL)
    payload = {"text": "I loved this movie"}
    resp = requests.post(f"{BASE_URL}/text/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("domain") == "text"
    assert data.get("text") == payload["text"]
    assert data.get("sentiment") == "positive"
    assert isinstance(data.get("confidence"), float)


def test_analyze_validation_error_integration():
    _must_reach(BASE_URL)
    resp = requests.post(f"{BASE_URL}/text/analyze", json={})
    assert resp.status_code == 422

