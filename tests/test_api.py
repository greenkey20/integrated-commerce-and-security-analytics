import requests
import pytest

BASE_URL = "http://127.0.0.1:8000"


def _must_reach(url: str):
    try:
        requests.get(url, timeout=1.0)
        return True
    except requests.RequestException:
        pytest.skip(f"Server not reachable at {BASE_URL}; start uvicorn and retry")


def test_root():
    _must_reach(BASE_URL)
    resp = requests.get(f"{BASE_URL}/")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("message") == "Text Analytics API v1.0"
    assert data.get("status") == "running"


def test_health():
    _must_reach(BASE_URL)
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "healthy"
    assert "timestamp" in data


def test_analyze_positive():
    _must_reach(BASE_URL)
    payload = {"text": "I loved this movie"}
    resp = requests.post(f"{BASE_URL}/text/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("domain") == "text"
    assert data.get("text") == payload["text"]
    assert data.get("sentiment") == "positive"
    assert isinstance(data.get("confidence"), float)


def test_analyze_validation_error():
    _must_reach(BASE_URL)
    # missing required 'text' field should produce 422
    resp = requests.post(f"{BASE_URL}/text/analyze", json={})
    assert resp.status_code == 422
