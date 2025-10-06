# ensure project root is importable
import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import asyncio
import json

from api_main import (
    root,
    health,
    analyze_text,
    customer_root,
    customer_catchall,
    retail_root,
    retail_catchall,
    security_root,
    security_catchall,
    TextInput,
)


def test_root_func():
    res = asyncio.run(root())
    assert getattr(res, "message") == "Integrated Analytics API"
    assert "text" in getattr(res, "domains")
    assert "text" in getattr(res, "active_domains")


def test_health_func():
    res = asyncio.run(health())
    assert getattr(res, "status") == "healthy"
    assert isinstance(getattr(res, "timestamp"), str)


def test_analyze_positive_func():
    payload = TextInput(text="I loved this movie")
    res = asyncio.run(analyze_text(payload))
    assert res.domain == "text"
    assert res.text == payload.text
    assert res.sentiment == "positive"
    assert isinstance(res.confidence, float)


def test_analyze_negative_func():
    payload = TextInput(text="This was terrible and bad")
    res = asyncio.run(analyze_text(payload))
    assert res.domain == "text"
    assert res.sentiment == "negative"


def _parse_json_response(resp):
    body = getattr(resp, "body", None)
    if body is None:
        body = resp.render()
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    return json.loads(body)


def test_customer_placeholders():
    resp = asyncio.run(customer_root())
    assert resp.status_code == 501
    payload = _parse_json_response(resp)
    assert payload["domain"] == "customer"

    resp2 = asyncio.run(customer_catchall("some/path"))
    assert resp2.status_code == 501
    payload2 = _parse_json_response(resp2)
    assert payload2["path"] == "/customer/some/path"


def test_retail_placeholders():
    resp = asyncio.run(retail_root())
    assert resp.status_code == 501
    payload = _parse_json_response(resp)
    assert payload["domain"] == "retail"

    resp2 = asyncio.run(retail_catchall("x"))
    assert resp2.status_code == 501
    payload2 = _parse_json_response(resp2)
    assert payload2["path"] == "/retail/x"


def test_security_placeholders():
    resp = asyncio.run(security_root())
    assert resp.status_code == 501
    payload = _parse_json_response(resp)
    assert payload["domain"] == "security"

    resp2 = asyncio.run(security_catchall("a/b"))
    assert resp2.status_code == 501
    payload2 = _parse_json_response(resp2)
    assert payload2["path"] == "/security/a/b"

