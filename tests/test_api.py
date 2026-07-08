from fastapi.testclient import TestClient

from app.main import app
from app.metrics import STORE

client = TestClient(app)


def setup_function():
    STORE.reset()


def test_chat_completions_openai_shape():
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "cheap",
            "messages": [{"role": "user", "content": "classify: I love it"}],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    # OpenAI-compatible fields
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]
    assert body["usage"]["total_tokens"] > 0
    # routerai metadata
    meta = body["routerai"]
    assert meta["tier"] == "cheap"
    assert meta["est_savings_usd"] >= 0
    assert meta["eval_score"] is not None


def test_metrics_endpoint_aggregates():
    for _ in range(3):
        client.post(
            "/v1/chat/completions",
            json={
                "model": "premium",
                "messages": [{"role": "user", "content": "prove step by step and analyze"}],
            },
        )
    m = client.get("/metrics").json()
    assert m["total_requests"] == 3
    assert m["baseline_cost_usd"] >= m["est_cost_usd"]
    assert "requests_by_model" in m


def test_health():
    assert client.get("/health").json()["status"] == "ok"
