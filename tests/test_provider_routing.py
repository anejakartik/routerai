"""Provider-aware routing (v0.3).

Before v0.3 the router picked the cheapest model in a tier with no idea which
provider served it. Every tier's cheapest candidate was a stub model, so with
ROUTERAI_PROVIDER=openai each request sent model="stub-small" to OpenAI, failed
twice, fell back to the stub, and was still billed and scored as a passing cheap
route. These tests pin the fix.
"""
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.config import NoRouteError, load_policy
from app.metrics import STORE
from app.persistence import SQLiteStore
from app.providers import Completion, Message, StubProvider, _RealProvider
from app.routing import route


def setup_function():
    STORE.reset()


class _Recording(_RealProvider):
    """A fake real provider that records model ids and rejects foreign ones."""

    def __init__(self, name, prefix):
        super().__init__(retries=1)
        self.name = name
        self.prefix = prefix
        self.models = []

    def _call(self, model, messages):
        self.models.append(model)
        if not model.startswith(self.prefix):
            raise RuntimeError(f"404: model {model} does not exist")
        return Completion(text="a real answer about " + messages[-1].content,
                          prompt_tokens=10, completion_tokens=30, provider=self.name)


def _msg(text="hello there"):
    return [Message(role="user", content=text)]


@pytest.mark.parametrize("tier", ["cheap", "balanced", "premium"])
def test_openai_only_never_receives_a_foreign_model_id(tier):
    openai = _Recording("openai", "gpt-")
    result = route(_msg(), quality_tier=tier, provider=openai)
    assert openai.models and all(m.startswith("gpt-") for m in openai.models)
    assert result.completion.provider == "openai"
    assert not result.fallback


def test_anthropic_only_routes_to_claude_models():
    anthropic = _Recording("anthropic", "claude-")
    assert route(_msg(), quality_tier="cheap", provider=anthropic).chosen_model == "claude-haiku-4-5"
    assert route(_msg(), quality_tier="premium", provider=anthropic).chosen_model == "claude-sonnet-5-5"


def test_multi_provider_picks_cheapest_across_both_and_dispatches(monkeypatch):
    openai = _Recording("openai", "gpt-")
    anthropic = _Recording("anthropic", "claude-")
    monkeypatch.setattr(
        "app.routing.get_providers", lambda: {"openai": openai, "anthropic": anthropic}
    )
    cheap = route(_msg(), quality_tier="cheap")
    premium = route(_msg(), quality_tier="premium")
    # gpt-4o-mini (0.0003) beats claude-haiku-4-5 (0.003); gpt-4o (0.005) beats sonnet (0.006)
    assert cheap.chosen_model == "gpt-4o-mini"
    assert premium.chosen_model == "gpt-4o"
    assert anthropic.models == []


def test_stub_only_still_routes_to_stub_models():
    result = route(_msg(), quality_tier="premium", provider=StubProvider())
    assert result.chosen_model == "stub-large"
    assert not result.fallback


class _Down(_RealProvider):
    name = "openai"

    def __init__(self):
        super().__init__(retries=0)

    def _call(self, model, messages):
        raise RuntimeError("outage")


def test_fallback_is_not_billed_saved_or_scored():
    result = route(_msg(), quality_tier="cheap", provider=_Down())
    assert result.fallback
    assert result.completion.provider == "stub"
    assert result.est_cost == result.baseline_cost == 0.0
    assert result.eval_score is None and result.passed_quality_bar is None

    m = STORE.summary()
    assert m["fallback_count"] == 1 and m["fallback_rate"] == 1.0
    assert m["est_savings_usd"] == 0.0
    # an outage must not read as a 100% quality pass rate
    assert m["quality_bar_pass_rate"] is None


def test_no_servable_model_raises(tmp_path):
    cfg = tmp_path / "routes.yaml"
    cfg.write_text(
        "baseline: gpt-4o\n"
        "models:\n  gpt-4o: {provider: openai, cost: 0.005}\n"
        "tiers:\n  cheap: [gpt-4o]\n"
    )
    policy = load_policy(cfg)
    with pytest.raises(NoRouteError):
        policy.cheapest("cheap", providers={"anthropic"})


def test_api_returns_503_when_no_model_is_servable(monkeypatch):
    from app.main import app

    monkeypatch.setattr(
        "app.routing.get_providers", lambda: {"nope": _Recording("nope", "x")}
    )
    resp = TestClient(app).post(
        "/v1/chat/completions",
        json={"model": "cheap", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 503


def test_legacy_float_config_is_servable_by_any_provider(tmp_path):
    cfg = tmp_path / "routes.yaml"
    cfg.write_text("models:\n  m1: 0.001\n  m2: 0.002\ntiers:\n  cheap: [m1, m2]\n")
    policy = load_policy(cfg)
    assert policy.provider_for("m1") is None
    assert policy.cheapest("cheap", providers={"openai"}) == "m1"


def test_pre_v03_database_gets_fallback_column(tmp_path):
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """CREATE TABLE decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, request_id TEXT NOT NULL,
                tier TEXT NOT NULL, chosen_model TEXT NOT NULL,
                baseline_model TEXT NOT NULL, total_tokens INTEGER NOT NULL,
                est_cost REAL NOT NULL, baseline_cost REAL NOT NULL,
                eval_score REAL, passed_quality_bar INTEGER,
                created_at TEXT NOT NULL DEFAULT (datetime('now')))"""
        )
        conn.execute(
            "INSERT INTO decisions (request_id, tier, chosen_model, baseline_model,"
            " total_tokens, est_cost, baseline_cost) VALUES ('r1','cheap','m','b',1,0,0)"
        )
    loaded = SQLiteStore(path).load_all()
    assert len(loaded) == 1 and loaded[0].fallback is False
