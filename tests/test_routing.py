from app.config import get_policy
from app.metrics import STORE
from app.providers import Message, StubProvider
from app.routing import classify, route


def setup_function():
    STORE.reset()


def test_stub_provider_is_deterministic():
    p = StubProvider()
    msgs = [Message(role="user", content="hello world")]
    a = p.complete("stub-small", msgs)
    b = p.complete("stub-small", msgs)
    assert a.text == b.text
    assert a.total_tokens == b.total_tokens > 0


def test_classifier_tiers():
    policy = get_policy()
    assert classify([Message("user", "hi there")], policy) == "cheap"
    assert classify(
        [Message("user", "prove step by step why this analyze holds")], policy
    ) == "premium"


def test_router_picks_cheapest_and_records():
    result = route([Message(role="user", content="classify this short text")])
    policy = get_policy()
    # chosen model must be the cheapest in its tier
    assert result.chosen_model == policy.cheapest(result.tier)
    # a decision was recorded
    assert len(STORE.decisions()) == 1


def test_explicit_tier_hint_is_honored():
    result = route([Message(role="user", content="anything")], quality_tier="premium")
    assert result.tier == "premium"


def test_savings_vs_baseline_nonnegative_for_cheap_tier():
    result = route([Message(role="user", content="short prompt")], quality_tier="cheap")
    assert result.baseline_cost >= result.est_cost


def test_eval_score_and_quality_bar_populated():
    result = route([Message(role="user", content="summarize the meeting notes please")])
    assert result.eval_score is not None
    assert result.passed_quality_bar is not None
