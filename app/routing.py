"""The router — the control plane.

Given a chat request, it:
  1. determines the quality tier (from an explicit hint or a heuristic classifier),
  2. picks the cheapest candidate model for that tier,
  3. calls the provider,
  4. (shadow) evaluates the response against the quality bar,
  5. records the decision + estimated savings vs the baseline premium model.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

from .config import Policy, get_policy
from .evaluator import evaluate
from .metrics import STORE, Decision
from .observability import TraceEvent, emit_event, now_ms
from .persistence import SQLiteStore
from .providers import Completion, Message, Provider, get_provider

# durable decision log (SQLite). Lazily created so importing the module never
# touches the filesystem until the first decision is recorded.
_DB: Optional[SQLiteStore] = None


def get_db() -> SQLiteStore:
    global _DB
    if _DB is None:
        _DB = SQLiteStore()
    return _DB


def set_db(db: Optional[SQLiteStore]) -> None:
    """Override the durable store (tests point this at a temp DB, or None to skip)."""
    global _DB
    _DB = db

# tiers a caller may request explicitly via the `quality_tier` field / model hint
_KNOWN_TIER_HINTS = {"cheap", "balanced", "premium"}


def classify(messages: List[Message], policy: Policy) -> str:
    """Heuristic complexity classifier -> quality tier.

    v0.1 uses cheap signals: prompt length and reasoning keywords. A fine-tuned
    classifier replaces this in the roadmap; the router only needs a tier string.
    """
    prompt = ""
    for m in reversed(messages):
        if m.role == "user":
            prompt = m.content
            break

    lowered = prompt.lower()
    n_words = len(prompt.split())

    hard_markers = ("prove", "derive", "step by step", "reason", "analyze",
                    "explain why", "legal", "diagnose", "architecture")
    if any(mk in lowered for mk in hard_markers) or n_words > 80:
        return "premium"
    if n_words < 15:
        return "cheap"
    return "balanced" if "balanced" in policy.tiers else policy.default_tier


@dataclass
class RouteResult:
    request_id: str
    tier: str
    chosen_model: str
    completion: Completion
    est_cost: float
    baseline_cost: float
    eval_score: Optional[float]
    passed_quality_bar: Optional[bool]


def _resolve_tier(requested: Optional[str], messages: List[Message], policy: Policy) -> str:
    if requested and requested in _KNOWN_TIER_HINTS and requested in policy.tiers:
        return requested
    return classify(messages, policy)


def route(
    messages: List[Message],
    quality_tier: Optional[str] = None,
    policy: Optional[Policy] = None,
    provider: Optional[Provider] = None,
    do_eval: Optional[bool] = None,
    request_id: Optional[str] = None,
) -> RouteResult:
    policy = policy or get_policy()
    provider = provider or get_provider()
    request_id = request_id or f"req-{uuid.uuid4().hex[:8]}"

    tier = _resolve_tier(quality_tier, messages, policy)
    chosen_model = policy.cheapest(tier)

    started = now_ms()
    completion = provider.complete(chosen_model, messages)
    latency_ms = round(now_ms() - started, 3)

    # cost is per 1K tokens
    est_cost = policy.cost_per_1k(chosen_model) * completion.total_tokens / 1000.0
    baseline_cost = policy.cost_per_1k(policy.baseline) * completion.total_tokens / 1000.0

    # shadow eval hook
    if do_eval is None:
        do_eval = policy.shadow_sample_rate >= 1.0 or _sample(request_id, policy.shadow_sample_rate)

    eval_score: Optional[float] = None
    passed: Optional[bool] = None
    if do_eval:
        prompt_text = next((m.content for m in reversed(messages) if m.role == "user"), "")
        eval_score = evaluate(prompt_text, completion.text)
        passed = eval_score >= policy.quality_bar

    decision = Decision(
        request_id=request_id,
        tier=tier,
        chosen_model=chosen_model,
        baseline_model=policy.baseline,
        total_tokens=completion.total_tokens,
        est_cost=round(est_cost, 8),
        baseline_cost=round(baseline_cost, 8),
        eval_score=eval_score,
        passed_quality_bar=passed,
    )

    # in-memory hot path (metrics) + durable log (survives restart)
    STORE.record(decision)
    db = get_db()
    if db is not None:
        db.record(decision)

    # observability: emit one tracelens-style trace per request
    emit_event(
        TraceEvent(
            request_id=request_id,
            ts=time.time(),
            tier=tier,
            model=chosen_model,
            provider=completion.provider,
            latency_ms=latency_ms,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            total_tokens=completion.total_tokens,
            cost=round(est_cost, 8),
            baseline_cost=round(baseline_cost, 8),
            eval_score=eval_score,
            passed=passed,
        )
    )

    return RouteResult(
        request_id=request_id,
        tier=tier,
        chosen_model=chosen_model,
        completion=completion,
        est_cost=round(est_cost, 8),
        baseline_cost=round(baseline_cost, 8),
        eval_score=eval_score,
        passed_quality_bar=passed,
    )


def _sample(request_id: str, rate: float) -> bool:
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    # deterministic sampling from the request id
    bucket = (hash(request_id) % 1000) / 1000.0
    return bucket < rate
