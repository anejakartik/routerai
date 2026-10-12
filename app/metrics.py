"""In-memory metrics store for routing decisions.

Tracks per-request: chosen model, tokens, estimated cost, eval score, and the
estimated cost had we always used the baseline (premium) model. Exposes an
aggregate summary including estimated savings vs always-premium.
"""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class Decision:
    request_id: str
    tier: str
    chosen_model: str
    baseline_model: str
    total_tokens: int
    est_cost: float
    baseline_cost: float
    eval_score: float | None
    passed_quality_bar: bool | None
    fallback: bool = False


@dataclass
class MetricsStore:
    _decisions: List[Decision] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, decision: Decision) -> None:
        with self._lock:
            self._decisions.append(decision)

    def reset(self) -> None:
        with self._lock:
            self._decisions.clear()

    def decisions(self) -> List[Decision]:
        with self._lock:
            return list(self._decisions)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            decisions = list(self._decisions)

        total_requests = len(decisions)
        total_cost = round(sum(d.est_cost for d in decisions), 6)
        baseline_cost = round(sum(d.baseline_cost for d in decisions), 6)
        est_savings = round(baseline_cost - total_cost, 6)
        savings_pct = round(100.0 * est_savings / baseline_cost, 2) if baseline_cost else 0.0

        scored = [d.eval_score for d in decisions if d.eval_score is not None]
        avg_eval = round(sum(scored) / len(scored), 4) if scored else None
        n_passed = sum(1 for d in decisions if d.passed_quality_bar)
        n_evaluated = sum(1 for d in decisions if d.passed_quality_bar is not None)

        n_fallback = sum(1 for d in decisions if d.fallback)

        by_model: Dict[str, int] = {}
        for d in decisions:
            by_model[d.chosen_model] = by_model.get(d.chosen_model, 0) + 1

        return {
            "total_requests": total_requests,
            "total_tokens": sum(d.total_tokens for d in decisions),
            "est_cost_usd": total_cost,
            "baseline_cost_usd": baseline_cost,
            "est_savings_usd": est_savings,
            "savings_pct_vs_premium": savings_pct,
            "avg_eval_score": avg_eval,
            "quality_bar_pass_rate": (
                round(n_passed / n_evaluated, 4) if n_evaluated else None
            ),
            # requests the stub answered because the routed provider failed;
            # excluded from cost, savings and pass rate above
            "fallback_count": n_fallback,
            "fallback_rate": round(n_fallback / total_requests, 4) if total_requests else 0.0,
            "requests_by_model": by_model,
            "recent_decisions": [asdict(d) for d in decisions[-10:]],
        }


# app-wide singleton
STORE = MetricsStore()
