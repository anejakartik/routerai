"""One-command demo: fires a few requests through routerai and prints the
routing decisions + aggregate savings vs always-premium.

Runs entirely in-process against the stub provider — NO API key, NO server.

    python demo.py
"""
from __future__ import annotations

from app.config import get_policy
from app.metrics import STORE
from app.observability import get_sink
from app.providers import Message, get_provider
from app.routing import get_db, route

SAMPLE_PROMPTS = [
    ("Classify the sentiment of: 'I love this!'", None),          # short -> cheap
    ("Summarize the following paragraph in one line.", None),      # medium -> balanced
    ("Prove step by step why the sum of the first n odd numbers "
     "is n squared, and analyze the reasoning carefully so a "
     "student could follow every derivation in detail.", None),   # hard -> premium
    ("Extract the date from: 'Meeting on 2026-07-01'.", "cheap"),  # explicit tier
    ("Explain why distributed consensus is hard.", "premium"),     # explicit tier
]


def main() -> None:
    STORE.reset()
    policy = get_policy()

    print("=" * 78)
    print("routerai v0.2 demo — stub provider, no API key needed")
    print(f"provider   : {get_provider().name}   event sink: "
          f"{getattr(get_sink(), 'name', 'unknown')}")
    print(f"baseline (always-premium) model: {policy.baseline} "
          f"@ ${policy.cost_per_1k(policy.baseline):.4f}/1K tokens")
    print(f"quality bar: {policy.quality_bar}")
    print("=" * 78)

    for i, (prompt, tier_hint) in enumerate(SAMPLE_PROMPTS, 1):
        result = route([Message(role="user", content=prompt)], quality_tier=tier_hint)
        savings = result.baseline_cost - result.est_cost
        pass_str = "PASS" if result.passed_quality_bar else "FAIL"
        print(f"\n[{i}] tier={result.tier:<9} model={result.chosen_model:<16} "
              f"tokens={result.completion.total_tokens:<4} "
              f"cost=${result.est_cost:.6f} baseline=${result.baseline_cost:.6f} "
              f"saved=${savings:.6f}")
        print(f"    eval={result.eval_score}  quality_bar={pass_str}")
        print(f"    prompt: {prompt[:64]}{'...' if len(prompt) > 64 else ''}")
        print(f"    reply : {result.completion.text[:80]}...")

    print("\n" + "=" * 78)
    print("AGGREGATE (GET /metrics)")
    print("=" * 78)
    summary = STORE.summary()
    for key in (
        "total_requests", "total_tokens", "est_cost_usd", "baseline_cost_usd",
        "est_savings_usd", "savings_pct_vs_premium", "avg_eval_score",
        "quality_bar_pass_rate", "requests_by_model",
    ):
        print(f"  {key:<24}: {summary[key]}")
    print("=" * 78)
    print(f"\n==> routerai saved an estimated {summary['savings_pct_vs_premium']}% "
          f"vs always using {policy.baseline}, "
          f"pass rate {summary['quality_bar_pass_rate']}.")

    # v0.2: decisions were also persisted to SQLite and one trace event per
    # request was emitted to the sink (feeds tracelens).
    try:
        print(f"==> persisted {get_db().count()} decision(s) to the durable log "
              f"(routerai.db); trace events emitted via the "
              f"'{getattr(get_sink(), 'name', 'unknown')}' sink.")
    except Exception:  # pragma: no cover - demo cosmetics only
        pass


if __name__ == "__main__":
    main()
