"""CI policy-gate (v0.2) — the *gate* leg of gate -> observe -> govern.

Mirrors evalstack's gate idea: run the routing eval suite over a fixed prompt set
against the deterministic stub provider, then FAIL the build (non-zero exit) if
either quality-bar pass-rate or estimated savings regresses past a threshold.

Because the stub provider + heuristic judge are deterministic, this is stable in
CI: a routing/policy change that quietly degrades quality or savings trips the
gate instead of silently shipping.

Usage:
    python scripts/eval_gate.py \
        --min-pass-rate 0.8 \
        --min-savings-pct 20 \
        [--json]

Thresholds may also come from env: ROUTERAI_MIN_PASS_RATE, ROUTERAI_MIN_SAVINGS_PCT.
Exit code 0 = gate passed, 1 = regression (build should fail), 2 = usage error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# ensure repo root is importable when run as `python scripts/eval_gate.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_policy  # noqa: E402
from app.metrics import MetricsStore, Decision  # noqa: E402
from app.observability import NullSink, set_sink  # noqa: E402
from app.providers import Message, get_provider  # noqa: E402
from app.routing import route, set_db  # noqa: E402

# a representative, fixed eval set spanning the tiers.
EVAL_PROMPTS = [
    ("Classify the sentiment of: 'I love this!'", None),
    ("Extract the date from: 'Meeting on 2026-07-01'.", "cheap"),
    ("Summarize the following paragraph in one line.", None),
    ("Translate 'good morning' into French.", "cheap"),
    ("List three benefits of unit testing.", "balanced"),
    ("Prove step by step why the sum of the first n odd numbers is n squared, "
     "and analyze the reasoning carefully.", None),
    ("Explain why distributed consensus is hard.", "premium"),
    ("Rewrite this sentence to be more concise.", "balanced"),
]


def run_eval() -> dict:
    """Run the fixed eval set through the router (isolated store, no side effects)."""
    # isolate: dedicated in-memory store, no DB, no event emission.
    store = MetricsStore()
    set_db(None)
    set_sink(NullSink())
    provider = get_provider("stub")  # deterministic gate, key-free

    # route() records into the global STORE; we recompute an isolated summary from
    # the returned RouteResult so the gate has no dependency on process globals.
    decisions = []
    for prompt, tier in EVAL_PROMPTS:
        result = route(
            [Message(role="user", content=prompt)],
            quality_tier=tier,
            provider=provider,
        )
        decisions.append(
            Decision(
                request_id=result.request_id,
                tier=result.tier,
                chosen_model=result.chosen_model,
                baseline_model=get_policy().baseline,
                total_tokens=result.completion.total_tokens,
                est_cost=result.est_cost,
                baseline_cost=result.baseline_cost,
                eval_score=result.eval_score,
                passed_quality_bar=result.passed_quality_bar,
            )
        )
    for d in decisions:
        store.record(d)
    return store.summary()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="routerai eval policy-gate")
    parser.add_argument(
        "--min-pass-rate", type=float,
        default=float(os.getenv("ROUTERAI_MIN_PASS_RATE", "0.8")),
        help="Fail if quality-bar pass rate is below this (0.0-1.0).",
    )
    parser.add_argument(
        "--min-savings-pct", type=float,
        default=float(os.getenv("ROUTERAI_MIN_SAVINGS_PCT", "20")),
        help="Fail if estimated savings vs premium is below this (percent).",
    )
    parser.add_argument("--json", action="store_true", help="Emit the summary as JSON.")
    args = parser.parse_args(argv)

    summary = run_eval()
    pass_rate = summary["quality_bar_pass_rate"] or 0.0
    savings_pct = summary["savings_pct_vs_premium"] or 0.0

    failures = []
    if pass_rate < args.min_pass_rate:
        failures.append(
            f"pass-rate {pass_rate:.3f} < min {args.min_pass_rate:.3f}"
        )
    if savings_pct < args.min_savings_pct:
        failures.append(
            f"savings {savings_pct:.2f}% < min {args.min_savings_pct:.2f}%"
        )

    if args.json:
        print(json.dumps(
            {
                "summary": summary,
                "thresholds": {
                    "min_pass_rate": args.min_pass_rate,
                    "min_savings_pct": args.min_savings_pct,
                },
                "failures": failures,
                "gate": "FAIL" if failures else "PASS",
            },
            indent=2,
        ))
    else:
        print("=" * 70)
        print("routerai eval policy-gate")
        print("=" * 70)
        print(f"  requests         : {summary['total_requests']}")
        print(f"  pass_rate        : {pass_rate:.3f}  (min {args.min_pass_rate})")
        print(f"  savings_pct      : {savings_pct:.2f}%  (min {args.min_savings_pct}%)")
        print(f"  avg_eval_score   : {summary['avg_eval_score']}")
        print("=" * 70)

    if failures:
        print("GATE FAILED: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
