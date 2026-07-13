"""Observability emit (v0.2) — the *observe* leg of gate -> observe -> govern.

routerai emits exactly one structured **trace event** per request. The schema is
the tracelens-style trace so the event can be shipped straight to
[tracelens](https://github.com/kartikaneja): model, latency_ms, tokens, cost,
eval_score, passed.

The sink is pluggable behind an `EventSink` protocol:
  - `JsonlSink` (default): append one JSON line per event to a file. Zero deps,
    tail-able, trivially ingestible.
  - `WebhookSink`: POST each event to an HTTP endpoint (feed a live tracelens).
  - `MultiSink` / `NullSink`: fan-out and no-op.

Selected via env: ROUTERAI_SINK = jsonl (default) | webhook | null.
Never raises into the request path — a broken sink degrades to a logged warning.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Protocol

logger = logging.getLogger("routerai.observability")

DEFAULT_EVENTS_PATH = Path(__file__).resolve().parent.parent / "events.jsonl"


@dataclass
class TraceEvent:
    """One tracelens-style trace per request."""
    request_id: str
    ts: float
    tier: str
    model: str
    provider: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    baseline_cost: float
    eval_score: Optional[float]
    passed: Optional[bool]

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


class EventSink(Protocol):
    def emit(self, event: TraceEvent) -> None:
        ...


class NullSink:
    """Discards events (useful in tests / to disable emit)."""

    name = "null"

    def emit(self, event: TraceEvent) -> None:  # noqa: D102
        return None


class JsonlSink:
    """Append one JSON line per event to a file (default sink)."""

    name = "jsonl"

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path or os.getenv("ROUTERAI_EVENTS_PATH") or DEFAULT_EVENTS_PATH)
        self._lock = threading.Lock()

    def emit(self, event: TraceEvent) -> None:
        try:
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(event.to_json() + "\n")
        except Exception as err:  # noqa: BLE001 - never break the request path
            logger.warning("JsonlSink emit failed: %s", err)


class WebhookSink:
    """POST each event to an HTTP endpoint (e.g. a live tracelens collector)."""

    name = "webhook"

    def __init__(self, url: str | None = None, timeout: float = 2.0):
        self.url = url or os.getenv("ROUTERAI_WEBHOOK_URL", "")
        self.timeout = timeout

    def emit(self, event: TraceEvent) -> None:
        if not self.url:
            logger.warning("WebhookSink has no URL (ROUTERAI_WEBHOOK_URL); dropping event")
            return
        try:
            import httpx  # lazy: only needed for the webhook sink

            httpx.post(self.url, json=json.loads(event.to_json()), timeout=self.timeout)
        except Exception as err:  # noqa: BLE001 - never break the request path
            logger.warning("WebhookSink emit failed: %s", err)


class MultiSink:
    """Fan an event out to several sinks."""

    name = "multi"

    def __init__(self, sinks: List[EventSink]):
        self.sinks = sinks

    def emit(self, event: TraceEvent) -> None:
        for sink in self.sinks:
            sink.emit(event)


def build_sink(name: str | None = None) -> EventSink:
    """Construct the configured sink. Defaults to JSONL append."""
    name = (name or os.getenv("ROUTERAI_SINK") or "jsonl").lower()
    if name == "null":
        return NullSink()
    if name == "webhook":
        return WebhookSink()
    if name == "jsonl":
        return JsonlSink()
    logger.warning("unknown ROUTERAI_SINK=%r; using jsonl", name)
    return JsonlSink()


# module-level default sink used by the router
_SINK: EventSink = build_sink()


def get_sink() -> EventSink:
    return _SINK


def set_sink(sink: EventSink) -> None:
    """Swap the active sink (used by tests and custom wiring)."""
    global _SINK
    _SINK = sink


def emit_event(event: TraceEvent) -> None:
    get_sink().emit(event)


def now_ms() -> float:
    return time.perf_counter() * 1000.0
