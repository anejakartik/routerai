import json

from app.observability import (
    JsonlSink,
    MultiSink,
    NullSink,
    TraceEvent,
    build_sink,
    get_sink,
    set_sink,
)


def _event(request_id="req-1"):
    return TraceEvent(
        request_id=request_id,
        ts=123.0,
        tier="cheap",
        model="stub-small",
        provider="stub",
        latency_ms=1.5,
        prompt_tokens=3,
        completion_tokens=4,
        total_tokens=7,
        cost=0.0001,
        baseline_cost=0.002,
        eval_score=0.9,
        passed=True,
    )


def test_trace_event_has_tracelens_fields():
    payload = json.loads(_event().to_json())
    for key in ("model", "latency_ms", "total_tokens", "cost", "eval_score", "passed"):
        assert key in payload


def test_jsonl_sink_appends_one_line_per_event(tmp_path):
    path = tmp_path / "events.jsonl"
    sink = JsonlSink(path)
    sink.emit(_event("a"))
    sink.emit(_event("b"))
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["request_id"] == "a"
    assert json.loads(lines[1])["request_id"] == "b"


def test_null_sink_is_noop():
    NullSink().emit(_event())  # must not raise


def test_multi_sink_fans_out(tmp_path):
    a = JsonlSink(tmp_path / "a.jsonl")
    b = JsonlSink(tmp_path / "b.jsonl")
    MultiSink([a, b]).emit(_event())
    assert (tmp_path / "a.jsonl").exists()
    assert (tmp_path / "b.jsonl").exists()


def test_build_sink_selection(monkeypatch):
    monkeypatch.setenv("ROUTERAI_SINK", "null")
    assert build_sink().name == "null"
    monkeypatch.setenv("ROUTERAI_SINK", "webhook")
    assert build_sink().name == "webhook"
    monkeypatch.delenv("ROUTERAI_SINK", raising=False)
    assert build_sink().name == "jsonl"


def test_set_and_get_sink_swaps_active_sink():
    original = get_sink()
    try:
        n = NullSink()
        set_sink(n)
        assert get_sink() is n
    finally:
        set_sink(original)


def test_router_emits_one_event(tmp_path):
    from app.providers import Message
    from app.routing import route, set_db

    events_path = tmp_path / "emitted.jsonl"
    original = get_sink()
    set_sink(JsonlSink(events_path))
    set_db(None)
    try:
        route([Message("user", "short prompt")])
    finally:
        set_sink(original)
    lines = events_path.read_text().strip().splitlines()
    assert len(lines) == 1
    ev = json.loads(lines[0])
    assert ev["provider"] == "stub"
    assert ev["total_tokens"] > 0
