from app.metrics import Decision
from app.persistence import SQLiteStore


def _decision(request_id="req-1", score=0.9, passed=True):
    return Decision(
        request_id=request_id,
        tier="cheap",
        chosen_model="stub-small",
        baseline_model="gpt-4o",
        total_tokens=42,
        est_cost=0.0001,
        baseline_cost=0.002,
        eval_score=score,
        passed_quality_bar=passed,
    )


def test_record_and_load_roundtrip(tmp_path):
    db = SQLiteStore(tmp_path / "t.db")
    db.record(_decision("a"))
    db.record(_decision("b", score=None, passed=None))

    loaded = db.load_all()
    assert len(loaded) == 2
    assert loaded[0].request_id == "a"
    assert loaded[0].passed_quality_bar is True
    # None round-trips through SQLite as None (not 0)
    assert loaded[1].eval_score is None
    assert loaded[1].passed_quality_bar is None


def test_count_and_clear(tmp_path):
    db = SQLiteStore(tmp_path / "t.db")
    db.record(_decision("a"))
    db.record(_decision("b"))
    assert db.count() == 2
    db.clear()
    assert db.count() == 0
    assert db.load_all() == []


def test_persists_across_new_connection(tmp_path):
    path = tmp_path / "persist.db"
    SQLiteStore(path).record(_decision("survivor"))
    # a fresh store object over the same file (simulates a restart)
    reopened = SQLiteStore(path)
    loaded = reopened.load_all()
    assert len(loaded) == 1
    assert loaded[0].request_id == "survivor"


def test_router_writes_to_db_and_metrics_reload(tmp_path):
    from app.metrics import STORE
    from app.observability import NullSink, set_sink
    from app.providers import Message
    from app.routing import route, set_db

    STORE.reset()
    set_sink(NullSink())
    db = SQLiteStore(tmp_path / "route.db")
    set_db(db)
    try:
        route([Message("user", "classify this short text")])
        # durable log got the row
        assert db.count() == 1
        # simulate startup reload into a clean in-memory store
        STORE.reset()
        for d in db.load_all():
            STORE.record(d)
        assert STORE.summary()["total_requests"] == 1
    finally:
        set_db(None)
