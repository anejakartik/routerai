"""Test-wide isolation: keep routerai's v0.2 side effects out of the repo.

Without this, routing during tests would write the real `routerai.db` and
`events.jsonl` at the repo root. We redirect persistence to a temp SQLite file
and the event sink to a NullSink for the whole session; individual tests that
exercise persistence/observability still create their own tmp_path stores.
"""
import pytest

from app import observability
from app import routing
from app.persistence import SQLiteStore


@pytest.fixture(autouse=True)
def _isolate_side_effects(tmp_path_factory):
    prev_sink = observability.get_sink()
    prev_db = routing._DB

    observability.set_sink(observability.NullSink())
    routing.set_db(SQLiteStore(tmp_path_factory.mktemp("db") / "test.db"))

    yield

    observability.set_sink(prev_sink)
    routing.set_db(prev_db)
