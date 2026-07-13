"""SQLite persistence for routing decisions (v0.2).

Every routing decision is appended to a SQLite table so the decision log survives
restarts. The in-memory `MetricsStore` remains the hot read path; on startup the
app reloads the persisted decisions so aggregate metrics (/metrics) are warm and
continuous across restarts.

Stdlib-only (`sqlite3`). Path is overridable via ROUTERAI_DB (default
`routerai.db` next to the repo root). Thread-safe: one lock, connections opened
per operation with `check_same_thread=False`.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional

from .metrics import Decision

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "routerai.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id         TEXT NOT NULL,
    tier               TEXT NOT NULL,
    chosen_model       TEXT NOT NULL,
    baseline_model     TEXT NOT NULL,
    total_tokens       INTEGER NOT NULL,
    est_cost           REAL NOT NULL,
    baseline_cost      REAL NOT NULL,
    eval_score         REAL,
    passed_quality_bar INTEGER,
    created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def db_path(path: str | os.PathLike | None = None) -> Path:
    return Path(path or os.getenv("ROUTERAI_DB") or DEFAULT_DB_PATH)


class SQLiteStore:
    """Durable append-log of routing decisions."""

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = db_path(path)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)

    def record(self, decision: Decision, timestamp: Optional[str] = None) -> None:
        """Persist one decision. `timestamp` optional (defaults to SQLite now())."""
        with self._lock, self._connect() as conn:
            if timestamp is None:
                conn.execute(
                    """INSERT INTO decisions
                       (request_id, tier, chosen_model, baseline_model, total_tokens,
                        est_cost, baseline_cost, eval_score, passed_quality_bar)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        decision.request_id, decision.tier, decision.chosen_model,
                        decision.baseline_model, decision.total_tokens,
                        decision.est_cost, decision.baseline_cost,
                        decision.eval_score,
                        _bool_to_int(decision.passed_quality_bar),
                    ),
                )
            else:
                conn.execute(
                    """INSERT INTO decisions
                       (request_id, tier, chosen_model, baseline_model, total_tokens,
                        est_cost, baseline_cost, eval_score, passed_quality_bar, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?)""",
                    (
                        decision.request_id, decision.tier, decision.chosen_model,
                        decision.baseline_model, decision.total_tokens,
                        decision.est_cost, decision.baseline_cost,
                        decision.eval_score,
                        _bool_to_int(decision.passed_quality_bar),
                        timestamp,
                    ),
                )

    def load_all(self) -> List[Decision]:
        """Reload every persisted decision as in-memory Decision objects."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """SELECT request_id, tier, chosen_model, baseline_model, total_tokens,
                          est_cost, baseline_cost, eval_score, passed_quality_bar
                   FROM decisions ORDER BY id ASC"""
            ).fetchall()
        return [
            Decision(
                request_id=r["request_id"],
                tier=r["tier"],
                chosen_model=r["chosen_model"],
                baseline_model=r["baseline_model"],
                total_tokens=r["total_tokens"],
                est_cost=r["est_cost"],
                baseline_cost=r["baseline_cost"],
                eval_score=r["eval_score"],
                passed_quality_bar=_int_to_bool(r["passed_quality_bar"]),
            )
            for r in rows
        ]

    def count(self) -> int:
        with self._lock, self._connect() as conn:
            return conn.execute("SELECT COUNT(*) AS n FROM decisions").fetchone()["n"]

    def clear(self) -> None:
        """Drop all rows (used by tests)."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM decisions")


def _bool_to_int(value: Optional[bool]) -> Optional[int]:
    return None if value is None else int(value)


def _int_to_bool(value: Optional[int]) -> Optional[bool]:
    return None if value is None else bool(value)
