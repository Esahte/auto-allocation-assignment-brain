"""
Reliable dashboard-to-OR-Tools event outbox.

This module is a small sender-side helper for services that publish lifecycle
events to OR-Tools over Socket.IO. Persist events before emitting them, then
retry until OR-Tools returns a matching successful callback ack.
"""

from __future__ import annotations

import json
import random
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import socketio


TERMINAL_STATUSES = {"acked", "dead_letter"}


def derive_event_id(event_type: str, payload: Dict[str, Any]) -> str:
    """Derive a stable id for retries when the caller did not provide one."""
    if payload.get("event_id"):
        return str(payload["event_id"])

    task_id = str(payload.get("id") or payload.get("task_id") or "")
    tookan_job_id = str(payload.get("tookan_job_id") or payload.get("job_id") or "")
    job_type = str(payload.get("job_type") if payload.get("job_type") is not None else "")
    agent_id = str(payload.get("agent_id") or "")
    lifecycle_time = str(
        payload.get("completed_at")
        or payload.get("cancelled_at")
        or payload.get("accepted_at")
        or payload.get("assigned_at")
        or payload.get("timestamp")
        or ""
    )
    return f"{event_type}:{task_id}:{tookan_job_id}:{job_type}:{agent_id}:{lifecycle_time}"


class ReliableEventOutbox:
    """SQLite-backed outbox with Socket.IO callback-ack retries."""

    def __init__(
        self,
        db_path: str | Path,
        sio: socketio.Client,
        ack_timeout_seconds: float = 5.0,
        max_attempts: int = 12,
        base_backoff_seconds: float = 2.0,
        max_backoff_seconds: float = 120.0,
    ):
        self.db_path = Path(db_path)
        self.sio = sio
        self.ack_timeout_seconds = ack_timeout_seconds
        self.max_attempts = max_attempts
        self.base_backoff_seconds = base_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reliable_event_outbox (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    last_error TEXT,
                    ack_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_reliable_event_outbox_pending
                ON reliable_event_outbox(status, next_attempt_at)
                """
            )

    def enqueue(self, event_type: str, payload: Dict[str, Any]) -> str:
        """Persist an event before emitting it. Safe to call repeatedly."""
        event_id = derive_event_id(event_type, payload)
        payload = dict(payload)
        payload["event_id"] = event_id
        now = time.time()

        with self._connect() as conn:
            existing = conn.execute(
                "SELECT status FROM reliable_event_outbox WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if existing and existing["status"] in TERMINAL_STATUSES:
                return event_id

            conn.execute(
                """
                INSERT INTO reliable_event_outbox (
                    event_id, event_type, payload_json, status,
                    attempts, next_attempt_at, created_at, updated_at
                )
                VALUES (?, ?, ?, 'pending', 0, 0, ?, ?)
                ON CONFLICT(event_id) DO UPDATE SET
                    event_type = excluded.event_type,
                    payload_json = excluded.payload_json,
                    status = CASE
                        WHEN reliable_event_outbox.status = 'acked' THEN 'acked'
                        ELSE 'pending'
                    END,
                    updated_at = excluded.updated_at
                """,
                (event_id, event_type, json.dumps(payload, separators=(",", ":")), now, now),
            )

        return event_id

    def emit(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Enqueue and immediately try to emit one event."""
        event_id = self.enqueue(event_type, payload)
        return self.emit_event(event_id)

    def emit_event(self, event_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM reliable_event_outbox WHERE event_id = ?",
                (event_id,),
            ).fetchone()

        if not row:
            return {"event_id": event_id, "success": False, "error": "event not found"}
        if row["status"] == "acked":
            return {
                "event_id": event_id,
                "success": True,
                "already_acked": True,
                "ack": json.loads(row["ack_json"] or "{}"),
            }
        if row["status"] == "dead_letter":
            return {
                "event_id": event_id,
                "success": False,
                "dead_letter": True,
                "error": row["last_error"],
            }

        event_type = row["event_type"]
        payload = json.loads(row["payload_json"])
        attempts = int(row["attempts"])

        try:
            ack = self.sio.call(event_type, payload, timeout=self.ack_timeout_seconds)
            if not self._ack_matches(event_id, ack):
                raise RuntimeError(f"ack did not match event_id={event_id}: {ack}")

            now = time.time()
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE reliable_event_outbox
                    SET status = 'acked',
                        ack_json = ?,
                        last_error = NULL,
                        attempts = attempts + 1,
                        updated_at = ?
                    WHERE event_id = ?
                    """,
                    (json.dumps(ack, separators=(",", ":")), now, event_id),
                )
            return {"event_id": event_id, "success": True, "ack": ack}
        except Exception as exc:
            attempts += 1
            status = "dead_letter" if attempts >= self.max_attempts else "pending"
            now = time.time()
            next_attempt_at = now + self._backoff_seconds(attempts)
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE reliable_event_outbox
                    SET status = ?,
                        attempts = ?,
                        next_attempt_at = ?,
                        last_error = ?,
                        updated_at = ?
                    WHERE event_id = ?
                    """,
                    (status, attempts, next_attempt_at, str(exc), now, event_id),
                )
            return {
                "event_id": event_id,
                "success": False,
                "retry_scheduled": status == "pending",
                "dead_letter": status == "dead_letter",
                "attempts": attempts,
                "error": str(exc),
            }

    def emit_due(self, limit: int = 100) -> Iterable[Dict[str, Any]]:
        """Emit all currently due pending events."""
        now = time.time()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_id
                FROM reliable_event_outbox
                WHERE status = 'pending' AND next_attempt_at <= ?
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (now, limit),
            ).fetchall()

        for row in rows:
            yield self.emit_event(row["event_id"])

    @staticmethod
    def _ack_matches(event_id: str, ack: Any) -> bool:
        return (
            isinstance(ack, dict)
            and ack.get("success") is True
            and str(ack.get("event_id")) == event_id
        )

    def _backoff_seconds(self, attempts: int) -> float:
        exponent = max(0, attempts - 1)
        delay = min(self.max_backoff_seconds, self.base_backoff_seconds * (2 ** exponent))
        jitter = random.uniform(0, min(delay * 0.25, 5.0))
        return delay + jitter
