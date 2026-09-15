"""Atomic SQLite snapshots and independently queryable audit records."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


class SQLiteCaseRepository:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            self._migrate_legacy_checkpoints(connection)
            connection.executescript(Path(__file__).with_name("schema.sql").read_text())

    @staticmethod
    def _migrate_legacy_checkpoints(connection: sqlite3.Connection) -> None:
        """Release the old table name needed by LangGraph without losing its rows."""
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(checkpoints)").fetchall()
        }
        if {"id", "case_id", "complete", "payload"} <= columns:
            connection.execute("ALTER TABLE checkpoints RENAME TO legacy_checkpoints")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def legacy_case(self, case_id: str) -> LegacyCaseRecord:
        """Read the minimal fields needed to list or export a pre-LangGraph case."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload FROM legacy_checkpoints WHERE case_id=? "
                "AND complete=1 ORDER BY id DESC LIMIT 1",
                (case_id,),
            ).fetchone()
        if row is None:
            raise KeyError("No complete checkpoint exists for this case.")
        return self._parse_legacy_payload(row[0])

    def list_legacy_cases(self) -> list[LegacyCaseRecord]:
        with self._connection() as connection:
            rows = connection.execute("SELECT payload FROM cases ORDER BY case_id").fetchall()
        return [self._parse_legacy_payload(row[0]) for row in rows]

    @staticmethod
    def _parse_legacy_payload(payload: str) -> LegacyCaseRecord:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("Legacy case payload is not an object.")
        case_id = value.get("case_id")
        phase = value.get("phase")
        document_path = value.get("document_path")
        if not isinstance(case_id, str) or not isinstance(phase, str):
            raise ValueError("Legacy case payload is missing its identity or phase.")
        if document_path is not None and not isinstance(document_path, str):
            raise ValueError("Legacy document path is invalid.")
        return LegacyCaseRecord(case_id, phase, document_path)

    def create_graph_case(self, case_id: str, initial_request: str) -> None:
        with self._connection() as connection:
            try:
                connection.execute(
                    "INSERT INTO graph_cases(case_id,initial_request,status) VALUES (?,?,?)",
                    (case_id, initial_request, "classifying"),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError("Case already exists.") from error

    def update_graph_case(
        self, case_id: str, *, status: str, document_path: str | None = None
    ) -> None:
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE graph_cases SET status=?, document_path=COALESCE(?,document_path), "
                "updated_at=CURRENT_TIMESTAMP WHERE case_id=?",
                (status, document_path, case_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown graph case: {case_id}")

    def graph_case(self, case_id: str) -> GraphCaseRecord:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT case_id,initial_request,status,document_path "
                "FROM graph_cases WHERE case_id=?",
                (case_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown graph case: {case_id}")
        return GraphCaseRecord(*row)

    def list_graph_cases(self) -> list[GraphCaseRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT case_id,initial_request,status,document_path "
                "FROM graph_cases ORDER BY created_at,case_id"
            ).fetchall()
        return [GraphCaseRecord(*row) for row in rows]

    def save_graph_message(self, case_id: str, *, role: str, content: str) -> None:
        if role not in ("user", "assistant"):
            raise ValueError("Unsupported conversation role.")
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO graph_messages(case_id,role,content) VALUES (?,?,?)",
                (case_id, role, content),
            )

    def save_graph_exchange(self, case_id: str, *, assistant: str, user: str) -> None:
        with self._connection() as connection:
            connection.executemany(
                "INSERT INTO graph_messages(case_id,role,content) VALUES (?,?,?)",
                (
                    (case_id, "assistant", assistant),
                    (case_id, "user", user),
                ),
            )

    def index_graph_document(self, case_id: str, draft_version: int, path: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO graph_documents(case_id,draft_version,path) VALUES (?,?,?)",
                (case_id, draft_version, path),
            )

    def graph_document(self, case_id: str) -> Path:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT path FROM graph_documents WHERE case_id=? "
                "ORDER BY draft_version DESC,id DESC LIMIT 1",
                (case_id,),
            ).fetchone()
        if row is None:
            raise ValueError("No approved generated document exists for this case.")
        return Path(row[0])

    def graph_messages(self, case_id: str) -> list[dict[str, str]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT role,content FROM graph_messages WHERE case_id=? ORDER BY id",
                (case_id,),
            ).fetchall()
        return [{"role": row[0], "content": row[1]} for row in rows]

    def begin_graph_model_run(
        self,
        *,
        run_id: str,
        case_id: str,
        capability: str,
        model: str,
        started_at: str,
    ) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO graph_model_runs("
                "run_id,case_id,capability,model,status,started_at) VALUES (?,?,?,?,?,?)",
                (run_id, case_id, capability, model, "running", started_at),
            )

    def finish_graph_model_run(
        self,
        run_id: str,
        *,
        status: str,
        finished_at: str,
        error_type: str | None = None,
    ) -> None:
        if status not in ("succeeded", "failed"):
            raise ValueError("Unsupported graph model-run status.")
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE graph_model_runs SET status=?,finished_at=?,error_type=? WHERE run_id=?",
                (status, finished_at, error_type, run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown graph model run: {run_id}")

    def graph_model_runs(self, case_id: str) -> list[GraphModelRunRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT run_id,case_id,capability,model,status,started_at,"
                "finished_at,error_type FROM graph_model_runs "
                "WHERE case_id=? ORDER BY rowid",
                (case_id,),
            ).fetchall()
        return [GraphModelRunRecord(*row) for row in rows]


@dataclass(frozen=True)
class GraphCaseRecord:
    case_id: str
    initial_request: str
    status: str
    document_path: str | None


@dataclass(frozen=True)
class LegacyCaseRecord:
    case_id: str
    phase: str
    document_path: str | None


@dataclass(frozen=True)
class GraphModelRunRecord:
    run_id: str
    case_id: str
    capability: str
    model: str
    status: str
    started_at: str
    finished_at: str | None
    error_type: str | None
