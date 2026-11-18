"""SQLite-backed fragment storage and deterministic lexical retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import re
import sqlite3
import uuid


class SourceKind(str, Enum):
    USER_STATEMENT = "user_statement"
    OBSERVATION = "observation"
    EXTERNAL_MATERIAL = "external_material"
    AGENT_INFERENCE = "agent_inference"


class QueryStatus(str, Enum):
    COMPLETE_WITH_MATCHES = "complete_with_matches"
    COMPLETE_NO_MATCHES = "complete_no_matches"
    NEEDS_CLARIFICATION = "needs_clarification"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


@dataclass(frozen=True)
class Record:
    fragment_id: str
    content: str
    source_kind: SourceKind
    source_ref: str
    recorded_at: str
    scope: str


@dataclass(frozen=True)
class QueryResult:
    status: QueryStatus
    matches: tuple[Record, ...]


def _features(text: str) -> set[str]:
    normalized = text.casefold()
    features: set[str] = set()
    for run in re.findall(r"[\u3400-\u9fff]+|[a-z0-9]+", normalized):
        if re.fullmatch(r"[\u3400-\u9fff]+", run):
            if len(run) == 1:
                features.add(run)
            else:
                features.update(run[index : index + 2] for index in range(len(run) - 1))
        else:
            features.add(run)
    return features


class MemoryStore:
    """Persist independent claims with provenance and support scoped searches."""

    def __init__(self, database: str | Path):
        self._connection = sqlite3.connect(str(database))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS fragments (
                fragment_id TEXT PRIMARY KEY,
                content TEXT NOT NULL CHECK (length(trim(content)) > 0),
                source_kind TEXT NOT NULL,
                source_ref TEXT NOT NULL CHECK (length(trim(source_ref)) > 0),
                recorded_at TEXT NOT NULL,
                scope TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def record(
        self,
        content: str,
        source_kind: SourceKind,
        *,
        source_ref: str,
        scope: str = "default",
    ) -> Record:
        if not isinstance(source_kind, SourceKind):
            raise ValueError("source_kind must be a SourceKind value")
        if not content.strip():
            raise ValueError("content must not be empty")
        if not source_ref.strip():
            raise ValueError("source_ref must not be empty")
        if not scope.strip():
            raise ValueError("scope must not be empty")

        record = Record(
            fragment_id=str(uuid.uuid4()),
            content=content,
            source_kind=source_kind,
            source_ref=source_ref,
            recorded_at=datetime.now(timezone.utc).isoformat(),
            scope=scope,
        )
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO fragments
                    (fragment_id, content, source_kind, source_ref, recorded_at, scope)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.fragment_id,
                    record.content,
                    record.source_kind.value,
                    record.source_ref,
                    record.recorded_at,
                    record.scope,
                ),
            )
        return record

    def get(self, fragment_id: str) -> Record:
        row = self._connection.execute(
            "SELECT * FROM fragments WHERE fragment_id = ?", (fragment_id,)
        ).fetchone()
        if row is None:
            raise KeyError(fragment_id)
        return self._to_record(row)

    def search(
        self,
        query: str,
        *,
        scope: str | None = None,
    ) -> QueryResult:
        """Search all fragments in scope and report the backend scan status."""
        if not query.strip():
            return QueryResult(QueryStatus.NEEDS_CLARIFICATION, ())

        try:
            rows = self._scan_scope(scope)
            if rows is None:
                return QueryResult(QueryStatus.INCOMPLETE, ())
            query_features = _features(query)
            if not query_features:
                return QueryResult(QueryStatus.NEEDS_CLARIFICATION, ())

            ranked: list[tuple[float, str, Record]] = []
            for row in rows:
                record = self._to_record(row)
                candidate_features = _features(record.content)
                overlap = len(query_features & candidate_features)
                score = overlap / len(query_features)
                if score >= 0.2:
                    ranked.append((score, record.recorded_at, record))
            ranked.sort(key=lambda item: (-item[0], item[1], item[2].fragment_id))
            matches = tuple(item[2] for item in ranked)
            status = (
                QueryStatus.COMPLETE_WITH_MATCHES
                if matches
                else QueryStatus.COMPLETE_NO_MATCHES
            )
            return QueryResult(status, matches)
        except sqlite3.Error:
            return QueryResult(QueryStatus.FAILED, ())

    def _scan_scope(self, scope: str | None) -> list[sqlite3.Row] | None:
        if scope is None:
            return self._connection.execute(
                "SELECT * FROM fragments ORDER BY recorded_at DESC, fragment_id"
            ).fetchall()
        return self._connection.execute(
            """SELECT * FROM fragments WHERE scope = ?
               ORDER BY recorded_at DESC, fragment_id""",
            (scope,),
        ).fetchall()

    @staticmethod
    def _to_record(row: sqlite3.Row) -> Record:
        return Record(
            fragment_id=row["fragment_id"],
            content=row["content"],
            source_kind=SourceKind(row["source_kind"]),
            source_ref=row["source_ref"],
            recorded_at=row["recorded_at"],
            scope=row["scope"],
        )
