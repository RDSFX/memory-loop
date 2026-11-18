"""SQLite-backed fragment storage and deterministic lexical retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
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


class ActivationReason(str, Enum):
    SEARCH = "search"


@dataclass(frozen=True)
class Record:
    fragment_id: str
    content: str
    source_kind: SourceKind
    source_ref: str
    recorded_at: str
    scope: str
    claim_id: str | None = None
    supports: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryResult:
    status: QueryStatus
    matches: tuple[Record, ...]
    trace_id: str | None = None


@dataclass(frozen=True)
class ContextFragment:
    fragment_id: str
    content: str
    source_kind: SourceKind
    source_ref: str
    reasons: tuple[ActivationReason, ...]
    verified_for_current_case: bool


@dataclass(frozen=True)
class DecisionExplanation:
    trace_id: str
    turn_id: str | None
    query: str
    query_status: QueryStatus
    ruleset: str
    fragment_id: str
    score: float | None
    selected: bool
    reason: str


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
                scope TEXT NOT NULL,
                claim_id TEXT,
                supports_json TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(fragments)")
        }
        if "claim_id" not in columns:
            self._connection.execute("ALTER TABLE fragments ADD COLUMN claim_id TEXT")
        if "supports_json" not in columns:
            self._connection.execute(
                "ALTER TABLE fragments ADD COLUMN supports_json TEXT NOT NULL DEFAULT '[]'"
            )
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS search_decisions (
                trace_id TEXT PRIMARY KEY,
                turn_id TEXT,
                query TEXT NOT NULL,
                scope TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                ruleset TEXT NOT NULL
            )"""
        )
        self._connection.execute(
            """CREATE TABLE IF NOT EXISTS search_decision_items (
                trace_id TEXT NOT NULL,
                fragment_id TEXT NOT NULL,
                score REAL NOT NULL,
                selected INTEGER NOT NULL,
                reason TEXT NOT NULL,
                PRIMARY KEY (trace_id, fragment_id),
                FOREIGN KEY (trace_id) REFERENCES search_decisions(trace_id)
            )"""
        )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS fragments_claim_id ON fragments(claim_id)"
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
        claim_id: str | None = None,
        supports: tuple[str, ...] = (),
    ) -> Record:
        if not isinstance(source_kind, SourceKind):
            raise ValueError("source_kind must be a SourceKind value")
        if not content.strip():
            raise ValueError("content must not be empty")
        if not source_ref.strip():
            raise ValueError("source_ref must not be empty")
        if not scope.strip():
            raise ValueError("scope must not be empty")
        if claim_id is not None and not claim_id.strip():
            raise ValueError("claim_id must not be empty when provided")
        if supports and source_kind is not SourceKind.AGENT_INFERENCE:
            raise ValueError("only agent inferences may link supporting fragments")
        supports = tuple(dict.fromkeys(supports))

        record = Record(
            fragment_id=str(uuid.uuid4()),
            content=content,
            source_kind=source_kind,
            source_ref=source_ref,
            recorded_at=datetime.now(timezone.utc).isoformat(),
            scope=scope,
            claim_id=claim_id,
            supports=supports,
        )
        with self._connection:
            for supporting_id in supports:
                exists = self._connection.execute(
                    "SELECT 1 FROM fragments WHERE fragment_id = ?", (supporting_id,)
                ).fetchone()
                if exists is None:
                    raise ValueError(f"supporting fragment does not exist: {supporting_id}")
            self._connection.execute(
                """
                INSERT INTO fragments
                    (fragment_id, content, source_kind, source_ref, recorded_at, scope,
                     claim_id, supports_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.fragment_id,
                    record.content,
                    record.source_kind.value,
                    record.source_ref,
                    record.recorded_at,
                    record.scope,
                    record.claim_id,
                    json.dumps(record.supports, ensure_ascii=False),
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
        turn_id: str | None = None,
    ) -> QueryResult:
        """Search all fragments in scope and report the backend scan status."""
        trace_id = str(uuid.uuid4())
        try:
            if not query.strip():
                self._save_decision(
                    trace_id, turn_id, query, scope, QueryStatus.NEEDS_CLARIFICATION, ()
                )
                return QueryResult(QueryStatus.NEEDS_CLARIFICATION, (), trace_id)
            rows = self._scan_scope(scope)
            if rows is None:
                self._save_decision(
                    trace_id, turn_id, query, scope, QueryStatus.INCOMPLETE, ()
                )
                return QueryResult(QueryStatus.INCOMPLETE, (), trace_id)
            query_features = _features(query)
            if not query_features:
                self._save_decision(
                    trace_id,
                    turn_id,
                    query,
                    scope,
                    QueryStatus.NEEDS_CLARIFICATION,
                    (),
                )
                return QueryResult(QueryStatus.NEEDS_CLARIFICATION, (), trace_id)

            ranked: list[tuple[float, str, Record]] = []
            evaluations: list[tuple[str, float, bool, str]] = []
            for row in rows:
                record = self._to_record(row)
                candidate_features = _features(record.content)
                overlap = len(query_features & candidate_features)
                score = overlap / len(query_features)
                selected = score >= 0.2
                evaluations.append(
                    (
                        record.fragment_id,
                        score,
                        selected,
                        "selected" if selected else "below_threshold",
                    )
                )
                if selected:
                    ranked.append((score, record.recorded_at, record))
            ranked.sort(key=lambda item: (-item[0], item[1], item[2].fragment_id))
            matches = tuple(item[2] for item in ranked)
            status = (
                QueryStatus.COMPLETE_WITH_MATCHES
                if matches
                else QueryStatus.COMPLETE_NO_MATCHES
            )
            self._save_decision(trace_id, turn_id, query, scope, status, evaluations)
            return QueryResult(status, matches, trace_id)
        except sqlite3.Error:
            try:
                self._save_decision(trace_id, turn_id, query, scope, QueryStatus.FAILED, ())
                return QueryResult(QueryStatus.FAILED, (), trace_id)
            except sqlite3.Error:
                return QueryResult(QueryStatus.FAILED, (), None)

    def independent_source_count(self, claim_id: str) -> int:
        """Count distinct non-agent source references, not repeated inferences."""
        row = self._connection.execute(
            """SELECT COUNT(DISTINCT source_ref) AS source_count
               FROM fragments
               WHERE claim_id = ? AND source_kind != ?""",
            (claim_id, SourceKind.AGENT_INFERENCE.value),
        ).fetchone()
        return int(row["source_count"])

    @staticmethod
    def context_for(result: QueryResult) -> tuple[ContextFragment, ...]:
        """Render search hits with provenance and without asserting current truth."""
        if result.status is not QueryStatus.COMPLETE_WITH_MATCHES:
            return ()
        return tuple(
            ContextFragment(
                fragment_id=record.fragment_id,
                content=record.content,
                source_kind=record.source_kind,
                source_ref=record.source_ref,
                reasons=(ActivationReason.SEARCH,),
                verified_for_current_case=False,
            )
            for record in result.matches
        )

    def explain(self, trace_id: str, fragment_id: str) -> DecisionExplanation:
        decision = self._connection.execute(
            "SELECT * FROM search_decisions WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if decision is None:
            raise KeyError(trace_id)
        item = self._connection.execute(
            """SELECT score, selected, reason FROM search_decision_items
               WHERE trace_id = ? AND fragment_id = ?""",
            (trace_id, fragment_id),
        ).fetchone()
        return DecisionExplanation(
            trace_id=trace_id,
            turn_id=decision["turn_id"],
            query=decision["query"],
            query_status=QueryStatus(decision["status"]),
            ruleset=decision["ruleset"],
            fragment_id=fragment_id,
            score=float(item["score"]) if item is not None else None,
            selected=bool(item["selected"]) if item is not None else False,
            reason=item["reason"] if item is not None else "not_scanned",
        )

    def _save_decision(
        self,
        trace_id: str,
        turn_id: str | None,
        query: str,
        scope: str | None,
        status: QueryStatus,
        evaluations: tuple[tuple[str, float, bool, str], ...]
        | list[tuple[str, float, bool, str]],
    ) -> None:
        with self._connection:
            self._connection.execute(
                """INSERT INTO search_decisions
                   (trace_id, turn_id, query, scope, status, created_at, ruleset)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    trace_id,
                    turn_id,
                    query,
                    scope,
                    status.value,
                    datetime.now(timezone.utc).isoformat(),
                    "cjk-bigram-v1",
                ),
            )
            self._connection.executemany(
                """INSERT INTO search_decision_items
                   (trace_id, fragment_id, score, selected, reason)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    (trace_id, fragment_id, score, int(selected), reason)
                    for fragment_id, score, selected, reason in evaluations
                ),
            )

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
            claim_id=row["claim_id"],
            supports=tuple(json.loads(row["supports_json"])),
        )
