"""SQLite-backed fragment storage and deterministic lexical retrieval."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import re
import sqlite3
from typing import TypeVar
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
    REMINDER = "reminder"
    ASSOCIATION = "association"


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
    correction: Record | None = None
    correction_source_ref: str | None = None


@dataclass(frozen=True)
class Reminder:
    reminder_id: str
    content: str
    trigger: str
    end_event: str | None
    source_ref: str
    scope: str
    principal: str
    fragment_id: str | None = None
    only_occurrence_id: str | None = None


@dataclass(frozen=True)
class ReminderNotice:
    reminder: Reminder
    occurrence_id: str
    context: ContextFragment | None

    @property
    def reminder_id(self) -> str:
        return self.reminder.reminder_id

    @property
    def delivery_key(self) -> tuple[str, str]:
        return self.reminder_id, self.occurrence_id


@dataclass(frozen=True)
class ReminderDelivery:
    reminder_id: str
    occurrence_id: str
    status: str
    error: str | None = None
    turn_id: str | None = None
    event: str | None = None
    fragment_id: str | None = None
    correction_id: str | None = None


@dataclass(frozen=True)
class Constraint:
    constraint_id: str
    content: str
    action: str
    source_ref: str
    scope: str
    principal: str


@dataclass(frozen=True)
class ActionCheck:
    allowed: bool
    constraints: tuple[Constraint, ...] = ()
    failed: bool = False


class ActionBlocked(PermissionError):
    def __init__(self, check: ActionCheck):
        self.check = check
        super().__init__("action check failed" if check.failed else "action blocked by constraint")


_T = TypeVar("_T")


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
    correction: Record | None = None
    correction_source_ref: str | None = None


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
    correction_id: str | None = None


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

    def __init__(
        self,
        database: str | Path,
        *,
        authorizer: Callable[[str, str, str], bool] | None = None,
    ):
        self._authorizer = authorizer
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
                correction_id TEXT,
                PRIMARY KEY (trace_id, fragment_id),
                FOREIGN KEY (trace_id) REFERENCES search_decisions(trace_id)
            )"""
        )
        decision_columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(search_decision_items)")
        }
        if "correction_id" not in decision_columns:
            self._connection.execute(
                "ALTER TABLE search_decision_items ADD COLUMN correction_id TEXT"
            )
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS fragments_claim_id ON fragments(claim_id)"
        )
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS reminders (
                reminder_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                trigger TEXT NOT NULL,
                end_event TEXT,
                source_ref TEXT NOT NULL,
                scope TEXT NOT NULL,
                principal TEXT NOT NULL,
                fragment_id TEXT REFERENCES fragments(fragment_id),
                only_occurrence_id TEXT,
                active INTEGER NOT NULL DEFAULT 1,
                revoked_by TEXT
            );
            CREATE TABLE IF NOT EXISTS situation_events (
                event TEXT NOT NULL,
                occurrence_id TEXT NOT NULL,
                scope TEXT NOT NULL,
                PRIMARY KEY (event, occurrence_id, scope)
            );
            CREATE TABLE IF NOT EXISTS reminder_deliveries (
                attempt INTEGER PRIMARY KEY,
                reminder_id TEXT NOT NULL,
                occurrence_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('pending', 'failed', 'delivered')),
                error TEXT,
                turn_id TEXT,
                event TEXT NOT NULL,
                fragment_id TEXT,
                correction_id TEXT,
                FOREIGN KEY (reminder_id) REFERENCES reminders(reminder_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS reminder_claim
                ON reminder_deliveries(reminder_id, occurrence_id) WHERE status != 'failed';
            CREATE INDEX IF NOT EXISTS reminder_history
                ON reminder_deliveries(reminder_id, occurrence_id, attempt);
            CREATE TABLE IF NOT EXISTS constraints (
                constraint_id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                action TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                scope TEXT NOT NULL,
                principal TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                revoked_by TEXT
            );
            CREATE TABLE IF NOT EXISTS corrections (
                old_fragment_id TEXT PRIMARY KEY,
                replacement_fragment_id TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                principal TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (old_fragment_id) REFERENCES fragments(fragment_id),
                FOREIGN KEY (replacement_fragment_id) REFERENCES fragments(fragment_id)
            );
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def _require_authority(self, principal: str, operation: str, scope: str) -> None:
        if not isinstance(principal, str) or not principal.strip() or self._authorizer is None:
            raise PermissionError("no verified authority is configured")
        try:
            allowed = self._authorizer(principal, operation, scope)
        except Exception as error:
            raise PermissionError("authority verification failed") from error
        if allowed is not True:
            raise PermissionError(f"principal is not authorized for {operation}")

    @staticmethod
    def _required(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must not be empty")
        return value

    def create_reminder(
        self,
        content: str,
        *,
        trigger: str,
        principal: str,
        source_ref: str,
        scope: str = "default",
        end_event: str | None = None,
        fragment_id: str | None = None,
        only_occurrence_id: str | None = None,
    ) -> Reminder:
        self._required(content, "content")
        self._required(trigger, "trigger")
        self._required(source_ref, "source_ref")
        self._required(scope, "scope")
        self._require_authority(principal, "reminder.create", scope)
        if end_event is not None:
            self._required(end_event, "end_event")
            if end_event == trigger:
                raise ValueError("trigger and end_event must differ")
        if fragment_id is not None and self.get(fragment_id).scope != scope:
            raise ValueError("reminder and referenced fragment must share a scope")
        if only_occurrence_id is not None:
            self._required(only_occurrence_id, "only_occurrence_id")
        reminder = Reminder(
            str(uuid.uuid4()), content, trigger, end_event, source_ref, scope, principal, fragment_id,
            only_occurrence_id,
        )
        with self._connection:
            self._connection.execute(
                """INSERT INTO reminders
                   (reminder_id, content, trigger, end_event, source_ref, scope, principal,
                    fragment_id, only_occurrence_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    reminder.reminder_id, content, trigger, end_event, source_ref, scope, principal,
                    fragment_id, only_occurrence_id,
                ),
            )
        return reminder

    def revoke_reminder(self, reminder_id: str, *, principal: str) -> None:
        row = self._connection.execute(
            "SELECT scope FROM reminders WHERE reminder_id = ?", (reminder_id,),
        ).fetchone()
        if row is None:
            raise KeyError(reminder_id)
        self._require_authority(principal, "reminder.revoke", row["scope"])
        with self._connection:
            self._connection.execute(
                "UPDATE reminders SET active = 0, revoked_by = ? WHERE reminder_id = ? AND active = 1",
                (principal, reminder_id),
            )

    def evaluate_reminders(
        self,
        event: str,
        occurrence_id: str,
        *,
        scope: str,
        deliver: Callable[[ReminderNotice], bool],
        turn_id: str | None = None,
    ) -> tuple[ReminderDelivery, ...]:
        """Process a trusted event and acknowledge only confirmed host delivery."""
        self._required(event, "event")
        self._required(occurrence_id, "occurrence_id")
        self._required(scope, "scope")
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO situation_events VALUES (?, ?, ?)",
                (event, occurrence_id, scope),
            )
            rows = self._connection.execute(
                """SELECT * FROM reminders AS r
                   WHERE active = 1 AND trigger = ? AND scope = ?
                   AND (only_occurrence_id IS NULL OR only_occurrence_id = ?) AND NOT EXISTS (
                       SELECT 1 FROM situation_events AS e
                       WHERE e.event = r.end_event AND e.scope = r.scope AND e.occurrence_id = ?
                   )
                   ORDER BY reminder_id""",
                (event, scope, occurrence_id, occurrence_id),
            ).fetchall()
        outcomes: list[ReminderDelivery] = []
        for row in rows:
            item = Reminder(
                row["reminder_id"], row["content"], row["trigger"], row["end_event"],
                row["source_ref"], row["scope"], row["principal"], row["fragment_id"],
                row["only_occurrence_id"],
            )
            with self._connection:
                self._connection.execute("BEGIN IMMEDIATE")
                claim = self._connection.execute(
                    """INSERT OR IGNORE INTO reminder_deliveries
                       (reminder_id, occurrence_id, status, turn_id, event, fragment_id)
                       SELECT r.reminder_id, ?, 'pending', ?, ?, r.fragment_id FROM reminders AS r
                       WHERE r.reminder_id = ? AND r.active = 1 AND NOT EXISTS (
                           SELECT 1 FROM situation_events AS e
                           WHERE e.event = r.end_event AND e.scope = r.scope AND e.occurrence_id = ?
                       )""",
                    (occurrence_id, turn_id, event, item.reminder_id, occurrence_id),
                )
                if claim.rowcount == 0:
                    previous = self._connection.execute(
                        """SELECT * FROM reminder_deliveries WHERE reminder_id = ?
                           AND occurrence_id = ? AND status != 'failed'""",
                        (item.reminder_id, occurrence_id),
                    ).fetchone()
                    if previous is not None:
                        outcome = self._delivery_from_row(previous)
                        outcomes.append(replace(
                            outcome,
                            status="already_delivered" if outcome.status == "delivered" else "pending",
                        ))
                    continue
                attempt = claim.lastrowid
                context = None
                correction_id = None
                if item.fragment_id is not None:
                    record = self.get(item.fragment_id)
                    context = self._context_fragment(record, ActivationReason.REMINDER)
                    correction_id = self._current_correction_id(record)
                    self._connection.execute(
                        "UPDATE reminder_deliveries SET correction_id = ? WHERE attempt = ?",
                        (correction_id, attempt),
                    )
            try:
                delivered = deliver(ReminderNotice(item, occurrence_id, context)) is True
                status = "delivered" if delivered else "failed"
                error = None if delivered else "delivery callback reported failure"
            except Exception as exception:
                status = "failed"
                error = f"{type(exception).__name__}: {exception}"
            with self._connection:
                self._connection.execute(
                    "UPDATE reminder_deliveries SET status = ?, error = ? WHERE attempt = ?",
                    (status, error, attempt),
                )
            outcomes.append(ReminderDelivery(
                item.reminder_id, occurrence_id, status, error, turn_id, event,
                item.fragment_id, correction_id,
            ))
        return tuple(outcomes)

    def reminder_history(
        self, reminder_id: str, occurrence_id: str,
    ) -> tuple[ReminderDelivery, ...]:
        rows = self._connection.execute(
            """SELECT * FROM reminder_deliveries WHERE reminder_id = ? AND occurrence_id = ?
               ORDER BY attempt""",
            (reminder_id, occurrence_id),
        )
        return tuple(self._delivery_from_row(row) for row in rows)

    @staticmethod
    def _delivery_from_row(row: sqlite3.Row) -> ReminderDelivery:
        return ReminderDelivery(
            row["reminder_id"], row["occurrence_id"], row["status"], row["error"], row["turn_id"],
            row["event"], row["fragment_id"], row["correction_id"],
        )

    def create_constraint(
        self,
        content: str,
        *,
        action: str,
        principal: str,
        source_ref: str,
        scope: str = "default",
    ) -> Constraint:
        self._required(content, "content")
        self._required(action, "action")
        self._required(source_ref, "source_ref")
        self._required(scope, "scope")
        self._require_authority(principal, "constraint.create", scope)
        constraint = Constraint(str(uuid.uuid4()), content, action, source_ref, scope, principal)
        with self._connection:
            self._connection.execute(
                """INSERT INTO constraints
                   (constraint_id, content, action, source_ref, scope, principal)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (constraint.constraint_id, content, action, source_ref, scope, principal),
            )
        return constraint

    def revoke_constraint(self, constraint_id: str, *, principal: str) -> None:
        row = self._connection.execute(
            "SELECT scope FROM constraints WHERE constraint_id = ?", (constraint_id,),
        ).fetchone()
        if row is None:
            raise KeyError(constraint_id)
        self._require_authority(principal, "constraint.revoke", row["scope"])
        with self._connection:
            self._connection.execute(
                "UPDATE constraints SET active = 0, revoked_by = ? WHERE constraint_id = ? AND active = 1",
                (principal, constraint_id),
            )

    def check_action(self, action: str, *, scope: str) -> ActionCheck:
        try:
            self._required(action, "action")
            self._required(scope, "scope")
            rows = self._connection.execute(
                """SELECT * FROM constraints WHERE active = 1 AND action = ?
                   AND scope = ? ORDER BY constraint_id""",
                (action, scope),
            ).fetchall()
            constraints = tuple(
                Constraint(
                    row["constraint_id"], row["content"], row["action"], row["source_ref"],
                    row["scope"], row["principal"],
                )
                for row in rows
            )
            return ActionCheck(not constraints, constraints)
        except (sqlite3.Error, ValueError):
            return ActionCheck(False, (), failed=True)

    def run_action(self, action: str, *, scope: str, execute: Callable[[], _T]) -> _T:
        """Run a host-bound action only if its immediate constraint check succeeds."""
        try:
            # Serialize constraint writes with the protected side effect.
            self._connection.execute("BEGIN IMMEDIATE")
        except sqlite3.Error as error:
            raise ActionBlocked(ActionCheck(False, failed=True)) from error
        with self._connection:
            check = self.check_action(action, scope=scope)
            if not check.allowed:
                raise ActionBlocked(check)
            return execute()

    def link_correction(
        self,
        old_fragment_id: str,
        replacement_fragment_id: str,
        *,
        principal: str,
        source_ref: str,
    ) -> None:
        self._required(source_ref, "source_ref")
        if old_fragment_id == replacement_fragment_id:
            raise ValueError("a fragment cannot correct itself")
        scope = self.get(old_fragment_id).scope
        self._require_authority(principal, "correction.link", scope)
        with self._connection:
            self._connection.execute("BEGIN IMMEDIATE")
            if self.get(old_fragment_id).correction is not None:
                raise ValueError("correct the current replacement rather than rewriting a correction")
            replacement = self.get(replacement_fragment_id)
            if replacement.scope != scope:
                raise ValueError("correction fragments must share a scope")
            if replacement.correction is not None:
                raise ValueError("replacement is already superseded")
            self._connection.execute(
                """INSERT INTO corrections
                   (old_fragment_id, replacement_fragment_id, source_ref, principal, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    old_fragment_id, replacement_fragment_id, source_ref, principal,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

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
        return self._with_correction(self._to_record(row))

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
            evaluations: list[tuple[str, float, bool, str, str | None]] = []
            for row in rows:
                record = self._to_record(row)
                candidate_features = _features(record.content)
                overlap = len(query_features & candidate_features)
                score = overlap / len(query_features)
                selected = score >= 0.2
                if selected:
                    record = self._with_correction(record)
                    ranked.append((score, record.recorded_at, record))
                evaluations.append(
                    (
                        record.fragment_id,
                        score,
                        selected,
                        "selected" if selected else "below_threshold",
                        self._current_correction_id(record),
                    )
                )
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
    def _context_fragment(record: Record, reason: ActivationReason) -> ContextFragment:
        return ContextFragment(
            fragment_id=record.fragment_id,
            content=record.content,
            source_kind=record.source_kind,
            source_ref=record.source_ref,
            reasons=(reason,),
            verified_for_current_case=False,
            correction=record.correction,
            correction_source_ref=record.correction_source_ref,
        )

    @staticmethod
    def context_for(
        result: QueryResult | None, *, reminders: tuple[ReminderNotice, ...] = (),
        associations: tuple[ContextFragment, ...] = (),
    ) -> tuple[ContextFragment, ...]:
        """Merge same-turn fragments without dropping any applicable reason."""
        fragments = {
            record.fragment_id: MemoryStore._context_fragment(record, ActivationReason.SEARCH)
            for record in result.matches
        } if result is not None and result.status is QueryStatus.COMPLETE_WITH_MATCHES else {}
        reminder_context = (notice.context for notice in reminders if notice.context is not None)
        for item in (*reminder_context, *associations):
            previous = fragments.get(item.fragment_id)
            if previous is not None:
                # Corrections only append, so the longer chain is the newer snapshot.
                base = max((previous, item), key=MemoryStore._correction_depth)
                item = replace(base, reasons=tuple(dict.fromkeys(previous.reasons + item.reasons)))
            fragments[item.fragment_id] = item
        return tuple(fragments.values())

    def explain(self, trace_id: str, fragment_id: str) -> DecisionExplanation:
        decision = self._connection.execute(
            "SELECT * FROM search_decisions WHERE trace_id = ?", (trace_id,)
        ).fetchone()
        if decision is None:
            raise KeyError(trace_id)
        item = self._connection.execute(
            """SELECT score, selected, reason, correction_id FROM search_decision_items
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
            correction_id=item["correction_id"] if item is not None else None,
        )

    def _save_decision(
        self,
        trace_id: str,
        turn_id: str | None,
        query: str,
        scope: str | None,
        status: QueryStatus,
        evaluations: tuple[tuple[str, float, bool, str, str | None], ...]
        | list[tuple[str, float, bool, str, str | None]],
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
                   (trace_id, fragment_id, score, selected, reason, correction_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    (trace_id, fragment_id, score, int(selected), reason, correction_id)
                    for fragment_id, score, selected, reason, correction_id in evaluations
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
    def _correction_depth(record: Record | ContextFragment) -> int:
        depth = 0
        while record.correction is not None:
            record = record.correction
            depth += 1
        return depth

    @staticmethod
    def _current_correction_id(record: Record) -> str | None:
        if record.correction is None:
            return None
        while record.correction is not None:
            record = record.correction
        return record.fragment_id

    def _with_correction(self, record: Record) -> Record:
        chain = []
        while True:
            replacement = self._connection.execute(
                """SELECT fragments.*, corrections.source_ref AS correction_source_ref
                   FROM corrections JOIN fragments
                     ON fragments.fragment_id = corrections.replacement_fragment_id
                   WHERE corrections.old_fragment_id = ?""",
                (record.fragment_id,),
            ).fetchone()
            if replacement is None:
                break
            chain.append((record, replacement["correction_source_ref"]))
            record = self._to_record(replacement)
        for prior, source_ref in reversed(chain):
            record = replace(prior, correction=record, correction_source_ref=source_ref)
        return record

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
