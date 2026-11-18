"""Offline turn orchestration and replay for host-supplied situation cues."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import sqlite3

from .store import (
    ActivationReason, ContextFragment, MemoryStore, QueryResult, QueryStatus,
    ReminderDelivery, ReminderNotice,
)


@dataclass(frozen=True)
class AssociationRule:
    fragment_id: str
    situation: str
    first_cue: str
    second_cue: str
    next_step: str
    question: str
    max_gap_turns: int = 2

    def __post_init__(self) -> None:
        for field in ("fragment_id", "situation", "first_cue", "second_cue", "next_step", "question"):
            MemoryStore._required(getattr(self, field), field)
        if self.first_cue == self.second_cue or type(self.max_gap_turns) is not int or self.max_gap_turns < 1:
            raise ValueError("an association requires distinct cues and at least one turn of retention")


@dataclass(frozen=True)
class SituationEvent:
    name: str
    occurrence_id: str


@dataclass(frozen=True)
class AssociationSuggestion:
    fragment_id: str
    next_step: str
    question: str


@dataclass(frozen=True)
class TurnResult:
    search: QueryResult | None
    reminders: tuple[ReminderDelivery, ...]
    context: tuple[ContextFragment, ...]
    context_payload: bytes
    associations: tuple[AssociationSuggestion, ...]
    association_failed: bool = False
    reminders_failed: bool = False


@dataclass(frozen=True)
class TurnExplanation:
    fragment_id: str
    ruleset: str
    association: str
    cue_evidence: tuple[tuple[str, str], ...]
    available_reasons: tuple[ActivationReason, ...]
    presented: bool
    mentioned: bool
    correction_id: str | None
    search_status: QueryStatus | None
    search_reason: str | None
    reminder_statuses: tuple[str, ...]
    budget_blocked: bool
    reminders_failed: bool


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class TurnEngine:
    """Handle serial host turns within a session and scope; store replay separately."""

    RULESET = "two-cue-v1"

    def __init__(self, store: MemoryStore, rules: tuple[AssociationRule, ...] = ()):
        self.store = store
        self._db = store._connection
        self.rules = tuple(rules)
        if len({rule.fragment_id for rule in self.rules}) != len(self.rules):
            raise ValueError("only one association rule per fragment is supported")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS replay_turns (
                replay_id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                scope TEXT NOT NULL,
                turn_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                ruleset TEXT NOT NULL,
                decision TEXT,
                UNIQUE (session_id, scope, turn_id),
                UNIQUE (session_id, scope, ordinal)
            );
            CREATE TABLE IF NOT EXISTS replay_touches (
                session_id TEXT NOT NULL,
                scope TEXT NOT NULL,
                rule_key TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                turn_id TEXT NOT NULL,
                PRIMARY KEY (session_id, scope, rule_key, ordinal)
            );
            CREATE TABLE IF NOT EXISTS replay_mentions (
                replay_id INTEGER NOT NULL REFERENCES replay_turns(replay_id),
                fragment_id TEXT NOT NULL REFERENCES fragments(fragment_id),
                PRIMARY KEY (replay_id, fragment_id)
            );
            """
        )

    def process_turn(
        self,
        session_id: str,
        turn_id: str,
        *,
        scope: str,
        situation: str,
        cues: tuple[str, ...] = (),
        planned_steps: tuple[str, ...] = (),
        query: str | None = None,
        event: SituationEvent | None = None,
        deliver: Callable[[ReminderNotice], bool] | None = None,
        context_budget_bytes: int = 4096,
    ) -> TurnResult:
        """Return explicit search answers and dispatch required reminders outside the budget."""
        for name, value in (("session_id", session_id), ("turn_id", turn_id),
                            ("scope", scope), ("situation", situation)):
            MemoryStore._required(value, name)
        if type(context_budget_bytes) is not int or context_budget_bytes < 0:
            raise ValueError("context budget must be a nonnegative byte count")
        if event is not None:
            MemoryStore._required(event.name, "event")
            MemoryStore._required(event.occurrence_id, "occurrence_id")
            if not callable(deliver):
                raise ValueError("a situation event requires a visible reminder delivery callback")
        cue_set = frozenset(MemoryStore._required(cue, "cue") for cue in cues)
        steps = frozenset(MemoryStore._required(step, "planned step") for step in planned_steps)
        replay_id, ordinal = self._reserve_turn(session_id, scope, turn_id)

        search = self.store.search(query, scope=scope, turn_id=turn_id) if query is not None else None
        notices: list[ReminderNotice] = []
        reminders: tuple[ReminderDelivery, ...] = ()
        reminders_failed = False
        if event is not None and deliver is not None:
            def dispatch(notice: ReminderNotice) -> bool:
                if deliver(notice) is not True:
                    return False
                notices.append(notice)
                return True

            try:
                reminders = self.store.evaluate_reminders(
                    event.name, event.occurrence_id, scope=scope, deliver=dispatch, turn_id=turn_id,
                )
            except sqlite3.Error:
                reminders_failed = True

        association_failed = False
        try:
            evaluations, candidates, touches = self._associations(
                session_id, scope, turn_id, ordinal, situation, cue_set, steps,
            )
        except sqlite3.Error:
            association_failed = True
            evaluations, candidates, touches = {}, {}, ()

        merged = self.store.context_for(
            search, reminders=tuple(notices), associations=tuple(item for item, _ in candidates.values()),
        )
        context: list[ContextFragment] = []
        payload = bytearray()
        suggestions: list[AssociationSuggestion] = []
        used_steps: set[str] = set()
        evidence = {
            fragment_id: {"association": status, "cue_evidence": cues_used}
            for fragment_id, (status, cues_used) in evaluations.items()
        }
        for item in merged:
            details = evidence.setdefault(item.fragment_id, {})
            suggestion = candidates[item.fragment_id][1] if item.fragment_id in candidates else None
            if suggestion is not None and suggestion.next_step in used_steps:
                details["association"] = "next_step_unchanged"
                suggestion = None
                reasons = tuple(reason for reason in item.reasons if reason is not ActivationReason.ASSOCIATION)
                if not reasons:
                    continue
                item = replace(item, reasons=reasons)
            details["reasons"] = item.reasons
            encoded = (_json(asdict(item)) + "\n").encode("utf-8")
            if len(encoded) > context_budget_bytes - len(payload):
                details["budget_blocked"] = True
                if suggestion is not None:
                    details["association"] = "budget_blocked"
                continue
            if item.correction is not None:
                details["correction_id"] = self.store._current_correction_id(item.correction) or item.correction.fragment_id
            details["presented"] = True
            context.append(item)
            payload.extend(encoded)
            if suggestion is not None:
                details["association"] = "presented"
                suggestions.append(suggestion)
                used_steps.add(suggestion.next_step)

        for delivery in reminders:
            if delivery.fragment_id is not None:
                details = evidence.setdefault(delivery.fragment_id, {})
                details.setdefault("reminder_statuses", []).append(delivery.status)
                if delivery.status == "delivered":
                    details["reminder_delivered"] = True
                    if delivery.correction_id is not None:
                        details.setdefault("correction_id", delivery.correction_id)
        decision = {
            "fragments": evidence,
            "association_failed": association_failed,
            "reminders_failed": reminders_failed,
            "search_status": search.status if search is not None else None,
            "search_trace_id": search.trace_id if search is not None else None,
            "budget": context_budget_bytes,
        }
        with self._db:
            self._db.execute("UPDATE replay_turns SET decision = ? WHERE replay_id = ?", (_json(decision), replay_id))
            self._db.executemany(
                "INSERT INTO replay_touches VALUES (?, ?, ?, ?, ?)",
                ((session_id, scope, key, ordinal, turn_id) for key in touches),
            )
        return TurnResult(
            search, reminders, tuple(context), bytes(payload), tuple(suggestions),
            association_failed, reminders_failed,
        )

    def _reserve_turn(self, session_id: str, scope: str, turn_id: str) -> tuple[int, int]:
        with self._db:
            self._db.execute("BEGIN IMMEDIATE")
            if self._db.execute(
                "SELECT 1 FROM replay_turns WHERE session_id = ? AND scope = ? AND turn_id = ?",
                (session_id, scope, turn_id),
            ).fetchone() is not None:
                raise ValueError("turn ID has already been used in this session and scope")
            ordinal = self._db.execute(
                "SELECT COALESCE(MAX(ordinal), 0) + 1 FROM replay_turns WHERE session_id = ? AND scope = ?",
                (session_id, scope),
            ).fetchone()[0]
            row = self._db.execute(
                """INSERT INTO replay_turns (session_id, scope, turn_id, ordinal, started_at, ruleset)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, scope, turn_id, ordinal, datetime.now(timezone.utc).isoformat(), self.RULESET),
            )
            return row.lastrowid, ordinal

    def _associations(
        self, session_id: str, scope: str, turn_id: str, ordinal: int,
        situation: str, cues: frozenset[str], planned_steps: frozenset[str],
    ) -> tuple[
        dict[str, tuple[str, tuple[tuple[str, str], ...]]],
        dict[str, tuple[ContextFragment, AssociationSuggestion]],
        tuple[str, ...],
    ]:
        decisions = {}
        candidates = {}
        touches = []
        for rule in self.rules:
            if rule.first_cue not in cues and rule.second_cue not in cues:
                continue
            try:
                record = self.store.get(rule.fragment_id)
            except KeyError:
                continue
            if record.scope != scope:
                continue
            fragment_id = rule.fragment_id
            current = tuple((turn_id, cue) for cue in (rule.first_cue, rule.second_cue) if cue in cues)
            if situation != rule.situation:
                decisions[fragment_id] = ("irrelevant", current)
                continue
            rule_key = _json((self.RULESET, asdict(rule)))
            if rule.first_cue in cues:
                touches.append(rule_key)
            if rule.second_cue not in cues:
                decisions[fragment_id] = ("weak_touch", current)
                continue
            first = self._db.execute(
                """SELECT ordinal, turn_id FROM replay_touches
                   WHERE session_id = ? AND scope = ? AND rule_key = ? AND ordinal < ?
                   ORDER BY ordinal DESC LIMIT 1""",
                (session_id, scope, rule_key, ordinal),
            ).fetchone()
            if first is None:
                decisions[fragment_id] = ("weak_touch" if rule.first_cue in cues else "no_candidate", current)
                continue
            evidence = ((first["turn_id"], rule.first_cue), (turn_id, rule.second_cue))
            if ordinal - first["ordinal"] > rule.max_gap_turns:
                decisions[fragment_id] = ("expired", evidence)
            elif rule.next_step in planned_steps:
                decisions[fragment_id] = ("next_step_unchanged", evidence)
            else:
                decisions[fragment_id] = ("eligible", evidence)
                candidates[fragment_id] = (
                    self.store._context_fragment(record, ActivationReason.ASSOCIATION),
                    AssociationSuggestion(fragment_id, rule.next_step, rule.question),
                )
        return decisions, candidates, tuple(touches)

    def _read_turn(self, session_id: str, scope: str, turn_id: str) -> tuple[sqlite3.Row, dict]:
        row = self._db.execute(
            "SELECT * FROM replay_turns WHERE session_id = ? AND scope = ? AND turn_id = ?",
            (session_id, scope, turn_id),
        ).fetchone()
        if row is None:
            raise KeyError((session_id, scope, turn_id))
        if row["decision"] is None:
            raise RuntimeError("turn did not finish; no complete decision was recorded")
        return row, json.loads(row["decision"])

    def explain(self, session_id: str, turn_id: str, fragment_id: str, *, scope: str) -> TurnExplanation:
        row, decision = self._read_turn(session_id, scope, turn_id)
        item = decision["fragments"].get(fragment_id, {})
        association = item.get("association", "failed" if decision["association_failed"] else "no_candidate")
        record = self._db.execute(
            "SELECT recorded_at FROM fragments WHERE fragment_id = ? AND scope = ?", (fragment_id, scope),
        ).fetchone()
        if not item and (record is None or record["recorded_at"] > row["started_at"]):
            association = "not_yet_stored"
        mentioned = self._db.execute(
            "SELECT 1 FROM replay_mentions WHERE replay_id = ? AND fragment_id = ?",
            (row["replay_id"], fragment_id),
        ).fetchone() is not None
        trace_id = decision["search_trace_id"]
        search_reason = self.store.explain(trace_id, fragment_id).reason if trace_id is not None else None
        return TurnExplanation(
            fragment_id, row["ruleset"], association, tuple(tuple(cue) for cue in item.get("cue_evidence", ())),
            tuple(ActivationReason(reason) for reason in item.get("reasons", ())),
            item.get("presented", False), mentioned, item.get("correction_id"),
            QueryStatus(decision["search_status"]) if decision["search_status"] is not None else None,
            search_reason, tuple(item.get("reminder_statuses", ())),
            item.get("budget_blocked", False), decision["reminders_failed"],
        )

    def mark_mentioned(self, session_id: str, turn_id: str, fragment_id: str, *, scope: str) -> None:
        """Record host-confirmed user mention, not merely context injection or delivery."""
        row, decision = self._read_turn(session_id, scope, turn_id)
        item = decision["fragments"].get(fragment_id, {})
        explicit_answer = ActivationReason.SEARCH.value in item.get("reasons", ())
        if not (item.get("presented") or item.get("reminder_delivered") or explicit_answer):
            raise ValueError("fragment was not available through a presentation, search answer or reminder")
        with self._db:
            self._db.execute("INSERT OR IGNORE INTO replay_mentions VALUES (?, ?)", (row["replay_id"], fragment_id))
