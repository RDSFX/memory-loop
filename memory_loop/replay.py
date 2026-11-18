"""Run the fixed A10 workload offline with `python -m memory_loop.replay`."""

from __future__ import annotations

import argparse
import json
from statistics import median
from time import perf_counter

from .store import ActionBlocked, MemoryStore, QueryStatus, SourceKind
from .turn import AssociationRule, SituationEvent, TurnEngine


_TRACE_TABLES = (
    "search_decisions", "search_decision_items", "reminder_deliveries",
    "replay_turns", "replay_touches", "replay_mentions",
)


def _trace_size(store: MemoryStore) -> tuple[int, int]:
    rows, size = 0, 0
    for table in _TRACE_TABLES:
        for row in store._connection.execute(f"SELECT * FROM {table}"):
            rows += 1
            size += len(json.dumps(tuple(row), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    return rows, size


def _case(unrelated: int, repeats: int) -> dict:
    store = MemoryStore(":memory:", authorizer=lambda who, op, scope: who == "user" and scope == "reports")
    try:
        observation = store.record(
            "\u6708\u62a5\u5bfc\u51fa\u5361\u4f4f\uff1b\u5173\u95ed\u5de5\u4f5c\u7c3f\u540e\u91cd\u8bd5\u6210\u529f\u3002",
            SourceKind.OBSERVATION, source_ref="run:export-17", scope="reports",
        )
        for index in range(unrelated):
            store.record(
                f"Travel itinerary for trip {index}", SourceKind.USER_STATEMENT,
                source_ref=f"trip:{index}", scope="reports",
            )
        store.create_reminder(
            "Check whether the target workbook is open.", trigger="investigate", principal="user",
            source_ref="turn:reminder", scope="reports", fragment_id=observation.fragment_id,
        )
        store.create_constraint(
            "Do not send the monthly report externally.", action="send_report", principal="user",
            source_ref="turn:constraint", scope="reports",
        )
        engine = TurnEngine(store, (AssociationRule(
            observation.fragment_id, "export", "stalled_last_step", "rename_succeeded",
            "check_original_open", "Is the original file open this time?",
        ),))
        engine.process_turn("fixed", "first", scope="reports", situation="export", cues=("stalled_last_step",))
        engine.process_turn("fixed", "aside", scope="reports", situation="unrelated")
        query = "\u6708\u62a5\u5bfc\u51fa\u5361\u4f4f"
        delivered = []

        def deliver(notice):
            delivered.append(notice.delivery_key)
            return True

        result = engine.process_turn(
            "fixed", "last", scope="reports", situation="export", cues=("rename_succeeded",),
            query=query, event=SituationEvent("investigate", "case:1"), deliver=deliver,
        )
        side_effects = []
        action_blocked = False
        try:
            store.run_action("send_report", scope="reports", execute=lambda: side_effects.append("send"))
        except ActionBlocked as error:
            action_blocked = not error.check.failed and bool(error.check.constraints)
        if result.search.status is not QueryStatus.COMPLETE_WITH_MATCHES:
            raise RuntimeError(f"unexpected replay search status: {result.search.status}")

        trace_rows, trace_utf8_bytes = _trace_size(store)
        timings = []
        for _ in range(repeats):
            start = perf_counter()
            store.search(query, scope="reports")
            timings.append((perf_counter() - start) * 1000)
        return {
            "unrelated_fragments": unrelated,
            "search_sources": [match.source_ref for match in result.search.matches],
            "reminders_delivered": len(delivered),
            "action_blocked": action_blocked and not side_effects,
            "association_suggestions": len(result.associations),
            "context_fragments": len(result.context),
            "context_utf8_bytes": len(result.context_payload),
            "context_reasons": [[reason.value for reason in item.reasons] for item in result.context],
            "search_median_ms": round(median(timings), 3),
            "trace_rows": trace_rows,
            "trace_logical_utf8_bytes": trace_utf8_bytes,
        }
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="number of unrelated fragments in the baseline")
    parser.add_argument("--repeats", type=int, default=7, help="timed explicit searches per size")
    args = parser.parse_args()
    if args.n < 1 or args.repeats < 1:
        parser.error("n and repeats must be positive")
    rows = [_case(size, args.repeats) for size in (args.n, 10 * args.n)]
    expected = {
        "search_sources": ["run:export-17"], "reminders_delivered": 1,
        "action_blocked": True, "association_suggestions": 1, "context_fragments": 1,
        "context_reasons": [["search", "reminder", "association"]],
    }
    passed = (
        all(all(row[key] == value for key, value in expected.items()) for row in rows)
        and rows[0]["context_utf8_bytes"] == rows[1]["context_utf8_bytes"]
    )
    print(json.dumps({"passed": passed, "cases": rows}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
