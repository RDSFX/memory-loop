import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from memory_loop import (
    ActionBlocked, ActivationReason, AssociationRule, MemoryStore, QueryStatus,
    SituationEvent, SourceKind, TurnEngine,
)


class TurnReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.database = Path(self.temp.name) / "memory.sqlite3"
        self.store = MemoryStore(self.database, authorizer=lambda who, op, scope: who == "user")
        self.addCleanup(self.store.close)
        self.old = self.store.record(
            "月报导出卡住；关闭目标工作簿后重试成功。", SourceKind.OBSERVATION,
            source_ref="run:export-17", claim_id="claim:export-17",
        )
        self.rule = AssociationRule(
            self.old.fragment_id, "export", "stalled_last_step", "rename_succeeded",
            "check_original_open", "这次原文件也开着吗？", max_gap_turns=2,
        )
        self.engine = TurnEngine(self.store, (self.rule,))

    def turn(self, name, cues=(), *, session="one", situation="export", **kwargs):
        return self.engine.process_turn(
            session, name, scope="default", situation=situation, cues=cues, **kwargs,
        )

    def explain(self, name, *, session="one", fragment_id=None):
        return self.engine.explain(session, name, fragment_id or self.old.fragment_id, scope="default")

    def prime(self, *, session="one"):
        return self.turn("first", ("stalled_last_step",), session=session)

    def test_a07_weak_touch_survives_intervening_turn_and_only_suggests_question(self):
        self.assertEqual(self.prime().context, ())
        self.assertEqual(self.explain("first").association, "weak_touch")
        self.assertEqual(self.turn("aside", situation="unrelated").context, ())
        result = self.turn("last", ("rename_succeeded",))
        self.assertEqual(result.associations[0].question, "这次原文件也开着吗？")
        self.assertEqual(result.context[0].reasons, (ActivationReason.ASSOCIATION,))
        self.assertEqual(result.context[0].source_ref, "run:export-17")
        self.assertFalse(result.context[0].verified_for_current_case)
        explanation = self.explain("last")
        self.assertEqual(explanation.cue_evidence, (
            ("first", "stalled_last_step"), ("last", "rename_succeeded"),
        ))
        self.assertTrue(explanation.presented)
        self.assertFalse(explanation.mentioned)
        self.engine.mark_mentioned("one", "last", self.old.fragment_id, scope="default")
        self.assertTrue(self.explain("last").mentioned)
        self.assertEqual(self.store.independent_source_count("claim:export-17"), 1)

    def test_a07_second_alone_expiry_wrong_situation_and_existing_step_stay_silent(self):
        self.prime()
        cases = (
            ("different_session", "other", "export", (), "no_candidate"),
            ("approval", "one", "approval", (), "irrelevant"),
            ("planned", "one", "export", ("check_original_open",), "next_step_unchanged"),
            ("expired", "one", "export", (), "expired"),
        )
        for name, session, situation, planned, expected in cases:
            with self.subTest(name=name):
                result = self.turn(
                    name, ("rename_succeeded",), session=session, situation=situation,
                    planned_steps=planned,
                )
                self.assertEqual(result.context, ())
                self.assertEqual(result.associations, ())
                self.assertEqual(self.explain(name, session=session).association, expected)

    def test_a07_rule_changes_do_not_reuse_an_old_weak_touch(self):
        self.prime()
        self.engine = TurnEngine(self.store, (replace(self.rule, next_step="check_destination"),))
        self.assertEqual(self.turn("changed", ("rename_succeeded",)).context, ())
        self.assertEqual(self.explain("changed").association, "no_candidate")
        self.assertEqual(self.explain("first").association, "weak_touch")

    def test_a08_all_reasons_and_correction_survive_deduplication(self):
        correction = self.store.record(
            "后来核实：重试前还更换了输出目录。", SourceKind.USER_STATEMENT,
            source_ref="turn:corrected",
        )
        self.store.link_correction(
            self.old.fragment_id, correction.fragment_id, principal="user", source_ref="turn:link",
        )
        reminder = self.store.create_reminder(
            "排查工作簿是否打开。", trigger="started", principal="user",
            source_ref="turn:reminder", fragment_id=self.old.fragment_id,
        )
        self.prime()
        deliver = Mock(return_value=True)
        result = self.turn(
            "all", ("rename_succeeded",), query="月报导出卡住",
            event=SituationEvent("started", "case:1"), deliver=deliver,
        )
        self.assertEqual(len(result.context), 1)
        self.assertEqual(result.context[0].reasons, (
            ActivationReason.SEARCH, ActivationReason.REMINDER, ActivationReason.ASSOCIATION,
        ))
        self.assertEqual(result.context[0].correction.fragment_id, correction.fragment_id)
        self.assertEqual(self.explain("all").correction_id, correction.fragment_id)
        deliver.assert_called_once()
        self.store.revoke_reminder(reminder.reminder_id, principal="user")
        later = self.turn(
            "no_reminder", ("rename_succeeded",), query="月报导出卡住",
            event=SituationEvent("started", "case:2"), deliver=deliver,
        )
        self.assertEqual(later.context[0].reasons, (ActivationReason.SEARCH, ActivationReason.ASSOCIATION))
        deliver.assert_called_once()

    def test_a08_budget_never_splits_a_fragment_from_its_correction(self):
        correction = self.store.record(
            "重新核实后，更换输出目录才是关键。", SourceKind.USER_STATEMENT,
            source_ref="turn:correction",
        )
        self.store.link_correction(
            self.old.fragment_id, correction.fragment_id, principal="user", source_ref="turn:link",
        )
        full = self.turn("full", query="月报导出卡住")
        self.assertEqual(len(full.context), 1)
        exact = self.turn("exact", query="月报导出卡住", context_budget_bytes=len(full.context_payload))
        self.assertEqual(exact.context_payload, full.context_payload)
        short = self.turn("short", query="月报导出卡住", context_budget_bytes=len(full.context_payload) - 1)
        self.assertEqual(short.context_payload, b"")
        self.assertEqual(short.search.matches[0].correction.fragment_id, correction.fragment_id)
        self.assertTrue(self.explain("short").budget_blocked)

    def test_a09_review_snapshots_budget_failure_and_not_yet_stored(self):
        self.prime()
        blocked = self.turn("blocked", ("rename_succeeded",), context_budget_bytes=0)
        self.assertEqual(blocked.context_payload, b"")
        self.assertFalse(self.explain("blocked").presented)
        self.assertEqual(self.explain("blocked").association, "budget_blocked")
        with patch.object(self.engine, "_associations", side_effect=sqlite3.OperationalError):
            failed = self.turn("failed", ("rename_succeeded",), query="月报导出卡住")
        self.assertTrue(failed.association_failed)
        self.assertEqual(failed.search.status, QueryStatus.COMPLETE_WITH_MATCHES)
        self.assertTrue(self.explain("failed").presented)
        self.assertEqual(self.explain("failed").association, "failed")
        later = self.store.record("海边旅行", SourceKind.USER_STATEMENT, source_ref="turn:later")
        reopened = MemoryStore(self.database)
        self.addCleanup(reopened.close)
        explanation = TurnEngine(reopened).explain("one", "blocked", later.fragment_id, scope="default")
        self.assertEqual(explanation.association, "not_yet_stored")
        prior = TurnEngine(reopened).explain("one", "blocked", self.old.fragment_id, scope="default")
        self.assertEqual(prior.association, "budget_blocked")
        self.assertEqual(prior.ruleset, "two-cue-v1")

    def test_a09_optional_failure_does_not_skip_reminder_delivery(self):
        self.store.create_reminder(
            "检查目标工作簿。", trigger="started", principal="user",
            source_ref="turn:reminder", fragment_id=self.old.fragment_id,
        )
        deliver = Mock(return_value=True)
        with patch.object(self.engine, "_associations", side_effect=sqlite3.OperationalError):
            failed = self.turn(
                "failed", ("rename_succeeded",), context_budget_bytes=0,
                event=SituationEvent("started", "case:1"), deliver=deliver,
            )
        self.assertTrue(failed.association_failed)
        self.assertEqual(failed.context, ())
        self.assertEqual(failed.reminders[0].status, "delivered")
        self.assertEqual(self.explain("failed").reminder_statuses, ("delivered",))
        deliver.assert_called_once()

    def test_a10_unrelated_growth_does_not_increase_context_or_change_obligations(self):
        self.store.create_reminder(
            "检查工作簿。", trigger="started", principal="user", source_ref="turn:reminder",
            fragment_id=self.old.fragment_id,
        )
        self.store.create_constraint(
            "月报不得外发。", action="send_report", principal="user", source_ref="turn:constraint",
        )
        snapshots = []
        for start, end in ((0, 10), (10, 100)):
            for index in range(start, end):
                self.store.record(
                    f"海边旅行住旅馆 {index}", SourceKind.USER_STATEMENT, source_ref=f"trip:{index}",
                )
            session = f"batch:{end}"
            self.prime(session=session)
            deliver = Mock(return_value=True)
            result = self.turn(
                "all", ("rename_succeeded",), session=session, query="月报导出卡住",
                event=SituationEvent("started", session), deliver=deliver,
            )
            self.assertEqual(result.reminders[0].status, "delivered")
            deliver.assert_called_once()
            self.assertEqual(result.search.matches[0].source_ref, "run:export-17")
            execute = Mock()
            with self.assertRaises(ActionBlocked):
                self.store.run_action("send_report", scope="default", execute=execute)
            execute.assert_not_called()
            snapshots.append((len(result.context), result.context_payload))
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(snapshots[0][0], 1)
        deliver = Mock(return_value=True)
        empty = self.turn(
            "zero", query="月报导出卡住", context_budget_bytes=0,
            event=SituationEvent("started", "zero"), deliver=deliver,
        )
        self.assertEqual(empty.context, ())
        self.assertEqual(empty.search.status, QueryStatus.COMPLETE_WITH_MATCHES)
        self.assertEqual(empty.reminders[0].status, "delivered")
        deliver.assert_called_once()
        self.engine.mark_mentioned("one", "zero", self.old.fragment_id, scope="default")
        self.assertTrue(self.explain("zero").mentioned)
        self.assertFalse(self.explain("zero").presented)
        self.assertEqual(self.turn("idle", situation="unrelated").context_payload, b"")
