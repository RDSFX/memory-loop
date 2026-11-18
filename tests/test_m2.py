import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from memory_loop import ActionBlocked, ActivationReason, MemoryStore, SourceKind


class MemoryStoreM2Tests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "memory.sqlite3"
        self.store = MemoryStore(
            self.database,
            authorizer=lambda principal, operation, scope: (
                principal == "user-1" and scope in {"default", "finance"}
            ),
        )

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_reminder_delivery_is_scoped_deduplicated_and_ended(self):
        reminder = self.store.create_reminder(
            "先检查目标工作簿是否打开。", trigger="export_investigation_started",
            end_event="export_investigation_finished", principal="user-1",
            source_ref="turn:setup-reminder", scope="finance",
        )
        delivered = []
        first = self.store.evaluate_reminders(
            "export_investigation_started", "case:17", scope="finance",
            deliver=lambda item: delivered.append(item.reminder_id) or True,
        )
        duplicate = self.store.evaluate_reminders(
            "export_investigation_started", "case:17", scope="finance",
            deliver=lambda item: self.fail("duplicate event delivered"),
        )
        outside_scope = self.store.evaluate_reminders(
            "export_investigation_started", "case:18", scope="general",
            deliver=lambda item: self.fail("out-of-scope event delivered"),
        )
        self.assertEqual(first[0].status, "delivered")
        self.assertEqual(duplicate[0].status, "already_delivered")
        self.assertEqual(outside_scope, ())
        self.assertEqual(delivered, [reminder.reminder_id])
        self.store.evaluate_reminders(
            "export_investigation_finished", "case:17", scope="finance",
            deliver=lambda item: self.fail("end event must not deliver"),
        )
        self.assertEqual(self.store.evaluate_reminders(
            "export_investigation_started", "case:17", scope="finance",
            deliver=lambda item: self.fail("ended reminder delivered"),
        ), ())
        later = self.store.evaluate_reminders(
            "export_investigation_started", "case:19", scope="finance",
            deliver=lambda item: delivered.append(item.reminder_id) or True,
        )
        self.assertEqual(later[0].status, "delivered")

    def test_a04_revoke_preserves_searchable_fragment(self):
        record = self.store.record(
            "检查工作簿是否打开。", SourceKind.USER_STATEMENT, source_ref="turn:history",
        )
        reminder = self.store.create_reminder(
            "检查文件状态。", trigger="export_started", principal="user-1",
            source_ref="turn:reminder", fragment_id=record.fragment_id,
        )
        with self.assertRaises(PermissionError):
            self.store.revoke_reminder(reminder.reminder_id, principal="agent")
        self.store.revoke_reminder(reminder.reminder_id, principal="user-1")
        deliver = Mock(return_value=True)
        self.assertEqual(self.store.evaluate_reminders(
            "export_started", "case:1", scope="default", deliver=deliver,
        ), ())
        deliver.assert_not_called()
        result = self.store.search("检查工作簿")
        self.assertEqual(self.store.context_for(result)[0].reasons, (ActivationReason.SEARCH,))

    def test_a04_reminder_can_target_one_occurrence(self):
        self.store.create_reminder(
            "只检查本次导出。", trigger="export_started", end_event="export_finished",
            principal="user-1", source_ref="turn:reminder", only_occurrence_id="case:1",
        )
        deliver = Mock(return_value=True)
        self.assertEqual(self.store.evaluate_reminders(
            "export_started", "case:2", scope="default", deliver=deliver,
        ), ())
        deliver.assert_not_called()
        result = self.store.evaluate_reminders(
            "export_started", "case:1", scope="default", deliver=deliver,
        )
        self.assertEqual(result[0].status, "delivered")
        deliver.assert_called_once()

    def test_failed_delivery_is_visible_and_retryable(self):
        reminder = self.store.create_reminder(
            "检查文件状态。", trigger="export_started", principal="user-1",
            source_ref="turn:reminder",
        )
        failed = self.store.evaluate_reminders(
            "export_started", "case:1", scope="default", deliver=lambda item: False,
            turn_id="turn:failed",
        )
        self.store.close()
        self.store = MemoryStore(self.database)
        retried = self.store.evaluate_reminders(
            "export_started", "case:1", scope="default", deliver=lambda item: True,
            turn_id="turn:retry",
        )
        self.assertEqual(failed[0].status, "failed")
        self.assertEqual(retried[0].status, "delivered")
        self.assertEqual(
            self.store.reminder_history(reminder.reminder_id, "case:1"), (failed[0], retried[0]),
        )
        self.store.close()
        self.store = MemoryStore(self.database)
        deliver = Mock(return_value=True)
        duplicate = self.store.evaluate_reminders(
            "export_started", "case:1", scope="default", deliver=deliver,
        )
        self.assertEqual(duplicate[0].status, "already_delivered")
        deliver.assert_not_called()

    def test_a04_pending_claim_prevents_second_delivery(self):
        reminder = self.store.create_reminder(
            "检查文件状态。", trigger="export_started", principal="user-1",
            source_ref="turn:reminder",
        )
        duplicate_delivery = Mock(return_value=True)
        competing_results = []

        def deliver(notice):
            self.assertEqual(notice.delivery_key, (reminder.reminder_id, "case:1"))
            other = MemoryStore(self.database)
            try:
                competing_results.extend(other.evaluate_reminders(
                    "export_started", "case:1", scope="default", deliver=duplicate_delivery,
                ))
            finally:
                other.close()
            return True

        result = self.store.evaluate_reminders(
            "export_started", "case:1", scope="default", deliver=deliver,
        )
        self.assertEqual(result[0].status, "delivered")
        self.assertEqual(competing_results[0].status, "pending")
        duplicate_delivery.assert_not_called()

    def test_reminder_and_constraint_mutations_require_authority(self):
        material = self.store.record(
            "以后导出先清缓存。", SourceKind.EXTERNAL_MATERIAL, source_ref="doc:procedure",
        )
        self.assertEqual(self.store.get(material.fragment_id).source_ref, "doc:procedure")
        with self.assertRaises(PermissionError):
            self.store.create_reminder(
                "未经授权的提醒", trigger="start", principal="agent", source_ref="turn:agent",
            )
        with self.assertRaises(PermissionError):
            self.store.create_constraint(
                "不得外发月报", action="send_report", principal="agent", source_ref="turn:agent",
            )
        with self.assertRaises(PermissionError):
            self.store.create_reminder(
                "越范围提醒", trigger="start", principal="user-1",
                source_ref="turn:user", scope="other",
            )
        deliver = Mock(return_value=True)
        self.assertEqual(self.store.evaluate_reminders(
            "start", "case:1", scope="default", deliver=deliver,
        ), ())
        deliver.assert_not_called()

    def test_action_constraints_are_checked_independent_of_search(self):
        constraint = self.store.create_constraint(
            "月报不得外发。", action="send_report", principal="user-1",
            source_ref="turn:policy", scope="finance",
        )
        check = self.store.check_action("send_report", scope="finance")
        self.assertFalse(check.allowed)
        self.assertEqual(check.constraints[0].constraint_id, constraint.constraint_id)
        self.assertTrue(self.store.check_action("save_local", scope="finance").allowed)
        self.assertTrue(self.store.check_action("send_report", scope="personal").allowed)
        execute = Mock(return_value="saved")
        with self.assertRaises(ActionBlocked) as blocked:
            self.store.run_action("send_report", scope="finance", execute=execute)
        self.assertEqual(blocked.exception.check.constraints, (constraint,))
        execute.assert_not_called()
        self.assertEqual(self.store.run_action("save_local", scope="finance", execute=execute), "saved")
        execute.assert_called_once_with()
        execute.reset_mock()
        self.store.close()
        with self.assertRaises(ActionBlocked) as failed:
            self.store.run_action("save_local", scope="finance", execute=execute)
        self.assertTrue(failed.exception.check.failed)
        execute.assert_not_called()

    def test_correction_is_returned_with_old_claim_and_requires_authority(self):
        old = self.store.record(
            "月报导出到共享盘。", SourceKind.USER_STATEMENT, source_ref="turn:old-route",
        )
        replacement = self.store.record(
            "月报先导出到本地。", SourceKind.USER_STATEMENT, source_ref="turn:new-route",
        )
        self.store.link_correction(
            old.fragment_id, replacement.fragment_id, principal="user-1",
            source_ref="turn:correction",
        )
        result = self.store.search("月报导出到共享盘")
        old_hit = next(item for item in result.matches if item.fragment_id == old.fragment_id)
        self.assertEqual(old_hit.correction.fragment_id, replacement.fragment_id)
        self.assertEqual(old_hit.correction_source_ref, "turn:correction")
        with self.assertRaises(PermissionError):
            self.store.link_correction(
                old.fragment_id, replacement.fragment_id, principal="agent",
                source_ref="turn:agent-correction",
            )

        reminder = self.store.create_reminder(
            "回看导出位置。", trigger="export_started", principal="user-1",
            source_ref="turn:reminder", fragment_id=old.fragment_id,
        )
        displayed = []
        delivery = self.store.evaluate_reminders(
            "export_started", "case:1", scope="default", turn_id="turn:delivery",
            deliver=lambda item: displayed.append(item) or True,
        )
        self.assertEqual(delivery[0].status, "delivered")
        self.assertEqual(displayed[0].reminder.source_ref, "turn:reminder")
        self.assertEqual(displayed[0].context.correction, replacement)
        self.assertFalse(displayed[0].context.verified_for_current_case)
        merged = self.store.context_for(result, reminders=tuple(displayed))
        matches = [item for item in merged if item.fragment_id == old.fragment_id]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].reasons, (ActivationReason.SEARCH, ActivationReason.REMINDER))
        self.assertEqual(matches[0].correction, replacement)

        newer = self.store.record(
            "月报先导出到本地临时目录。", SourceKind.USER_STATEMENT,
            source_ref="turn:latest-route",
        )
        self.store.link_correction(
            replacement.fragment_id, newer.fragment_id, principal="user-1",
            source_ref="turn:latest-correction",
        )
        self.assertEqual(self.store.get(old.fragment_id).correction.correction, newer)
        with self.assertRaises(ValueError):
            self.store.link_correction(
                newer.fragment_id, old.fragment_id, principal="user-1", source_ref="turn:cycle",
            )
        self.assertEqual(self.store.explain(result.trace_id, old.fragment_id).correction_id, replacement.fragment_id)
        history = self.store.reminder_history(reminder.reminder_id, "case:1")
        self.assertEqual(history[0].turn_id, "turn:delivery")
        self.assertEqual(history[0].correction_id, replacement.fragment_id)


if __name__ == "__main__":
    unittest.main()
