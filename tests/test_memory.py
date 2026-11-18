import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory_loop import ActivationReason, MemoryStore, QueryStatus, SourceKind


class MemoryStoreAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database = Path(self.temp_dir.name) / "memory.sqlite3"
        self.store = MemoryStore(self.database)

    def tearDown(self):
        self.store.close()
        self.temp_dir.cleanup()

    def test_a01_persists_distinct_sources_across_reopen(self):
        observed = self.store.record(
            "关闭已打开的目标工作簿后，月报导出重试成功。",
            SourceKind.OBSERVATION,
            source_ref="run:export-17",
        )
        inference = self.store.record(
            "文件可能被占用。",
            SourceKind.AGENT_INFERENCE,
            source_ref="turn:18",
        )
        self.store.close()
        self.store = MemoryStore(self.database)

        restored_observation = self.store.get(observed.fragment_id)
        restored_inference = self.store.get(inference.fragment_id)
        self.assertEqual(restored_observation.source_kind, SourceKind.OBSERVATION)
        self.assertEqual(restored_observation.source_ref, "run:export-17")
        self.assertEqual(restored_inference.source_kind, SourceKind.AGENT_INFERENCE)
        self.assertEqual(restored_inference.source_ref, "turn:18")

    def test_a01_rejects_missing_source(self):
        with self.assertRaises(ValueError):
            self.store.record("来源不可缺失", SourceKind.OBSERVATION, source_ref=" ")

    def test_a01_preserves_user_and_external_source_categories(self):
        for source_kind, source_ref in (
            (SourceKind.USER_STATEMENT, "turn:user-1"),
            (SourceKind.EXTERNAL_MATERIAL, "doc:procedure-1"),
        ):
            with self.subTest(source_kind=source_kind):
                record = self.store.record(
                    "导出前检查目标文件。",
                    source_kind,
                    source_ref=source_ref,
                )
                restored = self.store.get(record.fragment_id)
                self.assertEqual(restored.source_kind, source_kind)
                self.assertEqual(restored.source_ref, source_ref)

    def test_a02_searches_inactive_memory_and_preserves_provenance(self):
        observation = self.store.record(
            "月报导出卡住；关闭已打开的目标工作簿后，重试成功。",
            SourceKind.OBSERVATION,
            source_ref="run:export-17",
        )
        self.store.record(
            "文件可能被占用。",
            SourceKind.AGENT_INFERENCE,
            source_ref="turn:18",
        )

        result = self.store.search("上次月报导出卡住怎么恢复")

        self.assertEqual(result.status, QueryStatus.COMPLETE_WITH_MATCHES)
        self.assertTrue(result.matches)
        self.assertEqual(result.matches[0].fragment_id, observation.fragment_id)
        self.assertEqual(result.matches[0].source_kind, SourceKind.OBSERVATION)
        self.assertEqual(result.matches[0].source_ref, "run:export-17")

    def test_a08_search_context_includes_provenance_reason_and_unverified_state(self):
        self.store.record(
            "月报导出卡住；关闭工作簿后重试成功。",
            SourceKind.OBSERVATION,
            source_ref="run:export-17",
        )
        result = self.store.search("月报导出怎么恢复")

        context = self.store.context_for(result)

        self.assertEqual(len(context), 1)
        self.assertEqual(context[0].source_kind, SourceKind.OBSERVATION)
        self.assertEqual(context[0].source_ref, "run:export-17")
        self.assertEqual(context[0].reasons, (ActivationReason.SEARCH,))
        self.assertFalse(context[0].verified_for_current_case)

    def test_a02_only_complete_empty_search_means_no_record(self):
        missing = self.store.search("不存在的项目经历")
        self.assertEqual(missing.status, QueryStatus.COMPLETE_NO_MATCHES)

    def test_a02_rejects_incomplete_search_instead_of_claiming_no_record(self):
        with patch.object(self.store, "_scan_scope", return_value=None):
            result = self.store.search("月报怎么恢复")
        self.assertEqual(result.status, QueryStatus.INCOMPLETE)
        self.assertEqual(result.matches, ())

    def test_a02_database_failure_is_not_a_no_record_answer(self):
        self.store.close()

        result = self.store.search("月报怎么恢复")

        self.assertEqual(result.status, QueryStatus.FAILED)
        self.assertEqual(result.matches, ())

    def test_a09_search_failure_is_traceable_when_storage_is_available(self):
        with patch.object(self.store, "_scan_scope", side_effect=sqlite3.OperationalError):
            result = self.store.search("月报怎么恢复", turn_id="turn:failed-search")

        self.assertEqual(result.status, QueryStatus.FAILED)
        self.assertIsNotNone(result.trace_id)
        explanation = self.store.explain(result.trace_id, "missing-fragment")
        self.assertEqual(explanation.query_status, QueryStatus.FAILED)
        self.assertEqual(explanation.turn_id, "turn:failed-search")

    def test_a02_empty_query_requests_clarification(self):
        result = self.store.search("  ")
        self.assertEqual(result.status, QueryStatus.NEEDS_CLARIFICATION)

    def test_a03_repeated_agent_inferences_do_not_add_independent_evidence(self):
        observation = self.store.record(
            "关闭工作簿后重试成功。",
            SourceKind.OBSERVATION,
            source_ref="run:export-17",
            claim_id="claim:file-lock",
        )
        repeated_reference = self.store.record(
            "关闭工作簿后重试成功。",
            SourceKind.OBSERVATION,
            source_ref="run:export-17",
            claim_id="claim:file-lock",
        )
        inference_ids = []
        for turn in ("turn:18", "turn:19", "turn:20"):
            inference = self.store.record(
                "文件可能被占用。",
                SourceKind.AGENT_INFERENCE,
                source_ref=turn,
                claim_id="claim:file-lock",
                supports=(observation.fragment_id,),
            )
            inference_ids.append(inference.fragment_id)

        self.assertEqual(self.store.independent_source_count("claim:file-lock"), 1)
        self.assertEqual(
            self.store.get(inference_ids[0]).supports,
            (observation.fragment_id,),
        )
        self.assertEqual(repeated_reference.claim_id, observation.claim_id)

        self.store.record(
            "另一场排查中关闭工作簿后重试成功。",
            SourceKind.OBSERVATION,
            source_ref="run:export-22",
            claim_id="claim:file-lock",
        )
        self.assertEqual(self.store.independent_source_count("claim:file-lock"), 2)

    def test_a09_search_decision_can_be_explained_after_later_writes(self):
        relevant = self.store.record(
            "月报导出卡住；关闭工作簿后重试成功。",
            SourceKind.OBSERVATION,
            source_ref="run:export-17",
        )
        unrelated = self.store.record(
            "海边旅行订了周五的酒店。",
            SourceKind.USER_STATEMENT,
            source_ref="turn:vacation",
        )

        result = self.store.search("月报导出怎么恢复", turn_id="turn:ask-1")
        self.store.record(
            "后来新增的无关信息。",
            SourceKind.USER_STATEMENT,
            source_ref="turn:later",
        )

        self.assertIsNotNone(result.trace_id)
        selected = self.store.explain(result.trace_id, relevant.fragment_id)
        rejected = self.store.explain(result.trace_id, unrelated.fragment_id)
        self.assertEqual(selected.reason, "selected")
        self.assertTrue(selected.selected)
        self.assertEqual(rejected.reason, "below_threshold")
        self.assertFalse(rejected.selected)
        self.assertEqual(selected.turn_id, "turn:ask-1")
        self.store.close()
        self.store = MemoryStore(self.database)
        self.assertEqual(
            self.store.explain(result.trace_id, relevant.fragment_id).reason,
            "selected",
        )

    def test_store_migrates_the_initial_fragment_schema(self):
        self.store.close()
        self.database.unlink()
        connection = sqlite3.connect(self.database)
        connection.execute(
            """CREATE TABLE fragments (
               fragment_id TEXT PRIMARY KEY,
               content TEXT NOT NULL,
               source_kind TEXT NOT NULL,
               source_ref TEXT NOT NULL,
               recorded_at TEXT NOT NULL,
               scope TEXT NOT NULL
            )"""
        )
        connection.commit()
        connection.close()

        self.store = MemoryStore(self.database)
        record = self.store.record(
            "先前数据库已迁移。",
            SourceKind.USER_STATEMENT,
            source_ref="turn:migration",
        )
        self.assertEqual(self.store.get(record.fragment_id).supports, ())


if __name__ == "__main__":
    unittest.main()
