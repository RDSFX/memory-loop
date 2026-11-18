import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory_loop import MemoryStore, QueryStatus, SourceKind


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

    def test_a02_empty_query_requests_clarification(self):
        result = self.store.search("  ")
        self.assertEqual(result.status, QueryStatus.NEEDS_CLARIFICATION)


if __name__ == "__main__":
    unittest.main()
