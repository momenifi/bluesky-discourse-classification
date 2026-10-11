import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from bluesky_discourse.validate_data import validate
from bluesky_discourse.prepare_data import connected_groups, prepare


class ValidationTests(unittest.TestCase):
    def fixture(self, folder, duplicate=False):
        rows = []
        for index in range(500):
            group = ("paired_thread_source" if index < 200 else
                     "paired_thread_reply" if index < 400 else "independent_reply")
            rows.append({
                "annotation_id": f"gold_{index:04d}", "sample_group": group,
                "item_type": "original_post" if index < 200 else "reply",
                "language_hint": "en", "annotation_text": "Example text",
                "context_text": "Context" if index >= 200 else "",
                "uri": f"post:{index}", "root_uri": f"root:{index}",
                "source_uri": f"root:{index}", "parent_uri": "",
                "final_label": "Explanation",
                "final_label_source": "human_majority" if index < 442 else
                    "human_adjudication_with_llm_support",
            })
        if duplicate:
            rows[-1]["uri"] = rows[-2]["uri"]
        csv_path = folder / "gold.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        manifest_path = folder / "manifest.json"
        manifest_path.write_text(json.dumps({
            "rows": 500, "unique_annotation_ids": 500,
            "final_label_distribution": {"Explanation": 500},
            "majority_resolved_rows": 442, "human_adjudicated_rows": 58,
            "outputs": {"csv": {"sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest()}},
        }), encoding="utf-8")
        return csv_path, manifest_path

    def test_valid_release_with_originals_without_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp))
            report = validate(*paths)
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["unique_post_uris"], 500)

    def test_changed_input_fails_release_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path, manifest_path = self.fixture(Path(tmp))
            with csv_path.open("ab") as stream:
                stream.write(b"\n")
            report = validate(csv_path, manifest_path)
            self.assertFalse(report["valid"])
            self.assertTrue(any("SHA-256" in error for error in report["errors"]))

    def test_duplicate_post_in_different_threads_blocks_splitting(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = validate(*self.fixture(Path(tmp), duplicate=True))
            self.assertFalse(report["valid"])
            self.assertEqual(report["unique_post_uris"], 499)
            self.assertTrue(any("different root_uri" in error for error in report["errors"]))

    def test_preparation_deduplicates_and_preserves_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = self.fixture(Path(tmp), duplicate=True)
            original = paths[0].read_bytes()
            rows, report = prepare(*paths)
            self.assertEqual(len(rows), 499)
            self.assertEqual(report["exclusions"][0]["excluded_annotation_id"], "gold_0499")
            self.assertEqual(original, paths[0].read_bytes())

    def test_links_on_excluded_duplicates_connect_both_contexts(self):
        rows = [
            {"annotation_id": "a", "uri": "post:a", "source_uri": "root:a", "root_uri": "root:a", "parent_uri": ""},
            {"annotation_id": "b", "uri": "post:a", "source_uri": "root:b", "root_uri": "root:b", "parent_uri": ""},
            {"annotation_id": "c", "uri": "root:b", "source_uri": "", "root_uri": "root:b", "parent_uri": ""},
        ]
        groups = connected_groups(rows)
        self.assertEqual(len(set(groups.values())), 1)
        self.assertEqual(groups, connected_groups(list(reversed(rows))))


if __name__ == "__main__":
    unittest.main()
