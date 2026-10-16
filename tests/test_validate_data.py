import csv
import hashlib
import json
import tempfile
import unittest
import importlib.util
from pathlib import Path

from bluesky_discourse.validate_data import validate
from bluesky_discourse.prepare_data import assign_folds, connected_groups, model_text, prepare
from bluesky_discourse.baseline import evaluate
from bluesky_discourse.train_tfidf import build_pipeline, evaluate_text
from bluesky_discourse.validate_data import LABELS
from bluesky_discourse.train_transformer import EncodedPosts, class_weights, make_weighted_loss, split_rows
from bluesky_discourse.train_transformer import MODEL_ID, MODEL_REVISION, summarize_runs
from bluesky_discourse.build_gold_v3 import build_rows, read_table
from bluesky_discourse.prepare_data import prepare_v3


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

    def test_reply_text_preserves_content_and_excludes_label_metadata(self):
        row = {
            "item_type": "reply",
            "annotation_text": "  Not\ntrue!\t#science @someone https://example.org \U0001f600 ",
            "context_text": " A  published\nresult. ",
            "final_label": "SECRET_LABEL", "llm_rationale": "SECRET_RATIONALE",
        }
        result = model_text(row)
        self.assertEqual(result["text_only"],
                         "[REPLY] Not true! #science @someone https://example.org \U0001f600")
        self.assertEqual(result["text_with_context"],
                         result["text_only"] + " [CONTEXT] A published result.")
        self.assertNotIn("SECRET", " ".join(result.values()))

    def test_original_ignores_context_and_missing_reply_context_is_safe(self):
        original = model_text({"item_type": "original_post", "annotation_text": "New paper", "context_text": "Unused"})
        self.assertEqual(original, {"text_only": "[ORIGINAL] New paper", "text_with_context": "[ORIGINAL] New paper"})
        reply = model_text({"item_type": "reply", "annotation_text": "Thanks", "context_text": " \n "})
        self.assertEqual(reply["text_only"], reply["text_with_context"])
        self.assertNotIn("[CONTEXT]", reply["text_with_context"])

    def test_folds_cover_items_keep_groups_and_repeat_with_fixed_seed(self):
        rows = [{"annotation_id": f"id_{index:03d}", "uri": f"uri:{index}",
                 "group_id": f"group:{index // 2}", "final_label": "A" if index % 2 else "B"}
                for index in range(60)]
        first = [dict(row) for row in rows]
        second = [dict(row) for row in reversed(rows)]
        report = assign_folds(first)
        second_report = assign_folds(second)
        self.assertEqual({row["fold"] for row in first}, set(range(5)))
        for group in {row["group_id"] for row in first}:
            self.assertEqual(len({row["fold"] for row in first if row["group_id"] == group}), 1)
        self.assertEqual({row["annotation_id"]: row["fold"] for row in first},
                         {row["annotation_id"]: row["fold"] for row in second})
        self.assertEqual(report["assignment_sha256"], second_report["assignment_sha256"])

    def test_fold_assignment_rejects_repeated_post_uri(self):
        rows = [{"annotation_id": "a", "uri": "same", "group_id": "g", "final_label": "A"},
                {"annotation_id": "b", "uri": "same", "group_id": "h", "final_label": "A"}]
        with self.assertRaisesRegex(ValueError, "unique post URIs"):
            assign_folds(rows)


class BaselineTests(unittest.TestCase):
    def fixture(self):
        # The global majority is Critique, but fold 0's training majority is Explanation.
        return [{"annotation_id": f"id_{index:02d}", "uri": f"uri:{index}",
                 "group_id": f"group:{index}", "fold": 0 if index < 10 else index - 9,
                 "final_label": "Critique" if index < 10 else "Explanation"}
                for index in range(14)]

    def test_majority_uses_training_labels_and_predicts_each_item_once(self):
        predictions, report = evaluate(self.fixture())
        self.assertEqual(len(predictions), 14)
        self.assertEqual(len({row["annotation_id"] for row in predictions}), 14)
        self.assertEqual({row["predicted_label"] for row in predictions if row["fold"] == 0}, {"Explanation"})
        self.assertEqual(report["overall_out_of_fold"]["accuracy"], 0.0)
        self.assertEqual(sum(map(sum, report["confusion_matrix"])), 14)

    def test_baseline_rejects_connected_group_crossing_folds(self):
        rows = self.fixture()
        rows[-1]["group_id"] = rows[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "multiple folds"):
            evaluate(rows)

    def test_changing_text_or_metadata_cannot_change_baseline_predictions(self):
        rows = self.fixture()
        before = evaluate(rows)[0]
        for row in rows:
            row.update(annotation_text="Arbitrary changed text", llm_suggested_label="Other")
        self.assertEqual(before, evaluate(rows)[0])


class TextClassifierTests(unittest.TestCase):
    def fixture(self):
        return [{"annotation_id": f"id_{fold}_{index}", "uri": f"uri:{fold}:{index}",
                 "group_id": f"group:{fold}:{index}", "fold": fold, "final_label": label,
                 "text_only": f"[REPLY] example {label}",
                 "text_with_context": f"[REPLY] example {label} [CONTEXT] paper"}
                for fold in range(5) for index, label in enumerate(sorted(LABELS))]

    def test_test_only_words_do_not_enter_vocabulary_or_change_idf(self):
        model = build_pipeline()
        model.fit(["methodological flaws", "detailed explanation"], ["Critique", "Explanation"])
        vectorizer = model.named_steps["tfidf"]
        vocabulary = dict(vectorizer.vocabulary_)
        idf = vectorizer.idf_.copy()
        model.predict(["unseenwordonlytest"])
        self.assertNotIn("unseenwordonlytest", vectorizer.vocabulary_)
        self.assertEqual(vocabulary, vectorizer.vocabulary_)
        self.assertTrue((idf == vectorizer.idf_).all())

    def test_text_evaluation_covers_items_and_ignores_label_metadata(self):
        rows = self.fixture()
        first, report = evaluate_text(rows, "text_only")
        self.assertEqual(len(first), 35)
        self.assertEqual(len({row["annotation_id"] for row in first}), 35)
        self.assertEqual(set(report["predicted_label_counts"]), LABELS)
        for row in rows:
            row.update(llm_suggested_label="Other", ramin_label="Other")
        self.assertEqual(first, evaluate_text(rows, "text_only")[0])

    def test_label_column_is_rejected_as_model_input(self):
        with self.assertRaisesRegex(ValueError, "not label/metadata"):
            evaluate_text(self.fixture(), "final_label")

    def test_text_evaluation_rejects_group_overlap(self):
        rows = self.fixture()
        rows[-1]["group_id"] = rows[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "multiple folds"):
            evaluate_text(rows, "text_with_context")


class TransformerTests(unittest.TestCase):
    def fixture(self):
        return TextClassifierTests().fixture()

    def test_split_keeps_test_items_out_of_training(self):
        train, test = split_rows(self.fixture(), 0, "text_only")
        self.assertEqual((len(train), len(test)), (28, 7))
        self.assertFalse({row["group_id"] for row in train} & {row["group_id"] for row in test})
        before = class_weights(train)
        for row in test:
            row["final_label"] = "Explanation"
        self.assertEqual(before, class_weights(train))

    def test_weights_match_inverse_training_frequency(self):
        train, _ = split_rows(self.fixture(), 0, "text_only")
        train.append(dict(train[0], annotation_id="extra"))
        weights = class_weights(train)
        from bluesky_discourse.train_transformer import LABEL_ORDER
        index = LABEL_ORDER.index(train[0]["final_label"])
        self.assertAlmostEqual(weights[index], 29 / (7 * 5))

    def test_labels_cannot_be_selected_as_transformer_input(self):
        with self.assertRaisesRegex(ValueError, "Only text_only"):
            split_rows(self.fixture(), 0, "final_label")

    def test_transformer_split_rejects_group_overlap(self):
        rows = self.fixture()
        rows[-1]["group_id"] = rows[0]["group_id"]
        with self.assertRaisesRegex(ValueError, "multiple folds"):
            split_rows(rows, 0, "text_only")

    def test_dataset_contains_only_tokens_and_numeric_labels(self):
        def tokenizer(texts, **kwargs):
            self.assertEqual(texts, ["[REPLY] example Acknowledgment"])
            self.assertEqual(kwargs, {"truncation": True, "max_length": 256, "padding": False})
            return {"input_ids": [[101, 102]], "attention_mask": [[1, 1]]}
        dataset = EncodedPosts(tokenizer, self.fixture()[:1], "text_only")
        self.assertEqual(set(dataset[0]), {"input_ids", "attention_mask", "labels"})
        self.assertIsInstance(dataset[0]["labels"], int)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional PyTorch is not installed")
    def test_weighted_loss_supports_backward_without_pretrained_download(self):
        import torch
        from types import SimpleNamespace
        logits = torch.zeros((2, 7), requires_grad=True)
        loss = make_weighted_loss([1.0] * 7)(SimpleNamespace(logits=logits), torch.tensor([0, 1]))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
                         "Optional transformer packages are not installed")
    def test_tiny_offline_distilbert_forward_backward_and_optimizer_step(self):
        import torch
        from transformers import DistilBertConfig, DistilBertForSequenceClassification
        config = DistilBertConfig(vocab_size=32, max_position_embeddings=16,
                                 n_layers=1, n_heads=2, dim=16, hidden_dim=32,
                                 num_labels=7, dropout=0.0, attention_dropout=0.0)
        config._attn_implementation = "eager"
        model = DistilBertForSequenceClassification(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        output = model(input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
                       attention_mask=torch.ones((2, 3), dtype=torch.long))
        loss = make_weighted_loss([1.0] * 7)(output, torch.tensor([0, 1]))
        loss.backward()
        self.assertEqual(tuple(output.logits.shape), (2, 7))
        self.assertTrue(torch.isfinite(loss))
        before = model.classifier.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.classifier.weight))


class TransformerSummaryTests(unittest.TestCase):
    def fixture(self, folder):
        rows = TextClassifierTests().fixture()
        csv_path = folder / "training.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        preparation = {"training_rows": len(rows),
                       "training_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                       "folds": {"assignment_sha256": "synthetic-assignments"}}
        preparation_path = folder / "preparation.json"
        preparation_path.write_text(json.dumps(preparation), encoding="utf-8")
        for fold in range(5):
            run_folder = folder / f"distilbert_fold{fold}_text_only"
            run_folder.mkdir()
            test = [row for row in rows if row["fold"] == fold]
            predictions = [{"annotation_id": row["annotation_id"], "group_id": row["group_id"],
                            "fold": fold, "variant": "text_only", "actual_label": row["final_label"],
                            "predicted_label": row["final_label"], "correct": True} for row in test]
            predictions_path = run_folder / "predictions.csv"
            with predictions_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(predictions[0]), delimiter=";")
                writer.writeheader()
                writer.writerows(predictions)
            report = {
                "training_csv_sha256": preparation["training_csv_sha256"],
                "preparation_report_sha256": hashlib.sha256(preparation_path.read_bytes()).hexdigest(),
                "fold_assignment_sha256": "synthetic-assignments", "fold": fold,
                "text_column": "text_only", "status": "completed", "train_rows": 28, "test_rows": 7,
                "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
                "checks": {"no_group_overlap": True, "test_not_used_for_training_or_selection": True,
                           "each_test_item_predicted_once": True},
                "training_arguments": {"learning_rate": 2e-5, "output_dir": str(run_folder)},
                "tokenization": {"max_length": 256, "truncation": True, "padding": "dynamic per batch"},
                "attention_implementation": "eager", "loss": "synthetic weighted loss",
                "versions": {"test": "synthetic"}, "source_sha256": {"test": "synthetic"},
                "training_seconds": 1.0,
                "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
            }
            (run_folder / "report.json").write_text(json.dumps(report), encoding="utf-8")
        return csv_path, preparation_path, folder, "text_only"

    def test_summary_combines_exactly_one_prediction_per_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            predictions, report = summarize_runs(*self.fixture(Path(tmp)))
            self.assertEqual(len(predictions), 35)
            self.assertEqual(report["overall_out_of_fold"]["accuracy"], 1.0)
            self.assertEqual(report["overall_out_of_fold"]["macro_f1"], 1.0)
            self.assertEqual(report["total_training_seconds"], 5.0)

    def test_summary_requires_all_five_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            (Path(tmp) / "distilbert_fold4_text_only/report.json").unlink()
            with self.assertRaises(OSError):
                summarize_runs(*args)

    def test_summary_rejects_modified_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            path = Path(tmp) / "distilbert_fold4_text_only/predictions.csv"
            with path.open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaisesRegex(ValueError, "saved hash"):
                summarize_runs(*args)

    def test_summary_rejects_mixed_training_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            path = Path(tmp) / "distilbert_fold4_text_only/report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["training_arguments"]["learning_rate"] = 1e-3
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "settings, versions, or source hashes differ"):
                summarize_runs(*args)

    def test_summary_rejects_missing_and_repeated_test_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            folder = Path(tmp) / "distilbert_fold4_text_only"
            path = folder / "predictions.csv"
            with path.open(encoding="utf-8", newline="") as stream:
                predictions = list(csv.DictReader(stream, delimiter=";"))
            predictions[-1] = dict(predictions[0])
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(predictions[0]), delimiter=";")
                writer.writeheader()
                writer.writerows(predictions)
            report_path = folder / "report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["predictions_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            report_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing, repeated, or unexpected"):
                summarize_runs(*args)


class GoldV3Tests(unittest.TestCase):
    def fixture(self):
        original = [{"annotation_id": f"gold_{i:04d}", "annotation_text": "Unchanged text",
                     "doi": "10.1234/example", "final_label": "Contextualization",
                     "final_label_source": "human_majority", "ramin_label": "Contextualization",
                     "sarah_label": "Contextualization", "fakhri_label": "Contextualization"}
                    for i in range(1, 30)]
        original.append(dict(original[0], annotation_id="gold_0043", doi="10.1126", final_label="Explanation"))
        reviewed = [dict(row, new_label_Contextualization="Contextualization" if i < 29 else "", Comment_on_="")
                    for i, row in enumerate(original)]
        reviewed[0]["new_label_Contextualization"] = "Announcement"
        return list(original[0]), original, reviewed

    def test_v3_changes_only_reviewed_final_labels_and_preserves_human_labels(self):
        columns, original, reviewed = self.fixture()
        rows, report = build_rows(columns, original, reviewed)
        self.assertEqual(report["reviewed_rows"], 29)
        self.assertEqual(report["changed_rows"], 1)
        self.assertEqual(rows[0]["final_label"], "Announcement")
        self.assertEqual(rows[0]["final_label_v2"], "Contextualization")
        self.assertEqual(rows[0]["final_label_source"], "posthoc_handbook_review")
        self.assertEqual(rows[0]["ramin_label"], original[0]["ramin_label"])
        self.assertEqual(original[0]["final_label"], "Contextualization")

    def test_confirmed_spreadsheet_doi_mistake_is_recorded_and_not_propagated(self):
        columns, original, reviewed = self.fixture()
        reviewed[-1]["doi"] = "101.126"
        rows, report = build_rows(columns, original, reviewed)
        self.assertEqual(rows[-1]["doi"], "10.1126")
        self.assertEqual(len(report["ignored_review_metadata_edits"]), 1)

    def test_unexpected_text_changes_are_rejected(self):
        columns, original, reviewed = self.fixture()
        reviewed[0]["annotation_text"] = "Altered text"
        with self.assertRaisesRegex(ValueError, "Unexpected original-field edit"):
            build_rows(columns, original, reviewed)

    def test_missing_review_decision_is_rejected(self):
        columns, original, reviewed = self.fixture()
        reviewed[0]["new_label_Contextualization"] = ""
        with self.assertRaisesRegex(ValueError, "Missing or invalid review label"):
            build_rows(columns, original, reviewed)

    def test_review_outside_contextualization_is_rejected(self):
        columns, original, reviewed = self.fixture()
        reviewed[-1]["new_label_Contextualization"] = "Other"
        with self.assertRaisesRegex(ValueError, "outside the agreed"):
            build_rows(columns, original, reviewed)

    def test_empty_spreadsheet_columns_are_ignored_but_filled_ones_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.csv"
            path.write_text('annotation_id;;new_label_Contextualization\na;;Announcement\n', encoding="utf-8")
            columns, rows = read_table(path, review=True)
            self.assertNotIn("", columns)
            self.assertEqual(rows[0]["annotation_id"], "a")
            path.write_text('annotation_id;;new_label_Contextualization\na;unexpected;Announcement\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unnamed column content"):
                read_table(path, review=True)


class V3PreparationTests(unittest.TestCase):
    def fixture(self, folder):
        source_path, source_manifest_path = ValidationTests().fixture(folder, duplicate=True)
        with source_path.open(encoding="utf-8-sig", newline="") as stream:
            original = list(csv.DictReader(stream, delimiter=";"))
        other_labels = sorted(LABELS - {"Contextualization"})
        for index, row in enumerate(original):
            row["final_label"] = "Contextualization" if index < 29 else other_labels[index % 6]
        original[-1]["final_label"] = original[-2]["final_label"]
        self.write_csv(source_path, original)
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_manifest["outputs"]["csv"]["sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        from collections import Counter
        source_manifest["final_label_distribution"] = dict(Counter(row["final_label"] for row in original))
        source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
        old_rows, old_report = prepare(source_path, source_manifest_path)
        old_csv = folder / "training_v2.csv"
        old_report_path = folder / "preparation_v2.json"
        self.write_csv(old_csv, old_rows)
        old_report["training_csv_sha256"] = hashlib.sha256(old_csv.read_bytes()).hexdigest()
        old_report_path.write_text(json.dumps(old_report), encoding="utf-8")
        reviewed = [dict(row, new_label_Contextualization="Announcement" if index < 17 else
                         "Contextualization" if index < 29 else "", Comment_on_="")
                    for index, row in enumerate(original)]
        revised, review_report = build_rows(list(original[0]), original, reviewed)
        revised_csv = folder / "gold_v3.csv"
        revised_manifest_path = folder / "gold_v3_manifest.json"
        self.write_csv(revised_csv, revised)
        revised_manifest = {
            **review_report, "version": 3, "rows": 500,
            "inputs": {"v2_csv_sha256": old_report["input_csv_sha256"],
                       "v2_manifest_sha256": old_report["input_manifest_sha256"]},
            "outputs": {"csv": {"sha256": hashlib.sha256(revised_csv.read_bytes()).hexdigest()}},
        }
        revised_manifest_path.write_text(json.dumps(revised_manifest), encoding="utf-8")
        return (revised_csv, revised_manifest_path, old_csv, old_report_path), old_rows

    def write_csv(self, path, rows):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=";")
            writer.writeheader()
            writer.writerows(rows)

    def test_v3_keeps_reference_ids_groups_folds_and_model_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, old_rows = self.fixture(Path(tmp))
            rows, report = prepare_v3(*args)
            self.assertEqual(len(rows), 499)
            self.assertEqual(report["changed_training_labels"], 17)
            self.assertEqual(report["label_counts"]["Contextualization"], 12)
            self.assertEqual([(row["annotation_id"], row["group_id"], row["fold"], row["text_only"], row["text_with_context"]) for row in rows],
                             [(row["annotation_id"], row["group_id"], row["fold"], row["text_only"], row["text_with_context"]) for row in old_rows])

    def test_v3_rejects_an_altered_release_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = self.fixture(Path(tmp))
            with args[0].open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaisesRegex(ValueError, "CSV hash"):
                prepare_v3(*args)

    def test_v3_rejects_metadata_changes_even_if_manifest_hash_is_updated(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = self.fixture(Path(tmp))
            _, revised = read_table(args[0])
            revised[0]["annotation_text"] = "Unexpected changed text"
            self.write_csv(args[0], revised)
            manifest = json.loads(args[1].read_text(encoding="utf-8"))
            manifest["outputs"]["csv"]["sha256"] = hashlib.sha256(args[0].read_bytes()).hexdigest()
            args[1].write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Original metadata differs"):
                prepare_v3(*args)

    def test_v3_rejects_reference_fold_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, _ = self.fixture(Path(tmp))
            _, reference = read_table(args[2])
            reference[0]["fold"] = str((int(reference[0]["fold"]) + 1) % 5)
            self.write_csv(args[2], reference)
            report = json.loads(args[3].read_text(encoding="utf-8"))
            report["training_csv_sha256"] = hashlib.sha256(args[2].read_bytes()).hexdigest()
            args[3].write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fold assignments do not match"):
                prepare_v3(*args)


if __name__ == "__main__":
    unittest.main()
