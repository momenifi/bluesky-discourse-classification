"""Fine-tune pinned DistilBERT on one saved training partition; test once."""

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from statistics import mean, pstdev

from sklearn.metrics import classification_report, confusion_matrix

from .baseline import check_evaluation_rows, load_prepared_data, scores
from .validate_data import LABELS


MODEL_ID = "distilbert/distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
LABEL_ORDER = sorted(LABELS)
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_ORDER)}
TEXT_COLUMNS = ("text_only", "text_with_context")


def split_rows(rows: list[dict], fold: int, text_column: str) -> tuple[list[dict], list[dict]]:
    check_evaluation_rows(rows)
    if fold not in range(5):
        raise ValueError("Choose fold 0 through 4")
    if text_column not in TEXT_COLUMNS:
        raise ValueError("Only text_only or text_with_context can be used as input")
    if any(not row.get(text_column, "").strip() for row in rows):
        raise ValueError(f"Empty or missing {text_column}")
    ordered = sorted(rows, key=lambda row: row["annotation_id"])
    train = [row for row in ordered if int(row["fold"]) != fold]
    test = [row for row in ordered if int(row["fold"]) == fold]
    if {row["final_label"] for row in train} != LABELS:
        raise ValueError("All seven labels must be present in the training partition")
    return train, test


def class_weights(train: list[dict]) -> list[float]:
    """Balanced weights, computed exclusively from training labels."""
    counts = Counter(row["final_label"] for row in train)
    if set(counts) != LABELS:
        raise ValueError("All seven labels are needed to compute class weights")
    return [len(train) / (len(LABEL_ORDER) * counts[label]) for label in LABEL_ORDER]


class EncodedPosts:
    """Only tokenized text and numeric targets reach the Trainer."""

    def __init__(self, tokenizer, rows: list[dict], text_column: str, max_length: int = 256):
        self.encodings = tokenizer(
            [row[text_column] for row in rows], truncation=True,
            max_length=max_length, padding=False,
        )
        self.labels = [LABEL_TO_ID[row["final_label"]] for row in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {**{key: value[index] for key, value in self.encodings.items()},
                "labels": self.labels[index]}


def make_weighted_loss(weights: list[float]):
    import torch

    def weighted_loss(outputs, labels, num_items_in_batch=None):
        # With accumulation fixed to 1, use a weighted mean within each batch.
        return torch.nn.functional.cross_entropy(
            outputs.logits, labels,
            weight=torch.tensor(weights, device=outputs.logits.device, dtype=outputs.logits.dtype),
        )

    return weighted_loss


def summarize_runs(csv_path: Path, preparation_path: Path, runs_dir: Path,
                   text_column: str) -> tuple[list[dict], dict]:
    """Check and combine completed folds without loading a model or using CUDA."""
    rows, provenance = load_prepared_data(csv_path, preparation_path)
    predictions = []
    fold_results = []
    common_settings = None
    for fold in range(5):
        folder = runs_dir / f"distilbert_fold{fold}_{text_column}"
        report_bytes = (folder / "report.json").read_bytes()
        fold_report = json.loads(report_bytes.decode("utf-8-sig"))
        prediction_bytes = (folder / "predictions.csv").read_bytes()
        if hashlib.sha256(prediction_bytes).hexdigest() != fold_report["predictions_sha256"]:
            raise ValueError(f"Fold {fold}: predictions do not match the saved hash")
        if fold_report["status"] != "completed" or fold_report["fold"] != fold or fold_report["text_column"] != text_column:
            raise ValueError(f"Fold {fold}: mismatched or incomplete run")
        if any(fold_report[key] != value for key, value in provenance.items()):
            raise ValueError(f"Fold {fold}: data or folds differ from the current preparation")
        if fold_report["model_id"] != MODEL_ID or fold_report["model_revision"] != MODEL_REVISION:
            raise ValueError(f"Fold {fold}: unexpected pretrained checkpoint")
        if not all(fold_report["checks"].get(key) is True for key in (
            "no_group_overlap", "test_not_used_for_training_or_selection", "each_test_item_predicted_once"
        )):
            raise ValueError(f"Fold {fold}: recorded split checks did not pass")
        settings = {
            "training_arguments": {key: value for key, value in fold_report["training_arguments"].items()
                                   if key not in {"output_dir", "logging_dir", "run_name"}},
            "tokenization": {key: fold_report["tokenization"][key] for key in ("max_length", "truncation", "padding")},
            "attention_implementation": fold_report["attention_implementation"],
            "loss": fold_report["loss"], "versions": fold_report["versions"],
            "source_sha256": fold_report["source_sha256"],
        }
        if common_settings is not None and settings != common_settings:
            raise ValueError(f"Fold {fold}: training settings, versions, or source hashes differ across runs")
        common_settings = settings
        train, test = split_rows(rows, fold, text_column)
        if (fold_report["train_rows"], fold_report["test_rows"]) != (len(train), len(test)):
            raise ValueError(f"Fold {fold}: unexpected train/test sizes")
        expected = {row["annotation_id"]: row for row in test}
        with (folder / "predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            fold_predictions = list(csv.DictReader(stream, delimiter=";"))
        if len(fold_predictions) != len(test) or {row["annotation_id"] for row in fold_predictions} != set(expected):
            raise ValueError(f"Fold {fold}: missing, repeated, or unexpected test IDs")
        for prediction in fold_predictions:
            original = expected[prediction["annotation_id"]]
            if (prediction["actual_label"] != original["final_label"] or
                    prediction["group_id"] != original["group_id"] or
                    prediction["fold"] != str(fold) or prediction["variant"] != text_column or
                    prediction["predicted_label"] not in LABELS):
                raise ValueError(f"Fold {fold}: prediction metadata does not match the prepared data")
            prediction["fold"] = fold
            prediction["correct"] = prediction["actual_label"] == prediction["predicted_label"]
        fold_scores = scores([row["actual_label"] for row in fold_predictions],
                             [row["predicted_label"] for row in fold_predictions])
        predictions.extend(fold_predictions)
        fold_results.append({"fold": fold, "test_rows": len(test), "scores": fold_scores,
                             "training_seconds": fold_report["training_seconds"],
                             "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                             "predictions_sha256": fold_report["predictions_sha256"]})
    predictions.sort(key=lambda row: row["annotation_id"])
    if len(predictions) != len(rows) or {row["annotation_id"] for row in predictions} != {row["annotation_id"] for row in rows}:
        raise ValueError("Expected exactly one held-out prediction for every prepared item")
    actual = [row["actual_label"] for row in predictions]
    predicted = [row["predicted_label"] for row in predictions]
    report = {
        **provenance, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "text_column": text_column, "rows": len(predictions), "folds": 5,
        "common_run_settings": common_settings, "fold_results": fold_results,
        "overall_out_of_fold": scores(actual, predicted),
        "fold_mean_and_std": {
            metric: {"mean": mean(result["scores"][metric] for result in fold_results),
                     "std_population": pstdev(result["scores"][metric] for result in fold_results)}
            for metric in ("accuracy", "macro_f1", "weighted_f1")
        },
        "total_training_seconds": sum(result["training_seconds"] for result in fold_results),
        "predicted_label_counts": dict(sorted(Counter(predicted).items())),
        "label_order": LABEL_ORDER,
        "classification_report": classification_report(actual, predicted, labels=LABEL_ORDER, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=LABEL_ORDER).tolist(),
        "confusion_matrix_axes": "rows=actual, columns=predicted",
        "checks": {"all_five_folds_complete": True, "one_held_out_prediction_per_item": True,
                   "same_data_settings_and_versions": True, "no_group_overlap": True},
        "aggregation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return predictions, report


def summarize(args) -> None:
    predictions, report = summarize_runs(args.csv, args.preparation_report, args.runs_dir, args.text_column)
    outdir = args.outdir or args.runs_dir / f"distilbert_{args.text_column}_summary"
    if outdir.exists() and (not outdir.is_dir() or any(outdir.iterdir())):
        raise ValueError(f"Summary directory is not empty: {outdir}; choose another --outdir")
    outdir.mkdir(parents=True, exist_ok=True)
    predictions_path = outdir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(predictions)
    report["predictions_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    report_path = outdir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Combined {len(predictions)} held-out predictions from five folds")
    for metric, value in report["overall_out_of_fold"].items():
        print(f"Overall out-of-fold {metric}: {value:.6f}")
    print(f"Total training time across folds: {report['total_training_seconds']:.1f} seconds")
    for label in LABEL_ORDER:
        result = report["classification_report"][label]
        print(f"{label}: precision={result['precision']:.4f}, recall={result['recall']:.4f}, F1={result['f1-score']:.4f}")
    print(f"Predictions: {predictions_path}")
    print(f"Report: {report_path}")


def run(args) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        Trainer, TrainingArguments, set_seed,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one CUDA GPU must be visible. Use CUDA_VISIBLE_DEVICES for your allocated GPU.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run with python on one GPU, not torchrun or a distributed launcher")
    rows, provenance = load_prepared_data(args.csv, args.preparation_report)
    train, test = split_rows(rows, args.fold, args.text_column)
    outdir = args.outdir or Path(f"outputs/distilbert_fold{args.fold}_{args.text_column}")
    if outdir.exists() and (not outdir.is_dir() or any(outdir.iterdir())):
        raise ValueError(f"Output directory is not empty: {outdir}. Choose a new --outdir to preserve the earlier run.")
    weights = class_weights(train)
    print(f"GPU: {torch.cuda.get_device_name(0)} (visible CUDA device 0)", flush=True)
    print(f"Fold {args.fold}: {len(train)} training / {len(test)} held-out items", flush=True)
    print(f"Downloading/loading {MODEL_ID} at {MODEL_REVISION}", flush=True)
    started = time.perf_counter()
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    train_dataset = EncodedPosts(tokenizer, train, args.text_column)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, num_labels=len(LABEL_ORDER),
        id2label=dict(enumerate(LABEL_ORDER)), label2id=LABEL_TO_ID,
        use_safetensors=True, trust_remote_code=False, attn_implementation="eager",
    )
    training_args = TrainingArguments(
        output_dir=str(outdir), num_train_epochs=3, learning_rate=2e-5,
        per_device_train_batch_size=8, per_device_eval_batch_size=8,
        gradient_accumulation_steps=1, weight_decay=0.01,
        lr_scheduler_type="linear", warmup_ratio=0.1, optim="adamw_torch",
        seed=42, data_seed=42, full_determinism=True,
        fp16=False, bf16=False, tf32=False,
        dataloader_num_workers=0, eval_strategy="no", save_strategy="no",
        load_best_model_at_end=False, logging_strategy="steps", logging_steps=10,
        report_to=[], push_to_hub=False,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_loss_func=make_weighted_loss(weights),
    )
    torch.cuda.synchronize()
    train_started = time.perf_counter()
    training_result = trainer.train()
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - train_started

    # The held-out dataset is introduced only after the fixed training run ends.
    test_dataset = EncodedPosts(tokenizer, test, args.text_column)
    prediction_result = trainer.predict(test_dataset)
    predicted = [LABEL_ORDER[int(index)] for index in prediction_result.predictions.argmax(axis=-1)]
    actual = [row["final_label"] for row in test]
    predictions = [{
        "annotation_id": row["annotation_id"], "group_id": row["group_id"],
        "fold": args.fold, "variant": args.text_column,
        "actual_label": row["final_label"], "predicted_label": label,
        "correct": row["final_label"] == label,
    } for row, label in zip(test, predicted)]
    report = {
        **provenance, "status": "completed", "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "fold": args.fold, "text_column": args.text_column,
        "train_rows": len(train), "test_rows": len(test), "label_order": LABEL_ORDER,
        "training_label_counts": dict(sorted(Counter(row["final_label"] for row in train).items())),
        "class_weights": dict(zip(LABEL_ORDER, weights)),
        "loss": "Training-only balanced weights; cross entropy with weighted mean per batch",
        "training_arguments": training_args.to_dict(),
        "tokenization": {"max_length": 256, "truncation": True, "padding": "dynamic per batch",
                         "train_items_at_max_length": sum(len(ids) == 256 for ids in train_dataset.encodings["input_ids"]),
                         "test_items_at_max_length": sum(len(ids) == 256 for ids in test_dataset.encodings["input_ids"])},
        "attention_implementation": "eager", "training_seconds": training_seconds,
        "elapsed_seconds_including_load_and_test": time.perf_counter() - started,
        "training_metrics": training_result.metrics,
        "held_out_scores": scores(actual, predicted),
        "predicted_label_counts": dict(sorted(Counter(predicted).items())),
        "classification_report": classification_report(
            actual, predicted, labels=LABEL_ORDER, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=LABEL_ORDER).tolist(),
        "confusion_matrix_axes": "rows=actual, columns=predicted",
        "checks": {"no_group_overlap": True, "test_not_used_for_training_or_selection": True,
                   "each_test_item_predicted_once": len(predictions) == len(test)},
        "gpu": {"name": torch.cuda.get_device_name(0), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(0)},
        "versions": {"python": platform.python_version(), "pytorch_cuda": torch.version.cuda,
                     **{name: version(name) for name in ("torch", "transformers", "accelerate", "numpy", "scipy", "scikit-learn")}},
        "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                          for name in ("train_transformer.py", "baseline.py", "validate_data.py")},
        "model_saved": False,
    }
    outdir.mkdir(parents=True, exist_ok=True)
    predictions_path = outdir / "predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(predictions)
    report["predictions_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    report_path = outdir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Training time: {training_seconds:.1f} seconds")
    for metric, value in report["held_out_scores"].items():
        print(f"Fold {args.fold} held-out {metric}: {value:.6f}")
    print(f"Predictions: {predictions_path}")
    print(f"Report: {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("outputs/training_data.csv"))
    parser.add_argument("--preparation-report", type=Path, default=Path("outputs/preparation_report.json"))
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--text-column", choices=TEXT_COLUMNS, default="text_only")
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--summarize", action="store_true", help="Combine five completed runs; no GPU or training")
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    try:
        if args.summarize:
            summarize(args)
        else:
            run(args)
    except (ImportError, OSError, ValueError, KeyError, TypeError, RuntimeError, csv.Error) as exc:
        print(f"Transformer run failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
