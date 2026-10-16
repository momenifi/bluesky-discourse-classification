"""Fine-tune pinned DistilBERT on one saved training partition; test once."""

import argparse
import csv
import gc
import hashlib
import json
import math
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
SELECTION_PROTOCOL = {
    "inner_splits": 5, "inner_seed": 42, "max_epochs": 10, "patience": 2,
    "metric": "macro_f1", "tie_rule": "earliest", "refit_full_outer_training": True,
}


class EpochChoice:
    """Track validation scores only; equal scores keep the earlier epoch."""

    def __init__(self):
        self.best_epoch = None
        self.best_score = -math.inf
        self.bad_epochs = 0
        self.history = []

    def observe(self, epoch: int, score: float) -> bool:
        if epoch != len(self.history) + 1 or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Expected consecutive epochs and finite validation F1 in [0, 1]")
        self.history.append({"epoch": epoch, "macro_f1": float(score)})
        if score > self.best_score:
            self.best_epoch, self.best_score, self.bad_epochs = epoch, float(score), 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= SELECTION_PROTOCOL["patience"]


def validation_split(outer_train: list[dict]) -> tuple[list[dict], list[dict], dict]:
    """Select the first class-covered inner fold, without model-score search."""
    from sklearn.model_selection import StratifiedGroupKFold

    ordered = sorted(outer_train, key=lambda row: row["annotation_id"])
    if len({row["annotation_id"] for row in ordered}) != len(ordered):
        raise ValueError("Inner validation requires unique IDs")
    labels = [row["final_label"] for row in ordered]
    groups = [row["group_id"] for row in ordered]
    if set(labels) != LABELS:
        raise ValueError("Inner split requires all seven training labels")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for candidate, (fit_indices, validation_indices) in enumerate(splitter.split([[0]] * len(ordered), labels, groups)):
        fit = [ordered[int(index)] for index in fit_indices]
        validation = [ordered[int(index)] for index in validation_indices]
        if {row["final_label"] for row in fit} != LABELS or {row["final_label"] for row in validation} != LABELS:
            continue
        if {row["group_id"] for row in fit} & {row["group_id"] for row in validation}:
            raise ValueError("Inner training and validation groups overlap")
        return fit, validation, {
            "inner_fold": candidate, "fit_rows": len(fit), "validation_rows": len(validation),
            "fit_annotation_ids": [row["annotation_id"] for row in fit],
            "validation_annotation_ids": [row["annotation_id"] for row in validation],
            "fit_label_counts": dict(sorted(Counter(row["final_label"] for row in fit).items())),
            "validation_label_counts": dict(sorted(Counter(row["final_label"] for row in validation).items())),
        }
    raise ValueError("No inner split contains all seven labels on both sides; review the protocol, not the outer test scores")


def new_model():
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, num_labels=len(LABEL_ORDER),
        id2label=dict(enumerate(LABEL_ORDER)), label2id=LABEL_TO_ID,
        use_safetensors=True, trust_remote_code=False, attn_implementation="eager",
    )


def training_arguments(outdir: Path, epochs: int, *, validation: bool = False, use_cpu: bool = False):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(outdir), num_train_epochs=epochs, learning_rate=2e-5,
        per_device_train_batch_size=8, per_device_eval_batch_size=8,
        gradient_accumulation_steps=1, weight_decay=0.01,
        lr_scheduler_type="linear", warmup_ratio=0.1, optim="adamw_torch",
        seed=42, data_seed=42, full_determinism=True,
        use_cpu=use_cpu, dataloader_pin_memory=not use_cpu,
        fp16=False, bf16=False, tf32=False,
        dataloader_num_workers=0, eval_strategy="epoch" if validation else "no", save_strategy="no",
        load_best_model_at_end=False, logging_strategy="steps", logging_steps=10,
        report_to=[], push_to_hub=False,
    )


def select_epoch_count(outer_train: list[dict], tokenizer, text_column: str, outdir: Path) -> dict:
    import torch
    from transformers import DataCollatorWithPadding, Trainer, TrainerCallback, set_seed

    fit, validation, split_report = validation_split(outer_train)
    tracker = EpochChoice()

    class ValidationCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            epoch = int(round(state.epoch))
            if abs(state.epoch - epoch) > 1e-6:
                raise ValueError("Epoch selection requires evaluation at complete epochs")
            if tracker.observe(epoch, metrics["eval_macro_f1"]):
                control.should_training_stop = True
            return control

    def compute_metrics(result):
        actual = [LABEL_ORDER[int(index)] for index in result.label_ids]
        predicted = [LABEL_ORDER[int(index)] for index in result.predictions.argmax(axis=-1)]
        return scores(actual, predicted)

    print(f"Inner split: {len(fit)} fit / {len(validation)} validation items; outer test excluded", flush=True)
    set_seed(42)
    model = new_model()
    args = training_arguments(outdir, SELECTION_PROTOCOL["max_epochs"], validation=True)
    weights = class_weights(fit)
    trainer = Trainer(
        model=model, args=args, train_dataset=EncodedPosts(tokenizer, fit, text_column),
        eval_dataset=EncodedPosts(tokenizer, validation, text_column),
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_loss_func=make_weighted_loss(weights), compute_metrics=compute_metrics,
        callbacks=[ValidationCallback()],
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    output = trainer.train()
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    if tracker.best_epoch is None:
        raise ValueError("No validation epoch completed")
    result = {
        "protocol": dict(SELECTION_PROTOCOL), "split": split_report,
        "selected_epochs": tracker.best_epoch, "best_validation_macro_f1": tracker.best_score,
        "validation_history": tracker.history, "training_metrics": output.metrics,
        "training_arguments": args.to_dict(), "fit_class_weights": dict(zip(LABEL_ORDER, weights)),
        "training_seconds": seconds,
    }
    print(f"Validation selected {tracker.best_epoch} epochs; restarting the original checkpoint for full-training refit", flush=True)
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def check_selection_report(report: dict, outer_train: list[dict]) -> None:
    selection = report["epoch_selection"]
    if selection["protocol"] != SELECTION_PROTOCOL:
        raise ValueError("Unexpected epoch-selection protocol")
    tracker = EpochChoice()
    for index, item in enumerate(selection["validation_history"]):
        stopped = tracker.observe(item["epoch"], item["macro_f1"])
        if stopped and index != len(selection["validation_history"]) - 1:
            raise ValueError("Validation continued after its stopping rule")
    if (not tracker.history or len(tracker.history) > SELECTION_PROTOCOL["max_epochs"] or
            (len(tracker.history) < SELECTION_PROTOCOL["max_epochs"] and tracker.bad_epochs < SELECTION_PROTOCOL["patience"])):
        raise ValueError("Incomplete validation epoch selection")
    if (report["selected_epochs"] != tracker.best_epoch or selection["selected_epochs"] != tracker.best_epoch or
            selection["best_validation_macro_f1"] != tracker.best_score or
            report["training_arguments"]["num_train_epochs"] != tracker.best_epoch or
            report["training_metrics"]["epoch"] != tracker.best_epoch):
        raise ValueError("Selected and completed refit epochs differ")
    expected_fit, _, expected_report = validation_split(outer_train)
    if selection["split"] != expected_report:
        raise ValueError("Recorded inner split differs from deterministic training-only split")
    if selection["fit_class_weights"] != dict(zip(LABEL_ORDER, class_weights(expected_fit))):
        raise ValueError("Inner class weights differ from inner-training label counts")



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
                   text_column: str, *, selected_epochs: bool = False) -> tuple[list[dict], dict]:
    """Check and combine completed folds without loading a model or using CUDA."""
    rows, provenance = load_prepared_data(csv_path, preparation_path)
    predictions = []
    fold_results = []
    common_settings = None
    for fold in range(5):
        family = "distilbert_selected" if selected_epochs else "distilbert"
        folder = runs_dir / f"{family}_fold{fold}_{text_column}"
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
        protocol = fold_report.get("protocol", "fixed_3_epochs")
        if protocol != ("training_only_epoch_selection" if selected_epochs else "fixed_3_epochs"):
            raise ValueError("Do not mix fixed-epoch and validation-selected experiments")
        excluded_arguments = {"output_dir", "logging_dir", "run_name"}
        if selected_epochs:
            excluded_arguments.add("num_train_epochs")
        settings = {
            "training_arguments": {key: value for key, value in fold_report["training_arguments"].items()
                                   if key not in excluded_arguments},
            "protocol": protocol,
            "tokenization": {key: fold_report["tokenization"][key] for key in ("max_length", "truncation", "padding")},
            "attention_implementation": fold_report["attention_implementation"],
            "loss": fold_report["loss"], "versions": fold_report["versions"],
            "source_sha256": fold_report["source_sha256"],
        }
        if selected_epochs:
            settings["selection_protocol"] = fold_report["epoch_selection"]["protocol"]
            settings["selection_training_arguments"] = {
                key: value for key, value in fold_report["epoch_selection"]["training_arguments"].items()
                if key not in {"output_dir", "logging_dir", "run_name"}
            }
        if common_settings is not None and settings != common_settings:
            raise ValueError(f"Fold {fold}: training settings, versions, or source hashes differ across runs")
        common_settings = settings
        train, test = split_rows(rows, fold, text_column)
        if selected_epochs:
            check_selection_report(fold_report, train)
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
                             "selected_epochs": fold_report.get("selected_epochs", 3),
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
        "protocol": "training_only_epoch_selection" if selected_epochs else "fixed_3_epochs",
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
    predictions, report = summarize_runs(args.csv, args.preparation_report, args.runs_dir, args.text_column,
                                         selected_epochs=args.select_epochs)
    family = "distilbert_selected" if args.select_epochs else "distilbert"
    outdir = args.outdir or args.runs_dir / f"{family}_{args.text_column}_summary"
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
    if args.select_epochs:
        print("Selected epochs by fold:", {item["fold"]: item["selected_epochs"] for item in report["fold_results"]})
    for label in LABEL_ORDER:
        result = report["classification_report"][label]
        print(f"{label}: precision={result['precision']:.4f}, recall={result['recall']:.4f}, F1={result['f1-score']:.4f}")
    print(f"Predictions: {predictions_path}")
    print(f"Report: {report_path}")


def run(args) -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import (
        AutoTokenizer, DataCollatorWithPadding, Trainer, set_seed,
    )

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Exactly one CUDA GPU must be visible. Use CUDA_VISIBLE_DEVICES for your allocated GPU.")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Run with python on one GPU, not torchrun or a distributed launcher")
    rows, provenance = load_prepared_data(args.csv, args.preparation_report)
    train, test = split_rows(rows, args.fold, args.text_column)
    family = "distilbert_selected" if args.select_epochs else "distilbert"
    outdir = args.outdir or Path(f"outputs/{family}_fold{args.fold}_{args.text_column}")
    if outdir.exists() and (not outdir.is_dir() or any(outdir.iterdir())):
        raise ValueError(f"Output directory is not empty: {outdir}. Choose a new --outdir to preserve the earlier run.")
    weights = class_weights(train)
    print(f"GPU: {torch.cuda.get_device_name(0)} (visible CUDA device 0)", flush=True)
    print(f"Fold {args.fold}: {len(train)} training / {len(test)} held-out items", flush=True)
    print(f"Downloading/loading {MODEL_ID} at {MODEL_REVISION}", flush=True)
    started = time.perf_counter()
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    selection = select_epoch_count(train, tokenizer, args.text_column, outdir) if args.select_epochs else None
    epochs = selection["selected_epochs"] if selection else 3
    set_seed(42)
    train_dataset = EncodedPosts(tokenizer, train, args.text_column)
    model = new_model()
    training_args = training_arguments(outdir, epochs)
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
        compute_loss_func=make_weighted_loss(weights),
    )
    torch.cuda.synchronize()
    train_started = time.perf_counter()
    training_result = trainer.train()
    torch.cuda.synchronize()
    refit_seconds = time.perf_counter() - train_started
    training_seconds = refit_seconds + (selection["training_seconds"] if selection else 0.0)

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
        "protocol": "training_only_epoch_selection" if selection else "fixed_3_epochs",
        "selected_epochs": epochs, "epoch_selection": selection, "refit_training_seconds": refit_seconds,
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
    if selection:
        print(f"Selected epochs: {epochs}; validation-selected, full-training refit completed")
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
    parser.add_argument("--select-epochs", action="store_true", help="Select epochs on inner training-only validation, then refit")
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
