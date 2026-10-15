"""Evaluate a training-fold majority baseline on the saved five folds."""

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from statistics import mean, pstdev

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .validate_data import LABELS


def scores(actual: list[str], predicted: list[str]) -> dict:
    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, labels=sorted(LABELS), average="macro", zero_division=0),
        "weighted_f1": f1_score(actual, predicted, labels=sorted(LABELS), average="weighted", zero_division=0),
    }


def check_evaluation_rows(rows: list[dict]) -> None:
    """Check saved folds before either baseline or text-model evaluation."""
    if not rows:
        raise ValueError("No evaluation rows")
    for column in ("annotation_id", "uri", "group_id", "fold", "final_label"):
        if any(not str(row.get(column, "")).strip() for row in rows):
            raise ValueError(f"Missing or empty {column}")
    for column in ("annotation_id", "uri"):
        if len({row[column] for row in rows}) != len(rows):
            raise ValueError(f"Expected unique {column} values")
    if any(row["final_label"] not in LABELS for row in rows):
        raise ValueError("Unexpected final_label")
    if {str(row["fold"]) for row in rows} != {"0", "1", "2", "3", "4"}:
        raise ValueError("Expected saved folds 0 through 4")
    group_folds = {}
    for row in rows:
        fold = int(row["fold"])
        group = row["group_id"]
        if group in group_folds and group_folds[group] != fold:
            raise ValueError("A connected group appears in multiple folds")
        group_folds[group] = fold


def load_prepared_data(csv_path: Path, preparation_path: Path) -> tuple[list[dict], dict]:
    input_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    preparation_bytes = preparation_path.read_bytes()
    preparation = json.loads(preparation_bytes.decode("utf-8-sig"))
    if input_hash != preparation["training_csv_sha256"]:
        raise ValueError("Training CSV does not match its preparation report; regenerate both together")
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("CSV records have inconsistent column counts")
    if len(rows) != preparation["training_rows"]:
        raise ValueError("Row count does not match the preparation report")
    check_evaluation_rows(rows)
    return rows, {
        "training_csv_sha256": input_hash,
        "preparation_report_sha256": hashlib.sha256(preparation_bytes).hexdigest(),
        "fold_assignment_sha256": preparation["folds"]["assignment_sha256"],
    }


def evaluate(rows: list[dict]) -> tuple[list[dict], dict]:
    """Use test labels only for scoring, never for choosing predictions."""
    check_evaluation_rows(rows)
    predictions = []
    fold_results = []
    for fold in range(5):
        train = [row for row in rows if int(row["fold"]) != fold]
        test = [row for row in rows if int(row["fold"]) == fold]
        # DummyClassifier ignores these constant features; no post text is used.
        model = DummyClassifier(strategy="most_frequent")
        model.fit([[0]] * len(train), [row["final_label"] for row in train])
        predicted = model.predict([[0]] * len(test)).tolist()
        actual = [row["final_label"] for row in test]
        fold_results.append({
            "fold": fold, "train_rows": len(train), "test_rows": len(test),
            "training_label_counts": dict(sorted(Counter(row["final_label"] for row in train).items())),
            "predicted_label": predicted[0], "scores": scores(actual, predicted),
        })
        for row, label in zip(test, predicted):
            predictions.append({
                "annotation_id": row["annotation_id"], "group_id": row["group_id"],
                "fold": fold, "actual_label": row["final_label"],
                "predicted_label": label, "correct": row["final_label"] == label,
            })
    predictions.sort(key=lambda row: row["annotation_id"])
    actual = [row["actual_label"] for row in predictions]
    predicted = [row["predicted_label"] for row in predictions]
    summary = {
        "model": "DummyClassifier", "strategy": "most_frequent",
        "features": "None (constant dummy input)",
        "tie_rule": "First tied label in scikit-learn's sorted class order",
        "rows": len(rows), "fold_results": fold_results,
        "overall_out_of_fold": scores(actual, predicted),
        "fold_mean_and_std": {
            metric: {"mean": mean(result["scores"][metric] for result in fold_results),
                     "std_population": pstdev(result["scores"][metric] for result in fold_results)}
            for metric in ("accuracy", "macro_f1", "weighted_f1")
        },
        "label_order": sorted(LABELS),
        "classification_report": classification_report(
            actual, predicted, labels=sorted(LABELS), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=sorted(LABELS)).tolist(),
        "confusion_matrix_axes": "rows=actual, columns=predicted",
        "checks": {"unique_ids_and_post_uris": True, "no_group_overlap": True,
                   "one_held_out_prediction_per_item": len(predictions) == len(rows)},
    }
    return predictions, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("outputs/training_data.csv"))
    parser.add_argument("--preparation-report", type=Path, default=Path("outputs/preparation_report.json"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    predictions_path = args.outdir / "baseline_predictions.csv"
    report_path = args.outdir / "baseline_report.json"
    if {predictions_path.resolve(), report_path.resolve()} & {args.csv.resolve(), args.preparation_report.resolve()}:
        parser.error("Output paths must differ from input paths")
    try:
        rows, provenance = load_prepared_data(args.csv, args.preparation_report)
        predictions, report = evaluate(rows)
        report.update({
            **provenance,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "versions": {"python": platform.python_version(),
                         **{name: version(name) for name in ("scikit-learn", "numpy", "scipy")}},
        })
        args.outdir.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]), delimiter=";")
            writer.writeheader()
            writer.writerows(predictions)
        report["predictions_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        print(f"Baseline failed: {exc}")
        return 1
    for result in report["fold_results"]:
        print(f"Fold {result['fold']}: predict {result['predicted_label']}; accuracy={result['scores']['accuracy']:.4f}")
    for metric, value in report["overall_out_of_fold"].items():
        print(f"Out-of-fold {metric}: {value:.6f}")
    print(f"Predictions: {predictions_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
