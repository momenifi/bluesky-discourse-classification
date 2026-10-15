"""Evaluate fixed TF-IDF/logistic regression settings on saved grouped folds."""

import argparse
import csv
import hashlib
import json
import platform
import warnings
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from statistics import mean, pstdev

from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from .baseline import check_evaluation_rows, load_prepared_data, scores
from .validate_data import LABELS


TEXT_COLUMNS = ("text_only", "text_with_context")


def build_pipeline() -> Pipeline:
    """A fresh vocabulary, IDF weights, and classifier for each training fold."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("classifier", LogisticRegression(
            C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000,
            tol=1e-4, random_state=42,
        )),
    ])


def evaluate_text(rows: list[dict], text_column: str) -> tuple[list[dict], dict]:
    check_evaluation_rows(rows)
    if text_column not in TEXT_COLUMNS:
        raise ValueError("Select text_only or text_with_context, not label/metadata columns")
    if any(not row.get(text_column, "").strip() for row in rows):
        raise ValueError(f"Empty or missing input: {text_column}")
    ordered = sorted(rows, key=lambda row: row["annotation_id"])
    predictions = []
    fold_results = []
    for fold in range(5):
        train = [row for row in ordered if int(row["fold"]) != fold]
        test = [row for row in ordered if int(row["fold"]) == fold]
        model = build_pipeline()
        # Pipeline.fit learns TF-IDF and classifier weights only on this partition.
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit([row[text_column] for row in train], [row["final_label"] for row in train])
        predicted = model.predict([row[text_column] for row in test]).tolist()
        actual = [row["final_label"] for row in test]
        fold_results.append({
            "fold": fold, "train_rows": len(train), "test_rows": len(test),
            "training_label_counts": dict(sorted(Counter(row["final_label"] for row in train).items())),
            "vocabulary_size": len(model.named_steps["tfidf"].vocabulary_),
            "classifier_iterations": model.named_steps["classifier"].n_iter_.tolist(),
            "scores": scores(actual, predicted),
        })
        for row, label in zip(test, predicted):
            predictions.append({
                "variant": text_column, "annotation_id": row["annotation_id"],
                "group_id": row["group_id"], "fold": fold,
                "actual_label": row["final_label"], "predicted_label": label,
                "correct": row["final_label"] == label,
            })
    predictions.sort(key=lambda row: row["annotation_id"])
    actual = [row["actual_label"] for row in predictions]
    predicted = [row["predicted_label"] for row in predictions]
    report = {
        "rows": len(rows), "text_column": text_column, "fold_results": fold_results,
        "overall_out_of_fold": scores(actual, predicted),
        "predicted_label_counts": dict(sorted(Counter(predicted).items())),
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
    return predictions, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("outputs/training_data.csv"))
    parser.add_argument("--preparation-report", type=Path, default=Path("outputs/preparation_report.json"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    predictions_path = args.outdir / "tfidf_predictions.csv"
    report_path = args.outdir / "tfidf_report.json"
    if {predictions_path.resolve(), report_path.resolve()} & {args.csv.resolve(), args.preparation_report.resolve()}:
        parser.error("Output paths must differ from input paths")
    try:
        rows, provenance = load_prepared_data(args.csv, args.preparation_report)
        model = build_pipeline()
        report = {
            **provenance,
            "model": "TF-IDF + LogisticRegression", "hyperparameter_tuning": False,
            "tfidf_parameters": model.named_steps["tfidf"].get_params(),
            "classifier_parameters": model.named_steps["classifier"].get_params(),
            "versions": {"python": platform.python_version(),
                         **{name: version(name) for name in ("scikit-learn", "numpy", "scipy")}},
            "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("train_tfidf.py", "baseline.py", "validate_data.py")},
            "variants": {},
        }
        # Make default Python type parameters JSON-serializable without losing their identity.
        report["tfidf_parameters"]["dtype"] = str(report["tfidf_parameters"]["dtype"])
        all_predictions = []
        for column in TEXT_COLUMNS:
            print(f"Evaluating {column}...", flush=True)
            predictions, result = evaluate_text(rows, column)
            all_predictions.extend(predictions)
            report["variants"][column] = result
            print(f"  Accuracy={result['overall_out_of_fold']['accuracy']:.4f}; "
                  f"macro F1={result['overall_out_of_fold']['macro_f1']:.4f}", flush=True)
        args.outdir.mkdir(parents=True, exist_ok=True)
        with predictions_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(all_predictions[0]), delimiter=";")
            writer.writeheader()
            writer.writerows(all_predictions)
        report["predictions_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, csv.Error, ConvergenceWarning) as exc:
        print(f"Text-classifier evaluation failed: {exc}")
        return 1
    print(f"Predictions: {predictions_path}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
