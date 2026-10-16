"""Build a separate reviewed gold release from immutable v2 data and saved decisions."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .validate_data import LABELS, validate


REVIEW_COLUMN = "new_label_Contextualization"
COMMENT_COLUMN = "Comment_on_"
KNOWN_DOI_EDIT_IDS = {"gold_0043", "gold_0194", "gold_0275", "gold_0440", "gold_0444", "gold_0483"}
PARENT_INSPECTED_IDS = {"gold_0171", "gold_0403", "gold_0439", "gold_0444", "gold_0470", "gold_0476"}


def read_table(path: Path, *, review: bool = False) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter=";")
        header = next(reader)
        records = [record for record in reader if any(record)]
    if any(len(record) != len(header) for record in records):
        raise ValueError(f"Inconsistent column counts: {path}")
    if any(not name for name in header):
        if not review or any(record[index].strip() for record in records
                             for index, name in enumerate(header) if not name):
            raise ValueError(f"Unexpected unnamed column content: {path}")
    named = [name for name in header if name]
    if len(named) != len(set(named)):
        raise ValueError(f"Duplicate named columns: {path}")
    return named, [{name: record[index] for index, name in enumerate(header) if name}
                   for record in records]


def build_rows(original_columns: list[str], original: list[dict], reviewed: list[dict]) -> tuple[list[dict], dict]:
    original_by_id = {row["annotation_id"]: row for row in original}
    reviewed_by_id = {row["annotation_id"]: row for row in reviewed}
    if len(original_by_id) != len(original) or len(reviewed_by_id) != len(reviewed):
        raise ValueError("Annotation IDs must be unique")
    if original_by_id.keys() != reviewed_by_id.keys():
        raise ValueError("Review IDs must match the entire v2 release")
    if any(REVIEW_COLUMN not in row or COMMENT_COLUMN not in row for row in reviewed):
        raise ValueError("Missing review label or comment column")
    expected_review = {row["annotation_id"] for row in original if row["final_label"] == "Contextualization"}
    if len(expected_review) != 29:
        raise ValueError("Expected 29 original Contextualization items")
    decisions = []
    ignored_metadata_edits = []
    output = []
    for row in original:
        item_id = row["annotation_id"]
        review = reviewed_by_id[item_id]
        for column in original_columns:
            if column not in review:
                raise ValueError(f"Missing original column in review: {column}")
            if row[column] != review[column]:
                if (item_id in KNOWN_DOI_EDIT_IDS and column == "doi" and
                        row[column] == "10.1126" and review[column] == "101.126"):
                    ignored_metadata_edits.append({"annotation_id": item_id, "column": column,
                                                   "review_value": review[column], "preserved_v2_value": row[column]})
                else:
                    raise ValueError(f"Unexpected original-field edit: {item_id}, {column}")
        decision = review[REVIEW_COLUMN].strip()
        comment = review[COMMENT_COLUMN].strip()
        if item_id in expected_review:
            if decision not in LABELS:
                raise ValueError(f"Missing or invalid review label: {item_id}")
            decisions.append({"annotation_id": item_id, "old_label": row["final_label"],
                              "new_label": decision, "changed": decision != row["final_label"],
                              "comment": comment, "additional_parent_context_inspected": item_id in PARENT_INSPECTED_IDS})
        elif decision or comment:
            raise ValueError(f"Review data outside the agreed Contextualization audit: {item_id}")
        revised = dict(row)
        revised.update(final_label_v2=row["final_label"], final_label_source_v2=row["final_label_source"],
                       new_label_Contextualization=decision, Comment_on_=comment,
                       additional_parent_context_inspected=str(item_id in expected_review and item_id in PARENT_INSPECTED_IDS))
        if decision and decision != row["final_label"]:
            revised["final_label"] = decision
            revised["final_label_source"] = "posthoc_handbook_review"
        output.append(revised)
    report = {
        "reviewed_rows": len(decisions), "changed_rows": sum(item["changed"] for item in decisions),
        "retained_review_labels": sum(not item["changed"] for item in decisions),
        "reviewed_annotations": decisions, "ignored_review_metadata_edits": ignored_metadata_edits,
        "final_label_distribution": dict(sorted(Counter(row["final_label"] for row in output).items())),
        "final_label_source_counts": dict(sorted(Counter(row["final_label_source"] for row in output).items())),
    }
    return output, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/final_gold_dataset_v2.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/final_gold_dataset_v2_manifest.json"))
    parser.add_argument("--review", type=Path, default=Path("data/contextualization_review.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("data"))
    args = parser.parse_args()
    csv_output = args.outdir / "final_gold_dataset_v3.csv"
    manifest_output = args.outdir / "final_gold_dataset_v3_manifest.json"
    if {csv_output.resolve(), manifest_output.resolve()} & {args.source.resolve(), args.manifest.resolve(), args.review.resolve()}:
        parser.error("Output paths must differ from input paths")
    if csv_output.exists() or manifest_output.exists():
        parser.error("v3 outputs already exist; choose a different --outdir to preserve the release")
    try:
        validation = validate(args.source, args.manifest, resolve_duplicate_roots=True)
        if not validation["valid"]:
            raise ValueError("Source v2 validation failed: " + "; ".join(validation["errors"]))
        columns, original = read_table(args.source)
        _, reviewed = read_table(args.review, review=True)
        rows, report = build_rows(columns, original, reviewed)
        report.update({
            "dataset": "Bluesky scholarly discourse gold dataset v3", "version": 3,
            "rows": len(rows), "unique_annotation_ids": len({row["annotation_id"] for row in rows}),
            "unique_post_uris": len({row["uri"] for row in rows}), "allowed_labels": sorted(LABELS),
            "review_method": "Researcher handbook review with assistant support after model-error inspection; not blind or an independent rater",
            "review_scope": "Only the 29 items originally labeled Contextualization",
            "metadata_rule": "All original metadata and annotator fields come exclusively from validated v2",
            "evaluation_status": "Post-hoc label revision; future scores are exploratory, not an untouched test estimate",
            "context_note": "Additional parent replies inspected for six listed items; source texts and context_text are unchanged",
            "training_status": "500 annotation records; duplicate gold_0419 remains in the release and must still be excluded for training",
            "fold_status": "No new folds assigned; reuse the saved v2 item/group/fold mapping for subsequent comparison",
            "inputs": {"v2_csv_sha256": validation["csv_sha256"], "v2_manifest_sha256": validation["manifest_sha256"],
                       "review_csv_sha256": hashlib.sha256(args.review.read_bytes()).hexdigest()},
            "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        })
        args.outdir.mkdir(parents=True, exist_ok=True)
        with csv_output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        report["outputs"] = {"csv": {"path": csv_output.name, "sha256": hashlib.sha256(csv_output.read_bytes()).hexdigest()}}
        manifest_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, StopIteration, csv.Error) as exc:
        print(f"v3 build failed: {exc}")
        return 1
    print(f"Created {len(rows)} v3 annotation records")
    print(f"Reviewed {report['reviewed_rows']}; changed {report['changed_rows']}; retained {report['retained_review_labels']}")
    print(f"Contextualization remaining: {report['final_label_distribution'].get('Contextualization', 0)}")
    print(f"Ignored confirmed accidental DOI edits in review: {len(report['ignored_review_metadata_edits'])}")
    print(f"Dataset: {csv_output}")
    print(f"Manifest and change log: {manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
