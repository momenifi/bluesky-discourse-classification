"""Validate the released gold CSV without changing its contents."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


LABELS = {
    "Acknowledgment", "Announcement", "Contextualization", "Critique",
    "Evaluation", "Explanation", "Other",
}
EXPECTED_GROUPS = {
    "paired_thread_source": 200,
    "paired_thread_reply": 200,
    "independent_reply": 100,
}
REQUIRED_COLUMNS = {
    "annotation_id", "sample_group", "item_type", "language_hint",
    "annotation_text", "context_text", "uri", "root_uri", "final_label",
    "final_label_source", "source_uri", "parent_uri",
}


def validate(csv_path: Path, manifest_path: Path, *, resolve_duplicate_roots: bool = False) -> dict:
    """Return a deterministic report; errors block later training steps."""
    csv_bytes = csv_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    report = {
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "errors": [],
        "warnings": [],
    }
    errors = report["errors"]
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        headers = reader.fieldnames or []
        missing = sorted(REQUIRED_COLUMNS - set(headers))
        if missing:
            errors.append(f"Missing columns: {missing}")
        if len(headers) != len(set(headers)):
            errors.append("Duplicate CSV column names")
        rows = list(reader)
    report["rows"] = len(rows)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        errors.append("CSV records have inconsistent column counts")
    expected_hash = manifest["outputs"]["csv"]["sha256"].lower()
    if report["csv_sha256"] != expected_hash:
        errors.append("CSV SHA-256 does not match the release manifest")
    if len(rows) != 500 or len(rows) != manifest["rows"]:
        errors.append("Row count must be 500 and match the manifest")
    if errors:
        report["valid"] = False
        return report

    for row in rows:
        item_id = row["annotation_id"]
        for column in ("annotation_id", "annotation_text", "uri", "root_uri", "final_label"):
            if not row[column].strip():
                errors.append(f"{item_id}: empty {column}")
        if row["final_label"] not in LABELS:
            errors.append(f"{item_id}: invalid final_label")
        if row["language_hint"] != "en":
            errors.append(f"{item_id}: language_hint is not en")
        expected_type = {
            "paired_thread_source": "original_post",
            "paired_thread_reply": "reply",
            "independent_reply": "reply",
        }.get(row["sample_group"])
        if row["item_type"] != expected_type:
            errors.append(f"{item_id}: item_type does not match sample_group")
        if row["item_type"] == "reply" and not row["context_text"].strip():
            report["warnings"].append(f"{item_id}: reply has no context_text")

    ids = Counter(row["annotation_id"] for row in rows)
    report["unique_annotation_ids"] = len(ids)
    if len(ids) != manifest["unique_annotation_ids"] or any(n > 1 for n in ids.values()):
        errors.append("Annotation IDs are not unique or do not match the manifest")
    labels = dict(sorted(Counter(row["final_label"] for row in rows).items()))
    report["label_counts"] = labels
    if labels != manifest["final_label_distribution"]:
        errors.append("Final-label counts differ from the manifest")
    groups = dict(sorted(Counter(row["sample_group"] for row in rows).items()))
    report["sample_group_counts"] = groups
    if groups != EXPECTED_GROUPS:
        errors.append("Sample groups do not match the 200/200/100 design")
    sources = Counter(row["final_label_source"] for row in rows)
    report["final_label_source_counts"] = dict(sorted(sources.items()))
    if sources != Counter({
        "human_majority": manifest["majority_resolved_rows"],
        "human_adjudication_with_llm_support": manifest["human_adjudicated_rows"],
    }):
        errors.append("Final-label provenance counts differ from the manifest")

    uri_rows = defaultdict(list)
    for row in rows:
        uri_rows[row["uri"]].append(row)
    report["unique_post_uris"] = len(uri_rows)
    report["unique_root_uris"] = len({row["root_uri"] for row in rows})
    report["duplicate_posts"] = []
    for records in uri_rows.values():
        if len(records) < 2:
            continue
        item_ids = [row["annotation_id"] for row in records]
        same_root = len({row["root_uri"] for row in records}) == 1
        same_text = len({row["annotation_text"] for row in records}) == 1
        same_label = len({row["final_label"] for row in records}) == 1
        report["duplicate_posts"].append({
            "annotation_ids": item_ids,
            "same_root_uri": same_root,
            "same_text": same_text,
            "same_label": same_label,
        })
        report["warnings"].append(f"Repeated post URI: {item_ids}")
        if not same_root and not resolve_duplicate_roots:
            errors.append(f"{item_ids}: repeated post has different root_uri values; resolve before splitting")
        if not same_text or not same_label:
            errors.append(f"{item_ids}: repeated post has conflicting text or labels")
    report["valid"] = not errors
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/final_gold_dataset_v2.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/final_gold_dataset_v2_manifest.json"))
    parser.add_argument("--report", type=Path, default=Path("outputs/data_validation.json"))
    args = parser.parse_args()
    # Guard against accidentally replacing the input dataset with a report.
    if args.report.resolve() in {args.csv.resolve(), args.manifest.resolve()}:
        parser.error("The report path must differ from the input paths")
    try:
        report = validate(args.csv, args.manifest)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        print(f"Validation could not run: {exc}")
        return 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Rows: {report['rows']}")
    print(f"CSV SHA-256: {report['csv_sha256']}")
    for message in report["warnings"]:
        print(f"WARNING: {message}")
    for message in report["errors"]:
        print(f"ERROR: {message}")
    print(f"Report: {args.report}")
    print("PASS" if report["valid"] else "FAIL: resolve errors before training")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
