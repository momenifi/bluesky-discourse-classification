"""Prepare unique training items and connected groups from the gold release."""

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

from .validate_data import validate


LINK_COLUMNS = ("uri", "source_uri", "parent_uri", "root_uri")


def assign_folds(rows: list[dict], seed: int = 42) -> dict:
    """Assign each item to exactly one held-out fold, preserving groups."""
    import numpy
    import scipy
    import sklearn
    from sklearn.model_selection import StratifiedGroupKFold

    ordered = sorted(rows, key=lambda row: row["annotation_id"])
    if len({row["annotation_id"] for row in ordered}) != len(ordered):
        raise ValueError("Fold assignment requires unique annotation IDs")
    if len({row["uri"] for row in ordered}) != len(ordered):
        raise ValueError("Fold assignment requires unique post URIs")
    labels = [row["final_label"] for row in ordered]
    groups = [row["group_id"] for row in ordered]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    assigned = set()
    summaries = []
    for fold, (train_indices, test_indices) in enumerate(splitter.split(
        [[0]] * len(ordered), labels, groups
    )):
        train_groups = {groups[index] for index in train_indices}
        test_groups = {groups[index] for index in test_indices}
        if train_groups & test_groups:
            raise ValueError("A connected group crosses train/test boundaries")
        test_labels = Counter(labels[index] for index in test_indices)
        if set(test_labels) != set(labels):
            raise ValueError(f"Fold {fold} is missing a label; review the split design")
        for index in test_indices:
            index = int(index)
            if index in assigned:
                raise ValueError("An item was assigned to multiple test folds")
            ordered[index]["fold"] = fold
            assigned.add(index)
        summaries.append({
            "fold": fold, "test_rows": len(test_indices), "train_rows": len(train_indices),
            "test_groups": len(test_groups),
            "test_label_counts": dict(sorted(test_labels.items())),
        })
    if len(assigned) != len(ordered):
        raise ValueError("Some items were not assigned to a test fold")
    assignments = [(row["annotation_id"], row["fold"]) for row in ordered]
    return {
        "method": "StratifiedGroupKFold", "n_splits": 5, "shuffle": True,
        "seed": seed, "input_order": "annotation_id ascending",
        "group_column": "group_id", "label_column": "final_label",
        "assignment_sha256": hashlib.sha256(json.dumps(assignments).encode("utf-8")).hexdigest(),
        "checks": {"each_item_in_one_test_fold": True, "no_group_overlap": True,
                   "unique_post_uris": True, "all_labels_in_every_test_fold": True},
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                     "numpy": numpy.__version__, "scipy": scipy.__version__},
        "summary": summaries,
    }


def model_text(row: dict) -> dict[str, str]:
    """Build inputs only from item type, post text, and available context."""
    text = " ".join(row["annotation_text"].split())
    marker = {"original_post": "[ORIGINAL]", "reply": "[REPLY]"}[row["item_type"]]
    text_only = f"{marker} {text}"
    context = " ".join(row["context_text"].split()) if row["item_type"] == "reply" else ""
    return {
        "text_only": text_only,
        "text_with_context": f"{text_only} [CONTEXT] {context}" if context else text_only,
    }


def connected_groups(rows: list[dict]) -> dict[str, str]:
    """Join all recorded links, including links on subsequently excluded rows."""
    parents = {}

    def find(uri):
        parents.setdefault(uri, uri)
        while parents[uri] != uri:
            parents[uri] = parents[parents[uri]]
            uri = parents[uri]
        return uri

    for row in rows:
        links = [row[column].strip() for column in LINK_COLUMNS if row[column].strip()]
        for uri in links[1:]:
            left, right = find(links[0]), find(uri)
            parents[max(left, right)] = min(left, right)
        find(links[0])
    return {
        row["annotation_id"]: "group_" + hashlib.sha256(
            find(row["uri"].strip()).encode("utf-8")
        ).hexdigest()
        for row in rows
    }


def prepare(csv_path: Path, manifest_path: Path, *, seed: int = 42) -> tuple[list[dict], dict]:
    # Preparation explicitly resolves root conflicts using all recorded links.
    validation = validate(csv_path, manifest_path, resolve_duplicate_roots=True)
    if not validation["valid"]:
        raise ValueError("Input validation failed: " + "; ".join(validation["errors"]))
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    if "group_id" in rows[0]:
        raise ValueError("Expected the source release, not an already prepared CSV")
    group_ids = connected_groups(rows)
    priority = {"paired_thread_source": 0, "paired_thread_reply": 1, "independent_reply": 2}
    kept_by_uri = {}
    for row in sorted(rows, key=lambda item: (priority[item["sample_group"]], item["annotation_id"])):
        kept_by_uri.setdefault(row["uri"], row)
    kept = [dict(row, group_id=group_ids[row["annotation_id"]]) for row in rows
            if kept_by_uri[row["uri"]]["annotation_id"] == row["annotation_id"]]
    for row in kept:
        row.update(model_text(row))
    fold_report = assign_folds(kept, seed=seed)
    exclusions = [{
        "excluded_annotation_id": row["annotation_id"],
        "retained_annotation_id": kept_by_uri[row["uri"]]["annotation_id"],
        "reason": "Repeated post URI; prefer paired sample, then lowest annotation ID",
    } for row in rows if kept_by_uri[row["uri"]]["annotation_id"] != row["annotation_id"]]
    group_sizes = Counter(row["group_id"] for row in kept)
    report = {
        "input_csv_sha256": validation["csv_sha256"],
        "input_manifest_sha256": validation["manifest_sha256"],
        "input_rows": len(rows), "training_rows": len(kept),
        "sample_group_counts": dict(sorted(Counter(row["sample_group"] for row in kept).items())),
        "label_counts": dict(sorted(Counter(row["final_label"] for row in kept).items())),
        "group_count": len(group_sizes), "largest_group": max(group_sizes.values()),
        "group_rule": "Connected components of uri, source_uri, parent_uri, root_uri across all source rows",
        "exclusions": exclusions,
        "model_text_columns": ["text_only", "text_with_context"],
        "model_text_rule": "Collapse whitespace; prefix [ORIGINAL] or [REPLY]; append [CONTEXT] only for replies with context",
        "replies_with_context": sum(row["item_type"] == "reply" and bool(row["context_text"].strip()) for row in kept),
        "folds": fold_report,
    }
    return kept, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/final_gold_dataset_v2.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/final_gold_dataset_v2_manifest.json"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    csv_output = args.outdir / "training_data.csv"
    report_output = args.outdir / "preparation_report.json"
    if {csv_output.resolve(), report_output.resolve()} & {args.csv.resolve(), args.manifest.resolve()}:
        parser.error("Output paths must differ from input paths")
    try:
        rows, report = prepare(args.csv, args.manifest, seed=args.seed)
        args.outdir.mkdir(parents=True, exist_ok=True)
        with csv_output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        report["training_csv_sha256"] = hashlib.sha256(csv_output.read_bytes()).hexdigest()
        report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        print(f"Preparation failed: {exc}")
        return 1
    print(f"Prepared {len(rows)} unique items in {report['group_count']} connected groups")
    for item in report["exclusions"]:
        print(f"Excluded {item['excluded_annotation_id']}; retained {item['retained_annotation_id']}")
    print(f"Training CSV: {csv_output}")
    for fold in report["folds"]["summary"]:
        print(f"Fold {fold['fold']}: {fold['test_rows']} test / {fold['train_rows']} train items")
    print(f"Reproducibility report: {report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
