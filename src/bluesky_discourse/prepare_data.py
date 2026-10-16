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


def prepare_v3(csv_path: Path, manifest_path: Path, v2_csv: Path, v2_report_path: Path) -> tuple[list[dict], dict]:
    """Reuse the validated v2 sample, features, groups, and folds; revise labels."""
    from .baseline import load_prepared_data, check_evaluation_rows
    from .build_gold_v3 import read_table
    from .validate_data import LABELS

    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
    csv_hash = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if manifest["version"] != 3 or csv_hash != manifest["outputs"]["csv"]["sha256"]:
        raise ValueError("v3 source version or CSV hash does not match its manifest")
    _, source = read_table(csv_path)
    by_id = {row["annotation_id"]: row for row in source}
    if len(by_id) != len(source) or len(source) != manifest["rows"]:
        raise ValueError("v3 IDs are not unique or row count differs from the manifest")
    labels = Counter(row["final_label"] for row in source)
    if set(labels) != LABELS or labels != manifest["final_label_distribution"]:
        raise ValueError("v3 label counts do not match the release manifest")
    for decision in manifest["reviewed_annotations"]:
        reviewed = by_id[decision["annotation_id"]]
        if (reviewed["final_label"] != decision["new_label"] or
                reviewed["final_label_v2"] != decision["old_label"]):
            raise ValueError("v3 labels differ from the recorded review decisions")
    old_rows, old_provenance = load_prepared_data(v2_csv, v2_report_path)
    old_report = json.loads(v2_report_path.read_text(encoding="utf-8-sig"))
    if (old_report["input_csv_sha256"] != manifest["inputs"]["v2_csv_sha256"] or
            old_report["input_manifest_sha256"] != manifest["inputs"]["v2_manifest_sha256"]):
        raise ValueError("Reference preparation did not use the original v2 release")
    assignments = sorted((row["annotation_id"], int(row["fold"])) for row in old_rows)
    assignment_hash = hashlib.sha256(json.dumps(assignments).encode("utf-8")).hexdigest()
    if assignment_hash != old_report["folds"]["assignment_sha256"]:
        raise ValueError("Reference fold assignments do not match their recorded hash")
    selected_ids = {row["annotation_id"] for row in old_rows}
    excluded_ids = {item["excluded_annotation_id"] for item in old_report["exclusions"]}
    if selected_ids & excluded_ids or selected_ids | excluded_ids != set(by_id):
        raise ValueError("Reference sample and exclusions do not cover exactly the v3 release")
    group_ids = connected_groups(source)
    prepared = []
    changed_ids = []
    derived_columns = {"fold", "group_id", "text_only", "text_with_context", "final_label", "final_label_source"}
    for old in old_rows:
        row = dict(by_id[old["annotation_id"]])
        for column, value in old.items():
            if column not in derived_columns and row.get(column) != value:
                raise ValueError(f"Original metadata differs from v2: {old['annotation_id']}, {column}")
        if row["final_label_v2"] != old["final_label"] or row["final_label_source_v2"] != old["final_label_source"]:
            raise ValueError("Stored v2 label provenance differs from the reference preparation")
        if group_ids[row["annotation_id"]] != old["group_id"]:
            raise ValueError("Connected groups differ from the frozen v2 groups")
        text = model_text(row)
        if any(text[column] != old[column] for column in text):
            raise ValueError("Prepared model text differs from v2")
        row.update(text)
        row.update(group_id=old["group_id"], fold=int(old["fold"]))
        if row["final_label"] != old["final_label"]:
            changed_ids.append(row["annotation_id"])
        prepared.append(row)
    expected_changed = {item["annotation_id"] for item in manifest["reviewed_annotations"]
                        if item["changed"] and item["annotation_id"] in selected_ids}
    if set(changed_ids) != expected_changed:
        raise ValueError("Changed labels do not match the v3 review change log")
    check_evaluation_rows(prepared)
    all_labels = set(labels)
    fold_summaries = []
    for fold in range(5):
        test = [row for row in prepared if row["fold"] == fold]
        train = [row for row in prepared if row["fold"] != fold]
        test_counts = Counter(row["final_label"] for row in test)
        if {row["final_label"] for row in train} != all_labels:
            raise ValueError(f"Frozen fold {fold} lacks a training label; review the evaluation protocol")
        fold_summaries.append({
            "fold": fold, "test_rows": len(test), "train_rows": len(train),
            "test_groups": len({row["group_id"] for row in test}),
            "test_label_counts": {label: test_counts[label] for label in sorted(all_labels)},
            "missing_test_labels": sorted(all_labels - set(test_counts)),
        })
    group_sizes = Counter(row["group_id"] for row in prepared)
    folds = {
        "method": "Frozen v2 fold assignments", "n_splits": 5,
        "origin_method": old_report["folds"]["method"], "origin_seed": old_report["folds"]["seed"],
        "assignment_sha256": assignment_hash, "group_column": "group_id", "label_column": "final_label",
        "checks": {"each_item_in_one_test_fold": True, "no_group_overlap": True, "unique_post_uris": True,
                   "all_labels_in_every_test_fold": all(not item["missing_test_labels"] for item in fold_summaries)},
        "summary": fold_summaries,
    }
    report = {
        "gold_version": 3, "input_csv_sha256": csv_hash,
        "input_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "input_rows": len(source), "training_rows": len(prepared),
        "sample_group_counts": dict(sorted(Counter(row["sample_group"] for row in prepared).items())),
        "label_counts": dict(sorted(Counter(row["final_label"] for row in prepared).items())),
        "group_count": len(group_sizes), "largest_group": max(group_sizes.values()),
        "group_rule": old_report["group_rule"], "exclusions": old_report["exclusions"],
        "model_text_columns": old_report["model_text_columns"], "model_text_rule": old_report["model_text_rule"],
        "replies_with_context": sum(row["item_type"] == "reply" and bool(row["context_text"].strip()) for row in prepared),
        "reference_v2": old_provenance, "changed_training_labels": len(changed_ids),
        "changed_annotation_ids": sorted(changed_ids), "folds": folds,
        "checks": {"same_item_ids_as_v2": True, "same_groups_as_v2": True,
                   "same_model_text_as_v2": True, "same_folds_as_v2": True},
        "evaluation_status": "Exploratory post-hoc label revision using unchanged v2 folds",
        "preparation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return prepared, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-version", choices=("v2", "v3"), default="v2")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--v2-training-csv", type=Path, default=Path("outputs/training_data.csv"))
    parser.add_argument("--v2-preparation-report", type=Path, default=Path("outputs/preparation_report.json"))
    args = parser.parse_args()
    args.csv = args.csv or Path(f"data/final_gold_dataset_{args.gold_version}.csv")
    args.manifest = args.manifest or Path(f"data/final_gold_dataset_{args.gold_version}_manifest.json")
    args.outdir = args.outdir or (Path("outputs/v3") if args.gold_version == "v3" else Path("outputs"))
    if args.gold_version == "v3" and args.seed is not None:
        parser.error("V3 reuses existing folds; --seed is only valid for v2")
    csv_output = args.outdir / "training_data.csv"
    report_output = args.outdir / "preparation_report.json"
    if {csv_output.resolve(), report_output.resolve()} & {args.csv.resolve(), args.manifest.resolve()}:
        parser.error("Output paths must differ from input paths")
    if args.gold_version == "v3" and {csv_output.resolve(), report_output.resolve()} & {
        args.v2_training_csv.resolve(), args.v2_preparation_report.resolve()
    }:
        parser.error("V3 outputs must not overwrite the v2 reference preparation")
    try:
        if args.gold_version == "v3":
            rows, report = prepare_v3(args.csv, args.manifest, args.v2_training_csv, args.v2_preparation_report)
        else:
            rows, report = prepare(args.csv, args.manifest, seed=42 if args.seed is None else args.seed)
        args.outdir.mkdir(parents=True, exist_ok=True)
        with csv_output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        report["training_csv_sha256"] = hashlib.sha256(csv_output.read_bytes()).hexdigest()
        report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError, StopIteration, csv.Error) as exc:
        print(f"Preparation failed: {exc}")
        return 1
    print(f"Prepared {len(rows)} unique items in {report['group_count']} connected groups")
    if args.gold_version == "v3":
        print(f"Reused frozen v2 folds; revised {report['changed_training_labels']} training labels")
    for item in report["exclusions"]:
        print(f"Excluded {item['excluded_annotation_id']}; retained {item['retained_annotation_id']}")
    print(f"Training CSV: {csv_output}")
    for fold in report["folds"]["summary"]:
        print(f"Fold {fold['fold']}: {fold['test_rows']} test / {fold['train_rows']} train items")
        if fold.get("missing_test_labels"):
            print(f"  No test examples for: {', '.join(fold['missing_test_labels'])}; folds remain frozen")
    print(f"Reproducibility report: {report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
