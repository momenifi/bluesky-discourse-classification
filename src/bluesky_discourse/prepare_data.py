"""Prepare unique training items and connected groups from the gold release."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .validate_data import validate


LINK_COLUMNS = ("uri", "source_uri", "parent_uri", "root_uri")


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


def prepare(csv_path: Path, manifest_path: Path) -> tuple[list[dict], dict]:
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
    }
    return kept, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/final_gold_dataset_v2.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("data/final_gold_dataset_v2_manifest.json"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    csv_output = args.outdir / "training_data.csv"
    report_output = args.outdir / "preparation_report.json"
    if {csv_output.resolve(), report_output.resolve()} & {args.csv.resolve(), args.manifest.resolve()}:
        parser.error("Output paths must differ from input paths")
    try:
        rows, report = prepare(args.csv, args.manifest)
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
    print(f"Reproducibility report: {report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
