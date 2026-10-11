# Step 1: validate the gold dataset

Run from the repository root on GESIS after activating your environment:

```bash
python -m pip install -e .
python -m bluesky_discourse.validate_data
```

The validator uses Python's standard library, including the CSV parser with
the release's semicolon delimiter and UTF-8 BOM handling. It preserves inputs.

It checks the release hash, 500 rows, unique annotation IDs, seven allowed
labels, label counts, the 200/200/100 sample design, required text and thread
IDs, English language hints, and final-label provenance counts. Language hints
are checked as metadata; this step does not perform language detection.

Repeated post URIs are reported. Conflicting text, labels, or root identifiers
block training. Missing reply context is a warning; original posts may have
empty context.

The report is `outputs/data_validation.json`, ignored by Git. It records
SHA-256 hashes and counts without including post text. Exit code 0 means pass,
1 means validation errors, and 2 means invalid inputs or an execution error.

## Issue in the current release

`gold_0411` and `gold_0419` refer to the same post and have identical text and
the label Evaluation. One is a paired reply, the other an independent reply,
but they have different `root_uri` values. The current validator therefore
returns exit code 1. Resolve this in a documented preparation step before
assigning folds; otherwise the same post could appear in training and testing.
The source release is preserved unchanged.
