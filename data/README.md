# Gold dataset setup

The private repository includes `final_gold_dataset_v2.csv` and
`final_gold_dataset_v2_manifest.json`. Cloning retrieves both files.
The CSV contains posts, replies, context, annotations, and final labels.
Preserve the source CSV unchanged. Write derived files to `outputs/`.

An alternative copy of the private release is available from S3. With AWS CLI installed
and a profile authorized to read the release, run from the repository root:

```bash
aws s3 cp s3://discoure-bluesky-063093445177-eu-central-1-an/bluesky-discourse/releases/gold-v2/final_gold_dataset_v2/final_gold_dataset_v2.csv data/final_gold_dataset_v2.csv
aws s3 cp s3://discoure-bluesky-063093445177-eu-central-1-an/bluesky-discourse/releases/gold-v2/final_gold_dataset_v2/final_gold_dataset_v2_manifest.json data/final_gold_dataset_v2_manifest.json
```

Use `final_label` as the training target. Exclude annotator labels, LLM
suggestions, adjudication fields, and label provenance from model features.

Expected dataset: 500 unique annotation IDs, no missing final labels, and seven
discourse categories. Step 2 will verify these properties and record the input
file hash before we create training inputs.

## Reviewed v3 release

`final_gold_dataset_v2.csv` is restored to the exact original release bytes.
Do not edit it to record further review decisions.

- `contextualization_review.csv`: preserved snapshot of the researcher's edited
  export with the 29 decisions and comments. Its metadata is not authoritative:
  six confirmed accidental DOI edits are recorded and ignored by the builder.
- `final_gold_dataset_v3.csv`: 500 annotation records with 17 revised final
  labels and 12 remaining Contextualization items. Original metadata and
  annotator labels are preserved from v2.
- `final_gold_dataset_v3_manifest.json`: hashes, counts, the complete review
  decision log, comments, and review provenance.

These files are included in the private repository with byte-preserving Git
attributes. V3 is a post-hoc, assistant-supported handbook revision following
error analysis, not an independent or blind annotation round.

Rebuild without overwriting the released files:

```bash
python -m bluesky_discourse.build_gold_v3 --outdir outputs/rebuild_gold_v3
```

Prepare v3 while retaining the existing v2 folds:

```bash
python -m bluesky_discourse.prepare_data --gold-version v3
```

This requires the original `outputs/training_data.csv` and preparation report.
It writes separate outputs under `outputs/v3/`. The standalone validator remains
specific to v2; v3 preparation performs its own release and reference checks.
