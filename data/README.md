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
