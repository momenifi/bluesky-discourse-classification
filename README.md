# Bluesky discourse classification

Training workspace for classifying scholarly discourse in 500 English Bluesky
posts and replies. Implementation and training are carried out on the GESIS
server, step by step.

## Files

- `pyproject.toml`: Python package setup; training dependencies will be added
  when the corresponding code is implemented.
- `src/bluesky_discourse/`: package skeleton for the training code.
- `data/`: final gold CSV, release manifest, and dataset instructions.
- `docs/ANNOTATION_METHODOLOGY.md`: annotation and final-label procedure.
- `docs/annotation_handbook_v2.docx`: definitions of the annotation categories.
- `.gitignore`: includes the final CSV and manifest; excludes other local data,
  credentials, environments, and results.

The methodology references scripts in the original collection repository.
Those scripts are not required to train from the released gold dataset.

## GESIS setup

Publish this repository to a private Git remote, then run on the server:

```bash
git clone <repository-url>
cd bluesky-discourse-classification
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Python 3.10 or newer is required. The final dataset and manifest are included
in `data/` and will be retrieved when cloning the private repository.

## Training steps

1. Validate the CSV: expected rows, unique IDs, labels, text, and thread IDs.
2. Construct model input from post text and reply context.
3. Assign folds that keep related posts and replies together.
4. Evaluate a majority-class baseline.
5. Train TF-IDF plus logistic regression.
6. Inspect metrics and classification errors.
7. Compare a pretrained transformer if warranted.

The target is `final_label`: Acknowledgment, Announcement, Contextualization,
Critique, Evaluation, Explanation, or Other. Use text and conversational
context as features. Exclude annotator labels, LLM suggestions, adjudication
fields, and label provenance.

Fit each vectorizer only on its training fold. Report macro F1, accuracy,
weighted F1, per-class metrics, and a confusion matrix. Record dataset hashes,
dependency versions, fold assignments, seeds, and run settings for reproduction.

## Validate and prepare data

Run from the repository root in the activated environment:

```bash
python -m bluesky_discourse.validate_data
python -m bluesky_discourse.prepare_data
```

Validation flags one duplicate in the unchanged 500-row release: `gold_0411`
and `gold_0419` have the same post URI, text, and final label, but different
collection contexts and root identifiers. Its exit code is therefore 1.

Preparation validates the release hash and contents, then resolves this issue.
For repeated post URIs with identical text and labels, it prefers paired items
over independent replies, breaking ties by annotation ID. It retains
`gold_0411` and excludes `gold_0419`, producing 499 unique training items:
200 originals, 200 paired replies, and 99 independent replies. Conflicting
text or labels block preparation.

The two preparation outputs are `outputs/training_data.csv` and
`outputs/preparation_report.json`. The source release stays unchanged. The
report records hashes, counts, and exclusions.

The prepared CSV adds `group_id`, connecting recorded post, source, parent,
and root URIs across all source rows, including the excluded duplicate. This
conservatively keeps linked items in the same future fold even when recorded
root identifiers differ. These groups represent recorded connections, not
verified canonical Bluesky thread roots. `group_id` is for splitting, not a
model feature. No folds or text features have been created yet.
