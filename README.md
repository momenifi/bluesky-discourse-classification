# Bluesky discourse classification

Training workspace for classifying scholarly discourse in 500 English Bluesky
posts and replies. Implementation and training are carried out on the GESIS
server, step by step.

## Files

- `pyproject.toml`: Python package setup and pinned scikit-learn dependency.
- `src/bluesky_discourse/`: validation, preparation, baseline, TF-IDF, and transformer code.
- `data/`: final gold CSV, release manifest, and dataset instructions.
- `docs/ANNOTATION_METHODOLOGY.md`: annotation and final-label procedure.
- `docs/annotation_handbook_v2.docx`: definitions of the annotation categories.
- `.gitignore`: includes the final CSV and manifest; excludes other local data,
  credentials, environments, and results.
- `.gitattributes`: preserves the source CSV and manifest bytes across platforms.

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

Python 3.11 or newer is required. The final dataset and manifest are included
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

The CSV is parsed with a semicolon delimiter, UTF-8 BOM handling, and Python's
CSV parser, which handles quoted fields and embedded newlines. Validation
uses only the Python standard library. Preparation now additionally
uses scikit-learn 1.8.0 for fold assignment; install dependencies with
`python -m pip install -e .` before running it.

Validation checks required columns, consistent record widths, the release
SHA-256 hash, 500 rows, unique annotation IDs, nonempty item text and post/root
URIs, allowed final labels and their manifest counts, the 200/200/100 sample
design, item types, and final-label provenance counts (442 majority, 58 human
adjudications). It checks that `language_hint` is `en`; this does not independently
detect or verify the language. Empty context is allowed for originals and is
a warning for replies.

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

Standalone validation also writes `outputs/data_validation.json`. Exit codes
are 0 for pass, 1 for failed checks, and 2 for invalid inputs or execution errors.
The expected duplicate error in this source release must not be interpreted as
a successful standalone validation. Preparation explicitly resolves differing
root identifiers through connected grouping, while retaining other checks.

The prepared CSV adds `group_id`, connecting recorded post, source, parent,
and root URIs across all source rows, including the excluded duplicate. This
conservatively keeps linked items in the same future fold even when recorded
root identifiers differ. These groups represent recorded connections, not
verified canonical Bluesky thread roots. `group_id` is for splitting, not a
model feature. Numerical features are learned separately within each training
fold during text-classifier evaluation, not during preparation.

Grouping treats every nonempty `uri`, `source_uri`, `parent_uri`, and `root_uri`
in a row as linked. It merges overlapping links transitively across rows.
Each `group_id` is `group_` followed by the SHA-256 hash of the lexicographically
smallest URI in that component. Group IDs are independent of input row order.
This rule uses recorded links only; missing links can leave related discussions
unconnected. The grouping does not guarantee independence by author or publication.

## Model text

The same preparation command adds two columns to `outputs/training_data.csv`:

| Column | Input |
| --- | --- |
| `text_only` | `[ORIGINAL]` or `[REPLY]`, followed by the item text |
| `text_with_context` | The same text, plus `[CONTEXT]` and available reply context |

Original posts have identical inputs in both columns. Replies without context
also have identical inputs; no empty context marker is added. Context is the
available collection-source text, not necessarily the immediate parent reply.

Whitespace is collapsed to single spaces. Words, case, punctuation, URLs,
mentions, hashtags, negation, and emojis are preserved. No tokens or vocabulary
are learned at this stage. We will compare these two inputs with TF-IDF later.
Do not concatenate the entire CSV into model input: use only the selected text
column, with `final_label` as target and `group_id` for splitting.

Re-running preparation refreshes the same CSV and report, recording the text
construction rule and output hash. Grouped folds are described below.

## Completed preparation record

Verified on 2026-10-11. The original annotation release contains 500 annotation
records but 499 distinct post URIs. Annotation agreement statistics remain
those of the original 500-record release; training uses the derived 499 items.
No replacement item was sampled or annotated.

| Property | Prepared dataset |
| --- | ---: |
| Original posts | 200 |
| Paired replies | 200 |
| Independent replies | 99 |
| Total unique post URIs | 499 |
| Connected groups | 292 |
| Largest connected group | 4 items |
| Replies with added context | 299 |

| Final label | Source release | Prepared dataset |
| --- | ---: | ---: |
| Acknowledgment | 45 | 45 |
| Announcement | 119 | 119 |
| Contextualization | 29 | 29 |
| Critique | 50 | 50 |
| Evaluation | 56 | 55 |
| Explanation | 154 | 154 |
| Other | 47 | 47 |

The prepared CSV retains source metadata and annotations for traceability and
adds `group_id`, `text_only`, `text_with_context`, and `fold`. Retaining metadata
does not make it a model feature. The source CSV and manifest are copied without
modification into `data/` and included in the private Git repository. Generated
outputs are excluded from Git and can be recreated with the commands above.

Git line-ending conversion is disabled for the source CSV and manifest using
`.gitattributes`. When committing this setting, re-stage these two files so a
previously normalized Git version is replaced by the byte-preserved release:

```bash
git add .gitattributes
git add --renormalize data/final_gold_dataset_v2.csv data/final_gold_dataset_v2_manifest.json
```

Verified SHA-256 values before fold assignment was added (the source CSV and
manifest remain unchanged; the current prepared hash is recorded in the report):

```text
Source CSV:    2f6e04a7c6c509d04c6a7784aa3e99d39dc00113f1a6301ff9568c29b8d4feff
Manifest:      aea6f5d05bf412eb649d28e1aa13e932a99e87f24ee13e8e5ca6faa0c32ac225
Prepared CSV:  faf4d342ddb3de5424a19695511cc1653a5d628e6cc3d6f65b2a09ba2085b921
```

Run the verification suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

The verification suite covers valid release parsing, detection of changed
input hashes, conflicting duplicate roots, deduplication without input changes,
grouping through the excluded row, order-independent groups, text preservation,
exclusion of label metadata from text, and missing/unused context handling.
The actual prepared release was additionally checked for 499 nonempty text
inputs and identical text variants for original posts.

For each GESIS run, retain the preparation report alongside its generated CSV
and record the code revision and environment in the same ignored output folder:

```bash
git rev-parse HEAD > outputs/code_revision.txt
git status --short > outputs/working_tree.txt
python --version > outputs/python_version.txt
python -m pip freeze > outputs/environment.txt
```

Use a committed, clean code revision for reproducible experiments; a commit
hash alone does not capture uncommitted edits. Preparation is deterministic and
uses no random sampling during deduplication or text construction. Fold
assignment uses the fixed seed and recorded dependency versions described below.
The majority baseline and first text classifier are implemented below.

## Five evaluation folds

Re-run `python -m bluesky_discourse.prepare_data` to refresh the same two
preparation outputs, now including folds. No separate fold files are created.

`outputs/training_data.csv` stores each item's held-out fold in the `fold`
column, numbered 0 through 4. For evaluation round k, use `fold == k` for
testing and `fold != k` for training. Each item is tested once. Use these
same assignments for text-only and text-with-context comparisons.

The implementation is `assign_folds()` in `src/bluesky_discourse/prepare_data.py`.
It sorts input by annotation ID and uses
[StratifiedGroupKFold](https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
with `n_splits=5`, `shuffle=True`, and `random_state=42`. The groups are
`group_id` and the stratification labels are `final_label`. The algorithm aims
to balance label proportions subject to keeping groups intact; exact balance
is not guaranteed. The seed was fixed before any model evaluation.

`outputs/preparation_report.json`, under `folds`, records settings, installed
Python/scikit-learn/NumPy/SciPy versions, an assignment hash, and each fold's
train/test sizes, group counts, and test-label counts. The script checks
unique post URIs, assignment of every item exactly once, no connected group
crossing train/test boundaries, and all labels present in each test fold.
Tests also verify fixed-seed reproducibility under reordered input.

Verified on 2026-10-11: all nine tests passed, and the source release produced
the following held-out folds. All seven categories occur in each test fold.

| Fold | Test items | Training items |
| --- | ---: | ---: |
| 0 | 100 | 399 |
| 1 | 100 | 399 |
| 2 | 100 | 399 |
| 3 | 100 | 399 |
| 4 | 99 | 400 |

`--seed` permits an explicit alternative, but changing it creates a different
evaluation design and refreshes the outputs. Preserve the default assignments
for comparisons. Folds alone are not a separate validation set: future model
tuning must use only the training portion, with group-aware inner splits if
needed. Fit TF-IDF only on the training portion of each evaluation round.

## Majority baseline

The first evaluation deliberately ignores text. It measures what can be
achieved by always choosing the most frequent training label. This provides
a reference for the future TF-IDF plus logistic regression classifier; the
baseline itself is not logistic regression.

Run from the repository root after preparation:

```bash
python -m bluesky_discourse.baseline
```

Implementation: `evaluate()` in `src/bluesky_discourse/baseline.py`. For each
saved test fold k, it fits scikit-learn's
[DummyClassifier(strategy="most_frequent")](https://scikit-learn.org/1.8/modules/generated/sklearn.dummy.DummyClassifier.html)
using only `final_label` from the other four folds. A constant placeholder
feature is passed because no text or metadata features are needed. It then
predicts that training majority label for every item in fold k. In a tie,
scikit-learn selects the first tied label in its sorted class order.

The test labels are used only to score predictions. The majority is recomputed
separately for every training partition, not selected from the full dataset.
No TF-IDF, text classifier, tuning, or final deployment model is fitted here.
The baseline is deterministic and does not regenerate or modify folds.

The command checks that the prepared CSV matches the SHA-256 in its preparation
report, IDs and post URIs are unique, all five folds are present, and no group
crosses folds. It creates two files in the existing ignored `outputs/` folder:

- `baseline_predictions.csv`: one held-out prediction per item, containing ID,
  group, fold, actual label, predicted label, and whether it is correct.
- `baseline_report.json`: each fold's training label counts, majority choice,
  scores, combined held-out scores, per-label precision/recall/F1, and confusion
  matrix. It also records input/report/prediction/script hashes, the fold
  assignment hash, and installed Python/scikit-learn/NumPy/SciPy versions.

Scores include accuracy, macro F1, and weighted F1. Macro F1 averages the F1
scores of all seven labels equally; weighted F1 uses their test supports.
Undefined precision or F1 is set to zero. Overall scores combine the 499 held-out
predictions, each made without training on that item's fold. The JSON separately
reports the unweighted mean and population standard deviation across fold
scores; these are not confidence intervals and need not equal the combined score.
The confusion matrix uses actual labels as rows and predictions as columns,
with the recorded alphabetical label order.

Verified on 2026-10-15: Explanation was the training majority in every round.
The baseline correctly classified the 154 Explanation items and predicted
Explanation for the remaining 345 items as well.

| Metric | Combined held-out result |
| --- | ---: |
| Accuracy | 0.308617 (30.86%) |
| Macro F1 | 0.067381 |
| Weighted F1 | 0.145565 |

The high class imbalance allows 30.86% accuracy without inspecting any text;
the low macro F1 reflects failure to predict six categories. Compare the future
text model against these scores using the identical folds and metric settings.

All 12 tests passed. The baseline tests verify that a training majority can
differ from the full-dataset majority, every item has one held-out prediction,
groups crossing folds are rejected, and changing text or LLM metadata cannot
change baseline predictions. Existing validation and preparation tests also pass.

Source data and prepared fold assignments are unchanged by baseline evaluation.
To reproduce on GESIS, run setup, preparation, and baseline in that order,
and retain the environment and Git revision records described above. Re-running
the baseline refreshes its two outputs. The next section evaluates TF-IDF plus
logistic regression with all vocabulary learning restricted to training folds.

## TF-IDF plus logistic regression

Implementation: `src/bluesky_discourse/train_tfidf.py`. Run after setup and
preparation, from the repository root in the activated environment:

```bash
python -m bluesky_discourse.train_tfidf
```

This first experiment uses fixed settings, chosen before model scores were
examined. There is no hyperparameter search or selection of a new fold seed.
The command evaluates both `text_only` and `text_with_context` on the same
saved folds, using the same metric definitions as the majority baseline.

For each input variant and fold, `build_pipeline()` creates a fresh scikit-learn
[Pipeline](https://scikit-learn.org/1.8/modules/generated/sklearn.pipeline.Pipeline.html).
`fit()` receives only the four training folds' text and labels. It fits TF-IDF's
vocabulary and IDF weights, transforms the training text, and fits the classifier.
`predict()` transforms the held-out text using those fixed TF-IDF rules and
predicts labels. Test-only words are ignored; neither the vocabulary nor the
IDF weights are updated. Ten temporary pipelines are fitted in total: five
folds for each of the two input variants.

The [TfidfVectorizer](https://scikit-learn.org/1.8/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
uses word unigrams and bigrams (`ngram_range=(1, 2)`) and sublinear term frequency
(`sublinear_tf=True`, using 1 + log(count) for nonzero counts). Other defaults
include lowercasing, smoothed IDF, L2 vector normalization, no stop-word removal,
and no vocabulary-size limit. The default token pattern keeps word tokens of
at least two characters. Consequently punctuation and emojis preserved in the
prepared text are not standalone features in this word-based representation;
URLs and mentions are split into word tokens. Markers such as `[REPLY]` become
ordinary lowercased word tokens. The prepared source text is not changed.

[LogisticRegression](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html)
uses `C=1.0`, the default L2 regularization, `solver="lbfgs"`,
`class_weight="balanced"`, `max_iter=2000`, `tol=1e-4`, and `random_state=42`.
For these seven labels, it fits a multinomial classifier rather than a binary
majority-versus-rest rule. Balanced class weights are computed from each
training partition only. They increase minority-category contributions to the
training loss, without guaranteeing good minority-category performance.
A convergence warning stops evaluation instead of silently accepting an
unfinished fit.

Only the selected text column is passed as a feature. Annotation IDs, group
and fold IDs, labels, user metadata, LLM suggestions, and adjudication fields
are excluded from the feature input. The shared checks in `baseline.py` verify
prepared-file integrity and fold separation for both baseline and text models.

The command creates only two additional generated files in `outputs/`:

- `tfidf_predictions.csv`: 998 rows, one held-out prediction per item per variant
  (499 each), identified by `variant`, annotation ID, and fold. It includes
  actual and predicted labels and correctness, without copying post text.
- `tfidf_report.json`: fixed settings including vectorizer/classifier defaults,
  input/code/prediction hashes, fold assignment hash, dependency versions, and
  per-variant scores. It contains fold scores and vocabulary sizes, predicted
  label counts, per-category precision/recall/F1/support, and confusion matrices.

Overall scores use the combined held-out predictions. Fold means and population
standard deviations are also recorded separately, using all seven labels and
`zero_division=0`, matching the baseline. All comparisons below use the same
499 items and the same saved fold assignment hash.

First verified run on 2026-10-15:

| Model/input | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: |
| Majority baseline | 0.308617 | 0.067381 | 0.145565 |
| TF-IDF + logistic regression, text only | 0.525050 | 0.428154 | 0.498116 |
| TF-IDF + logistic regression, text with context | 0.468938 | 0.368140 | 0.445129 |

Both text variants predict all seven categories. However, neither correctly
classifies any of the 29 Contextualization items: its precision, recall, and
F1 are all zero. The text-only model predicts Contextualization five times,
and the context model six times, all incorrectly. Predicting a category is
therefore distinct from successfully recognizing its true examples.

Text-only performs better overall for these fixed settings; this does not
establish that context never helps. Context concatenation, available source
context, model representation, sample size, and annotation boundaries may
affect the results. These explanations have not yet been tested. Do not tune
on these held-out results and then treat them as an untouched test estimate.
Any later tuning should use group-aware splits within the training folds.

All 16 tests passed. The new tests verify that test-only words never enter the
vocabulary and prediction does not update IDF, each item receives one held-out
prediction, annotation metadata does not affect predictions, label columns
cannot be selected as input, and group overlap is rejected. Re-running the
baseline after sharing its input checks reproduced its previous scores.

The source release, prepared CSV, and fold assignments remain unchanged. This
command evaluates fold-specific models; it does not save a model fitted to all
499 items. Retain the report, predictions, environment, and committed code
revision for reproducibility. The next learning step is examining the confusion
matrix and the failed Contextualization examples before choosing improvements.

## Single-fold pretrained transformer run

Added on 2026-10-15: `src/bluesky_discourse/train_transformer.py`. This is a
one-fold DistilBERT experiment for checking GPU training and measuring runtime,
not yet a complete five-fold transformer evaluation. No pretrained transformer
training results are claimed until this script is run on GESIS.

The checkpoint is
[distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased),
pinned to revision `12040accade4e8a0f71eabdb258fecc2e7e948be` for both tokenizer
and model. The encoder starts with pretrained weights; the seven-category
classification head starts with new weights. All model parameters are fine-tuned.
A warning that the classification-head weights are newly initialized is expected.
The script uses safetensors and does not execute custom model repository code.

The first planned run uses `text_only`, held-out fold 0 (100 items), and training
folds 1-4 (399 items). Settings are fixed before this transformer run:

| Setting | Value |
| --- | --- |
| Epochs | 3 |
| Learning rate | 0.00002 |
| Train/evaluation batch size | 8 |
| Gradient accumulation | 1 |
| Optimizer | PyTorch AdamW |
| Weight decay | 0.01 |
| Learning-rate schedule | Linear, 10% warmup |
| Maximum input length | 256 tokenizer tokens, including special tokens |
| Padding | Dynamic within each batch |
| Loss | Balanced, weighted cross entropy |
| Class weights | N_train / (7 * training count for the class) |
| Random and data seeds | 42 |
| Precision | Float32; no FP16, BF16, or TF32 |
| Attention implementation | Eager |
| Evaluation during training | None |
| Early stopping / checkpoint selection | None |

Unlike TF-IDF, the tokenizer's vocabulary comes from the pinned pretrained
checkpoint and is not learned from this dataset. The model receives tokenized
text and numeric labels, not annotation metadata. Class weights are calculated
from training rows only. The script uses Hugging Face Trainer with a weighted
loss callback and trains for exactly three epochs. It introduces the held-out
dataset only after training ends and evaluates once. There is no held-out-score
based parameter adjustment or checkpoint selection. Each invocation starts from
the original checkpoint, including when a different fold is requested.

The run truncates inputs longer than 256 tokens. The report counts inputs that
reach this limit; that count includes both exactly-at-limit and truncated inputs.
Seeded deterministic algorithms are enabled, but different hardware, drivers,
or dependency versions can still affect results. Installed versions are recorded.

### Environment already verified on GESIS

- Conda environment: `bluesky-discourse`, Python 3.11.17.
- PyTorch 2.6.0+cu124; CUDA available.
- NumPy 2.4.6; scikit-learn 1.8.0.
- Transformers 4.57.6; Accelerate 1.12.0; `pip check` passed.
- All 16 pre-transformer tests and both CPU baselines reproduced successfully.
- Installed packages recorded in `outputs/environment.txt`.

`pyproject.toml` now declares the pinned optional `transformer` dependencies.
The GPU-enabled PyTorch wheel was installed separately using the official
CUDA 12.4 index before installing this optional dependency group.

### Transfer and execute

On Windows, commit and push these updated files to the private remote:

```powershell
git add README.md pyproject.toml src/bluesky_discourse/train_transformer.py tests/test_validate_data.py
git commit -m "Add reproducible single-fold DistilBERT experiment"
git push
```

On GESIS:

```bash
conda activate bluesky-discourse
cd ~/bluesky-discourse-classification
git pull --ff-only
python -m pip install -e '.[transformer]'
python -m pip check
python -m unittest discover -s tests -v
```

Use the server's GPU allocation procedure. If a scheduler has already restricted
the job to one GPU, keep its `CUDA_VISIBLE_DEVICES` setting and run the training
command below directly. For a manually allocated GPU, enter its physical number
when prompted; the prompt does not reserve a GPU:

```bash
read -r -p "Enter your allocated GPU number: " GPU_ID
export CUDA_VISIBLE_DEVICES="$GPU_ID"
python -c "import torch; print('Visible GPUs:', torch.cuda.device_count()); print('Selected:', torch.cuda.get_device_name(0))"
python -m bluesky_discourse.train_transformer --fold 0 --text-column text_only
```

Exactly one GPU must be visible. The script refuses CPU-only or multi-GPU
execution, preventing automatic use of all five shared GPUs. Logical CUDA
device 0 inside the process refers to the selected GPU. Do not use `torchrun`
or a distributed launcher. The first run requires network access to download
the checkpoint, which Hugging Face caches for later runs. GPU availability and
training duration have not yet been measured for this experiment.

### Outputs and interpretation

Only two result files are generated, in `outputs/distilbert_fold0_text_only/`:

- `predictions.csv`: the 100 held-out items' IDs, groups, fold, variant, actual
  labels, predicted labels, and correctness. Post text is not copied.
- `report.json`: held-out accuracy/macro F1/weighted F1, per-category metrics,
  confusion matrix, training-only class counts and weights, exact checkpoint
  revision, full training arguments, tokenization settings, training and total
  elapsed time, GPU name/memory, CUDA/dependency versions, and input/code hashes.

No intermediate checkpoints, fitted model, or online tracking runs are saved.
The script refuses a nonempty output directory to preserve earlier results.
For an intentional repeated run, provide a new `--outdir` and record why. A
single-fold result should be compared only with the corresponding fold of the
earlier models, not their combined five-fold scores. The next step, after
checking that this run completes, is to agree on the full evaluation protocol.
Avoid tuning based on the fold 0 test score and presenting that fold as untouched.

Record the code revision and environment after syncing, following the earlier
commands. Locally, 23 tests passed, including split separation, training-only
class weighting, feature-field restrictions, and weighted loss backpropagation.
A tiny randomly initialized DistilBERT also completed a CPU forward/backward
and optimizer-step test without downloads. Local testing used PyTorch 2.9.1+cpu
and Transformers 4.57.3. The full Trainer/CUDA run was subsequently executed
by the researcher on GESIS, with reported results recorded below.

## Combine the five transformer folds

The researcher reported successful GESIS runs with the fixed text-only settings:

| Fold | Training seconds | Held-out accuracy | Held-out macro F1 | Held-out weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 118.3 | 0.550000 | 0.379972 | 0.491520 |
| 4 | 96.2 | 0.626263 | 0.501404 | 0.597386 |

These are individual fold results, not the overall five-fold estimate. The
remaining-fold loop ran on GESIS; aggregation verifies the presence and
consistency of all five saved runs rather than assuming completion from a log.

The existing `train_transformer.py` now supports CPU-only aggregation. On
Windows, commit and push its update together with the README and tests:

```powershell
git add README.md src/bluesky_discourse/train_transformer.py tests/test_validate_data.py
git commit -m "Add checked five-fold transformer summary"
git push
```

Then on GESIS, in the active project environment:

```bash
git pull --ff-only
python -m unittest discover -s tests -v
python -m bluesky_discourse.train_transformer --summarize --text-column text_only
```

No model is loaded and no GPU or retraining is required. Inputs are
`outputs/training_data.csv`, its preparation report, and `report.json` plus
`predictions.csv` in each `outputs/distilbert_fold{k}_text_only/` directory,
for k from 0 through 4. `--runs-dir` can point to a different parent directory.

Aggregation checks the prepared input and fold hashes, prediction-file hashes,
completed statuses, matching checkpoint revision, compatible training settings,
identical package versions and recorded training-source hashes, and one test
prediction per prepared item. Actual labels, groups, variants, and fold IDs
must match the prepared CSV. No group may cross folds. Per-fold output paths,
log directories, and run names are allowed to differ; training settings are not.
This summary code can differ from the original training code: it records its
own hash and preserves the original training-source hashes in the report.

Only two summary files are added in `outputs/distilbert_text_only_summary/`:

- `predictions.csv`: all 499 held-out predictions, sorted by annotation ID.
- `report.json`: combined accuracy/macro F1/weighted F1, fold means and population
  standard deviations, per-category metrics, predicted-label counts, confusion
  matrix, summed training time, and input/run/report hashes and common settings.

The original five run directories remain unchanged. A nonempty summary
directory is rejected; use a new `--outdir` for an intentional repeat. The
summary prints overall metrics and per-category precision, recall, and F1.
Compare its `overall_out_of_fold` scores against the same field in the majority
and TF-IDF reports. Combined F1 is recomputed from all held-out predictions;
it is not obtained by averaging the five fold F1 values.

Locally, all 28 tests passed. Synthetic summary tests cover full prediction
coverage, missing runs, altered predictions, mixed settings, and duplicated
test IDs. Actual GESIS result files are not available in the local checkout,
so no combined transformer scores have been computed locally or inferred from
the displayed fold 0 and fold 4 results. Run aggregation on GESIS and retain its
two outputs with the environment and committed code revision.
