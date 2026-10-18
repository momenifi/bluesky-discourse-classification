# Annotation methodology

## Dataset

The gold dataset contains 500 English-language Bluesky items about scholarly
publications:

- 200 original posts;
- 200 replies linked to those original posts; and
- 100 independent replies sampled from other threads.

The sampling covers eight broad discipline groups. The paired sample supports
analysis of discourse development within threads, while the independent reply
sample reduces concentration on highly active threads.

## Annotation scheme

The annotation scheme is defined in `annotation_handbook_v2.docx`. Each item
receives one dominant discourse-act label:

- `Acknowledgment`
- `Announcement`
- `Contextualization`
- `Explanation`
- `Evaluation`
- `Critique`
- `Other`

Replies are interpreted with their available conversational context. Original
posts are interpreted from the post text itself.

## Human annotation

Three annotators, Annotator1, Annotator2, and Annotator3, labeled the 500 items separately.
The annotators did not see one another's labels during initial annotation.
Agreement was calculated on the original labels before majority resolution or
LLM-assisted adjudication.

The completed three-rater results are:

| Measure | Result |
| --- | ---: |
| Annotator1-Annotator2 agreement | 47.6% |
| Annotator1-Annotator2 Cohen's kappa | 0.363902 |
| Annotator1-Annotator3 agreement | 63.6% |
| Annotator1-Annotator3 Cohen's kappa | 0.556606 |
| Annotator2-Annotator3 agreement | 61.6% |
| Annotator2-Annotator3 Cohen's kappa | 0.521874 |
| Fleiss' kappa, three raters | 0.478197 |
| Unanimous rows | 211 |
| Exactly two annotators agree | 231 |
| All three annotators differ | 58 |

## Resolution and LLM assistance

The final label is produced with the following fixed rule:

1. If at least two human annotators selected the same label, that majority
   label is final. This resolves 442 items.
2. If all three human labels differ, the item enters adjudication. This applies
   to 58 items.
3. For adjudication, an LLM receives the item text, reply context when
   available, Annotation Handbook v2 rules, and the three candidate labels.
   Candidate labels are deduplicated, anonymized, and sorted, so annotator
   identity and label order are hidden.
4. The LLM returns a suggested label, alternative label, confidence, handbook
   rule, and short rationale.
5. A human adjudicator reviews the text, context, three human labels, and LLM
   suggestion and records `human_decision`. This human decision is the final
   label for the 58 adjudicated items.

The LLM is decision support, not an independent fourth annotator. It sees the
anonymized candidate labels, does not vote, and cannot directly assign a gold
label. LLM suggestions are excluded from Cohen's kappa and Fleiss' kappa.

The OpenWebUI API client intentionally sends no web-search or tool fields. The
documented initial suggestion run used GPT-5 via a Codex session. The review
workbook is the authoritative record of the suggestions actually reviewed and
the final human decisions.

## Final dataset fields

`final_gold_dataset_v2.csv` and `final_gold_dataset_v2.xlsx` retain the sampled
post metadata and add:

- `Annotator1_label`, `Annotator2_label`, and `Annotator3_label`;
- `majority_label` and `majority_resolution`;
- the LLM suggestion, confidence, rule, and rationale when adjudication was
  required;
- `human_adjudication_label` for all-different cases;
- `final_label`; and
- `final_label_source`, either `human_majority` or
  `human_adjudication_with_llm_support`.

The `gold_label` column from the working review workbook is not used. The
authoritative adjudication field is `human_decision`.

## Reproduction

From PowerShell in the `bluesky` directory, run:

```powershell
.\build_final_gold_dataset.ps1
```

This validates row counts, unique IDs, label values, majority/adjudication
coverage, and then creates the final CSV, XLSX, and provenance manifest in:

```text
output/gold_annotation/final_gold_dataset_v2/
```

The manifest records SHA-256 hashes for the base dataset, human comparison,
review workbook, handbook, and final outputs. Microsoft Excel is required to
produce the XLSX file. Use `-SkipXlsx` to reproduce only CSV and manifest output
without Excel.

## Recommended model-development split

For model development, split by discussion thread (`root_uri`) rather than by
individual row. This prevents an original post and its replies from appearing
in different train, validation, or test partitions. Stratification should also
consider `final_label`, `item_type`, and `discipline_hint` where feasible.
