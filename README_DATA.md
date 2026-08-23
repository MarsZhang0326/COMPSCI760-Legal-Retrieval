# COMPSCI 760 Legal Retrieval Project
## Data & Evaluation Handoff

### 1. Original Kaggle files

The project uses the following original competition files:

- `train.csv`
- `val.csv`
- `test.csv`
- `laws_de.csv`
- `court_considerations.csv`
- `sample_submission.csv`

The task is to retrieve and predict exact canonical Swiss legal citation IDs for each legal query.

---

## 2. Processed query files

Generated in the `processed/` folder:

- `train_clean.csv`
- `val_clean.csv`
- `test_clean.csv`
- `gold_train.json`
- `gold_val.json`
- `query_processing_summary.json`
- `corpus_coverage_report.json`
- `corpus_schema.json`

Query cleaning is intentionally conservative:

- leading and trailing whitespace removed
- line breaks, tabs, and repeated whitespace collapsed
- wording and punctuation preserved
- no forced lowercasing
- no stemming
- no stopword removal

Model-specific preprocessing should be implemented inside each retrieval method rather than in the shared data layer.

---

## 3. Query dataset summary

### Train

- 1,139 labelled queries
- 2,695 unique gold citations
- 4,659 total gold citation occurrences
- mean = 4.09 citations per query

### Validation

- 10 labelled queries
- 222 unique gold citations
- 251 total gold citation occurrences
- mean = 25.10 citations per query

### Test

- 40 unlabelled queries

Important: the train and validation sets have substantially different citation-count distributions, so validation results should be interpreted carefully.

---

## 4. Retrieval corpus summary

### Federal law corpus: `laws_de.csv`

- 175,933 rows
- 175,933 unique citation IDs
- columns include:
  - `citation`
  - `text`
  - `title`

### Federal Court corpus: `court_considerations.csv`

- 2,476,315 rows
- 1,985,178 unique citation IDs
- 491,137 repeated citation rows
- no missing citation values
- no missing text values
- columns:
  - `citation`
  - `text`

Repeated court citation IDs must be preserved because multiple different court passages can map to the same canonical citation.

Do **not** drop court rows solely because the citation string repeats.

---

## 5. Gold-to-corpus coverage

### Train

- 1,876 unique gold citations found in `laws_de.csv`
- 52 unique gold citations found in `court_considerations.csv`
- 1,928 / 2,695 found in the complete corpus
- full-corpus coverage = 71.54%
- 767 unique train gold citations are not present in the supplied retrieval corpus

### Validation

- 121 unique gold citations found in `laws_de.csv`
- 101 unique gold citations found in `court_considerations.csv`
- 222 / 222 found in the complete corpus
- full-corpus coverage = 100%

Therefore, all validation gold citations are retrievable from the supplied corpus.

The incomplete train coverage must be considered if supervised training or fine-tuning is attempted later.

---

## 6. Shared logical corpus schema

All retrieval methods should use the same logical schema:

- `doc_id`
- `citation`
- `text`
- `title`
- `source_type`

where:

- `source_type = "law"` for `laws_de.csv`
- `source_type = "court"` for `court_considerations.csv`
- court rows use an empty `title`
- `doc_id` is an internal row-level identifier
- `citation` is the canonical Kaggle prediction ID

Use the shared `data_loader.py` instead of implementing separate corpus loaders for BM25, Dense Retrieval, or Re-ranking.

---

## 7. Loading the data

Example:

```python
from data_loader import (
    load_train,
    load_val,
    load_test,
    load_laws,
    iter_court_chunks
)

train = load_train()
val = load_val()
test = load_test()
laws = load_laws()

for chunk in iter_court_chunks():
    # build or search the retrieval index
    pass
```

The court corpus is streamed in chunks so that the full ~2.4 GB file does not need to be loaded into memory at once.

---

## 8. Retrieval row vs. final citation prediction

Retrieval operates at **row level**.

Example:

- court row A -> citation X
- court row B -> citation X
- court row C -> citation Y

Rows A and B must both remain in the retrieval corpus because they contain different passages.

However, evaluation operates at **citation level**.

Before evaluation, final predictions must be deduplicated:

- X
- Y

not:

- X
- X
- Y

In short:

**row-level retrieval -> citation-level deduplication -> citation-level evaluation**

---

## 9. Shared evaluation

Use the shared `evaluation.py`.

Do **not** implement separate scoring logic for BM25, Dense Retrieval, or Re-ranking.

Prediction files must use this format:

```csv
query_id,predicted_citations
val_0001,"Art. ...;BGE ..."
val_0002,"..."
```

Multiple citations are separated by semicolons.

Example:

```python
from evaluation import evaluate_csv, print_summary

summary, details = evaluate_csv(
    "val_bm25_predictions.csv"
)

print_summary(summary)
```

The shared evaluator returns:

- Macro Precision
- Macro Recall
- Macro F1
- per-query TP
- per-query FP
- per-query FN
- per-query Precision
- per-query Recall
- per-query F1

---

## 10. Evaluation sanity checks

The shared evaluation pipeline has passed the following tests:

### Test 1: Perfect predictions

- Macro Precision = 1.0000
- Macro Recall = 1.0000
- Macro F1 = 1.0000

### Test 2: Empty predictions

- Macro Precision = 0.0000
- Macro Recall = 0.0000
- Macro F1 = 0.0000

### Test 3: CSV end-to-end test

A Kaggle-style prediction CSV was created and re-read through `evaluation.py`.

Result:

- Macro Precision = 1.0000
- Macro Recall = 1.0000
- Macro F1 = 1.0000
- CSV evaluation test passed

---

## 11. Rules for all retrieval experiments

To keep the ablation fair:

1. Use the same validation set.
2. Use the same original retrieval corpus.
3. Preserve canonical citation strings exactly.
4. Use the same `data_loader.py`.
5. Use the same `evaluation.py`.
6. Do not remove repeated court rows simply because citation IDs repeat.
7. Deduplicate final outputs at citation level before evaluation.
8. Record the retrieval `Top-K` setting.
9. Record latency.
10. Document model-specific tokenisation and preprocessing.
11. Do not change the evaluation script between models.
12. Avoid over-tuning to the very small 10-query validation set.

The intended comparison is:

**BM25 -> Dense Retrieval -> Dense + Cross-Encoder Re-ranking**

The optional LLM agent should only be added after the core retrieval pipeline is stable.

---

## 12. Recommended project structure

```text
llm-agentic-legal-information-retrieval/
│
├── train.csv
├── val.csv
├── test.csv
├── laws_de.csv
├── court_considerations.csv
├── sample_submission.csv
│
├── data_loader.py
├── evaluation.py
├── README_DATA.md
│
├── audit_court.py
├── prepare_queries.py
├── prepare_corpus.py
├── test_evaluation_csv.py
│
└── processed/
    ├── train_clean.csv
    ├── val_clean.csv
    ├── test_clean.csv
    ├── gold_train.json
    ├── gold_val.json
    ├── query_processing_summary.json
    ├── corpus_coverage_report.json
    ├── corpus_schema.json
    └── val_perfect_predictions.csv
```

The audit and preparation scripts are retained for reproducibility, but model developers mainly need:

- `data_loader.py`
- `evaluation.py`
- original corpus files
- cleaned query files in `processed/`
- this README

---

## 13. BM25 handoff

The BM25 workflow should be:

```text
load_val()
    ↓
load or stream the shared corpus
    ↓
build BM25 index
    ↓
retrieve top-K rows
    ↓
map rows to canonical citation IDs
    ↓
deduplicate citation IDs
    ↓
save val_bm25_predictions.csv
    ↓
evaluate with evaluation.py
    ↓
report Macro Precision / Recall / F1 and latency
```

Do not re-clean the shared data and do not create a separate evaluation function.
