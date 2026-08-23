# COMPSCI 760 Legal Retrieval Project

Group 13 course project for **COMPSCI 760**.

## Project

This project studies **cross-lingual Swiss legal citation retrieval** using the Kaggle competition:

**LLM Agentic Legal Information Retrieval**

The core experimental pipeline is:

**BM25 → Multilingual Dense Retrieval → Cross-Encoder Re-ranking**

An optional lightweight LLM agent may be tested after the core retrieval pipeline is stable.

## Research Questions

1. Does multilingual dense retrieval outperform a lexical BM25 baseline?
2. How much additional improvement does cross-encoder re-ranking provide?
3. If time permits, can lightweight LLM query reformulation improve retrieval recall?

## Shared Data and Evaluation Layer

The repository currently contains the shared data-processing and evaluation infrastructure:

- `data_loader.py` — common query/corpus loader
- `evaluation.py` — shared citation-level evaluation
- `prepare_queries.py` — reproducible query preprocessing
- `prepare_corpus.py` — corpus audit and coverage analysis
- `audit_court.py` — large court-corpus audit
- `final_preflight.py` — final consistency checks
- `README_DATA.md` — detailed data handoff documentation

### Confirmed dataset facts

- Train queries: **1,139**
- Validation queries: **10**
- Test queries: **40**
- Federal-law rows: **175,933**
- Federal-court rows: **2,476,315**
- Validation gold-to-corpus coverage: **222/222 (100%)**
- Train unique gold-to-corpus coverage: **1,928/2,695 (71.54%)**

## Data Setup

The original competition CSV files are **not stored in GitHub**.

Each team member should:

1. Join/accept the Kaggle competition rules.
2. Download the competition data directly from Kaggle.
3. Place the six original CSV files in the repository root:

```text
train.csv
val.csv
test.csv
laws_de.csv
court_considerations.csv
sample_submission.csv
```

These files are excluded by `.gitignore`.

## Reproduce the Shared Data Layer

After placing the Kaggle CSV files in the repository root:

```bash
python prepare_queries.py
python prepare_corpus.py
python final_preflight.py
```

A successful setup ends with:

```text
FINAL PREFLIGHT PASSED
Data & Evaluation package is ready for BM25.
```

## Evaluation

All retrieval systems must use the same `evaluation.py`.

Expected prediction format:

```csv
query_id,predicted_citations
val_0001,"Art. ...;BGE ..."
val_0002,"..."
```

Multiple citations are separated by semicolons.

The common evaluator reports:

- Macro Precision
- Macro Recall
- Macro F1
- per-query TP / FP / FN
- per-query Precision / Recall / F1

## Repository Structure

```text
COMPSCI760-Legal-Retrieval/
│
├── README.md
├── README_DATA.md
├── .gitignore
├── requirements.txt
│
├── data_loader.py
├── evaluation.py
│
├── audit_court.py
├── prepare_queries.py
├── prepare_corpus.py
├── test_evaluation_csv.py
├── final_preflight.py
│
├── processed/
│   ├── query_processing_summary.json
│   ├── corpus_coverage_report.json
│   └── corpus_schema.json
│
└── results/
```

Large raw data, embeddings, FAISS indexes, checkpoints, and other generated artifacts should not be committed to GitHub.

## Collaboration Rule

To keep the ablation fair:

- use the same validation set
- use the same corpus
- preserve canonical citation strings
- use the shared `data_loader.py`
- use the shared `evaluation.py`
- document model-specific preprocessing
- record Top-K and latency
- do not change the scoring logic between models

See `README_DATA.md` for full details.
