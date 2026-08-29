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

## BM25 Baseline

The BM25 lexical retrieval baseline has been implemented using Pyserini/Lucene.

The system builds separate indexes for:

- Federal laws
- Federal court considerations

The two retrieval sources are combined using weighted Reciprocal Rank Fusion (RRF).

### Main scripts

- `bm25_build_index.py` — builds the Lucene BM25 indexes
- `bm25_run_experiments.py` — runs parameter search, validation evaluation, and candidate generation

### Best validation configuration

The best configuration on the provided validation set is:

- BM25 `k1`: **1.8**
- BM25 `b`: **0.75**
- Law weight: **1.5**
- Court weight: **1.0**
- Prediction Top-K: **20**
- Macro Precision: **0.0250**
- Macro Recall: **0.0214**
- Macro F1: **0.0222**

The validation set contains only 10 queries, so these results should be interpreted as baseline validation results rather than a universal optimum.

### BM25 outputs

The main output files are stored in `results/`:

- `bm25_best_config.json` — selected BM25 configuration
- `bm25_best_predictions.csv` — validation predictions using the best configuration
- `bm25_candidates_top100.csv` — Top-100 BM25 candidates for downstream retrieval/re-ranking
- `bm25_ablation_summary.csv` — comparison of BM25 query-processing variants
- `bm25_final_grid_summary.csv` — full BM25 parameter-search results

The generated Lucene indexes are not stored in GitHub because they can be reproduced locally using `bm25_build_index.py`.
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
├── bm25_build_index.py
├── bm25_run_experiments.py
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
