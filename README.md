# COMPSCI 760 Legal Retrieval Project

Group 13 course project for **COMPSCI 760**.

## Project

This project studies **cross-lingual Swiss legal citation retrieval** using the Kaggle competition:

**LLM Agentic Legal Information Retrieval**

The core experimental pipeline is:

**BM25 + Multilingual Dense Retrieval → Fine-tuned Cross-Encoder Re-ranking**

## Research Questions

1. Does multilingual dense retrieval outperform a lexical BM25 baseline?
2. How much additional improvement does cross-encoder re-ranking provide?
3. If time permits, can lightweight LLM query reformulation improve retrieval recall?

## Final Results

Validation set (10 queries), same corpus and same `evaluation.py` for every system:

| System | Candidate pool | Macro P | Macro R | Macro F1 | Latency / query |
|---|---|---|---|---|---|
| BM25 (Top-20) | BM25 | 0.0250 | 0.0214 | 0.0222 | 90 ms (CPU) |
| Dense E5-base (Top-10) | Dense | 0.0100 | 0.0100 | 0.0100 | 381 ms (Mac MPS) |
| BM25 + dense fusion, no re-ranker (alpha = 0) | Both | 0.0143 | 0.0314 | 0.0186 | n/a |
| mMiniLM cross-encoder, not fine-tuned | Both | 0.0160 | 0.0267 | 0.0188 | 18.2 s (Mac MPS) |
| bge-reranker-v2-m3, not fine-tuned | Both | 0.0400 | 0.0372 | 0.0376 | 3.8 s (A100) |
| **bge-reranker-v2-m3, fine-tuned** | Both | **0.1559** | **0.0472** | **0.0676** | 3.8 s (A100) |

Exploratory Dense-only E5 comparison (newly executed retrieval):

| Evaluation set / gold | Model | Citation Recall@20 | Citation Recall@50 | Citation Recall@100 | Macro F1@10 |
|---|---|---|---|---|---|
| Train-derived independent (114 queries), full gold | Original pretrained E5 | 0.010234 | 0.014620 | 0.032164 | 0.004161 |
| Train-derived independent (114 queries), full gold | Fine-tuned E5 (epoch 2) | 0.078031 | 0.118239 | 0.139939 | 0.018003 |
| Official English validation (10 queries), full gold | Original pretrained E5 | 0.010000 | 0.014000 | 0.014000 | 0.010000 |
| Official English validation (10 queries), full gold | Fine-tuned E5 (epoch 2) | 0.000000 | 0.005159 | 0.005159 | 0.000000 |

Both models use the same complete corpus, shared `evaluation.py`, 10,000 passage-row search depth and Top-100 citation budget. Citation Recall is macro recall at the stated cutoff; Macro F1@10 uses fixed Top-10 predictions. Full-gold scoring retains missing gold in the denominators. This is a Dense-only fine-tuning experiment; E5-plus-reranking has not been evaluated. The historical fine-tuning gains discussed below refer to the BGE cross-encoder.

Kaggle test set, fine-tuned bge-reranker-v2-m3: **private F1 0.07505, public F1 0.00666** (submitted after the deadline).

Key findings:

- Fine-tuning raises validation macro F1 by about 80% over the same model without fine-tuning, and to about 3x the BM25 baseline. Most of the gain is precision: the fine-tuned model returns about 3 to 4 citations per query.
- **Candidate recall is the bottleneck.** The union of BM25 and dense top-100 holds only 5.1% (macro recall) of the validation gold citations (BM25 alone: 3.7%, dense alone: 1.4%). No re-ranker can score above this recall.
- Alpha and the cutoff are tuned on the same 10 validation queries, so validation scores are optimistic. The Kaggle private score is the fair estimate. The public and private scores differ a lot because each covers only part of the 40 test queries.

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

## Full Dense Retrieval

The full-corpus dense retrieval module uses multilingual sentence embeddings and FAISS vector search over the same law and court corpora used by BM25.

### Main scripts

- `dense_build_index.py` - encodes `laws_de.csv` and `court_considerations.csv`, then writes a FAISS index and row metadata
- `dense_run_experiments.py` - searches the dense index for validation or test queries and writes prediction/candidate files

### Default model

The default embedding model is:

```text
intfloat/multilingual-e5-base
```

For E5 models, documents are encoded as `passage: citation title text` and queries are encoded as `query: text`.

### Build the full dense index

Install the dense dependencies first:

```bash
python -m pip install -r requirements.txt
```

Run a small smoke test before the full build:

```bash
python dense_build_index.py --index-dir results/faiss_dense_smoke --max-docs 1000 --overwrite
python dense_run_experiments.py --index-dir results/faiss_dense_smoke --top-k-values 10,20 --output-prefix dense_smoke
```

Then build the full law plus court index:

```bash
python dense_build_index.py --overwrite
```

Run validation experiments:

```bash
python dense_run_experiments.py
```

Dense outputs are stored in `results/`:

- `dense_full_grid_summary.csv`
- `dense_full_best_predictions.csv`
- `dense_full_candidates_top100.csv`
- `dense_full_config.json`

The generated FAISS index and metadata are stored under `results/faiss_dense_full/` and are not committed to GitHub.

## Cross-Encoder Re-ranking

The re-ranking module takes the BM25 and dense candidates, scores each (query, passage) pair with a multilingual cross-encoder, and writes a new ranking. It can also fine-tune the cross-encoder on the training queries.

### Main files

- `rerank_cross_encoder.py` - loads candidates, gets passage text through `data_loader.py`, scores with the cross-encoder, and evaluates with `evaluation.py`
- `finetune_cross_encoder.py` - fine-tunes the cross-encoder on the training queries with hard negatives

### Method

1. Take the union of the BM25 and dense candidates for each query (up to 200 citations).
2. Look up the passage text for each candidate `doc_id`. The first run scans the court corpus once and caches the text in `results/ce_doc_text_cache.csv.gz`.
3. Score each pair with the cross-encoder. Passage format: `citation title text`. If a citation has more than one passage, it keeps the highest score.
4. Combine the scores:

```text
final = alpha * minmax(ce_score) + (1 - alpha) * minmax(first_stage_rrf)
```

`alpha = 1.0` is pure cross-encoder ranking. `alpha = 0.0` is the first-stage union with no re-ranker. The validation grid includes both, so one run gives the baseline and the re-ranker result.

5. Choose how many citations to predict per query (the cutoff):

| Mode | Keeps | Count per query |
|---|---|---|
| `topk` | the first N citations | same for every query |
| `threshold` | citations with `ce_score >= value` (model output, 0 to 1) | changes per query |
| `relative` | citations with `final_score >= value` (top citation = 1.0) | changes per query |

`threshold` and `relative` keep at least `--min-k` (default 1) and at most `--max-k` (default 50) citations. The gold count changes a lot between queries (validation: 10 to 47), so a variable cutoff can give a higher macro F1 than a fixed Top-K.

The validation grid tests every alpha with every cutoff setting and saves the best one.

### Default model

```text
BAAI/bge-reranker-v2-m3
```

This model is multilingual, so it can score English queries against German, French and Italian passages. For a fast smoke test, use `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.

### Run the re-ranker

Smoke test:

```bash
python rerank_cross_encoder.py --max-queries 2 --model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 --output-prefix ce_smoke
```

Validation grid:

```bash
python rerank_cross_encoder.py --max-length 1024
```

Test predictions (uses the best alpha and cutoff saved by the validation run):

```bash
python rerank_cross_encoder.py --split test \
    --bm25-candidates results/bm25_test_candidates_top100.csv \
    --dense-candidates results/dense_full_test_candidates_top100.csv
```

Useful options: `--sources bm25|dense|both`, `--first-stage-k`, `--alpha-values`, `--cutoff-modes`, `--top-k-values`, `--threshold-values`, `--relative-values`, `--min-k`, `--max-k`, `--max-length`, `--batch-size`, `--device cpu|cuda|mps`.

### First-stage candidates for train and test

Fine-tuning needs BM25 and dense candidates for the training queries, and test predictions need them for the test queries. Both first-stage scripts can now write them:

```bash
python bm25_run_experiments.py --candidates-split train     # or test, or val
python dense_run_experiments.py --split train               # or test
```

- BM25 uses the saved `results/bm25_best_config.json`, skips the parameter search, and writes `results/bm25_<split>_candidates_top100.csv`. Add `--candidate-output-k 500` for deeper candidates.
- Dense writes `results/dense_full_<split>_candidates_top100.csv`.
- Without these flags, both scripts run the validation experiment exactly as before.

### Fine-tuning

Command used for the final model (Colab A100, 8.4 minutes for 2 epochs):

```bash
python finetune_cross_encoder.py --device cuda --epochs 2 --max-length 512 \
    --groups-per-batch 4 --grad-accum 4 --output-dir results/ce_finetuned_bge
python rerank_cross_encoder.py --model results/ce_finetuned_bge --output-prefix ce_bge_ft --device cuda
```

Training result: held-out dev MRR rose from 0.447 to 0.778 (acc@1 0.61), and the training loss fell from about 2.2 to about 0.8.

Each training group is 1 gold passage plus 7 negatives. The loss is a softmax cross-entropy over the group.

- **Hard negatives (4):** BM25 and dense candidates for the query that are not gold. They come from the train candidate files above. Without these files, the script prints a warning and uses the other two kinds.
- **Neighbour negatives (2):** rows next to the gold row in the corpus, for example `Art. 221 Abs. 2 StPO` for the gold `Art. 221 Abs. 1 StPO`, or another consideration of the same court decision. They teach the exact-citation difference that the task scores.
- **Random negatives (1):** random law and court rows.

The script holds out 10% of the training queries and reports the gold passage's MRR within the held-out groups before training and after each epoch. It saves only the best epoch. It does not use the 10 official validation queries, so they stay a fair test.

Training data facts: 999 of the 1,139 training queries have at least one gold citation in the corpus, which gives 3,319 training groups. 3,262 of the positives are laws and 57 are court decisions, but about 40% of the validation gold citations are court decisions.

Hardware: fine-tuning `bge-reranker-v2-m3` needs a GPU with about 16 GB or more (Colab or a lab machine). On an Apple Silicon Mac, one training step took about 40 minutes, so we trained on Colab (see below). On a laptop, fine-tune `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` with `--base-model`.

Safety checks in the training loop: a batch with a non-finite loss and a step with a non-finite gradient norm are skipped and reported, and a `nan` score gets the worst rank in the dev check. On CUDA the script uses bfloat16 (A100, L4) or float16 with a gradient scaler (T4).

The fine-tuned model (about 2.3 GB) is not committed. Download link: `https://drive.google.com/drive/folders/1Lf1JR1amLI_HI-jm4Bl_VvJgzN9_tZHt?usp=sharing`.

### Outputs

Every file name starts with `--output-prefix` (default `ce_rerank`). The final runs used `ce_bge_zeroshot` and `ce_bge_ft`.

- `ce_rerank_grid_summary.csv` - alpha x cutoff validation results, with the average number of predicted citations
- `ce_rerank_best_predictions.csv` - validation predictions with the best setting
- `ce_rerank_candidates_top100.csv` - re-ranked candidates with first-stage ranks and all scores
- `ce_rerank_config.json` - best setting, metrics, candidate-pool recall, and latency
- `ce_rerank_test_predictions.csv` and `ce_rerank_test_candidates_top100.csv` - test predictions (Kaggle format) and ranked test candidates (`--split test`)
- `ce_finetuned*/` - fine-tuned model and `finetune_config.json` (not committed)

`candidate_pool_recall` is the share of gold citations that appear anywhere in the candidate pool. No re-ranker can score above this recall.

## Exploratory E5 Fine-Tuning Experiment

### Motivation and research question

The original Dense baseline had weak validation performance (Macro F1 0.0100 at Top-10), motivating an exploratory study of project-specific E5 fine-tuning for exact legal citations.

**Research question:** Can project-specific fine-tuning of multilingual E5 improve exact legal citation retrieval compared with the original pretrained E5 model?

This experiment is an optional branch, not a replacement for the original BM25, Dense and cross-encoder pipeline. E5 bi-encoder fine-tuning changes the query/document encoder; BGE cross-encoder fine-tuning changes the separate pair scorer.

### Main scripts

- `dense_e5_ft_run.py` - searches the completed fine-tuned E5 index and writes candidates/predictions using `data_loader.py` and `evaluation.py`

### Method

1. Split the original training queries into 911 train, 114 development and 114 independent evaluation queries (seed 760). All gold for one query stay together; duplicate/near-duplicate scenario groups stay in one split.
2. Link gold citations to real legal-text passages for positives. Missing gold are skipped for training and counted; full-gold evaluation retains them.
3. Train `intfloat/multilingual-e5-base` with contrastive learning, BM25/Dense hard negatives and random negatives. Known gold and other passages of a gold citation are excluded from negatives; known positives are protected in the batch loss. Unjudged negatives may still contain relevant citations.
4. Select the checkpoint by development full-gold Recall@100, not training loss. Epoch 2 scored 0.073521 versus 0.050585 for pretrained E5. Independent evaluation labels were not used for selection.

### Model and retrieval artifacts

Reuse the selected epoch-2 model and its matching full-corpus FAISS index (2,652,248 rows). Copy the completed archive artifacts into the optional branch directory:

- `e5_experiment/models/epoch_2/` contents → `results/dense_e5_ft_artifacts/model/`
- `e5_experiment/indexes/epoch_2/` contents → `results/dense_e5_ft_artifacts/index/`
- `e5_experiment/data/corpus.sqlite` → `results/dense_e5_ft_artifacts/corpus.sqlite`

Keep the packaged `protocol.json` and `corpus_snapshot_sha256.json` there, with archived filenames and bytes unchanged. Content hashes verify model/index/map identity after relocation; do not search the pretrained index with fine-tuned query vectors. No retraining, corpus encoding or index conversion is needed. Full training scripts, splits and logs remain in the separate experiment archive.

Retrieval preserves `query: text`, `passage: citation title text`, masked mean pooling, L2 normalization and 512-token truncation. It keeps 10,000 passage rows before deduplicating citations into at most 100 candidates per query. The existing encoder/search process separation resolves the OpenMP conflict; unset `KMP_DUPLICATE_LIB_OK`.

### Run the optional branch

Use the existing shared data preparation commands, then run in the E5 environment from the project root:

```bash
python dense_e5_ft_run.py --verify-only
python dense_e5_ft_run.py --split val --index-dir results/dense_e5_ft_artifacts/index --device auto --output-prefix dense_e5_ft
python dense_e5_ft_run.py --split test --index-dir results/dense_e5_ft_artifacts/index --device auto --output-prefix dense_e5_ft
```

Use a new `dense_e5_ft_` prefix for repeat runs. `--split train` uses the original shared train set, not the archive's derived independent evaluation set.

Outputs are stored in `results/`:

- `dense_e5_ft_val_candidates_top100.csv` and `dense_e5_ft_test_candidates_top100.csv` - Top-100 candidates for downstream re-ranking
- `dense_e5_ft_<split>_predictions.csv` - Dense-only Top-K predictions (default 10), separate from candidates
- `dense_e5_ft_val_summary.csv`, `dense_e5_ft_val_per_query.csv` and `dense_e5_ft_<split>_config.json` - evaluation and run records
- `dense_e5_ft_verified_results.csv` and `dense_e5_ft_provenance.json` - saved comparison metrics and evidence fingerprints

### Optional re-ranking follow-up

The existing reranker can consume this branch's candidate CSV through `--dense-candidates`. It retains `query_id,rank,doc_id,citation,source_type`; additional `row_rank,dense_score` columns are ignored. Preserve canonical citations and the original corpus row order. The reranking owner uses their cross-encoder checkpoint and shared corpus text, not E5 weights.

Run validation and then test in the existing CE environment with the same new output prefix; test reads that prefix's saved alpha/cutoff configuration. These are available follow-up commands, not completed combination results:

```bash
python rerank_cross_encoder.py --split val \
    --model results/ce_finetuned_bge \
    --dense-candidates results/dense_e5_ft_val_candidates_top100.csv \
    --output-prefix dense_e5_ft_ce_followup
python bm25_run_experiments.py --candidates-split test
python rerank_cross_encoder.py --split test \
    --model results/ce_finetuned_bge \
    --bm25-candidates results/bm25_test_candidates_top100.csv \
    --dense-candidates results/dense_e5_ft_test_candidates_top100.csv \
    --output-prefix dense_e5_ft_ce_followup
```

Use the owner's actual checkpoint path if different. Validation uses the existing BM25 validation candidates by default; both commands retain the default BM25-plus-Dense Top-100 union. Historical CE output prefixes are preserved.

### Results and discussion

The comparison in Final Results shows improvement on the predominantly German train-derived independent set, but deterioration on the English official validation set. BM25 figures in the experiment archive are historical candidate reevaluation; the E5 comparison uses newly executed retrieval for both models.

German-dominated training and its law-heavy positives (2,606 law versus 32 court passage pairs) differ from the English validation target, where 102 of 251 gold occurrences are court citations. These are possible explanations for limited transfer; their causal effects were not isolated. On derived evaluation, Top-100 court candidate share changed from 97.3% to 0%, replacing court dominance with law dominance rather than demonstrating that source bias was solved.

Official validation contains only ten questions and has already informed historical project tuning; it is not an untouched test set. Only one E5 training seed was run. The results support a limited exploratory finding, not improved English retrieval, stable generalization or gains after reranking.

## Setup Notes

### BM25 (Pyserini and Java)

Pyserini needs Java 21. Install both inside the project environment:

```bash
conda activate legal-ir
conda install -c conda-forge openjdk=21 -y
python -m pip install pyserini
conda env config vars set JAVA_HOME="$CONDA_PREFIX/lib/jvm" \
    JVM_PATH="$CONDA_PREFIX/lib/jvm/lib/server/libjvm.dylib" OPENAI_API_KEY=not-used
conda deactivate && conda activate legal-ir
```

- Run these after `conda activate legal-ir`. In `(base)`, `$CONDA_PREFIX` points to the wrong environment.
- `JVM_PATH` fixes `Unable to find libjvm.so` from `pyjnius` (the path above is for macOS; on Linux use `libjvm.so`).
- `OPENAI_API_KEY=not-used` works around a Pyserini import bug: it creates an OpenAI client at import time and stops if no key is set. No OpenAI call is made.
- Pyserini can upgrade numpy to 2.x. If faiss or torch then fails, run `python -m pip install "numpy>=1.26,<2.0"`.
- Some training queries have more than 1,024 unique terms. `bm25_run_experiments.py` raises Lucene's clause limit to 16,384 so these queries keep every term.

### Running on Google Colab (GPU)

The dense index build and cross-encoder fine-tuning are too slow on a laptop (dense index: about 46 hours on a Mac, versus about 2 hours on a Colab A100). The BM25 indexes stay local: copy only the candidate CSV files to Colab.

1. Zip the project without the large local folders and upload the zip to Google Drive:

   ```bash
   zip -r legal-ir-colab.zip COMPSCI760-Legal-Retrieval-main \
       -x "*/results/lucene_*" "*/results/faiss_*" "*/results/ce_finetuned*" "*/__pycache__/*" "*.bak"
   ```

2. In Colab, choose **Runtime > Change runtime type > A100 GPU** (or T4 on the free tier) and turn on High RAM for the dense index.
3. Mount Drive, unzip, install and check the GPU:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

   ```bash
   !unzip -o -q /content/drive/MyDrive/legal-ir-colab.zip -d /content/data
   %cd /content/data/COMPSCI760-Legal-Retrieval-main
   !pip install -q sentence-transformers faiss-cpu
   !nvidia-smi
   ```

4. Dense index with float16 (`--fp16`, CUDA only). Use the default `--index-dir`:

   ```bash
   !python dense_build_index.py --device cuda --fp16 --batch-size 256 --overwrite
   !python dense_run_experiments.py --split train
   !python dense_run_experiments.py --split test
   ```

5. Fine-tuning and re-ranking: the commands in the sections above, with `--device cuda`. Write models and results to `/content/drive/MyDrive/...`, because `/content` is deleted when the session ends.
6. Copy the result files back into `results/` on your computer.

## Reproducing the Final Results

Run in this order. Steps 5 to 8 need a GPU.

| Step | Command | Output |
|---|---|---|
| 1 | `python prepare_queries.py`, `python prepare_corpus.py`, `python final_preflight.py` | `processed/` |
| 2 | `python bm25_build_index.py` | `results/lucene_bm25_*` |
| 3 | `python bm25_run_experiments.py` | BM25 val results and val candidates |
| 4 | `python bm25_run_experiments.py --candidates-split train` (and `test`) | BM25 train and test candidates |
| 5 | `python dense_build_index.py --device cuda --fp16` | `results/faiss_dense_full/` |
| 6 | `python dense_run_experiments.py` (and `--split train`, `--split test`) | dense val, train and test candidates |
| 7 | `python rerank_cross_encoder.py --model BAAI/bge-reranker-v2-m3 --output-prefix ce_bge_zeroshot` | zero-shot baseline |
| 8 | `python finetune_cross_encoder.py ...` then `rerank_cross_encoder.py --model results/ce_finetuned_bge --output-prefix ce_bge_ft` (val, then `--split test`) | final re-ranker results and test predictions |

## Code Contributions and External Sources

| Member | Contribution |
|---|---|
| Qiyu Zhang | Data processing, corpus audit, shared data loader and evaluator (`prepare_*.py`, `audit_court.py`, `data_loader.py`, `evaluation.py`, `final_preflight.py`) |
| Yuting Shen | BM25 indexes, query processing, tuning and candidate pools (`bm25_*.py`) |
| Yi Zong | Dense retrieval with E5 and FAISS (`dense_*.py`) |
| Simon Lin | Cross-encoder re-ranking, fine-tuning, candidate pools for train and test, Colab workflow (`rerank_cross_encoder.py`, `finetune_cross_encoder.py`) |
| Xirui Liu | Integration and final comparison; exploratory E5 fine-tuning study, experiment evaluation and candidate handoff integration (`dense_e5_ft_run.py`) |

External sources used (not our work):

- Data: Kaggle competition *LLM Agentic Legal Information Retrieval*.
- Pretrained models from Hugging Face: `intfloat/multilingual-e5-base`, `BAAI/bge-reranker-v2-m3`, `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
- Libraries: Pyserini / Apache Lucene (BM25), FAISS, sentence-transformers, Hugging Face transformers, PyTorch, pandas, NumPy.
- Methods: BM25 (Robertson and Zaragoza, 2009), reciprocal rank fusion (Cormack et al., 2009), cross-encoder re-ranking (Nogueira and Cho, 2019).
- Generative AI helped with code debugging, script drafting and documentation. The group checked all code, numbers and conclusions.

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
├── dense_build_index.py
├── dense_run_experiments.py
├── dense_e5_ft_run.py
├── rerank_cross_encoder.py
├── finetune_cross_encoder.py
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
