# Retrieval Stability and Generalisation

This optional study checks whether retrieval gains depend on individual queries, cutoff settings, scenario groups or E5 training seeds. It separates historical artifact rescoring, archived full-corpus E5 evaluation and three-seed conditional candidate-pool scoring. The default project pipeline is unchanged.

## Run

Copy this directory to `experiments/retrieval_stability/`. From this directory:

```bash
python -m pip install -r requirements.txt
python scripts/final_verify.py
python scripts/test_stability.py
```

Recompute the analyses without weights or an index:

```bash
python scripts/analyze_stability.py
python scripts/analyze_parameters.py
python scripts/analyze_generalization.py
python scripts/compare_pool_rrf.py
```

Outputs go to `runs/reproduction/results/`; `--output-dir` changes the destination. Delivered `results/` remain unchanged. No adjacent query-expression package is needed.

## Data and main findings

`inputs/queries.json` preserves texts, full gold, IDs and the 911/114/114 train/dev/eval split, plus ten official English validation questions. Original candidates, predictions and grids needed for rescoring are under `inputs/historical/`. The shared evaluator, eight paired contrasts, statistics seed 76026 and 20,000 query/scenario bootstrap samples are unchanged. Full-gold and corpus-covered results remain separate, with exclusions reported.

| Comparison | Saved result | Qualification |
|---|---|---|
| English BGE, matched Top-20 | Macro F1 0.0376 to 0.0443 | One query improves; removing it eliminates the gain |
| English BGE, selected cutoff | Macro F1 0.0676 | Four improve, six tie; about 77.8% of this specific gain is cutoff-related |
| Derived E5, full corpus | Recall@100 0.0322 to 0.1399 | 26 improve, 84 tie, four worsen; scenario-bootstrap gain interval +0.0431 to +0.1770 |
| English E5, full corpus | F1@10 0.0100 to 0 | Negative transfer retained |
| Same frozen pool, seeds 760/761/762 | Derived F1@10 0.0337/0.0366/0.0334 | English pool F1@10 0.0527/0.0527/0.0458; these are not full-corpus gains |

All summary scores, paired deltas, parameter controls, coverage ceilings and supporting per-query scores remain in `results/`. Projections remove redundant columns only: every row and retained value is unchanged. Gold and ranked citations remain available. Source hashes and exact archived scoring/helper functions are recorded in `provenance/package_sources.json`. Seed training logs retain actual loss/update evidence; weights are excluded.

## Optional external artifacts

Default statistical reproduction is self-contained. To check or rescore existing vectors:

```bash
python scripts/final_verify.py --check-artifacts --artifact-dir <stability_archive> --e5-archive <e5_archive>
python scripts/analyze_seed_pool.py --artifact-dir <stability_archive>
```

Use the original study's `retrieval_stability_study/` and `e5_experiment/` archives, or configure `seed_artifact_root` and `e5_archive` relative to this directory. The stability archive must contain `inputs/seed_passages.jsonl`, `inputs/seed_embeddings/{base,seed_760,seed_761,seed_762}_{queries,documents}.npy`, and `seed_runs/seed_{761,762}/` with original weights, samples and resume/protocol files. The E5 archive supplies original `data/`, `models/` and `indexes/{base,epoch_2}/` with all manifests/shards and the mapping snapshot. Bundled identity records define the required content; cleaned descriptive metadata must not replace external model files.

Full optional reproduction retains the separate `train_seed.py`, `prepare_seed_pool.py`, `extract_archived_vectors.py` and `encode_new_seed_pool.py` workers. It requires Transformers 4.49.0, faiss-cpu 1.10.0, safetensors 0.8.0 and PyTorch 2.6.0 (tested CUDA wheel: cu126), installed separately. Follow the original `rsg_clean.zip` reproduction sequence in a fresh research copy. FAISS extraction and PyTorch jobs remain separate; never use `KMP_DUPLICATE_LIB_OK`. E5 prefixes, masked mean pooling, normalization and identity checks are preserved.

## Limitations and checks

Ten-question validation was historically tuned. The predominantly German holdout is independent of E5 training/dev, not BGE training. Only one seed has full-corpus evaluation; three seeds have fixed-pool evidence and only 21/114 queries improve under all three. Leave-one-out is sensitivity analysis, not retraining cross-validation. Statistical intervals cannot remove selection bias. No independent English whole-pipeline test, new Cross-Encoder inference or independently verified Kaggle result is claimed. Repeated deterministic retrieval does not establish generalisation.

`validation/checks.json` records actual packaging checks and unexecuted work. `CHECKSUMS.json` verifies delivered files; generated runs are excluded. Figures, long reports, snapshots and redundant outputs remain in the original full cleaned ZIP. No large model/index, cache, personal path or actual hardware record is included.
