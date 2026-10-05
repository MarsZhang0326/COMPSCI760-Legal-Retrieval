"""
Cross-encoder re-ranking for the COMPSCI 760 legal retrieval pipeline.

Pipeline position:

    BM25 top-100  ┐
                  ├─> candidate union ─> cross-encoder ─> score fusion ─> Top-K
    Dense top-100 ┘

Inputs:
    - first-stage candidate files written by bm25_run_experiments.py and
      dense_run_experiments.py (citation-level, one doc_id per row)
    - the original corpus files, read through the shared data_loader.py,
      to get the passage text for each candidate doc_id

Outputs (under results/, prefix set by --output-prefix):
    - <prefix>_grid_summary.csv      alpha x Top-K validation grid
    - <prefix>_best_predictions.csv  Kaggle-style predictions, best setting
    - <prefix>_candidates_top100.csv re-ranked candidates with all scores
    - <prefix>_config.json           best configuration and metrics
    - <prefix>_test_predictions.csv  test predictions (--split test)

Score fusion:
    final = alpha * minmax(ce_score) + (1 - alpha) * minmax(first_stage_rrf)

    alpha = 1.0  -> pure cross-encoder ranking
    alpha = 0.0  -> first-stage ranking only (the "no re-ranker" baseline)

Cutoff (how many citations to predict per query):
    topk       same N for every query
    threshold  citations with ce_score >= value
    relative   citations with final_score >= value (top citation = 1.0)

The validation grid tests every alpha with every cutoff setting.

Fine-tuned models from finetune_cross_encoder.py load with
--model results/ce_finetuned (or another output directory).

All evaluation goes through the shared evaluation.py.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import iter_court_chunks, load_laws, load_test, load_val
from evaluation import evaluate_csv, load_gold


try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_BM25_CANDIDATES = RESULTS_DIR / "bm25_candidates_top100.csv"
DEFAULT_DENSE_CANDIDATES = RESULTS_DIR / "dense_full_candidates_top100.csv"
DEFAULT_TEXT_CACHE = RESULTS_DIR / "ce_doc_text_cache.csv.gz"
DEFAULT_OUTPUT_PREFIX = "ce_rerank"

DEFAULT_TOP_K_VALUES = [5, 10, 20, 30, 50]
DEFAULT_ALPHA_VALUES = [1.0, 0.8, 0.6, 0.4, 0.0]
DEFAULT_THRESHOLD_VALUES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
DEFAULT_RELATIVE_VALUES = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5]

RRF_K = 60
CANDIDATE_OUTPUT_K = 100


# ============================================================
# Arguments
# ============================================================

def parse_number_list(value, cast):
    output = [
        cast(item.strip())
        for item in value.split(",")
        if item.strip()
    ]

    if not output:
        raise argparse.ArgumentTypeError(
            "At least one value is required."
        )

    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Re-rank BM25 and dense retrieval candidates "
            "with a multilingual cross-encoder."
        )
    )

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
        help="Run the validation grid or produce test predictions.",
    )

    parser.add_argument(
        "--sources",
        choices=["both", "bm25", "dense"],
        default="both",
        help="First-stage candidate sources to re-rank.",
    )

    parser.add_argument(
        "--bm25-candidates",
        type=Path,
        default=DEFAULT_BM25_CANDIDATES,
        help="BM25 candidate CSV.",
    )

    parser.add_argument(
        "--dense-candidates",
        type=Path,
        default=DEFAULT_DENSE_CANDIDATES,
        help="Dense retrieval candidate CSV.",
    )

    parser.add_argument(
        "--first-stage-k",
        type=int,
        default=100,
        help="Candidates taken from each source per query.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Cross-encoder model. Multilingual options: "
            "BAAI/bge-reranker-v2-m3 (default, strongest), "
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (fast)."
        ),
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum tokens per (query, passage) pair.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Cross-encoder batch size.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device, e.g. cpu, cuda, or mps.",
    )

    parser.add_argument(
        "--top-k-values",
        type=lambda v: parse_number_list(v, int),
        default=DEFAULT_TOP_K_VALUES,
        help="Comma-separated Top-K values for the validation grid.",
    )

    parser.add_argument(
        "--alpha-values",
        type=lambda v: parse_number_list(v, float),
        default=DEFAULT_ALPHA_VALUES,
        help="Comma-separated cross-encoder weights for the validation grid.",
    )

    parser.add_argument(
        "--cutoff-modes",
        type=lambda v: [m.strip() for m in v.split(",") if m.strip()],
        default=list(CUTOFF_MODES),
        help="Comma-separated cutoff modes for the grid: topk,threshold,relative.",
    )

    parser.add_argument(
        "--threshold-values",
        type=lambda v: parse_number_list(v, float),
        default=DEFAULT_THRESHOLD_VALUES,
        help="ce_score thresholds (0 to 1) for the threshold mode.",
    )

    parser.add_argument(
        "--relative-values",
        type=lambda v: parse_number_list(v, float),
        default=DEFAULT_RELATIVE_VALUES,
        help="final_score cutoffs (0 to 1, top = 1.0) for the relative mode.",
    )

    parser.add_argument(
        "--min-k",
        type=int,
        default=1,
        help="Minimum citations per query for threshold and relative modes.",
    )

    parser.add_argument(
        "--max-k",
        type=int,
        default=50,
        help="Maximum citations per query for threshold and relative modes.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Test mode: use a fixed Top-K cutoff. Overrides the saved best cutoff.",
    )

    parser.add_argument(
        "--cutoff-mode",
        choices=CUTOFF_MODES,
        default=None,
        help="Test mode: cutoff mode. Defaults to the best validation cutoff.",
    )

    parser.add_argument(
        "--cutoff-value",
        type=float,
        default=None,
        help="Test mode: value for --cutoff-mode.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Test alpha. Defaults to the best validation alpha.",
    )

    parser.add_argument(
        "--text-cache",
        type=Path,
        default=DEFAULT_TEXT_CACHE,
        help="Cache file for candidate passage text.",
    )

    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Smoke-test limit on the number of queries.",
    )

    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix for result files written under results/.",
    )

    return parser.parse_args(argv)


def build_output_paths(output_prefix):
    prefix = output_prefix.strip()

    if not prefix or "/" in prefix or "\\" in prefix:
        raise ValueError("--output-prefix must be a plain file prefix.")

    return {
        "grid_summary": RESULTS_DIR / f"{prefix}_grid_summary.csv",
        "best_predictions": RESULTS_DIR / f"{prefix}_best_predictions.csv",
        "candidates": RESULTS_DIR / f"{prefix}_candidates_top100.csv",
        "best_config": RESULTS_DIR / f"{prefix}_config.json",
        "test_predictions": RESULTS_DIR / f"{prefix}_test_predictions.csv",
        "temporary_predictions":
            RESULTS_DIR / f"_{prefix}_temporary_predictions.csv",
    }


# ============================================================
# Candidate loading
# ============================================================

def read_candidate_file(path, source_name, first_stage_k):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing {source_name} candidate file: {path}"
        )

    df = pd.read_csv(
        path,
        keep_default_na=False,
        dtype={"query_id": str, "doc_id": str, "citation": str},
    )

    required = {"query_id", "rank", "doc_id", "citation", "source_type"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} missing columns: {missing}"
        )

    df["rank"] = df["rank"].astype(int)
    df["citation"] = df["citation"].str.strip()
    df = df[df["citation"] != ""]
    df = df[df["rank"] <= first_stage_k]

    df = df[["query_id", "rank", "doc_id", "citation", "source_type"]].copy()
    df["source"] = source_name

    return df


def load_candidates(args):
    """
    Return one row per (query_id, citation, doc_id).

    A citation can appear in both sources with different doc_ids
    (for example two court passages with the same citation). All
    doc_ids are kept and scored; the citation takes the best score.
    """

    frames = []

    if args.sources in {"both", "bm25"}:
        frames.append(read_candidate_file(
            args.bm25_candidates, "bm25", args.first_stage_k
        ))

    if args.sources in {"both", "dense"}:
        frames.append(read_candidate_file(
            args.dense_candidates, "dense", args.first_stage_k
        ))

    raw = pd.concat(frames, ignore_index=True)

    # Citation-level first-stage rank per source
    ranks = (
        raw.groupby(["query_id", "citation", "source"])["rank"]
        .min()
        .unstack("source")
    )

    for source in ["bm25", "dense"]:
        if source not in ranks.columns:
            ranks[source] = np.nan

    ranks = ranks.rename(columns={
        "bm25": "bm25_rank",
        "dense": "dense_rank",
    })

    rrf = np.zeros(len(ranks))

    for column in ["bm25_rank", "dense_rank"]:
        values = ranks[column].to_numpy(dtype=float)
        rrf += np.where(
            np.isnan(values), 0.0, 1.0 / (RRF_K + values)
        )

    ranks["first_stage_rrf"] = rrf
    ranks = ranks.reset_index()

    pairs = (
        raw[["query_id", "citation", "doc_id", "source_type"]]
        .drop_duplicates(["query_id", "citation", "doc_id"])
    )

    return pairs.merge(
        ranks[["query_id", "citation", "bm25_rank",
               "dense_rank", "first_stage_rrf"]],
        on=["query_id", "citation"],
        how="left",
    )


# ============================================================
# Passage text lookup
# ============================================================

def build_passage_text(citation, title, text):
    parts = [
        str(part or "").strip()
        for part in [citation, title, text]
    ]

    return " ".join(part for part in parts if part)


def load_text_cache(path):
    if not Path(path).exists():
        return {}

    cache = pd.read_csv(
        path,
        keep_default_na=False,
        dtype=str,
    )

    return dict(zip(cache["doc_id"], cache["passage"]))


def save_text_cache(path, passages):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "doc_id": list(passages.keys()),
        "passage": list(passages.values()),
    }).to_csv(
        path,
        index=False,
    )


def fetch_passages(doc_ids, cache_path):
    """
    Map doc_id -> passage text through the shared data_loader.

    Law rows are read in one pass. The court file is streamed and
    the scan stops as soon as every needed court row is found.
    Results are cached so later runs skip the 2.4 GB court scan.
    """

    passages = load_text_cache(cache_path)
    needed = set(doc_ids) - set(passages)

    if not needed:
        print(f"Passage text: all {len(set(doc_ids)):,} rows from cache")
        return passages

    needed_law = {d for d in needed if d.startswith("law_")}
    needed_court = {d for d in needed if d.startswith("court_")}

    print(
        f"Passage text: {len(needed_law):,} law rows and "
        f"{len(needed_court):,} court rows to load"
    )

    if needed_law:
        laws = load_laws()
        laws = laws[laws["doc_id"].isin(needed_law)]

        for row in laws.itertuples(index=False):
            passages[row.doc_id] = build_passage_text(
                row.citation, row.title, row.text
            )

    if needed_court:
        remaining = set(needed_court)
        start = time.perf_counter()

        for chunk in iter_court_chunks():
            hits = chunk[chunk["doc_id"].isin(remaining)]

            for row in hits.itertuples(index=False):
                passages[row.doc_id] = build_passage_text(
                    row.citation, row.title, row.text
                )
                remaining.discard(row.doc_id)

            if not remaining:
                break

        print(
            f"Court scan: {time.perf_counter() - start:.1f} s, "
            f"{len(remaining):,} rows not found"
        )

    unresolved = needed - set(passages)

    if unresolved:
        raise ValueError(
            f"{len(unresolved)} candidate doc_ids not found in corpus, "
            f"e.g. {sorted(unresolved)[:5]}"
        )

    save_text_cache(cache_path, passages)

    return passages


# ============================================================
# Cross-encoder scoring
# ============================================================

def load_cross_encoder(args):
    if CrossEncoder is None:
        raise SystemExit(
            "Missing dependency: sentence-transformers\n"
            "Install it with:\n"
            "  python -m pip install -r requirements.txt"
        )

    return CrossEncoder(
        args.model,
        max_length=args.max_length,
        device=args.device,
    )


def score_candidates(model, queries, candidates, passages, batch_size):
    """
    Score every (query, doc_id) pair. One predict call per query,
    so latency is recorded per query.

    Returns the candidate frame with a ce_score column and a
    {query_id: latency_ms} dict.
    """

    query_text = dict(zip(queries["query_id"], queries["query"]))
    scored = []
    latencies_ms = {}

    for query_id, group in candidates.groupby("query_id", sort=False):
        pairs = [
            (query_text[query_id], passages[doc_id])
            for doc_id in group["doc_id"]
        ]

        start = time.perf_counter()

        scores = model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms[query_id] = elapsed_ms

        group = group.copy()
        group["ce_score"] = np.asarray(scores, dtype=float).reshape(-1)
        scored.append(group)

        print(
            f"{query_id} | pairs={len(pairs):4d} | "
            f"rerank={elapsed_ms:8.1f} ms"
        )

    return pd.concat(scored, ignore_index=True), latencies_ms


def to_citation_level(scored):
    """
    One row per (query_id, citation). The citation keeps the
    best-scoring passage and its doc_id.
    """

    best = (
        scored.sort_values("ce_score", ascending=False)
        .drop_duplicates(["query_id", "citation"])
    )

    return best.reset_index(drop=True)


# ============================================================
# Fusion and ranking
# ============================================================

def minmax_per_query(df, column):
    grouped = df.groupby("query_id")[column]
    low = grouped.transform("min")
    high = grouped.transform("max")
    span = (high - low).replace(0, 1.0)

    return (df[column] - low) / span


def rank_citations(citations, alpha):
    """
    Return the citation frame sorted by the fused score within
    each query. Ties break on the first-stage score, then the
    citation string, so the output is deterministic.
    """

    df = citations.copy()
    df["final_score"] = (
        alpha * minmax_per_query(df, "ce_score")
        + (1 - alpha) * minmax_per_query(df, "first_stage_rrf")
    )

    return df.sort_values(
        ["query_id", "final_score", "first_stage_rrf", "citation"],
        ascending=[True, False, False, True],
    )


# ============================================================
# Cutoff: how many citations to predict per query
# ============================================================
#
# topk       keep the first N citations (same N for every query)
# threshold  keep citations with ce_score >= value
#            (ce_score is the model's sigmoid output, 0 to 1)
# relative   keep citations with final_score >= value
#            (final_score is min-max scaled per query, so the
#            top citation is 1.0 and value is a share of the range)
#
# threshold and relative give a different count per query.
# Every mode keeps at least --min-k and at most --max-k citations.
# The order of the kept citations is always the final_score order.

CUTOFF_MODES = ["topk", "threshold", "relative"]


def select_citations(ranked_df, mode, value, min_k=1, max_k=50):
    """
    Return {query_id: [citation, ...]} after the cutoff.
    ranked_df must come from rank_citations (sorted per query).
    """

    if mode not in CUTOFF_MODES:
        raise ValueError(f"Unknown cutoff mode: {mode}")

    selected = {}

    for query_id, group in ranked_df.groupby("query_id", sort=False):
        if mode == "topk":
            keep = group.head(int(value))
        else:
            column = "ce_score" if mode == "threshold" else "final_score"
            keep = group[group[column] >= value]

            if len(keep) < min_k:
                keep = group.head(min_k)

            keep = keep.head(max_k)

        selected[query_id] = keep["citation"].tolist()

    return selected


def cutoff_settings(args, alpha):
    """
    The (mode, value) pairs to test for one alpha.
    threshold is skipped for alpha = 0, because the ranking then
    does not use the cross-encoder.
    """

    settings = []

    if "topk" in args.cutoff_modes:
        settings += [("topk", k) for k in args.top_k_values]

    if "threshold" in args.cutoff_modes and alpha > 0:
        settings += [("threshold", t) for t in args.threshold_values]

    if "relative" in args.cutoff_modes:
        settings += [("relative", r) for r in args.relative_values]

    return settings


def format_cutoff(mode, value):
    if mode == "topk":
        return f"Top-{int(value)}"

    return f"{mode}>={value:g}"


def write_prediction_file(queries, selected, output_file):
    rows = [
        {
            "query_id": query_id,
            "predicted_citations":
                ";".join(selected.get(query_id, [])),
        }
        for query_id in queries["query_id"]
    ]

    pd.DataFrame(rows).to_csv(output_file, index=False)


def candidate_recall(citations, gold):
    """
    Share of gold citations present anywhere in the candidate pool.
    This is the ceiling for any re-ranker on these candidates.
    """

    pool = citations.groupby("query_id")["citation"].apply(set).to_dict()
    per_query = [
        len(pool.get(query_id, set()) & gold_set) / len(gold_set)
        for query_id, gold_set in gold.items()
        if gold_set
    ]

    return float(np.mean(per_query)) if per_query else 0.0


# ============================================================
# Main
# ============================================================

def load_queries(split, candidates, max_queries):
    queries = load_val() if split == "val" else load_test()
    queries = queries.copy()
    queries["query_id"] = queries["query_id"].astype(str)
    queries["query"] = queries["query"].fillna("").astype(str)

    missing = set(queries["query_id"]) - set(candidates["query_id"])

    if missing and max_queries is None:
        raise ValueError(
            f"No candidates for {len(missing)} {split} queries, "
            f"e.g. {sorted(missing)[:5]}. Check the candidate files "
            f"match --split {split}."
        )

    if max_queries is not None:
        queries = queries[
            queries["query_id"].isin(candidates["query_id"])
        ].head(max_queries)

    return queries.reset_index(drop=True)


def relative_path(path):
    path = Path(path).resolve()

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_best_config(path):
    if not Path(path).exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def run(args, model=None):
    """
    Run the full re-ranking step and return a result dict.
    Pass a loaded model to reuse it across notebook calls.
    """

    output_paths = build_output_paths(args.output_prefix)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("============================================================")
    print("CROSS-ENCODER RE-RANKING")
    print("============================================================")
    print(f"Split: {args.split}")
    print(f"Sources: {args.sources} (top-{args.first_stage_k} each)")
    print(f"Model: {args.model}")

    candidates = load_candidates(args)
    queries = load_queries(args.split, candidates, args.max_queries)
    candidates = candidates[
        candidates["query_id"].isin(queries["query_id"])
    ]

    print(f"Queries: {len(queries):,}")
    print(
        f"Candidate pairs: {len(candidates):,} "
        f"({candidates.groupby('query_id').size().mean():.1f} per query)"
    )

    passages = fetch_passages(
        candidates["doc_id"].unique().tolist(),
        args.text_cache,
    )

    if model is None:
        model = load_cross_encoder(args)

    scored, latencies_ms = score_candidates(
        model, queries, candidates, passages, args.batch_size
    )

    citations = to_citation_level(scored)
    avg_latency_ms = float(np.mean(list(latencies_ms.values())))

    if args.split == "test":
        best = read_best_config(output_paths["best_config"])
        alpha = args.alpha if args.alpha is not None else best.get("alpha", 1.0)

        if args.top_k is not None:
            mode, value = "topk", args.top_k
        elif args.cutoff_mode is not None:
            mode, value = args.cutoff_mode, args.cutoff_value
        elif "cutoff_mode" in best:
            mode, value = best["cutoff_mode"], best["cutoff_value"]
        else:
            mode = "topk"
            value = best.get("prediction_top_k", max(args.top_k_values))

        min_k = best.get("min_k", args.min_k)
        max_k = best.get("max_k", args.max_k)

        ranked_df = rank_citations(citations, alpha)
        selected = select_citations(ranked_df, mode, value, min_k, max_k)
        write_prediction_file(
            queries, selected, output_paths["test_predictions"]
        )

        # Ranked test candidates, used by agent_rerank.py
        test_candidates_file = RESULTS_DIR / f"{args.output_prefix}_test_candidates_top100.csv"
        save_candidate_file(ranked_df, test_candidates_file)
        print(f"Ranked test candidates: {test_candidates_file}")

        avg_predicted = float(np.mean([len(v) for v in selected.values()]))

        print("\n============================================================")
        print("CROSS-ENCODER TEST PREDICTIONS COMPLETE")
        print("============================================================")
        print(f"alpha={alpha} | cutoff={format_cutoff(mode, value)}")
        print(f"Average citations per query: {avg_predicted:.1f}")
        print(f"Average re-rank latency: {avg_latency_ms:.1f} ms/query")
        print(f"Predictions: {output_paths['test_predictions']}")

        return {
            "ranked_df": ranked_df,
            "alpha": alpha,
            "cutoff_mode": mode,
            "cutoff_value": value,
            "avg_latency_ms": avg_latency_ms,
            "model": model,
        }

    # ---------------- validation grid ----------------

    gold = load_gold()
    gold = {q: gold[q] for q in queries["query_id"] if q in gold}
    pool_recall = candidate_recall(citations, gold)

    print(f"\nCandidate pool recall (ceiling): {pool_recall:.4f}")

    grid_rows = []
    ranked_by_alpha = {}

    for alpha in args.alpha_values:
        ranked_df = rank_citations(citations, alpha)
        ranked_by_alpha[alpha] = ranked_df

        for mode, value in cutoff_settings(args, alpha):
            selected = select_citations(
                ranked_df, mode, value, args.min_k, args.max_k
            )

            write_prediction_file(
                queries, selected,
                output_paths["temporary_predictions"],
            )

            if args.max_queries is None:
                summary, _ = evaluate_csv(
                    output_paths["temporary_predictions"]
                )
            else:
                summary = evaluate_subset(
                    output_paths["temporary_predictions"], gold
                )

            avg_predicted = float(
                np.mean([len(v) for v in selected.values()])
            )

            grid_rows.append({
                "model": args.model,
                "sources": args.sources,
                "first_stage_k": args.first_stage_k,
                "alpha": alpha,
                "cutoff_mode": mode,
                "cutoff_value": value,
                "avg_predicted": avg_predicted,
                "macro_precision": summary["macro_precision"],
                "macro_recall": summary["macro_recall"],
                "macro_f1": summary["macro_f1"],
                "avg_latency_ms": avg_latency_ms,
            })

            print(
                f"alpha={alpha:.2f} | {format_cutoff(mode, value):<16} | "
                f"avg n={avg_predicted:5.1f} | "
                f"P={summary['macro_precision']:.4f} | "
                f"R={summary['macro_recall']:.4f} | "
                f"F1={summary['macro_f1']:.4f}"
            )

    grid_df = pd.DataFrame(grid_rows).sort_values(
        ["macro_f1", "macro_recall", "macro_precision", "alpha"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    grid_df.to_csv(output_paths["grid_summary"], index=False)

    best = grid_df.iloc[0]
    best_alpha = float(best["alpha"])
    best_mode = str(best["cutoff_mode"])
    best_value = float(best["cutoff_value"])

    if best_mode == "topk":
        best_value = int(best_value)

    best_ranked_df = ranked_by_alpha[best_alpha]
    best_selected = select_citations(
        best_ranked_df, best_mode, best_value, args.min_k, args.max_k
    )

    write_prediction_file(
        queries, best_selected,
        output_paths["best_predictions"],
    )

    # Best result for each cutoff mode, for the report
    mode_best = (
        grid_df.drop_duplicates("cutoff_mode")
        [["cutoff_mode", "alpha", "cutoff_value", "avg_predicted", "macro_f1"]]
    )

    save_candidate_file(
        best_ranked_df, output_paths["candidates"]
    )

    best_config = {
        "system": "Cross-encoder re-ranking",
        "model": args.model,
        "sources": args.sources,
        "bm25_candidates": relative_path(args.bm25_candidates),
        "dense_candidates": relative_path(args.dense_candidates),
        "first_stage_k": args.first_stage_k,
        "max_length": args.max_length,
        "passage_format": "citation title text",
        "fusion": "alpha * minmax(ce) + (1 - alpha) * minmax(first_stage_rrf)",
        "rrf_k": RRF_K,
        "alpha": best_alpha,
        "cutoff_mode": best_mode,
        "cutoff_value": best_value,
        "min_k": args.min_k,
        "max_k": args.max_k,
        "prediction_top_k": best_value if best_mode == "topk" else None,
        "avg_predicted": float(best["avg_predicted"]),
        "candidate_pool_recall": pool_recall,
        "macro_precision": float(best["macro_precision"]),
        "macro_recall": float(best["macro_recall"]),
        "macro_f1": float(best["macro_f1"]),
        "avg_latency_ms": avg_latency_ms,
        "n_queries": len(queries),
    }

    with open(output_paths["best_config"], "w", encoding="utf-8") as file:
        json.dump(best_config, file, indent=2, ensure_ascii=False)

    if output_paths["temporary_predictions"].exists():
        output_paths["temporary_predictions"].unlink()

    print("\n============================================================")
    print("CROSS-ENCODER RE-RANKING COMPLETE")
    print("============================================================")
    print(
        f"Best alpha: {best_alpha} | "
        f"Best cutoff: {format_cutoff(best_mode, best_value)} | "
        f"avg n={best['avg_predicted']:.1f}"
    )
    print(f"Macro Precision: {best['macro_precision']:.4f}")
    print(f"Macro Recall: {best['macro_recall']:.4f}")
    print(f"Macro F1: {best['macro_f1']:.4f}")
    print(f"Average re-rank latency: {avg_latency_ms:.1f} ms/query")
    print("\nBest result per cutoff mode:")
    print(mode_best.to_string(index=False))
    print("\nFILES SAVED:")

    for key in ["grid_summary", "best_predictions", "candidates", "best_config"]:
        print(f"  {output_paths[key]}")

    return {
        "grid_df": grid_df,
        "mode_best": mode_best,
        "best_config": best_config,
        "ranked_df": best_ranked_df,
        "scored_pairs": scored,
        "model": model,
    }


def evaluate_subset(prediction_file, gold):
    """
    Smoke-test path: evaluate only the queries that were run.
    Uses the shared evaluator's functions, not new scoring logic.
    """

    from evaluation import evaluate_predictions, parse_citations

    df = pd.read_csv(prediction_file, keep_default_na=False)
    predictions = {
        str(row.query_id): parse_citations(row.predicted_citations)
        for row in df.itertuples(index=False)
    }

    summary, _ = evaluate_predictions(predictions, gold)

    return summary


def save_candidate_file(ranked_df, output_file):
    df = ranked_df.copy()
    df["rank"] = df.groupby("query_id").cumcount() + 1
    df = df[df["rank"] <= CANDIDATE_OUTPUT_K]

    df[[
        "query_id", "rank", "doc_id", "citation", "source_type",
        "bm25_rank", "dense_rank", "first_stage_rrf",
        "ce_score", "final_score",
    ]].to_csv(output_file, index=False)


def main():
    args = parse_args()

    if args.first_stage_k <= 0:
        raise ValueError("--first-stage-k must be positive.")

    for alpha in args.alpha_values + ([args.alpha] if args.alpha is not None else []):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha values must be between 0 and 1.")

    unknown = set(args.cutoff_modes) - set(CUTOFF_MODES)

    if unknown:
        raise ValueError(f"Unknown cutoff modes: {unknown}")

    if not 1 <= args.min_k <= args.max_k:
        raise ValueError("Need 1 <= --min-k <= --max-k.")

    if (args.cutoff_mode is None) != (args.cutoff_value is None):
        raise ValueError("Give --cutoff-mode and --cutoff-value together.")

    run(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
