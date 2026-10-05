import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import load_test, load_train, load_val
from evaluation import evaluate_csv


try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DEFAULT_INDEX_DIR = RESULTS_DIR / "faiss_dense_full"

DEFAULT_TOP_K_VALUES = [10, 20, 50]
DEFAULT_OUTPUT_PREFIX = "dense_full"
CANDIDATE_OUTPUT_K = 100
INITIAL_SEARCH_DEPTH = 1000
MAX_SEARCH_DEPTH = 10000


def require_dependencies():
    missing = []

    if faiss is None:
        missing.append("faiss-cpu")

    if SentenceTransformer is None:
        missing.append("sentence-transformers")

    if missing:
        raise SystemExit(
            "Missing dense retrieval dependencies: "
            + ", ".join(missing)
            + "\nInstall them with:\n"
            + "  python -m pip install -r requirements.txt"
        )


def parse_top_k_values(value):
    output = []

    for item in value.split(","):
        item = item.strip()

        if item:
            output.append(
                int(item)
            )

    if not output:
        raise argparse.ArgumentTypeError(
            "At least one top-k value is required."
        )

    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run full-corpus dense retrieval experiments using a "
            "previously built FAISS index."
        )
    )

    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory containing dense.faiss and metadata.csv.gz.",
    )

    parser.add_argument(
        "--split",
        choices=["val", "test", "train"],
        default="val",
        help=(
            "val: run the validation grid. test: write test predictions "
            "and test candidates. train: write training-query candidates "
            "(for cross-encoder fine-tuning)."
        ),
    )

    parser.add_argument(
        "--top-k-values",
        type=parse_top_k_values,
        default=DEFAULT_TOP_K_VALUES,
        help="Comma-separated Top-K values for validation evaluation.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Prediction Top-K for test mode. Defaults to the best "
            "validation Top-K if available."
        ),
    )

    parser.add_argument(
        "--candidate-output-k",
        type=int,
        default=CANDIDATE_OUTPUT_K,
        help="Number of dense candidates to save per query.",
    )

    parser.add_argument(
        "--search-depth",
        type=int,
        default=INITIAL_SEARCH_DEPTH,
        help="Initial row-level FAISS search depth per query.",
    )

    parser.add_argument(
        "--max-search-depth",
        type=int,
        default=MAX_SEARCH_DEPTH,
        help="Maximum row-level FAISS search depth per query.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Query embedding batch size.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Optional SentenceTransformer device, e.g. cpu, cuda, or mps.",
    )

    parser.add_argument(
        "--faiss-threads",
        type=int,
        default=None,
        help="Optional number of FAISS CPU threads.",
    )

    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix for result files written under results/.",
    )

    return parser.parse_args()


def uses_e5_prompt(model_name):
    return "e5" in model_name.lower()


def build_query_text(query, model_name):
    query = str(query or "").strip()

    if uses_e5_prompt(model_name):
        return "query: " + query

    return query


def encode_texts(model, texts, batch_size):
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(
        embeddings,
        dtype=np.float32,
    )


def load_index_config(index_dir):
    config_file = index_dir / "dense_index_config.json"

    if not config_file.exists():
        raise FileNotFoundError(
            f"Missing dense index config: {config_file}"
        )

    with open(
        config_file,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_metadata(index_dir):
    metadata_file = index_dir / "metadata.csv.gz"

    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Missing dense metadata: {metadata_file}"
        )

    metadata = pd.read_csv(
        metadata_file,
        keep_default_na=False,
    )

    required = {
        "doc_int_id",
        "doc_id",
        "citation",
        "source_type",
    }

    missing = required - set(metadata.columns)

    if missing:
        raise ValueError(
            f"Dense metadata missing columns: {missing}"
        )

    expected_ids = np.arange(
        len(metadata),
        dtype=np.int64,
    )

    actual_ids = metadata["doc_int_id"].to_numpy(
        dtype=np.int64
    )

    if not np.array_equal(
        expected_ids,
        actual_ids,
    ):
        raise ValueError(
            "metadata.csv.gz must be ordered by contiguous doc_int_id."
        )

    return metadata


def build_output_paths(output_prefix):
    prefix = output_prefix.strip()

    if not prefix:
        raise ValueError("--output-prefix cannot be empty.")

    if "/" in prefix or "\\" in prefix:
        raise ValueError("--output-prefix must be a file prefix, not a path.")

    return {
        "grid_summary":
            RESULTS_DIR / f"{prefix}_grid_summary.csv",

        "best_predictions":
            RESULTS_DIR / f"{prefix}_best_predictions.csv",

        "candidates":
            RESULTS_DIR / f"{prefix}_candidates_top100.csv",

        "best_config":
            RESULTS_DIR / f"{prefix}_config.json",

        "test_predictions":
            RESULTS_DIR / f"{prefix}_test_predictions.csv",

        "temporary_predictions":
            RESULTS_DIR / f"_{prefix}_temporary_predictions.csv",
    }


def retrieve_unique_citations(
    index,
    metadata,
    query_embedding,
    target_k,
    search_depth,
    max_search_depth,
):
    depth = max(
        search_depth,
        target_k * 5,
    )

    depth = min(
        depth,
        index.ntotal,
    )

    while True:
        scores, ids = index.search(
            query_embedding,
            depth,
        )

        output = []
        seen = set()

        for row_rank, (score, doc_int_id) in enumerate(
            zip(scores[0], ids[0]),
            start=1,
        ):
            if doc_int_id < 0:
                continue

            item = metadata.iloc[
                int(doc_int_id)
            ]

            citation = str(
                item["citation"]
            ).strip()

            if not citation or citation in seen:
                continue

            seen.add(
                citation
            )

            output.append({
                "doc_id":
                    item["doc_id"],

                "citation":
                    citation,

                "source_type":
                    item["source_type"],

                "row_rank":
                    row_rank,

                "dense_score":
                    float(score),
            })

            if len(output) >= target_k:
                return output

        if depth >= index.ntotal:
            return output

        if depth >= max_search_depth:
            return output

        depth = min(
            depth * 2,
            max_search_depth,
            index.ntotal,
        )


def write_prediction_file(
    queries,
    ranked_results,
    top_k,
    output_file,
):
    rows = []

    for i, row in queries.iterrows():
        citations = [
            item["citation"]
            for item in ranked_results[i][:top_k]
        ]

        rows.append({
            "query_id":
                row["query_id"],

            "predicted_citations":
                ";".join(citations),
        })

    pd.DataFrame(
        rows
    ).to_csv(
        output_file,
        index=False,
    )


def evaluate_ranked_results(
    queries,
    ranked_results,
    top_k,
    temporary_prediction_file,
):
    write_prediction_file(
        queries,
        ranked_results,
        top_k,
        temporary_prediction_file,
    )

    summary, _ = evaluate_csv(
        temporary_prediction_file
    )

    return summary


def save_candidate_file(
    queries,
    ranked_results,
    candidate_output_k,
    output_file,
):
    rows = []

    for i, row in queries.iterrows():
        query_id = row["query_id"]

        for rank, item in enumerate(
            ranked_results[i][:candidate_output_k],
            start=1,
        ):
            rows.append({
                "query_id":
                    query_id,

                "rank":
                    rank,

                "doc_id":
                    item["doc_id"],

                "citation":
                    item["citation"],

                "source_type":
                    item["source_type"],

                "row_rank":
                    item["row_rank"],

                "dense_score":
                    item["dense_score"],
            })

    pd.DataFrame(
        rows
    ).to_csv(
        output_file,
        index=False,
    )


def run_retrieval(
    queries,
    model,
    model_name,
    index,
    metadata,
    target_k,
    batch_size,
    search_depth,
    max_search_depth,
):
    query_texts = [
        build_query_text(
            query,
            model_name,
        )
        for query in queries["query"].fillna("").astype(str).tolist()
    ]

    embedding_start = time.perf_counter()

    query_embeddings = encode_texts(
        model,
        query_texts,
        batch_size,
    )

    embedding_elapsed_ms = (
        time.perf_counter()
        - embedding_start
    ) * 1000

    embedding_latency_ms = (
        embedding_elapsed_ms
        / len(queries)
    )

    ranked_results = []
    query_latencies_ms = []

    for i, row in queries.iterrows():
        query_id = row["query_id"]

        start = time.perf_counter()

        embedding = query_embeddings[
            i : i + 1
        ]

        ranked = retrieve_unique_citations(
            index,
            metadata,
            embedding,
            target_k,
            search_depth,
            max_search_depth,
        )

        elapsed_ms = (
            time.perf_counter()
            - start
        ) * 1000

        total_latency_ms = (
            embedding_latency_ms
            + elapsed_ms
        )

        ranked_results.append(
            ranked
        )

        query_latencies_ms.append(
            total_latency_ms
        )

        print(
            f"{query_id} | "
            f"candidates={len(ranked)} | "
            f"search={elapsed_ms:.1f} ms | "
            f"total={total_latency_ms:.1f} ms"
        )

    return ranked_results, query_latencies_ms


def choose_test_top_k(args, best_config_file):
    if args.top_k is not None:
        return args.top_k

    if best_config_file.exists():
        with open(
            best_config_file,
            "r",
            encoding="utf-8",
        ) as file:
            config = json.load(file)

        return int(
            config["prediction_top_k"]
        )

    return max(
        args.top_k_values
    )


def main():
    args = parse_args()
    require_dependencies()

    if args.candidate_output_k <= 0:
        raise ValueError("--candidate-output-k must be positive.")

    if args.search_depth <= 0:
        raise ValueError("--search-depth must be positive.")

    if args.max_search_depth < args.search_depth:
        raise ValueError("--max-search-depth must be >= --search-depth.")

    output_paths = build_output_paths(
        args.output_prefix
    )

    if args.faiss_threads is not None:
        faiss.omp_set_num_threads(
            args.faiss_threads
        )

    index_dir = args.index_dir
    index_file = index_dir / "dense.faiss"

    if not index_file.exists():
        raise FileNotFoundError(
            f"Missing dense FAISS index: {index_file}\n"
            "Run dense_build_index.py first."
        )

    index_config = load_index_config(
        index_dir
    )

    model_name = index_config["model"]

    print("============================================================")
    print("FULL DENSE RETRIEVAL EXPERIMENT")
    print("============================================================")
    print(f"Split: {args.split}")
    print(f"Model: {model_name}")
    print(f"Index: {index_file}")

    index = faiss.read_index(
        str(index_file)
    )

    metadata = load_metadata(
        index_dir
    )

    if len(metadata) != index.ntotal:
        raise ValueError(
            "Metadata row count does not match FAISS index size: "
            f"{len(metadata):,} vs {index.ntotal:,}"
        )

    print(f"Indexed documents: {index.ntotal:,}")

    model = SentenceTransformer(
        model_name,
        device=args.device,
    )

    if args.split == "val":
        queries = load_val().reset_index(
            drop=True
        )
        target_k = max(
            max(args.top_k_values),
            args.candidate_output_k,
        )
    elif args.split == "train":
        queries = load_train().reset_index(
            drop=True
        )
        target_k = args.candidate_output_k
    else:
        queries = load_test().reset_index(
            drop=True
        )
        test_top_k = choose_test_top_k(
            args,
            output_paths["best_config"],
        )
        target_k = max(
            test_top_k,
            args.candidate_output_k,
        )

    print(f"Queries: {len(queries):,}")
    print(f"Retrieval target K: {target_k}")

    ranked_results, query_latencies_ms = run_retrieval(
        queries,
        model,
        model_name,
        index,
        metadata,
        target_k,
        args.batch_size,
        args.search_depth,
        args.max_search_depth,
    )

    avg_latency_ms = (
        sum(query_latencies_ms)
        / len(query_latencies_ms)
    )

    if args.split in {"test", "train"}:
        split_candidates_file = (
            RESULTS_DIR
            / f"{args.output_prefix}_{args.split}_candidates_top{args.candidate_output_k}.csv"
        )

        save_candidate_file(
            queries,
            ranked_results,
            args.candidate_output_k,
            split_candidates_file,
        )

        print(f"\n{args.split} candidates: {split_candidates_file}")

    if args.split == "train":
        print(f"Average search latency: {avg_latency_ms:.2f} ms/query")
        return

    if args.split == "test":
        write_prediction_file(
            queries,
            ranked_results,
            test_top_k,
            output_paths["test_predictions"],
        )

        print("\n============================================================")
        print("DENSE TEST PREDICTIONS COMPLETE")
        print("============================================================")
        print(f"Top-K: {test_top_k}")
        print(f"Average search latency: {avg_latency_ms:.2f} ms/query")
        print(f"Predictions: {output_paths['test_predictions']}")
        return

    grid_rows = []

    for top_k in args.top_k_values:
        summary = evaluate_ranked_results(
            queries,
            ranked_results,
            top_k,
            output_paths["temporary_predictions"],
        )

        row = {
            "model":
                model_name,

            "top_k":
                top_k,

            "search_depth":
                args.search_depth,

            "max_search_depth":
                args.max_search_depth,

            "macro_precision":
                summary["macro_precision"],

            "macro_recall":
                summary["macro_recall"],

            "macro_f1":
                summary["macro_f1"],

            "avg_latency_ms":
                avg_latency_ms,
        }

        grid_rows.append(
            row
        )

        print(
            f"Top-{top_k:<2d} | "
            f"P={summary['macro_precision']:.4f} | "
            f"R={summary['macro_recall']:.4f} | "
            f"F1={summary['macro_f1']:.4f}"
        )

    grid_df = pd.DataFrame(
        grid_rows
    ).sort_values(
        by=[
            "macro_f1",
            "macro_recall",
            "macro_precision",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    grid_df.to_csv(
        output_paths["grid_summary"],
        index=False,
    )

    best = grid_df.iloc[0]
    best_top_k = int(
        best["top_k"]
    )

    write_prediction_file(
        queries,
        ranked_results,
        best_top_k,
        output_paths["best_predictions"],
    )

    save_candidate_file(
        queries,
        ranked_results,
        args.candidate_output_k,
        output_paths["candidates"],
    )

    best_config = {
        "system":
            "Full-corpus dense retrieval",

        "model":
            model_name,

        "index_dir":
            str(index_dir.relative_to(ROOT)),

        "index_document_count":
            int(index.ntotal),

        "prediction_top_k":
            best_top_k,

        "candidate_output_k":
            args.candidate_output_k,

        "search_depth":
            args.search_depth,

        "max_search_depth":
            args.max_search_depth,

        "macro_precision":
            float(best["macro_precision"]),

        "macro_recall":
            float(best["macro_recall"]),

        "macro_f1":
            float(best["macro_f1"]),

        "avg_latency_ms":
            float(avg_latency_ms),

        "query_format":
            (
                "query: text"
                if uses_e5_prompt(model_name)
                else "text"
            ),

        "passage_format":
            index_config.get(
                "passage_format",
                "",
            ),
    }

    with open(
        output_paths["best_config"],
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            best_config,
            file,
            indent=2,
            ensure_ascii=False,
        )

    if output_paths["temporary_predictions"].exists():
        output_paths["temporary_predictions"].unlink()

    print("\n============================================================")
    print("FULL DENSE RETRIEVAL COMPLETE")
    print("============================================================")
    print(f"Best Top-K: {best_top_k}")
    print(f"Macro Precision: {best['macro_precision']:.4f}")
    print(f"Macro Recall: {best['macro_recall']:.4f}")
    print(f"Macro F1: {best['macro_f1']:.4f}")
    print(f"Average search latency: {avg_latency_ms:.2f} ms/query")
    print("\nFILES SAVED:")
    print(f"Grid summary:\n{output_paths['grid_summary']}")
    print(f"\nBest predictions:\n{output_paths['best_predictions']}")
    print(f"\nTop-{args.candidate_output_k} candidates:\n{output_paths['candidates']}")
    print(f"\nBest configuration:\n{output_paths['best_config']}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
