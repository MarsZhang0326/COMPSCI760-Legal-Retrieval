import os
import re
import json
import base64
import shutil
import time
from pathlib import Path

# ============================================================
# Pyserini environment
# ============================================================

# Pyserini 0.37 imports its OpenAI encoder even though
# this experiment only uses local Lucene BM25.
# No OpenAI API request is made.
os.environ.setdefault(
    "OPENAI_API_KEY",
    "PYSERINI_BM25_NOT_USED"
)

java_exe = shutil.which("java")

if java_exe:
    os.environ["JAVA_HOME"] = str(
        Path(java_exe).resolve().parent.parent
    )


# Must import Pyserini AFTER environment setup.
from pyserini.search.lucene import LuceneSearcher
from pyserini.analysis import JWhiteSpaceAnalyzer

# Long queries can have more than 1,024 unique terms. Lucene's default
# limit is 1,024 clauses per query, so raise it to keep every term.
from jnius import autoclass
autoclass("org.apache.lucene.search.IndexSearcher").setMaxClauseCount(16384)

import pandas as pd

from data_loader import load_val
from evaluation import evaluate_csv


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

LAW_INDEX = (
    ROOT
    / "results"
    / "lucene_bm25_law"
)

COURT_INDEX = (
    ROOT
    / "results"
    / "lucene_bm25_court"
)

RESULTS_DIR = (
    ROOT
    / "results"
)

FINAL_DIR = (
    RESULTS_DIR
    / "bm25_final"
)

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TEMP_PREDICTION_FILE = (
    FINAL_DIR
    / "_temporary_predictions.csv"
)

GRID_SUMMARY_FILE = (
    RESULTS_DIR
    / "bm25_final_grid_summary.csv"
)

ABLATION_SUMMARY_FILE = (
    RESULTS_DIR
    / "bm25_ablation_summary.csv"
)

BEST_PREDICTIONS_FILE = (
    RESULTS_DIR
    / "bm25_best_predictions.csv"
)

BEST_CANDIDATES_FILE = (
    RESULTS_DIR
    / "bm25_candidates_top100.csv"
)

BEST_CONFIG_FILE = (
    RESULTS_DIR
    / "bm25_best_config.json"
)


# ============================================================
# Experiment settings
# ============================================================

# Standard BM25 parameter grid.
K1_VALUES = [
    0.9,
    1.2,
    1.5,
    1.8
]

B_VALUES = [
    0.4,
    0.75,
    0.9
]

# Law/Court fusion weights.
#
# (law_weight, court_weight)
SOURCE_WEIGHT_CONFIGS = [
    (1.0, 1.0),
    (1.5, 1.0),
    (2.0, 1.0),
    (3.0, 1.0),
    (1.0, 1.5),
    (1.0, 2.0),
]

TOP_K_VALUES = [
    10,
    20,
    50
]

# Number of unique citations retrieved from each source.
#
# We retrieve substantially more than final Top-K because
# weighted fusion and citation deduplication happen later.
SOURCE_K = 300

# Final candidate file for downstream Dense/Reranking work.
CANDIDATE_OUTPUT_K = 100

# Reciprocal Rank Fusion constant.
RRF_K = 60

# Row-level search depth.
INITIAL_SEARCH_DEPTH = 500
MAX_SEARCH_DEPTH = 5000

# Exact citation lookup depth.
EXACT_SEARCH_DEPTH = 50


# ============================================================
# Query modes
# ============================================================

MODE_RAW = "raw"
MODE_CITATION = "citation_aware"
MODE_HYBRID = "citation_hybrid"

QUERY_MODES = [
    MODE_RAW,
    MODE_CITATION,
    MODE_HYBRID
]


# ============================================================
# Language-neutral tokenisation
# ============================================================

def tokenize_neutral(text):
    """
    Same tokenisation strategy used when the split indexes
    were built.

    No English stemming is used.

    Examples kept reasonably intact:
        StPO
        Art. 221
        BGE 137 IV 122
        1B_210/2023
    """

    text = str(text).lower()

    tokens = re.findall(
        r"\w+(?:[./_-]\w+)*",
        text,
        flags=re.UNICODE
    )

    return " ".join(tokens)


# ============================================================
# Citation normalisation
# ============================================================

def normalize_citation(citation):
    """
    Normalised representation used ONLY for exact matching.

    The original canonical citation is always preserved
    for final predictions.
    """

    citation = str(citation)

    # Remove letter-level subreference:
    # Art. 221 Abs. 1 lit. b StPO
    # ->
    # Art. 221 Abs. 1 StPO
    citation = re.sub(
        r"\s+lit\.\s*[a-zA-Z]",
        "",
        citation,
        flags=re.IGNORECASE
    )

    citation = citation.lower()

    citation = re.sub(
        r"\s+",
        " ",
        citation
    ).strip()

    return citation


# ============================================================
# Explicit citation extraction
# ============================================================

def extract_explicit_citations(query):
    """
    Extract citations explicitly written in the user query.

    No gold labels are used.

    Supports:
      - statutes/articles
      - BGE citations
      - Federal Court case numbers
    """

    query = str(query)

    citations = []


    # --------------------------------------------------------
    # Statutory article
    #
    # Example:
    # Art. 221 Abs. 1 lit. b StPO
    # --------------------------------------------------------

    article_pattern = (
        r"\bArt\.\s*\d+[a-zA-Z]*"
        r"(?:\s+Abs\.\s*\d+[a-zA-Z]*)?"
        r"(?:\s+lit\.\s*[a-zA-Z])?"
        r"\s+[A-Z][A-Za-z0-9\-]*"
    )

    article_matches = re.findall(
        article_pattern,
        query,
        flags=re.IGNORECASE
    )

    for citation in article_matches:

        citation = re.sub(
            r"\s+lit\.\s*[a-zA-Z]",
            "",
            citation,
            flags=re.IGNORECASE
        )

        citation = re.sub(
            r"\s+",
            " ",
            citation
        ).strip()

        citations.append(
            citation
        )


    # --------------------------------------------------------
    # BGE
    #
    # Example:
    # BGE 137 IV 122 E. 6.2
    # --------------------------------------------------------

    bge_pattern = (
        r"\bBGE\s+\d+\s+[IVX]+\s+\d+"
        r"(?:\s+E\.\s*[\d.]+)?"
    )

    bge_matches = re.findall(
        bge_pattern,
        query,
        flags=re.IGNORECASE
    )

    citations.extend(
        bge_matches
    )


    # --------------------------------------------------------
    # Federal Court case
    #
    # Examples:
    # 1B_210/2023
    # 7B_496/2025 E. 3.2
    # --------------------------------------------------------

    case_pattern = (
        r"\b\d+[A-Z]+_\d+/\d{4}"
        r"(?:\s+E\.\s*[\d.]+)?"
    )

    case_matches = re.findall(
        case_pattern,
        query,
        flags=re.IGNORECASE
    )

    citations.extend(
        case_matches
    )


    # Remove duplicate extracted citations
    # while preserving order.
    output = []
    seen = set()

    for citation in citations:

        citation = re.sub(
            r"\s+",
            " ",
            citation
        ).strip()

        key = normalize_citation(
            citation
        )

        if key not in seen:

            seen.add(
                key
            )

            output.append(
                citation
            )

    return output


# ============================================================
# Citation-aware query
# ============================================================

def build_citation_aware_query(query):
    """
    Give explicit legal references more lexical importance.

    This uses only information already present in the query.
    """

    query = str(query)

    citations = extract_explicit_citations(
        query
    )

    if not citations:
        return query

    citation_text = " ".join(
        citations
    )

    return (
        citation_text
        + " "
        + citation_text
        + " "
        + citation_text
        + " "
        + query
    )


# ============================================================
# Lucene document ID decoding
# ============================================================

def decode_hit(hit):
    """
    Index ID format created by bm25_build_index.py:

        original_doc_id | base64(canonical citation)
    """

    parts = hit.docid.split(
        "|",
        1
    )

    if len(parts) != 2:

        raise ValueError(
            f"Unexpected Lucene document ID: "
            f"{hit.docid}"
        )

    original_doc_id = parts[0]

    encoded_citation = parts[1]

    citation = (
        base64.urlsafe_b64decode(
            encoded_citation.encode(
                "ascii"
            )
        )
        .decode(
            "utf-8"
        )
        .strip()
    )

    return {
        "doc_id":
            original_doc_id,

        "citation":
            citation,

        "bm25_score":
            float(hit.score)
    }


# ============================================================
# Retrieve unique citations from one source
# ============================================================

def retrieve_unique(
    searcher,
    query,
    target_k
):
    """
    Lucene retrieves rows.

    Court data can contain multiple rows for the same
    canonical citation.

    Therefore:
        row retrieval
          ->
        citation deduplication
          ->
        ranked unique citations
    """

    depth = max(
        INITIAL_SEARCH_DEPTH,
        target_k * 3
    )

    while True:

        hits = searcher.search(
            query,
            k=depth
        )

        results = []
        seen = set()


        for hit in hits:

            item = decode_hit(
                hit
            )

            citation = item[
                "citation"
            ]

            if (
                citation
                and citation not in seen
            ):

                seen.add(
                    citation
                )

                item[
                    "source_rank"
                ] = len(results) + 1

                results.append(
                    item
                )


            if len(results) >= target_k:

                return results


        # Lucene returned fewer documents than requested.
        if len(hits) < depth:

            return results


        if depth >= MAX_SEARCH_DEPTH:

            return results


        depth = min(
            depth * 2,
            MAX_SEARCH_DEPTH
        )


# ============================================================
# Exact citation candidate lookup
# ============================================================

def find_exact_candidates(
    law_searcher,
    court_searcher,
    original_query
):
    """
    Find canonical corpus citations that exactly match
    citations explicitly written in the query.

    This is NOT gold-label lookup.

    Search flow:
        query citation
          ->
        search law/court indexes
          ->
        compare canonical citation text
          ->
        keep only genuine exact corpus matches
    """

    explicit_citations = (
        extract_explicit_citations(
            original_query
        )
    )

    if not explicit_citations:
        return []


    output = []
    seen = set()


    for explicit in explicit_citations:

        target_key = normalize_citation(
            explicit
        )

        explicit_query = (
            tokenize_neutral(
                explicit
            )
        )


        for source_type, searcher in [
            ("law", law_searcher),
            ("court", court_searcher)
        ]:

            hits = searcher.search(
                explicit_query,
                k=EXACT_SEARCH_DEPTH
            )


            for hit in hits:

                item = decode_hit(
                    hit
                )

                citation_key = (
                    normalize_citation(
                        item["citation"]
                    )
                )


                if citation_key == target_key:

                    citation = item[
                        "citation"
                    ]


                    if citation in seen:
                        break


                    seen.add(
                        citation
                    )


                    output.append({
                        "doc_id":
                            item[
                                "doc_id"
                            ],

                        "citation":
                            citation,

                        "source_type":
                            source_type,

                        "source_rank":
                            0,

                        "bm25_score":
                            item[
                                "bm25_score"
                            ],

                        "fusion_score":
                            None,

                        "exact_match":
                            True
                    })

                    break


    return output


# ============================================================
# Weighted Reciprocal Rank Fusion
# ============================================================

def weighted_rrf(
    law_results,
    court_results,
    law_weight,
    court_weight
):
    """
    Source-balanced RRF.

    Because law and court BM25 score distributions are not
    necessarily comparable, fusion uses ranks instead of
    directly mixing raw BM25 scores.
    """

    fused_scores = {}
    metadata = {}


    source_data = [
        (
            "law",
            law_results,
            law_weight
        ),

        (
            "court",
            court_results,
            court_weight
        )
    ]


    for (
        source_type,
        results,
        source_weight
    ) in source_data:

        for rank, item in enumerate(
            results,
            start=1
        ):

            citation = item[
                "citation"
            ]

            score = (
                source_weight
                / (
                    RRF_K
                    + rank
                )
            )


            fused_scores[
                citation
            ] = (
                fused_scores.get(
                    citation,
                    0.0
                )
                + score
            )


            # Law/court citation overlap is expected to be
            # negligible, but preserve the strongest source
            # metadata if overlap occurs.
            if citation not in metadata:

                metadata[
                    citation
                ] = {
                    "doc_id":
                        item[
                            "doc_id"
                        ],

                    "citation":
                        citation,

                    "source_type":
                        source_type,

                    "source_rank":
                        rank,

                    "bm25_score":
                        item[
                            "bm25_score"
                        ],

                    "exact_match":
                        False
                }


    ranked_citations = sorted(
        fused_scores.keys(),
        key=lambda citation: (
            -fused_scores[
                citation
            ],
            metadata[
                citation
            ][
                "source_rank"
            ],
            citation
        )
    )


    output = []

    for citation in ranked_citations:

        item = metadata[
            citation
        ].copy()

        item[
            "fusion_score"
        ] = fused_scores[
            citation
        ]

        output.append(
            item
        )

    return output


# ============================================================
# Exact-citation hybrid
# ============================================================

def prepend_exact_candidates(
    exact_candidates,
    fused_results
):
    """
    Explicit citations genuinely found in the corpus are
    placed before normal fused BM25 results.

    Remaining results retain their weighted-RRF order.
    """

    output = []
    seen = set()


    for item in exact_candidates:

        citation = item[
            "citation"
        ]

        if citation not in seen:

            seen.add(
                citation
            )

            output.append(
                item.copy()
            )


    for item in fused_results:

        citation = item[
            "citation"
        ]

        if citation not in seen:

            seen.add(
                citation
            )

            output.append(
                item.copy()
            )


    return output


# ============================================================
# Prediction file helper
# ============================================================

def write_prediction_file(
    validation,
    ranked_results,
    top_k,
    output_file
):
    rows = []


    for i, row in validation.iterrows():

        citations = [
            item["citation"]
            for item
            in ranked_results[i][
                :top_k
            ]
        ]


        rows.append({
            "query_id":
                row[
                    "query_id"
                ],

            "predicted_citations":
                ";".join(
                    citations
                )
        })


    pd.DataFrame(
        rows
    ).to_csv(
        output_file,
        index=False
    )


# ============================================================
# Evaluation helper
# ============================================================

def evaluate_ranked_results(
    validation,
    ranked_results,
    top_k
):
    write_prediction_file(
        validation,
        ranked_results,
        top_k,
        TEMP_PREDICTION_FILE
    )

    summary, details = (
        evaluate_csv(
            TEMP_PREDICTION_FILE
        )
    )

    return summary


# ============================================================
# Load validation data
# ============================================================

print(
    "============================================================"
)

print(
    "FINAL SPLIT BM25 EXPERIMENT"
)

print(
    "============================================================"
)


# ============================================================
# Candidate-only mode (train / val / test)
# ------------------------------------------------------------
# Writes first-stage candidates for any query split, for example
# training queries for cross-encoder fine-tuning:
#
#   python bm25_run_experiments.py --candidates-split train
#   python bm25_run_experiments.py --candidates-split test
#   python bm25_run_experiments.py --candidates-split val --candidate-output-k 500
#
# This mode skips the parameter search. It uses the saved best
# configuration in results/bm25_best_config.json, writes
#   results/bm25_<split>_candidates_top<K>.csv
# with the same columns as bm25_candidates_top100.csv, and exits.
#
# Without --candidates-split, the validation experiment below
# runs exactly as before.
# ============================================================

import argparse
import sys

_cli = argparse.ArgumentParser(add_help=False)

_cli.add_argument(
    "--candidates-split",
    choices=["train", "val", "test"],
    default=None
)

_cli.add_argument(
    "--candidate-output-k",
    type=int,
    default=CANDIDATE_OUTPUT_K
)

_cli_args, _ = _cli.parse_known_args()


if _cli_args.candidates_split is not None:

    from data_loader import load_train, load_test

    split = _cli_args.candidates_split
    output_k = _cli_args.candidate_output_k

    if not BEST_CONFIG_FILE.exists():
        sys.exit(
            f"Missing {BEST_CONFIG_FILE}. "
            "Run the validation experiment first."
        )

    with open(BEST_CONFIG_FILE, "r", encoding="utf-8") as file:
        cfg = json.load(file)

    split_queries = {
        "train": load_train,
        "val": load_val,
        "test": load_test,
    }[split]().reset_index(drop=True)

    source_k = max(SOURCE_K, output_k)

    print("============================================================")
    print(f"BM25 CANDIDATES: {split} ({len(split_queries):,} queries)")
    print(
        f"mode={cfg['mode']} | k1={cfg['k1']} | b={cfg['b']} | "
        f"law_weight={cfg['law_weight']} | "
        f"court_weight={cfg['court_weight']} | top-{output_k}"
    )
    print("============================================================")

    cand_law_searcher = LuceneSearcher(str(LAW_INDEX))
    cand_court_searcher = LuceneSearcher(str(COURT_INDEX))

    for searcher in [cand_law_searcher, cand_court_searcher]:
        searcher.set_analyzer(JWhiteSpaceAnalyzer())
        searcher.set_bm25(float(cfg["k1"]), float(cfg["b"]))

    candidate_rows = []
    start_all = time.perf_counter()

    for i, row in split_queries.iterrows():

        original_query = str(row["query"] or "")

        if cfg["mode"] == MODE_RAW:
            query = tokenize_neutral(original_query)
        else:
            query = tokenize_neutral(
                build_citation_aware_query(original_query)
            )

        law_results = retrieve_unique(
            cand_law_searcher, query, source_k
        )

        court_results = retrieve_unique(
            cand_court_searcher, query, source_k
        )

        ranked = weighted_rrf(
            law_results,
            court_results,
            float(cfg["law_weight"]),
            float(cfg["court_weight"])
        )

        if cfg["mode"] == MODE_HYBRID:
            ranked = prepend_exact_candidates(
                find_exact_candidates(
                    cand_law_searcher,
                    cand_court_searcher,
                    original_query
                ),
                ranked
            )

        for rank, item in enumerate(ranked[:output_k], start=1):
            candidate_rows.append({
                "query_id": row["query_id"],
                "rank": rank,
                "doc_id": item.get("doc_id", ""),
                "citation": item["citation"],
                "source_type": item.get("source_type", ""),
                "source_rank": item.get("source_rank", ""),
                "bm25_score": item.get("bm25_score", ""),
                "fusion_score": item.get("fusion_score", ""),
                "exact_match": item.get("exact_match", False),
            })

        if (i + 1) % 50 == 0 or i + 1 == len(split_queries):
            print(
                f"{i + 1:,}/{len(split_queries):,} queries | "
                f"{time.perf_counter() - start_all:.0f} s"
            )

    output_file = (
        RESULTS_DIR
        / f"bm25_{split}_candidates_top{output_k}.csv"
    )

    pd.DataFrame(candidate_rows).to_csv(output_file, index=False)

    print(f"\nCandidates saved: {output_file}")

    sys.exit(0)


val = load_val()

print(
    f"Validation queries: "
    f"{len(val):,}"
)

print(
    f"Law index: "
    f"{LAW_INDEX}"
)

print(
    f"Court index: "
    f"{COURT_INDEX}"
)


# ============================================================
# Open indexes
# ============================================================

law_searcher = LuceneSearcher(
    str(LAW_INDEX)
)

court_searcher = LuceneSearcher(
    str(COURT_INDEX)
)


# Indexes were built with -pretokenized,
# therefore query side must use WhitespaceAnalyzer.
law_searcher.set_analyzer(
    JWhiteSpaceAnalyzer()
)

court_searcher.set_analyzer(
    JWhiteSpaceAnalyzer()
)


print(
    f"Law index documents: "
    f"{law_searcher.num_docs:,}"
)

print(
    f"Court index documents: "
    f"{court_searcher.num_docs:,}"
)


# ============================================================
# Prepare raw/citation-aware queries once
# ============================================================

original_queries = (
    val["query"]
    .fillna("")
    .astype(str)
    .tolist()
)


query_texts = {
    MODE_RAW: [
        tokenize_neutral(
            query
        )
        for query
        in original_queries
    ],

    MODE_CITATION: [
        tokenize_neutral(
            build_citation_aware_query(
                query
            )
        )
        for query
        in original_queries
    ]
}


# ============================================================
# Parameter search
# ============================================================

grid_rows = []

retrieval_configuration_count = (
    len(K1_VALUES)
    * len(B_VALUES)
    * 2
)

retrieval_configuration_number = 0


try:

    for k1 in K1_VALUES:

        for b in B_VALUES:

            law_searcher.set_bm25(
                k1,
                b
            )

            court_searcher.set_bm25(
                k1,
                b
            )


            # Only TWO actual retrieval versions are needed.
            #
            # citation_hybrid reuses citation-aware retrieval
            # and then adds verified explicit citations.
            for retrieval_mode in [
                MODE_RAW,
                MODE_CITATION
            ]:

                retrieval_configuration_number += 1


                print(
                    "\n"
                    "============================================================"
                )

                print(
                    f"RETRIEVAL "
                    f"{retrieval_configuration_number}/"
                    f"{retrieval_configuration_count}"
                )

                print(
                    f"Mode: "
                    f"{retrieval_mode}"
                )

                print(
                    f"k1={k1}, b={b}"
                )

                print(
                    "============================================================"
                )


                per_query_law = []
                per_query_court = []
                per_query_exact = []
                query_latencies_ms = []


                # ------------------------------------------------
                # Retrieve law/court results ONCE per k1/b/mode.
                # Fusion weights reuse the same retrieval.
                # ------------------------------------------------

                for i, query in enumerate(
                    query_texts[
                        retrieval_mode
                    ]
                ):

                    query_id = (
                        val.iloc[i][
                            "query_id"
                        ]
                    )


                    start = (
                        time.perf_counter()
                    )


                    law_results = (
                        retrieve_unique(
                            law_searcher,
                            query,
                            SOURCE_K
                        )
                    )


                    court_results = (
                        retrieve_unique(
                            court_searcher,
                            query,
                            SOURCE_K
                        )
                    )


                    # Exact candidates are only relevant to
                    # citation-aware/hybrid processing.
                    if (
                        retrieval_mode
                        == MODE_CITATION
                    ):

                        exact_results = (
                            find_exact_candidates(
                                law_searcher,
                                court_searcher,
                                original_queries[i]
                            )
                        )

                    else:

                        exact_results = []


                    elapsed_ms = (
                        (
                            time.perf_counter()
                            - start
                        )
                        * 1000
                    )


                    query_latencies_ms.append(
                        elapsed_ms
                    )

                    per_query_law.append(
                        law_results
                    )

                    per_query_court.append(
                        court_results
                    )

                    per_query_exact.append(
                        exact_results
                    )


                    print(
                        f"{query_id} | "
                        f"law={len(law_results)} | "
                        f"court={len(court_results)} | "
                        f"exact={len(exact_results)} | "
                        f"{elapsed_ms:.1f} ms"
                    )


                avg_latency_ms = (
                    sum(
                        query_latencies_ms
                    )
                    / len(
                        query_latencies_ms
                    )
                )


                # ------------------------------------------------
                # Try different source weights WITHOUT
                # repeating retrieval.
                # ------------------------------------------------

                for (
                    law_weight,
                    court_weight
                ) in SOURCE_WEIGHT_CONFIGS:

                    base_fused_results = []


                    for i in range(
                        len(val)
                    ):

                        fused = weighted_rrf(
                            per_query_law[i],
                            per_query_court[i],
                            law_weight,
                            court_weight
                        )

                        base_fused_results.append(
                            fused
                        )


                    # --------------------------------------------
                    # RAW retrieval produces RAW mode only.
                    #
                    # Citation-aware retrieval produces:
                    #   1. citation_aware
                    #   2. citation_hybrid
                    # --------------------------------------------

                    if retrieval_mode == MODE_RAW:

                        modes_to_evaluate = {
                            MODE_RAW:
                                base_fused_results
                        }

                    else:

                        hybrid_results = []


                        for i in range(
                            len(val)
                        ):

                            hybrid = (
                                prepend_exact_candidates(
                                    per_query_exact[i],
                                    base_fused_results[i]
                                )
                            )

                            hybrid_results.append(
                                hybrid
                            )


                        modes_to_evaluate = {
                            MODE_CITATION:
                                base_fused_results,

                            MODE_HYBRID:
                                hybrid_results
                        }


                    # --------------------------------------------
                    # Top-10 / Top-20 / Top-50
                    # --------------------------------------------

                    for (
                        evaluation_mode,
                        ranked_results
                    ) in modes_to_evaluate.items():

                        for top_k in TOP_K_VALUES:

                            summary = (
                                evaluate_ranked_results(
                                    val,
                                    ranked_results,
                                    top_k
                                )
                            )


                            result_row = {
                                "mode":
                                    evaluation_mode,

                                "k1":
                                    k1,

                                "b":
                                    b,

                                "law_weight":
                                    law_weight,

                                "court_weight":
                                    court_weight,

                                "top_k":
                                    top_k,

                                "macro_precision":
                                    summary[
                                        "macro_precision"
                                    ],

                                "macro_recall":
                                    summary[
                                        "macro_recall"
                                    ],

                                "macro_f1":
                                    summary[
                                        "macro_f1"
                                    ],

                                "avg_latency_ms":
                                    avg_latency_ms
                            }


                            grid_rows.append(
                                result_row
                            )


                            print(
                                f"{evaluation_mode:16s} | "
                                f"L:C="
                                f"{law_weight}:"
                                f"{court_weight} | "
                                f"Top-{top_k:<2d} | "
                                f"P="
                                f"{summary['macro_precision']:.4f} | "
                                f"R="
                                f"{summary['macro_recall']:.4f} | "
                                f"F1="
                                f"{summary['macro_f1']:.4f}"
                            )


finally:

    # Do NOT close here yet if we need final best retrieval.
    pass


# ============================================================
# Grid summary
# ============================================================

grid_df = pd.DataFrame(
    grid_rows
)


grid_df = grid_df.sort_values(
    by=[
        "macro_f1",
        "macro_recall",
        "macro_precision"
    ],
    ascending=[
        False,
        False,
        False
    ]
).reset_index(
    drop=True
)


grid_df.to_csv(
    GRID_SUMMARY_FILE,
    index=False
)


# ============================================================
# Best configuration overall
# ============================================================

best = grid_df.iloc[0]


best_mode = str(
    best[
        "mode"
    ]
)

best_k1 = float(
    best[
        "k1"
    ]
)

best_b = float(
    best[
        "b"
    ]
)

best_law_weight = float(
    best[
        "law_weight"
    ]
)

best_court_weight = float(
    best[
        "court_weight"
    ]
)

best_top_k = int(
    best[
        "top_k"
    ]
)


# ============================================================
# Ablation summary
# ============================================================

ablation_rows = []


for mode in QUERY_MODES:

    mode_rows = grid_df[
        grid_df[
            "mode"
        ]
        == mode
    ]


    if len(mode_rows) == 0:
        continue


    mode_best = (
        mode_rows
        .sort_values(
            by=[
                "macro_f1",
                "macro_recall",
                "macro_precision"
            ],
            ascending=[
                False,
                False,
                False
            ]
        )
        .iloc[0]
    )


    ablation_rows.append({
        "mode":
            mode,

        "best_k1":
            mode_best[
                "k1"
            ],

        "best_b":
            mode_best[
                "b"
            ],

        "law_weight":
            mode_best[
                "law_weight"
            ],

        "court_weight":
            mode_best[
                "court_weight"
            ],

        "top_k":
            int(
                mode_best[
                    "top_k"
                ]
            ),

        "macro_precision":
            mode_best[
                "macro_precision"
            ],

        "macro_recall":
            mode_best[
                "macro_recall"
            ],

        "macro_f1":
            mode_best[
                "macro_f1"
            ],

        "avg_latency_ms":
            mode_best[
                "avg_latency_ms"
            ]
    })


ablation_df = pd.DataFrame(
    ablation_rows
)


ablation_df.to_csv(
    ABLATION_SUMMARY_FILE,
    index=False
)


# ============================================================
# Re-run ONLY the winning configuration
# ============================================================

print(
    "\n\n"
    "============================================================"
)

print(
    "RETRIEVING FINAL BEST CONFIGURATION"
)

print(
    "============================================================"
)


law_searcher.set_bm25(
    best_k1,
    best_b
)

court_searcher.set_bm25(
    best_k1,
    best_b
)


if best_mode == MODE_RAW:

    final_queries = (
        query_texts[
            MODE_RAW
        ]
    )

else:

    final_queries = (
        query_texts[
            MODE_CITATION
        ]
    )


final_ranked_results = []
final_query_latencies = []


for i, query in enumerate(
    final_queries
):

    query_id = (
        val.iloc[i][
            "query_id"
        ]
    )


    start = (
        time.perf_counter()
    )


    law_results = retrieve_unique(
        law_searcher,
        query,
        SOURCE_K
    )

    court_results = retrieve_unique(
        court_searcher,
        query,
        SOURCE_K
    )


    fused_results = weighted_rrf(
        law_results,
        court_results,
        best_law_weight,
        best_court_weight
    )


    if best_mode == MODE_HYBRID:

        exact_results = (
            find_exact_candidates(
                law_searcher,
                court_searcher,
                original_queries[i]
            )
        )

        ranked = (
            prepend_exact_candidates(
                exact_results,
                fused_results
            )
        )

    else:

        ranked = fused_results


    elapsed_ms = (
        (
            time.perf_counter()
            - start
        )
        * 1000
    )


    final_query_latencies.append(
        elapsed_ms
    )


    final_ranked_results.append(
        ranked
    )


    print(
        f"{query_id} | "
        f"final candidates="
        f"{len(ranked)} | "
        f"{elapsed_ms:.1f} ms"
    )


# ============================================================
# Save best prediction file
# ============================================================

write_prediction_file(
    val,
    final_ranked_results,
    best_top_k,
    BEST_PREDICTIONS_FILE
)


# ============================================================
# Save downstream Top-100 candidates
# ============================================================

candidate_rows = []


for i, row in val.iterrows():

    query_id = row[
        "query_id"
    ]


    candidates = (
        final_ranked_results[i]
        [
            :CANDIDATE_OUTPUT_K
        ]
    )


    for rank, item in enumerate(
        candidates,
        start=1
    ):

        candidate_rows.append({
            "query_id":
                query_id,

            "rank":
                rank,

            "doc_id":
                item.get(
                    "doc_id",
                    ""
                ),

            "citation":
                item[
                    "citation"
                ],

            "source_type":
                item.get(
                    "source_type",
                    ""
                ),

            "source_rank":
                item.get(
                    "source_rank",
                    ""
                ),

            "bm25_score":
                item.get(
                    "bm25_score",
                    ""
                ),

            "fusion_score":
                item.get(
                    "fusion_score",
                    ""
                ),

            "exact_match":
                item.get(
                    "exact_match",
                    False
                )
        })


pd.DataFrame(
    candidate_rows
).to_csv(
    BEST_CANDIDATES_FILE,
    index=False
)


# ============================================================
# Save best configuration JSON
# ============================================================

best_config = {
    "system":
        "Split Lucene BM25",

    "mode":
        best_mode,

    "k1":
        best_k1,

    "b":
        best_b,

    "law_weight":
        best_law_weight,

    "court_weight":
        best_court_weight,

    "rrf_k":
        RRF_K,

    "source_k":
        SOURCE_K,

    "prediction_top_k":
        best_top_k,

    "candidate_output_k":
        CANDIDATE_OUTPUT_K,

    "macro_precision":
        float(
            best[
                "macro_precision"
            ]
        ),

    "macro_recall":
        float(
            best[
                "macro_recall"
            ]
        ),

    "macro_f1":
        float(
            best[
                "macro_f1"
            ]
        ),

    "grid_avg_latency_ms":
        float(
            best[
                "avg_latency_ms"
            ]
        ),

    "final_avg_latency_ms":
        float(
            sum(
                final_query_latencies
            )
            / len(
                final_query_latencies
            )
        ),

    "indexing":
        {
            "law_index":
                "results/lucene_bm25_law",

            "court_index":
                "results/lucene_bm25_court",

            "analyzer":
                "WhitespaceAnalyzer",

            "tokenization":
                "language-neutral regex",

            "citation_weighting":
                "canonical citation repeated 5 times in document representation"
        },

    "query_processing":
        {
            "raw":
                "language-neutral tokenisation only",

            "citation_aware":
                "explicit citations extracted and repeated in query",

            "citation_hybrid":
                "citation-aware BM25 plus verified exact corpus citation candidates"
        }
}


with open(
    BEST_CONFIG_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        best_config,
        file,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# Clean temporary evaluation file
# ============================================================

if TEMP_PREDICTION_FILE.exists():

    TEMP_PREDICTION_FILE.unlink()


# ============================================================
# Close Lucene indexes
# ============================================================

law_searcher.close()

court_searcher.close()


# ============================================================
# Final terminal summary
# ============================================================

print(
    "\n\n"
    "============================================================"
)

print(
    "FINAL BM25 PARAMETER SEARCH COMPLETE"
)

print(
    "============================================================"
)


print(
    "\nBEST RESULT FOR EACH ABLATION MODE:\n"
)

print(
    ablation_df.to_string(
        index=False
    )
)


print(
    "\n"
    "============================================================"
)

print(
    "BEST OVERALL BM25"
)

print(
    "============================================================"
)


print(
    f"Mode: "
    f"{best_mode}"
)

print(
    f"k1: "
    f"{best_k1}"
)

print(
    f"b: "
    f"{best_b}"
)

print(
    f"Law weight: "
    f"{best_law_weight}"
)

print(
    f"Court weight: "
    f"{best_court_weight}"
)

print(
    f"Top-K: "
    f"{best_top_k}"
)

print(
    f"Macro Precision: "
    f"{best['macro_precision']:.4f}"
)

print(
    f"Macro Recall: "
    f"{best['macro_recall']:.4f}"
)

print(
    f"Macro F1: "
    f"{best['macro_f1']:.4f}"
)

print(
    f"Average latency: "
    f"{best['avg_latency_ms']:.2f} ms/query"
)


print(
    "\nFILES SAVED:"
)

print(
    f"Grid summary:"
    f"\n{GRID_SUMMARY_FILE}"
)

print(
    f"\nAblation summary:"
    f"\n{ABLATION_SUMMARY_FILE}"
)

print(
    f"\nBest predictions:"
    f"\n{BEST_PREDICTIONS_FILE}"
)

print(
    f"\nTop-100 candidates:"
    f"\n{BEST_CANDIDATES_FILE}"
)

print(
    f"\nBest configuration:"
    f"\n{BEST_CONFIG_FILE}"
)

print(
    "\n"
    "============================================================"
)

print(
    "BM25 MODULE COMPLETE"
)

print(
    "============================================================"
)