"""Frozen project scoring/citation rules; function bodies preserved exactly."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
CUTOFF_MODES=["topk","threshold","relative"]

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
