"""Frozen project scoring/citation rules; function bodies preserved exactly."""
from pathlib import Path
import re
import numpy as np
import pandas as pd
CUTOFF_MODES=["topk","threshold","relative"]

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
