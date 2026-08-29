import os
import re
import base64
import shutil
import time
from pathlib import Path

os.environ.setdefault(
    "OPENAI_API_KEY",
    "PYSERINI_BM25_NOT_USED"
)

java_exe = shutil.which("java")
if java_exe:
    os.environ["JAVA_HOME"] = str(
        Path(java_exe).resolve().parent.parent
    )

from pyserini.index.lucene import LuceneIndexer

from data_loader import (
    load_laws,
    iter_court_chunks
)


ROOT = Path(__file__).resolve().parent

LAW_INDEX = (
    ROOT / "results" / "lucene_bm25_law"
)

COURT_INDEX = (
    ROOT / "results" / "lucene_bm25_court"
)

COURT_CHUNK_SIZE = 50_000

# Give the document's OWN citation extra weight.
# This is fixed for every document and does not use gold labels.
CITATION_REPEAT = 5


def encode_citation(citation):
    return base64.urlsafe_b64encode(
        str(citation).encode("utf-8")
    ).decode("ascii")


def tokenize_neutral(text):
    """
    Language-neutral lexical tokenisation.

    Keeps legal identifiers such as:
      StPO
      BGE
      1B_210/2023
      Art. 221

    No English stemming is applied.
    """

    text = str(text).lower()

    tokens = re.findall(
        r"\w+(?:[./_-]\w+)*",
        text,
        flags=re.UNICODE
    )

    return " ".join(tokens)


def build_contents(citation, title, text):

    citation = str(citation or "").strip()
    title = str(title or "").strip()
    text = str(text or "").strip()

    # Repeat canonical citation so the document's own ID
    # is stronger than incidental citations mentioned in text.
    citation_part = " ".join(
        [citation] * CITATION_REPEAT
    )

    combined = (
        citation_part
        + " "
        + title
        + " "
        + text
    )

    return tokenize_neutral(combined)


def build_doc(row):

    citation = str(row.citation).strip()

    lucene_id = (
        f"{row.doc_id}|"
        f"{encode_citation(citation)}"
    )

    contents = build_contents(
        citation,
        row.title,
        row.text
    )

    return {
        "id": lucene_id,
        "contents": contents
    }


# ============================================================
# LAW INDEX
# ============================================================

print("\n====================================")
print("BUILD LAW BM25 INDEX")
print("====================================")

laws = load_laws()

law_indexer = LuceneIndexer(
    args=[
        "-index",
        str(LAW_INDEX),
        "-pretokenized"
    ],
    threads=2
)

law_docs = [
    build_doc(row)
    for row in laws.itertuples(index=False)
]

law_indexer.add_batch_dict(
    law_docs
)

law_indexer.close()

print(
    f"Law indexed: {len(law_docs):,}"
)


# ============================================================
# COURT INDEX
# ============================================================

print("\n====================================")
print("BUILD COURT BM25 INDEX")
print("====================================")

court_indexer = LuceneIndexer(
    args=[
        "-index",
        str(COURT_INDEX),
        "-pretokenized"
    ],
    threads=2
)

total = 0
start = time.perf_counter()

for chunk_no, chunk in enumerate(
    iter_court_chunks(
        chunksize=COURT_CHUNK_SIZE
    ),
    start=1
):

    docs = [
        build_doc(row)
        for row in chunk.itertuples(index=False)
    ]

    court_indexer.add_batch_dict(
        docs
    )

    total += len(docs)

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"Chunk {chunk_no:02d} | "
        f"Court indexed: {total:,} | "
        f"{elapsed / 60:.1f} min"
    )


court_indexer.close()

print("\n====================================")
print("SPLIT BM25 INDEX BUILD COMPLETE")
print("====================================")

print(
    f"Law rows: {len(laws):,}"
)

print(
    f"Court rows: {total:,}"
)