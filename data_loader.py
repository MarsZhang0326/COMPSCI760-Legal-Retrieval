import pandas as pd
from pathlib import Path

# ============================================================
# Default paths
# ============================================================

ROOT = Path(__file__).resolve().parent

LAW_FILE = ROOT / "laws_de.csv"
COURT_FILE = ROOT / "court_considerations.csv"

TRAIN_FILE = ROOT / "processed" / "train_clean.csv"
VAL_FILE = ROOT / "processed" / "val_clean.csv"
TEST_FILE = ROOT / "processed" / "test_clean.csv"


# ============================================================
# Query loaders
# ============================================================

def load_train():
    """
    Load cleaned training queries.
    """
    return pd.read_csv(TRAIN_FILE)


def load_val():
    """
    Load cleaned validation queries.
    """
    return pd.read_csv(VAL_FILE)


def load_test():
    """
    Load cleaned test queries.
    """
    return pd.read_csv(TEST_FILE)


# ============================================================
# Law corpus
# ============================================================

def load_laws():
    """
    Load the complete federal-law corpus.

    Logical schema:
        doc_id
        citation
        text
        title
        source_type
    """

    df = pd.read_csv(
        LAW_FILE,
        low_memory=False
    )

    required = {"citation", "text"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"laws_de.csv missing columns: {missing}"
        )

    df = df.copy()

    # Unique row-level internal ID
    df.insert(
        0,
        "doc_id",
        [f"law_{i:07d}" for i in range(len(df))]
    )

    if "title" not in df.columns:
        df["title"] = ""

    df["title"] = df["title"].fillna("")

    df["source_type"] = "law"

    return df[
        [
            "doc_id",
            "citation",
            "text",
            "title",
            "source_type"
        ]
    ]


# ============================================================
# Court corpus
# ============================================================

def iter_court_chunks(chunksize=100_000):
    """
    Stream the court corpus in chunks.

    This avoids loading the full ~2.4 GB file into RAM.

    Important:
    repeated citation strings are preserved because
    different court passages may map to the same citation.
    """

    row_offset = 0

    for chunk in pd.read_csv(
        COURT_FILE,
        chunksize=chunksize,
        low_memory=False
    ):

        required = {"citation", "text"}

        missing = required - set(chunk.columns)

        if missing:
            raise ValueError(
                f"court_considerations.csv "
                f"missing columns: {missing}"
            )

        chunk = chunk.copy()

        n = len(chunk)

        chunk.insert(
            0,
            "doc_id",
            [
                f"court_{i:08d}"
                for i in range(
                    row_offset,
                    row_offset + n
                )
            ]
        )

        chunk["title"] = ""
        chunk["source_type"] = "court"

        row_offset += n

        yield chunk[
            [
                "doc_id",
                "citation",
                "text",
                "title",
                "source_type"
            ]
        ]


# ============================================================
# Unified corpus iterator
# ============================================================

def iter_full_corpus(chunksize=100_000):
    """
    Yield both corpora using exactly the same schema.

    First:
        all law rows

    Then:
        court rows chunk-by-chunk
    """

    laws = load_laws()

    yield laws

    for court_chunk in iter_court_chunks(
        chunksize=chunksize
    ):
        yield court_chunk


# ============================================================
# Retrieval text helper
# ============================================================

def build_retrieval_text(df, include_title=True):
    """
    Create retrieval text without modifying the canonical
    citation IDs.

    For laws:
        title + text

    For court:
        text

    BM25 modelling can later decide whether to use title.
    """

    df = df.copy()

    if include_title:

        title = (
            df["title"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        text = (
            df["text"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        df["retrieval_text"] = (
            title + " " + text
        ).str.strip()

    else:

        df["retrieval_text"] = (
            df["text"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return df


# ============================================================
# Quick sanity check
# ============================================================

if __name__ == "__main__":

    print("===== QUERY DATA =====")

    train = load_train()
    val = load_val()
    test = load_test()

    print(f"Train: {len(train):,}")
    print(f"Validation: {len(val):,}")
    print(f"Test: {len(test):,}")

    print("\n===== LAW CORPUS =====")

    laws = load_laws()

    print(f"Law rows: {len(laws):,}")
    print(laws.columns.tolist())
    print(laws.head(2))

    print("\n===== FIRST COURT CHUNK =====")

    first_chunk = next(
        iter_court_chunks(chunksize=100_000)
    )

    print(
        f"First court chunk: "
        f"{len(first_chunk):,}"
    )

    print(first_chunk.columns.tolist())
    print(first_chunk.head(2))

    print("\n===== DATA LOADER CHECK PASSED =====")