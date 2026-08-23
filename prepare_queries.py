import pandas as pd
import json
import re
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

TRAIN_FILE = "train.csv"
VAL_FILE = "val.csv"
TEST_FILE = "test.csv"

OUTPUT_DIR = Path("processed")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def clean_query(text):
    """
    Conservative query cleaning:
    - preserve all words, punctuation and special characters
    - strip leading/trailing whitespace
    - replace line breaks/tabs/multiple spaces with one space
    """
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_gold(value):
    """
    Split Kaggle semicolon-separated citation labels.

    Important:
    citation strings are treated as canonical IDs.
    We do NOT lowercase, remove punctuation, or modify
    internal formatting.
    """
    if pd.isna(value):
        return []

    return [
        citation.strip()
        for citation in str(value).split(";")
        if citation.strip()
    ]


def process_labelled_file(path):
    df = pd.read_csv(path)

    required = {"query_id", "query", "gold_citations"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}"
        )

    out = pd.DataFrame()

    out["query_id"] = df["query_id"].astype(str)
    out["query_raw"] = df["query"].astype(str)
    out["query"] = df["query"].apply(clean_query)

    # Keep the original Kaggle label string exactly
    out["gold_citations"] = df["gold_citations"].astype(str)

    # Useful diagnostic field
    out["gold_count"] = df["gold_citations"].apply(
        lambda x: len(parse_gold(x))
    )

    return out


def process_test_file(path):
    df = pd.read_csv(path)

    required = {"query_id", "query"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing required columns: {missing}"
        )

    out = pd.DataFrame()

    out["query_id"] = df["query_id"].astype(str)
    out["query_raw"] = df["query"].astype(str)
    out["query"] = df["query"].apply(clean_query)

    return out


# ============================================================
# 1. Process files
# ============================================================

train = process_labelled_file(TRAIN_FILE)
val = process_labelled_file(VAL_FILE)
test = process_test_file(TEST_FILE)


# ============================================================
# 2. Basic integrity checks
# ============================================================

assert train["query_id"].is_unique
assert val["query_id"].is_unique
assert test["query_id"].is_unique

assert not train["query_id"].isin(val["query_id"]).any()
assert not train["query_id"].isin(test["query_id"]).any()
assert not val["query_id"].isin(test["query_id"]).any()

assert train["query"].str.len().gt(0).all()
assert val["query"].str.len().gt(0).all()
assert test["query"].str.len().gt(0).all()


# ============================================================
# 3. Build gold dictionaries
# ============================================================

gold_train = {
    row["query_id"]: parse_gold(row["gold_citations"])
    for _, row in train.iterrows()
}

gold_val = {
    row["query_id"]: parse_gold(row["gold_citations"])
    for _, row in val.iterrows()
}


# ============================================================
# 4. Save clean CSV files
# ============================================================

train.to_csv(
    OUTPUT_DIR / "train_clean.csv",
    index=False,
    encoding="utf-8"
)

val.to_csv(
    OUTPUT_DIR / "val_clean.csv",
    index=False,
    encoding="utf-8"
)

test.to_csv(
    OUTPUT_DIR / "test_clean.csv",
    index=False,
    encoding="utf-8"
)


# ============================================================
# 5. Save structured gold labels
# ============================================================

with open(
    OUTPUT_DIR / "gold_train.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        gold_train,
        f,
        ensure_ascii=False,
        indent=2
    )

with open(
    OUTPUT_DIR / "gold_val.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        gold_val,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 6. Save summary
# ============================================================

summary = {
    "train_queries": len(train),
    "validation_queries": len(val),
    "test_queries": len(test),

    "train_unique_gold_citations":
        len(set(c for values in gold_train.values() for c in values)),

    "validation_unique_gold_citations":
        len(set(c for values in gold_val.values() for c in values)),

    "train_gold_occurrences":
        sum(len(v) for v in gold_train.values()),

    "validation_gold_occurrences":
        sum(len(v) for v in gold_val.values()),

    "train_mean_gold_per_query":
        float(train["gold_count"].mean()),

    "train_median_gold_per_query":
        float(train["gold_count"].median()),

    "validation_mean_gold_per_query":
        float(val["gold_count"].mean()),

    "validation_median_gold_per_query":
        float(val["gold_count"].median())
}

with open(
    OUTPUT_DIR / "query_processing_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 7. Print results
# ============================================================

print("\n===== QUERY PROCESSING COMPLETE =====")

print(f"Train queries: {len(train):,}")
print(f"Validation queries: {len(val):,}")
print(f"Test queries: {len(test):,}")

print(
    f"Train unique gold citations: "
    f"{summary['train_unique_gold_citations']:,}"
)

print(
    f"Validation unique gold citations: "
    f"{summary['validation_unique_gold_citations']:,}"
)

print(
    f"Train gold occurrences: "
    f"{summary['train_gold_occurrences']:,}"
)

print(
    f"Validation gold occurrences: "
    f"{summary['validation_gold_occurrences']:,}"
)

print(
    f"Train mean gold/query: "
    f"{summary['train_mean_gold_per_query']:.2f}"
)

print(
    f"Validation mean gold/query: "
    f"{summary['validation_mean_gold_per_query']:.2f}"
)

print("\nGenerated files:")
print(" - processed/train_clean.csv")
print(" - processed/val_clean.csv")
print(" - processed/test_clean.csv")
print(" - processed/gold_train.json")
print(" - processed/gold_val.json")
print(" - processed/query_processing_summary.json")