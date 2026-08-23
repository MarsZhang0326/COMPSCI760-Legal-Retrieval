import pandas as pd
import json
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

LAW_FILE = "laws_de.csv"
COURT_FILE = "court_considerations.csv"

GOLD_TRAIN_FILE = "processed/gold_train.json"
GOLD_VAL_FILE = "processed/gold_val.json"

OUTPUT_DIR = Path("processed")
CHUNK_SIZE = 100_000


# ============================================================
# Helper
# ============================================================

def flatten_gold(gold_dict):
    return {
        citation
        for citations in gold_dict.values()
        for citation in citations
    }


# ============================================================
# 1. Load gold citation sets
# ============================================================

with open(GOLD_TRAIN_FILE, "r", encoding="utf-8") as f:
    gold_train_dict = json.load(f)

with open(GOLD_VAL_FILE, "r", encoding="utf-8") as f:
    gold_val_dict = json.load(f)

train_gold = flatten_gold(gold_train_dict)
val_gold = flatten_gold(gold_val_dict)

print("===== GOLD LABELS =====")
print(f"Train unique gold citations: {len(train_gold):,}")
print(f"Validation unique gold citations: {len(val_gold):,}")


# ============================================================
# 2. Audit laws_de.csv
# ============================================================

print("\n===== PROCESSING LAW CORPUS =====")

laws = pd.read_csv(
    LAW_FILE,
    low_memory=False
)

required_law_columns = {"citation", "text"}

missing = required_law_columns - set(laws.columns)

if missing:
    raise ValueError(
        f"{LAW_FILE} missing columns: {missing}"
    )

law_citations = set(
    laws["citation"]
    .dropna()
    .astype(str)
)

train_found_law = train_gold & law_citations
val_found_law = val_gold & law_citations

print(f"Law rows: {len(laws):,}")
print(f"Unique law citations: {len(law_citations):,}")
print(
    f"Train gold found in laws: "
    f"{len(train_found_law):,}/{len(train_gold):,}"
)
print(
    f"Validation gold found in laws: "
    f"{len(val_found_law):,}/{len(val_gold):,}"
)


# ============================================================
# 3. Scan court corpus
# ============================================================

print("\n===== PROCESSING COURT CORPUS =====")

court_rows = 0

court_citations = set()

train_found_court = set()
val_found_court = set()

for chunk_number, chunk in enumerate(
    pd.read_csv(
        COURT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ),
    start=1
):

    print(f"Processing court chunk {chunk_number} ...")

    court_rows += len(chunk)

    citations = set(
        chunk["citation"]
        .dropna()
        .astype(str)
    )

    court_citations.update(citations)

    train_found_court.update(
        train_gold & citations
    )

    val_found_court.update(
        val_gold & citations
    )


print("\nCourt scan complete.")

print(f"Court rows: {court_rows:,}")
print(
    f"Unique court citations: "
    f"{len(court_citations):,}"
)

print(
    f"Train gold found in court: "
    f"{len(train_found_court):,}/{len(train_gold):,}"
)

print(
    f"Validation gold found in court: "
    f"{len(val_found_court):,}/{len(val_gold):,}"
)


# ============================================================
# 4. Combined corpus coverage
# ============================================================

train_found_all = (
    train_found_law |
    train_found_court
)

val_found_all = (
    val_found_law |
    val_found_court
)

train_missing = sorted(
    train_gold - train_found_all
)

val_missing = sorted(
    val_gold - val_found_all
)


train_coverage = (
    len(train_found_all) / len(train_gold)
    if train_gold else 0
)

val_coverage = (
    len(val_found_all) / len(val_gold)
    if val_gold else 0
)


# ============================================================
# 5. Corpus overlap
# ============================================================

citation_overlap = (
    law_citations &
    court_citations
)


# ============================================================
# 6. Save coverage report
# ============================================================

report = {

    "corpus": {
        "law_rows": int(len(laws)),
        "law_unique_citations":
            int(len(law_citations)),

        "court_rows":
            int(court_rows),

        "court_unique_citations":
            int(len(court_citations)),

        "law_court_citation_overlap":
            int(len(citation_overlap))
    },

    "train": {
        "unique_gold":
            int(len(train_gold)),

        "found_in_laws":
            int(len(train_found_law)),

        "found_in_court":
            int(len(train_found_court)),

        "found_in_full_corpus":
            int(len(train_found_all)),

        "coverage":
            float(train_coverage),

        "missing_count":
            int(len(train_missing)),

        "missing_citations":
            train_missing
    },

    "validation": {
        "unique_gold":
            int(len(val_gold)),

        "found_in_laws":
            int(len(val_found_law)),

        "found_in_court":
            int(len(val_found_court)),

        "found_in_full_corpus":
            int(len(val_found_all)),

        "coverage":
            float(val_coverage),

        "missing_count":
            int(len(val_missing)),

        "missing_citations":
            val_missing
    }
}


with open(
    OUTPUT_DIR / "corpus_coverage_report.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 7. Save corpus schema / handoff information
# ============================================================

schema = {
    "logical_schema": {
        "doc_id":
            "Unique row-level internal identifier",
        "citation":
            "Canonical Kaggle prediction ID",
        "text":
            "Text used for retrieval",
        "title":
            "Law title when available; empty for court rows",
        "source_type":
            "law or court"
    },

    "source_mapping": {
        "laws_de.csv": {
            "citation": "citation",
            "text": "text",
            "title": "title",
            "source_type": "law"
        },

        "court_considerations.csv": {
            "citation": "citation",
            "text": "text",
            "title": None,
            "source_type": "court"
        }
    },

    "important_rules": [
        "Do not drop rows solely because citation values repeat.",
        "Each corpus row remains a retrieval unit.",
        "Predicted citations must be deduplicated at citation level before evaluation.",
        "Do not modify canonical citation strings.",
        "BM25-specific tokenisation, stemming and stopword removal belong to the modelling stage."
    ]
}


with open(
    OUTPUT_DIR / "corpus_schema.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        schema,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 8. Final output
# ============================================================

print("\n===== FULL CORPUS AUDIT COMPLETE =====")

print(
    f"Train full-corpus coverage: "
    f"{len(train_found_all):,}/{len(train_gold):,} "
    f"({train_coverage:.2%})"
)

print(
    f"Validation full-corpus coverage: "
    f"{len(val_found_all):,}/{len(val_gold):,} "
    f"({val_coverage:.2%})"
)

print(
    f"Law/Court citation overlap: "
    f"{len(citation_overlap):,}"
)

print("\nGenerated files:")
print(
    " - processed/corpus_coverage_report.json"
)
print(
    " - processed/corpus_schema.json"
)