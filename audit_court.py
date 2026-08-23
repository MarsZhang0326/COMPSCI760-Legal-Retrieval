import pandas as pd
import json
from pathlib import Path

COURT_FILE = "court_considerations.csv"
VAL_FILE = "val.csv"

CHUNK_SIZE = 100_000

# -----------------------------
# 1. Load validation gold citations
# -----------------------------
val = pd.read_csv(VAL_FILE)

val_gold = set()

for value in val["gold_citations"].dropna():
    for citation in str(value).split(";"):
        citation = citation.strip()
        if citation:
            val_gold.add(citation)

print(f"Validation unique gold citations: {len(val_gold)}")


# -----------------------------
# 2. Scan court corpus in chunks
# -----------------------------
total_rows = 0
missing_citation = 0
missing_text = 0

seen_citations = set()
duplicate_citations = 0

matched_val_gold = set()

sample_chunks = []

for i, chunk in enumerate(
    pd.read_csv(
        COURT_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False
    )
):
    print(f"Processing chunk {i + 1} ...")

    total_rows += len(chunk)

    # Basic missing values
    if "citation" in chunk.columns:
        missing_citation += chunk["citation"].isna().sum()

    if "text" in chunk.columns:
        missing_text += chunk["text"].isna().sum()

    # Citation audit
    citations = chunk["citation"].dropna().astype(str)

    for citation in citations:
        if citation in seen_citations:
            duplicate_citations += 1
        else:
            seen_citations.add(citation)

        if citation in val_gold:
            matched_val_gold.add(citation)

    # Save only a small sample
    if len(sample_chunks) < 1:
        sample_chunks.append(chunk.head(1000))


# -----------------------------
# 3. Coverage results
# -----------------------------
missing_val_gold = sorted(val_gold - matched_val_gold)

summary = {
    "court_file": COURT_FILE,
    "columns": list(chunk.columns),
    "total_rows": int(total_rows),
    "unique_citations": int(len(seen_citations)),
    "duplicate_citation_rows": int(duplicate_citations),
    "missing_citation_values": int(missing_citation),
    "missing_text_values": int(missing_text),
    "validation_unique_gold_citations": int(len(val_gold)),
    "validation_gold_found_in_court": int(len(matched_val_gold)),
    "validation_gold_not_found_in_court": int(len(missing_val_gold)),
    "matched_validation_citations": sorted(matched_val_gold),
    "missing_validation_citations": missing_val_gold
}


# -----------------------------
# 4. Save results
# -----------------------------
with open(
    "court_audit_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

if sample_chunks:
    pd.concat(sample_chunks).to_csv(
        "court_sample.csv",
        index=False,
        encoding="utf-8-sig"
    )

print("\n===== AUDIT COMPLETE =====")
print(f"Rows: {total_rows:,}")
print(f"Unique citations: {len(seen_citations):,}")
print(f"Duplicate citation rows: {duplicate_citations:,}")
print(f"Missing citation values: {missing_citation:,}")
print(f"Missing text values: {missing_text:,}")
print(
    f"Validation gold found in court: "
    f"{len(matched_val_gold)}/{len(val_gold)}"
)

print("\nGenerated:")
print(" - court_audit_summary.json")
print(" - court_sample.csv")