import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed"

EXPECTED_ROOT_FILES = [
    "train.csv",
    "val.csv",
    "test.csv",
    "laws_de.csv",
    "court_considerations.csv",
    "sample_submission.csv",
    "data_loader.py",
    "evaluation.py",
    "README_DATA.md",
]

EXPECTED_PROCESSED_FILES = [
    "train_clean.csv",
    "val_clean.csv",
    "test_clean.csv",
    "gold_train.json",
    "gold_val.json",
    "query_processing_summary.json",
    "corpus_coverage_report.json",
    "corpus_schema.json",
    "val_perfect_predictions.csv",
]


def clean_query(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_gold(value):
    if pd.isna(value):
        return []
    return [
        c.strip()
        for c in str(value).split(";")
        if c.strip()
    ]


def fail(message):
    raise AssertionError(message)


def check(condition, message):
    if not condition:
        fail(message)


print("===== FINAL DATA & EVALUATION PREFLIGHT =====\n")

# ------------------------------------------------------------------
# 1. Required files
# ------------------------------------------------------------------
print("[1/8] Checking project files ...")

missing_root = [f for f in EXPECTED_ROOT_FILES if not (ROOT / f).exists()]
missing_processed = [f for f in EXPECTED_PROCESSED_FILES if not (PROCESSED / f).exists()]

check(not missing_root, f"Missing root files: {missing_root}")
check(not missing_processed, f"Missing processed files: {missing_processed}")

court_size = (ROOT / "court_considerations.csv").stat().st_size
check(court_size > 2_000_000_000, "court_considerations.csv appears unexpectedly small.")

print("  PASS: required files are present.")


# ------------------------------------------------------------------
# 2. Raw query files
# ------------------------------------------------------------------
print("[2/8] Checking raw query files ...")

train_raw = pd.read_csv(ROOT / "train.csv", keep_default_na=False)
val_raw = pd.read_csv(ROOT / "val.csv", keep_default_na=False)
test_raw = pd.read_csv(ROOT / "test.csv", keep_default_na=False)

check(len(train_raw) == 1139, f"Unexpected train rows: {len(train_raw)}")
check(len(val_raw) == 10, f"Unexpected val rows: {len(val_raw)}")
check(len(test_raw) == 40, f"Unexpected test rows: {len(test_raw)}")

check(set(train_raw.columns) == {"query_id", "query", "gold_citations"},
      f"Unexpected train columns: {train_raw.columns.tolist()}")
check(set(val_raw.columns) == {"query_id", "query", "gold_citations"},
      f"Unexpected val columns: {val_raw.columns.tolist()}")
check(set(test_raw.columns) == {"query_id", "query"},
      f"Unexpected test columns: {test_raw.columns.tolist()}")

for name, df in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
    check(df["query_id"].is_unique, f"{name}: query_id is not unique")
    check((df["query"].astype(str).str.len() > 0).all(), f"{name}: empty query found")

check(set(train_raw["query_id"]).isdisjoint(set(val_raw["query_id"])), "train/val ID overlap")
check(set(train_raw["query_id"]).isdisjoint(set(test_raw["query_id"])), "train/test ID overlap")
check(set(val_raw["query_id"]).isdisjoint(set(test_raw["query_id"])), "val/test ID overlap")

print("  PASS: raw query schemas, counts, and IDs are correct.")


# ------------------------------------------------------------------
# 3. Processed query files reproduce the intended cleaning exactly
# ------------------------------------------------------------------
print("[3/8] Checking processed query files ...")

train_clean = pd.read_csv(PROCESSED / "train_clean.csv", keep_default_na=False)
val_clean = pd.read_csv(PROCESSED / "val_clean.csv", keep_default_na=False)
test_clean = pd.read_csv(PROCESSED / "test_clean.csv", keep_default_na=False)

for name, raw, clean, labelled in [
    ("train", train_raw, train_clean, True),
    ("val", val_raw, val_clean, True),
    ("test", test_raw, test_clean, False),
]:
    check(len(raw) == len(clean), f"{name}: row count changed during cleaning")
    check(raw["query_id"].astype(str).tolist() == clean["query_id"].astype(str).tolist(),
          f"{name}: query_id order/content changed")
    check(raw["query"].astype(str).tolist() == clean["query_raw"].astype(str).tolist(),
          f"{name}: query_raw does not preserve the original query")
    expected_clean = raw["query"].map(clean_query).tolist()
    check(expected_clean == clean["query"].astype(str).tolist(),
          f"{name}: cleaned query differs from the documented whitespace-only cleaning")

    if labelled:
        expected_counts = raw["gold_citations"].map(lambda x: len(parse_gold(x))).tolist()
        check(expected_counts == clean["gold_count"].astype(int).tolist(),
              f"{name}: gold_count mismatch")
        check(raw["gold_citations"].astype(str).tolist() == clean["gold_citations"].astype(str).tolist(),
              f"{name}: canonical gold citation strings were modified")

print("  PASS: processed queries preserve raw content and apply only documented whitespace cleaning.")


# ------------------------------------------------------------------
# 4. Gold JSON files exactly match raw labels
# ------------------------------------------------------------------
print("[4/8] Checking gold label JSON files ...")

with open(PROCESSED / "gold_train.json", "r", encoding="utf-8") as f:
    gold_train = json.load(f)
with open(PROCESSED / "gold_val.json", "r", encoding="utf-8") as f:
    gold_val = json.load(f)

expected_gold_train = {
    str(row.query_id): parse_gold(row.gold_citations)
    for row in train_raw.itertuples(index=False)
}
expected_gold_val = {
    str(row.query_id): parse_gold(row.gold_citations)
    for row in val_raw.itertuples(index=False)
}

check(gold_train == expected_gold_train, "gold_train.json does not exactly match raw train labels")
check(gold_val == expected_gold_val, "gold_val.json does not exactly match raw val labels")

train_unique_gold = len({c for values in gold_train.values() for c in values})
val_unique_gold = len({c for values in gold_val.values() for c in values})
train_occ = sum(len(v) for v in gold_train.values())
val_occ = sum(len(v) for v in gold_val.values())

check(train_unique_gold == 2695, f"Unexpected train unique gold: {train_unique_gold}")
check(val_unique_gold == 222, f"Unexpected val unique gold: {val_unique_gold}")
check(train_occ == 4659, f"Unexpected train gold occurrences: {train_occ}")
check(val_occ == 251, f"Unexpected val gold occurrences: {val_occ}")

print("  PASS: gold JSON files exactly reproduce canonical Kaggle labels.")


# ------------------------------------------------------------------
# 5. Law corpus and saved full-corpus audit
# ------------------------------------------------------------------
print("[5/8] Checking corpus integrity and coverage report ...")

laws = pd.read_csv(ROOT / "laws_de.csv", low_memory=False)

check(len(laws) == 175933, f"Unexpected laws rows: {len(laws)}")
check({"citation", "text", "title"}.issubset(laws.columns),
      f"Unexpected laws columns: {laws.columns.tolist()}")
check(laws["citation"].notna().all(), "laws: missing citation")
check(laws["text"].notna().all(), "laws: missing text")
check(laws["citation"].nunique() == 175933, "laws: citation IDs should be unique")

with open(PROCESSED / "corpus_coverage_report.json", "r", encoding="utf-8") as f:
    coverage = json.load(f)

check(coverage["corpus"]["law_rows"] == 175933, "coverage report: wrong law row count")
check(coverage["corpus"]["court_rows"] == 2476315, "coverage report: wrong court row count")
check(coverage["corpus"]["court_unique_citations"] == 1985178,
      "coverage report: wrong court unique citation count")
check(coverage["corpus"]["law_court_citation_overlap"] == 0,
      "coverage report: unexpected law/court citation overlap")

check(coverage["train"]["found_in_full_corpus"] == 1928,
      "coverage report: unexpected train found count")
check(coverage["train"]["unique_gold"] == 2695,
      "coverage report: unexpected train unique gold count")
check(abs(coverage["train"]["coverage"] - 0.7153988868274582) < 1e-12,
      "coverage report: unexpected train coverage")

check(coverage["validation"]["found_in_full_corpus"] == 222,
      "coverage report: validation must have 222/222 coverage")
check(coverage["validation"]["unique_gold"] == 222,
      "coverage report: unexpected validation unique gold count")
check(abs(coverage["validation"]["coverage"] - 1.0) < 1e-12,
      "coverage report: validation coverage must equal 1.0")
check(coverage["validation"]["missing_count"] == 0,
      "coverage report: validation should have no missing citations")

print("  PASS: law corpus and saved full-corpus coverage report are consistent.")


# ------------------------------------------------------------------
# 6. Shared data loader
# ------------------------------------------------------------------
print("[6/8] Checking shared data_loader.py ...")

sys.path.insert(0, str(ROOT))
from data_loader import load_train, load_val, load_test, load_laws, iter_court_chunks

dl_train = load_train()
dl_val = load_val()
dl_test = load_test()
dl_laws = load_laws()
first_court = next(iter_court_chunks(chunksize=1000))

check(len(dl_train) == 1139, "data_loader: wrong train row count")
check(len(dl_val) == 10, "data_loader: wrong val row count")
check(len(dl_test) == 40, "data_loader: wrong test row count")
check(len(dl_laws) == 175933, "data_loader: wrong law row count")
check(len(first_court) == 1000, "data_loader: court chunk size mismatch")

expected_schema = ["doc_id", "citation", "text", "title", "source_type"]
check(dl_laws.columns.tolist() == expected_schema, "data_loader: law schema mismatch")
check(first_court.columns.tolist() == expected_schema, "data_loader: court schema mismatch")
check(dl_laws["source_type"].eq("law").all(), "data_loader: bad law source_type")
check(first_court["source_type"].eq("court").all(), "data_loader: bad court source_type")

print("  PASS: shared data loader works for queries, law corpus, and streamed court corpus.")


# ------------------------------------------------------------------
# 7. Shared evaluation
# ------------------------------------------------------------------
print("[7/8] Checking evaluation.py ...")

from evaluation import load_gold, evaluate_predictions, evaluate_csv

gold = load_gold()

perfect_predictions = {qid: citations.copy() for qid, citations in gold.items()}
perfect_summary, _ = evaluate_predictions(perfect_predictions, gold)
check(perfect_summary["n_queries"] == 10, "evaluation: wrong query count")
check(abs(perfect_summary["macro_f1"] - 1.0) < 1e-12, "evaluation: perfect F1 should be 1")

empty_predictions = {qid: set() for qid in gold}
empty_summary, _ = evaluate_predictions(empty_predictions, gold)
check(abs(empty_summary["macro_f1"] - 0.0) < 1e-12, "evaluation: empty F1 should be 0")

csv_summary, _ = evaluate_csv(PROCESSED / "val_perfect_predictions.csv")
check(abs(csv_summary["macro_precision"] - 1.0) < 1e-12, "CSV evaluation: precision should be 1")
check(abs(csv_summary["macro_recall"] - 1.0) < 1e-12, "CSV evaluation: recall should be 1")
check(abs(csv_summary["macro_f1"] - 1.0) < 1e-12, "CSV evaluation: F1 should be 1")

print("  PASS: evaluation logic and CSV end-to-end path are correct.")


# ------------------------------------------------------------------
# 8. Submission-format sanity check
# ------------------------------------------------------------------
print("[8/8] Checking Kaggle submission schema ...")

sample = pd.read_csv(ROOT / "sample_submission.csv", keep_default_na=False)
check({"query_id", "predicted_citations"}.issubset(sample.columns),
      f"sample submission columns unexpected: {sample.columns.tolist()}")

# Important: sample file is only an example; final submission must cover all test IDs.
check(len(test_raw) == 40, "test must contain 40 queries")

print("  PASS: submission schema is understood; final submission must use all 40 test query IDs.")

print("\n==============================================")
print("FINAL PREFLIGHT PASSED")
print("Data & Evaluation package is ready for BM25.")
print("==============================================")
print("\nKey confirmed facts:")
print("  Train queries: 1,139")
print("  Validation queries: 10")
print("  Test queries: 40")
print("  Law rows: 175,933")
print("  Court rows: 2,476,315")
print("  Validation corpus coverage: 222/222 (100%)")
print("  Train corpus coverage: 1,928/2,695 (71.54%)")
print("  Shared loader: PASS")
print("  Shared evaluator: PASS")
