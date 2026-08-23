import pandas as pd
import json
from pathlib import Path

from evaluation import evaluate_csv, print_summary


ROOT = Path(__file__).resolve().parent

GOLD_FILE = ROOT / "processed" / "gold_val.json"
OUTPUT_FILE = ROOT / "processed" / "val_perfect_predictions.csv"


# ------------------------------------------------------------
# 1. Load gold validation citations
# ------------------------------------------------------------

with open(
    GOLD_FILE,
    "r",
    encoding="utf-8"
) as f:
    gold = json.load(f)


# ------------------------------------------------------------
# 2. Create a Kaggle-style prediction CSV
#    using perfect predictions
# ------------------------------------------------------------

rows = []

for query_id, citations in gold.items():

    rows.append({
        "query_id": query_id,
        "predicted_citations": ";".join(citations)
    })


predictions = pd.DataFrame(rows)

predictions.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ------------------------------------------------------------
# 3. Evaluate the actual CSV
# ------------------------------------------------------------

summary, details = evaluate_csv(
    OUTPUT_FILE
)

print(
    "===== CSV END-TO-END TEST ====="
)

print_summary(summary)


# ------------------------------------------------------------
# 4. Assertions
# ------------------------------------------------------------

assert summary["n_queries"] == 10

assert abs(
    summary["macro_precision"] - 1.0
) < 1e-12

assert abs(
    summary["macro_recall"] - 1.0
) < 1e-12

assert abs(
    summary["macro_f1"] - 1.0
) < 1e-12


print(
    "\n===== CSV EVALUATION TEST PASSED ====="
)

print(
    f"Generated: {OUTPUT_FILE}"
)