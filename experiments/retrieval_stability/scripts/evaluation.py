import pandas as pd
import json
from pathlib import Path


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parent

DEFAULT_GOLD_VAL = (
    ROOT / "processed" / "gold_val.json"
)


# ============================================================
# Citation parsing
# ============================================================

def parse_citations(value):
    """
    Convert a semicolon-separated citation string into a set.

    Kaggle scoring is citation-set based:
    - order does not matter
    - exact citation strings matter
    - duplicate predictions should not receive extra credit

    Only separator whitespace is removed.
    Citation contents themselves are NOT normalised.
    """

    if value is None:
        return set()

    if isinstance(value, float) and pd.isna(value):
        return set()

    value = str(value).strip()

    if value == "":
        return set()

    return {
        citation.strip()
        for citation in value.split(";")
        if citation.strip()
    }


# ============================================================
# Per-query metrics
# ============================================================

def score_query(predicted, gold):
    """
    Calculate citation-level Precision, Recall and F1
    for one query.

    predicted and gold must be sets of citation strings.
    """

    predicted = set(predicted)
    gold = set(gold)

    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)

    # Precision
    if len(predicted) == 0:
        precision = 0.0
    else:
        precision = tp / len(predicted)

    # Recall
    if len(gold) == 0:
        recall = 0.0
    else:
        recall = tp / len(gold)

    # F1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# Load gold labels
# ============================================================

def load_gold(path=DEFAULT_GOLD_VAL):
    """
    Load gold labels created by prepare_queries.py.
    """

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        data = json.load(f)

    return {
        query_id: set(citations)
        for query_id, citations in data.items()
    }


# ============================================================
# Evaluate prediction dictionary
# ============================================================

def evaluate_predictions(
    predictions,
    gold
):
    """
    Evaluate:
        predictions = {
            query_id: set/list of citations
        }

        gold = {
            query_id: set/list of citations
        }

    Returns:
        summary metrics
        per-query dataframe
    """

    gold_ids = set(gold.keys())
    prediction_ids = set(predictions.keys())

    missing_query_ids = (
        gold_ids - prediction_ids
    )

    extra_query_ids = (
        prediction_ids - gold_ids
    )

    if missing_query_ids:
        raise ValueError(
            "Missing predictions for query IDs: "
            + ", ".join(sorted(missing_query_ids))
        )

    if extra_query_ids:
        raise ValueError(
            "Predictions contain unexpected query IDs: "
            + ", ".join(sorted(extra_query_ids))
        )

    rows = []

    for query_id in sorted(gold.keys()):

        pred_set = set(
            predictions[query_id]
        )

        gold_set = set(
            gold[query_id]
        )

        metrics = score_query(
            pred_set,
            gold_set
        )

        rows.append({
            "query_id": query_id,

            "n_gold":
                len(gold_set),

            "n_predicted":
                len(pred_set),

            "tp":
                metrics["tp"],

            "fp":
                metrics["fp"],

            "fn":
                metrics["fn"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "f1":
                metrics["f1"],

            "gold_citations":
                ";".join(sorted(gold_set)),

            "predicted_citations":
                ";".join(sorted(pred_set))
        })

    results = pd.DataFrame(rows)

    summary = {
        "n_queries":
            len(results),

        "macro_precision":
            float(results["precision"].mean()),

        "macro_recall":
            float(results["recall"].mean()),

        "macro_f1":
            float(results["f1"].mean())
    }

    return summary, results


# ============================================================
# Evaluate Kaggle-style CSV file
# ============================================================

def evaluate_csv(
    prediction_file,
    gold_file=DEFAULT_GOLD_VAL
):
    """
    Prediction file format:

    query_id,predicted_citations
    val_0001,"Art. ...;BGE ..."
    val_0002,"..."
    """

    df = pd.read_csv(
        prediction_file,
        keep_default_na=False
    )

    required = {
        "query_id",
        "predicted_citations"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Prediction file missing columns: "
            f"{missing}"
        )

    if df["query_id"].duplicated().any():
        duplicates = (
            df.loc[
                df["query_id"].duplicated(),
                "query_id"
            ]
            .tolist()
        )

        raise ValueError(
            f"Duplicate query IDs: {duplicates}"
        )

    predictions = {
        str(row["query_id"]):
            parse_citations(
                row["predicted_citations"]
            )

        for _, row in df.iterrows()
    }

    gold = load_gold(gold_file)

    return evaluate_predictions(
        predictions,
        gold
    )


# ============================================================
# Print helper
# ============================================================

def print_summary(summary):

    print("\n===== EVALUATION RESULTS =====")

    print(
        f"Queries: "
        f"{summary['n_queries']}"
    )

    print(
        f"Macro Precision: "
        f"{summary['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall: "
        f"{summary['macro_recall']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{summary['macro_f1']:.4f}"
    )


# ============================================================
# Built-in sanity tests
# ============================================================

if __name__ == "__main__":

    gold = load_gold()

    # --------------------------------------------------------
    # TEST 1: Perfect predictions
    # Expected Macro F1 = 1.0000
    # --------------------------------------------------------

    perfect_predictions = {
        query_id: citations.copy()
        for query_id, citations in gold.items()
    }

    perfect_summary, _ = (
        evaluate_predictions(
            perfect_predictions,
            gold
        )
    )

    print(
        "===== SANITY TEST 1: "
        "PERFECT PREDICTIONS ====="
    )

    print_summary(
        perfect_summary
    )

    assert abs(
        perfect_summary["macro_f1"]
        - 1.0
    ) < 1e-12


    # --------------------------------------------------------
    # TEST 2: Empty predictions
    # Gold sets are non-empty,
    # so expected Macro F1 = 0.0000
    # --------------------------------------------------------

    empty_predictions = {
        query_id: set()
        for query_id in gold
    }

    empty_summary, _ = (
        evaluate_predictions(
            empty_predictions,
            gold
        )
    )

    print(
        "\n===== SANITY TEST 2: "
        "EMPTY PREDICTIONS ====="
    )

    print_summary(
        empty_summary
    )

    assert abs(
        empty_summary["macro_f1"]
        - 0.0
    ) < 1e-12


    print(
        "\n===== "
        "ALL EVALUATION TESTS PASSED "
        "====="
    )