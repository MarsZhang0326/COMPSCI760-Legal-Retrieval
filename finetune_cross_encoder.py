"""
Fine-tune a multilingual cross-encoder on the training queries.

Training data
-------------
Each training example is a group: 1 positive passage + N negatives.
The loss is a softmax cross-entropy over the group, so the model
learns to rank the positive above its negatives.

Positives
    Corpus rows whose citation is a gold citation of the query.
    Gold citations that are not in the corpus (about 28% of
    unique train gold citations) are skipped.

Negatives (three kinds, per group)
    hard       BM25 and dense candidates for the query that are not
               gold. Needs train candidate files:
                 python bm25_run_experiments.py --candidates-split train
                 python dense_run_experiments.py --split train
    neighbour  corpus rows next to a positive row: other paragraphs
               and articles of the same act, or other considerations
               of the same court decision. These teach the model the
               exact-citation distinction the task scores on.
    random     random law and court rows. They keep the model
               calibrated on clearly unrelated text.

    If a pool is short for a query, the other pools fill the gap.
    Without train candidate files, the script uses neighbour and
    random negatives only, and prints a warning.

Validation
----------
A share of the training queries (--dev-fraction) is held out.
The script reports the positive's MRR and accuracy@1 inside the
held-out groups before training and after each epoch, and keeps
the best epoch. The 10 official validation queries are not used
here, so rerank_cross_encoder.py can still evaluate on them fairly.

Output
------
results/ce_finetuned/          Hugging Face model + tokenizer
results/ce_finetuned/finetune_config.json

Use the model with:
    python rerank_cross_encoder.py --model results/ce_finetuned
"""

import argparse
import contextlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import iter_court_chunks, load_laws, load_train
from evaluation import load_gold
from rerank_cross_encoder import (
    DEFAULT_MODEL,
    build_passage_text,
    read_candidate_file,
)


try:
    import torch
    import torch.nn.functional as F
    from sentence_transformers import CrossEncoder
    from transformers import get_linear_schedule_with_warmup
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
GOLD_TRAIN_FILE = ROOT / "processed" / "gold_train.json"

DEFAULT_OUTPUT_DIR = RESULTS_DIR / "ce_finetuned"
DEFAULT_HARD_NEGATIVE_FILES = [
    RESULTS_DIR / "bm25_train_candidates_top100.csv",
    RESULTS_DIR / "dense_full_train_candidates_top100.csv",
]

MAX_ROWS_PER_GOLD_CITATION = 2


# ============================================================
# Arguments
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fine-tune a cross-encoder on the training queries."
    )

    parser.add_argument("--base-model", default=DEFAULT_MODEL,
                        help="Model to start from.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Where to save the fine-tuned model.")

    parser.add_argument("--hard-negative-files", type=Path, nargs="*", default=None,
                        help=("Train candidate CSVs (BM25 / dense format). "
                              "Default: the standard train candidate files that exist."))
    parser.add_argument("--hard-negative-depth", type=int, default=50,
                        help="Use candidates up to this rank as hard negatives.")

    parser.add_argument("--hard-negatives", type=int, default=4,
                        help="Hard negatives per group.")
    parser.add_argument("--neighbour-negatives", type=int, default=2,
                        help="Neighbour negatives per group.")
    parser.add_argument("--random-negatives", type=int, default=1,
                        help="Random negatives per group.")
    parser.add_argument("--neighbour-window", type=int, default=3,
                        help="Rows before and after a positive row to use as neighbours.")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--groups-per-batch", type=int, default=2,
                        help="Groups per forward pass (pairs = groups x (1 + negatives)).")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Forward passes per optimizer step.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Save GPU memory at some speed cost.")

    parser.add_argument("--dev-fraction", type=float, default=0.1,
                        help="Share of training queries held out for the dev check.")
    parser.add_argument("--max-train-queries", type=int, default=None,
                        help="Smoke-test limit on training queries.")
    parser.add_argument("--random-pool-size", type=int, default=5000,
                        help="Size of the random-negative pool (half law, half court).")

    parser.add_argument("--device", default=None, help="cpu, cuda, or mps.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=20,
                        help="Print progress every N optimizer steps.")

    return parser.parse_args(argv)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)


# ============================================================
# Build training data
# ============================================================

def act_name(title):
    """Law title up to the section part: identifies the act."""
    return str(title or "").split(" - ")[0].strip()


def load_hard_negatives(args, gold, query_ids):
    """
    Return {query_id: [doc_id, ...]} of non-gold candidates,
    best rank first, and the set of candidate doc_ids.
    """

    if args.hard_negative_files is None:
        files = [p for p in DEFAULT_HARD_NEGATIVE_FILES if p.exists()]
    else:
        files = list(args.hard_negative_files)

        for path in files:
            if not path.exists():
                raise FileNotFoundError(f"Missing hard-negative file: {path}")

    if not files:
        print(
            "WARNING: no train candidate files found. Training uses "
            "neighbour and random negatives only.\n"
            "  For BM25/dense hard negatives, run:\n"
            "    python bm25_run_experiments.py --candidates-split train\n"
            "    python dense_run_experiments.py --split train"
        )
        return {}, files

    frames = [
        read_candidate_file(path, path.stem, args.hard_negative_depth)
        for path in files
    ]

    df = pd.concat(frames, ignore_index=True)
    df = df[df["query_id"].isin(query_ids)]
    df = df[[
        citation not in gold.get(query_id, set())
        for query_id, citation in zip(df["query_id"], df["citation"])
    ]]

    df = (
        df.sort_values(["query_id", "rank"])
        .drop_duplicates(["query_id", "doc_id"])
    )

    hard = {
        query_id: group["doc_id"].tolist()
        for query_id, group in df.groupby("query_id")
    }

    print(
        f"Hard negatives: {len(df):,} from {len(files)} file(s), "
        f"{len(hard):,} queries covered"
    )

    return hard, files


def build_training_data(args):
    rng = random.Random(args.seed)

    train = load_train()
    train["query_id"] = train["query_id"].astype(str)
    train["query"] = train["query"].fillna("").astype(str)

    gold = {q: set(c) for q, c in load_gold(GOLD_TRAIN_FILE).items()}

    if args.max_train_queries is not None:
        train = train.sample(
            n=min(args.max_train_queries, len(train)),
            random_state=args.seed,
        )

    query_ids = set(train["query_id"])
    all_gold = set().union(*(gold.get(q, set()) for q in query_ids))

    hard, hard_files = load_hard_negatives(args, gold, query_ids)
    hard_ids = {d for ids in hard.values() for d in ids}

    passages = {}
    citation_of = {}

    # ---------------- laws ----------------

    laws = load_laws().reset_index(drop=True)
    law_citations = laws["citation"].astype(str).str.strip().tolist()
    law_acts = [act_name(t) for t in laws["title"]]
    law_pos = {c: i for i, c in enumerate(law_citations)}

    def add_law(i):
        row = laws.iloc[i]
        doc_id = row["doc_id"]

        if doc_id not in passages:
            passages[doc_id] = build_passage_text(
                row["citation"], row["title"], row["text"]
            )
            citation_of[doc_id] = law_citations[i]

        return doc_id

    gold_rows = {}          # citation -> [doc_id, ...]
    neighbours_of = {}      # positive doc_id -> [neighbour doc_id, ...]

    for citation in all_gold:
        i = law_pos.get(citation)

        if i is None:
            continue

        doc_id = add_law(i)
        gold_rows[citation] = [doc_id]

        lo = max(0, i - args.neighbour_window)
        hi = min(len(laws), i + args.neighbour_window + 1)

        neighbours_of[doc_id] = [
            add_law(j)
            for j in range(lo, hi)
            if j != i and law_acts[j] == law_acts[i]
        ]

    for doc_id in hard_ids:
        if doc_id.startswith("law_"):
            add_law(int(doc_id.split("_")[1]))

    random_law = [
        add_law(i)
        for i in rng.sample(range(len(laws)), min(args.random_pool_size // 2, len(laws)))
    ]

    del laws

    # ---------------- court (one streamed pass) ----------------

    needed_court = {d for d in hard_ids if d.startswith("court_")}
    random_court_candidates = []
    court_rows_seen = 0
    start = time.perf_counter()

    for chunk in iter_court_chunks():
        chunk = chunk.reset_index(drop=True)
        citations = chunk["citation"].astype(str).str.strip().tolist()
        doc_ids = chunk["doc_id"].tolist()

        def add_court(j):
            doc_id = doc_ids[j]

            if doc_id not in passages:
                passages[doc_id] = build_passage_text(
                    citations[j], "", chunk.at[j, "text"]
                )
                citation_of[doc_id] = citations[j]

            return doc_id

        for j, citation in enumerate(citations):
            if citation in all_gold:
                rows = gold_rows.setdefault(citation, [])

                if len(rows) >= MAX_ROWS_PER_GOLD_CITATION:
                    continue

                doc_id = add_court(j)
                rows.append(doc_id)

                lo = max(0, j - args.neighbour_window)
                hi = min(len(chunk), j + args.neighbour_window + 1)

                neighbours_of[doc_id] = [
                    add_court(k) for k in range(lo, hi) if k != j
                ]

            elif doc_ids[j] in needed_court:
                add_court(j)

        per_chunk = max(1, args.random_pool_size // 2 // 25)
        for j in rng.sample(range(len(chunk)), min(per_chunk, len(chunk))):
            random_court_candidates.append(add_court(j))

        court_rows_seen += len(chunk)

    print(
        f"Court scan: {court_rows_seen:,} rows in "
        f"{time.perf_counter() - start:.0f} s"
    )

    random_pool = random_law + random_court_candidates[: args.random_pool_size // 2]

    missing_hard = needed_court - set(passages)

    if missing_hard:
        print(f"WARNING: {len(missing_hard):,} hard-negative court rows not found")

    # ---------------- per-query records ----------------

    records = []
    n_gold_total = 0
    n_gold_found = 0

    for row in train.itertuples(index=False):
        query_gold = gold.get(row.query_id, set())
        n_gold_total += len(query_gold)

        positives = []

        for citation in sorted(query_gold):
            rows = gold_rows.get(citation, [])

            if rows:
                n_gold_found += 1
                positives += rows

        if not positives:
            continue

        def not_gold(doc_id):
            return citation_of.get(doc_id) not in query_gold

        neighbour_ids = []

        for doc_id in positives:
            neighbour_ids += neighbours_of.get(doc_id, [])

        records.append({
            "query_id": row.query_id,
            "query": row.query,
            "gold": query_gold,
            "positives": positives,
            "hard": [d for d in hard.get(row.query_id, []) if d in passages and not_gold(d)],
            "neighbour": list(dict.fromkeys(d for d in neighbour_ids if not_gold(d))),
        })

    print(
        f"Training queries with a positive in the corpus: "
        f"{len(records):,} of {len(train):,}"
    )
    print(
        f"Gold citations found in corpus: {n_gold_found:,} of {n_gold_total:,} "
        f"({n_gold_found / max(n_gold_total, 1):.1%})"
    )

    return records, passages, citation_of, random_pool, hard_files


# ============================================================
# Groups
# ============================================================

def sample_negatives(record, random_pool, citation_of, args, rng):
    wanted = {
        "hard": args.hard_negatives,
        "neighbour": args.neighbour_negatives,
        "random": args.random_negatives,
    }

    total = sum(wanted.values())
    chosen = []
    used = set()

    def take(pool, n):
        options = [d for d in pool if d not in used]
        picked = rng.sample(options, min(n, len(options)))
        used.update(picked)
        chosen.extend(picked)

    random_ok = [
        d for d in rng.sample(random_pool, min(len(random_pool), 200))
        if citation_of.get(d) not in record["gold"]
    ]

    take(record["hard"], wanted["hard"])
    take(record["neighbour"], wanted["neighbour"])
    take(random_ok, wanted["random"])

    # Fill any shortfall: hard first, then neighbours, then random
    for pool in [record["hard"], record["neighbour"], random_ok]:
        if len(chosen) < total:
            take(pool, total - len(chosen))

    return chosen


def build_groups(records, random_pool, citation_of, args, rng):
    groups = []

    for record in records:
        for positive in record["positives"]:
            negatives = sample_negatives(
                record, random_pool, citation_of, args, rng
            )

            groups.append({
                "query_id": record["query_id"],
                "query": record["query"],
                "doc_ids": [positive] + negatives,
            })

    return groups


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ============================================================
# Model
# ============================================================

def group_logits(hf_model, tokenizer, groups, passages, max_length, device, autocast):
    pairs_q = []
    pairs_p = []

    for group in groups:
        for doc_id in group["doc_ids"]:
            pairs_q.append(group["query"])
            pairs_p.append(passages[doc_id])

    sizes = {len(g["doc_ids"]) for g in groups}
    assert len(sizes) == 1, f"Groups have different sizes: {sizes}"

    encoded = tokenizer(
        pairs_q,
        pairs_p,
        padding=True,
        truncation="longest_first",
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with autocast():
        logits = hf_model(**encoded).logits

    return logits.float().view(len(groups), -1)


def evaluate_groups(hf_model, tokenizer, groups, passages, args, device, autocast):
    """MRR and accuracy@1 of the positive (index 0) within each group."""

    if not groups:
        return {"mrr": float("nan"), "acc_at_1": float("nan"), "n_groups": 0}

    hf_model.eval()
    reciprocal_ranks = []

    with torch.no_grad():
        for batch in batches(groups, max(1, args.groups_per_batch * 2)):
            logits = group_logits(
                hf_model, tokenizer, batch, passages,
                args.max_length, device, autocast,
            )

            # Ties count against the positive, so a model that gives
            # every passage the same score does not look good.
            positive = logits[:, :1]
            rank = 1 + (~(logits[:, 1:] < positive)).sum(dim=1)
            reciprocal_ranks += (1.0 / rank.float()).tolist()

    hf_model.train()
    rr = np.array(reciprocal_ranks)

    return {
        "mrr": float(rr.mean()),
        "acc_at_1": float((rr == 1.0).mean()),
        "n_groups": len(groups),
    }


def make_autocast(device):
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return (lambda: torch.autocast("cuda", dtype=dtype)), dtype

    return contextlib.nullcontext, None


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if torch is None:
        raise SystemExit(
            "Missing dependencies: torch, sentence-transformers, transformers\n"
            "Install them with:\n"
            "  python -m pip install -r requirements.txt"
        )

    set_seed(args.seed)
    rng = random.Random(args.seed)

    print("============================================================")
    print("CROSS-ENCODER FINE-TUNING")
    print("============================================================")
    print(f"Base model: {args.base_model}")

    records, passages, citation_of, random_pool, hard_files = build_training_data(args)

    if not records:
        raise SystemExit("No training queries with a positive in the corpus.")

    # Split by query, so no query appears in both train and dev
    rng.shuffle(records)
    n_dev = int(round(len(records) * args.dev_fraction))
    dev_records = records[:n_dev]
    train_records = records[n_dev:]

    dev_groups = build_groups(
        dev_records, random_pool, citation_of, args, random.Random(args.seed + 1)
    )

    group_size = 1 + args.hard_negatives + args.neighbour_negatives + args.random_negatives
    n_train_groups = sum(len(r["positives"]) for r in train_records)
    steps_per_epoch = math.ceil(n_train_groups / args.groups_per_batch / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs

    print(f"Train queries: {len(train_records):,} | dev queries: {len(dev_records):,}")
    print(
        f"Train groups: {n_train_groups:,} x {group_size} passages | "
        f"dev groups: {len(dev_groups):,}"
    )
    print(f"Optimizer steps: {total_steps:,} ({steps_per_epoch:,} per epoch)")

    model = CrossEncoder(
        args.base_model,
        max_length=args.max_length,
        device=args.device,
    )

    hf_model = model.model
    tokenizer = model.tokenizer
    device = next(hf_model.parameters()).device

    if hf_model.config.num_labels != 1:
        raise SystemExit(
            f"{args.base_model} has {hf_model.config.num_labels} output labels. "
            "Use a re-ranker model with a single relevance score."
        )

    if args.gradient_checkpointing:
        hf_model.gradient_checkpointing_enable()

    autocast, amp_dtype = make_autocast(device)
    scaler = None

    if amp_dtype == torch.float16:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda")
        else:
            scaler = torch.cuda.amp.GradScaler()

    print(f"Device: {device} | mixed precision: {amp_dtype}")

    optimizer = torch.optim.AdamW(
        hf_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=max(total_steps, 1),
    )

    history = []
    before = evaluate_groups(hf_model, tokenizer, dev_groups, passages, args, device, autocast)
    history.append({"epoch": 0, **before})
    print(f"\nDev before training: MRR={before['mrr']:.4f} | acc@1={before['acc_at_1']:.4f}")

    best_mrr = before["mrr"]
    saved_epoch = None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hf_model.train()
    step = 0
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        # New negatives every epoch
        train_groups = build_groups(
            train_records, random_pool, citation_of, args,
            random.Random(args.seed + 100 + epoch),
        )
        rng.shuffle(train_groups)

        running_loss = 0.0
        running_count = 0
        optimizer.zero_grad()

        batch_list = list(batches(train_groups, args.groups_per_batch))

        for b, batch in enumerate(batch_list, start=1):
            logits = group_logits(
                hf_model, tokenizer, batch, passages,
                args.max_length, device, autocast,
            )

            target = torch.zeros(len(batch), dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, target) / args.grad_accum

            if not torch.isfinite(loss):
                ids = [g["query_id"] for g in batch]
                print(f"WARNING: non-finite loss at batch {b}, queries {ids}. Skipped.")
                optimizer.zero_grad()
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += loss.item() * args.grad_accum
            running_count += 1

            if b % args.grad_accum == 0 or b == len(batch_list):
                if scaler is not None:
                    scaler.unscale_(optimizer)

                grad_norm = torch.nn.utils.clip_grad_norm_(hf_model.parameters(), 1.0)

                if scaler is None and not torch.isfinite(grad_norm):
                    print(f"WARNING: non-finite grad norm before step {step + 1}. Skipped.")
                    optimizer.zero_grad()
                    continue

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if device.type == "mps":
                    torch.mps.empty_cache()

                if step % args.log_every == 0 or step == total_steps:
                    elapsed = time.perf_counter() - start
                    eta = elapsed / step * (total_steps - step)
                    print(
                        f"epoch {epoch} | step {step:,}/{total_steps:,} | "
                        f"loss={running_loss / running_count:.4f} | "
                        f"{elapsed / 60:.1f} min elapsed | ~{eta / 60:.1f} min left"
                    )
                    running_loss = 0.0
                    running_count = 0

        result = evaluate_groups(hf_model, tokenizer, dev_groups, passages, args, device, autocast)
        history.append({"epoch": epoch, **result})
        print(f"Dev after epoch {epoch}: MRR={result['mrr']:.4f} | acc@1={result['acc_at_1']:.4f}")

        # Keep the best epoch. Without a dev set, keep the last epoch.
        if not dev_groups or result["mrr"] > best_mrr:
            best_mrr = result["mrr"]
            saved_epoch = epoch
            hf_model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"Saved model: {args.output_dir}")

    config = {
        "system": "Cross-encoder fine-tuning",
        "base_model": args.base_model,
        "output_dir": str(args.output_dir),
        "saved_epoch": saved_epoch,
        "loss": "softmax cross-entropy over 1 positive + N negatives",
        "negatives_per_group": {
            "hard": args.hard_negatives,
            "neighbour": args.neighbour_negatives,
            "random": args.random_negatives,
        },
        "hard_negative_files": [str(p) for p in hard_files],
        "hard_negative_depth": args.hard_negative_depth,
        "neighbour_window": args.neighbour_window,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "groups_per_batch": args.groups_per_batch,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "seed": args.seed,
        "n_train_queries": len(train_records),
        "n_dev_queries": len(dev_records),
        "n_train_groups": n_train_groups,
        "dev_history": history,
        "dev_query_ids": sorted(r["query_id"] for r in dev_records),
        "training_minutes": (time.perf_counter() - start) / 60,
    }

    if saved_epoch is None:
        print(
            "\nNo epoch beat the base model on the dev groups, so no model was saved. "
            "Try a lower --learning-rate or more hard negatives."
        )
    else:
        with open(args.output_dir / "finetune_config.json", "w", encoding="utf-8") as file:
            json.dump(config, file, indent=2, ensure_ascii=False)

    print("\n============================================================")
    print("FINE-TUNING COMPLETE")
    print("============================================================")
    print(f"Dev MRR: {before['mrr']:.4f} -> {best_mrr:.4f} (saved epoch: {saved_epoch})")

    if saved_epoch is not None:
        print("\nEvaluate on the validation queries with:")
        print(f"  python rerank_cross_encoder.py --model {args.output_dir} --output-prefix ce_finetuned")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
