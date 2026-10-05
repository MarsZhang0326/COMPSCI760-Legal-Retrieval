import argparse
import csv
import gzip
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from data_loader import iter_court_chunks, load_laws


try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
DEFAULT_INDEX_DIR = RESULTS_DIR / "faiss_dense_full"

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
COURT_CHUNK_SIZE = 50_000
BATCH_SIZE = 64


def require_dependencies():
    missing = []

    if faiss is None:
        missing.append("faiss-cpu")

    if SentenceTransformer is None:
        missing.append("sentence-transformers")

    if missing:
        raise SystemExit(
            "Missing dense retrieval dependencies: "
            + ", ".join(missing)
            + "\nInstall them with:\n"
            + "  python -m pip install -r requirements.txt"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a full-corpus FAISS dense retrieval index "
            "over laws_de.csv and court_considerations.csv."
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="SentenceTransformer model name.",
    )

    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directory for the FAISS index and metadata.",
    )

    parser.add_argument(
        "--sources",
        choices=["full", "law", "court"],
        default="full",
        help="Corpus source to index. Use full for laws plus court.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Embedding batch size.",
    )

    parser.add_argument(
        "--court-chunk-size",
        type=int,
        default=COURT_CHUNK_SIZE,
        help="Rows per streamed court chunk.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Optional SentenceTransformer device, e.g. cpu, cuda, or mps.",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Encode in half precision (CUDA only). About 2x faster on a GPU.",
    )

    parser.add_argument(
        "--faiss-threads",
        type=int,
        default=None,
        help="Optional number of FAISS CPU threads.",
    )

    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help=(
            "Optional smoke-test limit. Omit this argument for the "
            "full dense index."
        ),
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Print build progress after this many indexed documents.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing index directory before building.",
    )

    return parser.parse_args()


def uses_e5_prompt(model_name):
    return "e5" in model_name.lower()


def build_passage_text(row, model_name):
    citation = str(row.citation or "").strip()
    title = str(row.title or "").strip()
    text = str(row.text or "").strip()

    passage = " ".join(
        part for part in [citation, title, text] if part
    )

    if uses_e5_prompt(model_name):
        return "passage: " + passage

    return passage


def iter_source_chunks(args):
    if args.sources in {"full", "law"}:
        yield "law", load_laws()

    if args.sources in {"full", "court"}:
        for chunk in iter_court_chunks(
            chunksize=args.court_chunk_size
        ):
            yield "court", chunk


def encode_texts(model, texts, batch_size):
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(
        embeddings,
        dtype=np.float32,
    )


def open_metadata_writer(path):
    handle = gzip.open(
        path,
        "wt",
        encoding="utf-8",
        newline="",
    )

    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "doc_int_id",
            "doc_id",
            "citation",
            "source_type",
        ],
    )

    writer.writeheader()

    return handle, writer


def flush_batch(
    model,
    index,
    metadata_writer,
    batch_texts,
    batch_metadata,
    batch_size,
):
    if not batch_texts:
        return 0

    embeddings = encode_texts(
        model,
        batch_texts,
        batch_size,
    )

    ids = np.asarray(
        [item["doc_int_id"] for item in batch_metadata],
        dtype=np.int64,
    )

    index.add_with_ids(
        embeddings,
        ids,
    )

    metadata_writer.writerows(
        batch_metadata
    )

    return len(batch_texts)


def main():
    args = parse_args()
    require_dependencies()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    if args.max_docs is not None and args.max_docs <= 0:
        raise ValueError("--max-docs must be positive when provided.")

    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive.")

    if args.faiss_threads is not None:
        faiss.omp_set_num_threads(
            args.faiss_threads
        )

    index_dir = args.index_dir.resolve()
    index_file = index_dir / "dense.faiss"
    metadata_file = index_dir / "metadata.csv.gz"
    config_file = index_dir / "dense_index_config.json"

    if index_dir.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Index directory already exists: {index_dir}\n"
                "Use --overwrite to rebuild it."
            )

        shutil.rmtree(
            index_dir
        )

    index_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("============================================================")
    print("BUILD FULL DENSE FAISS INDEX")
    print("============================================================")
    print(f"Model: {args.model}")
    print(f"Sources: {args.sources}")
    print(f"Index directory: {index_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max docs: {args.max_docs or 'full corpus'}")
    print(f"Progress every: {args.progress_every:,} docs")

    model = SentenceTransformer(
        args.model,
        device=args.device,
    )

    if args.fp16:
        if model.device.type != "cuda":
            raise SystemExit("--fp16 needs a CUDA GPU. Remove --fp16 on CPU or MPS.")
        model.half()
        print("Precision: float16")

    probe_text = (
        "passage: dimension probe"
        if uses_e5_prompt(args.model)
        else "dimension probe"
    )

    probe_embedding = encode_texts(
        model,
        [probe_text],
        batch_size=1,
    )

    embedding_dim = int(
        probe_embedding.shape[1]
    )

    index = faiss.IndexIDMap2(
        faiss.IndexFlatIP(
            embedding_dim
        )
    )

    metadata_handle, metadata_writer = open_metadata_writer(
        metadata_file
    )

    start = time.perf_counter()
    total_docs = 0
    batch_texts = []
    batch_metadata = []
    chunk_number = 0
    next_progress = args.progress_every

    try:
        for source_name, chunk in iter_source_chunks(args):
            chunk_number += 1

            for row in chunk.itertuples(index=False):
                if (
                    args.max_docs is not None
                    and total_docs >= args.max_docs
                ):
                    break

                batch_texts.append(
                    build_passage_text(
                        row,
                        args.model,
                    )
                )

                batch_metadata.append({
                    "doc_int_id":
                        total_docs,

                    "doc_id":
                        row.doc_id,

                    "citation":
                        str(row.citation).strip(),

                    "source_type":
                        row.source_type,
                })

                total_docs += 1

                if len(batch_texts) >= args.batch_size:
                    flush_batch(
                        model,
                        index,
                        metadata_writer,
                        batch_texts,
                        batch_metadata,
                        args.batch_size,
                    )

                    batch_texts.clear()
                    batch_metadata.clear()

                if total_docs >= next_progress:
                    elapsed = time.perf_counter() - start
                    docs_per_minute = (
                        total_docs / max(elapsed / 60, 1e-9)
                    )

                    print(
                        f"Progress | "
                        f"source={source_name} | "
                        f"indexed={total_docs:,} | "
                        f"{docs_per_minute:,.0f} docs/min | "
                        f"{elapsed / 60:.1f} min",
                        flush=True,
                    )

                    next_progress += args.progress_every

            elapsed = time.perf_counter() - start

            print(
                f"Chunk {chunk_number:03d} | "
                f"source={source_name} | "
                f"indexed={total_docs:,} | "
                f"{elapsed / 60:.1f} min",
                flush=True,
            )

            if (
                args.max_docs is not None
                and total_docs >= args.max_docs
            ):
                break

        flush_batch(
            model,
            index,
            metadata_writer,
            batch_texts,
            batch_metadata,
            args.batch_size,
        )

    finally:
        metadata_handle.close()

    faiss.write_index(
        index,
        str(index_file),
    )

    elapsed = time.perf_counter() - start

    config = {
        "system":
            "Full-corpus dense retrieval",

        "model":
            args.model,

        "sources":
            args.sources,

        "index_type":
            "IndexIDMap2(IndexFlatIP)",

        "similarity":
            "inner product over L2-normalized embeddings",

        "embedding_dim":
            embedding_dim,

        "document_count":
            total_docs,

        "passage_format":
            (
                "passage: citation title text"
                if uses_e5_prompt(args.model)
                else "citation title text"
            ),

        "metadata_file":
            str(metadata_file),

        "index_file":
            str(index_file),

        "elapsed_minutes":
            elapsed / 60,
    }

    with open(
        config_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            config,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n============================================================")
    print("DENSE INDEX BUILD COMPLETE")
    print("============================================================")
    print(f"Documents indexed: {total_docs:,}")
    print(f"Embedding dim: {embedding_dim}")
    print(f"Elapsed: {elapsed / 60:.1f} min")
    print(f"Index: {index_file}")
    print(f"Metadata: {metadata_file}")
    print(f"Config: {config_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")
