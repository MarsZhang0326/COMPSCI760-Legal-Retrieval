"""Additional fine-tuned E5 Dense branch using completed sharded FAISS artifacts.

Queries use the shared data_loader; citation metrics use the shared evaluation.
PyTorch encoding is isolated in a subprocess of this file to avoid OpenMP
conflicts with FAISS. This entry point does not train, index or migrate data.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
from data_loader import load_test, load_train, load_val
from evaluation import evaluate_predictions, parse_citations

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / 'results/dense_e5_ft_artifacts'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def model_identity(directory):
    files = sorted(list(directory.glob('*.safetensors')) + list(directory.glob('pytorch_model*.bin')))
    if not files:
        raise FileNotFoundError(f'Missing model weights: {directory}')
    identity = {f.name: sha256(f) for f in files}
    identity['config'] = sha256(directory / 'config.json')
    return hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def encode_worker(request_file):
    """Only this subprocess imports torch/transformers; it never imports FAISS."""
    if 'KMP_DUPLICATE_LIB_OK' in os.environ:
        raise RuntimeError('Unset KMP_DUPLICATE_LIB_OK; unsafe OpenMP overrides are rejected')
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    request = read_json(request_file)
    model_dir = Path(request['model_dir'])
    if model_identity(model_dir) != request['model_sha256']:
        raise ValueError('Query model identity changed')
    selected_device = request['device']
    if selected_device == 'auto':
        selected_device = ('cuda' if torch.cuda.is_available() else
                           'mps' if torch.backends.mps.is_available() else 'cpu')
    torch.set_num_threads(request['threads'])
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).to(selected_device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    texts = ['query: ' + re.sub(r'\s+', ' ', str(q)).strip() for q in request['queries']]
    vectors = []
    from contextlib import nullcontext
    context = torch.autocast('cuda', dtype=torch.float16) if selected_device == 'cuda' else nullcontext()
    with torch.no_grad(), context:
        for start in range(0, len(texts), request['batch_size']):
            batch = tokenizer(texts[start:start + request['batch_size']], padding=True,
                              truncation=True, max_length=512, return_tensors='pt')
            batch = {key: value.to(selected_device) for key, value in batch.items()}
            hidden = model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            vectors.append(functional.normalize(pooled, p=2, dim=1).float().cpu().numpy())
    if 'faiss' in sys.modules:
        raise RuntimeError('FAISS unexpectedly imported in the encoder process')
    result = np.concatenate(vectors).astype(np.float32)
    with open(request['vectors_file'], 'wb') as output:
        np.save(output, result, allow_pickle=False)
    Path(request['worker_record']).write_text(json.dumps({
        'device': selected_device, 'faiss_imported': False,
        'model_sha256': request['model_sha256'], 'vectors_sha256': sha256(request['vectors_file']),
        'query_count': len(result), 'pooling': 'attention-mask mean, L2 normalize',
        'query_prefix': 'query: ', 'max_length': 512}), encoding='utf-8')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['val', 'test', 'train'], default='val')
    parser.add_argument('--index-dir', type=Path, default=ARTIFACTS / 'index')
    parser.add_argument('--model-dir', type=Path, help='Default: <index-dir>/../model')
    parser.add_argument('--corpus-map', type=Path, help='Default: <index-dir>/../corpus.sqlite')
    parser.add_argument('--artifact-dir', type=Path, default=ARTIFACTS,
                        help='Directory containing original protocol and corpus checksum metadata')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--faiss-threads', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=10, help='Fixed prediction cutoff; not tuned on validation')
    parser.add_argument('--output-prefix', default='dense_e5_ft')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--max-queries', type=int)
    parser.add_argument('--smoke-shards', type=int,
                        help='Explicit limited-corpus check; requires a smoke output prefix; no quality metrics')
    parser.add_argument('--verify-only', action='store_true', help='Check artifacts without running retrieval')
    parser.add_argument('--overwrite', action='store_true', help='Replace only this branch prefix, never legacy files')
    return parser.parse_args(argv)


def verify_artifacts(args):
    index_dir = args.index_dir.resolve()
    model_dir = (args.model_dir or index_dir.parent / 'model').resolve()
    mapping = (args.corpus_map or index_dir.parent / 'corpus.sqlite').resolve()
    metadata_dir = args.artifact_dir.resolve()
    branch = read_json(ROOT / 'results/dense_e5_ft_provenance.json')
    if (sha256(ROOT / 'evaluation.py') != branch['shared_evaluator_sha256'] or
        sha256(ROOT / 'data_loader.py') != branch['shared_data_loader_sha256']):
        raise ValueError('Shared loader/evaluator changed; review compatibility with the frozen experiment')
    manifest = read_json(index_dir / 'manifest.json')
    complete = read_json(index_dir / 'complete.json')
    if complete['manifest_sha256'] != sha256(index_dir / 'manifest.json'):
        raise ValueError('Index manifest checksum mismatch')
    protocol_sha = sha256(metadata_dir / 'protocol.json')
    if protocol_sha != manifest['protocol_sha256'] or protocol_sha != branch['protocol_sha256']:
        raise ValueError('Frozen protocol mismatch')
    identity = model_identity(model_dir)
    if identity != manifest['model_sha256'] or identity != branch['selected_model_sha256']:
        raise ValueError('Selected model/index mismatch')
    if (manifest['max_length'] != 512 or manifest['dimension'] != 768 or
        manifest['format'] != 'passage: citation title text' or
        manifest['pooling'] != 'attention-mask mean, L2 normalize'):
        raise ValueError('E5 input/pooling/dimension mismatch')
    snapshot_file = metadata_dir / 'corpus_snapshot_sha256.json'
    if sha256(snapshot_file) != branch['corpus_snapshot_metadata_sha256']:
        raise ValueError('Corpus checksum metadata changed')
    snapshot = read_json(snapshot_file)
    if mapping.stat().st_size != snapshot['bytes'] or sha256(mapping) != snapshot['sha256']:
        raise ValueError('Corpus mapping identity mismatch')
    expected_start = 0
    for shard_number, item in enumerate(complete['shards']):
        start, end = map(int, Path(item['file']).stem.split('_')[1:])
        if start != expected_start or end <= start:
            raise ValueError('Non-contiguous shard row ranges')
        expected_start = end
        file = index_dir / item['file']
        if args.smoke_shards is None and not file.is_file():
            raise FileNotFoundError(f'Missing complete-index shard: {file}')
        if args.verify_only and (args.smoke_shards is None or shard_number < args.smoke_shards):
            if sha256(file) != item['sha256']:
                raise ValueError(f'Shard checksum mismatch: {file.name}')
    if expected_start != complete['rows'] or complete['rows'] != manifest['corpus']['rows']:
        raise ValueError('Index/corpus row count mismatch')
    conn = sqlite3.connect(mapping.as_uri() + '?mode=ro', uri=True)
    if conn.execute('SELECT COUNT(*) FROM docs').fetchone()[0] != complete['rows']:
        raise ValueError('Corpus mapping row count mismatch')
    conn.close()
    # Archived model_path and snapshot path are descriptive, not runtime locations.
    return index_dir, model_dir, mapping, manifest, complete


def search(args, queries, verified):
    if 'torch' in sys.modules or 'KMP_DUPLICATE_LIB_OK' in os.environ:
        raise RuntimeError('Use a fresh FAISS-only process and unset KMP_DUPLICATE_LIB_OK')
    index_dir, model_dir, mapping, manifest, complete = verified
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='dense_e5_ft_tmp_', dir=args.output_dir) as temporary:
        temp = Path(temporary)
        request = {'model_dir': str(model_dir), 'model_sha256': manifest['model_sha256'],
                   'queries': queries['query'].astype(str).tolist(), 'device': args.device,
                   'batch_size': args.batch_size, 'threads': args.faiss_threads,
                   'vectors_file': str(temp / 'queries.npy'), 'worker_record': str(temp / 'encoder.json')}
        (temp / 'request.json').write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--encode-worker',
                        str(temp / 'request.json')], check=True)
        worker = read_json(temp / 'encoder.json')
        if worker['vectors_sha256'] != sha256(temp / 'queries.npy'):
            raise ValueError('Query vector transfer checksum mismatch')
        qvec = np.load(temp / 'queries.npy', allow_pickle=False)
    if qvec.dtype != np.float32 or qvec.shape != (len(queries), 768) or not np.isfinite(qvec).all():
        raise ValueError('Invalid query vectors')
    if np.max(np.abs(np.linalg.norm(qvec, axis=1) - 1)) >= .003:
        raise ValueError('Query vectors are not normalized')
    import faiss
    faiss.omp_set_num_threads(args.faiss_threads)
    scores = np.empty((len(queries), 0), dtype=np.float32)
    ids = np.empty((len(queries), 0), dtype=np.int64)
    shards = complete['shards'] if args.smoke_shards is None else complete['shards'][:args.smoke_shards]
    started = time.perf_counter()
    for number, item in enumerate(shards, 1):
        file = index_dir / item['file']
        if sha256(file) != item['sha256']:
            raise ValueError(f'Shard checksum mismatch: {file.name}')
        index = faiss.read_index(str(file))
        start, end = map(int, Path(item['file']).stem.split('_')[1:])
        if index.ntotal != end - start or index.d != 768:
            raise ValueError('Shard header mismatch')
        local_scores, local_ids = index.search(qvec, min(10000, index.ntotal))
        scores = np.concatenate([scores, local_scores], axis=1)
        ids = np.concatenate([ids, local_ids], axis=1)
        order = np.lexsort((ids, -scores), axis=1)[:, :min(10000, scores.shape[1])]
        scores = np.take_along_axis(scores, order, axis=1)
        ids = np.take_along_axis(ids, order, axis=1)
        del index
        if number % 20 == 0 or number == len(shards):
            print(f'Searched {number}/{len(shards)} shards; global passage budget=10000', flush=True)
    conn = sqlite3.connect(mapping.as_uri() + '?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    ranked = []
    for row_ids, row_scores in zip(ids, scores):
        hits, seen = [], set()
        for row_rank, (did, score) in enumerate(zip(row_ids, row_scores), 1):
            row = conn.execute('SELECT doc_id,citation,source_type FROM docs WHERE id=?', (int(did),)).fetchone()
            if row is None:
                raise ValueError('Index row is absent from mapping')
            citation = row['citation']
            if citation and citation not in seen:
                seen.add(citation)
                hits.append({'doc_id': row['doc_id'], 'citation': citation, 'source_type': row['source_type'],
                             'row_rank': row_rank, 'dense_score': float(score)})
            if len(hits) == 100:
                break
        ranked.append(hits)
    conn.close()
    return ranked, {'encoder': worker, 'torch_imported_in_search': 'torch' in sys.modules,
                    'shards_searched': len(shards), 'search_seconds': time.perf_counter() - started}


def write_outputs(args, queries, ranked, verified, runtime):
    stem = args.output_dir / f'{args.output_prefix}_{args.split}'
    files = {kind: Path(str(stem) + suffix) for kind, suffix in {
        'candidates': '_candidates_top100.csv', 'predictions': '_predictions.csv',
        'summary': '_summary.csv', 'per_query': '_per_query.csv', 'config': '_config.json'}.items()}
    if not args.overwrite and any(file.exists() for file in files.values()):
        raise FileExistsError('Choose a new dense_e5_ft_ output prefix or use explicit --overwrite')
    # Small serializers use the exact old Dense schema. Importing the old Dense
    # module itself would eagerly load FAISS and sentence-transformers/torch.
    candidates = [{'query_id': str(q.query_id), 'rank': rank, **hit}
                  for q, hits in zip(queries.itertuples(index=False), ranked)
                  for rank, hit in enumerate(hits, 1)]
    pd.DataFrame(candidates, columns=['query_id', 'rank', 'doc_id', 'citation', 'source_type',
                                    'row_rank', 'dense_score']).to_csv(files['candidates'], index=False)
    pd.DataFrame([{'query_id': str(q.query_id), 'predicted_citations': ';'.join(
        h['citation'] for h in hits[:args.top_k])} for q, hits in zip(queries.itertuples(index=False), ranked)]
    ).to_csv(files['predictions'], index=False)
    if args.split != 'test' and args.smoke_shards is None:
        mapping = verified[2]
        conn = sqlite3.connect(mapping.as_uri() + '?mode=ro', uri=True)
        full = {str(q.query_id): parse_citations(q.gold_citations) for q in queries.itertuples(index=False)}
        available = {g for labels in full.values() for g in labels
                     if conn.execute('SELECT 1 FROM docs WHERE citation=? LIMIT 1', (g,)).fetchone()}
        conn.close()
        summaries, details = [], []
        by_id = dict(zip(queries.query_id.astype(str), ranked))
        for mode in ['full', 'in_corpus']:
            gold = full if mode == 'full' else {qid: labels & available for qid, labels in full.items() if labels & available}
            if not gold:
                continue
            for k in sorted({10, 20, 50, 100, args.top_k}):
                predictions = {qid: [h['citation'] for h in by_id[qid][:k]] for qid in gold}
                metrics, per_query = evaluate_predictions(predictions, gold)
                summaries.append({'split': args.split, 'gold_mode': mode, 'k': k, **metrics,
                    'excluded_queries': len(full) - len(gold),
                    'excluded_gold_occurrences': 0 if mode == 'full' else sum(len(g - available) for g in full.values())})
                details.append(per_query.assign(split=args.split, gold_mode=mode, k=k))
        pd.DataFrame(summaries).to_csv(files['summary'], index=False)
        if details:
            pd.concat(details, ignore_index=True).to_csv(files['per_query'], index=False)
    config = {'branch': 'fine-tuned E5 bi-encoder; no Cross-Encoder', 'split': args.split,
        'checkpoint': 'epoch_2', 'model_sha256': verified[3]['model_sha256'],
        'protocol_sha256': verified[3]['protocol_sha256'], 'index_dir': str(verified[0]),
        'model_dir': str(verified[1]), 'corpus_map': str(verified[2]), 'top_k': args.top_k,
        'candidate_budget': 100, 'global_passage_budget': 10000, 'max_length': 512,
        'query_count': len(queries), 'limited_corpus_smoke_check': args.smoke_shards is not None,
        'max_queries_limit': args.max_queries, 'runtime': runtime,
        'evaluation_scope': 'None: smoke/unlabeled test' if args.smoke_shards is not None or args.split == 'test'
                            else 'Fixed cutoffs, full and in-corpus gold; no validation parameter selection'}
    files['config'].write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(files['candidates'])
    print(files['predictions'])


def main(argv=None):
    args = parse_args(argv)
    if args.batch_size < 1 or args.faiss_threads < 1 or not 1 <= args.top_k <= 100:
        raise ValueError('Batch size/threads must be positive; top-k must be between 1 and 100')
    if (args.output_prefix != 'dense_e5_ft' and not args.output_prefix.startswith('dense_e5_ft_')) or not re.fullmatch(r'[A-Za-z0-9_-]+', args.output_prefix):
        raise ValueError('Output prefix must be a simple filename beginning with dense_e5_ft_')
    if args.smoke_shards is not None and (args.smoke_shards < 1 or 'smoke' not in args.output_prefix.lower()):
        raise ValueError('Limited-corpus checks require positive --smoke-shards and a smoke output prefix')
    if args.max_queries is not None and args.max_queries < 1:
        raise ValueError('--max-queries must be positive')
    verified = verify_artifacts(args)
    if args.verify_only:
        print(json.dumps({'verified': True, 'checkpoint': 'epoch_2', 'rows': verified[4]['rows'],
                          'shards': len(verified[4]['shards']), 'model_sha256': verified[3]['model_sha256'],
                          'limited_shards_allowed': args.smoke_shards is not None}, indent=2))
        return
    queries = {'val': load_val, 'test': load_test, 'train': load_train}[args.split]().copy()
    if args.max_queries is not None:
        queries = queries.head(args.max_queries)
    if queries.empty or queries.query_id.astype(str).duplicated().any() or queries['query'].isna().any():
        raise ValueError('Queries must be nonempty, have unique IDs and contain query text')
    ranked, runtime = search(args, queries, verified)
    write_outputs(args, queries, ranked, verified, runtime)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--encode-worker':
        encode_worker(sys.argv[2])
    else:
        main()
