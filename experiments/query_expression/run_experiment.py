"""Run the frozen query-expression study or check its saved results without retrieval."""
import argparse
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def resolve_config(root, args):
    config = json.loads((root / 'config.json').read_text(encoding='utf8'))
    for key in ('model_dir', 'index_dir', 'corpus_map'):
        value = getattr(args, key) or config[key]
        config[key] = str((root / value).resolve())
    for key in ('query_file', 'gold_file'):
        config[key] = str((root / config[key]).resolve())
    if args.device:
        config['device'] = args.device
    return config


def verify(root, config, check_artifacts=False):
    # Only pandas and the shared evaluator are used here; no encoder or FAISS load.
    import pandas as pd
    import evaluation
    assert 'KMP_DUPLICATE_LIB_OK' not in os.environ, 'Unset the duplicate-runtime bypass'
    assert 'torch' not in sys.modules and 'faiss' not in sys.modules
    retained = json.loads((root / 'provenance/retained_sources.json').read_text())
    for relative, record in retained.items():
        expected = record.get('public_sha256', record['archive_sha256'])
        assert sha(root / relative) == expected, relative
    manifest = json.loads((root / 'inputs/variant_manifest.json').read_text())
    assert sha(root / 'provenance/original_config.json') == manifest['config_sha256']
    assert sha(config['query_file']) == manifest['query_variants_sha256']
    gold = evaluation.load_gold(config['gold_file'])
    queries = pd.read_csv(config['query_file'], keep_default_na=False)
    assert len(gold) == 10 and len(queries) == 30
    assert not queries.duplicated(['query_id', 'variant']).any()
    assert set(queries.variant) == {'en', 'de', 'legal_original'}
    for variant, group in queries.groupby('variant'):
        assert set(group.query_id) == set(gold)
        assert group['query'].str.len().min() > 0
    numeric_atoms = lambda text: set(re.findall(r'\d+(?:[.,]\d+)*', text))
    for qid in gold:
        group = queries[queries.query_id == qid].set_index('variant')['query']
        assert numeric_atoms(group['en']) == numeric_atoms(group['de'])
        assert group['en'] in group['legal_original']
    checks = pd.read_csv(root / 'validation/transformation_checks.csv')
    assert set(checks.query_id) == set(gold)
    assert checks.numeric_atoms_equal.all() and checks.legal_keeps_full_original.all()
    candidates = pd.read_csv(root / 'results/all_candidates.csv', keep_default_na=False)
    assert {'query_id', 'variant', 'rank', 'id', 'doc_id', 'citation', 'source_type'} <= set(candidates)
    assert len(candidates) == 6000 and set(candidates.variant) == set(config['variants'])
    assert set(candidates.query_id) == set(gold)
    assert set(candidates.source_type) <= {'law', 'court'}
    for _, group in candidates.groupby(['query_id', 'variant']):
        assert len(group) == group.citation.nunique() == 100
        assert group['rank'].tolist() == list(range(1, 101))
        assert group.citation.str.strip().equals(group.citation) and (group.citation != '').all()
    metrics = pd.read_csv(root / 'results/metrics.csv')
    per_query = pd.read_csv(root / 'results/per_query_metrics.csv')
    comparison = pd.read_csv(root / 'results/main_comparison.csv')
    assert len(metrics) == 24 and len(per_query) == 240 and len(comparison) == 6
    assert set(metrics.gold_mode) == {'full'} and set(metrics.split) == {'official_val'}
    for variant in config['variants']:
        group = candidates[candidates.variant == variant]
        for k in config['cutoffs']:
            predictions = {qid: group[(group.query_id == qid) & (group['rank'] <= k)].citation.tolist() for qid in gold}
            fresh, details = evaluation.evaluate_predictions(predictions, gold)
            saved = metrics[(metrics.variant == variant) & (metrics.k == k)]
            assert len(saved) == 1
            for name in ('macro_precision', 'macro_recall', 'macro_f1'):
                assert abs(fresh[name] - saved.iloc[0][name]) < 1e-12
            stored = per_query[(per_query.variant == variant) & (per_query.k == k)].set_index('query_id')
            assert set(stored.index) == set(gold) and len(stored) == 10
            for record in details.itertuples(index=False):
                for name in ('tp', 'fp', 'fn', 'precision', 'recall', 'f1'):
                    assert abs(getattr(record, name) - stored.loc[record.query_id, name]) < 1e-12
            row = comparison[comparison.variant == variant].iloc[0]
            if k in (20, 50, 100):
                assert abs(fresh['macro_recall'] - row['recall' + str(k)]) < 1e-12
            if k == 10:
                assert abs(fresh['macro_f1'] - row.macro_f1_10) < 1e-12
    for qid, group in candidates.groupby('query_id'):
        original = group[group.variant == 'en'].citation.tolist()
        for control in ('en_deep2x', 'en_repeat2x'):
            assert original == group[group.variant == control].citation.tolist()
    identity = json.loads((root / 'provenance/artifacts.json').read_text())
    encoder = json.loads((root / 'provenance/encoder_record.json').read_text())
    search = json.loads((root / 'provenance/search_record_first_complete.json').read_text())
    assert encoder['model_sha256'] == search['model_sha256'] == identity['model_sha256']
    assert search['manifest_sha256'] == identity['manifest_sha256']
    assert search['mapping_sha256'] == identity['mapping_sha256']
    assert search['complete'] and search['shards_searched'] == 266 and not search['cross_encoder_executed']
    artifact_checks = 'Not requested'
    if check_artifacts:
        # Hash local model weights, not full index shards or the 3.2 GB mapping.
        # A complete reproduction checks all shard/mapping hashes in its search worker.
        from encode_queries import model_identity
        model = Path(config['model_dir'])
        index = Path(config['index_dir'])
        mapping = Path(config['corpus_map'])
        index_manifest = json.loads((index / 'manifest.json').read_text())
        complete = json.loads((index / 'complete.json').read_text())
        assert sha(index / 'manifest.json') == complete['manifest_sha256'] == identity['manifest_sha256']
        assert model_identity(model) == index_manifest['model_sha256'] == identity['model_sha256']
        assert sha(model / 'tokenizer.json') == encoder['tokenizer_sha256']
        assert index_manifest['corpus']['rows'] == complete['rows'] == 2652248
        assert index_manifest['corpus']['input_manifest_sha256'] == identity['input_manifest_sha256']
        assert index_manifest['format'] == 'passage: citation title text'
        assert index_manifest['pooling'] == 'attention-mask mean, L2 normalize'
        assert index_manifest['max_length'] == 512 and index_manifest['dimension'] == 768
        assert len(complete['shards']) == 266 and all((index / s['file']).is_file() for s in complete['shards'])
        snapshot = json.loads((mapping.parent / 'corpus_snapshot_sha256.json').read_text())
        assert snapshot['sha256'] == identity['mapping_sha256'] and mapping.stat().st_size == snapshot['bytes']
        conn = sqlite3.connect(mapping.as_uri() + '?mode=ro&immutable=1', uri=True)
        try:
            for row in candidates[['id', 'doc_id', 'citation', 'source_type']].drop_duplicates().itertuples(index=False):
                actual = conn.execute('SELECT doc_id,citation,source_type FROM docs WHERE id=?', (int(row.id),)).fetchone()
                assert actual == (row.doc_id, row.citation, row.source_type)
        finally:
            conn.close()
        artifact_checks = 'Model/tokenizer hashes; manifest/completion identity; shard presence; mapping size/record; all candidate row IDs'
    result = {'passed': True, 'queries': 10, 'groups': 6, 'real_saved_candidates': 6000,
              'metric_configurations_rechecked': 24, 'per_query_metrics_rechecked': 240,
              'verification_scope': 'Saved-result rescoring and packaging checks, not new retrieval',
              'artifact_checks': artifact_checks, 'full_index_hash_scan': False,
              'encoder_executed': False, 'retrieval_executed': False, 'cross_encoder_executed': False}
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--verify-only', action='store_true', help='Rescore saved candidates and check formats without retrieval')
    p.add_argument('--check-artifacts', action='store_true', help='Also check local external artifacts; use with --verify-only')
    p.add_argument('--model-dir', type=Path, help='Override pretrained encoder location')
    p.add_argument('--index-dir', type=Path, help='Override its matching sharded index location')
    p.add_argument('--corpus-map', type=Path, help='Override full corpus.sqlite location; keep its snapshot JSON beside it')
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'))
    p.add_argument('--output-dir', type=Path, help='Fresh/resumable run directory; default <package>/runs/reproduction')
    args = p.parse_args()
    if args.check_artifacts and not args.verify_only:
        p.error('--check-artifacts requires --verify-only')
    root = args.root.resolve()
    config = resolve_config(root, args)
    if args.verify_only:
        verify(root, config, args.check_artifacts)
        return
    assert 'KMP_DUPLICATE_LIB_OK' not in os.environ, 'Unset the duplicate-runtime bypass'
    identity = json.loads((root / 'provenance/artifacts.json').read_text())
    index = Path(config['index_dir'])
    assert sha(index / 'manifest.json') == identity['manifest_sha256'], 'Expected archived pretrained index'
    assert config['expected_mapping_sha256'] == identity['mapping_sha256']
    assert config['candidate_budget'] == 100 and config['base_row_depth'] == 1000 and config['deep_row_depth'] == 2000
    assert config['rrf_k'] == 60 and config['fusion_weights'] == [1.0, 1.0]
    assert config['variants'] == ['en', 'de', 'legal_original', 'en_de_rrf', 'en_deep2x', 'en_repeat2x']
    assert config['cutoffs'] == [10, 20, 50, 100] and config['max_length'] == 512
    assert sha(config['query_file']) == json.loads((root / 'inputs/variant_manifest.json').read_text())['query_variants_sha256']
    retained = json.loads((root / 'provenance/retained_sources.json').read_text())
    assert sha(config['gold_file']) == retained['inputs/gold_official_val.json']['archive_sha256']
    out = (args.output_dir or (root / 'runs/reproduction')).resolve()
    assert out != root and (root not in out.parents or out.is_relative_to(root / 'runs')), 'Use a dedicated run directory'
    assert out != Path(config['model_dir']) and out != index and out != Path(config['corpus_map']).parent
    (out / 'inputs').mkdir(parents=True, exist_ok=True)
    (out / 'results').mkdir(exist_ok=True)
    config_file = out / 'config.json'
    if config_file.exists():
        assert json.loads(config_file.read_text()) == config, 'Run configuration changed; choose a new --output-dir'
    else:
        config_file.write_text(json.dumps(config, indent=2), encoding='utf8')
    if not (out / 'inputs/query_vectors.npy').exists() or not (out / 'inputs/encoder_record.json').exists():
        subprocess.run([sys.executable, str(root / 'encode_queries.py'), '--root', str(out)], check=True)
    # Fresh subprocesses prevent simultaneous loading of PyTorch and FAISS OpenMP runtimes.
    subprocess.run([sys.executable, str(root / 'search_and_evaluate.py'), '--root', str(out)], check=True)
    print('Completed all six Dense-only groups; results: ' + str(out / 'results'))


if __name__ == '__main__':
    main()
