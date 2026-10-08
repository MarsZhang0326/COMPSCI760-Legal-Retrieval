"""Verify compact-package integrity and rescore actual saved artifacts without retrieval."""
from pathlib import Path
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from evaluation import evaluate_predictions, parse_citations

def rescore_saved_artifacts():
    rows=json.loads((ROOT/'inputs/queries.json').read_text(encoding='utf8'))
    source=json.loads((ROOT/'inputs/gold_sources.json').read_text(encoding='utf8'))
    covered={citation for citation,types in source.items() if types}
    all_gold={row['query_id']:set(row['gold']) for row in rows}
    splits={split:[row['query_id'] for row in rows if row['split']==split]
            for split in ('eval','official_val')}
    comparisons=0
    def compare(frame, ids, k, mode, saved):
        nonlocal comparisons
        gold={qid:all_gold[qid] for qid in ids}
        if mode in ('covered','in_corpus'):
            gold={qid:labels&covered for qid,labels in gold.items() if labels&covered}
        rankings={qid:g.sort_values('rank').citation.tolist()[:int(k)]
                  for qid,g in frame.groupby('query_id') if qid in gold}
        metric,_=evaluate_predictions(rankings,gold)
        for key in ('macro_precision','macro_recall','macro_f1'):
            assert abs(metric[key]-float(saved[key]))<1e-12, (key,metric[key],saved[key])
        assert metric['n_queries']==int(saved['n_queries'])
        comparisons+=1
    for prefix in ('final_original_E5','final_finetuned_E5'):
        location=ROOT/'inputs/archived_e5_results/results'
        frame=pd.read_csv(location/(prefix+'_candidates.csv'),keep_default_na=False)
        summary=pd.read_csv(location/(prefix+'_summary.csv'))
        for saved in summary.to_dict('records'):
            compare(frame,splits[saved['split']],saved['k'],saved['gold_mode'],saved)
    summary=pd.read_csv(ROOT/'results/seed_pool_metrics.csv')
    for label in ('base','seed_760','seed_761','seed_762'):
        frame=pd.read_csv(ROOT/'results'/f'seed_pool_{label}_rankings.csv',keep_default_na=False)
        for saved in summary[summary.system=='Pool_'+label].to_dict('records'):
            compare(frame,splits[saved['split']],saved['k'],saved['gold_mode'],saved)
    selected=0
    for prefix,name in (('bm25','bm25_best_config.json'),('dense_full','dense_full_config.json'),
                        ('ce_bge_zeroshot','ce_bge_zeroshot_config.json'),('ce_bge_ft','ce_bge_ft_config.json'),
                        ('ce_mini','ce_mini_config.json')):
        location=ROOT/'inputs/historical'
        predictions=pd.read_csv(location/(prefix+'_best_predictions.csv'),keep_default_na=False)
        gold={qid:all_gold[qid] for qid in splits['official_val']}
        score,_=evaluate_predictions({r.query_id:parse_citations(r.predicted_citations)
                                     for r in predictions.itertuples(index=False)},gold)
        saved=json.loads((location/name).read_text('utf8'))
        for key in ('macro_precision','macro_recall','macro_f1'):
            assert abs(score[key]-saved[key])<1e-12
        selected+=1
    return {'archived_full_corpus_and_seed_pool_configurations_rechecked':comparisons,
            'historical_selected_predictions_rechecked':selected,
            'scope':'Saved real-artifact rescoring only; no new encoding, retrieval or training'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-artifacts',action='store_true',help='Also hash original external vectors/checkpoints and check index manifests; no encoding/search')
    parser.add_argument('--artifact-dir',type=Path,help='Override the existing stability research archive')
    parser.add_argument('--e5-archive',type=Path,help='Override the existing full E5 archive')
    args=parser.parse_args()
    config=json.loads((ROOT/'config.json').read_text('utf8'))
    manifest=json.loads((ROOT/'CHECKSUMS.json').read_text('utf8'))
    assert all(sha(ROOT/name)==digest for name,digest in manifest.items()), 'Public package checksum mismatch'
    rows=json.loads((ROOT/'inputs/queries.json').read_text('utf8'))
    assert len({r['query_id'] for r in rows})==len(rows)==1149
    assert {s:sum(r['split']==s for r in rows) for s in ('train','dev','eval','official_val')}=={'train':911,'dev':114,'eval':114,'official_val':10}
    sources=json.loads((ROOT/'inputs/gold_sources.json').read_text('utf8'))
    assert all(c in sources for r in rows for c in r['gold'])
    pool=pd.read_csv(ROOT/'inputs/seed_candidate_pool.csv',keep_default_na=False)
    meta=json.loads((ROOT/'inputs/seed_pool_record.json').read_text('utf8'))
    assert sha(ROOT/'inputs/seed_candidate_pool.csv')==meta['pool_sha256']
    assert sha(ROOT/'inputs/seed_queries.csv')==meta['query_sha256']
    assert not meta['gold_injected'] and pool.query_id.nunique()==124
    assert (pool.groupby('query_id').size()<=200).all()
    for source in meta['sources']:
        assert sha(ROOT/'inputs/historical'/source['file'])==source.get('public_sha256',source['sha256'])
    actual=set(zip(pool.query_id,pool.doc_id,pool.citation))
    reference=set()
    for branch in ('bm25','dense_full'):
        for suffix in ('_candidates_top100.csv','_train_candidates_top100.csv'):
            frame=pd.read_csv(ROOT/'inputs/historical'/(branch+suffix),keep_default_na=False)
            reference.update(zip(frame.query_id,frame.doc_id,frame.citation))
    assert actual<=reference
    for label in ('base','seed_760','seed_761','seed_762'):
        frame=pd.read_csv(ROOT/'results'/f'seed_pool_{label}_rankings.csv',keep_default_na=False)
        assert set(zip(frame.query_id,frame.citation))==set(zip(pool.query_id,pool.citation))
        assert set(zip(frame.query_id,frame.doc_id,frame.citation))<=actual
        for _,group in frame.groupby('query_id'):
            assert len(group)==group.citation.nunique()
            assert group['rank'].tolist()==list(range(1,len(group)+1))
    paired=pd.read_csv(ROOT/'results/paired_query_deltas.csv')
    stats=pd.read_csv(ROOT/'results/paired_statistics.csv').set_index('comparison')
    assert paired.comparison.nunique()==len(stats)==8
    for name,group in paired.groupby('comparison'):
        assert abs(group.delta.mean()-stats.loc[name,'mean_delta'])<1e-12
    leave=pd.read_csv(ROOT/'results/leave_one_query_out.csv')
    for r in leave.itertuples(index=False):
        subset=paired[(paired.comparison==r.comparison)&(paired.query_id!=r.removed_query)]
        assert abs(subset.delta.mean()-r.mean_delta)<1e-12
    # Evidence of real historical training is retained; no training is executed here.
    audit=json.loads((ROOT/'provenance/new_training_audit.json').read_text('utf8'))
    assert {r['seed'] for r in audit}=={761,762}
    for seed in (761,762):
        run=ROOT/'seed_runs'/f'seed_{seed}'
        assert json.loads((run/'training_status.json').read_text('utf8'))['exit_code']==0
        logs=[json.loads(line) for line in (run/'logs/train.jsonl').read_text('utf8').splitlines()]
        assert len(logs)==660 and [r['step'] for r in logs]==list(range(1,661))
        assert all(np.isfinite(r['loss_last_microbatch']) for r in logs)
    artifact_checks=False
    if args.check_artifacts:
        import archived_experiment as e5
        artifact=(args.artifact_dir or ROOT/config['seed_artifact_root']).resolve()
        archive=(args.e5_archive or ROOT/config['e5_archive']).resolve()
        assert sha(artifact/'inputs/seed_passages.jsonl')==meta['passages_sha256']
        mapping=archive/'data/corpus.sqlite'
        snapshot=json.loads((mapping.parent/'corpus_snapshot_sha256.json').read_text('utf8'))
        audit=json.loads((ROOT/'provenance/source_audit.json').read_text('utf8'))
        assert mapping.stat().st_size==snapshot['bytes']
        assert snapshot['sha256']==audit['mapping_sha256_verified']
        for label in ('base','seed_760','seed_761','seed_762'):
            record=json.loads((ROOT/'inputs/seed_embeddings'/(label+'_record.json')).read_text('utf8'))
            for kind in ('documents','queries'):
                assert sha(artifact/'inputs/seed_embeddings'/(label+'_'+kind+'.npy'))==record[kind+'_vectors_sha256']
            model=archive/'models'/('base' if label=='base' else 'epoch_2') if label in ('base','seed_760') else artifact/'seed_runs'/label/'models/epoch_2'
            assert e5.model_identity(model)==record['model_sha256']
        for name,label in (('base','base'),('epoch_2','seed_760')):
            folder=archive/'indexes'/name
            index=json.loads((folder/'manifest.json').read_text('utf8'))
            complete=json.loads((folder/'complete.json').read_text('utf8'))
            record=json.loads((ROOT/'inputs/seed_embeddings'/(label+'_record.json')).read_text('utf8'))
            assert sha(folder/'manifest.json')==complete['manifest_sha256']
            assert index['model_sha256']==record['model_sha256']
            assert index['dimension']==768 and index['corpus']['rows']==complete['rows']==2652248
            assert index['max_length']==512
            assert index['format']=='passage: citation title text' and index['pooling']=='attention-mask mean, L2 normalize'
            assert all((folder/shard['file']).exists() for shard in complete['shards'])
        artifact_checks=True
    scores=rescore_saved_artifacts()
    result={'passed':True,'scope':'Compact-package checks and saved-artifact rescoring; not new model inference',
            'scores':scores,'paired_contrasts':8,'saved_training_log_rows_checked':1320,
            'query_split_counts':{'train':911,'dev':114,'eval':114,'official_val':10},
            'external_identity_checks_executed':artifact_checks,'full_index_or_mapping_hash_scan':False,
            'training_executed':False,'full_retrieval_executed':False,'cross_encoder_executed':False}
    output=ROOT/'runs/verification';output.mkdir(parents=True,exist_ok=True)
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
