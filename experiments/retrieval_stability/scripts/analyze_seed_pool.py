"""Real fixed-pool E5 scoring with shared candidates and citation evaluator.

These results diagnose ranking conditional on a frozen candidate pool. They do
not measure new-seed full-corpus retrieval or improvements to candidate coverage.
"""
import argparse
from pathlib import Path
import json, sys, time
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze_stability as study

ROOT=study.ROOT

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/study.CONFIG['default_output'])
    parser.add_argument('--artifact-dir',type=Path,help='Existing research archive with inputs/seed_embeddings and seed_passages.jsonl')
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);study.OUT=out
    pool=pd.read_csv(ROOT/'inputs/seed_candidate_pool.csv',keep_default_na=False)
    artifact=(args.artifact_dir or (ROOT/study.CONFIG['seed_artifact_root'])).resolve()
    passage_file=ROOT/'inputs/seed_passages.jsonl'
    if not passage_file.exists():passage_file=artifact/'inputs/seed_passages.jsonl'
    documents=pd.read_json(passage_file,lines=True)
    queries=pd.read_csv(ROOT/'inputs/seed_queries.csv',keep_default_na=False)
    doc_position={doc_id:i for i,doc_id in enumerate(documents.doc_id)}
    qposition={qid:i for i,qid in enumerate(queries.query_id)}
    embeddings=artifact/'inputs/seed_embeddings';frames=[];runtime=[]
    for label in ('base','seed_760','seed_761','seed_762'):
        record=json.loads((ROOT/'inputs/seed_embeddings'/(label+'_record.json')).read_text(encoding='utf8'))
        assert record['document_order_sha256']==study.sha(passage_file)
        assert record['query_order_sha256']==study.sha(ROOT/'inputs/seed_queries.csv')
        dfile=embeddings/(label+'_documents.npy');qfile=embeddings/(label+'_queries.npy')
        assert study.sha(dfile)==record['documents_vectors_sha256'] and study.sha(qfile)==record['queries_vectors_sha256']
        dvec=np.load(dfile,allow_pickle=False);qvec=np.load(qfile,allow_pickle=False)
        assert dvec.shape==(len(documents),768) and qvec.shape==(124,768)
        assert np.isfinite(dvec).all() and np.isfinite(qvec).all()
        assert np.max(np.abs(np.linalg.norm(dvec,axis=1)-1))<.003
        ranked=[]
        for qid,group in pool.groupby('query_id'):
            started=time.perf_counter()
            positions=[doc_position[doc_id] for doc_id in group.doc_id]
            group=group.copy();group['score']=dvec[positions]@qvec[qposition[qid]]
            group=group.sort_values(['score','doc_id'],ascending=[False,True]).drop_duplicates('citation')
            group['rank']=range(1,len(group)+1);group['encoder']=label
            ranked.append(group)
            runtime.append({'query_id':qid,'encoder':label,'dot_product_sort_seconds':time.perf_counter()-started,
                            'candidate_passage_pairs':len(positions),'unique_pool_citations':len(group),
                            'scope':'Fixed-pool scoring only; query/document encoding and archive I/O excluded'})
        frame=pd.concat(ranked,ignore_index=True);frames.append(frame)
        frame.to_csv(out/('seed_pool_'+label+'_rankings.csv'),index=False)
        for split in ('eval','official_val'):
            ids=queries[queries.split==split].query_id.tolist()
            for k in [10,20,50,100]:
                pred=study.candidate_predictions(frame[frame.query_id.isin(ids)],k)
                for mode in ('full','covered'):
                    study.evaluate('Pool_'+label,pred,ids,split,k,
                                   'New real fixed-pool dot-product rescoring; base vectors reused, all three tuned-seed pool embeddings use matched batching',
                                   scope='conditional_candidate_rescoring',gold_mode=mode)
    metrics=pd.DataFrame(study.SUMMARY);details=pd.concat(study.DETAILS,ignore_index=True)
    metrics.to_csv(out/'seed_pool_metrics.csv',index=False)
    details.to_csv(out/'seed_pool_per_query.csv',index=False)
    pd.DataFrame(runtime).to_csv(out/'seed_pool_scoring_times.csv',index=False)
    # Actual variability across three trained checkpoints on exactly the same pool.
    spread=[];consistency=[];pairs=[]
    for split in ('eval','official_val'):
        for k in [10,20,50,100]:
            trial=metrics[(metrics.split==split)&(metrics.k==k)&(metrics.gold_mode=='full')&metrics.system.str.startswith('Pool_seed_')]
            assert len(trial)==3
            for metric in ('macro_precision','macro_recall','macro_f1'):
                spread.append({'split':split,'k':k,'metric':metric,'seeds_completed':3,'minimum':trial[metric].min(),
                               'maximum':trial[metric].max(),'mean':trial[metric].mean(),'sample_std':trial[metric].std(ddof=1),
                               'scope':'Three seeds; frozen samples/split/final epoch and fixed candidate pool. No seed-population CI claimed.'})
            baseline=details[(details.system=='Pool_base')&(details.split==split)&(details.k==k)&(details.gold_mode=='full')].set_index('query_id')
            seeds={seed:details[(details.system=='Pool_'+seed)&(details.split==split)&(details.k==k)&(details.gold_mode=='full')].set_index('query_id') for seed in ('seed_760','seed_761','seed_762')}
            for qid in baseline.index:
                values={seed:table.loc[qid,'f1']-baseline.loc[qid,'f1'] for seed,table in seeds.items()}
                consistency.append({'split':split,'k':k,'query_id':qid,'all_three_improve_f1':all(v>1e-12 for v in values.values()),
                                    'at_least_one_improves_f1':any(v>1e-12 for v in values.values()),'at_least_one_worsens_f1':any(v< -1e-12 for v in values.values()),
                                    'all_three_tie_f1':all(abs(v)<=1e-12 for v in values.values()),
                                    **{seed+'_delta_f1':value for seed,value in values.items()}})
            for seed,table in seeds.items():
                for qid in baseline.index:pairs.append({'split':split,'k':k,'seed':seed,'query_id':qid,
                                                        'delta_f1':table.loc[qid,'f1']-baseline.loc[qid,'f1'],
                                                        'delta_recall':table.loc[qid,'recall']-baseline.loc[qid,'recall']})
    pd.DataFrame(spread).to_csv(out/'seed_variability.csv',index=False)
    pd.DataFrame(consistency).to_csv(out/'seed_query_consistency.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'seed_pool_paired_deltas.csv',index=False)
    ceiling=[]
    for split in ('eval','official_val'):
        ids=queries[queries.split==split].query_id.tolist()
        pred={qid:pool[pool.query_id==qid].citation.tolist() for qid in ids}
        metric,_=study.evaluate_predictions(pred,{qid:study.GOLD[qid] for qid in ids})
        ceiling.append({'split':split,'queries':len(ids),'full_pool_citation_macro_recall_ceiling':metric['macro_recall'],
                        'encoder_independent':True,'gold_injected':False})
    pd.DataFrame(ceiling).to_csv(out/'seed_pool_coverage_ceiling.csv',index=False)
    print(metrics[(metrics.gold_mode=='full')&(metrics.k==10)][['system','split','macro_f1','macro_recall','macro_precision']].to_string(index=False))

if __name__=='__main__':main()
