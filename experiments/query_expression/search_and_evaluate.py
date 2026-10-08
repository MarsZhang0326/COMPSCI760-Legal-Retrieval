"""Full archived-corpus exact search for frozen E5 query-expression views.

No PyTorch import, no corpus encoding, and no Cross-Encoder scores. Restart
resumes after the last saved shard. All published retrieval metrics are newly
computed with the frozen shared citation evaluator.
"""
import argparse, hashlib, importlib.util, json, os, sqlite3, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import faiss

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def dedup(ids,scores,lookup,limit=100):
    seen=set(); hits=[]
    for row_rank,(did,score) in enumerate(zip(ids,scores),1):
        doc=lookup(int(did)); citation=doc['citation']
        if not citation or citation in seen: continue
        seen.add(citation)
        hits.append({**doc,'rank':len(hits)+1,'row_rank':row_rank,'dense_score':float(score)})
        if len(hits)==limit: break
    return hits

def rrf(left,right,k=60):
    scores={}; representatives={}
    for hits in (left,right):
        for h in hits:
            citation=h['citation']; scores[citation]=scores.get(citation,0)+1/(k+h['rank'])
            old=representatives.get(citation)
            if old is None or (-h['dense_score'],h['doc_id'])<(-old['dense_score'],old['doc_id']):
                representatives[citation]=h
    ordered=sorted(scores,key=lambda c:(-scores[c],-representatives[c]['dense_score'],representatives[c]['doc_id']))[:100]
    return [{**representatives[c],'rank':i+1,'fusion_score':scores[c]} for i,c in enumerate(ordered)]

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent); args=p.parse_args()
    root=args.root.resolve(); c=json.loads((root/'config.json').read_text())
    assert 'torch' not in sys.modules and 'KMP_DUPLICATE_LIB_OK' not in os.environ
    faiss.omp_set_num_threads(c['faiss_threads'])
    index_dir=(root/c['index_dir']).resolve(); mapping=(root/c['corpus_map']).resolve()
    manifest=json.loads((index_dir/'manifest.json').read_text()); complete=json.loads((index_dir/'complete.json').read_text())
    encoder=json.loads((root/'inputs/encoder_record.json').read_text())
    assert sha(index_dir/'manifest.json')==complete['manifest_sha256']
    assert manifest['model_sha256']==encoder['model_sha256']
    assert manifest['corpus']['rows']==complete['rows']==2652248
    assert manifest['pooling']=='attention-mask mean, L2 normalize' and manifest['dimension']==768
    assert manifest['format']=='passage: citation title text' and manifest['max_length']==512
    assert sha(root/c['query_file'])==encoder['query_file_sha256']
    assert sha(root/'inputs/query_vectors.npy')==encoder['vectors_sha256']
    vectors=np.load(root/'inputs/query_vectors.npy',allow_pickle=False)
    rows=pd.read_csv(root/'inputs/query_vector_rows.csv',keep_default_na=False)
    assert vectors.shape==(30,768) and np.isfinite(vectors).all() and np.max(np.abs(np.linalg.norm(vectors,axis=1)-1))<.003
    qids=sorted(rows.query_id.unique())
    arrays={v:vectors[rows.index[rows.variant==v].tolist()] for v in ('en','de','legal_original')}
    assert all(rows[rows.variant==v].query_id.tolist()==qids for v in arrays)
    arrays.update(en_deep2x=arrays['en'],en_repeat_component=arrays['en'])
    depth={v:c['deep_row_depth'] if v=='en_deep2x' else c['base_row_depth'] for v in arrays}
    state_path=root/'results/search_resume.npz'
    identity=sha(root/'config.json')+encoder['vectors_sha256']+complete['manifest_sha256']
    scores={v:np.empty((10,0),np.float32) for v in arrays}; ids={v:np.empty((10,0),np.int64) for v in arrays}
    times={v:np.zeros(10) for v in arrays}; next_shard=0; common_io=0.0
    if state_path.exists():
        state=np.load(state_path,allow_pickle=False)
        assert str(state['identity'])==identity
        next_shard=int(state['next_shard']); common_io=float(state['common_io'])
        for v in arrays: scores[v]=state[v+'_scores'];ids[v]=state[v+'_ids'];times[v]=state[v+'_times']
    invocation_start_shard=next_shard
    run_started=time.perf_counter()
    for number,item in enumerate(complete['shards']):
        if number<next_shard: continue
        start=time.perf_counter(); file=index_dir/item['file']
        assert sha(file)==item['sha256'],file.name
        idx=faiss.read_index(str(file))
        lo,hi=map(int,Path(item['file']).stem.split('_')[1:]); assert idx.ntotal==hi-lo and idx.d==768
        assert idx.metric_type==faiss.METRIC_INNER_PRODUCT
        common_io+=time.perf_counter()-start
        for v,matrix in arrays.items():
            ds=[];di=[]
            for i in range(10):
                started=time.perf_counter()
                d,indices=idx.search(matrix[i:i+1],min(depth[v],idx.ntotal))
                times[v][i]+=time.perf_counter()-started
                ds.append(d[0]);di.append(indices[0])
            local_scores=np.asarray(ds);local_ids=np.asarray(di)
            merged_s=np.concatenate([scores[v],local_scores],axis=1)
            merged_i=np.concatenate([ids[v],local_ids],axis=1)
            started=time.perf_counter()
            order=np.lexsort((merged_i,-merged_s),axis=1)[:,:depth[v]]
            scores[v]=np.take_along_axis(merged_s,order,axis=1)
            ids[v]=np.take_along_axis(merged_i,order,axis=1)
            times[v]+=((time.perf_counter()-started)/10)
        del idx
        if (number+1)%10==0 or number+1==len(complete['shards']):
            state={'identity':np.asarray(identity),'next_shard':np.asarray(number+1),'common_io':np.asarray(common_io)}
            for v in arrays:state[v+'_scores']=scores[v];state[v+'_ids']=ids[v];state[v+'_times']=times[v]
            temp=state_path.with_suffix('.npz.tmp')
            with open(temp,'wb') as f:np.savez(f,**state)
            os.replace(temp,state_path)
            print(f'Searched and checked {number+1}/{len(complete["shards"])} shards, all five views/controls',flush=True)
    snapshot=mapping.parent/'corpus_snapshot_sha256.json'
    expected=json.loads(snapshot.read_text()); assert expected['sha256']==c['expected_mapping_sha256']; assert expected['input_manifest_sha256']==manifest['corpus']['input_manifest_sha256']; assert mapping.stat().st_size==expected['bytes'] and sha(mapping)==expected['sha256']
    conn=sqlite3.connect(mapping.as_uri()+'?mode=ro&immutable=1',uri=True);conn.row_factory=sqlite3.Row
    cache={}
    def lookup(did):
        if did not in cache:
            record=conn.execute('SELECT id,doc_id,citation,source_type FROM docs WHERE id=?',(did,)).fetchone()
            assert record is not None
            cache[did]=dict(record)
        return cache[did]
    pools={}; candidates=[]; runtime=[]
    enc=pd.read_csv(root/'results/encoding_times.csv')
    for qi,qid in enumerate(qids):
        local={}
        for v in arrays:
            started=time.perf_counter();hits=dedup(ids[v][qi],scores[v][qi],lookup)
            assert len(hits)==100,(qid,v,len(hits))
            local[v]=hits
            elapsed=time.perf_counter()-started
            encv='en' if v in ('en_deep2x','en_repeat_component') else v
            encode_seconds=float(enc[(enc.query_id==qid)&(enc.variant==encv)].encode_seconds.iloc[0])
            runtime.append({'query_id':qid,'variant':v,'query_embeddings':1,'full_search_passes':1,'raw_row_budget':depth[v],
                            'encoding_seconds':encode_seconds,'search_merge_seconds':float(times[v][qi]),'dedup_fusion_seconds':elapsed,
                            'unique_candidates':100,'failure':False})
        start=time.perf_counter();local['en_de_rrf']=rrf(local['en'],local['de'],c['rrf_k']);rrftime=time.perf_counter()-start
        start=time.perf_counter();local['en_repeat2x']=rrf(local['en'],local['en_repeat_component'],c['rrf_k']);repeattime=time.perf_counter()-start
        for v in ('en_de_rrf','en_repeat2x'):
            views=('en','de') if v=='en_de_rrf' else ('en','en_repeat_component')
            component=[r for r in runtime if r['query_id']==qid and r['variant'] in views]
            runtime.append({'query_id':qid,'variant':v,'query_embeddings':2 if v=='en_de_rrf' else 1,'full_search_passes':2,
                            'raw_row_budget':2000,'encoding_seconds':sum(r['encoding_seconds'] for r in component) if v=='en_de_rrf' else component[0]['encoding_seconds'],
                            'search_merge_seconds':sum(r['search_merge_seconds'] for r in component),
                            'dedup_fusion_seconds':rrftime if v=='en_de_rrf' else repeattime,'unique_candidates':100,'failure':False})
        for v in c['variants']:
            pools[(qid,v)]=local[v]
            candidates.extend({'query_id':qid,'variant':v,**hit} for hit in local[v])
    conn.close()
    frame=pd.DataFrame(candidates)
    frame.to_csv(root/'results/all_candidates.csv',index=False)
    pd.DataFrame(runtime).to_csv(root/'results/retrieval_times.csv',index=False)
    spec=importlib.util.spec_from_file_location('frozen_evaluation',Path(__file__).resolve().parent/'evaluation.py')
    evaluation=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluation)
    gold={qid:set(labels) for qid,labels in json.loads((root/c['gold_file']).read_text()).items()}
    conn=sqlite3.connect(mapping.as_uri()+'?mode=ro&immutable=1',uri=True)
    sources={citation:conn.execute('SELECT source_type FROM docs WHERE citation=? LIMIT 1',(citation,)).fetchone()[0]
             for labels in gold.values() for citation in labels}
    conn.close(); summary=[]; details=[]
    for v in c['variants']:
        subset=frame[frame.variant==v].copy()
        subset[['query_id','rank','doc_id','citation','source_type','row_rank','dense_score']].to_csv(root/f'results/query_expr_{v}_candidates_top100.csv',index=False)
        for k in c['cutoffs']:
            predictions={qid:[hit['citation'] for hit in pools[(qid,v)][:k]] for qid in gold}
            metric,detail=evaluation.evaluate_predictions(predictions,gold)
            record={'variant':v,'split':'official_val','gold_mode':'full','k':k,**metric,
                    'origin':'New full-corpus retrieval in this query-expression experiment','candidate_budget':100}
            for source in ('law','court'):
                eligible={qid:{g for g in labels if sources[g]==source} for qid,labels in gold.items()}
                eligible={qid:labels for qid,labels in eligible.items() if labels}
                hits=sum(len(set(predictions[qid])&labels) for qid,labels in eligible.items())
                n=sum(map(len,eligible.values()))
                record[source+'_gold_occurrences']=n;record[source+'_hits']=hits;record[source+'_micro_recall']=hits/n
                record[source+'_macro_recall']=sum(len(set(predictions[qid])&labels)/len(labels) for qid,labels in eligible.items())/len(eligible)
                record[source+'_candidate_share']=float((subset[subset['rank']<=k].source_type==source).mean())
            summary.append(record);details.append(detail.assign(variant=v,split='official_val',gold_mode='full',k=k))
            pd.DataFrame([{'query_id':qid,'predicted_citations':';'.join(predictions[qid])} for qid in sorted(predictions)]).to_csv(
                root/f'results/query_expr_{v}_predictions_top{k}.csv',index=False)
    pd.DataFrame(summary).to_csv(root/'results/metrics.csv',index=False)
    detail=pd.concat(details,ignore_index=True);detail.to_csv(root/'results/per_query_metrics.csv',index=False)
    comparison=[]
    for v in c['variants']:
        values={r['k']:r for r in summary if r['variant']==v}
        comparison.append({'variant':v,'queries':10,'gold_mode':'full',
                           'recall20':values[20]['macro_recall'],'recall50':values[50]['macro_recall'],
                           'recall100':values[100]['macro_recall'],'precision10':values[10]['macro_precision'],
                           'recall10':values[10]['macro_recall'],'macro_f1_10':values[10]['macro_f1'],
                           'law_hits100':values[100]['law_hits'],'court_hits100':values[100]['court_hits']})
    pd.DataFrame(comparison).to_csv(root/'results/main_comparison.csv',index=False)
    record={'complete':True,'shards_searched':len(complete['shards']),'corpus_rows':complete['rows'],'model_sha256':manifest['model_sha256'],
            'manifest_sha256':sha(index_dir/'manifest.json'),'mapping_sha256':expected['sha256'],'torch_imported':False,
            'common_shard_checksum_and_io_seconds':common_io,'current_invocation_wall_seconds':time.perf_counter()-run_started,
            'cross_encoder_executed':False,'query_file_sha256':encoder['query_file_sha256'],
            'invocation_start_shard':invocation_start_shard,
            'full_search_performed_this_invocation':invocation_start_shard==0}
    (root/'results/search_record.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    if not (root/'results/search_record_first_complete.json').exists():
        (root/'results/search_record_first_complete.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print(pd.DataFrame(summary).query('k == 100')[['variant','macro_recall','law_hits','court_hits','law_candidate_share']].to_string(index=False),flush=True)

if __name__=='__main__':main()
