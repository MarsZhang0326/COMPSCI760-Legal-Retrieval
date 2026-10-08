"""Freeze a real BM25/E5 candidate pool without adding any gold citations."""
import hashlib, json, sqlite3
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    c=json.loads((ROOT/'config.json').read_text(encoding='utf8'));archive=(ROOT/c['e5_archive']).resolve()
    rows=json.loads((ROOT/'inputs/queries.json').read_text(encoding='utf8'))
    rows=[r for r in rows if r['split'] in ('eval','official_val')]
    qids={r['query_id'] for r in rows}
    pd.DataFrame([{'query_id':r['query_id'],'query':r['query'],'split':r['split']} for r in rows]).to_csv(ROOT/'inputs/seed_queries.csv',index=False)
    frames=[];sources=[]
    for branch in ('bm25','dense_full'):
        for name in (branch+'_train_candidates_top100.csv',branch+'_candidates_top100.csv'):
            file=ROOT/'inputs/historical'/name
            frame=pd.read_csv(file,keep_default_na=False);frame=frame[frame.query_id.isin(qids)&(frame['rank']<=100)].copy()
            frame['candidate_branch']=branch;frames.append(frame[['query_id','doc_id','citation','source_type','candidate_branch','rank']])
            sources.append({'file':name,'sha256':sha(file)})
    raw=pd.concat(frames,ignore_index=True)
    pool=raw.drop_duplicates(['query_id','citation','doc_id']).copy()
    pool=pool.sort_values(['query_id','doc_id','citation']).reset_index(drop=True)
    assert set(pool.query_id)==qids and len(qids)==124
    conn=sqlite3.connect((archive/'data/corpus.sqlite').resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    documents=[]
    for doc_id in sorted(set(pool.doc_id)):
        doc=conn.execute('SELECT id,doc_id,citation,source_type,title,text FROM docs WHERE doc_id=?',(doc_id,)).fetchone()
        assert doc is not None
        did,doc_id,citation,source,title,text=doc
        documents.append({'id':did,'doc_id':doc_id,'citation':citation,'source_type':source,
                          'input':'passage: '+' '.join(str(x).strip() for x in (citation,title,text) if x)})
    conn.close()
    docs=pd.DataFrame(documents)
    check=pool.merge(docs[['doc_id','citation','source_type']],on='doc_id',suffixes=('','_actual'))
    assert (check.citation==check.citation_actual).all() and (check.source_type==check.source_type_actual).all()
    pool.to_csv(ROOT/'inputs/seed_candidate_pool.csv',index=False)
    docs.to_json(ROOT/'inputs/seed_passages.jsonl',orient='records',lines=True,force_ascii=False)
    counts=pool.groupby('query_id').agg(passage_pairs=('doc_id','size'),unique_citations=('citation','nunique')).reset_index()
    counts.to_csv(ROOT/'results/seed_candidate_budgets.csv',index=False)
    record={'queries':124,'independent_e5_eval_queries':114,'historical_validation_queries':10,
            'unique_documents':len(docs),'candidate_pairs':len(pool),'gold_injected':False,
            'candidate_budget':'Exactly the union of two historical Top-100 branches, maximum 200 per query; same pool for every encoder',
            'final_budgets':[10,20,50,100],'sources':sources,'pool_sha256':sha(ROOT/'inputs/seed_candidate_pool.csv'),
            'passages_sha256':sha(ROOT/'inputs/seed_passages.jsonl'),
            'query_sha256':sha(ROOT/'inputs/seed_queries.csv'),
            'scope':'New conditional candidate rescoring experiment, NOT full-corpus retrieval by the new seeds'}
    (ROOT/'inputs/seed_pool_record.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
