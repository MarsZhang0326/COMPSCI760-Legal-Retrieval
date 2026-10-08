"""Reuse matching old query/document vectors in a FAISS-only process."""
import hashlib, json, re, sys
from pathlib import Path
import faiss
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    assert 'torch' not in sys.modules
    c=json.loads((ROOT/'config.json').read_text(encoding='utf8'));archive=(ROOT/c['e5_archive']).resolve()
    docs=pd.read_json(ROOT/'inputs/seed_passages.jsonl',lines=True)
    queries=pd.read_csv(ROOT/'inputs/seed_queries.csv',keep_default_na=False)
    wanted={r.query_id:'query: '+re.sub(r'\s+',' ',r.query).strip() for r in queries.itertuples(index=False)}
    out=ROOT/'inputs/seed_embeddings';out.mkdir(exist_ok=True)
    faiss.omp_set_num_threads(4)
    for label,index_name in (('base','base'),('seed_760','epoch_2')):
        folder=archive/'indexes'/index_name
        manifest=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
        complete=json.loads((folder/'complete.json').read_text(encoding='utf8'))
        assert sha(folder/'manifest.json')==complete['manifest_sha256']
        assert manifest['corpus']['rows']==2652248
        identity=manifest['model_sha256']
        found=[]
        for file in (archive/'data/query_embeddings').glob('*.json'):
            info=json.loads(file.read_text(encoding='utf8'))
            if 'vectors_sha256' not in info or info['identity']['model_sha256']!=identity:continue
            cache={r['query_id']:r['input'] for r in info['identity']['queries']}
            if all(cache.get(q)==text for q,text in wanted.items()):found.append((file,info))
        assert found,'No compatible full query cache'
        file,info=found[0];vector_file=file.with_suffix('.npy');assert sha(vector_file)==info['vectors_sha256']
        old=np.load(vector_file,allow_pickle=False)
        position={r['query_id']:i for i,r in enumerate(info['identity']['queries'])}
        qvec=old[[position[qid] for qid in queries.query_id]].copy()
        dvec=np.empty((len(docs),768),np.float32);covered=np.zeros(len(docs),bool)
        for number,shard in enumerate(complete['shards']):
            lo,hi=map(int,Path(shard['file']).stem.split('_')[1:])
            mask=(docs.id>=lo)&(docs.id<hi)
            if not mask.any():continue
            path=folder/shard['file'];assert sha(path)==shard['sha256']
            idx=faiss.read_index(str(path));assert idx.ntotal==hi-lo and idx.d==768
            for position in np.flatnonzero(mask):
                dvec[position]=idx.reconstruct(int(docs.iloc[position].id))
                covered[position]=True
            del idx
            if number%40==0:print(label,number,flush=True)
        assert covered.all() and np.isfinite(dvec).all()
        assert np.max(np.abs(np.linalg.norm(dvec,axis=1)-1))<.003
        np.save(out/(label+'_queries.npy'),qvec,allow_pickle=False)
        np.save(out/(label+'_documents.npy'),dvec,allow_pickle=False)
        record={'model_sha256':identity,'model':label,'document_vectors_reconstructed':len(dvec),
                'query_vectors_reused':len(qvec),'new_encoding':False,'torch_imported':False,
                'index_manifest_sha256':complete['manifest_sha256'],'query_cache':str(file),
                'document_order_sha256':sha(ROOT/'inputs/seed_passages.jsonl'),
                'query_order_sha256':sha(ROOT/'inputs/seed_queries.csv'),
                'documents_vectors_sha256':sha(out/(label+'_documents.npy')),
                'queries_vectors_sha256':sha(out/(label+'_queries.npy'))}
        (out/(label+'_record.json')).write_text(json.dumps(record,indent=2),encoding='utf8')
        print(json.dumps(record),flush=True)

if __name__=='__main__':main()
