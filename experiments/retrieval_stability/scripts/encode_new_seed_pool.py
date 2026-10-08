"""Encode only the frozen diagnostic pool with the two new E5 checkpoints.

This PyTorch-only worker never imports FAISS or builds a full-corpus index.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, time
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import archived_experiment as e5

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,required=True);args=p.parse_args()
    assert 'faiss' not in sys.modules and 'KMP_DUPLICATE_LIB_OK' not in os.environ
    archive=(ROOT/e5.read_json(ROOT/'config.json')['e5_archive']).resolve()
    run=archive if args.seed==760 else (ROOT/e5.read_json(ROOT/'config.json')['seed_artifact_root']).resolve()/'seed_runs'/f'seed_{args.seed}'
    c=e5.read_json(run/'config.json');model_path=run/'models/epoch_2'
    assert e5.read_json(run/'models/training_complete.json')['epochs']==2
    torch.set_num_threads(c['threads']);torch.manual_seed(args.seed)
    expected=e5.read_json(ROOT/'inputs/seed_embeddings'/f'seed_{args.seed}_record.json')
    assert e5.model_identity(model_path)==expected['model_sha256'], 'Frozen checkpoint identity mismatch'
    model,tokenizer=e5.load_encoder(c,model_path)
    queries=pd.read_csv(ROOT/'inputs/seed_queries.csv',keep_default_na=False)
    docs=pd.read_json(ROOT/'inputs/seed_passages.jsonl',lines=True)
    record=e5.read_json(ROOT/'inputs/seed_pool_record.json')
    assert e5.sha(ROOT/'inputs/seed_queries.csv')==record['query_sha256']
    assert e5.sha(ROOT/'inputs/seed_passages.jsonl')==record['passages_sha256']
    out=ROOT/'inputs/seed_embeddings';out.mkdir(exist_ok=True)
    started=time.perf_counter()
    qvec=e5.encode(model,tokenizer,[e5.query_input(q) for q in queries['query']],c)
    # Stream chunks and preserve the same per-batch precision/ordering as archived encoding.
    chunks=[]
    for start in range(0,len(docs),256):
        chunks.append(e5.encode(model,tokenizer,docs['input'].iloc[start:start+256].tolist(),c))
        if start%2048==0:print(json.dumps({'seed':args.seed,'documents_complete':min(start+256,len(docs)),'documents_total':len(docs),'seconds':time.perf_counter()-started}),flush=True)
    dvec=np.concatenate(chunks).astype(np.float32)
    assert dvec.shape==(len(docs),768) and qvec.shape==(len(queries),768)
    assert np.isfinite(dvec).all() and np.isfinite(qvec).all()
    assert np.max(np.abs(np.linalg.norm(dvec,axis=1)-1))<.003
    label=f'seed_{args.seed}'
    if args.seed==760 and (out/(label+'_record.json')).exists():
        old=e5.read_json(out/(label+'_record.json'))
        if old.get('new_encoding') is False:
            for suffix in ('_queries.npy','_documents.npy','_record.json'):
                (out/(label+suffix)).rename(out/(label+'_archive'+suffix))
    np.save(out/(label+'_queries.npy'),qvec,allow_pickle=False)
    np.save(out/(label+'_documents.npy'),dvec,allow_pickle=False)
    metadata={'seed':args.seed,'epoch_policy':'Frozen final epoch 2; no new retrieval-dev checkpoint choice',
              'model_sha256':e5.model_identity(model_path),'document_vectors_newly_encoded':len(dvec),
              'queries_encoded':len(qvec),'faiss_imported':False,'seconds':time.perf_counter()-started,
              'document_order_sha256':e5.sha(ROOT/'inputs/seed_passages.jsonl'),
              'query_order_sha256':e5.sha(ROOT/'inputs/seed_queries.csv'),
              'documents_vectors_sha256':e5.sha(out/(label+'_documents.npy')),
              'queries_vectors_sha256':e5.sha(out/(label+'_queries.npy')),
              'scope':'Real fixed-pool rescoring inputs; no full-corpus index or candidate search. All three tuned seeds use identical frozen-pool batching.',
              'device':'configured','hardware_recorded':False}
    (out/(label+'_record.json')).write_text(json.dumps(metadata,indent=2),encoding='utf8')
    print(json.dumps(metadata),flush=True)

if __name__=='__main__':main()
