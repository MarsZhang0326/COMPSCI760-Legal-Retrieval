"""PyTorch-only pretrained E5 query encoding worker; FAISS is never imported."""
import argparse, hashlib, importlib.metadata, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def model_identity(path):
    files=sorted(list(path.glob('*.safetensors'))+list(path.glob('pytorch_model*.bin')))
    assert files,'Missing encoder weights'
    identity={p.name:sha(p) for p in files}; identity['config']=sha(path/'config.json')
    return hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent); args=p.parse_args()
    root=args.root.resolve(); c=json.loads((root/'config.json').read_text())
    assert 'KMP_DUPLICATE_LIB_OK' not in os.environ
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel,AutoTokenizer
    assert 'faiss' not in __import__('sys').modules
    torch.manual_seed(c['seed']); torch.set_num_threads(c['faiss_threads'])
    model_dir=(root/c['model_dir']).resolve()
    manifest=json.loads((root/c['index_dir']/'manifest.json').read_text())
    identity=model_identity(model_dir); assert identity==manifest['model_sha256']
    device='cuda' if c['device']=='auto' and torch.cuda.is_available() else 'cpu' if c['device']=='auto' else c['device']
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    model=AutoModel.from_pretrained(model_dir,local_files_only=True).to(device).eval()
    # Float32 encoding is fixed for every view; no E5 fine-tuning takes place.
    frame=pd.read_csv(root/c['query_file'],keep_default_na=False)
    text=['query: '+q for q in frame['query']]
    token_lengths=[len(x) for x in tokenizer(text,truncation=False)['input_ids']]
    assert max(token_lengths)<=c['max_length'],'A frozen variant would lose facts to truncation'
    records=[]; vectors=[]
    with torch.no_grad():
        # Same per-query execution rule for every view; record actual individual latency.
        for i,input_text in enumerate(text):
            if device=='cuda': torch.cuda.synchronize()
            started=time.perf_counter()
            batch=tokenizer([input_text],padding=True,truncation=True,max_length=c['max_length'],return_tensors='pt')
            batch={k:v.to(device) for k,v in batch.items()}
            hidden=model(**batch).last_hidden_state
            mask=batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            vector=F.normalize((hidden*mask).sum(1)/mask.sum(1).clamp_min(1),p=2,dim=1).float().cpu().numpy()
            if device=='cuda': torch.cuda.synchronize()
            vectors.append(vector)
            records.append({'query_id':frame.iloc[i].query_id,'variant':frame.iloc[i].variant,
                            'input_tokens':token_lengths[i],'truncated':False,'encode_seconds':time.perf_counter()-started,'failure':False})
    matrix=np.concatenate(vectors).astype(np.float32)
    assert matrix.shape==(30,768) and np.isfinite(matrix).all()
    np.save(root/'inputs/query_vectors.npy',matrix,allow_pickle=False)
    frame[['query_id','variant']].to_csv(root/'inputs/query_vector_rows.csv',index=False)
    pd.DataFrame(records).to_csv(root/'results/encoding_times.csv',index=False)
    metadata={'device':'configured',
              'model_sha256':identity,'encoder_revision':c['encoder_revision'],'max_tokens':max(token_lengths),
              'vectors_sha256':sha(root/'inputs/query_vectors.npy'),'query_file_sha256':sha(root/c['query_file']),
              'dtype':'float32; no autocast','actual_query_batch_size':1,
              'timing_policy':'One query per forward pass for per-query timing; first query includes warm-up',
              'faiss_imported':False,'norm_min':float(np.linalg.norm(matrix,axis=1).min()),
              'dependencies':{n:importlib.metadata.version(n) for n in ('torch','transformers','numpy','pandas')},
              'tokenizer_sha256':sha(model_dir/'tokenizer.json')}
    (root/'inputs/encoder_record.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__': main()
