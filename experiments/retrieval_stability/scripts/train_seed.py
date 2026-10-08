"""Train one E5 seed with frozen archive samples; never encode the full corpus."""
from pathlib import Path
import argparse, json, os, sys, time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import archived_experiment as e5

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,choices=[761,762],required=True);p.add_argument('--resume',action='store_true');args=p.parse_args()
    assert 'faiss' not in sys.modules and 'KMP_DUPLICATE_LIB_OK' not in os.environ
    run=(ROOT/e5.read_json(ROOT/'config.json')['seed_artifact_root']).resolve()/'seed_runs'/f'seed_{args.seed}'
    c=json.loads((run/'config.json').read_text(encoding='utf8'))
    protocol=json.loads((run/'protocol.json').read_text(encoding='utf8'))
    assert e5.sha(run/'data/train_samples.jsonl')==protocol['samples_sha256']
    assert c['seed']==args.seed and c['epochs']==2
    if (run/'models/training_complete.json').exists() and (run/'models/epoch_2/model.safetensors').exists():
        print('Training already complete; preserving checkpoint');return
    original_read=e5.read_json
    base_location=(ROOT/original_read(ROOT/'config.json')['e5_archive']).resolve()/'models/base'
    def relocated_read(path):
        record=original_read(path)
        if Path(path).resolve()==(run/'models/base.json').resolve():
            record={**record,'path':str(base_location)}
        return record
    # Runtime path override only: preserve external configs, protocol hashes and resume locks.
    e5.read_json=relocated_read
    e5.ROOT=run;started=time.perf_counter()
    e5.train(c,resume=args.resume)
    (run/'training_status.json').write_text(json.dumps({'seed':args.seed,'exit_code':0,'wall_seconds':time.perf_counter()-started,
                                                       'scope':'Real full training on frozen samples; no full-corpus index'},indent=2),encoding='utf8')

if __name__=='__main__':main()
