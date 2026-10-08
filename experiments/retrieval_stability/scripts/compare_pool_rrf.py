"""Add a same-pool deterministic RRF comparator and seed source breakdown."""
import argparse
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze_stability as study

ROOT=study.ROOT

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/study.CONFIG['default_output'])
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);study.OUT=out
    queries=pd.read_csv(ROOT/'inputs/seed_queries.csv');qids=set(queries.query_id)
    raw=[]
    for branch in ('bm25','dense_full'):
        for file in (branch+'_train_candidates_top100.csv',branch+'_candidates_top100.csv'):
            data=pd.read_csv(ROOT/'inputs/historical'/file,keep_default_na=False)
            data=data[data.query_id.isin(qids)].copy();data['source_branch']=branch
            raw.append(data[['query_id','citation','rank','source_branch']])
    raw=pd.concat(raw);raw['first_stage_rrf']=1/(60+raw['rank'])
    frame=raw.groupby(['query_id','citation'],as_index=False).first_stage_rrf.sum()
    frame=frame.sort_values(['query_id','first_stage_rrf','citation'],ascending=[True,False,True])
    frame['rank']=frame.groupby('query_id').cumcount()+1
    frame.to_csv(out/'seed_pool_rrf_rankings.csv',index=False)
    for split in ('eval','official_val'):
        ids=queries[queries.split==split].query_id.tolist()
        for k in (10,20,50,100):
            pred=study.candidate_predictions(frame[frame.query_id.isin(ids)],k)
            for mode in ('full','covered'):study.evaluate('Pool_RRF',pred,ids,split,k,'New deterministic same-pool RRF scoring',scope='conditional_candidate_rescoring',gold_mode=mode)
    pd.DataFrame(study.SUMMARY).to_csv(out/'seed_pool_rrf_baseline_metrics.csv',index=False)
    pd.concat(study.DETAILS).to_csv(out/'seed_pool_rrf_per_query.csv',index=False)
    counts=[]
    for label in ('base','seed_760','seed_761','seed_762'):
        ranked=pd.read_csv(ROOT/'results'/f'seed_pool_{label}_rankings.csv',keep_default_na=False)
        for qid,group in ranked.groupby('query_id'):
            for k in (10,100):
                top=group[group['rank']<=k]
                counts.append({'encoder':label,'query_id':qid,'split':study.BY_ID[qid]['split'],'k':k,
                               'law':int((top.source_type=='law').sum()),'court':int((top.source_type=='court').sum()),
                               'scope':'Frozen pool source composition, not complete-corpus retrieval bias'})
    pd.DataFrame(counts).to_csv(out/'seed_pool_sources_per_query.csv',index=False)
    print(pd.DataFrame(study.SUMMARY).query('gold_mode=="full" and k==10')[['split','macro_f1','macro_recall']].to_string(index=False))

if __name__=='__main__':main()
