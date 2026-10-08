"""Query-family influence and citation-unseen diagnostics of real saved retrieval."""
import argparse
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze_stability as study

ROOT=study.ROOT

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/study.CONFIG['default_output'])
    args=parser.parse_args();r=args.output_dir.resolve();r.mkdir(parents=True,exist_ok=True)
    reference=r if (r/'paired_query_deltas.csv').exists() else ROOT/'results'
    pairs=pd.read_csv(reference/'paired_query_deltas.csv')
    influence=[]
    for name,group in pairs.groupby('comparison'):
        total=group.delta.sum();n=len(group)
        for gid,family in group.groupby('group_id'):
            if len(family)==n:continue
            value=(total-family.delta.sum())/(n-len(family))
            influence.append({'comparison':name,'removed_scenario':gid,'removed_queries':len(family),
                              'remaining_queries':n-len(family),'delta_without_scenario':value,'full_delta':total/n,
                              'direction_reversed':bool(value*(total/n)<-1e-12),'gain_eliminated':abs(value)<1e-12})
    pd.DataFrame(influence).to_csv(r/'leave_one_scenario_out.csv',index=False)
    archive=study.ARCHIVE
    membership=json.loads((ROOT/'inputs/training_citation_membership.json').read_text(encoding='utf8'))
    positive_citations=set(membership['positive_citations'])
    negative_citations=set(membership['negative_citations'])
    eval_ids=[q for q,record in study.BY_ID.items() if record['split']=='eval']
    gold={qid:study.GOLD[qid] for qid in eval_ids}
    labels=set().union(*gold.values());covered=labels&set(study.COVERED)
    record={'eval_queries':len(eval_ids),'unique_eval_gold':len(labels),'unique_covered_eval_gold':len(covered),
            'training_positive_citations':len(positive_citations),
            'eval_covered_gold_citations_seen_in_training_positives':len(covered&positive_citations),
            'eval_covered_gold_citations_unseen_in_training_positives':len(covered-positive_citations),
            'eval_covered_gold_absent_positive_but_seen_as_negative':len((covered-positive_citations)&negative_citations),
            'eval_covered_gold_absent_both_positive_and_negative':len(covered-positive_citations-negative_citations),
            'unseen_definition':'Absent from positive supervision only; some citations were encoded as negatives. Not a guarantee of absence from all training text or multilingual E5 pretraining.',
            'query_independence_vs_citation_independence':'Unseen questions can still target previously supervised citations; same-corpus label reuse is allowed but does not prove unseen-citation transfer.'}
    result=[]
    for system,file in [('E5_base','final_original_E5_candidates.csv'),('E5_ft','final_finetuned_E5_candidates.csv')]:
        frame=pd.read_csv(archive/'results'/file,keep_default_na=False)
        for k in [10,20,50,100]:
            pred=study.candidate_predictions(frame[frame.query_id.isin(eval_ids)],k)
            for category in ('seen_positive_citation','unseen_positive_citation'):
                eligible={q:{label for label in labels if label in study.COVERED and ((label in positive_citations)==(category=='seen_positive_citation'))}
                          for q,labels in gold.items()}
                eligible={q:labels for q,labels in eligible.items() if labels}
                for source in ('all','law','court'):
                    selected={q:{label for label in labels if source=='all' or source in study.COVERED[label]} for q,labels in eligible.items()}
                    selected={q:labels for q,labels in selected.items() if labels}
                    n=sum(map(len,selected.values()));hits=sum(len(set(pred[q])&labels) for q,labels in selected.items())
                    result.append({'system':system,'k':k,'category':category,'source':source,'eligible_queries':len(selected),
                                   'gold_occurrences':n,'hits':hits,'micro_recall':hits/n if n else np.nan,
                                   'macro_recall':np.mean([len(set(pred[q])&labels)/len(labels) for q,labels in selected.items()]) if selected else np.nan,
                                   'scope':'Covered-gold subset recall diagnosis only; labels/questions excluded explicitly, not full-gold precision or F1'})
    pd.DataFrame(result).to_csv(r/'seen_unseen_citation_recall.csv',index=False)
    train=pd.read_csv(reference/'recomputed_per_query.csv')
    comparison=[]
    for k in (10,20,100):
        a=train[(train.system=='BM25')&(train.split=='original_train_frozen')&(train.gold_mode=='full')&(train.k.astype(str)==str(k))].set_index('query_id')
        b=train[(train.system=='Dense_pretrained_history')&(train.split=='original_train_frozen')&(train.gold_mode=='full')&(train.k.astype(str)==str(k))].set_index('query_id')
        difference=a.f1-b.f1
        comparison.append({'split':'original_train_frozen','queries':len(difference),'k':k,'metric':'f1','mean_BM25_minus_Dense':difference.mean(),
                           'BM25_better_queries':int((difference>1e-12).sum()),'same_queries':int((difference.abs()<=1e-12).sum()),
                           'Dense_better_queries':int((difference< -1e-12).sum()),
                           'scope':'Frozen historical candidates on original train labels; no project fine-tuning in either model. Historical run provenance limitations remain.'})
    pd.DataFrame(comparison).to_csv(r/'frozen_baseline_train_pairs.csv',index=False)
    (r/'generalization_diagnostic_record.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print(json.dumps(record,indent=2))
    print(pd.DataFrame(result).query('k==100 and source=="all"').to_string(index=False))

if __name__=='__main__':main()
