"""Matched historical grids and new fixed-alpha cutoff sensitivity, without CE inference."""
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
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'inputs/historical'
    ft=pd.read_csv(source/'ce_bge_ft_grid_summary.csv');zero=pd.read_csv(source/'ce_bge_zeroshot_grid_summary.csv')
    keys=['sources','first_stage_k','alpha','cutoff_mode','cutoff_value']
    matched=ft.merge(zero,on=keys,suffixes=('_ft','_zero'),validate='one_to_one')
    assert len(matched)==79
    for metric in ('macro_precision','macro_recall','macro_f1'):matched['delta_'+metric]=matched[metric+'_ft']-matched[metric+'_zero']
    matched.to_csv(out/'ce_matched_parameter_grid.csv',index=False)
    near=matched[(matched.alpha.isin([.6,.8,1.]))&(matched.cutoff_mode=='relative')&(matched.cutoff_value.isin([.5,.6,.7]))]
    near.to_csv(out/'ce_near_best_parameter_grid.csv',index=False)
    parameter=[]
    for name,frame in [('BM25',pd.read_csv(source/'bm25_final_grid_summary.csv')),('Dense',pd.read_csv(source/'dense_full_grid_summary.csv')),
                       ('BGE_zero',zero),('BGE_ft',ft),('Mini',pd.read_csv(source/'ce_mini_grid_summary.csv'))]:
        best=frame.macro_f1.max()
        parameter.append({'system':name,'historical_settings':len(frame),'best_f1':best,
                          'settings_at_95pct_best':int((frame.macro_f1>=.95*best-1e-12).sum()),
                          'settings_at_90pct_best':int((frame.macro_f1>=.9*best-1e-12).sum()),
                          'f1_median_all_settings':frame.macro_f1.median(),'f1_min_all_settings':frame.macro_f1.min(),
                          'scope':'Historical same-ten-query selection grid; not independent performance distribution'})
    pd.DataFrame(parameter).to_csv(out/'parameter_robustness_summary.csv',index=False)
    ids=sorted(q for q in study.GOLD if q.startswith('val_'));gold={q:study.GOLD[q] for q in ids}
    current=pd.read_csv(source/'ce_bge_ft_candidates_top100.csv',keep_default_na=False).sort_values(['query_id','rank'])
    rows=[];predictions={};setting=0
    for mode,values in [('topk',[5,10,20,30,50]),('relative',[.5,.55,.6,.65,.7,.75,.8,.85,.9,.95])]:
        for value in values:
            pred=study.CE['select_citations'](current,mode,value,1,50)
            summary,detail=study.evaluate_predictions(pred,gold)
            key=f'{mode}:{value:g}';predictions[key]=detail.set_index('query_id')
            rows.append({'setting':key,'mode':mode,'value':value,'alpha':.8,**summary,'average_output':detail.n_predicted.mean()})
    pd.DataFrame(rows).to_csv(out/'new_same_alpha_cutoff_grid.csv',index=False)
    selection=[]
    for held_out in ids:
        fitting=[q for q in ids if q!=held_out]
        # Deterministic tie rule; these settings were fixed before checking new scores.
        ordered=sorted(predictions,key=lambda key:(-predictions[key].loc[fitting,'f1'].mean(),-predictions[key].loc[fitting,'recall'].mean(),key))
        chosen=ordered[0]
        selection.append({'held_out_query':held_out,'selected_on_nine':chosen,
                          'nine_query_f1':predictions[chosen].loc[fitting,'f1'].mean(),
                          'held_out_f1':predictions[chosen].loc[held_out,'f1'],
                          'frozen_relative_0_6_held_out_f1':predictions['relative:0.6'].loc[held_out,'f1'],
                          'scope':'Cutoff label-withholding diagnostic only; model and alpha already selected with all ten queries. NOT retraining cross-validation or independent test.'})
    pd.DataFrame(selection).to_csv(out/'cutoff_nine_query_selection_diagnostic.csv',index=False)
    no_ce=zero[zero.alpha==0].sort_values(['macro_f1','macro_recall','macro_precision'],ascending=False).iloc[0]
    matched_zero=matched[matched.alpha==0]
    assert np.max(np.abs(matched_zero.delta_macro_f1))<1e-12
    records={'matched_settings':79,'ft_better_same_rule_settings':int((matched.delta_macro_f1>1e-12).sum()),
             'ft_same_settings':int((matched.delta_macro_f1.abs()<=1e-12).sum()),'ft_worse_settings':int((matched.delta_macro_f1< -1e-12).sum()),
             'near_best_nine_cells_f1_min':near.macro_f1_ft.min(),'near_best_nine_cells_f1_max':near.macro_f1_ft.max(),
             'best_no_ce_historical':{'macro_f1':no_ce.macro_f1,'cutoff_mode':no_ce.cutoff_mode,'cutoff_value':no_ce.cutoff_value},
             'nine_query_cutoff_choices':pd.Series([r['selected_on_nine'] for r in selection]).value_counts().to_dict(),
             'nine_query_selection_mean_held_out_f1':float(np.mean([r['held_out_f1'] for r in selection])),
             'selection_bias_corrected':False,
             'score_export_limit':'CE files only retain Top-100 after selected ranking. Full-input CE scores unavailable. Alpha grid statistics remain historical; new cutoff analysis uses stored full-input-normalized final_score at alpha 0.8.'}
    (out/'parameter_analysis_notes.json').write_text(json.dumps(records,indent=2),encoding='utf8')
    print(json.dumps(records,indent=2))

if __name__=='__main__':main()
