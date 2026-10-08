"""Recompute frozen artifacts and perform query-paired stability analysis.

No training, retrieval or Cross-Encoder inference occurs in this script.
Full-gold results never discard missing corpus labels. Covered-gold diagnostics
explicitly exclude queries with no covered gold. Confidence intervals condition
on saved systems/settings and do not remove selection bias.
"""
import argparse
from pathlib import Path
import ast, hashlib, importlib.metadata, itertools, json, re, sqlite3, sys, unicodedata
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluation import evaluate_predictions, parse_citations

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'config.json').read_text(encoding='utf8'))
ARCHIVE=(ROOT/CONFIG['statistics_archive']).resolve()
ROWS=json.loads((ROOT/'inputs/queries.json').read_text(encoding='utf8'))
BY_ID={r['query_id']:r for r in ROWS}
GOLD={qid:set(r['gold']) for qid,r in BY_ID.items()}
COVERED={citation:sources for citation,sources in json.loads((ROOT/'inputs/gold_sources.json').read_text(encoding='utf8')).items() if sources}
OUT=ROOT/'results'
SUMMARY=[];DETAILS=[];CHECKS=[]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def pure_functions(path,names,namespace=None):
    source=Path(path).read_text(encoding='utf8');tree=ast.parse(source)
    scope={'np':np,'pd':pd,'Path':Path,'re':re,'CUTOFF_MODES':['topk','threshold','relative']}
    scope.update(namespace or {})
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope

CE=pure_functions(ROOT/'scripts/archived_rerank_rules.py',
                  ['rank_citations','minmax_per_query','select_citations','read_candidate_file'])
EXPLICIT=pure_functions(ROOT/'scripts/archived_bm25_rules.py',['normalize_citation','extract_explicit_citations'])['extract_explicit_citations']

def candidate_predictions(frame,k):
    return {qid:group.sort_values('rank').head(k).citation.tolist() for qid,group in frame.groupby('query_id')}

def evaluate(label,predictions,ids,split,k,origin,scope='full_corpus',gold_mode='full'):
    ids=list(ids);excluded=sum(len(GOLD[q]-set(COVERED)) for q in ids)
    if gold_mode=='covered':
        gold={q:GOLD[q]&set(COVERED) for q in ids if GOLD[q]&set(COVERED)}
    else:gold={q:GOLD[q] for q in ids}
    predictions={q:predictions[q] for q in gold}
    metric,detail=evaluate_predictions(predictions,gold)
    record={'system':label,'split':split,'gold_mode':gold_mode,'k':k,'scope':scope,'origin':origin,**metric,
            'avg_predictions':float(detail.n_predicted.mean()),'zero_hit_queries':int((detail.tp==0).sum()),
            'excluded_queries':len(ids)-len(gold),'excluded_gold_occurrences':excluded if gold_mode=='covered' else 0}
    for source in ('law','court'):
        def source_of(citation):
            if citation in COVERED:return COVERED[citation][0]
            return 'law' if citation.startswith('Art.') else 'court'
        eligible={q:{c for c in labels if source_of(c)==source} for q,labels in gold.items()}
        eligible={q:labels for q,labels in eligible.items() if labels}
        count=sum(map(len,eligible.values()));hits=sum(len(set(predictions[q])&labels) for q,labels in eligible.items())
        record[source+'_gold']=count;record[source+'_hits']=hits
        record[source+'_micro_recall']=hits/count if count else np.nan
        record[source+'_macro_recall']=np.mean([len(set(predictions[q])&labels)/len(labels) for q,labels in eligible.items()]) if eligible else np.nan
    record['missing_gold_source_policy']='Art. prefix heuristic for absent labels only; covered labels use corpus metadata'
    SUMMARY.append(record)
    detail=detail.assign(system=label,split=split,gold_mode=gold_mode,k=k,scope=scope,origin=origin)
    detail['group_id']=detail.query_id.map(lambda q:BY_ID[q].get('group_id',q))
    DETAILS.append(detail)
    return record,detail

def query_audit():
    # Recompute similarity on the combined cohort, including official validation.
    texts=[' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',r['query']).casefold())) for r in ROWS]
    matrix=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=1).fit_transform(texts)
    similarities=(matrix@matrix.T).toarray()
    parent=list(range(len(ROWS)));pairs=[];exact=[];cross=[]
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in range(len(ROWS)):
        for j in range(i):
            same=texts[i]==texts[j]
            prefix=min(len(texts[i]),len(texts[j]))>=260 and texts[i][:180]==texts[j][:180]
            similarity=float(similarities[i,j])
            if same or prefix or similarity>=.8:
                parent[find(i)]=find(j)
                item={'query_a':ROWS[j]['query_id'],'query_b':ROWS[i]['query_id'],'split_a':ROWS[j]['split'],
                      'split_b':ROWS[i]['split'],'similarity':similarity,'exact':same,'scenario_prefix':prefix}
                pairs.append(item)
                if same:exact.append(item)
                if ROWS[j]['split']!=ROWS[i]['split']:cross.append(item)
    clusters={ROWS[i]['query_id']:min(ROWS[j]['query_id'] for j in range(len(ROWS)) if find(j)==find(i)) for i in range(len(ROWS))}
    assert not cross,'Identifiable cross-split leakage requires revisiting evaluation claims'
    assert len(set(BY_ID))==len(ROWS)
    for r in ROWS:r['group_id']=clusters[r['query_id']]
    pd.DataFrame(pairs).to_csv(OUT/'recomputed_near_duplicate_pairs.csv',index=False)
    pd.DataFrame([{'query_id':q,'group_id':g,'split':BY_ID[q]['split']} for q,g in clusters.items()]).to_csv(OUT/'recomputed_query_groups.csv',index=False)
    metadata=[]
    for r in ROWS:
        query=r['query'];explicit=EXPLICIT(query);gold=GOLD[r['query_id']]
        # Transparent lexical markers, not expert legal-domain classifications.
        criminal=bool(re.search(r'\b(StGB|StPO|criminal|crime|prosecut|Straf|Staatsanw)',query,re.I))
        insurance=bool(re.search(r'\b(IVG|UVG|ATSG|Versicher|insurance|disability|Invalid)',query,re.I))
        private=bool(re.search(r'\b(?:OR|ZGB|ZPO)\b',query) or re.search(r'\b(contract|property|tenancy|Vertrag|Eigentum|Miet)',query,re.I))
        environment=bool(re.search(r'\b(USG|UVPV|environment|Lärm|noise|Umwelt)',query,re.I))
        lang='English lexical markers' if re.search(r'\b(the|whether|which|under)\b',query,re.I) else 'German lexical markers' if re.search(r'\b(der|die|das|ist|welche)\b',query,re.I) else 'Other/undetermined lexical markers'
        metadata.append({'query_id':r['query_id'],'split':r['split'],'group_id':r['group_id'],'characters':len(query),
                         'gold_count':len(gold),'covered_gold_count':len(gold&set(COVERED)),
                         'missing_gold_count':len(gold-set(COVERED)),
                         'covered_law_gold':sum('law' in COVERED[c] for c in gold if c in COVERED),
                         'covered_court_gold':sum('court' in COVERED[c] for c in gold if c in COVERED),
                         'full_law_gold_heuristic':sum(c.startswith('Art.') for c in gold),
                         'explicit_citations':';'.join(explicit),'explicit_count':len(explicit),
                         'has_explicit':bool(explicit),'explicit_gold_exact_hits':len(set(explicit)&gold),
                         'language_marker':lang,'criminal_marker':criminal,'insurance_marker':insurance,
                         'private_marker':private,'environment_marker':environment})
    metadata=pd.DataFrame(metadata);metadata.to_csv(OUT/'query_metadata.csv',index=False)
    stats=[]
    groups=[('all_original_train',metadata[metadata.split!='official_val'])]+list(metadata.groupby('split'))
    for split,group in groups:
        stats.append({'split':split,'queries':len(group),'scenario_groups':group.group_id.nunique(),
                      'gold_occurrences':int(group.gold_count.sum()),'gold_mean':group.gold_count.mean(),
                      'gold_median':group.gold_count.median(),'gold_min':group.gold_count.min(),'gold_max':group.gold_count.max(),
                      'covered_occurrences':int(group.covered_gold_count.sum()),'missing_occurrences':int(group.missing_gold_count.sum()),
                      'no_covered_gold_queries':int((group.covered_gold_count==0).sum()),
                      'covered_law_gold':int(group.covered_law_gold.sum()),'covered_court_gold':int(group.covered_court_gold.sum()),
                      'explicit_queries':int(group.has_explicit.sum()),'explicit_query_share':group.has_explicit.mean(),
                      'character_median':group.characters.median(),
                      **{x+'_queries':int(group[x].sum()) for x in ['criminal_marker','insurance_marker','private_marker','environment_marker']}})
    pd.DataFrame(stats).to_csv(OUT/'dataset_distributions.csv',index=False)
    pd.crosstab(metadata.split,metadata.language_marker).to_csv(OUT/'query_language_markers.csv')
    result={'queries':len(ROWS),'near_duplicate_pairs':len(pairs),'normalized_exact_pairs':len(exact),
            'cross_split_identifiable_pairs':cross,'groups':len(set(clusters.values())),
            'official_query_id_overlap_with_original_train':list(set(q for q in GOLD if q.startswith('val_'))&set(q for q in GOLD if q.startswith('train_'))),
            'official_max_similarity_to_train':float(similarities[-10:,:-10].max()),
            'method':'Exhaustive normalized char_wb 3-5 gram TF-IDF; cosine >=0.8 or exact or 180-character shared prefix',
            'semantic_paraphrase_guarantee':False,'language_and_type_labels':'Transparent regex/lexical markers; not human or semantic classifications'}
    (OUT/'query_overlap_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return metadata

def cached_systems():
    official=[r['query_id'] for r in ROWS if r['split']=='official_val']
    train=[r['query_id'] for r in ROWS if r['split']!='official_val']
    source_rows=[]
    for system,prefix in [('BM25','bm25'),('Dense_pretrained_history','dense_full')]:
        frame=pd.read_csv(ROOT/CONFIG['historical_inputs']/f'{prefix}_candidates_top100.csv',keep_default_na=False)
        full=pd.read_csv(ROOT/CONFIG['historical_inputs']/f'{prefix}_train_candidates_top100.csv',keep_default_na=False)
        combined=pd.concat([frame,full])
        for (qid,),group in combined.groupby(['query_id']):
            assert group.citation.nunique()==100 and len(group)==100
        for split,ids in [('official_val',official),('original_train_frozen',train),('eval',[r['query_id'] for r in ROWS if r['split']=='eval'])]:
            for k in CONFIG['cutoffs']:
                pred=candidate_predictions(combined[combined.query_id.isin(ids)],k)
                for mode in ('full','covered'):evaluate(system,pred,ids,split,k,'New rescoring of historical candidate files',gold_mode=mode)
        for split,name in [('train',f'{prefix}_train_candidates_top100.csv'),('official_val',f'{prefix}_candidates_top100.csv'),('test',f'{prefix}_test_candidates_top100.csv')]:
            current=pd.read_csv(ROOT/CONFIG['historical_inputs']/name,keep_default_na=False)
            for qid,group in current.groupby('query_id'):
                source_rows.append({'system':system,'split':split,'query_id':qid,'law':int((group.source_type=='law').sum()),
                                    'court':int((group.source_type=='court').sum()),'total':len(group)})
    pd.DataFrame(source_rows).to_csv(OUT/'candidate_source_per_query.csv',index=False)
    bests=[('BM25_best','bm25','bm25_best_config.json'),('Dense_best','dense_full','dense_full_config.json'),
           ('BGE_zero_best','ce_bge_zeroshot','ce_bge_zeroshot_config.json'),('BGE_ft_best','ce_bge_ft','ce_bge_ft_config.json'),
           ('Mini_best','ce_mini','ce_mini_config.json')]
    for label,prefix,config_file in bests:
        config=json.loads((ROOT/CONFIG['historical_inputs']/config_file).read_text(encoding='utf8'))
        pred_frame=pd.read_csv(ROOT/CONFIG['historical_inputs']/f'{prefix}_best_predictions.csv',keep_default_na=False)
        pred={r.query_id:parse_citations(r.predicted_citations) for r in pred_frame.itertuples(index=False)}
        record,_=evaluate(label,pred,official,'official_val','selected','New rescoring of historical selected prediction file')
        errors={m:abs(record[m]-config[m]) for m in ('macro_precision','macro_recall','macro_f1')}
        assert max(errors.values())<1e-12
        CHECKS.append({'check':label+'_selected_predictions_vs_config','passed':True,'max_error':max(errors.values())})
    # Reconstruct equal-source union RRF with the project's exact deterministic tie rule.
    frames=[]
    for prefix in ('bm25','dense_full'):
        data=pd.read_csv(ROOT/CONFIG['historical_inputs']/f'{prefix}_candidates_top100.csv',keep_default_na=False)
        data['component']=prefix;frames.append(data[['query_id','citation','rank','component']])
    pooled=pd.concat(frames);pooled['rrf']=1/(60+pooled['rank'])
    union=pooled.groupby(['query_id','citation'],as_index=False).rrf.sum().rename(columns={'rrf':'first_stage_rrf'})
    union=union.sort_values(['query_id','first_stage_rrf','citation'],ascending=[True,False,True])
    union['rank']=union.groupby('query_id').cumcount()+1
    union.to_csv(OUT/'reconstructed_first_stage_union.csv',index=False)
    pred=candidate_predictions(union,20);evaluate('Union_RRF_top20',pred,official,'official_val',20,'New deterministic recombination of historical candidates',scope='fixed_union')
    evaluate('Union_full_ceiling',candidate_predictions(union,200),official,'official_val',200,'New candidate coverage computation',scope='fixed_union')
    for label,prefix in [('BGE_zero','ce_bge_zeroshot'),('BGE_ft','ce_bge_ft'),('Mini','ce_mini')]:
        data=pd.read_csv(ROOT/CONFIG['historical_inputs']/f'{prefix}_candidates_top100.csv',keep_default_na=False)
        assert len(data)==1000
        for qid,group in data.groupby('query_id'):
            assert set(group.citation)<=set(union[union.query_id==qid].citation)
        for k in [5,10,20,30,50,100]:
            evaluate(label+'_fixed',candidate_predictions(data,k),official,'official_val',k,'New fixed-K rescoring of historical CE ranking',scope='fixed_union')
        # Saved final_score uses full-input normalization at the selected alpha.
        # Restrict cutoff changes to same alpha; all cutoffs cap at 50 <= saved depth.
        for threshold in [.5,.55,.6,.65,.7,.75,.8,.85,.9,.95]:
            pred=CE['select_citations'](data.sort_values(['query_id','rank']),'relative',threshold,1,50)
            evaluate(label+'_relative',pred,official,'official_val',threshold,'New same-alpha cutoff rescoring of historical CE final_score',scope='fixed_union')
        # Historical full grids retain complete-input scores for alpha sensitivity.
        grid=pd.read_csv(ROOT/CONFIG['historical_inputs']/f'{prefix}_grid_summary.csv')
        grid.to_csv(OUT/f'{prefix}_historical_grid.csv',index=False)
        config=json.loads((ROOT/CONFIG['historical_inputs']/f'{prefix}_config.json').read_text(encoding='utf8'))
        best=grid.iloc[0]
        assert abs(best.macro_f1-config['macro_f1'])<1e-12
    # Archive held-out/full-corpus E5 results: re-evaluate using current evaluator.
    for label,file in [('E5_base_full','final_original_E5_candidates.csv'),('E5_ft_full','final_finetuned_E5_candidates.csv')]:
        data=pd.read_csv(ARCHIVE/'results'/file,keep_default_na=False)
        for qid,group in data.groupby('query_id'):
            source_rows.append({'system':label,'split':BY_ID[qid]['split'],'query_id':qid,
                                'law':int((group.source_type=='law').sum()),'court':int((group.source_type=='court').sum()),'total':len(group)})
        original=pd.read_csv(ARCHIVE/'results'/file.replace('_candidates','_summary'))
        for split in ('eval','official_val'):
            ids=[r['query_id'] for r in ROWS if r['split']==split]
            for k in CONFIG['cutoffs']:
                pred=candidate_predictions(data[data.query_id.isin(ids)],k)
                for mode in ('full','covered'):
                    record,_=evaluate(label,pred,ids,split,k,'New rescoring of archived newly-executed full-corpus E5 retrieval',gold_mode=mode)
                    archived=original[(original['split']==split)&(original.gold_mode==('in_corpus' if mode=='covered' else 'full'))&(original.k==k)]
                    assert len(archived)==1
                    error=max(abs(record[x]-archived.iloc[0][x]) for x in ('macro_precision','macro_recall','macro_f1'))
                    assert error<1e-12
                    CHECKS.append({'check':label+'_'+split+'_'+mode+'_'+str(k),'passed':True,'max_error':error})
    dev=[]
    for label in ('base','epoch_1','epoch_2'):
        data=pd.read_csv(ARCHIVE/'results'/f'dev_{label}_candidates.csv',keep_default_na=False)
        for k in CONFIG['cutoffs']:
            record,_=evaluate('E5_dev_'+label,candidate_predictions(data,k),[r['query_id'] for r in ROWS if r['split']=='dev'],'dev',k,'New rescoring of archived checkpoint-selection rankings')
            dev.append(record)
    selection=json.loads((ARCHIVE/'models/selection.json').read_text(encoding='utf8'))
    verified=[r for r in dev if r['k']==100]
    selected=max(verified,key=lambda r:r['macro_recall'])['system'].removeprefix('E5_dev_')
    assert selected==selection['deployment_best']['name']=='epoch_2'
    pd.DataFrame(dev).to_csv(OUT/'dev_checkpoint_sensitivity.csv',index=False)
    # All frozen query-expression groups; these remain historical validation.
    qdata=pd.read_csv(ROOT/CONFIG['query_candidates'],keep_default_na=False)
    for label,group in qdata.groupby('variant'):
        for k in CONFIG['cutoffs']:evaluate('Query_'+label,candidate_predictions(group,k),official,'official_val',k,'New rescoring of archived query-expression experiment')
    pd.DataFrame(source_rows).to_csv(OUT/'candidate_source_per_query.csv',index=False)

def paired_statistics(detail):
    comparisons=[
        ('CE_selected_ft_minus_zero','BGE_ft_best','BGE_zero_best','official_val','selected','f1','Selected settings differ; combines model and cutoff'),
        ('CE_fixed20_ft_minus_zero','BGE_ft_fixed','BGE_zero_fixed','official_val',20,'f1','Same historical union, alpha=0.8, fixed Top-20'),
        ('CE_cutoff_ft_relative_minus_top20','BGE_ft_relative','BGE_ft_fixed','official_val',(.6,20),'f1','Same fine-tuned model/ranking; cutoff effect'),
        ('CE_zero_minus_union20','BGE_zero_fixed','Union_RRF_top20','official_val',20,'f1','Same first-stage union, fixed Top-20; reranking/fusion effect'),
        ('E5_eval_f1_ft_minus_base','E5_ft_full','E5_base_full','eval',10,'f1','Unseen archived E5 eval; full corpus, fixed Top-10'),
        ('E5_eval_recall_ft_minus_base','E5_ft_full','E5_base_full','eval',100,'recall','Unseen archived E5 eval; full corpus, fixed Top-100'),
        ('E5_val_f1_ft_minus_base','E5_ft_full','E5_base_full','official_val',10,'f1','Ten historically tuned English questions'),
        ('Query_de_f1_minus_en','Query_de','Query_en','official_val',10,'f1','Frozen expression experiment; ten historical validation questions')]
    pairs=[];summary=[];leave=[]
    rng=np.random.default_rng(CONFIG['statistics_seed']);B=CONFIG['bootstrap_samples']
    for name,left,right,split,k,metric,note in comparisons:
        ka,kb=k if isinstance(k,tuple) else (k,k)
        a=detail[(detail.system==left)&(detail.split==split)&(detail.k.astype(str)==str(ka))&(detail.gold_mode=='full')].set_index('query_id')
        b=detail[(detail.system==right)&(detail.split==split)&(detail.k.astype(str)==str(kb))&(detail.gold_mode=='full')].set_index('query_id')
        assert set(a.index)==set(b.index) and len(a)>0,(name,ka,kb)
        ids=sorted(a.index);values=(a.loc[ids,metric]-b.loc[ids,metric]).to_numpy()
        for qid,value in zip(ids,values):pairs.append({'comparison':name,'query_id':qid,'group_id':BY_ID[qid]['group_id'],'metric':metric,
                                                     'left':a.loc[qid,metric],'right':b.loc[qid,metric],'delta':value,'left_tp':a.loc[qid,'tp'],'right_tp':b.loc[qid,'tp']})
        sampled=rng.integers(0,len(values),size=(B,len(values)))
        iid=values[sampled].mean(axis=1)
        groups=pd.DataFrame({'value':values,'group':[BY_ID[q]['group_id'] for q in ids]}).groupby('group').value.agg(['sum','count'])
        pick=rng.integers(0,len(groups),size=(B,len(groups)))
        cluster=groups['sum'].to_numpy()[pick].sum(axis=1)/groups['count'].to_numpy()[pick].sum(axis=1)
        active=values[np.abs(values)>1e-12]
        if len(active)<=18:
            perms=np.array(list(itertools.product([-1,1],repeat=len(active))))@active/len(values) if len(active) else np.zeros(1)
            p=float(np.mean(np.abs(perms)>=abs(values.mean())-1e-12));method='Exact paired sign-flip; conditional exchangeability diagnostic'
        else:
            perms=(rng.choice([-1,1],size=(B,len(active)))@active)/len(values)
            p=(int((np.abs(perms)>=abs(values.mean())-1e-12).sum())+1)/(B+1);method='Monte Carlo paired sign-flip; conditional exchangeability diagnostic'
        group_sums=groups['sum'].to_numpy();nonzero=group_sums[np.abs(group_sums)>1e-12]
        if len(nonzero)<=18:
            perms=np.array(list(itertools.product([-1,1],repeat=len(nonzero))))@nonzero/len(values) if len(nonzero) else np.zeros(1)
            cluster_p=float(np.mean(np.abs(perms)>=abs(values.mean())-1e-12))
        else:
            perms=rng.choice([-1,1],size=(B,len(nonzero)))@nonzero/len(values)
            cluster_p=(int((np.abs(perms)>=abs(values.mean())-1e-12).sum())+1)/(B+1)
        improved=int((values>1e-12).sum());worse=int((values< -1e-12).sum())
        loo=(values.sum()-values)/(len(values)-1)
        for qid,delta in zip(ids,loo):leave.append({'comparison':name,'removed_query':qid,'remaining_queries':len(values)-1,'mean_delta':delta,
                                                   'full_mean_delta':values.mean(),'sign_changed':bool(np.sign(delta)!=np.sign(values.mean()))})
        positive=np.sort(values[values>0])[::-1]
        summary.append({'comparison':name,'metric':metric,'queries':len(values),'scenario_groups':len(groups),'mean_delta':values.mean(),
                        'median_delta':float(np.median(values)),'improved':improved,'tied':len(values)-improved-worse,'worsened':worse,
                        'largest_gain':values.max(),'largest_loss':values.min(),
                        'top_positive_query_share_of_positive_gain':positive[0]/positive.sum() if len(positive) else np.nan,
                        'top2_share_of_positive_gain':positive[:2].sum()/positive.sum() if len(positive) else np.nan,
                        'query_bootstrap_low':float(np.quantile(iid,.025)),'query_bootstrap_high':float(np.quantile(iid,.975)),
                        'cluster_bootstrap_low':float(np.quantile(cluster,.025)),'cluster_bootstrap_high':float(np.quantile(cluster,.975)),
                        'paired_signflip_p':p,'cluster_signflip_p':cluster_p,'test_method':method,
                        'sign_test_p':float(binomtest(improved,improved+worse,.5).pvalue) if improved+worse else 1.,
                        'loo_min_delta':loo.min(),'loo_max_delta':loo.max(),'loo_sign_changes':int((np.sign(loo)!=np.sign(values.mean())).sum()),
                        'interpretation_scope':note,'selection_bias_removed':False})
    # Eight predeclared contrasts; Holm controls multiplicity within this reporting family,
    # but cannot repair historical selection on these labels or unrecorded earlier searches.
    order=np.argsort([r['cluster_signflip_p'] for r in summary]);running=0.
    for position,index in enumerate(order):
        running=max(running,min(1.,(len(order)-position)*summary[index]['cluster_signflip_p']))
        summary[index]['holm_cluster_p']=running
    pd.DataFrame(summary).to_csv(OUT/'paired_statistics.csv',index=False)
    pd.DataFrame(pairs).to_csv(OUT/'paired_query_deltas.csv',index=False)
    pd.DataFrame(leave).to_csv(OUT/'leave_one_query_out.csv',index=False)

def main():
    global OUT
    parser=argparse.ArgumentParser(description='Recompute saved-artifact stability statistics; no model inference')
    parser.add_argument('--output-dir',type=Path,default=ROOT/CONFIG['default_output'])
    args=parser.parse_args()
    OUT=args.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
    metadata=query_audit();cached_systems()
    summary=pd.DataFrame(SUMMARY);detail=pd.concat(DETAILS,ignore_index=True)
    summary.to_csv(OUT/'recomputed_metrics.csv',index=False)
    detail.to_csv(OUT/'recomputed_per_query.csv',index=False)
    paired_statistics(detail)
    # Stratification is descriptive: small subgroups are not independently tuned.
    joined=detail.merge(metadata[['query_id','has_explicit','gold_count','criminal_marker','insurance_marker','private_marker','environment_marker']],on='query_id')
    records=[]
    for columns,group in joined[(joined.gold_mode=='full')&(joined.scope=='full_corpus')].groupby(['system','split','k','has_explicit'],dropna=False):
        records.append({'system':columns[0],'split':columns[1],'k':columns[2],'has_explicit':columns[3],'queries':len(group),
                        'macro_precision':group.precision.mean(),'macro_recall':group.recall.mean(),'macro_f1':group.f1.mean()})
    pd.DataFrame(records).to_csv(OUT/'explicit_citation_stratification.csv',index=False)
    env={name:importlib.metadata.version(name) for name in ('numpy','pandas','scipy','scikit-learn')}
    record={'all_passed':True,'python':sys.version,'dependencies':env,'recomputations':CHECKS,
            'new_model_scoring':False,'new_full_corpus_retrieval':False,'seed_training_recorded_separately':True,
            'bootstrap_samples':CONFIG['bootstrap_samples'],'statistics_seed':CONFIG['statistics_seed'],
            'paired_loqo_is_retraining_cv':False,
            'statistical_references':['https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html',
                                      'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html'],
            'selection_caveat':'Intervals and tests condition on saved choices; historical ten-query selection bias remains; no p-value establishes transfer to English/new query distributions.'}
    (OUT/'statistical_run_checks.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print(summary[(summary.gold_mode=='full')&(summary.system.str.endswith('best'))][['system','macro_precision','macro_recall','macro_f1']].to_string(index=False))
    print(pd.read_csv(OUT/'paired_statistics.csv')[['comparison','mean_delta','improved','tied','worsened','loo_min_delta','loo_max_delta','cluster_bootstrap_low','cluster_bootstrap_high']].to_string(index=False))

if __name__=='__main__':main()
