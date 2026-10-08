"""Meaningful metric, pairing, identity and real-artifact regression checks."""
from pathlib import Path
import json, sys, unittest
import numpy as np
import pandas as pd
from scipy.stats import permutation_test
sys.path.insert(0,str(Path(__file__).resolve().parent))
from evaluation import evaluate_predictions, parse_citations

ROOT=Path(__file__).resolve().parents[1]

class Checks(unittest.TestCase):
    def test_citation_set_and_gold_denominator(self):
        # Toy inputs are unit tests only, not experimental scores.
        score,_=evaluate_predictions({'q':['a','a']},{'q':{'a','missing'}})
        self.assertEqual(score['macro_precision'],1)
        self.assertEqual(score['macro_recall'],.5)
        self.assertEqual(parse_citations('a; a'),{'a'})

    def test_unmatched_query_ids_rejected(self):
        with self.assertRaises(ValueError):evaluate_predictions({'other':['a']},{'q':{'a'}})

    def test_real_paired_signflip_matches_scipy(self):
        details=pd.read_csv(ROOT/'results/paired_query_deltas.csv')
        stats=pd.read_csv(ROOT/'results/paired_statistics.csv').set_index('comparison')
        for name in ('CE_selected_ft_minus_zero','CE_fixed20_ft_minus_zero'):
            frame=details[details.comparison==name]
            result=permutation_test((frame.left.to_numpy(),frame.right.to_numpy()),lambda a,b:np.mean(a-b),
                                    permutation_type='samples',n_resamples=np.inf,vectorized=False,alternative='two-sided')
            self.assertAlmostEqual(result.pvalue,stats.loc[name,'paired_signflip_p'],places=12)

    def test_leave_one_query_formula(self):
        pairs=pd.read_csv(ROOT/'results/paired_query_deltas.csv');leave=pd.read_csv(ROOT/'results/leave_one_query_out.csv')
        for row in leave.itertuples(index=False):
            selected=pairs[(pairs.comparison==row.comparison)&(pairs.query_id!=row.removed_query)]
            self.assertAlmostEqual(selected.delta.mean(),row.mean_delta,places=12)

    def test_coverage_diagnostic_is_explicit(self):
        metrics=pd.read_csv(ROOT/'results/recomputed_metrics.csv')
        full=metrics[(metrics.system=='BM25')&(metrics.split=='original_train_frozen')&(metrics.gold_mode=='full')&(metrics.k.astype(str)=='20')].iloc[0]
        covered=metrics[(metrics.system=='BM25')&(metrics.split=='original_train_frozen')&(metrics.gold_mode=='covered')&(metrics.k.astype(str)=='20')].iloc[0]
        self.assertEqual(full.n_queries,1139);self.assertEqual(full.excluded_queries,0)
        self.assertEqual(covered.n_queries,999);self.assertEqual(covered.excluded_queries,140)
        self.assertEqual(covered.excluded_gold_occurrences,1340)

    def test_frozen_seed_pool_has_no_added_gold(self):
        pool=pd.read_csv(ROOT/'inputs/seed_candidate_pool.csv',keep_default_na=False)
        record=json.loads((ROOT/'inputs/seed_pool_record.json').read_text(encoding='utf8'))
        self.assertFalse(record['gold_injected']);self.assertEqual(pool.query_id.nunique(),124)
        self.assertTrue((pool.groupby('query_id').size()<=200).all())
        refs=[]
        for branch in ('bm25','dense_full'):
            for name in (branch+'_train_candidates_top100.csv',branch+'_candidates_top100.csv'):
                refs.append(pd.read_csv(ROOT/'inputs/historical'/name,keep_default_na=False))
        refs=pd.concat(refs)
        reference=set(zip(refs.query_id,refs.doc_id,refs.citation));actual=set(zip(pool.query_id,pool.doc_id,pool.citation))
        self.assertTrue(actual<=reference)

    def test_completed_seed_scores_if_available(self):
        if not (ROOT/'results/seed_pool_metrics.csv').exists():self.skipTest('New seed scoring still in progress')
        metrics=pd.read_csv(ROOT/'results/seed_pool_metrics.csv')
        self.assertEqual(set(metrics.system),{'Pool_base','Pool_seed_760','Pool_seed_761','Pool_seed_762'})
        rankings={label:pd.read_csv(ROOT/'results'/f'seed_pool_{label}_rankings.csv',keep_default_na=False) for label in ('base','seed_760','seed_761','seed_762')}
        sets=[set(zip(frame.query_id,frame.citation)) for frame in rankings.values()]
        self.assertTrue(all(value==sets[0] for value in sets))

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    record={'passed':result.wasSuccessful(),'tests':result.testsRun,'skipped':len(result.skipped),
            'failures':len(result.failures),'errors':len(result.errors),
            'scope':'Unit tests and saved-real-artifact regressions; toy tests are not retrieval results'}
    output=ROOT/'runs/verification'
    output.mkdir(parents=True,exist_ok=True)
    (output/'unit_test_results.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
