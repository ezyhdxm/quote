"""Cache equivalence, absent versus incomplete state, and same-slot pairing diagnostics."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import quote_quality_core as qc
import quote_quality_population as qp
from tests import test_quality_step45 as test45
raw, T = test45.raw, test45.T


def direct_reference(quotes, queries, **kwargs):
    """Original per-bond full route, independent of the optimized query eligibility."""
    rows = []
    for bond, q in queries.groupby('cusip', sort=False):
        events = qc.event_history(quotes.loc[quotes.cusip.eq(bond)])['events']
        times = pd.DatetimeIndex(q.time).sort_values().unique()
        pieces = [qc.side_features_fast(events.loc[events.side.eq(side)], times,
                    kwargs.get('age_min', 30), kwargs.get('allow_exact', True)).add_prefix('bcq_'+side+'_')
                  for side in ['bid', 'ask']]
        pairs = qc.pair_snapshots(events, times, **kwargs)
        pieces.append(qc.pair_features(pairs, times).add_prefix('bcq_'))
        f = pd.concat(pieces, axis=1)
        f['bcq_has_quote'] = (f.bcq_bid_n_dealers+f.bcq_ask_n_dealers).gt(0).astype(float)
        rows.append(q.merge(f, left_on='time', right_index=True, how='left'))
    return pd.concat(rows, ignore_index=True).set_index('row_id').reindex(queries.row_id).reset_index()


class QuoteCacheChecks(unittest.TestCase):
    def test_narrow_events_preserve_full_row_repeat_semantics(self):
        quotes = raw([(0,'A','bid',100,2),(0,'A','bid',100,2),(1,'A','bid',np.nan,2)])
        quotes['source_detail'] = ['first', 'different', 'incomplete']
        quotes = pd.concat([quotes, quotes.iloc[[0]]], ignore_index=True)
        full = qc.event_history(quotes)
        narrow = qc.prepare_quote_events(quotes)
        pd.testing.assert_frame_equal(full['events'], narrow['events'])
        self.assertNotIn('raw', narrow)
        self.assertEqual(narrow['events'].repeats.tolist(), [1, 0])
        self.assertEqual(set(narrow['timings']), {'normalize_s','aggregate_s','history_s'})

    def test_fast_route_matches_original_for_empty_days_and_incomplete_latest(self):
        quotes = raw([(0,'A','bid',105,0),(0,'A','ask',100,0),
                      (2,'A','bid',np.nan,2),(1440,'B','bid',110,2)])
        queries = pd.DataFrame(dict(row_id=np.arange(8), cusip=['X']*7+['MISSING'],
            time=[T+pd.Timedelta(minutes=m) for m in [-1,0,1,2,3,1440,2880,0]]))
        for allow_exact in [True, False]:
            expected = direct_reference(quotes, queries, allow_exact=allow_exact)
            cached = qc.prepare_quote_events(quotes)
            result = qc.build_quote_features(quotes, queries, allow_exact=allow_exact, event_cache=cached)
            pd.testing.assert_frame_equal(expected, result)
            self.assertTrue(result.attrs['quote_feature_timings']['event_cache_reused'])
        incomplete = qc.build_quote_features(quotes, queries).set_index('row_id')
        self.assertEqual(incomplete.loc[4,'bcq_bid_n_incomplete'], 1)
        self.assertEqual(incomplete.loc[4,'bcq_bid_n_dealers'], 0)
        self.assertTrue(pd.isna(incomplete.loc[4,'bcq_pair_gap']))

    def test_no_state_queries_skip_dealer_and_pair_calculations(self):
        quotes = raw([(0,'A','bid',105,2),(0,'A','ask',100,2)])
        queries = pd.DataFrame(dict(row_id=[1,2,3], cusip=['X','X','ABSENT'],
            time=[T-pd.Timedelta(minutes=1),T+pd.Timedelta(days=1),T]))
        cached = qc.prepare_quote_events(quotes)
        with patch.object(qc, 'side_features_fast', side_effect=AssertionError('unexpected side work')), \
             patch.object(qc, 'pair_snapshots', side_effect=AssertionError('unexpected pair work')), \
             patch.object(qc, 'prepare_quote_events', side_effect=AssertionError('cache ignored')):
            result = qc.build_quote_features(quotes, queries, event_cache=cached)
        self.assertTrue(result.bcq_has_quote.eq(0).all())
        self.assertTrue(result.bcq_bid_n_incomplete.eq(0).all())
        self.assertTrue(result.bcq_bid_center_equal.isna().all())
        self.assertEqual(result.attrs['quote_feature_timings']['no_state_unique_queries'], 3)


class SameSlotChecks(unittest.TestCase):
    def setUp(self):
        self.quotes = raw([(0,'A','bid',70,0),(0,'A','ask',100,0),
            (0,'B','bid',90,2),(0,'B','bid',110,5),(0,'B','ask',100,2),(0,'B','ask',200,99),
            (0,'C','bid',105,2),(0,'C','ask',100,2)])
        self.events = qc.prepare_quote_events(self.quotes)['events']
        self.times = pd.DatetimeIndex([T,T+pd.Timedelta(minutes=20),T+pd.Timedelta(minutes=40)])

    def test_masks_from_cached_state_equal_fresh_asof_at_each_policy(self):
        cached = qc.pair_snapshots(self.events, self.times)
        for age, sync in [(10,0),(30,1),(60,5)]:
            actual = qc.pair_policy_masks(cached, age, sync)
            expected = qc.pair_snapshots(self.events, self.times, age, sync)
            pd.testing.assert_frame_equal(actual, expected)

    def test_abc_keeps_b_c_slots_equal_and_allows_matching_to_increase_crossing(self):
        pairs = qc.pair_snapshots(self.events, [T])
        result = qc.pair_policy_comparison(pairs)
        self.assertEqual(result.n_slots.tolist(), [3,2,2])
        self.assertEqual(result.iloc[1].all_fraction, 0)
        self.assertEqual(result.iloc[2].all_fraction, .5)
        self.assertNotEqual(result.iloc[0].gap_mean_bps, result.iloc[1].gap_mean_bps)
        self.assertNotEqual(result.iloc[1].gap_mean_bps, result.iloc[2].gap_mean_bps)
        self.assertEqual(len(pairs), 3)  # diagnostic never deletes unknown quantities

    def test_extreme_provenance_keeps_quantities_and_deduplicates_grid_state(self):
        pairs = qc.pair_snapshots(self.events, self.times)
        sources = qc.pair_extreme_sources(self.quotes, pairs, limit_each=1)
        low = sources.loc[sources.extreme.eq('Low gap boundary') & sources.boundary_candidate]
        self.assertEqual(set(zip(low.side, low.spread, low.quantity)), {('bid',90,2),('ask',200,99)})
        high = sources.loc[sources.extreme.eq('High gap boundary') & sources.boundary_candidate]
        self.assertEqual(set(zip(high.side, high.spread, high.quantity)), {('bid',110,5),('ask',100,2)})
        self.assertEqual(len(sources), 8)  # all four B raw candidates for each extreme once

    def test_dealer_change_only_renders_and_apply_reuses_bond_day_state(self):
        state = test45.NotebookChecks().load(4)
        cached_pairs = next(iter(state['step4_pair_cache'].values()))['pairs']
        with patch.dict(state, pair_snapshots=lambda *a,**k: (_ for _ in ()).throw(AssertionError('rebuilt state')),
                        event_history=lambda *a,**k: (_ for _ in ()).throw(AssertionError('rebuilt events'))):
            current = state['step4_dealer'].value
            state['step4_dealer'].value = next(v for _,v in state['step4_dealer'].options if v != current)
            before = state['step4_result']['pairs']
            state['step4_age'].value = 10
            self.assertIs(state['step4_result']['pairs'], before)
            state['apply_step4']()
            self.assertTrue(state['step4_result']['timings']['pair_cache_reused'])
        self.assertIs(next(iter(state['step4_pair_cache'].values()))['pairs'], cached_pairs)
        self.assertTrue(bytes(state['step4_image'].value).startswith(b'\x89PNG'))

    def test_freeze_is_explicit_once_and_keeps_same_case_ids(self):
        with patch.object(qp, 'fixed_case_manifest', wraps=qp.fixed_case_manifest) as freeze:
            state = test45.NotebookChecks().load(4)
            self.assertEqual(freeze.call_count, 0)
            state['freeze_step4_cases']()
            self.assertEqual(freeze.call_count, 1)
            ids = state['step4_manifest'].case_id.tolist()
            self.assertEqual(len(ids), len(set(ids)))
            self.assertIn('fixed_case', state['step4_result'])
            state['freeze_step4_cases']()
            self.assertEqual(freeze.call_count, 1)
            self.assertEqual(state['step4_manifest'].case_id.tolist(), ids)
            self.assertEqual(state['step4_result']['fixed_case']['case_id'], state['step4_case'].value)


if __name__ == '__main__':
    unittest.main()
