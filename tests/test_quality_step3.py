"""Causal event/snapshot checks and the notebook's four research workflows."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from test_quality_step2 import fixture

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'quote_quality_step3.py'


def load_dashboard(quotes=None, trades=None):
    default_q, default_t = fixture()
    quotes = default_q if quotes is None else quotes
    trades = default_t if trades is None else trades
    state = {'__name__': 'step3_test'}
    def read(path, *a, **k):
        return (trades if Path(path).name == 'data_ig.parquet' else quotes).copy(deep=True)
    with contextlib.redirect_stdout(io.StringIO()), patch('pandas.read_parquet', side_effect=read), \
         patch.object(Path, 'exists', return_value=True), patch('IPython.display.display') as display:
        exec(compile(SCRIPT.read_text(), str(SCRIPT), 'exec'), state)
    return state, display


def raw_rows(records):
    """records = (minute, dealer, spread, quantity), repeated minute is a set."""
    base = pd.Timestamp('2026-03-02T10:00:00', tz='America/New_York')
    return pd.DataFrame([dict(firm=d, cusip='BOND_A', side='bid', quote_timestamp_ET=base + pd.Timedelta(minutes=m), spread=s, quantity=q) for m,d,s,q in records])


class Step3Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state, _ = load_dashboard()

    def events(self, records):
        return self.state['event_history'](raw_rows(records))['events']

    def snapshots(self, events, minutes, age=30):
        base = pd.Timestamp('2026-03-02T10:00:00', tz='America/New_York')
        times = pd.DatetimeIndex([base + pd.Timedelta(minutes=m) for m in minutes])
        return self.state['asof_features'](events, times, age)

    def test_candidate_sets_keep_zero_negative_and_ignore_row_multiplicity(self):
        e = self.events([(0,'A',-1,0),(0,'A',0,0),(0,'A',100,2),(0,'A',100,2),
                         (1,'A',-1,0),(1,'A',0,0),(1,'A',100,2)])
        self.assertEqual(e.iloc[0]['spread_set'], (-1,0,100))
        self.assertEqual(e.iloc[0]['center'], 0)
        self.assertEqual(e.iloc[0]['repeats'], 1)
        self.assertTrue(e.iloc[1]['pair_refresh'])
        self.assertEqual(e.iloc[1]['center_delta'], 0)
        self.assertTrue(e['change_age_min'].isna().all())

    def test_exact_time_aba_and_quantity_condition_switch_are_causal(self):
        e = self.events([(0,'A',10,2),(1,'A',20,2),(2,'A',10,2),(3,'A',12,5)])
        self.assertEqual(e['observed_aba'].tolist(), [False,False,True,False])
        self.assertTrue(pd.isna(e.iloc[0]['guarded_delta']))
        self.assertEqual(e.iloc[1]['guarded_delta'], 10)
        self.assertTrue(pd.isna(e.iloc[3]['guarded_delta']))
        self.assertTrue(e.iloc[3]['condition_changed'])
        raw = raw_rows([(0,'A',10,2),(0,'A',20,2)])
        raw.loc[1,'quote_timestamp_ET'] += pd.Timedelta(nanoseconds=1)
        g = self.state['event_history'](raw)['events']
        self.assertEqual(len(g), 2)

    def test_unknown_change_age_refresh_gap_and_day_boundary(self):
        e = self.events([(0,'A',10,0),(10,'A',10,0),(20,'A',12,0),(30,'A',12,0),
                         (120,'A',13,0),(121,'A',14,0),(1440,'A',14,0)])
        self.assertTrue(e.loc[:1,'change_age_min'].isna().all())
        self.assertEqual(e.loc[2:3,'change_age_min'].tolist(), [0,10])
        self.assertTrue(pd.isna(e.iloc[4]['change_age_min']))
        self.assertTrue(pd.isna(e.iloc[-1]['change_age_min']))
        slots, _ = self.snapshots(e, [15,25,35,1441])
        self.assertEqual(slots['message_age_min'].tolist(), [5,5,5,1])
        self.assertTrue(pd.isna(slots.iloc[0]['spread_set_change_age_min']))
        self.assertEqual(slots.iloc[1]['spread_set_change_age_min'], 5)
        self.assertEqual(slots.iloc[2]['spread_set_change_age_min'], 15)
        _, absent = self.snapshots(e.iloc[:-1], [1441])
        self.assertEqual(absent.iloc[0]['n_dealers'], 0)
        self.assertTrue(pd.isna(absent.iloc[0]['center_equal']))

    def test_fixed_window_change_counts_do_not_depend_on_datetime_unit(self):
        raw = raw_rows([(0,'A',10,1),(5,'A',11,1),(10,'A',12,1),(35,'A',13,1)])
        ns = self.state['event_history'](raw)['events']
        raw['quote_timestamp_ET'] = raw['quote_timestamp_ET'].dt.as_unit('us')
        us = self.state['event_history'](raw)['events']
        self.assertEqual(ns['changes_30m'].tolist(), [0,1,2,2])
        self.assertEqual(us['changes_30m'].tolist(), ns['changes_30m'].tolist())

    def test_future_append_does_not_rewrite_event_or_asof_features(self):
        past = raw_rows([(0,'A',10,1),(0,'B',11,1),(10,'A',12,1),(20,'B',13,1),(35,'A',14,1)])
        future = raw_rows([(40,'A',999,1),(40,'NEW',-999,None),(1440,'B',500,1)])
        short = self.state['event_history'](past)['events']
        long = self.state['event_history'](pd.concat([past,future],ignore_index=True))['events']
        cols = ['firm','quote_timestamp_ET','spread_set','quantity_set','center_delta','guarded_delta','change_age_min','changes_30m','observed_aba']
        old = long.loc[long['quote_timestamp_ET'].le(past['quote_timestamp_ET'].max())]
        pd.testing.assert_frame_equal(short[cols].reset_index(drop=True),old[cols].reset_index(drop=True))
        _, a = self.snapshots(short,[0,10,20,30,35])
        _, b = self.snapshots(long,[0,10,20,30,35])
        pd.testing.assert_frame_equal(a,b)

    def test_fixed_horizon_delta_is_query_batch_invariant_and_guards_composition(self):
        e = self.events([(0,'A',10,1),(0,'B',20,1),(20,'A',12,1),(30,'A',14,1),
                         (35,'A',15,2),(40,'NEW',25,1)])
        _, one = self.snapshots(e,[30])
        _, many = self.snapshots(e,[0,5,10,20,25,30,35,40])
        pd.testing.assert_series_equal(one.iloc[0],many.loc[one.index[0]])
        self.assertEqual(one.iloc[0]['center_delta_30m'], 2)
        self.assertTrue(pd.isna(many.iloc[-2]['center_delta_30m']))
        self.assertEqual(many.iloc[-1]['composition_changed_30m'], 1)
        self.assertTrue(pd.isna(many.iloc[-1]['center_delta_30m']))

    def test_incomplete_latest_event_blocks_numeric_aggregation_without_fallback(self):
        e = self.events([(0,'A',10,1),(5,'A',20,1),(5,'A','bad',1)])
        slots,f = self.snapshots(e,[0,5,10])
        self.assertEqual(f['n_dealers'].tolist(), [1,0,0])
        self.assertEqual(f['n_incomplete'].tolist(), [0,1,1])
        self.assertTrue(f.iloc[1:]['center_equal'].isna().all())
        self.assertEqual(slots.iloc[-1]['center'],20)  # Still retained for research.

    def test_dealer_equal_not_message_or_candidate_weighted(self):
        records=[(0,'A',10,0)]*20 + [(0,'B',20,0),(0,'B',40,0)]
        e=self.events(records)
        _,f=self.snapshots(e,[0])
        self.assertEqual(f.iloc[0]['center_equal'],20)  # (10 + median(20,40)) / 2
        self.assertEqual(f.iloc[0]['n_dealers'],2)
        self.assertEqual(f.iloc[0]['multi_fraction'],0.5)

    def test_influence_uses_only_other_fresh_dealers_and_keeps_unsupported_values(self):
        e=self.events([(0,'A',100,0),(0,'B',10,0),(0,'C',10,0),(0,'D',10,0),
                       (40,'A',100,0),(50,'FUTURE',-999,0)])
        slots,f=self.snapshots(e,[0,40],age=30)
        target=slots.loc[slots['firm'].eq('A')]
        self.assertEqual(target.iloc[0]['peer_center'],10)
        self.assertEqual(target.iloc[0]['clipped_center'],20)
        self.assertLess(f.iloc[0]['center_candidate_clip'],f.iloc[0]['center_equal'])
        self.assertEqual(target.iloc[1]['n_peers'],0)
        self.assertTrue(pd.isna(target.iloc[1]['peer_residual']))
        self.assertEqual(target.iloc[1]['clipped_center'],100)
        self.assertEqual(target.iloc[1]['dealer_weight'],1)

    def test_age_rules_report_lost_coverage_and_equal_age_decay(self):
        e=self.events([(0,'A',10,0),(0,'B',20,0)])
        _,f=self.snapshots(e,[0,31],age=30)
        self.assertEqual(f.iloc[1]['n_dealers'],2)
        self.assertEqual(f.iloc[1]['n_fresh_dealers'],0)
        self.assertTrue(pd.isna(f.iloc[1]['center_max_age']))
        self.assertEqual(f.iloc[1]['center_decay'],15)
        self.assertEqual(f.iloc[1]['max_decay_weight_share'],0.5)

    def test_quantity_contrasts_require_complete_all_positive_unique_mapping(self):
        raw=raw_rows([(0,'A',10,1),(0,'A',20,2),(0,'A',20,2),
                      (1,'A',11,1),(1,'A',12,1),(1,'A',20,2),
                      (2,'A',10,0),(2,'A',20,2),
                      (3,'A',10,1),(3,'A','bad',2),
                      (4,'A',10,0.12345671),(4,'A',20,0.12345672)])
        r=self.state['event_history'](raw)
        cells,contrasts=self.state['quantity_evidence'](r['raw'])
        self.assertEqual(contrasts['quote_timestamp_ET'].nunique(),2)
        self.assertEqual(contrasts['within_event_residual'].tolist(),[-5,5,-5,5])
        self.assertEqual(r['events'].iloc[-1]['candidate_count'],2)
        self.assertEqual(len(r['events'].iloc[-1]['quantity_set']),2)

    def test_all_views_empty_case_and_rerun_have_one_image_no_table(self):
        state,display=load_dashboard()
        image_id=state['step3_image'].model_id
        for view in state['VIEWS']:
            state['step3_view'].value=view
            self.assertEqual(state['step3_image'].model_id,image_id)
            self.assertTrue(state['step3_image'].value.tobytes().startswith(b'\x89PNG'))
            self.assertEqual(len(state['step3_figure'].axes),3)
        old_view=state['step3_view']
        state['step3_issuer'].value='Gamma'  # Missing firm, no keyed events.
        self.assertTrue(state['step3_features'].empty)
        cell=SCRIPT.read_text().split('# %% 3. Choose a case; compare rules; save one PNG\n')[1]
        with contextlib.redirect_stdout(io.StringIO()): exec(compile(cell,'cell3','exec'),state)
        fig=state['step3_figure']
        old_view.value=state['VIEWS'][0]
        self.assertIs(state['step3_figure'],fig)
        self.assertEqual(display.call_count,2)
        self.assertTrue(all(isinstance(call.args[0],state['widgets'].VBox) for call in display.call_args_list))
        self.assertEqual(plt.get_fignums(),[])


if __name__ == '__main__':
    unittest.main()
