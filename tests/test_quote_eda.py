"""Checks for the interactive notebook's cleaning and dropdown behavior."""
import contextlib
import io
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fixture():
    times = pd.date_range('2026-03-02 09:55', '2026-03-02 11:00', freq='5min')
    rows = [(bond,firm,side,when,60+10*i+j*j*.03+offset,1.)
            for i,bond in enumerate(['BOND00001','BOND00002','BOND00003'])
            for j,when in enumerate(times) for firm in ['A','B','C']
            for side,offset in [('bid',2),('ask',-2)]]
    q = pd.DataFrame(rows,columns=['cusip','firm','side','quote_timestamp_ET','spread','quantity'])
    t = pd.DataFrame({'CUSIP':['BOND00001','BOND00002','BOND00003'],
        'EFFECTIVE_DATETIME_TS':pd.to_datetime(['2026-02-02 10:00']*3),
        'BM_SPREAD':[.6,.7,.8],'QUANTITY':1.,'EFF_SIDE':'D'})
    m = pd.DataFrame({'CUSIP':t.CUSIP,'ISSUER':'TEST','YRS_TO_MATURITY':[1.,5.,10.]})
    return q,t,m


def load(q,t,m,**settings):
    with contextlib.redirect_stdout(io.StringIO()), patch('IPython.display.display'):
        ns = runpy.run_path(str(ROOT/'quote_eda.py'),init_globals={'bcq_df':q,'data_ig':t,
            'bond_info_df':m,'EDA_OVERRIDES':{'ISSUER':'TEST','START':'2026-03-02 10:00',
            'END':'2026-03-02 11:00','SHOW_PLOTS':False,'SHOW_WIDGETS':False,
            'MIN_COMMON_OBS':3,**settings}})
    return ns


def state(r,when,firm='A',side='bid',bond='BOND00001'):
    frame = r['states']
    return frame.loc[frame.time.eq(pd.Timestamp(when,tz='America/New_York')) & frame.cusip.eq(bond)
                     & frame.firm.eq(firm) & frame.side.eq(side)].iloc[0]


class DashboardChecks(unittest.TestCase):
    def test_universe_independent_of_quote_window_and_prefilter(self):
        q,t,m=fixture()
        t=t.loc[t.CUSIP.ne('BOND00003')]
        full=load(q,t,m)['result']
        filtered=load(q.loc[q.cusip.isin(t.CUSIP)],t,m)['result']
        self.assertEqual(set(full['coverage'].index),{'BOND00001','BOND00002'})
        self.assertTrue(full['coverage'].window_trades.eq(0).all())
        self.assertTrue(full['coverage'].n_trades.eq(1).all())
        pd.testing.assert_frame_equal(full['levels'],filtered['levels'])

    def test_explicit_universe_dates(self):
        q,t,m=fixture()
        t.loc[t.CUSIP.eq('BOND00003'),'EFFECTIVE_DATETIME_TS']=pd.Timestamp('2025-11-30')
        r=load(q,t,m,TRADE_UNIVERSE_START='2025-12-01',TRADE_UNIVERSE_END='2026-03-01')['result']
        self.assertEqual(len(r['coverage']),2)

    def test_event_time_strictly_before_and_future_independence(self):
        q,t,m=fixture()
        base=load(q,t,m)['result']
        self.assertEqual(state(base,'2026-03-02 10:05').event_time,
                         pd.Timestamp('2026-03-02 10:00',tz='America/New_York'))
        self.assertTrue((base['states'].event_time < base['states'].time).all())
        q.loc[q.quote_timestamp_ET.ge(pd.Timestamp('2026-03-02 10:20')),'spread']+=1000
        changed=load(q.sample(frac=1,random_state=4),t,m)['result']
        cutoff=pd.Timestamp('2026-03-02 10:20',tz='America/New_York')
        pd.testing.assert_frame_equal(base['levels'].loc[lambda x:x.index.get_level_values('time')<=cutoff],
                                     changed['levels'].loc[lambda x:x.index.get_level_values('time')<=cutoff])

    def test_expiry_and_invalid_latest_never_revive_old_quote(self):
        q,t,m=fixture()
        target=q.cusip.eq('BOND00001') & q.firm.eq('A') & q.side.eq('bid')
        q=q.loc[~(target & q.quote_timestamp_ET.gt(pd.Timestamp('2026-03-02 10:00')))].copy()
        r=load(q,t,m)['result']
        self.assertEqual(state(r,'2026-03-02 10:35').reason,'expired')
        row=q.loc[q.cusip.eq('BOND00001') & q.firm.eq('A') & q.side.eq('bid')].iloc[0].copy()
        row['quote_timestamp_ET']=pd.Timestamp('2026-03-02 10:06')
        row['spread']=np.nan
        r=load(pd.concat([q,row.to_frame().T],ignore_index=True),t,m)['result']
        self.assertEqual(state(r,'2026-03-02 10:10').reason,'nonfinite')

    def test_large_timestamp_conflict_invalidates_latest_state(self):
        q,t,m=fixture()
        row=q.loc[q.cusip.eq('BOND00001') & q.firm.eq('A') & q.side.eq('bid')
                  & q.quote_timestamp_ET.eq(pd.Timestamp('2026-03-02 10:00'))].iloc[0].copy()
        row['spread']+=10
        a=load(pd.concat([q,row.to_frame().T],ignore_index=True),t,m)['result']
        b=load(pd.concat([row.to_frame().T,q],ignore_index=True),t,m)['result']
        self.assertEqual(state(a,'2026-03-02 10:05').reason,'timestamp conflict')
        self.assertEqual(state(b,'2026-03-02 10:05').reason,'timestamp conflict')
        self.assertTrue(pd.isna(state(a,'2026-03-02 10:05').spread))

    def test_tight_timestamp_batch_median_and_feed_sequence(self):
        q,t,m=fixture()
        q['seq']=1
        row=q.loc[q.cusip.eq('BOND00001') & q.firm.eq('A') & q.side.eq('bid')
                  & q.quote_timestamp_ET.eq(pd.Timestamp('2026-03-02 10:00'))].iloc[0].copy()
        old=row.spread
        row['spread']+=1
        row['quantity']=2
        row['seq']=2
        combined=pd.concat([row.to_frame().T,q],ignore_index=True)
        a=load(combined,t,m)['result']
        self.assertAlmostEqual(state(a,'2026-03-02 10:05').spread,old+.5)
        self.assertTrue(pd.isna(state(a,'2026-03-02 10:05').quantity))
        b=load(combined,t,m,QUOTE_SEQUENCE_COL='seq')['result']
        self.assertAlmostEqual(state(b,'2026-03-02 10:05').spread,old+1)

    def test_one_sided_zero_placeholder_rejects_only_identified_bad_leg(self):
        q,t,m=fixture()
        q.loc[q.firm.eq('A') & q.side.eq('bid'),'spread']=0
        r=load(q,t,m)['result']
        self.assertEqual(state(r,'2026-03-02 10:05').reason,'contextual zero')
        self.assertEqual(state(r,'2026-03-02 10:05',side='ask').reason,'kept')
        self.assertFalse(r['clean'].spread.eq(0).any())
        self.assertTrue((r['levels'].bid.dropna()>=r['levels'].ask.reindex(r['levels'].bid.dropna().index)).all())

    def test_conflicting_rows_at_same_maximum_sequence_are_not_arbitrarily_resolved(self):
        q,t,m=fixture()
        q['seq']=1
        row=q.loc[q.cusip.eq('BOND00001') & q.firm.eq('A') & q.side.eq('bid')
                  & q.quote_timestamp_ET.eq(pd.Timestamp('2026-03-02 10:00'))].iloc[0].copy()
        row['spread']+=10
        r=load(pd.concat([q,row.to_frame().T],ignore_index=True),t,m,QUOTE_SEQUENCE_COL='seq')['result']
        self.assertEqual(state(r,'2026-03-02 10:05').reason,'timestamp conflict')

    def test_zero_and_negative_prices_are_not_rejected_by_sign(self):
        q,t,m=fixture()
        q['spread']=(pd.Timestamp('2026-03-02 10:10')-q.quote_timestamp_ET).dt.total_seconds()/300
        r=load(q,t,m,CHANGE_LAG='10min')['result']
        self.assertEqual(state(r,'2026-03-02 10:15').reason,'kept')
        self.assertEqual(state(r,'2026-03-02 10:20').reason,'kept')
        self.assertTrue(r['clean'].spread.eq(0).any())
        self.assertTrue(r['clean'].spread.lt(0).any())
        self.assertTrue(r['matched'].spread.eq(0).any())
        self.assertTrue(r['matched'].spread_old.eq(0).any())

    def test_crossed_pair_rejects_both_when_bad_leg_unknown(self):
        q,t,m=fixture()
        q.loc[q.firm.eq('A') & q.side.eq('ask'),'spread']+=10
        r=load(q,t,m)['result']
        self.assertEqual(state(r,'2026-03-02 10:05').reason,'crossed dealer pair')
        self.assertEqual(state(r,'2026-03-02 10:05',side='ask').reason,'crossed dealer pair')
        self.assertFalse(r['pairs'].loc[r['pairs'].clean,'crossed'].any())

    def test_aggregate_crossing_with_no_common_pair_is_withheld(self):
        q,t,m=fixture()
        q=q.loc[(q.firm.eq('A') & q.side.eq('bid')) | (q.firm.eq('B') & q.side.eq('ask'))].copy()
        q.loc[q.side.eq('bid'),'spread']=60
        q.loc[q.side.eq('ask'),'spread']=70
        r=load(q,t,m)['result']
        self.assertTrue(r['pairs'].empty)
        self.assertTrue(r['levels'].isna().all().all())
        self.assertGreater(r['audit']['suppressed_consensus_crossings'],0)

    def test_common_pair_consensus_cannot_cross(self):
        q,t,m=fixture()
        q=q.loc[q.firm.ne('C')].copy()
        q=q.loc[~(q.firm.eq('A') & q.side.eq('ask') & q.quote_timestamp_ET.ge(pd.Timestamp('2026-03-02 10:00')))]
        q.loc[q.firm.eq('A'),'spread']=40
        r=load(q,t,m,PAIR_MAX_GAP='1min')['result']
        self.assertTrue((r['levels'].dropna().bid>=r['levels'].dropna().ask).all())

    def test_unknown_sizes_supported_known_mismatch_unpaired(self):
        q,t,m=fixture()
        q['quantity']=0
        r=load(q,t,m)['result']
        self.assertTrue(r['pairs'].unknown_size.all())
        self.assertTrue(r['pairs'].eligible.all())
        self.assertFalse(r['pairs'].same_size.any())
        self.assertTrue(r['changes'].count().ge(3).all())
        q.loc[q.side.eq('bid'),'quantity']=1
        q.loc[q.side.eq('ask'),'quantity']=2
        r=load(q,t,m)['result']
        self.assertFalse(r['pairs'].eligible.any())
        self.assertTrue(r['changes'].count().ge(3).all())

    def test_peer_filter_and_refresh_age(self):
        q,t,m=fixture()
        q.loc[q.firm.eq('A') & q.side.eq('bid'),'spread']=100
        r=load(q,t,m)['result']
        self.assertEqual(state(r,'2026-03-02 10:05').reason,'peer outlier')
        self.assertEqual(state(r,'2026-03-02 10:30').age_min,5)
        self.assertEqual(state(r,'2026-03-02 10:30').change_age_min,35)

    def test_dropdown_switches_issuer_and_display_toggles_do_not_recompute(self):
        q,t,m=fixture()
        m.loc[m.CUSIP.eq('BOND00003'),'ISSUER']='OTHER'
        ns=load(q,t,m)
        with contextlib.redirect_stdout(io.StringIO()):
            ns['issuer_box'].value='OTHER'
        callback_globals=ns['refresh'].__globals__
        other=callback_globals['result']
        self.assertEqual(other['issuer'],'OTHER')
        self.assertEqual(list(other['coverage'].index),['BOND00003'])
        with contextlib.redirect_stdout(io.StringIO()):
            ns['raw_box'].value=True
            ns['trend_box'].value=True
        self.assertIs(callback_globals['result'],other)
        self.assertTrue(all(trace.visible for trace in other['figures'][1].data if trace.meta in ['raw','trend']))

    def test_display_trend_never_fills_missing_points_or_changes_analysis(self):
        q,t,m=fixture()
        q=q.loc[q.quote_timestamp_ET.le(pd.Timestamp('2026-03-02 10:00'))]
        ns=load(q,t,m)
        r=ns['result']
        for side in ['bid','ask']:
            clean_trace=next(trace for trace in r['figures'][1].data if trace.name==f'clean {side}')
            trend_trace=next(trace for trace in r['figures'][1].data if trace.name==f'display trend {side}')
            self.assertTrue(np.array_equal(pd.isna(clean_trace.y),pd.isna(trend_trace.y)))
            self.assertFalse(clean_trace.connectgaps)
            self.assertFalse(trend_trace.connectgaps)
        before=r['changes'].copy()
        with contextlib.redirect_stdout(io.StringIO()):
            ns['trend_box'].value=True
        pd.testing.assert_frame_equal(before,r['changes'])

    def test_one_sided_quotes_and_pca_meaningful_subset(self):
        q,t,m=fixture()
        r=load(q.loc[q.side.eq('bid')],t,m)['result']
        self.assertTrue(r['pairs'].empty)
        self.assertTrue(r['coverage'].ask_fresh.eq(0).all())
        self.assertGreater(len(r['pca_variance']),0)
        self.assertAlmostEqual(r['pca_variance'].sum(),1.)
        self.assertEqual(len(r['pca_loadings']),3)

    def test_timezone_conversion(self):
        q,t,m=fixture()
        base=load(q,t,m)['result']
        q['quote_timestamp_ET']=q.quote_timestamp_ET.dt.tz_localize('America/New_York').dt.tz_convert('UTC')
        changed=load(q,t,m)['result']
        pd.testing.assert_frame_equal(base['levels'],changed['levels'])


if __name__=='__main__':
    unittest.main()
