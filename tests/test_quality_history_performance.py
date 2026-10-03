"""Exact rolling-change equivalence across contiguous history segments."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import quote_quality_core as qc

T = pd.Timestamp('2026-03-02 10:00', tz='America/New_York')


def quotes(records):
    return pd.DataFrame([dict(cusip=bond, firm=firm, side=side, spread=spread,
        quantity=quantity, quote_timestamp_ET=T+pd.Timedelta(minutes=minute))
        for minute, bond, firm, side, spread, quantity in records])


def reference_changes(events):
    """The former pandas-indexed loop, retained only as an independent oracle."""
    result = pd.Series(0, index=events.index, dtype='int64', name='changes_30m')
    segments = events.groupby(events.history_break.cumsum()).groups
    for indices in segments.values():
        times = events.loc[indices, qc.KEYS[-1]].array.as_unit('ns').asi8
        counts = np.r_[0, events.loc[indices, 'spread_changed'].to_numpy().cumsum()]
        left = np.searchsorted(times, times-qc.LOOKBACK_MIN*60*10**9, side='right')
        result.loc[indices] = counts[1:]-counts[left]
    return result


class RollingHistoryChecks(unittest.TestCase):
    def check_reference(self, raw):
        events = qc.prepare_quote_events(raw)['events']
        pd.testing.assert_series_equal(events.changes_30m, reference_changes(events))
        return events

    def test_strict_left_boundary_includes_current_change(self):
        events = self.check_reference(quotes([
            (0,'X','A','bid',100,2),(5,'X','A','bid',101,2),
            (35,'X','A','bid',102,2),(65,'X','A','bid',103,2),
            (125,'X','A','bid',104,2),(186,'X','A','bid',105,2)]))
        self.assertEqual(events.changes_30m.tolist(), [0,1,1,1,1,0])
        self.assertEqual(events.history_break.tolist(), [True,False,False,False,False,True])
        # A change one nanosecond inside the window remains included.
        nanosecond = quotes([(0,'X','A','bid',100,2),(5,'X','A','bid',101,2),
                             (35,'X','A','bid',102,2)])
        nanosecond.loc[2,'quote_timestamp_ET'] -= pd.Timedelta(nanoseconds=1)
        self.assertEqual(self.check_reference(nanosecond).changes_30m.tolist(), [0,1,2])

    def test_series_side_day_incomplete_and_condition_boundaries(self):
        raw = quotes([
            (0,'X','A','bid',100,2),(5,'X','A','bid',101,2),
            (6,'X','A','bid',np.nan,2),(7,'X','A','bid',102,2),
            (8,'X','A','bid',103,2),(8,'X','A','bid',105,0),
            (1440,'X','A','bid',110,2),(1445,'X','A','bid',111,2),
            (0,'X','A','ask',90,2),(10,'X','A','ask',91,2),
            (0,'X','B','bid',120,2),(1,'X','B','bid',121,2),
            (0,'Y','A','bid',200,2),(0,'Z','C','ask',np.nan,0)])
        events = self.check_reference(raw.sample(frac=1, random_state=2026))
        bid = events.loc[events.cusip.eq('X') & events.firm.eq('A') & events.side.eq('bid')]
        self.assertEqual(bid.changes_30m.tolist(), [0,1,0,0,1,0,1])
        self.assertEqual(bid.complete.tolist(), [True,True,False,True,True,True,True])
        self.assertTrue(bid.iloc[4].condition_changed)
        self.assertEqual(bid.iloc[4].changes_30m, 1)  # spread changes still count when quantity/count changes
        singleton = events.loc[events.cusip.isin(['Y','Z'])]
        self.assertTrue(singleton.changes_30m.eq(0).all())
        self.assertTrue(singleton.history_break.all())

    def test_empty_events_and_progress_finish_with_segment_count(self):
        raw = quotes([(0,'X','A','bid',100,2)])
        reports = []
        empty = qc.prepare_quote_events(raw.iloc[:0], lambda *args: reports.append(args))['events']
        pd.testing.assert_series_equal(empty.changes_30m, reference_changes(empty))
        self.assertEqual(reports[-1][:3], ('events',0,0))
        reports.clear()
        self.check_reference(raw)
        qc.prepare_quote_events(raw, lambda *args: reports.append(args))
        self.assertEqual(reports[-1][:3], ('events',1,1))

    def test_twenty_thousand_single_event_segments(self):
        # High segment cardinality, unlike a dense single-series row-count fixture.
        size = 20_000
        raw = pd.DataFrame(dict(cusip=['B'+str(i) for i in range(size)],
            firm=['A']*size, side=['bid']*size, quote_timestamp_ET=pd.DatetimeIndex([T]*size),
            spread=np.full(size,100.), quantity=np.full(size,2.)))
        events = qc.prepare_quote_events(raw)['events']
        self.assertEqual(len(events), size)
        self.assertEqual(int(events.history_break.sum()), size)
        self.assertTrue(events.changes_30m.eq(0).all())
        self.assertEqual(events.changes_30m.dtype, np.dtype('int64'))


if __name__ == '__main__':
    unittest.main()
