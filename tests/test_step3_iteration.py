"""Common-dealer diagnostics and case navigation preserve causal quote semantics."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from test_quality_step3 import load_dashboard, raw_rows


class Step3IterationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state, _ = load_dashboard()

    def features(self, records, minutes):
        events = self.state['event_history'](raw_rows(records))['events']
        base = pd.Timestamp('2026-03-02 10:00', tz='America/New_York')
        times = pd.DatetimeIndex([base + pd.Timedelta(minutes=m) for m in minutes])
        return self.state['asof_features'](events, times)

    def test_common_changes_survive_condition_switch_and_roster_entry(self):
        # A changes quantity while NEW enters: retain A's observed change as a
        # diagnostic, with explicit retention, while keeping guarded feature NaN.
        records = [(0, 'A', 10, 1), (30, 'A', 14, 2), (30, 'NEW', 30, 0)]
        _, f = self.features(records, [30])
        row = f.iloc[0]
        self.assertEqual(row['n_common_dealers_30m'], 1)
        self.assertEqual(row['common_dealer_delta_30m'], 4)
        self.assertEqual(row['aggregate_delta_30m'], 12)
        self.assertEqual(row['common_retention_30m'], .5)
        self.assertEqual(row['past_common_retention_30m'], 1)
        self.assertEqual(row['common_condition_changed_fraction_30m'], 1)
        self.assertTrue(np.isnan(row['center_delta_30m']))
        _, many = self.features(records, [0, 5, 10, 25, 30])
        pd.testing.assert_series_equal(row, many.loc[row.name])

    def test_incomplete_current_and_day_boundary_do_not_revive_common_price(self):
        records = [(0, 'A', 10, 1), (30, 'A', 20, 1), (30, 'A', 'bad', 1)]
        _, f = self.features(records, [30, 1440])
        self.assertEqual(f.iloc[0]['n_incomplete'], 1)
        self.assertEqual(f.iloc[0]['n_common_dealers_30m'], 0)
        self.assertTrue(f.common_dealer_delta_30m.isna().all())
        self.assertTrue(f.aggregate_delta_30m.isna().all())

    def test_apply_batches_age_and_display_change_reuses_case(self):
        state, _ = load_dashboard()
        original = state['asof_features']
        with patch.dict(state, {'asof_features': unittest.mock.Mock(wraps=original)}):
            measure = state['asof_features']
            state['step3_view'].value = state['VIEWS'][0]
            dealers = [v for _, v in state['step3_dealer'].options]
            if len(dealers) > 1:
                state['step3_dealer'].value = dealers[1]
            self.assertEqual(measure.call_count, 0)
            state['step3_age'].value = 10
            self.assertEqual(measure.call_count, 0)
            self.assertTrue(state['step3_save'].disabled)
            self.assertEqual(bytes(state['step3_image'].value), b'')
            state['refresh_step3']()
            self.assertEqual(measure.call_count, 1)
            self.assertFalse(state['step3_save'].disabled)
            state['step3_age'].value = 30
            state['refresh_step3']()
            self.assertEqual(measure.call_count, 1)  # Already cached initial case.


if __name__ == '__main__':
    unittest.main()
