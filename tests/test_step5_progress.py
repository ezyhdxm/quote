"""Progress reporting must expose real work without changing feature or model results."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quote_quality_core as qc


def quotes():
    start = pd.Timestamp('2026-03-02 10:00', tz='America/New_York')
    return pd.DataFrame([
        dict(cusip='X', firm=firm, side=side, spread=60 + minute / 10 + shift,
             quantity=2, quote_timestamp_ET=start + pd.Timedelta(minutes=minute))
        for minute in range(0, 60, 10)
        for firm in ['A', 'B']
        for side, shift in [('bid', 3), ('ask', -3)]
    ])


def record_into(events):
    return lambda stage, completed=None, total=None, detail='': events.append((stage, completed, total, detail))


class FeatureProgressChecks(unittest.TestCase):
    def test_history_and_empty_history_are_unchanged(self):
        for raw in [quotes(), quotes().iloc[:0]]:
            reports = []
            observed = qc.event_history(raw, progress=record_into(reports))
            expected = qc.event_history(raw)
            pd.testing.assert_frame_equal(observed['events'], expected['events'])
            pd.testing.assert_frame_equal(observed['raw'], expected['raw'])
            self.assertEqual(observed['unkeyed'], expected['unkeyed'])
            self.assertEqual([r[0] for r in reports], ['events'] * len(reports))
            self.assertTrue(all(r[1:3] == (None, None) for r in reports[:3]))
            self.assertEqual(reports[-1][1], reports[-1][2])

    def test_feature_progress_keeps_no_quote_queries_and_finishes_after_history(self):
        q = quotes()
        queries = pd.DataFrame({'row_id': [9, 7, 5], 'cusip': ['X', 'MISSING', 'X'],
                                'time': [q.quote_timestamp_ET.min()] * 3})
        reports = []
        observed = qc.build_quote_features(q, queries, progress=record_into(reports))
        pd.testing.assert_frame_equal(observed, qc.build_quote_features(q, queries))
        self.assertEqual(observed.row_id.tolist(), [9, 7, 5])
        self.assertEqual(observed.bcq_has_quote.tolist(), [1., 0., 1.])
        features = [r for r in reports if r[0] == 'features']
        self.assertEqual([r[1] for r in features], [0, 0, 1, 1, 2])
        self.assertTrue(all(r[2] == 2 for r in features))
        self.assertEqual(reports[0][0], 'features')
        self.assertEqual(reports[-1][:3], ('features', 2, 2))
        self.assertIn('MISSING', reports[-1][3])
        empty_reports = []
        empty = qc.build_quote_features(q, queries.iloc[:0], progress=record_into(empty_reports))
        pd.testing.assert_frame_equal(empty, queries.iloc[:0])
        self.assertEqual(empty_reports[0][:3], ('features', 0, 0))


@unittest.skipUnless(find_spec('lightgbm'), 'LightGBM is required for real-fit progress checks')
class ModelProgressChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        q = quotes()
        frame = pd.DataFrame({column: np.linspace(0, 1, 50) for column in qc.BASE_FEATURES})
        frame['PREV_TRADE_TYPE'] = 'B'
        frame['TRADE_TYPE'] = 'S'
        frame['PREV_BM_SPREAD'] = 0.6
        frame['D_BM_SPREAD'] = np.linspace(-0.03, 0.04, len(frame))
        frame['BM_SPREAD'] = frame.PREV_BM_SPREAD + frame.D_BM_SPREAD
        frame['cusip'] = 'X'
        frame['row_id'] = np.arange(len(frame))
        frame['time'] = pd.date_range(q.quote_timestamp_ET.min(), periods=len(frame), freq='min')
        frame['split'] = ['Train'] * 30 + ['Validation'] * 10 + ['Test'] * 10
        frame['refit_train'] = frame.row_id.lt(40)
        features = qc.build_quote_features(q, frame[['row_id', 'cusip', 'time']])
        cls.frame = frame.merge(features.drop(columns=['cusip', 'time']), on='row_id', validate='one_to_one')
        cls.params = dict(objective='mae', boosting_type='dart', n_estimators=12,
                          num_leaves=4, min_child_samples=2, n_jobs=1,
                          verbosity=-1, random_state=2026)

    def test_callback_preserves_predictions_and_counts_real_iterations(self):
        reports = []
        expected, _ = qc.run_comparison(self.frame, self.params)
        observed, models = qc.run_comparison(self.frame, self.params, progress=record_into(reports))
        pd.testing.assert_frame_equal(observed, expected)
        self.assertEqual(reports[0][:3], ('models', 0, 8))
        self.assertEqual(reports[-1][:3], ('models', 8, 8))
        self.assertEqual(len(models), 8)
        completed = [r for r in reports if r[0] == 'models' and r[3].startswith('Finished')]
        self.assertEqual([r[1] for r in completed], list(range(1, 9)))
        fits = [r for r in reports if r[0] == 'fit']
        self.assertEqual([r[1] for r in fits], [0, 10, 12] * 8)
        self.assertTrue(all(r[2] == 12 for r in fits))
        # Finishing training does not finish the model; prediction happens before completion.
        for index, event in enumerate(reports):
            if event[0] == 'models' and event[3].startswith('Finished'):
                self.assertTrue(reports[index - 1][3].startswith('Predicting'))
                self.assertEqual(reports[index - 1][1], event[1] - 1)

    def test_locked_test_counts_only_requested_versions_and_errors_do_not_finish(self):
        reports = []
        expected, _ = qc.run_comparison(self.frame, self.params, 'Test', 'Candidate clip')
        observed, _ = qc.run_comparison(self.frame, self.params, 'Test', 'Candidate clip', record_into(reports))
        pd.testing.assert_frame_equal(observed, expected)
        self.assertEqual(reports[-1][:3], ('models', 3, 3))
        failed = []
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            qc.run_comparison(self.frame.iloc[:0], self.params, progress=record_into(failed))
        self.assertFalse(any(r[3].startswith('Finished') for r in failed))


if __name__ == '__main__':
    unittest.main()
