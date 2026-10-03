"""Progress reporting must expose real work without changing feature or model results."""
from importlib.util import find_spec
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

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

    def test_finite_validation_versions_match_full_predictions_and_requested_order(self):
        full, _ = qc.run_comparison(self.frame, self.params)
        versions = ['Age decay','Base','Reliability','Quote levels']
        reports = []
        frame_before = self.frame.copy(deep=True)
        params_before = self.params.copy()
        observed, models = qc.run_comparison(self.frame, self.params, 'Validation', None,
            record_into(reports), versions)
        expected = pd.concat([full.loc[full.model.eq(name)] for name in versions], ignore_index=True)
        pd.testing.assert_frame_equal(observed, expected)
        pd.testing.assert_frame_equal(self.frame, frame_before)
        self.assertEqual(self.params, params_before)
        self.assertEqual(list(models), versions)
        self.assertEqual(observed.model.drop_duplicates().tolist(), versions)
        self.assertEqual(models['Base'].feature_name_, qc.BASE_FEATURES)
        for model in models.values():
            for key, value in self.params.items():
                self.assertEqual(model.get_params()[key], value)
        self.assertEqual(reports[0][:3], ('models',0,4))
        self.assertEqual(reports[-1][:3], ('models',4,4))
        completed = [r[1] for r in reports if r[0]=='models' and r[3].startswith('Finished')]
        self.assertEqual(completed, [1,2,3,4])
        self.assertEqual([r[1] for r in reports if r[0]=='fit'], [0,10,12]*4)
        self.assertTrue(observed.stage.eq('Validation').all())
        self.assertEqual(set(observed.row_id), set(self.frame.loc[self.frame.split.eq('Validation'),'row_id']))
        self.assertTrue(observed.train_n.eq(30).all())
        # Changing every test target and input cannot alter validation fits/rows.
        changed = self.frame.copy(deep=True)
        test = changed.split.eq('Test')
        for col in qc.BASE_FEATURES:
            changed.loc[test,col] = 'UNSEEN_TEST' if col in qc.BASE_CAT_FEATURES else -999.0
        changed.loc[test,[col for col in changed if col.startswith('bcq_')]] = -999.0
        changed.loc[test,['D_BM_SPREAD','BM_SPREAD']] = np.nan
        heldout_changed, _ = qc.run_comparison(changed, self.params, versions=versions)
        pd.testing.assert_frame_equal(observed, heldout_changed)

    def test_invalid_validation_versions_and_explicit_test_versions_fail_before_fit(self):
        with patch('lightgbm.LGBMRegressor.fit', side_effect=AssertionError('unexpected fit')):
            for versions, message in [([], 'nonempty'), (['Base','Base'], 'unique'),
                (['Base','Unknown'], 'Unknown'), (['Quote levels'], 'include Base'),
                ('Base', 'nonempty'), ([None,'Base'], 'model names'), (1, 'nonempty')]:
                with self.subTest(versions=versions), self.assertRaisesRegex(ValueError, message):
                    qc.run_comparison(self.frame, self.params, versions=versions)
            reports = []
            for versions in [[], ['Base'], ['Base','Quote levels','Reliability','Age decay']]:
                with self.subTest(test_versions=versions), self.assertRaisesRegex(ValueError, 'only supported for Validation'):
                    qc.run_comparison(self.frame, self.params, 'Test', 'Age decay',
                        record_into(reports), versions)
            self.assertFalse(reports)


if __name__ == '__main__':
    unittest.main()
