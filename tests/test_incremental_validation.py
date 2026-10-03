# TEST SETUP LOGIC: 合成fixtures与断言；测试通过不代表真实预测增益。
"""Incremental validation must fit once, preserve the original cohort and recover."""
from copy import deepcopy
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quote_quality_core as qc
import quote_quality_incremental as qi


DIR = ['bcq_bid_move_mean_bps', 'bcq_bid_move_common_n',
       'bcq_ask_move_mean_bps', 'bcq_ask_move_common_n']
ISS = ['bcq_bid_issuer_move_mean_bps', 'bcq_bid_issuer_move_other_bond_n',
       'bcq_ask_issuer_move_mean_bps', 'bcq_ask_issuer_move_other_bond_n']


# TEST LOGIC: RecorderRegressor；仅用于复现输入或核对行为。
class RecorderRegressor:
    constructed = []
    fits = []
    predicts = []

    # TEST FIXTURE LOGIC: __init__；仅用于复现输入或核对行为。
    def __init__(self, **params):
        self.params = deepcopy(params)
        self.n_estimators = params['n_estimators']
        self.constructed.append(self)

    # TEST FIXTURE LOGIC: fit；仅用于复现输入或核对行为。
    def fit(self, x, y, **kwargs):
        self.fits.append((x.copy(), y.copy(), deepcopy(kwargs.get('categorical_feature'))))
        for iteration in range(self.n_estimators):
            env = type('Env', (), {'iteration': iteration, 'begin_iteration': 0,
                                   'end_iteration': self.n_estimators})()
            for callback in kwargs.get('callbacks', []):
                callback(env)
        return self

    # TEST FIXTURE LOGIC: predict；仅用于复现输入或核对行为。
    def predict(self, x):
        self.predicts.append(x.copy())
        return np.full(len(x), .015)


# TEST FIXTURE LOGIC: fixture；仅用于复现输入或核对行为。
def fixture():
    start = pd.Timestamp('2026-03-02 10:00', tz='America/New_York')
    frame = pd.DataFrame({column: np.linspace(0, 1, 60) for column in qc.BASE_FEATURES})
    frame['PREV_TRADE_TYPE'] = 'B'
    frame['TRADE_TYPE'] = ['B'] * 40 + ['VALIDATION_ONLY'] * 10 + ['TEST_ONLY'] * 10
    frame['PREV_BM_SPREAD'] = .6
    frame['D_BM_SPREAD'] = np.linspace(.01, .03, 60)
    frame['BM_SPREAD'] = frame.PREV_BM_SPREAD + frame.D_BM_SPREAD
    frame['row_id'] = np.arange(60)
    frame['cusip'], frame['ISSUER'], frame['SECTOR'] = 'X', 'ISS', 'Financial'
    frame['time'] = pd.date_range(start, periods=60, freq='min')
    frame['split'] = ['Train'] * 40 + ['Validation'] * 10 + ['Test'] * 10
    raw = pd.DataFrame([dict(cusip='X', firm=firm, side=side,
        quote_timestamp_ET=start + pd.Timedelta(minutes=minute), spread=70 + minute / 10 + j,
        quantity=2) for minute in range(0, 61, 10) for j, firm in enumerate(['A', 'B'])
        for side in ['bid', 'ask']])
    f = qc.build_quote_features(raw, frame[['row_id', 'cusip', 'time']])
    frame = frame.merge(f.drop(columns=['cusip', 'time']), on='row_id', validate='one_to_one')
    frame['bcq_bid_center_decay'] = 123.5  # Distinguishable from center_equal.
    params = dict(objective='mae', boosting_type='dart', n_estimators=12,
                  learning_rate=.2, num_leaves=4, max_bin=31, n_jobs=1,
                  random_state=2026, min_child_samples=2, verbosity=-1)
    columns = qc.model_versions(frame)[1]
    metadata = dict(base_features=list(qc.BASE_FEATURES), target='D_BM_SPREAD',
        anchor='PREV_BM_SPREAD', error_multiplier=100, quote_spread_unit='bps',
        allow_exact_quotes=True, age_min=30, sync_min=1, lgb_params=params,
        validation_versions=['Base', 'Quote levels', 'Reliability', 'Age decay'],
        model_columns={name: cols for name, (cols, _) in columns.items()},
        settings_origin='captured_before_fitting', event_cache_key='original-events',
        locked_choice=None, test_available=False, input_files={'quotes': {'size': 12}},
        split_dates={'Validation': ['2026-03-02']})
    predictions = []
    for name in metadata['validation_versions']:
        p = frame.loc[frame.split.eq('Validation'), ['row_id', 'time']].copy()
        p['model'], p['stage'], p['pred_spread'] = name, 'Validation', .611
        p['error_bps'] = (p.pred_spread - frame.set_index('row_id').loc[p.row_id, 'BM_SPREAD'].to_numpy()) * 100
        p['abs_error_bps'], p['train_n'] = p.error_bps.abs(), 40
        predictions.append(p)
    sidecar = frame.loc[frame.split.isin(['Train', 'Validation']), ['row_id', 'cusip', 'time']].copy()
    for c in DIR + ISS:
        sidecar[c] = 2. if c.endswith('_n') else np.linspace(-1, 1, len(sidecar))
    sidecar.attrs = dict(source_frame_sha256=qi.frame_fingerprint(frame),
        event_cache_key=metadata['event_cache_key'], feature_dictionary={'example': {'unit': 'bps'}},
        movement_config=dict(lookback_min=30., age_min=30., allow_exact=True,
            include_issuer=True, same_et_day=True, issuer_min_other_bonds=2,
            issuer_vote='one bond; guarded common-dealer mean change'))
    return frame, sidecar, pd.concat(predictions, ignore_index=True), metadata


# TEST LOGIC: IncrementalValidationChecks；仅用于复现输入或核对行为。
class IncrementalValidationChecks(unittest.TestCase):
    # TEST FIXTURE LOGIC: setUp；仅用于复现输入或核对行为。
    def setUp(self):
        self.frame, self.sidecar, self.reference, self.metadata = fixture()
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.features = patch.object(qi, '_feature_lists', return_value=(DIR, ISS))
        self.features.start(); self.addCleanup(self.features.stop)
        RecorderRegressor.constructed, RecorderRegressor.fits, RecorderRegressor.predicts = [], [], []
        self.regressor = patch('lightgbm.LGBMRegressor', RecorderRegressor)
        self.regressor.start(); self.addCleanup(self.regressor.stop)
        self.addCleanup(self.tmp.cleanup)

    # TEST FIXTURE LOGIC: run_block；仅用于复现输入或核对行为。
    def run_block(self, block='Direction', **kwargs):
        return qi.run_incremental_validation(self.frame, self.sidecar, self.reference,
            self.metadata, block, folder=self.folder, **kwargs)

    # TEST LOGIC: test_one_fit_uses_exact_original_budget_train_categories_and_age_decay；仅用于复现输入或核对行为。
    def test_one_fit_uses_exact_original_budget_train_categories_and_age_decay(self):
        reports = []
        original = self.frame.copy(deep=True)
        params = deepcopy(self.metadata['lgb_params'])
        out, model, info = self.run_block(progress=lambda *args: reports.append(args))
        self.assertEqual(len(RecorderRegressor.constructed), 1)
        self.assertEqual(len(RecorderRegressor.fits), 1)
        self.assertEqual(model.params, params)
        x, y, cats = RecorderRegressor.fits[0]
        self.assertEqual(len(x), 40); self.assertEqual(set(y.index), set(range(40)))
        self.assertEqual(cats, qc.BASE_CAT_FEATURES)
        self.assertEqual(list(x), self.metadata['model_columns']['Age decay'] + DIR)
        np.testing.assert_allclose(x.bcq_bid_anchor_gap, 63.5)
        self.assertEqual(list(x.TRADE_TYPE.cat.categories), ['B'])
        self.assertTrue(RecorderRegressor.predicts[0].TRADE_TYPE.isna().all())
        self.assertEqual(set(out.row_id), set(range(40, 50)))
        self.assertTrue(out.model.eq('Age decay + Direction').all())
        self.assertTrue(out.stage.eq('Validation').all())
        np.testing.assert_allclose(out.pred_spread, .615)
        truth = self.frame.set_index('row_id').loc[out.row_id, 'BM_SPREAD'].to_numpy()
        np.testing.assert_allclose(out.error_bps, (out.pred_spread - truth) * 100)
        pd.testing.assert_frame_equal(self.frame, original)
        self.assertEqual(params, self.metadata['lgb_params'])
        self.assertFalse(info['locked_test_supported'])
        self.assertFalse(info['reused'])
        self.assertEqual([r[1] for r in reports if r[0] == 'fit'], [0, 10, 12])
        self.assertEqual(reports[-1][:3], ('models', 1, 1))

    # TEST LOGIC: test_repeated_completed_block_recovers_and_test_values_do_not_change_identity；仅用于复现输入或核对行为。
    def test_repeated_completed_block_recovers_and_test_values_do_not_change_identity(self):
        out, _, info = self.run_block()
        before = (Path(info['snapshot_path']) / 'completed.json').read_bytes()
        test = self.frame.split.eq('Test')
        self.frame.loc[test, ['D_BM_SPREAD', 'BM_SPREAD']] = np.nan
        self.frame.loc[test, 'COUPON'] = -99999.
        self.frame.loc[test, 'SECTOR'] = 'UNSEEN_TEST'
        again, model, recovered = self.run_block()
        pd.testing.assert_frame_equal(out, again)
        self.assertIsNone(model); self.assertTrue(recovered['reused'])
        self.assertEqual(len(RecorderRegressor.fits), 1)
        self.assertEqual(before, (Path(info['snapshot_path']) / 'completed.json').read_bytes())

    # TEST LOGIC: test_two_blocks_are_independent_and_no_original_model_is_refitted；仅用于复现输入或核对行为。
    def test_two_blocks_are_independent_and_no_original_model_is_refitted(self):
        self.run_block('Direction'); out, _, info = self.run_block('Issuer')
        self.assertEqual(len(RecorderRegressor.fits), 2)
        self.assertEqual(list(RecorderRegressor.fits[1][0]), self.metadata['model_columns']['Age decay'] + ISS)
        self.assertFalse(set(DIR).intersection(RecorderRegressor.fits[1][0]))
        self.assertTrue(out.model.eq('Age decay + Issuer').all())
        self.assertEqual(info['reference'], 'Age decay')
        with self.assertRaisesRegex(ValueError, 'Only Direction'):
            self.run_block('Test')
        self.assertEqual(len(RecorderRegressor.fits), 2)

    # TEST LOGIC: test_changed_completed_request_refuses_additional_fit；仅用于复现输入或核对行为。
    def test_changed_completed_request_refuses_additional_fit(self):
        self.run_block()
        for kind in ['values', 'config', 'params', 'reference']:
            frame, sidecar, reference, metadata = fixture()
            if kind == 'values': sidecar.loc[0, DIR[0]] += 1
            if kind == 'config': sidecar.attrs['movement_config']['issuer_vote'] = 'changed'
            if kind == 'params': metadata['lgb_params']['n_estimators'] += 1
            if kind == 'reference':
                reference.loc[reference.model.eq('Age decay'), 'pred_spread'] += .001
                reference['error_bps'] = (reference.pred_spread - frame.set_index('row_id').loc[reference.row_id, 'BM_SPREAD'].to_numpy()) * 100
                reference['abs_error_bps'] = reference.error_bps.abs()
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'different saved request'):
                qi.run_incremental_validation(frame, sidecar, reference, metadata,
                    'Direction', folder=self.folder)
        self.assertEqual(len(RecorderRegressor.fits), 1)

    # TEST LOGIC: test_locked_exposed_unknown_budget_and_provenance_fail_before_fit；仅用于复现输入或核对行为。
    def test_locked_exposed_unknown_budget_and_provenance_fail_before_fit(self):
        for key, value in [('locked_choice', 'Age decay'), ('test_available', True),
                           ('saved_lock_guard', True), ('test_exposed', True)]:
            meta = dict(self.metadata, **{key: value})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'locked'):
                qi.run_incremental_validation(self.frame, self.sidecar, self.reference, meta,
                    'Direction', folder=self.folder)
        for missing in ['locked_choice', 'test_available', 'lgb_params']:
            meta = deepcopy(self.metadata); meta.pop(missing)
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                qi.run_incremental_validation(self.frame, self.sidecar, self.reference, meta,
                    'Direction', folder=self.folder)
        for origin in [None, 'legacy_live_settings_at_capture']:
            meta = dict(self.metadata, settings_origin=origin)
            with self.subTest(origin=origin), self.assertRaisesRegex(ValueError, 'provenance|Legacy'):
                qi.run_incremental_validation(self.frame, self.sidecar, self.reference, meta,
                    'Direction', folder=self.folder)
        self.assertFalse(RecorderRegressor.constructed)

    # TEST LOGIC: test_source_row_time_quote_key_and_reference_mismatches_fail_before_fit；仅用于复现输入或核对行为。
    def test_source_row_time_quote_key_and_reference_mismatches_fail_before_fit(self):
        for kind in ['frame', 'duplicate', 'test_row', 'time', 'cusip', 'event_key', 'missing_event_key', 'no_provenance', 'inf']:
            frame, sidecar, reference, metadata = fixture()
            if kind == 'frame': frame.loc[0, 'COUPON'] += .1
            if kind == 'duplicate': sidecar.loc[1, 'row_id'] = 0
            if kind == 'test_row': sidecar.loc[0, 'row_id'] = 59
            if kind == 'time': sidecar.loc[0, 'time'] += pd.Timedelta(seconds=1)
            if kind == 'cusip': sidecar.loc[0, 'cusip'] = 'OTHER'
            if kind == 'event_key': sidecar.attrs['event_cache_key'] = 'other-quotes'
            if kind == 'missing_event_key':
                sidecar.attrs['event_cache_key'] = None
                metadata['event_cache_key'] = None
            if kind == 'no_provenance': sidecar.attrs = {}
            if kind == 'inf': sidecar.loc[0, DIR[0]] = np.inf
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                qi.run_incremental_validation(frame, sidecar, reference, metadata,
                    'Direction', folder=self.folder)
        for kind in ['timestamp', 'cohort', 'error', 'train_n']:
            p = self.reference.copy()
            if kind == 'timestamp': p.loc[0, 'time'] += pd.Timedelta(seconds=1)
            if kind == 'cohort': p = p.iloc[1:]
            if kind == 'error': p.loc[0, 'error_bps'] += 1
            if kind == 'train_n': p['train_n'] = 99
            with self.subTest(reference=kind), self.assertRaises(ValueError):
                qi.run_incremental_validation(self.frame, self.sidecar, p, self.metadata,
                    'Direction', folder=self.folder)
        self.assertFalse(RecorderRegressor.constructed)

    # TEST LOGIC: test_unknown_movement_is_unassessed_despite_finite_counts；仅用于复现输入或核对行为。
    def test_unknown_movement_is_unassessed_despite_finite_counts(self):
        for split in ['Train', 'Validation']:
            sidecar = self.sidecar.copy(deep=True)
            ids = self.frame.loc[self.frame.split.eq(split), 'row_id']
            sidecar.loc[sidecar.row_id.isin(ids), [DIR[0], DIR[2]]] = np.nan
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, 'unassessed'):
                qi.run_incremental_validation(self.frame, sidecar, self.reference, self.metadata,
                    'Direction', folder=self.folder)
        self.assertFalse(RecorderRegressor.constructed)

    # TEST LOGIC: test_sidecar_roundtrip_manifest_not_parquet_attrs_binds_content；仅用于复现输入或核对行为。
    def test_sidecar_roundtrip_manifest_not_parquet_attrs_binds_content(self):
        sentinel = self.folder / 'latest.json'; sentinel.write_text('ORIGINAL_STEP5_POINTER')
        saved = qi.save_movement_sidecar(self.frame, self.sidecar, self.metadata, self.folder)
        restored, manifest = qi.load_movement_sidecar(self.frame, self.metadata, self.folder)
        pd.testing.assert_frame_equal(restored, self.sidecar)
        self.assertEqual(Path(manifest['snapshot_path']), saved)
        self.assertEqual(sentinel.read_text(), 'ORIGINAL_STEP5_POINTER')
        # No reliance on Arrow attrs: the companion manifest supplies all binding fields.
        no_attrs = restored.copy(); no_attrs.attrs = {}
        with patch('quote_quality_incremental.pd.read_parquet', return_value=no_attrs):
            loaded, _ = qi.load_movement_sidecar(self.frame, self.metadata, self.folder)
            self.assertEqual(loaded.attrs['source_frame_sha256'], qi.frame_fingerprint(self.frame))
        (saved / 'movement_features.parquet').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            qi.load_movement_sidecar(self.frame, self.metadata, self.folder)
        changed = self.frame.copy(); changed.loc[0, 'COUPON'] += 1
        with self.assertRaises(FileNotFoundError):
            qi.load_movement_sidecar(changed, self.metadata, self.folder)

    # TEST LOGIC: test_save_interruption_preserves_old_sidecar_pointer_and_original_pointer；仅用于复现输入或核对行为。
    def test_save_interruption_preserves_old_sidecar_pointer_and_original_pointer(self):
        first = qi.save_movement_sidecar(self.frame, self.sidecar, self.metadata, self.folder)
        original_pointer = first.parent / 'latest.json'
        old = original_pointer.read_bytes()
        with patch('quote_quality_incremental._atomic_json', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                qi.save_movement_sidecar(self.frame, self.sidecar, self.metadata, self.folder)
        self.assertEqual(old, original_pointer.read_bytes())
        restored, _ = qi.load_movement_sidecar(self.frame, self.metadata, self.folder)
        pd.testing.assert_frame_equal(restored, self.sidecar)

    # TEST LOGIC: test_failed_fit_has_no_completed_record_and_is_not_silently_retrained；仅用于复现输入或核对行为。
    def test_failed_fit_has_no_completed_record_and_is_not_silently_retrained(self):
        with patch.object(RecorderRegressor, 'fit', side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt): self.run_block()
        self.assertFalse(list(self.folder.rglob('completed.json')))
        with self.assertRaisesRegex(RuntimeError, 'incomplete prior attempt'):
            self.run_block()
        self.assertEqual(len(RecorderRegressor.constructed), 1)

    # TEST LOGIC: test_completed_run_checksum_failure_refuses_fit；仅用于复现输入或核对行为。
    def test_completed_run_checksum_failure_refuses_fit(self):
        _, _, info = self.run_block()
        result = Path(info['snapshot_path']) / 'validation_predictions.parquet'
        result.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.run_block()
        self.assertEqual(len(RecorderRegressor.fits), 1)

    # TEST LOGIC: test_real_movement_contract_build_save_load_and_one_candidate_fit；仅用于复现输入或核对行为。
    def test_real_movement_contract_build_save_load_and_one_candidate_fit(self):
        from quote_quality_movement import build_movement_features, DIRECTION_FEATURES
        self.features.stop()
        start = self.frame.time.min()
        raw = pd.DataFrame([dict(cusip='X', firm=firm, side=side,
            quote_timestamp_ET=start + pd.Timedelta(minutes=minute), spread=60 + minute,
            quantity=2) for minute in range(0, 61, 10) for firm in ['A', 'B']
            for side in ['bid', 'ask']])
        cache = qc.prepare_quote_events(raw)
        cache['cache_key'] = self.metadata['event_cache_key']
        sidecar = build_movement_features(self.frame, cache, include_issuer=False)
        saved = qi.save_movement_sidecar(self.frame, sidecar, self.metadata, self.folder)
        restored, info = qi.load_movement_sidecar(self.frame, self.metadata, self.folder)
        self.assertEqual(len(restored), 50)
        self.assertEqual(Path(info['snapshot_path']), saved)
        out, _, run = qi.run_incremental_validation(self.frame, restored, self.reference,
            self.metadata, 'Direction', folder=self.folder)
        self.assertEqual(len(RecorderRegressor.fits), 1)
        self.assertEqual(list(RecorderRegressor.fits[0][0]), self.metadata['model_columns']['Age decay'] + DIRECTION_FEATURES)
        self.assertEqual(len(out), 10)
        self.assertGreater(run['support_counts']['Train'], 0)


if __name__ == '__main__':
    unittest.main()
