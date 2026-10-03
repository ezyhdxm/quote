# TEST SETUP LOGIC: 合成fixtures与断言；测试通过不代表真实预测增益。
"""Exercise real movement -> persisted sidecar -> one tiny LightGBM fit -> review.

Synthetic predictions are fixture inputs, never evidence of predictive benefit.
"""
from importlib.util import find_spec
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

import quote_quality_core as core
import quote_quality_movement as movement
import quote_quality_incremental as incremental
import quote_quality_diagnostics as diagnostics


# TEST LOGIC: ResearchIntegrationChecks；仅用于复现输入或核对行为。
@unittest.skipUnless(find_spec('lightgbm'), 'LightGBM required for integration smoke check')
class ResearchIntegrationChecks(unittest.TestCase):
    # TEST LOGIC: test_real_feature_schema_save_reload_two_single_fits_and_restore；仅用于复现输入或核对行为。
    def test_real_feature_schema_save_reload_two_single_fits_and_restore(self):
        start = pd.Timestamp('2026-03-02 10:00', tz='America/New_York')
        records, trades = [], []
        for k, bond in enumerate(['A', 'B', 'C']):
            for minute in range(60):
                row = {name: 1. for name in core.BASE_FEATURES}
                row.update(row_id=len(trades), cusip=bond, ISSUER='Issuer', SECTOR='Financial',
                    time=start + pd.Timedelta(minutes=minute),
                    split='Train' if minute < 40 else 'Validation' if minute < 50 else 'Test',
                    TRADE_TYPE='B', PREV_TRADE_TYPE='S', QUANTITY=100000., PREV_QUANTITY=100000.,
                    PREV_BM_SPREAD=.6, D_BM_SPREAD=np.sin(minute/10)/1000,
                    BM_SPREAD=.6 + np.sin(minute/10)/1000)
                trades.append(row)
            for minute in range(0, 60, 5):
                for firm, offset, quantity in [('D1', 0., 0), ('D2', .5, None)]:
                    for side, spread in [('bid', 65.), ('ask', 60.)]:
                        records.append(dict(cusip=bond, firm=firm, side=side,
                            quote_timestamp_ET=start + pd.Timedelta(minutes=minute),
                            spread=spread + k + minute*(k+1)/10 + offset, quantity=quantity))
        frame, quotes = pd.DataFrame(trades), pd.DataFrame(records)
        events = core.prepare_quote_events(quotes)
        events['cache_key'] = 'synthetic-known-event-key'
        old = core.build_quote_features(quotes, frame[['row_id', 'cusip', 'time']], event_cache=events)
        frame = frame.merge(old.drop(columns=['cusip', 'time']), on='row_id', validate='one_to_one')
        original = frame.copy(deep=True)
        specs = core.model_versions(frame)[1]
        params = dict(objective='mae', boosting_type='dart', n_estimators=4, learning_rate=.2,
            num_leaves=4, max_bin=31, min_child_samples=2, n_jobs=1, random_state=2026, verbosity=-1)
        versions = ['Base', 'Quote levels', 'Reliability', 'Age decay']
        meta = dict(base_features=core.BASE_FEATURES, target='D_BM_SPREAD', anchor='PREV_BM_SPREAD',
            error_multiplier=100, quote_spread_unit='bps', age_min=30, sync_min=1,
            allow_exact_quotes=True, lgb_params=params, validation_versions=versions,
            model_columns={name: columns for name, (columns, _) in specs.items()},
            event_cache_key=events['cache_key'], settings_origin='captured_before_fitting',
            locked_choice=None, test_available=False)
        predictions = []
        for name in versions:
            p = frame.loc[frame.split.eq('Validation'), ['row_id', 'time']].copy()
            p['stage'], p['model'], p['train_n'] = 'Validation', name, 120
            p['pred_spread'] = .6
            p['error_bps'] = (p.pred_spread - frame.set_index('row_id').loc[p.row_id, 'BM_SPREAD'].to_numpy())*100
            p['abs_error_bps'] = p.error_bps.abs(); predictions.append(p)
        predictions = pd.concat(predictions, ignore_index=True)
        sidecar = movement.build_movement_features(frame, events)
        self.assertEqual(len(sidecar), 150)
        self.assertTrue(sidecar.loc[sidecar.time.ge(start + pd.Timedelta(minutes=30)), 'bcq_bid_issuer_move_other_bond_n'].eq(2).all())
        with TemporaryDirectory() as folder:
            incremental.save_movement_sidecar(frame, sidecar, meta, folder=folder)
            restored, _ = incremental.load_movement_sidecar(frame, meta, folder=folder)
            pd.testing.assert_frame_equal(sidecar, restored)
            for block in ['Direction', 'Issuer']:
                new, model, info = incremental.run_incremental_validation(frame, restored, predictions, meta, block, folder=folder)
                self.assertEqual(model.n_estimators, 4)
                self.assertEqual(len(new), 30)
                self.assertTrue(np.isfinite(new.pred_spread).all())
                again, model2, recovered = incremental.run_incremental_validation(frame, restored, predictions, meta, block, folder=folder)
                self.assertIsNone(model2); self.assertTrue(recovered['reused'])
                pd.testing.assert_frame_equal(new, again)
                review = diagnostics.diagnostics(frame, pd.concat([predictions, new]), info['candidate'])
                self.assertEqual(review['reference'], 'Age decay')
                self.assertEqual(review['n'], 30)
        pd.testing.assert_frame_equal(frame, original)


if __name__ == '__main__': unittest.main()
