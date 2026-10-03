"""Same-target SECTOR review must not retrain or open locked test."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import quote_quality_saved as saved


def fixture():
    times = pd.date_range('2026-03-19 10:00', periods=6, freq='12h', tz='America/New_York')
    f = pd.DataFrame(dict(row_id=range(6), time=times, split='Validation',
                          SECTOR=['Tech', 'Tech', 'Energy', 'Energy', None, ''],
                          BM_SPREAD=.6, QUANTITY=[50_000, 2_000_000] * 3,
                          bcq_has_quote=[0, 1, 1, 1, 0, 1], bcq_n_pair=[0, 1, 0, 1, 0, 1],
                          bcq_n_size_time_pair=[0, 0, 0, 1, 0, 1]))
    parts = []
    for name, e in [('Base', [2.] * 6), ('Quote levels', [1., 2., 3., 2., 1., 2.]),
                    ('Reliability', [1., 1., 2., 2., 1., 1.])]:
        p = f[['row_id', 'time']].copy()
        p['model'] = name; p['stage'] = 'Validation'; p['train_n'] = 30
        p['abs_error_bps'] = e; p['error_bps'] = e; p['pred_spread'] = .6 + np.asarray(e) / 100
        parts.append(p)
    return f, pd.concat(parts, ignore_index=True)


class SavedValidationChecks(unittest.TestCase):
    def test_target_sector_and_identical_row_deltas(self):
        f, p = fixture()
        r = saved.sector_diagnostics(f, p, 'Quote levels')
        self.assertEqual(r['reference'], 'Base')
        self.assertEqual(r['summary'].loc['Unknown', 'n'], 2)
        self.assertEqual(r['summary'].loc['Tech', 'delta_mae'], -.5)
        self.assertEqual(r['summary'].loc['Energy', 'delta_mae'], .5)
        self.assertEqual(r['coverage'].loc['Unknown', 'Any quote'], .5)
        self.assertEqual(r['n'], 6)
        self.assertEqual(saved.sector_diagnostics(f, p, 'Reliability')['reference'], 'Quote levels')

    def test_unequal_rows_duplicate_stale_time_and_wrong_target_are_rejected(self):
        f, p = fixture()
        for bad in [p.iloc[1:], pd.concat([p, p.iloc[:1]])]:
            with self.assertRaises(ValueError): saved.validation_rows(f, bad)
        bad = p.copy(); bad.loc[0, 'time'] += pd.Timedelta(seconds=1)
        with self.assertRaises(ValueError): saved.validation_rows(f, bad)
        bad = f.copy(); bad.loc[0, 'BM_SPREAD'] = .7
        with self.assertRaises(ValueError): saved.validation_rows(bad, p)
        with self.assertRaises(ValueError): saved.sector_diagnostics(f.drop(columns='SECTOR'), p)
        with self.assertRaises(ValueError): saved.validation_rows(f, p.assign(stage='Test'))
        with self.assertRaises(ValueError): saved.validation_rows(f, p.drop(columns='error_bps'))
        with self.assertRaises(ValueError): saved.validation_rows(f, p.assign(abs_error_bps=-1.))

    def test_load_reads_validation_only_and_checks_dates(self):
        f, p = fixture()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            f.to_parquet(folder / 'model_features.parquet', index=False)
            p.to_parquet(folder / 'validation_predictions.parquet', index=False)
            manifest = dict(split_dates={'Validation': sorted(f.time.dt.strftime('%Y-%m-%d').unique())}, locked_choice='Base')
            (folder / 'experiment.json').write_text(json.dumps(manifest))
            real_read = pd.read_parquet
            with patch('pandas.read_parquet', wraps=real_read) as read:
                a, b, m = saved.load_saved_validation(folder)
            self.assertEqual([Path(c.args[0]).name for c in read.call_args_list], ['model_features.parquet', 'validation_predictions.parquet'])
            self.assertEqual(m['locked_choice'], 'Base')
            pd.testing.assert_frame_equal(a, f)
            manifest['split_dates']['Validation'] = ['2026-03-01']
            (folder / 'experiment.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): saved.load_saved_validation(folder)

    def test_complete_png_without_large_table(self):
        f, p = fixture()
        result = saved.sector_diagnostics(f, p)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'sectors.png'
            saved.sector_figure(result).savefig(path)
            self.assertGreater(path.stat().st_size, 1000)

    def test_trade_rule_queries_keep_absence_and_mask_unsupported_fallback(self):
        import quote_quality_core as core
        f, _ = fixture()
        f['cusip'] = 'X'
        q = pd.DataFrame([dict(cusip='X', firm='A', side='bid', spread=60., quantity=0,
                               quote_timestamp_ET=f.time.iloc[0])])
        features = core.build_quote_features(q, f[['row_id', 'cusip', 'time']])
        f = f.drop(columns=['bcq_has_quote', 'bcq_n_pair', 'bcq_n_size_time_pair']).merge(
            features.drop(columns=['time', 'cusip']), on='row_id', validate='one_to_one')
        f['bcq_bid_center_candidate_clip'] += 100  # unsupported fallback must never be scored as assessed
        result = saved.trade_rule_diagnostics(f, f.time.iloc[0], f.time.iloc[2])
        self.assertEqual(result['n'] + result['outside_file_date_n'], len(f))
        self.assertLess(result['n'], len(f))
        clip = result['effects'].loc[result['effects'].rule.eq('Clip')]
        self.assertEqual(clip.n.sum(), 0)
        self.assertTrue(clip.p95.isna().all())
        self.assertEqual(int(result['summary'].quote_n.sum()), 2)
        self.assertGreater(int(result['summary'].n.sum()), int(result['summary'].quote_n.sum()))
        with tempfile.TemporaryDirectory() as temp:
            saved.trade_rule_figure(result).savefig(Path(temp) / 'rule_effects.png')


if __name__ == '__main__': unittest.main()
