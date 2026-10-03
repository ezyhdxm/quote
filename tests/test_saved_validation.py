# TEST SETUP LOGIC: 合成fixtures与断言；测试通过不代表真实预测增益。
"""Same-target SECTOR review must not retrain or open locked test."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import quote_quality_saved as saved


# TEST FIXTURE LOGIC: fixture；仅用于复现输入或核对行为。
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


# TEST FIXTURE LOGIC: Actual rule inputs span both DST offsets and include targets outside file dates.
def rule_fixture():
    import quote_quality_core as core
    frame, _ = fixture()
    frame['cusip'] = 'X'
    frame['time'] = pd.DatetimeIndex(['2026-02-28 23:59', '2026-03-01 00:00',
        '2026-03-07 23:59', '2026-03-08 03:01', '2026-03-31 23:59', '2026-04-02 00:00'],
        tz='America/New_York')
    quotes = pd.DataFrame([dict(cusip='X', firm='A', side='bid', spread=0., quantity=0,
        quote_timestamp_ET=frame.time.iloc[1])])
    features = core.build_quote_features(quotes, frame[['row_id', 'cusip', 'time']])
    return frame.drop(columns=['bcq_has_quote', 'bcq_n_pair', 'bcq_n_size_time_pair']).merge(
        features.drop(columns=['time', 'cusip']), on='row_id', validate='one_to_one')


# TEST LOGIC: SavedValidationChecks；仅用于复现输入或核对行为。
class SavedValidationChecks(unittest.TestCase):
    # TEST LOGIC: test_target_sector_and_identical_row_deltas；仅用于复现输入或核对行为。
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

    # TEST LOGIC: test_unequal_rows_duplicate_stale_time_and_wrong_target_are_rejected；仅用于复现输入或核对行为。
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

    # TEST LOGIC: test_load_reads_validation_only_and_checks_dates；仅用于复现输入或核对行为。
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

    # TEST LOGIC: test_complete_png_without_large_table；仅用于复现输入或核对行为。
    def test_complete_png_without_large_table(self):
        f, p = fixture()
        result = saved.sector_diagnostics(f, p)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'sectors.png'
            saved.sector_figure(result).savefig(path)
            self.assertGreater(path.stat().st_size, 1000)

    # TEST LOGIC: test_trade_rule_queries_keep_absence_and_mask_unsupported_fallback；仅用于复现输入或核对行为。
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

    # TEST LOGIC: Both DST offsets retain exact NY dates; UTC endpoints convert, naive ones remain wall time.
    def test_rule_dates_across_dst_preserve_naive_and_aware_endpoint_semantics(self):
        frame = rule_fixture()
        retained = frame.copy(deep=True)
        endpoints = [
            ('2026-03-01T00:00:00-05:00', '2026-04-01T23:59:00-04:00'),
            (pd.Timestamp('2026-03-01T00:00:00-05:00'), pd.Timestamp('2026-04-01T23:59:00-04:00')),
            ('2026-03-01 00:30:00', '2026-04-01 23:59:00'),
            ('2026-03-02T03:00:00Z', '2026-04-02T03:59:00Z'),
            ('2026-03-01 00:30:00', '2026-04-01T23:59:00-04:00')]
        for start, end in endpoints:
            with self.subTest(start=start, end=end):
                result = saved.trade_rule_diagnostics(frame, start, end)
                self.assertEqual(result['start'], pd.Timestamp('2026-03-01', tz='America/New_York'))
                self.assertEqual(result['end'], pd.Timestamp('2026-04-01', tz='America/New_York'))
                self.assertEqual(result['n'], 4)
                self.assertEqual(result['outside_file_date_n'], 2)
                self.assertEqual(result['summary'].n.sum(), 4)
                # TEST LOGIC: The sole zero-spread quote remains valid; three later no-quote days stay in n.
                self.assertEqual(result['summary'].quote_n.sum(), 1)
        pd.testing.assert_frame_equal(frame, retained)

    # TEST LOGIC: Invalid or reversed bounds still fail before sector aggregation or plotting.
    def test_rule_dates_reject_invalid_and_reversed_bounds(self):
        frame = rule_fixture()
        for start, end in [('bad', '2026-04-01'), ('2026-03-01', None),
                           ('2026-04-01T00:00-04:00', '2026-03-01T00:00-05:00')]:
            with self.subTest(start=start, end=end):
                with self.assertRaisesRegex(ValueError, 'actual quote file start and end'):
                    saved.trade_rule_diagnostics(frame, start, end)


# TEST LOGIC: Step5CheckpointChecks；仅用于复现输入或核对行为。
class Step5CheckpointChecks(unittest.TestCase):
    # TEST LOGIC: test_unsaved_and_frame_only_are_clear_and_never_train_or_read_test；仅用于复现输入或核对行为。
    def test_unsaved_and_frame_only_are_clear_and_never_train_or_read_test(self):
        frame, _ = fixture()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with self.assertRaisesRegex(FileNotFoundError, 'No saved Step5 checkpoint/export'):
                saved.load_saved_validation(folder)
            with self.assertRaisesRegex(FileNotFoundError, 'No saved Step5 checkpoint/export'):
                saved.load_step5_metadata(folder)
            snapshot = saved.save_step5_checkpoint(frame, folder=folder)
            self.assertTrue((snapshot / 'model_features.parquet').is_file())
            with patch('pandas.read_parquet', side_effect=AssertionError('no validation to read')):
                with self.assertRaisesRegex(ValueError, 'no completed validation'):
                    saved.load_saved_validation(folder)

    # TEST LOGIC: test_round_trip_preserves_metadata_sector_and_existing_flat_export；仅用于复现输入或核对行为。
    def test_round_trip_preserves_metadata_sector_and_existing_flat_export(self):
        frame, predictions = fixture()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'experiment.json').write_text('legacy export remains unchanged')
            metadata = dict(locked_choice='Reliability', age_min=30, sync_min=1,
                            lgb_params={'boosting_type': 'dart', 'n_jobs': 8}, custom='research pilot')
            first = saved.save_step5_checkpoint(frame, predictions, metadata=metadata, folder=folder)
            first_bytes = {p.name: p.read_bytes() for p in first.iterdir()}
            second = saved.save_step5_checkpoint(frame, predictions, metadata=metadata, folder=folder)
            self.assertNotEqual(first, second)
            self.assertEqual((folder / 'experiment.json').read_text(), 'legacy export remains unchanged')
            self.assertEqual(first_bytes, {p.name: p.read_bytes() for p in first.iterdir()})
            a, b, manifest = saved.load_saved_validation(folder)
            pd.testing.assert_frame_equal(a, frame)
            pd.testing.assert_frame_equal(b, predictions)
            for key, value in metadata.items():
                self.assertEqual(manifest[key], value)
            self.assertTrue(manifest['validation_available'])
            self.assertEqual(manifest['error_multiplier'], 100)
            self.assertEqual(manifest['target'], 'D_BM_SPREAD')
            self.assertEqual(manifest['anchor'], 'PREV_BM_SPREAD')
            self.assertEqual(manifest['split_dates']['Validation'], sorted(frame.time.dt.strftime('%Y-%m-%d').unique()))
            self.assertEqual(set(manifest['validation_files_sha256']), {'model_features.parquet', 'validation_predictions.parquet'})

    # TEST LOGIC: test_validation_restore_never_reads_or_checks_test_predictions；仅用于复现输入或核对行为。
    def test_validation_restore_never_reads_or_checks_test_predictions(self):
        frame, predictions = fixture()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            snapshot = saved.save_step5_checkpoint(frame, predictions,
                predictions.assign(stage='Test'), metadata={'locked_choice': 'Base'}, folder=folder)
            # A damaged test file still cannot trigger test reads during validation recovery.
            (snapshot / 'test_predictions.parquet').write_bytes(b'test must not be read here')
            real_read = pd.read_parquet
            real_hash = saved._file_sha256
            with patch('pandas.read_parquet', wraps=real_read) as read, \
                    patch.object(saved, '_file_sha256', wraps=real_hash) as digest:
                _, _, manifest = saved.load_saved_validation(folder)
            self.assertTrue(manifest['test_available'])
            self.assertEqual([Path(call.args[0]).name for call in read.call_args_list],
                             ['model_features.parquet', 'validation_predictions.parquet'])
            self.assertNotIn('test_predictions.parquet', [Path(call.args[0]).name for call in digest.call_args_list])

    # TEST LOGIC: test_interrupted_write_keeps_previous_pointer_and_snapshot_usable；仅用于复现输入或核对行为。
    def test_interrupted_write_keeps_previous_pointer_and_snapshot_usable(self):
        frame, predictions = fixture()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            previous = saved.save_step5_checkpoint(frame, predictions, folder=folder)
            pointer = (folder / 'latest.json').read_bytes()
            files = {p.name: p.read_bytes() for p in previous.iterdir()}
            real_write = pd.DataFrame.to_parquet
            # TEST FIXTURE LOGIC: interrupt；仅用于复现输入或核对行为。
            def interrupt(source, path, *args, **kwargs):
                if Path(path).name == 'validation_predictions.parquet':
                    raise KeyboardInterrupt('simulated write interruption')
                return real_write(source, path, *args, **kwargs)
            with patch.object(pd.DataFrame, 'to_parquet', autospec=True, side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    saved.save_step5_checkpoint(frame, predictions, folder=folder)
            self.assertEqual((folder / 'latest.json').read_bytes(), pointer)
            self.assertEqual({p.name: p.read_bytes() for p in previous.iterdir()}, files)
            a, b, _ = saved.load_saved_validation(folder)
            pd.testing.assert_frame_equal(a, frame)
            pd.testing.assert_frame_equal(b, predictions)
            with patch.object(Path, 'replace', side_effect=OSError('simulated pointer publication failure')):
                with self.assertRaisesRegex(OSError, 'publication failure'):
                    saved.save_step5_checkpoint(frame, predictions, folder=folder)
            self.assertEqual((folder / 'latest.json').read_bytes(), pointer)
            self.assertFalse(list(folder.glob('.latest-*.json')))
            saved.load_saved_validation(folder)

    # TEST LOGIC: test_first_interruption_does_not_publish_a_pointer；仅用于复现输入或核对行为。
    def test_first_interruption_does_not_publish_a_pointer(self):
        frame, predictions = fixture()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            with patch.object(pd.DataFrame, 'to_parquet', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    saved.save_step5_checkpoint(frame, predictions, folder=folder)
            self.assertFalse((folder / 'latest.json').exists())
            with self.assertRaisesRegex(FileNotFoundError, 'No saved Step5 checkpoint/export'):
                saved.load_saved_validation(folder)

    # TEST LOGIC: test_missing_and_wrong_snapshot_checksums_are_rejected；仅用于复现输入或核对行为。
    def test_missing_and_wrong_snapshot_checksums_are_rejected(self):
        frame, predictions = fixture()
        for mutation in ['manifest', 'missing', 'wrong']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                snapshot = saved.save_step5_checkpoint(frame, predictions, folder=folder)
                manifest_path = snapshot / 'experiment.json'
                manifest = json.loads(manifest_path.read_text())
                if mutation == 'missing':
                    del manifest['validation_files_sha256']['validation_predictions.parquet']
                elif mutation == 'wrong':
                    manifest['validation_files_sha256']['validation_predictions.parquet'] = '0' * 64
                    manifest['files_sha256']['validation_predictions.parquet'] = '0' * 64
                else:
                    manifest['locked_choice'] = 'tampered'
                manifest_path.write_text(json.dumps(manifest))
                if mutation != 'manifest':
                    pointer = json.loads((folder / 'latest.json').read_text())
                    pointer['experiment_sha256'] = saved._file_sha256(manifest_path)
                    (folder / 'latest.json').write_text(json.dumps(pointer))
                with patch('pandas.read_parquet', side_effect=AssertionError('reject before reading parquet')):
                    with self.assertRaisesRegex(ValueError, 'checksum|checksums'):
                        saved.load_saved_validation(folder)

    # TEST LOGIC: test_bad_snapshot_path_and_missing_data_do_not_fall_back；仅用于复现输入或核对行为。
    def test_bad_snapshot_path_and_missing_data_do_not_fall_back(self):
        frame, predictions = fixture()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            snapshot = saved.save_step5_checkpoint(frame, predictions, folder=folder)
            pointer = json.loads((folder / 'latest.json').read_text())
            for path in ['../external', '/tmp/external']:
                (folder / 'latest.json').write_text(json.dumps(dict(pointer, snapshot=path)))
                with self.assertRaisesRegex(ValueError, 'inside its output folder'):
                    saved.load_saved_validation(folder)
            (folder / 'latest.json').write_text(json.dumps(pointer))
            (snapshot / 'model_features.parquet').unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'validation file is missing'):
                saved.load_saved_validation(folder)

    # TEST LOGIC: test_metadata_restores_lock_without_validation_or_any_parquet_access；仅用于复现输入或核对行为。
    def test_metadata_restores_lock_without_validation_or_any_parquet_access(self):
        frame, predictions = fixture()
        for completed in [False, True]:
            with self.subTest(completed=completed), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                snapshot = saved.save_step5_checkpoint(frame, predictions if completed else None,
                    predictions.assign(stage='Test') if completed else None,
                    metadata={'locked_choice': 'Reliability'}, folder=folder)
                # Metadata restoration is independent of all feature/metric files.
                for path in snapshot.glob('*.parquet'):
                    path.write_bytes(b'must not read or hash during metadata restoration')
                real_hash = saved._file_sha256
                with patch('pandas.read_parquet', side_effect=AssertionError('metadata cannot open parquet')), \
                        patch.object(saved, '_file_sha256', wraps=real_hash) as digest:
                    manifest, restored_path = saved.load_step5_metadata(folder)
                self.assertEqual(manifest['locked_choice'], 'Reliability')
                self.assertEqual(manifest['validation_available'], completed)
                self.assertEqual(restored_path, snapshot.resolve())
                self.assertEqual([Path(call.args[0]).name for call in digest.call_args_list], ['experiment.json'])
                payload = json.loads((snapshot / 'experiment.json').read_text())
                payload['locked_choice'] = None
                (snapshot / 'experiment.json').write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    saved.load_step5_metadata(folder)

    # TEST LOGIC: test_metadata_old_flat_lock_works_without_prediction_files；仅用于复现输入或核对行为。
    def test_metadata_old_flat_lock_works_without_prediction_files(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            payload = {'locked_choice': 'Age decay', 'validation_available': False}
            (folder / 'experiment.json').write_text(json.dumps(payload))
            with patch('pandas.read_parquet', side_effect=AssertionError('metadata cannot open parquet')):
                manifest, restored_path = saved.load_step5_metadata(folder)
            self.assertEqual(manifest, payload)
            self.assertEqual(restored_path, folder.resolve())


if __name__ == '__main__': unittest.main()
