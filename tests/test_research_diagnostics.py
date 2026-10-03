# TEST SETUP LOGIC: 合成fixtures与断言；测试通过不代表真实预测增益。
"""Read-only research review: common cohorts, fixed cuts and frozen local cases."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import quote_quality_diagnostics as review


# TEST FIXTURE LOGIC: fixture；仅用于复现输入或核对行为。
def fixture():
    n = 30
    time = pd.Series([pd.Timestamp('2026-03-19', tz='America/New_York') +
                      pd.Timedelta(days=i // 10, hours=9 + i % 10) for i in range(n)])
    frame = pd.DataFrame(dict(row_id=range(n), time=time, split='Validation', BM_SPREAD=.6,
        cusip=[f'B{i % 10}' for i in range(n)], ISSUER=[f'I{i % 10}' for i in range(n)],
        SECTOR=['Energy', 'Utility', None] * 10, QUANTITY=[50_000, 500_000, 2_000_000] * 10,
        TRADE_TYPE=['B', 'S', None] * 10, NUM_OF_ISSUER_TRADES_SINCE_PREV=[0, 2, np.nan] * 10))
    for side in ['bid', 'ask']:
        count = np.asarray([0, 1, 2, 4, 6] * 6, dtype=float)
        if side == 'ask': count[1::5] = 0
        frame[f'bcq_{side}_n_dealers'] = count
        frame[f'bcq_{side}_n_peer_supported'] = np.where(count >= 4, 1, 0)
        frame[f'bcq_{side}_median_message_age_min'] = [0, 5, 5.01, 30, 30.01] * 6
        frame[f'bcq_{side}_median_change_age_min'] = [np.nan, 5, 30, 31, -1] * 6
        frame[f'bcq_{side}_mean_candidate_gap'] = [0, 2, 2.1, 10, 10.1] * 6
        frame[f'bcq_{side}_dispersion_bps'] = [0, 1, 5, 20, np.nan] * 6
        frame[f'bcq_{side}_zero_quantity_fraction'] = [np.nan, 0, .5, 1, 0] * 6
        frame[f'bcq_{side}_unknown_quantity_fraction'] = [np.nan, 0, 1, 0, .5] * 6
    frame['bcq_n_pair'] = [0, 0, 1, 2, 1] * 6
    frame['bcq_n_size_time_pair'] = [0, 0, 0, 1, 1] * 6
    frame['bcq_has_quote'] = (frame.bcq_bid_n_dealers + frame.bcq_ask_n_dealers).gt(0).astype(int)
    frame['PREV_EFFECTIVE_DATETIME_TS'] = frame.time - pd.to_timedelta([1, 5, 5.1, 30, 31] * 6, unit='min')
    train = frame.iloc[:5].copy()
    train['row_id'] += 100
    train['split'] = 'Train'
    train['time'] -= pd.Timedelta(days=4)
    frame = pd.concat([frame, train], ignore_index=True)
    errors = {'Base': np.full(n, 3.), 'Quote levels': np.full(n, 2.5),
              'Reliability': 2 + np.arange(n) % 4 / 2,
              'Age decay': .5 + np.arange(n) % 6 * 2.5}
    errors['Age decay + Direction'] = errors['Age decay'] - .2
    errors['Age decay + Issuer'] = errors['Age decay'] + .1
    parts = []
    for model, values in errors.items():
        p = frame.loc[frame.split.eq('Validation'), ['row_id', 'time']].copy()
        p['model'], p['stage'] = model, 'Validation'
        p['error_bps'], p['abs_error_bps'] = values, values
        p['pred_spread'] = .6 + values / 100
        parts.append(p)
    return frame, pd.concat(parts, ignore_index=True)


# TEST FIXTURE LOGIC: impacts_fixture；仅用于复现输入或核对行为。
def impacts_fixture(frame):
    parts = []
    for side in ['bid', 'ask']:
        z = frame.loc[frame.split.eq('Validation'), ['cusip', 'time']].copy()
        z['day'] = z.time.dt.normalize()
        z = z[['cusip', 'day']].drop_duplicates().reset_index(drop=True)
        z['side'] = side
        z['max_center_effect_bps'] = np.arange(len(z)) % 4
        z['coverage_loss_fraction'] = (np.arange(len(z)) % 3) / 3
        z['impact_queries'] = 1
        parts.append(z)
    impacts = pd.concat(parts, ignore_index=True)
    impacts.attrs['query_source'] = 'saved eligible trade queries'
    return impacts


# TEST LOGIC: ResearchDiagnosticsChecks；仅用于复现输入或核对行为。
class ResearchDiagnosticsChecks(unittest.TestCase):
    # TEST LOGIC: test_each_family_partitions_targets_and_contributions_add；仅用于复现输入或核对行为。
    def test_each_family_partitions_targets_and_contributions_add(self):
        frame, predictions = fixture()
        original_frame, original_predictions = frame.copy(deep=True), predictions.copy(deep=True)
        with (patch('quote_quality_core.build_quote_features', side_effect=AssertionError('must reuse')),
              patch('quote_quality_core.run_comparison', side_effect=AssertionError('must not fit'))):
            result = review.diagnostics(frame, predictions)
        self.assertEqual(result['n'], 30)
        self.assertEqual(result['reference'], 'Reliability')
        for _, group in result['groups'].groupby('family'):
            self.assertEqual(group.n.sum(), 30)
            self.assertAlmostEqual(group.contribution_bps.sum(), result['overall'].delta_mae)
            self.assertAlmostEqual(group.total_loss_delta.sum(), result['rows'].loss_delta.sum())
        self.assertAlmostEqual(result['overall'].delta_p95,
                               result['rows'].selected_error.quantile(.95) - result['rows'].reference_error.quantile(.95))
        self.assertGreater(result['overall'].selected_gt10, 0)
        self.assertNotAlmostEqual(result['overall'].delta_p95, result['rows'].loss_delta.quantile(.95))
        pd.testing.assert_frame_equal(frame, original_frame)
        pd.testing.assert_frame_equal(predictions, original_predictions)

    # TEST LOGIC: test_fixed_age_gap_boundaries_unknown_and_source_flags；仅用于复现输入或核对行为。
    def test_fixed_age_gap_boundaries_unknown_and_source_flags(self):
        frame, predictions = fixture()
        result = review.diagnostics(frame, predictions)
        rows = result['rows']
        self.assertEqual(rows.loc[0, 'quote_state'], 'No quote')
        self.assertEqual(rows.loc[1, 'quote_state'], 'One-sided')
        self.assertEqual(rows.loc[2, 'quote_state'], 'Two-sided')
        self.assertEqual(rows.loc[1, 'bid_message_age'], '0-5')
        self.assertEqual(rows.loc[2, 'bid_message_age'], '5-30')
        self.assertEqual(rows.loc[3, 'bid_message_age'], '5-30')
        self.assertEqual(rows.loc[4, 'bid_message_age'], '>30')
        self.assertEqual(rows.loc[4, 'bid_change_age'], 'Unknown')
        self.assertEqual(rows.loc[1, 'bid_candidate_gap'], '0-2')
        self.assertEqual(rows.loc[3, 'bid_candidate_gap'], '2-10')
        self.assertEqual(rows.loc[2, 'issuer_trade_count'], 'Unknown')
        self.assertEqual(rows.loc[2, 'SECTOR'], 'Unknown')
        self.assertTrue(result['definitions'].cut_source.eq('predeclared fixed cuts').all())
        changed = frame.copy(); changed.loc[0, 'bcq_bid_mean_candidate_gap'] = 99999
        second = review.diagnostics(changed, predictions)
        pd.testing.assert_frame_equal(result['definitions'], second['definitions'])
        self.assertEqual(second['rows'].loc[1, 'bid_candidate_gap'], '0-2')
        missing = review.diagnostics(frame.drop(columns='bcq_ask_n_dealers'), predictions)
        self.assertTrue(missing['rows'].quote_state.eq('Unknown').all())
        self.assertFalse(missing['definitions'].set_index('family').loc['quote_state', 'available'])

    # TEST LOGIC: test_new_models_use_age_decay_reference_and_reject_wrong_reference；仅用于复现输入或核对行为。
    def test_new_models_use_age_decay_reference_and_reject_wrong_reference(self):
        frame, predictions = fixture()
        for name in ['Age decay + Direction', 'Age decay + Issuer']:
            result = review.diagnostics(frame, predictions, name)
            self.assertEqual(result['reference'], 'Age decay')
            with self.assertRaisesRegex(ValueError, 'must compare directly'):
                review.diagnostics(frame, predictions, name, 'Reliability')
        self.assertEqual(review.diagnostics(frame, predictions, 'Reliability')['reference'], 'Quote levels')

    # TEST LOGIC: test_strict_saved_join_rejects_stale_or_incomplete_cohorts_and_test；仅用于复现输入或核对行为。
    def test_strict_saved_join_rejects_stale_or_incomplete_cohorts_and_test(self):
        frame, predictions = fixture()
        for bad in [predictions.iloc[1:], pd.concat([predictions, predictions.iloc[:1]]),
                    predictions.assign(stage='Test')]:
            with self.assertRaises(ValueError): review.diagnostics(frame, bad)
        bad = predictions.copy(); bad.loc[0, 'time'] += pd.Timedelta(seconds=1)
        with self.assertRaises(ValueError): review.diagnostics(frame, bad)
        wrong = frame.copy(); wrong.loc[0, 'BM_SPREAD'] += .1
        with self.assertRaises(ValueError): review.diagnostics(wrong, predictions)

    # TEST LOGIC: test_lodo_is_recomputed_from_fixed_rows_without_refit；仅用于复现输入或核对行为。
    def test_lodo_is_recomputed_from_fixed_rows_without_refit(self):
        frame, predictions = fixture()
        result = review.diagnostics(frame, predictions)
        rows = result['rows']
        expected = [rows.loc[rows.day.ne(day), 'loss_delta'].mean() for day in rows.day.unique()]
        self.assertAlmostEqual(result['overall'].lodo_min, min(expected))
        self.assertAlmostEqual(result['overall'].lodo_max, max(expected))
        self.assertAlmostEqual(result['overall'].day_equal_delta, rows.groupby('day').loss_delta.mean().mean())

    # TEST LOGIC: test_full_png_and_tables_saved_without_overwriting_or_loading_test；仅用于复现输入或核对行为。
    def test_full_png_and_tables_saved_without_overwriting_or_loading_test(self):
        frame, predictions = fixture()
        with tempfile.TemporaryDirectory() as folder, patch('IPython.display.display'):
            with patch('pandas.read_parquet', side_effect=AssertionError('live results need no reads')):
                first = review.show_validation_diagnostics(frame, predictions, output_folder=folder, parquet=True)
                second = review.show_validation_diagnostics(frame, predictions, output_folder=folder)
            self.assertEqual(len(review.figure(first).axes), 6)
            self.assertEqual(len(review.sector_tail_figure(first).axes), 6)
            self.assertGreater(first['png_path'].stat().st_size, 10000)
            self.assertGreater(first['tail_png_path'].stat().st_size, 10000)
            self.assertNotEqual(first['output_folder'], second['output_folder'])
            for name in ['groups.csv', 'daily.csv', 'overall.csv', 'definitions.csv', 'groups.parquet']:
                self.assertTrue((first['output_folder'] / name).is_file())
            with self.assertRaises(ValueError): review.show_validation_diagnostics(frame=frame, output_folder=folder)

    # TEST LOGIC: test_saved_entry_reads_validation_only_even_when_test_file_is_damaged；仅用于复现输入或核对行为。
    def test_saved_entry_reads_validation_only_even_when_test_file_is_damaged(self):
        from quote_quality_saved import save_step5_checkpoint
        frame, predictions = fixture()
        with tempfile.TemporaryDirectory() as folder, patch('IPython.display.display'):
            source = Path(folder) / 'step5'
            snapshot = save_step5_checkpoint(frame, predictions, predictions.assign(stage='Test'), folder=source)
            (snapshot / 'test_predictions.parquet').write_bytes(b'never read this held-out artifact')
            read_parquet = pd.read_parquet
            with patch('pandas.read_parquet', wraps=read_parquet) as reads:
                result = review.show_validation_diagnostics(folder=source, output_folder=Path(folder) / 'review')
            self.assertEqual(result['n'], 30)
            self.assertEqual([Path(call.args[0]).name for call in reads.call_args_list],
                             ['model_features.parquet', 'validation_predictions.parquet'])

    # TEST LOGIC: test_tail_panels_keep_all_sector_support_and_correct_reference；仅用于复现输入或核对行为。
    def test_tail_panels_keep_all_sector_support_and_correct_reference(self):
        frame, predictions = fixture()
        result = review.diagnostics(frame, predictions, 'Age decay + Direction')
        fig = review.sector_tail_figure(result)
        self.assertIn('minus Age decay', fig.texts[0].get_text())
        self.assertIn('P95 delta=', fig.texts[1].get_text())
        self.assertIn('>10bps share delta=', fig.texts[1].get_text())
        self.assertEqual(len(fig.axes[0].patches), result['groups'].family.eq('SECTOR').sum())
        self.assertTrue(any('n=' in label.get_text() for label in fig.axes[0].get_yticklabels()))
        with self.assertRaises(ValueError): review.sector_tail_figure(dict(result, reference='Reliability'))


# TEST LOGIC: FrozenTradeCaseChecks；仅用于复现输入或核对行为。
class FrozenTradeCaseChecks(unittest.TestCase):
    # TEST LOGIC: test_stable_cases_include_no_quote_and_do_not_access_quote_or_event_paths；仅用于复现输入或核对行为。
    def test_stable_cases_include_no_quote_and_do_not_access_quote_or_event_paths(self):
        frame, _ = fixture(); impacts = impacts_fixture(frame)
        with (patch('quote_quality_core.prepare_quote_events', side_effect=AssertionError('no events')),
              patch('quote_quality_population.population_tables', side_effect=AssertionError('no population rebuild')),
              patch('pandas.read_parquet', side_effect=AssertionError('no raw reads'))):
            result = review.fixed_trade_case_manifest(frame, impacts)
        second = review.fixed_trade_case_manifest(frame.sample(frac=1, random_state=7), impacts.sample(frac=1, random_state=9))
        self.assertEqual(result.case_id.tolist(), second.case_id.tolist())
        self.assertEqual(len(result), 18)
        self.assertEqual(result.selection.value_counts().to_dict(), {'Random': 6, 'Typical': 6, 'High impact': 6})
        self.assertFalse(result.case_id.duplicated().any())
        self.assertTrue(result.loc[result.selection.eq('Random'), 'quote_query_fraction'].eq(0).any())
        self.assertTrue(result.firm.isna().all())
        self.assertTrue(result.scope_flag.str.contains('no population-rate claim').all())
        self.assertTrue(result.impact_source.eq('saved eligible trade queries').all())
        self.assertTrue(result.day.ge(pd.Timestamp('2026-03-19', tz='America/New_York')).all())

    # TEST LOGIC: test_existing_manifest_and_disk_file_remain_frozen；仅用于复现输入或核对行为。
    def test_existing_manifest_and_disk_file_remain_frozen(self):
        frame, _ = fixture(); impacts = impacts_fixture(frame)
        first = review.fixed_trade_case_manifest(frame, impacts)
        unchanged = review.fixed_trade_case_manifest(frame, impacts.assign(max_center_effect_bps=1e9), existing=first)
        pd.testing.assert_frame_equal(first, unchanged)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cases.csv'
            created = review.fixed_trade_case_manifest(frame, impacts, manifest_path=path)
            before = path.read_bytes()
            retained = review.fixed_trade_case_manifest(frame, impacts.assign(max_center_effect_bps=1e9), manifest_path=path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(created.case_id.tolist(), retained.case_id.tolist())

    # TEST LOGIC: test_unassessed_high_impact_is_not_filled_with_raw_gap_cases；仅用于复现输入或核对行为。
    def test_unassessed_high_impact_is_not_filled_with_raw_gap_cases(self):
        frame, _ = fixture(); impacts = impacts_fixture(frame)
        result = review.fixed_trade_case_manifest(frame, impacts.assign(max_center_effect_bps=0, coverage_loss_fraction=0))
        self.assertEqual(len(result), 12)
        self.assertNotIn('High impact', result.selection.tolist())
        with self.assertRaisesRegex(ValueError, 'one row'):
            review.fixed_trade_case_manifest(frame, pd.concat([impacts, impacts.iloc[:1]]))

    # TEST LOGIC: test_missing_dealer_count_is_unassessed_instead_of_false_no_quote；仅用于复现输入或核对行为。
    def test_missing_dealer_count_is_unassessed_instead_of_false_no_quote(self):
        frame, _ = fixture(); impacts = impacts_fixture(frame)
        result = review.fixed_trade_case_manifest(frame.drop(columns='bcq_bid_n_dealers'), impacts)
        bid = result.loc[result.side.eq('bid')]
        self.assertTrue(bid.quote_query_fraction.isna().all())
        self.assertTrue(bid.known_support_queries.eq(0).all())


if __name__ == '__main__':
    unittest.main()
