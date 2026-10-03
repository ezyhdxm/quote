# TEST SETUP LOGIC: 合成fixtures与断言；测试通过不代表真实预测增益。
"""Hot add-on must preserve live state and save completed work before enabling fits."""
import ast
from contextlib import ExitStack
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import quote_quality_research as research


# TEST FIXTURE LOGIC: fixture；仅用于复现输入或核对行为。
def fixture():
    frame = pd.DataFrame(dict(row_id=range(6),
        time=pd.date_range('2026-03-19 10:00', periods=6, freq='min', tz='America/New_York'),
        split='Validation', BM_SPREAD=.6, NUM_OF_ISSUER_TRADES_SINCE_PREV=[0, 1, 0, 2, 0, 1]))
    pred = frame[['row_id', 'time']].copy()
    pred['model'] = 'Age decay'; pred['stage'] = 'Validation'
    pred['pred_spread'] = .61; pred['error_bps'] = 1.; pred['abs_error_bps'] = 1.
    return frame, pred


# TEST LOGIC: ResearchControlsChecks；仅用于复现输入或核对行为。
class ResearchControlsChecks(unittest.TestCase):
    # TEST LOGIC: test_live_context_no_loader_and_retains_saved_lock；仅用于复现输入或核对行为。
    def test_live_context_no_loader_and_retains_saved_lock(self):
        frame, predictions = fixture()
        ns = dict(step5_frame=frame, step5_predictions=predictions,
            step5_result_metadata={'locked_choice': 'Age decay', 'test_available': True},
            step5_event_cache={'events': 'identity only'})
        with patch('quote_quality_saved.load_saved_validation', side_effect=AssertionError('live results')):
            context = research.research_context(ns)
        self.assertIs(context['frame'], frame)
        self.assertIs(context['predictions'], predictions)
        self.assertIs(context['event_cache'], ns['step5_event_cache'])
        self.assertEqual(context['metadata']['locked_choice'], 'Age decay')
        self.assertTrue(context['metadata']['test_available'])
        self.assertNotIn('saved_lock_guard', ns['step5_result_metadata'])

    # TEST LOGIC: test_busy_rejected_before_read_and_disk_does_not_borrow_unmatched_live_cache；仅用于复现输入或核对行为。
    def test_busy_rejected_before_read_and_disk_does_not_borrow_unmatched_live_cache(self):
        with patch('quote_quality_saved.load_saved_validation', side_effect=AssertionError('no read')):
            with self.assertRaisesRegex(ValueError, 'busy'):
                research.research_context({'step5_busy': True})
        frame, predictions = fixture()
        with patch('quote_quality_saved.load_saved_validation', return_value=(frame, predictions, {})):
            context = research.research_context({'step5_event_cache': {'events': 'unmatched'}})
        self.assertIsNone(context['event_cache'])

    # TEST LOGIC: test_saved_only_opens_controls_without_events_or_fitting；仅用于复现输入或核对行为。
    def test_saved_only_opens_controls_without_events_or_fitting(self):
        frame, predictions = fixture()
        with TemporaryDirectory() as temp:
            with patch('quote_quality_saved.load_saved_validation', return_value=(frame, predictions, {})), \
                 patch('quote_quality_incremental.load_movement_sidecar', side_effect=FileNotFoundError), \
                 patch('quote_quality_incremental.run_incremental_validation', side_effect=AssertionError('no automatic fit')):
                session = research.ResearchSession(output_folder=temp)
            self.assertTrue(session.buttons['features'].disabled)
            self.assertTrue(session.buttons['Direction'].disabled)
            self.assertFalse(session.buttons['losses'].disabled)
            session.dashboard.close()

    # TEST LOGIC: test_save_failure_retains_built_sidecar_and_retry_does_not_rebuild；仅用于复现输入或核对行为。
    def test_save_failure_retains_built_sidecar_and_retry_does_not_rebuild(self):
        frame, predictions = fixture()
        ns = dict(step5_frame=frame, step5_predictions=predictions,
            step5_event_cache={'cache_key': 'same'},
            step5_result_metadata={'event_cache_key': 'same', 'quote_file_dates': ['2026-03-01', '2026-04-01']})
        sidecar = frame[['row_id', 'time']].copy()
        with TemporaryDirectory() as temp, \
             patch('quote_quality_incremental.load_movement_sidecar', side_effect=FileNotFoundError), \
             patch('quote_quality_movement.build_movement_features', return_value=sidecar) as build, \
             patch('quote_quality_incremental.save_movement_sidecar', side_effect=[OSError('disk unavailable'), Path(temp)]), \
             patch('quote_quality_research.movement_coverage_figure', return_value=None), \
             patch.object(research.ResearchSession, '_render', return_value=Path(temp)/'support.png'):
            session = research.ResearchSession(ns, output_folder=temp)
            session.run('features')
            self.assertIs(session.sidecar, sidecar)
            self.assertFalse(session.sidecar_saved)
            self.assertFalse(session.buttons['features'].disabled)
            self.assertTrue(session.buttons['Direction'].disabled)
            session.run('features')
            self.assertTrue(session.sidecar_saved)
            self.assertEqual(build.call_count, 1)
            self.assertIs(ns['step5_frame'], frame)
            self.assertFalse(ns['step5_research_busy'])
            session.dashboard.close()

    # TEST LOGIC: test_dynamic_test_lock_blocks_fit_after_controls_open；仅用于复现输入或核对行为。
    def test_dynamic_test_lock_blocks_fit_after_controls_open(self):
        frame, predictions = fixture()
        ns = dict(step5_frame=frame, step5_predictions=predictions, step5_result_metadata={})
        with TemporaryDirectory() as temp, \
             patch('quote_quality_incremental.load_movement_sidecar', side_effect=FileNotFoundError), \
             patch('quote_quality_incremental.run_incremental_validation', side_effect=AssertionError('must stay blocked')) as run:
            session = research.ResearchSession(ns, output_folder=temp)
            session.sidecar = pd.DataFrame(); session.sidecar_saved = True
            ns['step5_locked'] = 'Age decay'
            session.run('Direction')
            self.assertEqual(run.call_count, 0)
            self.assertTrue(session.buttons['Direction'].disabled)
            self.assertIs(ns['step5_predictions'], predictions)
            session.dashboard.close()

    # TEST LOGIC: test_coverage_does_not_count_zero_support_as_available_and_png_complete；仅用于复现输入或核对行为。
    def test_coverage_does_not_count_zero_support_as_available_and_png_complete(self):
        from quote_quality_movement import DIRECTION_FEATURES, ISSUER_FEATURES
        frame, _ = fixture(); frame['split'] = ['Train'] * 3 + ['Validation'] * 3
        sidecar = frame[['row_id', 'time']].copy()
        sidecar['cusip'] = 'X'
        for column in DIRECTION_FEATURES + ISSUER_FEATURES: sidecar[column] = np.nan
        fig = research.movement_coverage_figure(frame, sidecar)
        self.assertEqual(len(fig.axes), 4)
        self.assertTrue(all(bar.get_height() == 0 for bar in fig.axes[0].patches))
        with TemporaryDirectory() as temp:
            path = Path(temp)/'coverage.png'; fig.savefig(path)
            self.assertGreater(path.stat().st_size, 1000)

    # TEST LOGIC: test_notebook_is_three_unexecuted_cells_and_never_calls_loader；仅用于复现输入或核对行为。
    def test_notebook_is_three_unexecuted_cells_and_never_calls_loader(self):
        path = Path(__file__).resolve().parents[1]/'quote_quality_step5_research.ipynb'
        nb = json.loads(path.read_text())
        cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(cells), 3)
        self.assertTrue(all(c['execution_count'] is None and not c['outputs'] for c in cells))
        code = ''.join(''.join(c['source']) for c in cells)
        self.assertNotIn('read_parquet', code)
        self.assertNotIn('run_comparison', code)
        self.assertNotIn('run_step5', code)


# TEST LOGIC: 主Step5末cell可独立运行；验证真实入口不会启动原loader、聚合或模型。
class IntegratedResearchEntryChecks(unittest.TestCase):
    # TEST FIXTURE LOGIC: 每个入口测试使用独立目录，保存样本不进入真实输出。
    def setUp(self):
        previous = Path.cwd()
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(os.chdir, previous)
        os.chdir(temporary.name)
        self.root = Path(__file__).resolve().parents[1]
        self.marker = '# %% 4. Continue completed validation research'
        self.source = (self.root / 'quote_quality_step5.py').read_text().split(self.marker, 1)[1]

    # TEST FIXTURE LOGIC: 合成frame、预测、cache均保留原对象，用identity断言发现状态重建。
    def live_namespace(self):
        frame, predictions = fixture()
        return dict(step5_frame=frame, step5_predictions=predictions,
            step5_test_predictions=None, step5_locked=None,
            step5_result_metadata={'locked_choice': None, 'test_available': False},
            step5_event_cache={'cache_key': 'retained-live-events'})

    # TEST FIXTURE LOGIC: 除指定的saved验证文件外禁止读Parquet；禁止所有重建与fit入口。
    def execute_entry(self, namespace, parquet_reader=None):
        with ExitStack() as stack:
            display = stack.enter_context(patch('IPython.display.display'))
            if parquet_reader is None:
                parquet_reader = AssertionError('entry must not read raw Parquet')
            stack.enter_context(patch('pandas.read_parquet', side_effect=parquet_reader))
            stack.enter_context(patch('quote_quality_incremental.load_movement_sidecar', side_effect=FileNotFoundError))
            for target in ['quote_quality_core.prepare_quote_events', 'quote_quality_core.build_quote_features',
                           'quote_quality_core.run_comparison', 'quote_quality_movement.build_movement_features',
                           'quote_quality_incremental.run_incremental_validation']:
                stack.enter_context(patch(target, side_effect=AssertionError('opening controls must not compute or fit')))
            exec(compile(self.source, 'step5_research_entry', 'exec'), namespace)
        session = namespace.get('step5_research')
        if session is not None:
            self.addCleanup(session.dashboard.close)
        return display

    # TEST LOGIC: 已完成live validation直接开面板，所有原结果和事件cache保留identity。
    def test_entry_uses_live_completed_results_without_loader_build_or_fit(self):
        ns = self.live_namespace()
        retained = dict(ns)
        with patch('quote_quality_saved.load_saved_validation', side_effect=AssertionError('live results are available')):
            display = self.execute_entry(ns)
        session = ns['step5_research']
        self.assertIs(session.namespace, ns)
        self.assertIs(session.frame, retained['step5_frame'])
        self.assertIs(session.predictions, retained['step5_predictions'])
        self.assertIs(session.event_cache, retained['step5_event_cache'])
        for name, value in retained.items():
            self.assertIs(ns[name], value)
        display.assert_called_once_with(session.dashboard)

    # TEST LOGIC: fresh kernel只读校验通过的保存frame/validation，locked状态保留且不借用残留cache。
    def test_entry_restores_saved_validation_without_loading_raw_or_test(self):
        from quote_quality_saved import save_step5_checkpoint
        frame, predictions = fixture()
        snapshot = save_step5_checkpoint(frame, predictions, metadata={'locked_choice': 'Age decay'})
        (snapshot / 'test_predictions.parquet').write_bytes(b'this file must never be read')
        real_read = pd.read_parquet
        reads = []
        # TEST FIXTURE LOGIC: 仅允许两个保存文件，记录真实读取以证明没有raw或test路径。
        def read_validation_only(path, *args, **kwargs):
            self.assertIn(Path(path).name, ['model_features.parquet', 'validation_predictions.parquet'])
            reads.append(Path(path).name)
            return real_read(path, *args, **kwargs)
        ns = {'step5_event_cache': {'cache_key': 'unmatched'}}
        self.execute_entry(ns, read_validation_only)
        session = ns['step5_research']
        pd.testing.assert_frame_equal(session.frame, frame)
        pd.testing.assert_frame_equal(session.predictions, predictions)
        self.assertEqual(reads, ['model_features.parquet', 'validation_predictions.parquet'])
        self.assertIsNone(session.event_cache)
        self.assertEqual(session.metadata['locked_choice'], 'Age decay')
        self.assertTrue(session.buttons['features'].disabled)
        self.assertTrue(session.buttons['Direction'].disabled)

    # TEST LOGIC: 尚无保存结果时只显示提示，未完成的live frame和cache不清空。
    def test_missing_validation_shows_message_and_preserves_partial_work(self):
        ns = self.live_namespace()
        ns['step5_predictions'] = None
        retained = dict(ns)
        display = self.execute_entry(ns)
        self.assertNotIn('step5_research', ns)
        self.assertTrue(display.called)
        self.assertIn('No saved Step5', display.call_args.args[0].data)
        for name, value in retained.items():
            self.assertIs(ns[name], value)

    # TEST LOGIC: 实际保存文件checksum损坏时拒绝恢复，错误不会触发重训。
    def test_corrupt_checkpoint_shows_error_without_fit_or_parquet_read(self):
        from quote_quality_saved import save_step5_checkpoint
        frame, predictions = fixture()
        snapshot = save_step5_checkpoint(frame, predictions)
        path = snapshot / 'validation_predictions.parquet'
        path.write_bytes(path.read_bytes() + b'changed after save')
        ns = {}
        display = self.execute_entry(ns)
        self.assertNotIn('step5_research', ns)
        self.assertIn('checksum', display.call_args.args[0].data.lower())

    # TEST LOGIC: 磁盘不可读同样是可见错误，不能把故障解释成需要重训。
    def test_unreadable_checkpoint_shows_io_error_and_preserves_live_objects(self):
        ns = self.live_namespace()
        ns['step5_predictions'] = None
        frame = ns['step5_frame']
        with patch('quote_quality_saved.load_saved_validation', side_effect=OSError('checkpoint disk unavailable')):
            display = self.execute_entry(ns)
        self.assertIs(ns['step5_frame'], frame)
        self.assertNotIn('step5_research', ns)
        self.assertIn('checkpoint disk unavailable', display.call_args.args[0].data)

    # TEST LOGIC: 三种busy来源均在读取、显示或重建前拒绝，原会话及其sidecar保持不变。
    def test_busy_entry_is_rejected_before_any_load_or_session_reconstruction(self):
        for state in [{'step5_busy': True}, {'step5_research_busy': True},
                      {'step5_research': SimpleNamespace(busy=True, sidecar=object())}]:
            with self.subTest(state=list(state)):
                ns = self.live_namespace()
                ns.update(state)
                retained = dict(ns)
                with patch('quote_quality_research.show_research', side_effect=AssertionError('must not reconstruct')), \
                     patch('IPython.display.display', side_effect=AssertionError('must not replace UI')):
                    with self.assertRaisesRegex(RuntimeError, 'Step5 is running'):
                        exec(compile(self.source, 'step5_research_entry', 'exec'), ns)
                for name, value in retained.items():
                    self.assertIs(ns[name], value)

    # TEST LOGIC: 重跑末cell只重新display，保留未保存sidecar、追加预测和已显示的状态。
    def test_rerun_reuses_existing_session_and_retains_unsaved_incremental_work(self):
        ns = self.live_namespace()
        self.execute_entry(ns)
        session = ns['step5_research']
        sidecar = pd.DataFrame({'row_id': [1], 'movement': [2.]})
        appended_predictions = session.predictions.copy()
        session.sidecar = sidecar
        session.predictions = appended_predictions
        session.status.value = 'Completed Direction; sidecar is still in memory'
        with patch('quote_quality_research.show_research', side_effect=AssertionError('must reuse existing session')):
            display = self.execute_entry(ns)
        self.assertIs(ns['step5_research'], session)
        self.assertIs(session.sidecar, sidecar)
        self.assertIs(session.predictions, appended_predictions)
        self.assertEqual(session.status.value, 'Completed Direction; sidecar is still in memory')
        display.assert_called_once_with(session.dashboard)

    # TEST LOGIC: 已提交notebook自带末cell，和配对Python执行逻辑一致，无保存输出。
    def test_main_notebook_contains_the_same_independent_entry_cell(self):
        notebook = json.loads((self.root / 'quote_quality_step5.ipynb').read_text())
        cells = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
        self.assertEqual(len(cells), 4)
        self.assertIsNone(cells[-1]['execution_count'])
        self.assertFalse(cells[-1]['outputs'])
        self.assertEqual(ast.dump(ast.parse(''.join(cells[-1]['source']))), ast.dump(ast.parse(self.source)))


if __name__ == '__main__': unittest.main()
