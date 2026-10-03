"""Causal pairing, existing-target ablations and complete notebook workflows (synthetic data)."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import quote_quality_core as qc

T=pd.Timestamp('2026-03-02 10:00',tz='America/New_York')


def raw(records):
    # minute, dealer, side, spread, raw quantity
    return pd.DataFrame([dict(cusip='X',firm=firm,side=side,spread=s,quantity=q,
        quote_timestamp_ET=T+pd.Timedelta(minutes=minute)) for minute,firm,side,s,q in records])


def fixture():
    dates=pd.bdate_range('2026-02-16',periods=35,tz='America/New_York')
    quotes=[];trades=[]
    for dayno,day in enumerate(dates):
        for bond in ['X','Y']:
            for minute in range(0,61,5):
                level=60+dayno*.1+minute*.02+(bond=='Y')*10
                row={k:1. for k in qc.BASE_FEATURES}
                row.update(CUSIP=bond,ISSUER='SYNTHETIC '+bond,SECTOR='Energy' if bond=='X' else 'Utility',EFFECTIVE_DATETIME_TS=day+pd.Timedelta(hours=10,minutes=minute),
                    PREV_BM_SPREAD=level/100,QUANTITY=50_000 if minute%10 else 2_000_000,
                    PREV_TRADE_TYPE='B',TRADE_TYPE='S' if minute%10 else 'B',D_BM_SPREAD=(minute-30)/1000,
                    BM_SPREAD=level/100+(minute-30)/1000)
                trades.append(row)
            if dayno<10 or bond=='Y':continue  # no-quote bonds/history must stay in evaluation
            for minute in [0,10,30,55]:
                for firm,offset in [('A',0),('B',1),('C',-1),('D',2)]:
                    for side,sg in [('bid',3),('ask',-3)]:
                        level=60+dayno*.1+minute*.02+sg+offset
                        candidates=[(level,2)] if firm!='A' else [(level,2),(level-40,0)]
                        for value,size in candidates:
                            quotes.append(dict(cusip=bond,firm=firm,side=side,spread=value,quantity=size,
                                quote_timestamp_UTC=(day+pd.Timedelta(hours=10,minutes=minute)).tz_convert('UTC')))
    return pd.DataFrame(quotes),pd.DataFrame(trades)


def canonical(quotes):
    q=quotes.copy();q['quote_timestamp_ET']=q.quote_timestamp_UTC.dt.tz_convert('America/New_York')
    q['ISSUER']='SYNTHETIC '+q.cusip
    return q


def model_frame():
    q,t=fixture();q=canonical(q)
    t=t.assign(time=t.EFFECTIVE_DATETIME_TS,cusip=t.CUSIP,row_id=np.arange(len(t)))
    t['split'],t['refit_train']=qc.chronological_split(t,q.quote_timestamp_ET.max())
    f=qc.build_quote_features(q,t[['row_id','cusip','time']])
    return t.merge(f.drop(columns=['cusip','time']),on='row_id',validate='one_to_one')


class PairChecks(unittest.TestCase):
    def pairs(self,records,minutes=(0,),**kwargs):
        return qc.pair_snapshots(qc.event_history(raw(records))['events'],[T+pd.Timedelta(minutes=m) for m in minutes],**kwargs)

    def test_crossing_direction_locked_and_set_classes(self):
        p=self.pairs([(0,'A','bid',100,0),(0,'A','bid',120,0),(0,'A','ask',110,0),
                      (0,'B','bid',100,2),(0,'B','ask',100,2),
                      (0,'C','bid',-10,2),(0,'C','ask',-5,2)]).set_index('firm')
        self.assertEqual(p['cross'].to_dict(),{'A':'Some','B':'None','C':'All'})
        self.assertEqual((p.loc['A','gap_low'],p.loc['A','gap_high']),(-10,10))
        self.assertEqual(p.loc['B','gap_center'],0)
        self.assertEqual(p.loc['C','gap_center'],-5)

    def test_raw_size_matches_never_use_zero_and_keep_positive_multi(self):
        p=self.pairs([(0,'A','bid',90,0),(0,'A','ask',80,0),
                      (0,'B','bid',90,2),(0,'B','bid',110,2),(0,'B','ask',100,2),
                      (0,'B','ask',200,5),(0,'C','bid',90,2),(0,'C','ask',80,5)]).set_index('firm')
        self.assertFalse(p.loc['A','size_time_pair']);self.assertFalse(p.loc['C','size_time_pair'])
        self.assertTrue(p.loc['B','size_time_pair']);self.assertEqual(p.loc['B','cross_matched'],'Some')
        self.assertEqual(p.loc['B','matched_gap_high'],10)
        self.assertEqual(p.loc['B','matched_mid'],100)  # not pooled with the unmatched size-5 ask

    def test_asynchrony_age_exact_time_and_no_overnight(self):
        records=[(0,'A','bid',105,2),(3,'A','ask',100,2)]
        p=self.pairs(records,minutes=(-1,0,3,31,34,1440))
        self.assertEqual(len(p),4)  # no before-first or overnight state
        at3=p.loc[p.time.eq(T+pd.Timedelta(minutes=3))].iloc[0]
        self.assertTrue(at3.fresh_pair);self.assertFalse(at3.same_time);self.assertFalse(at3.size_time_pair)
        self.assertFalse(p.loc[p.time.eq(T+pd.Timedelta(minutes=31)),'fresh_pair'].iloc[0])
        exact=self.pairs(records,minutes=(3,),sync_min=3).iloc[0]
        self.assertTrue(exact.size_time_pair)
        strict=self.pairs(records,minutes=(3,),allow_exact=False).iloc[0]
        self.assertFalse(strict.both)

    def test_latest_incomplete_blocks_old_valid_pair(self):
        p=self.pairs([(0,'A','bid',105,2),(0,'A','ask',100,2),(1,'A','bid',np.nan,2)],minutes=(2,))
        self.assertTrue(p.iloc[0].both);self.assertFalse(p.iloc[0].complete)
        self.assertEqual(p.iloc[0]['cross'],'Unassessed');self.assertTrue(pd.isna(p.iloc[0].gap_center))

    def test_future_append_duplicates_and_query_batches_do_not_rewrite_features(self):
        records=[(0,'A','bid',-5,0),(0,'A','bid',5,0),(0,'A','ask',0,0),(2,'A','bid',7,2)]
        q=raw(records);queries=pd.DataFrame({'row_id':[50,21,22],'cusip':['X','X','Y'],'time':[T,T+pd.Timedelta(minutes=3),T]})
        original=qc.build_quote_features(q,queries)
        altered=qc.build_quote_features(pd.concat([q,q.iloc[:2],raw([(100,'A','bid',999,2)])]),queries)
        pd.testing.assert_frame_equal(original,altered)
        single=qc.build_quote_features(q,queries.iloc[[1]])
        pd.testing.assert_frame_equal(original.iloc[[1]].reset_index(drop=True),single)
        self.assertEqual(original.row_id.tolist(),[50,21,22]);self.assertEqual(original.bcq_has_quote.tolist(),[1.,1.,0.])
        self.assertEqual(original.bcq_bid_n_dealers.tolist(),[1,1,0])
        with self.assertRaises(ValueError):qc.build_quote_features(q,pd.concat([queries,queries.iloc[[0]]]))

    def test_no_candidate_multiplicity_weight_and_no_dealer_cross_pairing(self):
        p=self.pairs([(0,'A','bid',100,2),(0,'A','bid',120,2),(0,'A','ask',100,2),
                      (0,'B','bid',110,2),(0,'B','ask',100,2),(0,'C','bid',999,2)])
        f=qc.pair_features(p,[T]).iloc[0]
        self.assertEqual(f.n_pair,2);self.assertEqual(f.pair_gap,10);self.assertEqual(f.n_size_time_pair,2)

    def test_fast_side_matches_step3_reference_and_peer_support(self):
        # Reuse the previous notebook's independent reference, avoiding a second production implementation.
        source=ast.parse((ROOT/'quote_quality_step3.py').read_text())
        node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='asof_features')
        state=dict(np=np,pd=pd,DEFAULT_AGE_MIN=30,LOOKBACK_MIN=30,KEYS=qc.KEYS,
                   MIN_PEERS=3,CLIP_FLOOR_BPS=10.,MAD_MULTIPLIER=4.)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'reference','exec'),state)
        records=[(0,'A','bid',-100,0),(0,'A','bid',200,0),(0,'B','bid',50,2),
                 (0,'C','bid',51,2),(0,'D','bid',49,2),(10,'B','bid',52,2),
                 (20,'D','bid',np.nan,2),(70,'A','bid',201,0)]
        events=qc.event_history(raw(records))['events'];times=pd.DatetimeIndex([T+pd.Timedelta(minutes=m) for m in [-1,0,10,21,60,75,1440]])
        _,ref=state['asof_features'](events,times)
        fast=qc.side_features_fast(events,times)
        pd.testing.assert_frame_equal(ref[fast.columns].rename_axis('time'),fast,check_dtype=False,atol=1e-8)
        self.assertEqual(fast.loc[T,'n_clipped_dealers'],1)
        self.assertEqual(fast.loc[T,'n_changed_centers'],0)  # clipped extremes, same median
        self.assertEqual(fast.iloc[-1].n_dealers,0)
        strict=qc.side_features_fast(events,[T],allow_exact=False)
        self.assertEqual(strict.iloc[0].n_dealers,0)


class TrainingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.frame=model_frame()

    def test_baseline_anchor_units_feature_sets_and_coverage(self):
        f=self.frame;x,specs=qc.model_versions(f)
        self.assertEqual(len(specs['Base'][0]),14)
        self.assertEqual(specs['Base'][0][0],'D_CPP_BM_SPREAD')
        self.assertEqual(specs['Base'][0][-1],'PREV_BM_SPREAD_STD_GROUP_BY_TYPE')
        self.assertEqual(len(specs),8)
        covered=x.bcq_has_quote.gt(0)
        np.testing.assert_allclose(x.loc[covered,'bcq_bid_anchor_gap'],x.loc[covered,'bcq_bid_center_equal']-100*x.loc[covered,'PREV_BM_SPREAD'])
        self.assertTrue(f.loc[f.CUSIP.eq('Y'),'bcq_has_quote'].eq(0).all())
        self.assertEqual(len(f),35*2*13)

    def test_fixed_split_embargo_and_refit_excludes_test(self):
        f=self.frame
        for label,n in [('Train',23),('Validation',5),('Test',5)]:
            self.assertEqual(f.loc[f.split.eq(label),'time'].dt.normalize().nunique(),n)
        self.assertFalse((f.refit_train&f.split.eq('Test')).any())
        self.assertLess(f.loc[f.refit_train,'time'].max(),f.loc[f.split.eq('Test'),'time'].min())
        with self.assertRaises(ValueError):qc.chronological_split(f,pd.NaT)

    def test_actual_training_same_rows_native_target_and_locked_baselines(self):
        params=dict(objective='mae',boosting_type='dart',n_estimators=4,num_leaves=4,
                    min_child_samples=2,n_jobs=1,verbosity=-1,random_state=2026)
        f=self.frame.copy();f.loc[f.split.eq('Test'),'TRADE_TYPE']='UNSEEN'
        pred,models=qc.run_comparison(f,params)
        self.assertEqual(pred.model.nunique(),8)
        self.assertEqual(len({tuple(g.row_id) for _,g in pred.groupby('model')}),1)
        self.assertEqual(len(models['Base'].feature_name_),14)
        row=pred.iloc[0];target=f.set_index('row_id').loc[row.row_id,'BM_SPREAD']
        self.assertAlmostEqual(row.error_bps,(row.pred_spread-target)*100)
        test,_=qc.run_comparison(f,params,'Test','Candidate clip')
        self.assertEqual(set(test.model),{'Base','Reliability','Candidate clip'})
        self.assertEqual(len({tuple(g.row_id) for _,g in test.groupby('model')}),1)
        f['bcq_has_quote']=0
        with self.assertRaisesRegex(ValueError,'unassessed'):qc.run_comparison(f,params)


class NotebookChecks(unittest.TestCase):
    VERSIONS=('Base','Quote levels','Reliability','Age decay')

    def setUp(self):
        # Notebook snapshots and narrow caches must never leak into the checkout.
        previous=Path.cwd()
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.addCleanup(os.chdir,previous)
        os.chdir(temporary.name)
        self.folder=Path('outputs/quote_quality_step5')

    def load(self,step):
        q,t=fixture();state={'__name__':'notebook_check'}
        with contextlib.redirect_stdout(io.StringIO()),patch('pandas.read_parquet',side_effect=lambda path,*a,**k:(t if Path(path).name=='data_ig.parquet' else q).copy()),patch.object(Path,'exists',return_value=True),patch('IPython.display.display') as display:
            exec(compile((ROOT/f'quote_quality_step{step}.py').read_text(),f'step{step}','exec'),state)
        self.assertEqual(display.call_count,1)
        return state

    def configure(self,s):
        s['LGB_PARAMS'].update(n_estimators=3,num_leaves=4,min_child_samples=2,n_jobs=1)

    def controls_cell(self,step=5):
        marker='# %% 3. Research controls' if step==5 else '# %% 3. Case controls'
        return (ROOT/f'quote_quality_step{step}.py').read_text().split(marker,1)[1]

    def checkpoint(self):
        pointer=json.loads((self.folder/'latest.json').read_text())
        snapshot=self.folder/pointer['snapshot']
        return snapshot,json.loads((snapshot/'experiment.json').read_text())

    def rerun_controls(self,s):
        with patch('IPython.display.display'),contextlib.redirect_stdout(io.StringIO()):
            exec(compile(self.controls_cell(),'step5_controls','exec'),s)

    def test_step4_controls_rerun_and_combined_png(self):
        s=self.load(4);identity=s['step4_image'].model_id
        self.assertEqual(len(s['step4_figure'].axes),6)
        s['step4_age'].value=10;s['step4_sync'].value=0
        self.assertEqual(s['step4_image'].model_id,identity)
        self.assertEqual(bytes(s['step4_image'].value),b'')
        self.assertTrue(s['step4_save'].disabled)
        s['apply_step4']()
        self.assertTrue(bytes(s['step4_image'].value).startswith(b'\x89PNG'))
        self.assertEqual((s['step4_result']['age'],s['step4_result']['sync']),(10,0))
        old=s['step4_controls'][0];old_refresh=s['refresh_step4']
        with patch('IPython.display.display'),contextlib.redirect_stdout(io.StringIO()):
            exec(self.controls_cell(4),s)
        self.assertNotIn(old_refresh,old._trait_notifiers.get('value',{}).get('change',[]))

    def test_step5_build_validate_lock_test_are_automatic_checkpoints(self):
        import quote_quality_saved as saved
        s=self.load(5);self.configure(s)
        self.assertEqual(tuple(s['VALIDATION_VERSIONS']),self.VERSIONS)
        self.assertEqual(len(s['step5_figure'].axes),4)
        s['run_step5']('build');frame=s['step5_frame']
        first,metadata=self.checkpoint()
        self.assertEqual(first,s['step5_last_checkpoint'])
        self.assertFalse(metadata['validation_available'])
        self.assertFalse(metadata['test_available'])
        self.assertIsNone(metadata['locked_choice'])
        self.assertEqual(metadata['validation_versions'],list(self.VERSIONS))
        restored=pd.read_parquet(first/'model_features.parquet')
        pd.testing.assert_frame_equal(restored,frame)
        self.assertEqual(set(restored.SECTOR),{'Energy','Utility'})
        original_files={p.name:p.read_bytes() for p in first.iterdir()}
        real_comparison=s['run_comparison'];models_seen=[];test_started=[]
        def compare(frame,params,stage='Validation',selected=None,**kwargs):
            if stage=='Test':
                # The held-out fit cannot begin until its choice is recoverable.
                before,manifest=self.checkpoint()
                self.assertEqual(manifest['locked_choice'],'Age decay')
                self.assertTrue(manifest['validation_available'])
                self.assertFalse(manifest['test_available'])
                self.assertFalse((before/'test_predictions.parquet').exists())
                self.assertEqual(selected,'Age decay')
                test_started.append(before)
            predictions,models=real_comparison(frame,params,stage,selected,**kwargs)
            models_seen.append(tuple(models))
            return predictions,models
        calls=Mock(side_effect=compare)
        with patch.dict(s,run_comparison=calls):
            s['run_step5']('validate')
            validation=s['step5_predictions']
            self.assertEqual(models_seen,[self.VERSIONS])
            self.assertEqual(validation.model.drop_duplicates().tolist(),list(self.VERSIONS))
            self.assertEqual(calls.call_args.kwargs['versions'],self.VERSIONS)
            second,metadata=self.checkpoint()
            self.assertNotEqual(first,second)
            self.assertTrue(metadata['validation_available'])
            self.assertFalse(metadata['test_available'])
            pd.testing.assert_frame_equal(pd.read_parquet(second/'validation_predictions.parquet'),validation)
            self.assertEqual((s['step5_progress'].value,s['step5_progress'].max),(4,4))
            self.assertEqual(s['step5_progress'].bar_style,'success')
            self.assertTrue(s['step5_clock_stop'].is_set())
            s['step5_selected'].value='Age decay'
            s['run_step5']('test');test=s['step5_test_predictions']
            self.assertEqual(s['step5_locked'],'Age decay')
            self.assertEqual(models_seen[-1],('Base','Reliability','Age decay'))
            self.assertEqual(len(test_started),1)
            self.assertTrue(s['step5_selected'].disabled)
            self.assertTrue(s['step5_validate'].disabled)
            final,metadata=self.checkpoint()
            self.assertNotEqual(final,test_started[0])
            self.assertTrue(metadata['test_available'])
            self.assertEqual(metadata['locked_choice'],'Age decay')
            pd.testing.assert_frame_equal(pd.read_parquet(final/'test_predictions.parquet'),test)
            pointer=(self.folder/'latest.json').read_bytes()
            s['run_step5']('test')
            self.assertIs(s['step5_frame'],frame)
            self.assertIs(s['step5_predictions'],validation)
            self.assertIs(s['step5_test_predictions'],test)
            self.assertEqual(calls.call_count,2)
            self.assertEqual((self.folder/'latest.json').read_bytes(),pointer)
        self.assertEqual({p.name:p.read_bytes() for p in first.iterdir()},original_files)
        # Validation recovery reads its own files, even after the locked test exists.
        real_read=pd.read_parquet
        with patch('lightgbm.LGBMRegressor.fit',side_effect=AssertionError('viewer must not fit')), \
                patch('pandas.read_parquet',wraps=real_read) as reads:
            restored,validation_saved,manifest=saved.load_saved_validation(self.folder)
        pd.testing.assert_frame_equal(restored,frame)
        pd.testing.assert_frame_equal(validation_saved,validation)
        self.assertEqual([Path(call.args[0]).name for call in reads.call_args_list],
                         ['model_features.parquet','validation_predictions.parquet'])
        self.assertEqual(manifest['locked_choice'],'Age decay')
        s['save_step5']();self.assertTrue((self.folder/'dashboard.png').is_file())

    def test_step5_completed_validation_is_reused_without_fit_or_build(self):
        s=self.load(5);self.configure(s)
        s['run_step5']('validate')
        frame,predictions=s['step5_frame'],s['step5_predictions']
        pointer=(self.folder/'latest.json').read_bytes()
        with patch.dict(s,run_comparison=Mock(side_effect=AssertionError('do not refit')),
                         build_quote_features=Mock(side_effect=AssertionError('do not rebuild'))):
            s['run_step5']('validate')
            self.assertIs(s['step5_frame'],frame)
            self.assertIs(s['step5_predictions'],predictions)
            self.assertEqual((self.folder/'latest.json').read_bytes(),pointer)
            self.assertIn('Completed validation remains',s['step5_status'].value)
            self.assertFalse(s['step5_save'].disabled)

    def test_step5_controls_rerun_preserves_complete_work_and_locked_choice(self):
        s=self.load(5);self.configure(s)
        s['run_step5']('validate');s['step5_selected'].value='Age decay';s['run_step5']('test')
        frame,predictions,test=s['step5_frame'],s['step5_predictions'],s['step5_test_predictions']
        cache,metadata,last=s['step5_event_cache'],s['step5_result_metadata'],s['step5_last_checkpoint']
        old_button=s['step5_validate'];pointer=(self.folder/'latest.json').read_bytes()
        with patch('lightgbm.LGBMRegressor.fit',side_effect=AssertionError('UI rerun must not fit')), \
                patch.dict(s,build_quote_features=Mock(side_effect=AssertionError('UI rerun must not rebuild'))):
            self.rerun_controls(s)
        self.assertIs(s['step5_frame'],frame)
        self.assertIs(s['step5_predictions'],predictions)
        self.assertIs(s['step5_test_predictions'],test)
        self.assertIs(s['step5_event_cache'],cache)
        self.assertIs(s['step5_result_metadata'],metadata)
        self.assertEqual(s['step5_last_checkpoint'],last)
        self.assertEqual(s['step5_locked'],'Age decay')
        self.assertEqual(s['step5_selected'].value,'Age decay')
        self.assertTrue(s['step5_selected'].disabled)
        self.assertTrue(s['step5_validate'].disabled)
        self.assertEqual((self.folder/'latest.json').read_bytes(),pointer)
        self.assertFalse(old_button._click_handlers.callbacks)

    def test_step5_controls_rerun_keeps_frame_only_and_validation_only(self):
        for phase in ['build','validate']:
            with self.subTest(phase=phase):
                s=self.load(5);self.configure(s);s['run_step5'](phase)
                frame,predictions=s['step5_frame'],s['step5_predictions']
                pointer=(self.folder/'latest.json').read_bytes()
                with patch('lightgbm.LGBMRegressor.fit',side_effect=AssertionError('UI rebuild must not fit')), \
                        patch.dict(s,build_quote_features=Mock(side_effect=AssertionError('UI rebuild must not build'))):
                    self.rerun_controls(s)
                self.assertIs(s['step5_frame'],frame)
                self.assertIs(s['step5_predictions'],predictions)
                self.assertIsNone(s['step5_test_predictions'])
                self.assertIsNone(s['step5_locked'])
                self.assertFalse(s['step5_export'].disabled)
                self.assertEqual(s['step5_validate'].disabled,phase=='validate')
                self.assertEqual((self.folder/'latest.json').read_bytes(),pointer)

    def test_step5_new_kernel_preserves_saved_lock_without_reading_test_or_fit(self):
        import quote_quality_saved as saved
        s=self.load(5);self.configure(s)
        s['run_step5']('validate');s['step5_selected'].value='Age decay';s['run_step5']('test')
        pointer=(self.folder/'latest.json').read_bytes()
        final,metadata=self.checkpoint()
        # The recovered lock comes from metadata, never from held-out predictions.
        (final/'test_predictions.parquet').write_bytes(b'locked test file must not be read')
        real_read=pd.read_parquet;real_hash=saved._file_sha256
        with patch('lightgbm.LGBMRegressor.fit',side_effect=AssertionError('restart must not fit')):
            restored=self.load(5)
        self.assertIsNone(restored['step5_frame'])
        self.assertIsNone(restored['step5_predictions'])
        self.assertIsNone(restored['step5_test_predictions'])
        self.assertEqual(restored['step5_locked'],'Age decay')
        self.assertTrue(restored['step5_saved_lock_guard'])
        self.assertTrue(restored['step5_selected'].disabled)
        for name in ['step5_build','step5_validate','step5_test']:self.assertTrue(restored[name].disabled)
        with patch.dict(restored,run_comparison=Mock(side_effect=AssertionError('persisted lock forbids fit')),
                        build_quote_features=Mock(side_effect=AssertionError('persisted lock forbids new full build'))):
            for action in ['build','validate','test']:restored['run_step5'](action)
        self.assertEqual((self.folder/'latest.json').read_bytes(),pointer)
        with patch('pandas.read_parquet',wraps=real_read) as reads, \
                patch.object(saved,'_file_sha256',wraps=real_hash) as hashes:
            frame,predictions,manifest=saved.load_saved_validation(self.folder)
        self.assertEqual(manifest['locked_choice'],'Age decay')
        self.assertEqual(set(predictions.model),set(self.VERSIONS))
        self.assertNotIn('test_predictions.parquet',[Path(call.args[0]).name for call in reads.call_args_list])
        self.assertNotIn('test_predictions.parquet',[Path(call.args[0]).name for call in hashes.call_args_list])
        pd.testing.assert_frame_equal(frame,s['step5_frame'])

    def test_step5_controls_rebuild_is_rejected_while_running_without_mutation(self):
        s=self.load(5)
        identities={name:s[name] for name in ['step5_dashboard','step5_validate','step5_image',
                    'step5_progress_state','step5_clock_stop','step5_frame','step5_predictions']}
        before_image=bytes(s['step5_image'].value)
        before_status=s['step5_status'].value
        callbacks=list(s['step5_validate']._click_handlers.callbacks)
        s['step5_busy']=True
        with self.assertRaisesRegex(RuntimeError,'Step5 is running'):
            self.rerun_controls(s)
        self.assertTrue(s['step5_busy'])
        for name,value in identities.items():self.assertIs(s[name],value)
        self.assertEqual(bytes(s['step5_image'].value),before_image)
        self.assertEqual(s['step5_status'].value,before_status)
        self.assertEqual(s['step5_validate']._click_handlers.callbacks,callbacks)
        s['step5_busy']=False

    def test_step5_changed_settings_or_input_reject_fit_and_test_keep_old_results(self):
        for changed in ['params','input']:
            with self.subTest(changed=changed):
                s=self.load(5);self.configure(s);s['run_step5']('build')
                frame=s['step5_frame'];metadata_before=s['step5_result_metadata']['lgb_params'].copy()
                original_target=s['model_data'].loc[s['model_data'].index[0],'D_BM_SPREAD']
                if changed=='params':s['LGB_PARAMS']['n_estimators']+=1
                else:s['model_data'].loc[s['model_data'].index[0],'D_BM_SPREAD']+=1
                with patch.dict(s,run_comparison=Mock(side_effect=AssertionError('changed experiment must not fit'))):
                    s['run_step5']('validate')
                self.assertIs(s['step5_frame'],frame)
                self.assertIsNone(s['step5_predictions'])
                self.assertEqual(s['step5_result_metadata']['lgb_params'],metadata_before)
                self.assertIn('Inputs or settings changed',s['step5_status'].value)
                # Reviewable validation remains usable, while an altered experiment
                # cannot lock/open test or publish another pointer.
                if changed=='params':s['LGB_PARAMS'].update(metadata_before)
                else:s['model_data'].loc[s['model_data'].index[0],'D_BM_SPREAD']=original_target
                s['run_step5']('validate');predictions=s['step5_predictions']
                self.assertIsNotNone(predictions)
                pointer=(self.folder/'latest.json').read_bytes()
                if changed=='params':s['LGB_PARAMS']['n_estimators']+=1
                else:s['model_data'].loc[s['model_data'].index[0],'D_BM_SPREAD']+=1
                with patch.dict(s,run_comparison=Mock(side_effect=AssertionError('changed experiment must not open test'))):
                    s['run_step5']('test')
                self.assertIs(s['step5_frame'],frame)
                self.assertIs(s['step5_predictions'],predictions)
                self.assertIsNone(s['step5_test_predictions'])
                self.assertIsNone(s['step5_locked'])
                self.assertEqual((self.folder/'latest.json').read_bytes(),pointer)
                self.assertIn('Inputs or settings changed',s['step5_status'].value)

    def test_step5_validation_checkpoint_failure_keeps_memory_export_retries(self):
        import quote_quality_saved as saved
        s=self.load(5);self.configure(s);s['run_step5']('build')
        frame=s['step5_frame'];pointer=(self.folder/'latest.json').read_bytes()
        with patch.object(saved,'save_step5_checkpoint',side_effect=OSError('synthetic disk full')):
            try:s['run_step5']('validate')
            except OSError:pass  # Memory retention also holds for a surfaced save error.
        predictions=s['step5_predictions']
        self.assertIsNotNone(predictions)
        self.assertEqual(set(predictions.model),set(self.VERSIONS))
        self.assertIs(s['step5_frame'],frame)
        self.assertEqual((self.folder/'latest.json').read_bytes(),pointer)
        self.assertFalse(s['step5_busy']);self.assertTrue(s['step5_clock_stop'].is_set())
        self.assertFalse(s['step5_export'].disabled)
        self.assertIn('retained',s['step5_status'].value)
        with patch.dict(s,run_comparison=Mock(side_effect=AssertionError('export retry must not fit'))):
            s['export_step5']()
        self.assertIs(s['step5_predictions'],predictions)
        restored,pred_saved,manifest=saved.load_saved_validation(self.folder)
        pd.testing.assert_frame_equal(restored,frame)
        pd.testing.assert_frame_equal(pred_saved,predictions)
        self.assertTrue(manifest['validation_available'])

    def test_step5_interrupt_stops_timer_and_retains_completed_features(self):
        s=self.load(5);self.configure(s);s['run_step5']('build');built=s['step5_frame']
        with patch.dict(s,run_comparison=lambda *a,**k:(_ for _ in ()).throw(KeyboardInterrupt())):
            s['run_step5']('validate')
        self.assertIs(s['step5_frame'],built)
        self.assertFalse(s['step5_busy'])
        self.assertTrue(s['step5_clock_stop'].is_set())
        self.assertEqual(s['step5_progress'].bar_style,'warning')
        self.assertIn('Interrupted',s['step5_status'].value)
        self.assertFalse(s['step5_validate'].disabled)
        self.assertFalse(s['step5_export'].disabled)
        self.assertTrue(s['step5_save'].disabled)


if __name__=='__main__':unittest.main()
