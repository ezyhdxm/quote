# %% [markdown]
# # Step 5 — BondCliQ increment over the existing BASE_FEATURES model
# Run three cells. Loading is included; output is one shareable dashboard.

# %% 1. Load data
from pathlib import Path
import numpy as np
import pandas as pd
from io import BytesIO
from textwrap import fill
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import ipywidgets as widgets
from IPython.display import display
from html import escape
from threading import Event, Thread
from time import perf_counter

PIPELINE_CSV = Path("data/pipeline/data_pipeline.csv_20260506")
BENCHMARK_CSV = Path("data/pipeline/DailyCloseUSTBenchmarks.csv_20260506")
RAW_QUOTES_FILE = Path("data/bondcliq/quotes_pretrade_260301_260401_Wells_quotes2.parquet")
DATA_IG_CACHE = Path("data/pipeline/data_ig.parquet")

def to_ny_datetime(series):
    values = pd.to_datetime(series, errors="coerce", format="mixed")
    if values.dt.tz is None:
        return values.dt.tz_localize("America/New_York")
    return values.dt.tz_convert("America/New_York")

if DATA_IG_CACHE.exists():
    data_ig = pd.read_parquet(DATA_IG_CACHE)
else:
    from data import load_merged_prints
    data_ig = load_merged_prints(
        pipeline_csv=PIPELINE_CSV, benchmark_csv=BENCHMARK_CSV,
        liquid_trade_count_threshold=15, qav_threshold_ig=0.005, qav_threshold_hy=0.2,
    )
    data_ig["EFFECTIVE_DATETIME_TS"] = to_ny_datetime(data_ig["EFFECTIVE_DATETIME_TS"])
    DATA_IG_CACHE.parent.mkdir(parents=True, exist_ok=True)
    data_ig.to_parquet(DATA_IG_CACHE, index=False)

bcq_df = pd.read_parquet(RAW_QUOTES_FILE)
bcq_df = bcq_df.loc[bcq_df["cusip"].isin(data_ig["CUSIP"])].copy()
bcq_df["quote_timestamp_ET"] = pd.to_datetime(
    bcq_df["quote_timestamp_UTC"], utc=True, errors="coerce", format="mixed",
).dt.tz_convert("America/New_York")
cusip_issuer = (data_ig[["ISSUER", "CUSIP"]].dropna()
                .drop_duplicates("CUSIP").set_index("CUSIP")["ISSUER"])
bcq_df["ISSUER"] = bcq_df["cusip"].map(cusip_issuer)


from quote_quality_core import (BASE_FEATURES, BASE_CAT_FEATURES, build_quote_features,
    chronological_split, run_comparison, representative_issuers, model_versions)

# %% [markdown]
# ## Same prediction task, finite quote alternatives
# BASE_FEATURES are copied from the supplied training code. They already exist in data_ig.
# Every model predicts D_BM_SPREAD and adds PREV_BM_SPREAD; evaluate against BM_SPREAD in bps.
# Missing quote rows stay in every version. No adaptive anchor / extended feature sets are added.
# Run validation first; lock one version for the final test. All charts appear together.

# %% 2. Experiment settings and plot
TIME_COL = 'EFFECTIVE_DATETIME_TS'
TARGET_COL = 'D_BM_SPREAD'
ANCHOR_COL = 'PREV_BM_SPREAD'
AGE_MIN, SYNC_MIN = 30, 1
ALLOW_EXACT_QUOTES = True  # event time = known time; set False for strict-before sensitivity.
# Short-file pilot: use the final quote date as the predeclared experiment end.
# These split settings differ from the older 60/10/10-day notebook to fit the short quote period.
VALIDATION_DAYS, TEST_DAYS, EMBARGO_DAYS, MIN_TRAIN_DAYS = 5, 5, 2, 10
LGB_PARAMS = dict(objective='mae',boosting_type='dart',n_estimators=400,learning_rate=.2,
    num_leaves=127,max_bin=511,max_depth=-1,min_child_samples=20,min_split_gain=0.,
    subsample=1.,subsample_freq=0,colsample_bytree=1.,reg_alpha=0.,reg_lambda=0.,
    n_jobs=8,verbosity=-1,random_state=2026)

required=BASE_FEATURES+[TARGET_COL,ANCHOR_COL,'BM_SPREAD',TIME_COL,'CUSIP','ISSUER']
missing=sorted(set(required)-set(data_ig.columns))
if missing:raise ValueError('data_ig is missing required existing model columns: '+', '.join(missing))
model_data=data_ig.copy().reset_index(drop=True)
model_data[[TARGET_COL,ANCHOR_COL,'BM_SPREAD']]=model_data[[TARGET_COL,ANCHOR_COL,'BM_SPREAD']].apply(pd.to_numeric,errors='coerce')
model_data['row_id']=np.arange(len(model_data))  # Stable within this loaded frame; never join on time alone.
model_data['time']=to_ny_datetime(model_data[TIME_COL])
model_data['cusip']=model_data.CUSIP
valid_target=np.isfinite(model_data[[TARGET_COL,ANCHOR_COL,'BM_SPREAD']].apply(pd.to_numeric,errors='coerce')).all(axis=1)
valid_keys=model_data[['cusip','time']].notna().all(axis=1)
excluded_target_rows=int((~(valid_target&valid_keys)).sum())
model_data=model_data.loc[valid_target&valid_keys].copy()
quote_end=bcq_df.quote_timestamp_ET.max()
model_data['split'],model_data['refit_train']=chronological_split(model_data,quote_end,
    VALIDATION_DAYS,TEST_DAYS,EMBARGO_DAYS,MIN_TRAIN_DAYS)
# Include no-quote events within the same date experiment; no quote-coverage row filtering.
model_data=model_data.loc[model_data.time.dt.normalize().le(quote_end.normalize())].copy()


def validation_figure(frame, predictions=None, focus=None, stage='Readiness', title='All issuers'):
    fig=Figure(figsize=(17,11),facecolor='white');axes=fig.subplots(2,2)
    a,b,c,d=axes.flat;fig.subplots_adjust(top=.84,bottom=.15,hspace=.45,wspace=.26)
    fig.suptitle(f'Step 5 | {stage} | {title}\nSame BASE_FEATURES, D_BM_SPREAD target and PREV_BM_SPREAD anchor',fontsize=16,y=.98)
    if predictions is None or predictions.empty:
        caption=f'{len(frame):,} eligible experiment rows | {excluded_target_rows:,} invalid target/key rows excluded once | quote coverage does not remove rows'
    else:
        caption=(f'{predictions.row_id.nunique():,} {stage.lower()} trades | {int(predictions.train_n.iloc[0]):,} training rows | '
                 f'{predictions.time.min():%Y-%m-%d} to {predictions.time.max():%Y-%m-%d} ET | same evaluation rows for every version')
    fig.text(.5,.90,caption,ha='center',fontsize=10)
    if predictions is None or predictions.empty:
        splits=['Train','Validation','Test'];n=frame.groupby('split').size().reindex(splits,fill_value=0)
        covered=frame.loc[frame.bcq_has_quote.gt(0)].groupby('split').size().reindex(splits,fill_value=0)
        a.bar(splits,n,color='#BBBBBB',label='All target rows');a.bar(splits,covered,color='#277F8E',label='Any quote')
        for i,k in enumerate(splits):a.text(i,n[k],f'{covered[k]:,}/{n[k]:,}',ha='center',va='bottom',fontsize=9)
        a.set_title('Training and evaluation support');a.set_ylabel('Rows (covered / all)')
        daily=frame.assign(day=frame.time.dt.normalize()).groupby('day').agg(coverage=('bcq_has_quote','mean'),pairs=('bcq_n_pair',lambda x:x.gt(0).mean()),matched=('bcq_n_size_time_pair',lambda x:x.gt(0).mean()))
        for col,label,color in [('coverage','Any single-side quote','#277F8E'),('pairs','Fresh same-dealer pair','#C9563D'),('matched','Size + short-gap pair','#8560A5')]:
            b.plot(daily.index,daily[col],marker='.',label=label,color=color)
        b.set_ylim(-.02,1.05);b.set_title('Daily feature coverage');b.set_ylabel('Fraction of all target rows');b.tick_params(axis='x',rotation=25)
        shifts=[];labels=[];support=[]
        for side in ['bid','ask']:
            for col,label in [('center_decay','Decay'),('center_max_age','Max age'),('center_candidate_clip','Clip')]:
                delta=(frame[f'bcq_{side}_{col}']-frame[f'bcq_{side}_center_equal']).abs()
                if col=='center_candidate_clip':delta=delta.where(frame[f'bcq_{side}_n_peer_supported'].gt(0))
                delta=delta.dropna();support.append(len(delta))
                shifts.append([delta.median(),delta.quantile(.95)] if len(delta) else [np.nan,np.nan]);labels.append(f'{side}\n{label}')
        vals=np.asarray(shifts);xx=np.arange(len(labels));c.bar(xx-.18,vals[:,0],.36,label='Median',color='#277F8E');c.bar(xx+.18,vals[:,1],.36,label='95th percentile',color='#C9563D')
        for i,count in enumerate(support):
            c.text(i,0,f'n={count:,}' if count else 'Unassessed',rotation=90,ha='center',va='bottom',fontsize=8)
        c.set_xticks(xx,labels);c.set_title('Rule effect; clip requires peer support');c.set_ylabel('|level change| (bps)')
        d.axis('off');d.text(.03,.92,'Ready for the existing target',fontsize=14,weight='bold')
        d.text(.03,.79,'1. Inspect train / validation / test quote coverage.\n2. Run validation on all issuers and all target rows.\n3. Select a version from validation only.\n4. Run the locked final test once.\n\nClipping falls back when peers are insufficient.\nA zero numerical change alone does not validate it.\nNo quote label or target is inferred from these plots.',va='top',fontsize=12,linespacing=1.6)
    else:
        summary=predictions.groupby('model').abs_error_bps.mean().sort_values()
        bars=a.barh(summary.index,summary,color=['#C9563D' if n=='Base' else '#277F8E' for n in summary.index])
        a.bar_label(bars,fmt='%.3f',padding=3,fontsize=9);a.margins(x=.18)
        a.set_xlabel('MAE (bps)');a.set_title(f'{stage}: all target rows, same population')
        chosen=focus if focus in summary.index else summary.index[0]
        wide=predictions.pivot(index='row_id',columns='model',values='abs_error_bps')
        reference='Base' if chosen in ['Base','Quote levels'] else ('Quote levels' if chosen=='Reliability' else 'Reliability')
        if reference not in wide:reference='Base'
        delta=wide[chosen]-wide[reference]
        keys=predictions.drop_duplicates('row_id').set_index('row_id').reindex(delta.index)
        daily=delta.groupby(keys.time.dt.normalize()).mean()
        b.axhline(0,color='#777777');b.plot(daily.index,daily,marker='o',color='#277F8E');b.tick_params(axis='x',rotation=25)
        b.set_title(f'{chosen} minus {reference}: daily loss');b.set_ylabel('MAE difference (bps; lower is better)')
        masks={'No quote':keys.bcq_has_quote.eq(0),'Covered':keys.bcq_has_quote.gt(0),'Paired':keys.bcq_n_pair.gt(0)}
        for value in keys.TRADE_TYPE.dropna().unique():masks[f'Type {value}']=keys.TRADE_TYPE.eq(value)
        masks['<=100K par']=pd.to_numeric(keys.QUANTITY,errors='coerce').le(100_000)
        masks['>=1MM par']=pd.to_numeric(keys.QUANTITY,errors='coerce').ge(1_000_000)
        stats=[(label,delta[mask].mean(),int(mask.sum())) for label,mask in masks.items() if mask.any()]
        c.barh([f'{k} (n={n:,})' for k,v,n in stats],[v for k,v,n in stats],color='#277F8E');c.axvline(0,color='#777777');c.set_xlabel('MAE difference (bps)');c.set_title('Same-row subgroup differences; groups overlap')
        tails=predictions.groupby('model').abs_error_bps.agg(p95=lambda v:v.quantile(.95),gt10=lambda v:v.gt(10).mean())
        d.barh(tails.index,tails.p95,color='#8560A5');d.set_xlabel('95th percentile absolute error (bps)');d.set_title('Tail errors on all target rows')
    for ax in [a,b,c,d]:
        ax.grid(alpha=.15);h,l=ax.get_legend_handles_labels()
        if h:ax.legend(fontsize=9)
    fig.text(.08,.075,'No full-sample model selection: validation chooses the version; the final test is evaluated only after the choice is locked.',fontsize=11)
    fig.text(.08,.04,'Pilot windows: final quote date, prior training history, 5 validation dates + 5 test dates, 2-date training embargo. All variants share the same rows and training budget.',fontsize=10)
    return fig

# %% [markdown]
# ## Run and export
# Preview one issuer for speed; training always builds the full experiment, with no screenshot-selected sample.
# Build all features writes no file until Export is clicked. Export keeps row_id, original baseline columns and features.
# Validation reports MAE, day-level differences, coverage/side/size slices and tail errors together.
# Test refits the locked choice and its comparison baselines on pre-test data. Re-running controls never tunes on test.
# Progress shows completed bonds/models and current training iterations; elapsed time updates every second.
# Preprocessing has no reliable percentage. Keep the existing run; progress cannot attach to an older running cell.

# %% 3. Research controls
if 'step5_dashboard' in globals():
    if 'step5_clock_stop' in globals():step5_clock_stop.set()
    for button,callback in step5_callbacks:button.on_click(callback,remove=True)
    step5_selected.unobserve(show_step5_choice,names='value')
    step5_dashboard.close()
step5_issuer=widgets.Dropdown(options=representative_issuers(bcq_df),description='Preview:',layout=widgets.Layout(width='850px'))
step5_preview=widgets.Button(description='Preview issuer')
step5_build=widgets.Button(description='Build all features')
step5_validate=widgets.Button(description='Run validation',button_style='primary')
step5_test=widgets.Button(description='Run locked test',disabled=True)
step5_selected=widgets.Dropdown(options=['Base'],description='Choice:',disabled=True)
step5_export=widgets.Button(description='Export features')
step5_save=widgets.Button(description='Save all PNG',icon='download',disabled=True)
step5_status=widgets.HTML();step5_image=widgets.Image(format='png',layout=widgets.Layout(width='100%',max_width='1600px'))
step5_progress=widgets.IntProgress(description='Bonds:',min=0,max=1,value=0,style={'description_width':'initial'},layout=widgets.Layout(width='650px'))
step5_rounds=widgets.IntProgress(description='Iterations:',min=0,max=1,value=0,layout=widgets.Layout(width='650px',display='none'))
step5_detail=widgets.HTML();step5_elapsed=widgets.HTML()
step5_clock_stop=Event()
step5_progress_state={'last_update':0.,'context':''}
step5_frame=None;step5_predictions=None;step5_test_predictions=None;step5_locked=None;step5_busy=False


def report_step5(stage,completed=None,total=None,detail=''):
    # Throttle UI messages, not the calculation or counts. Always show model transitions and completion.
    now=perf_counter()
    final=total is not None and completed==total
    transition=(stage=='features' and step5_progress_state.get('phase')!='features') or (stage=='fit' and step5_rounds.layout.display=='none')
    if stage in ['features','models']:step5_progress_state['context']=detail
    if stage!='models' and not final and not transition and now-step5_progress_state['last_update']<.2:return
    step5_progress_state['last_update']=now
    if stage in ['features','models']:
        unit='Bonds' if stage=='features' else 'Models'
        step5_progress.description=f'{unit}: {int(completed or 0):,}/{int(total or 0):,}'
        step5_progress.max=max(1,int(total or 0));step5_progress.value=int(completed or 0)
        step5_rounds.layout.display='none'
        step5_progress_state['phase']=stage
    elif stage=='fit':
        step5_rounds.layout.display=''
        step5_rounds.max=max(1,int(total or 0));step5_rounds.value=int(completed or 0)
    context=step5_progress_state['context'] if stage=='events' else ''
    count=f' — {completed:,}/{total:,}' if completed is not None and total is not None else ''
    step5_detail.value=escape(f'{context} | {detail}' if context else detail)+escape(count)


def step5_clock(stop,started,widget):
    # This timer only updates display text; no training or data work runs in the thread.
    while not stop.wait(1):
        seconds=int(perf_counter()-started)
        widget.value=f'Elapsed: {seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'


def run_step5(action='preview'):
    global step5_frame,step5_predictions,step5_test_predictions,step5_locked,step5_figure,step5_busy,step5_clock_stop
    if step5_busy:return
    step5_busy=True
    for button in [step5_preview,step5_build,step5_validate,step5_test,step5_export,step5_save]:button.disabled=True
    step5_selected.disabled=True;step5_issuer.disabled=True
    step5_image.value=b''
    step5_progress.value=0;step5_progress.max=1;step5_progress.bar_style='info'
    step5_rounds.layout.display='none';step5_rounds.bar_style='info'
    step5_progress_state.update(last_update=0.,context='',phase='')
    step5_clock_stop=Event();started=perf_counter();step5_elapsed.value='Elapsed: 00:00:00'
    clock=Thread(target=step5_clock,args=(step5_clock_stop,started,step5_elapsed),daemon=True);clock.start()
    try:
        step5_status.value='Running. Counts show completed work; bond sizes and model times vary.'
        report_step5('events',detail='Preparing input rows')
        if action=='preview':
            sample=model_data.loc[model_data.ISSUER.eq(step5_issuer.value)]
            if sample.empty:raise ValueError('No eligible target rows for this issuer through the quote end date. Choose another issuer.')
            q=sample[['row_id','cusip','time']]
            features=build_quote_features(bcq_df.loc[bcq_df.ISSUER.eq(step5_issuer.value)],q,AGE_MIN,SYNC_MIN,ALLOW_EXACT_QUOTES,progress=report_step5)
            shown=sample.merge(features.drop(columns=['cusip','time']),on='row_id',validate='one_to_one')
            step5_figure=validation_figure(shown,title=step5_issuer.value)
        else:
            if step5_frame is None:
                features=build_quote_features(bcq_df,model_data[['row_id','cusip','time']],AGE_MIN,SYNC_MIN,ALLOW_EXACT_QUOTES,progress=report_step5)
                step5_frame=model_data.merge(features.drop(columns=['cusip','time']),on='row_id',validate='one_to_one')
            else:
                n_bonds=step5_frame.cusip.nunique()
                report_step5('features',n_bonds,n_bonds,'Reusing completed in-memory feature frame')
            if action=='validate':
                if step5_locked is not None:raise ValueError('Test choice is already locked. Start a new experiment explicitly before changing it.')
                step5_predictions,_=run_comparison(step5_frame,LGB_PARAMS,progress=report_step5)
                order=step5_predictions.groupby('model').abs_error_bps.mean().sort_values().index.tolist()
                step5_selected.options=order;step5_selected.value=order[0]
                step5_figure=validation_figure(step5_frame,step5_predictions,step5_selected.value,'Validation')
            elif action=='test':
                if step5_predictions is None:raise ValueError('Run validation before opening the final test')
                if step5_locked is None:step5_locked=step5_selected.value
                step5_selected.disabled=True
                if step5_test_predictions is None:step5_test_predictions,_=run_comparison(step5_frame,LGB_PARAMS,'Test',step5_locked,progress=report_step5)
                step5_figure=validation_figure(step5_frame,step5_test_predictions,step5_locked,'Locked test')
            else:step5_figure=validation_figure(step5_frame)
        step5_detail.value='Rendering the four-panel dashboard...'
        with BytesIO() as buffer:
            step5_figure.savefig(buffer,format='png',dpi=110);step5_image.value=buffer.getvalue()
        step5_save.disabled=False
        step5_progress.bar_style='success';step5_rounds.bar_style='success'
        step5_detail.value='Complete. Features and predictions remain in this kernel; use Export features to save.'
        step5_status.value='Finished. All four panels are ready to share; no large tables printed.'
    except KeyboardInterrupt:
        step5_status.value='Interrupted. Any fully built feature frame remains in this kernel; partial model results are not saved.'
        step5_progress.bar_style='warning';step5_rounds.bar_style='warning'
    except Exception as error:
        step5_status.value=escape(f'{type(error).__name__}: {error}')
        step5_progress.bar_style='danger';step5_rounds.bar_style='danger'
        step5_save.disabled=True
        if not isinstance(error,ValueError):raise
    finally:
        step5_clock_stop.set();clock.join(timeout=.2)
        seconds=int(perf_counter()-started)
        step5_elapsed.value=f'Elapsed: {seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'
        step5_busy=False
        for button in [step5_preview,step5_build]:button.disabled=False
        step5_issuer.disabled=False
        step5_export.disabled=step5_frame is None
        step5_selected.disabled=step5_predictions is None or step5_locked is not None
        step5_validate.disabled=step5_locked is not None
        step5_test.disabled=step5_predictions is None


def export_step5(_=None):
    if step5_frame is None:
        step5_status.value='Build all features before exporting.';return
    folder=Path('outputs/quote_quality_step5');folder.mkdir(parents=True,exist_ok=True)
    export,specs=model_versions(step5_frame)
    export.to_parquet(folder/'model_features.parquet',index=False)
    for label,pred in [('validation',step5_predictions),('test',step5_test_predictions)]:
        if pred is not None:pred.to_parquet(folder/f'{label}_predictions.parquet',index=False)
    import json
    manifest=dict(base_features=BASE_FEATURES,target=TARGET_COL,anchor=ANCHOR_COL,
        error_multiplier=100,quote_spread_unit='bps',allow_exact_quotes=ALLOW_EXACT_QUOTES,
        age_min=AGE_MIN,sync_min=SYNC_MIN,lgb_params=LGB_PARAMS,locked_choice=step5_locked,
        val_days=VALIDATION_DAYS,test_days=TEST_DAYS,embargo_days=EMBARGO_DAYS,
        model_columns={name:columns for name,(columns,_) in specs.items()},
        split_dates={label:sorted(group.time.dt.strftime('%Y-%m-%d').unique()) for label,group in step5_frame.groupby('split')})
    (folder/'experiment.json').write_text(json.dumps(manifest,indent=2))
    step5_status.value=f'Saved features, experiment settings and available predictions to {folder}'


def save_step5(_=None):
    folder=Path('outputs/quote_quality_step5');folder.mkdir(parents=True,exist_ok=True)
    step5_figure.savefig(folder/'dashboard.png',dpi=180);step5_status.value=f'Saved {folder}/dashboard.png'

def show_step5_choice(change=None):
    global step5_figure
    if step5_busy or step5_predictions is None or step5_locked is not None:return
    step5_figure=validation_figure(step5_frame,step5_predictions,step5_selected.value,'Validation')
    with BytesIO() as buffer:
        step5_figure.savefig(buffer,format='png',dpi=110);step5_image.value=buffer.getvalue()
    step5_save.disabled=False


step5_callbacks=[(step5_preview,lambda _:run_step5('preview')),(step5_build,lambda _:run_step5('build')),
    (step5_validate,lambda _:run_step5('validate')),(step5_test,lambda _:run_step5('test')),
    (step5_save,save_step5),(step5_export,export_step5)]
for button,callback in step5_callbacks:button.on_click(callback)
step5_selected.observe(show_step5_choice,names='value')
step5_dashboard=widgets.VBox([step5_issuer,widgets.HBox([step5_preview,step5_build,step5_validate]),widgets.HBox([step5_selected,step5_test,step5_export,step5_save]),step5_status,step5_progress,step5_rounds,step5_detail,step5_elapsed,step5_image])
display(step5_dashboard)
run_step5('preview')
