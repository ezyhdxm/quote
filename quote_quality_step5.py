# %% [markdown]
# # Step 5 — BondCliQ increment over the existing BASE_FEATURES model
# First run: cells 1–3 include loading and the original dashboard.
# Completed validation: run only the final research cell in the existing idle kernel.

# %% 1. Load data
# SETUP LOGIC: Import notebook/figure controls; no data or training work is performed by the imports.
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

# CONFIGURATION LOGIC: Keep existing loader paths unchanged, including the traded-bond data_ig cache.
PIPELINE_CSV = Path("data/pipeline/data_pipeline.csv_20260506")
BENCHMARK_CSV = Path("data/pipeline/DailyCloseUSTBenchmarks.csv_20260506")
RAW_QUOTES_FILE = Path("data/bondcliq/quotes_pretrade_260301_260401_Wells_quotes2.parquet")
DATA_IG_CACHE = Path("data/pipeline/data_ig.parquet")

# SETUP LOGIC: Define to_ny_datetime; calling is explicit, not triggered by this declaration.
def to_ny_datetime(series):
    # CORE LOGIC: STEP 1 — Normalize scalar timestamps to ET without inventing missing times.
    # Input: aware input=2026-03-19 14:00 UTC; naive input=2026-03-19 10:00.
    # Output: Both represent 2026-03-19 10:00 America/New_York; invalid input becomes NaT.
    # Trick: Naive timestamps are treated as already local ET; aware timestamps are converted, not relocalized.
    values = pd.to_datetime(series, errors="coerce", format="mixed")
    if values.dt.tz is None:
        return values.dt.tz_localize("America/New_York")
    return values.dt.tz_convert("America/New_York")

# CACHEING LOGIC: Reuse the existing loader/cache; changing comments does not rerun this cell.
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

# CACHEING LOGIC: Read the original quote Parquet without changing loader behavior.
bcq_df = pd.read_parquet(RAW_QUOTES_FILE)
# CORE LOGIC: STEP 1 — Restrict quotes to the existing three-month traded-bond universe and derive known ET time.
# Input: data_ig.CUSIP=['X']; raw quote cusips=['X','Y']; X known timestamp=2026-03-19 14:00 UTC.
# Output: Keep X, exclude Y; X known time=2026-03-19 10:00 ET; exact time is not rounded.
bcq_df = bcq_df.loc[bcq_df["cusip"].isin(data_ig["CUSIP"])].copy()
bcq_df["quote_timestamp_ET"] = pd.to_datetime(
    bcq_df["quote_timestamp_UTC"], utc=True, errors="coerce", format="mixed",
).dt.tz_convert("America/New_York")
# CORE LOGIC: STEP 2 — Attach the existing issuer lookup used by the preview.
# Input: data_ig rows X/Acme, X/Acme, Y/Beta, Z/missing issuer; quote cusips X and Z.
# Output: Lookup has X:Acme and Y:Beta; X maps to Acme, unmatched Z remains missing.
# Trick: This existing first-per-CUSIP preview lookup is not the causal donor mapping of the new issuer-movement module.
cusip_issuer = (data_ig[["ISSUER", "CUSIP"]].dropna()
                .drop_duplicates("CUSIP").set_index("CUSIP")["ISSUER"])
bcq_df["ISSUER"] = bcq_df["cusip"].map(cusip_issuer)


# SETUP LOGIC: Bind the shared computations without building any quote features at import.
from quote_quality_core import (BASE_FEATURES, BASE_CAT_FEATURES, build_quote_features,
    chronological_split, run_comparison, representative_issuers, model_versions)
from quote_quality_cache import prepare_quote_events

# %% [markdown]
# ## Same prediction task, finite quote alternatives
# BASE_FEATURES are copied from the supplied training code. They already exist in data_ig.
# Every model predicts D_BM_SPREAD and adds PREV_BM_SPREAD; evaluate against BM_SPREAD in bps.
# Missing quote rows stay in every version. No adaptive anchor / extended feature sets are added.
# Run validation first; lock one version for the final test. All charts appear together.

# %% 2. Experiment settings and plot
# CONFIGURATION LOGIC: Declare the original target/anchor, 30min age, 1min sync, exact known-time policy and fixed four-model budget.
TIME_COL = 'EFFECTIVE_DATETIME_TS'
TARGET_COL = 'D_BM_SPREAD'
ANCHOR_COL = 'PREV_BM_SPREAD'
AGE_MIN, SYNC_MIN = 30, 1
ALLOW_EXACT_QUOTES = True  # event time = known time; set False for strict-before sensitivity.
# Short-file pilot: use the final quote date as the predeclared experiment end.
# These split settings differ from the older 60/10/10-day notebook to fit the short quote period.
VALIDATION_DAYS, TEST_DAYS, EMBARGO_DAYS, MIN_TRAIN_DAYS = 5, 5, 2, 10
# Recover only the existing comparison chain after an unsaved kernel result was lost.
# This is declared before fitting; it does not change the model or evaluation rows.
VALIDATION_VERSIONS = ('Base', 'Quote levels', 'Reliability', 'Age decay')
LGB_PARAMS = dict(objective='mae',boosting_type='dart',n_estimators=400,learning_rate=.2,
    num_leaves=127,max_bin=511,max_depth=-1,min_child_samples=20,min_split_gain=0.,
    subsample=1.,subsample_freq=0,colsample_bytree=1.,reg_alpha=0.,reg_lambda=0.,
    n_jobs=8,verbosity=-1,random_state=2026)

# CORE LOGIC: STEP 3 — Require BASE14 and create a stable row_id for each existing target trade.
# Input: data_ig has 3 rows with EFFECTIVE_DATETIME_TS, CUSIP, ISSUER and all 14 BASE columns; anchor text='0.60'.
# Output: row_id=[0,1,2], anchor becomes 0.60, time is ET and cusip copies CUSIP; an absent COUPON column rejects; a missing COUPON value is retained.
# Trick: Two trades at the same timestamp retain distinct row_id values; joins never collapse them by time.
required=BASE_FEATURES+[TARGET_COL,ANCHOR_COL,'BM_SPREAD',TIME_COL,'CUSIP','ISSUER']
missing=sorted(set(required)-set(data_ig.columns))
if missing:raise ValueError('data_ig is missing required existing model columns: '+', '.join(missing))
model_data=data_ig.copy().reset_index(drop=True)
model_data[[TARGET_COL,ANCHOR_COL,'BM_SPREAD']]=model_data[[TARGET_COL,ANCHOR_COL,'BM_SPREAD']].apply(pd.to_numeric,errors='coerce')
model_data['row_id']=np.arange(len(model_data))  # Stable within this loaded frame; never join on time alone.
model_data['time']=to_ny_datetime(model_data[TIME_COL])
model_data['cusip']=model_data.CUSIP
# CORE LOGIC: STEP 4 — Exclude invalid target/key rows once, not according to quote coverage.
# Input: rows 0/1/2 have finite targets [True,False,True], valid keys [True,True,False]; row 0 has no quote.
# Output: excluded_target_rows=2; only no-quote row 0 remains eligible.
valid_target=np.isfinite(model_data[[TARGET_COL,ANCHOR_COL,'BM_SPREAD']].apply(pd.to_numeric,errors='coerce')).all(axis=1)
valid_keys=model_data[['cusip','time']].notna().all(axis=1)
excluded_target_rows=int((~(valid_target&valid_keys)).sum())
model_data=model_data.loc[valid_target&valid_keys].copy()
# CORE LOGIC: STEP 5 — Apply the existing chronological pilot dates and original refit mask.
# Input: 22 ET weekday trade dates from 2026-03-02 through 03-31; final quote day=03-31; val/test/embargo=5/5/2.
# Output: Train=03-02..03-13, Validation=03-18..03-24, Test=03-25..03-31; refit includes through 03-20.
# Trick: The two refit embargo dates are the final two Validation dates; no extra two dates are inserted before Test.
quote_end=bcq_df.quote_timestamp_ET.max()
model_data['split'],model_data['refit_train']=chronological_split(model_data,quote_end,
    VALIDATION_DAYS,TEST_DAYS,EMBARGO_DAYS,MIN_TRAIN_DAYS)
# Include no-quote events within the same date experiment; no quote-coverage row filtering.
model_data=model_data.loc[model_data.time.dt.normalize().le(quote_end.normalize())].copy()


# PLOTTING LOGIC: Define validation_figure; calling is explicit, not triggered by this declaration.
def validation_figure(frame, predictions=None, focus=None, stage='Readiness', title='All issuers'):
    # PLOTTING LOGIC: Create one figure and the fixed four-panel layout.
    fig=Figure(figsize=(17,11),facecolor='white');axes=fig.subplots(2,2)
    a,b,c,d=axes.flat;fig.subplots_adjust(top=.84,bottom=.15,hspace=.45,wspace=.26)
    fig.suptitle(f'Step 5 | {stage} | {title}\nSame BASE_FEATURES, D_BM_SPREAD target and PREV_BM_SPREAD anchor',fontsize=16,y=.98)
    # CORE LOGIC: STEP 1 — Report the actual target and training populations for this view.
    # Input: completed predictions have row_ids=[7,7,8,8], train_n=40 and Validation dates=03-19..03-20.
    # Output: Caption reports 2 validation trades and 40 training rows, not 4 prediction records.
    if predictions is None or predictions.empty:
        caption=f'{len(frame):,} eligible experiment rows | {excluded_target_rows:,} invalid target/key rows excluded once | quote coverage does not remove rows'
    else:
        caption=(f'{predictions.row_id.nunique():,} {stage.lower()} trades | {int(predictions.train_n.iloc[0]):,} training rows | '
                 f'{predictions.time.min():%Y-%m-%d} to {predictions.time.max():%Y-%m-%d} ET | same evaluation rows for every version')
    # PLOTTING LOGIC: Place the prepared population caption in the figure.
    fig.text(.5,.90,caption,ha='center',fontsize=10)
    # CORE LOGIC: STEP 2 — Compute split-level support over all eligible targets.
    # Input: Train/Validation/Test row counts=[4,2,2]; quoted row counts=[3,1,0].
    # Output: n=[4,2,2], covered=[3,1,0]; no-quote rows remain in each denominator.
    if predictions is None or predictions.empty:
        splits=['Train','Validation','Test'];n=frame.groupby('split').size().reindex(splits,fill_value=0)
        covered=frame.loc[frame.bcq_has_quote.gt(0)].groupby('split').size().reindex(splits,fill_value=0)
        # PLOTTING LOGIC: Draw the split support counts and labels.
        a.bar(splits,n,color='#BBBBBB',label='All target rows');a.bar(splits,covered,color='#277F8E',label='Any quote')
        for i,k in enumerate(splits):a.text(i,n[k],f'{covered[k]:,}/{n[k]:,}',ha='center',va='bottom',fontsize=9)
        a.set_title('Training and evaluation support');a.set_ylabel('Rows (covered / all)')
        # CORE LOGIC: STEP 3 — Compute daily coverage, fresh-pair and size-time-pair fractions.
        # Input: one ET day has two targets with has_quote=[1,0], n_pair=[1,0], n_size_time_pair=[0,0].
        # Output: Daily fractions are coverage=0.5, pairs=0.5, matched=0.
        daily=frame.assign(day=frame.time.dt.normalize()).groupby('day').agg(coverage=('bcq_has_quote','mean'),pairs=('bcq_n_pair',lambda x:x.gt(0).mean()),matched=('bcq_n_size_time_pair',lambda x:x.gt(0).mean()))
        # PLOTTING LOGIC: Draw the already computed daily coverage curves.
        for col,label,color in [('coverage','Any single-side quote','#277F8E'),('pairs','Fresh same-dealer pair','#C9563D'),('matched','Size + short-gap pair','#8560A5')]:
            b.plot(daily.index,daily[col],marker='.',label=label,color=color)
        b.set_ylim(-.02,1.05);b.set_title('Daily feature coverage');b.set_ylabel('Fraction of all target rows');b.tick_params(axis='x',rotation=25)
        # CORE LOGIC: STEP 4 — Measure same-target rule shifts and exclude unsupported clip effects from assessment.
        # Input: bid equal centers=[100,100], decay=[101,103], clip=[100,102], peer support=[0,1].
        # Output: Decay shifts=[1,3], median=2, P95=2.9; Clip assessed shifts=[2], support=1.
        # Trick: A zero fallback from insufficient clipping peers is not counted as an assessed zero effect.
        shifts=[];labels=[];support=[]
        for side in ['bid','ask']:
            for col,label in [('center_decay','Decay'),('center_max_age','Max age'),('center_candidate_clip','Clip')]:
                delta=(frame[f'bcq_{side}_{col}']-frame[f'bcq_{side}_center_equal']).abs()
                if col=='center_candidate_clip':delta=delta.where(frame[f'bcq_{side}_n_peer_supported'].gt(0))
                delta=delta.dropna();support.append(len(delta))
                shifts.append([delta.median(),delta.quantile(.95)] if len(delta) else [np.nan,np.nan]);labels.append(f'{side}\n{label}')
        # PLOTTING LOGIC: Draw rule-effect summaries and readiness text; this does not validate a prediction model.
        vals=np.asarray(shifts);xx=np.arange(len(labels));c.bar(xx-.18,vals[:,0],.36,label='Median',color='#277F8E');c.bar(xx+.18,vals[:,1],.36,label='95th percentile',color='#C9563D')
        for i,count in enumerate(support):
            c.text(i,0,f'n={count:,}' if count else 'Unassessed',rotation=90,ha='center',va='bottom',fontsize=8)
        c.set_xticks(xx,labels);c.set_title('Rule effect; clip requires peer support');c.set_ylabel('|level change| (bps)')
        d.axis('off');d.text(.03,.92,'Ready for the existing target',fontsize=14,weight='bold')
        d.text(.03,.79,'1. Inspect train / validation / test quote coverage.\n2. Run validation on all issuers and all target rows.\n3. Select a version from validation only.\n4. Run the locked final test once.\n\nClipping falls back when peers are insufficient.\nA zero numerical change alone does not validate it.\nNo quote label or target is inferred from these plots.',va='top',fontsize=12,linespacing=1.6)
    else:
        # CORE LOGIC: STEP 5 — Compute common-population model MAE.
        # Input: Base absolute errors=[1,3], Quote levels=[1,2] for the same row_ids 7/8.
        # Output: Sorted MAE is Quote levels=1.5 bps, Base=2.0 bps.
        summary=predictions.groupby('model').abs_error_bps.mean().sort_values()
        # PLOTTING LOGIC: Draw the computed MAE ranking with formatted labels.
        bars=a.barh(summary.index,summary,color=['#C9563D' if n=='Base' else '#277F8E' for n in summary.index])
        a.bar_label(bars,fmt='%.3f',padding=3,fontsize=9);a.margins(x=.18)
        a.set_xlabel('MAE (bps)');a.set_title(f'{stage}: all target rows, same population')
        # CORE LOGIC: STEP 6 — Use the direct prior model and align daily losses by target id.
        # Input: chosen=Age decay; rows 7/8 have Age errors=[1.9,2.1], Reliability=[2.0,2.1], on 03-19/03-20.
        # Output: Reference=Reliability; delta=[-0.1,0] bps; daily values are -0.1 and 0.
        chosen=focus if focus in summary.index else summary.index[0]
        wide=predictions.pivot(index='row_id',columns='model',values='abs_error_bps')
        reference='Base' if chosen in ['Base','Quote levels'] else ('Quote levels' if chosen=='Reliability' else 'Reliability')
        if reference not in wide:reference='Base'
        delta=wide[chosen]-wide[reference]
        keys=predictions.drop_duplicates('row_id').set_index('row_id').reindex(delta.index)
        daily=delta.groupby(keys.time.dt.normalize()).mean()
        # PLOTTING LOGIC: Draw the prepared daily paired-loss differences.
        b.axhline(0,color='#777777');b.plot(daily.index,daily,marker='o',color='#277F8E');b.tick_params(axis='x',rotation=25)
        b.set_title(f'{chosen} minus {reference}: daily loss');b.set_ylabel('MAE difference (bps; lower is better)')
        # CORE LOGIC: STEP 7 — Compute fixed overlapping coverage/type/par-size slices.
        # Input: rows 7/8 have has_quote=[1,0], QUANTITY=[100000,1000000], loss deltas=[-0.1,0].
        # Output: Covered/<=100K contains row 7 with delta=-0.1; No quote/>=1MM contains row 8 with delta=0.
        # Trick: Slices overlap; these are descriptive comparisons, not independent sample partitions.
        masks={'No quote':keys.bcq_has_quote.eq(0),'Covered':keys.bcq_has_quote.gt(0),'Paired':keys.bcq_n_pair.gt(0)}
        for value in keys.TRADE_TYPE.dropna().unique():masks[f'Type {value}']=keys.TRADE_TYPE.eq(value)
        masks['<=100K par']=pd.to_numeric(keys.QUANTITY,errors='coerce').le(100_000)
        masks['>=1MM par']=pd.to_numeric(keys.QUANTITY,errors='coerce').ge(1_000_000)
        stats=[(label,delta[mask].mean(),int(mask.sum())) for label,mask in masks.items() if mask.any()]
        # PLOTTING LOGIC: Draw the prepared slice means with sample counts.
        c.barh([f'{k} (n={n:,})' for k,v,n in stats],[v for k,v,n in stats],color='#277F8E');c.axvline(0,color='#777777');c.set_xlabel('MAE difference (bps)');c.set_title('Same-row subgroup differences; groups overlap')
        # CORE LOGIC: STEP 8 — Compute model tail errors over the unchanged target population.
        # Input: model absolute errors=[1,3] bps for two validation targets.
        # Output: P95=2.9 bps and fraction above 10bps=0.
        tails=predictions.groupby('model').abs_error_bps.agg(p95=lambda v:v.quantile(.95),gt10=lambda v:v.gt(10).mean())
        # PLOTTING LOGIC: Draw tail bars, legends and the experiment footnotes.
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
# Completed features and predictions are saved automatically to independent snapshots.
# Export features retries saving; old exports and completed snapshots are never overwritten.
# Validation reports MAE, day-level differences, coverage/side/size slices and tail errors together.
# Test refits the locked choice and its comparison baselines on pre-test data. Re-running controls never tunes on test.
# Progress shows completed bonds/models and current training iterations; elapsed time updates every second.
# Preprocessing has no reliable percentage. Keep the existing run; progress cannot attach to an older running cell.

# %% 3. Research controls
# CORE LOGIC: STEP 6 — Protect an active kernel run from control reconstruction.
# Input: step5_busy=True during validation, or step5_research_busy=True during movement building.
# Output: Raise RuntimeError; no reload, new loader or second fit begins.
if (globals().get('step5_busy',False) or globals().get('step5_research_busy',False)
        or getattr(globals().get('step5_research'), 'busy', False)):
    raise RuntimeError('Step5 is running. Wait for completion; rebuilding controls cannot attach to active work.')
# A hot update reads only the revised functions; no loader, features or fits run.
# SETUP LOGIC: Hot-load revised function definitions without executing the loader or fitting models.
import importlib
import quote_quality_core as step5_core
import quote_quality_saved as step5_saved
importlib.reload(step5_core);importlib.reload(step5_saved)
run_comparison=step5_core.run_comparison
model_versions=step5_core.model_versions
if 'VALIDATION_VERSIONS' not in globals():
    VALIDATION_VERSIONS=('Base','Quote levels','Reliability','Age decay')
# UI LOGIC: Reuse prior display choice and detach old handlers before constructing replacement controls.
step5_previous_choice=globals().get('step5_selected')
step5_previous_choice=getattr(step5_previous_choice,'value',None)
if 'step5_dashboard' in globals():
    if 'step5_clock_stop' in globals():step5_clock_stop.set()
    for button,callback in step5_callbacks:button.on_click(callback,remove=True)
    step5_selected.unobserve(show_step5_choice,names='value')
    step5_dashboard.close()
step5_issuer=widgets.Dropdown(options=representative_issuers(bcq_df),description='Preview:',layout=widgets.Layout(width='850px'))
step5_preview=widgets.Button(description='Preview issuer')
step5_build=widgets.Button(description='Build all features')
step5_validate=widgets.Button(description=f'Run validation ({len(VALIDATION_VERSIONS)})',button_style='primary')
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
# Rebuilding the UI must retain completed work and the locked test choice.
# CORE LOGIC: STEP 7 — Preserve completed experiment objects and any test choice lock.
# Input: globals already contain a 1000-row step5_frame and step5_locked='Age decay'.
# Output: Those objects/lock remain unchanged; only absent state keys get their explicit defaults.
# Trick: The initial busy guard has already stopped this reconstruction if a run is active.
for name,default in [('step5_frame',None),('step5_predictions',None),
                     ('step5_test_predictions',None),('step5_locked',None),
                     ('step5_event_cache',None),('step5_last_checkpoint',None),
                     ('step5_result_metadata',None),('step5_saved_lock_guard',False)]:
    if name not in globals():globals()[name]=default
step5_busy=False
# CORE LOGIC: STEP 8 — Restore persisted lock metadata without opening test predictions.
# Input: fresh kernel has no frame/predictions/lock; saved metadata locked_choice='Age decay', test_available=True.
# Output: step5_locked=Age decay, saved_lock_guard=True and checkpoint path is retained; no Parquet/test metrics load.
if step5_frame is None and step5_predictions is None and step5_locked is None:
    try:
        saved_metadata,saved_path=step5_saved.load_step5_metadata()
        if saved_metadata.get('locked_choice') is not None or saved_metadata.get('test_available'):
            step5_locked=saved_metadata.get('locked_choice')
            step5_saved_lock_guard=True
            step5_last_checkpoint=saved_path
    # CACHEING LOGIC: Check only whether an existing pointer/test file indicates a broken saved experiment.
    except FileNotFoundError:
        saved_folder=Path('outputs/quote_quality_step5')
        # CORE LOGIC: STEP 9 — Treat a broken existing save as unknown test exposure, not a fresh unopened test.
        # Input: latest.json exists but experiment.json is missing.
        # Output: saved_lock_guard=True; UI cannot start another fit/test.
        if (saved_folder/'latest.json').is_file() or (saved_folder/'test_predictions.parquet').is_file():
            step5_saved_lock_guard=True  # broken existing save is not evidence of an unopened test
    # CORE LOGIC: STEP 10 — Block fitting if the saved lock record cannot be verified.
    # Input: experiment.json checksum is invalid or reading it raises OSError.
    # Output: saved_lock_guard=True and the verification error is displayed; no model starts.
    except (ValueError,OSError) as error:
        step5_saved_lock_guard=True
        step5_status.value=escape(f'Saved experiment cannot be verified: {error}. Review it before fitting.')


# SETUP LOGIC: Define step5_experiment_metadata; calling is explicit, not triggered by this declaration.
def step5_experiment_metadata(include_model_columns=True):
    # File stats and exact settings identify this run without hashing 39M raw quotes again.
    # CORE LOGIC: STEP 1 — Record the feature specification of the actual existing frame.
    # Input: include_model_columns=False and step5_frame=None; this is a metadata-only context check.
    # Output: specs={}; model_versions is not called and the returned model_columns map is empty.
    specs=model_versions(step5_frame)[1] if include_model_columns else {}
    # CACHEING LOGIC: Record file size/mtime provenance without rehashing the 39M raw quote rows.
    files={}
    for path in [DATA_IG_CACHE,RAW_QUOTES_FILE]:
        if path.is_file():
            stat=path.stat()
            files[str(path)]=dict(size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    # CORE LOGIC: STEP 2 — Record the original prediction task, budget and date settings explicitly.
    # Input: target=D_BM_SPREAD, anchor=PREV_BM_SPREAD, multiplier=100, trees=400, age/sync=30/1, val/test/embargo=5/5/2.
    # Output: Returned metadata retains those exact values, four version names and the existing event cache key.
    return dict(base_features=BASE_FEATURES,target=TARGET_COL,anchor=ANCHOR_COL,
        error_multiplier=100,quote_spread_unit='bps',allow_exact_quotes=ALLOW_EXACT_QUOTES,
        age_min=AGE_MIN,sync_min=SYNC_MIN,lgb_params=LGB_PARAMS,
        val_days=VALIDATION_DAYS,test_days=TEST_DAYS,embargo_days=EMBARGO_DAYS,
        validation_versions=list(VALIDATION_VERSIONS),
        model_columns={name:columns for name,(columns,_) in specs.items()},
        quote_file_dates=[bcq_df.quote_timestamp_ET.min().isoformat(),bcq_df.quote_timestamp_ET.max().isoformat()],
        input_files=files,event_cache_key=step5_event_cache.get('cache_key') if step5_event_cache is not None else None)


# SETUP LOGIC: Define checkpoint_step5; calling is explicit, not triggered by this declaration.
def checkpoint_step5():
    # SETUP LOGIC: Update only the explicit checkpoint/metadata state names.
    global step5_last_checkpoint,step5_result_metadata
    # CORE LOGIC: STEP 1 — Require a completed full feature frame before any export.
    # Input: step5_frame=None while only a 467-row issuer preview exists.
    # Output: ValueError; an issuer preview cannot be exported as the full experiment.
    if step5_frame is None:raise ValueError('No completed full feature frame to save')
    # SETUP LOGIC: Use the immutable snapshot writer.
    from quote_quality_saved import save_step5_checkpoint
    # CORE LOGIC: STEP 2 — Capture experiment settings once and preserve their actual provenance.
    # Input: step5_frame has 50 rows, lgb_params.n_estimators=400, predictions=None, result_metadata=None; later models are Base/Quote levels/Reliability/Age decay.
    # Output: step5_result_metadata retains n_estimators=400, settings_origin=captured_before_fitting; later metadata.validation_versions is Base/Quote levels/Reliability/Age decay.
    # Trick: Capturing after existing predictions instead marks legacy_live_settings_at_capture, which cannot prove the old fitting budget.
    if step5_result_metadata is None:
        from copy import deepcopy
        step5_result_metadata=deepcopy(step5_experiment_metadata())
        step5_result_metadata['settings_origin']='legacy_live_settings_at_capture' if step5_predictions is not None else 'captured_before_fitting'
    metadata=dict(step5_result_metadata,locked_choice=step5_locked)
    if step5_predictions is not None:
        metadata['validation_versions']=step5_predictions.model.drop_duplicates().tolist()
    # CACHEING LOGIC: Save features/available predictions and persisted choice to a new snapshot; propagate failure.
    step5_detail.value='Saving completed results; previous snapshots remain available...'
    step5_last_checkpoint=save_step5_checkpoint(step5_frame,step5_predictions,
        step5_test_predictions,metadata=metadata,folder=Path('outputs/quote_quality_step5'))
    step5_detail.value=escape(f'Saved completed results to {step5_last_checkpoint}')
    return step5_last_checkpoint


# SETUP LOGIC: Define step5_matches_saved_context; calling is explicit, not triggered by this declaration.
def step5_matches_saved_context():
    # Existing results remain viewable, but changed inputs/settings cannot silently
    # drive another fit or held-out test in their experiment.
    # CORE LOGIC: STEP 1 — Require the exact original target/base/split frame before another fit or test.
    # Input: saved row 7 COUPON=5, time=03-19 10:00 ET; current row 7 COUPON=6.
    # Output: Return False; unchanged rows/base values pass the frame comparison.
    # Trick: The legacy metadata=None branch preserves compatibility, not verified provenance for new incremental fitting.
    columns=['row_id','cusip','time',TARGET_COL,ANCHOR_COL,'BM_SPREAD','split','refit_train']+BASE_FEATURES
    columns=list(dict.fromkeys(columns))
    if not step5_frame[columns].reset_index(drop=True).equals(model_data[columns].reset_index(drop=True)):return False
    if step5_result_metadata is None:return True  # legacy live results; do not invent old run metadata
    # CORE LOGIC: STEP 2 — Compare actual parameter, source and as-of settings to their saved values.
    # Input: saved trees=400, age=30, exact=True; current trees=401 with all other fields unchanged.
    # Output: Return False; identical settings return True.
    current=step5_experiment_metadata(include_model_columns=False)
    fields=['target','anchor','base_features','error_multiplier','allow_exact_quotes',
            'age_min','sync_min','lgb_params','val_days','test_days','embargo_days',
            'validation_versions','quote_file_dates','input_files']
    return all(step5_result_metadata.get(field)==current.get(field) for field in fields)


# UI LOGIC: Define report_step5; calling is explicit, not triggered by this declaration.
def report_step5(stage,completed=None,total=None,detail=''):
    # Throttle UI messages, not the calculation or counts. Always show model transitions and completion.
    # UI LOGIC: Throttle display changes to 0.2s, while reporting every model transition and final count.
    now=perf_counter()
    final=total is not None and completed==total
    transition=(stage=='features' and step5_progress_state.get('phase')!='features') or (stage=='fit' and step5_rounds.layout.display=='none')
    # Trick: model/iteration counts are completed work; event preparation has no estimated completion percentage.
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


# UI LOGIC: Define step5_clock; calling is explicit, not triggered by this declaration.
def step5_clock(stop,started,widget):
    # This timer only updates display text; no training or data work runs in the thread.
    # UI LOGIC: Update elapsed text each second; this thread does no feature or model work.
    while not stop.wait(1):
        seconds=int(perf_counter()-started)
        widget.value=f'Elapsed: {seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'


# SETUP LOGIC: Define run_step5; calling is explicit, not triggered by this declaration.
def run_step5(action='preview'):
    # SETUP LOGIC: Use the existing kernel experiment objects; no reset is implied.
    global step5_frame,step5_predictions,step5_test_predictions,step5_locked,step5_figure,step5_busy,step5_clock_stop,step5_event_cache
    # CORE LOGIC: STEP 1 — Reject an overlapping action or a saved locked/unverifiable experiment.
    # Input: action=test, step5_busy=False, step5_saved_lock_guard=True; or research_busy=True.
    # Output: Return after the guard message; no feature construction or fit/test starts.
    if (step5_busy or globals().get('step5_research_busy',False)
            or getattr(globals().get('step5_research'), 'busy', False)):return
    if action!='preview' and step5_saved_lock_guard:
        step5_status.value='Saved experiment has a locked test or cannot be verified. Review saved validation with the helper; no new fit/test runs here.'
        return
    # CORE LOGIC: STEP 2 — Reuse completed validation and reject changed experiment inputs.
    # Input: action=validate with 100454 saved validation target rows; changed model parameters would also fail the context check.
    # Output: Existing validation redraws with zero new fits; changed inputs/settings block further fit/test.
    if action=='validate' and step5_predictions is not None:
        step5_status.value='Completed validation remains in memory. Use Choice or the SECTOR helper to review it.'
        show_step5_choice();return
    if action in ['validate','test'] and step5_frame is not None and not step5_matches_saved_context():
        step5_status.value='Existing results retained for review. Inputs or settings changed; do not fit/test under this experiment.'
        return
    # UI LOGIC: Disable conflicting controls, clear the old image and start the elapsed display.
    step5_busy=True
    for button in [step5_preview,step5_build,step5_validate,step5_test,step5_export,step5_save]:button.disabled=True
    step5_selected.disabled=True;step5_issuer.disabled=True
    step5_image.value=b''
    step5_progress.value=0;step5_progress.max=1;step5_progress.bar_style='info'
    step5_rounds.layout.display='none';step5_rounds.bar_style='info'
    step5_progress_state.update(last_update=0.,context='',phase='')
    step5_clock_stop=Event();started=perf_counter();step5_elapsed.value='Elapsed: 00:00:00'
    clock=Thread(target=step5_clock,args=(step5_clock_stop,started,step5_elapsed),daemon=True);clock.start()
    # UI LOGIC: Show the actual preparation stage; no speculative completion percentage is assigned.
    try:
        step5_status.value='Running. Counts show completed work; bond sizes and model times vary.'
        report_step5('events',detail='Preparing input rows')
        # CORE LOGIC: STEP 3 — Build only the requested issuer preview without replacing full experiment results.
        # Input: selected issuer=HPS with 467 eligible rows; step5_frame already contains the full experiment.
        # Output: q/shown contain those 467 rows; original step5_frame and predictions remain unchanged.
        if action=='preview':
            sample=model_data.loc[model_data.ISSUER.eq(step5_issuer.value)]
            if sample.empty:raise ValueError('No eligible target rows for this issuer through the quote end date. Choose another issuer.')
            q=sample[['row_id','cusip','time']]
            features=build_quote_features(bcq_df.loc[bcq_df.ISSUER.eq(step5_issuer.value)],q,AGE_MIN,SYNC_MIN,ALLOW_EXACT_QUOTES,progress=report_step5)
            shown=sample.merge(features.drop(columns=['cusip','time']),on='row_id',validate='one_to_one')
            # PLOTTING LOGIC: Render the issuer readiness preview.
            step5_figure=validation_figure(shown,title=step5_issuer.value)
        # UI LOGIC: Route non-preview work to the retained full experiment.
        else:
            if step5_frame is None:
                # CACHEING LOGIC: Prepare quote events only when no retained event cache exists.
                if step5_event_cache is None:
                    step5_event_cache=prepare_quote_events(bcq_df,progress=report_step5)
                # CORE LOGIC: STEP 4 — Build full quote features once and join each target by row_id.
                # Input: model_data rows 7/8 share time=10:00 ET but have different target ids; row 8 has no quote.
                # Output: step5_frame retains both rows; no-quote row 8 gets zero counts/NaN levels, not deletion.
                features=build_quote_features(bcq_df,model_data[['row_id','cusip','time']],AGE_MIN,SYNC_MIN,ALLOW_EXACT_QUOTES,progress=report_step5,event_cache=step5_event_cache)
                step5_frame=model_data.merge(features.drop(columns=['cusip','time']),on='row_id',validate='one_to_one')
                # CACHEING LOGIC: Persist the completed full frame before fitting.
                checkpoint_step5()
            # UI LOGIC: Report reuse of the existing full frame instead of rebuilding it.
            else:
                n_bonds=step5_frame.cusip.nunique()
                report_step5('features',n_bonds,n_bonds,'Reusing completed in-memory feature frame')
            # CORE LOGIC: STEP 5 — Train the predeclared four models on the original Train population.
            # Input: action=validate, choice not locked, versions=[Base,Quote levels,Reliability,Age decay], original budget=400 trees.
            # Output: run_comparison returns four Validation models on identical target ids; a locked choice rejects.
            if action=='validate':
                if step5_locked is not None:raise ValueError('Test choice is already locked. Start a new experiment explicitly before changing it.')
                step5_predictions,_=run_comparison(step5_frame,LGB_PARAMS,progress=report_step5,versions=VALIDATION_VERSIONS)
                # CACHEING LOGIC: Persist completed validation predictions without replacing older snapshots.
                checkpoint_step5()
                # CORE LOGIC: STEP 6 — Order the completed validation choices by same-population MAE.
                # Input: MAE Base=2.039, Quote levels=2.020, Reliability=2.013, Age decay=2.009 bps.
                # Output: Displayed order starts Age decay, Reliability, Quote levels, Base; the default choice is not yet a test lock.
                order=step5_predictions.groupby('model').abs_error_bps.mean().sort_values().index.tolist()
                step5_selected.options=order;step5_selected.value=order[0]
                # PLOTTING LOGIC: Render the retained validation choice.
                step5_figure=validation_figure(step5_frame,step5_predictions,step5_selected.value,'Validation')
            # CORE LOGIC: STEP 7 — Require validation and lock the selected choice before the normal test path.
            # Input: step5_predictions is present, Choice=Age decay, step5_locked=None.
            # Output: step5_locked becomes Age decay and checkpoint is called before the first held-out test fit.
            # Trick: If this save fails, the in-memory lock remains set; this existing error path is not changed by these comments.
            elif action=='test':
                if step5_predictions is None:raise ValueError('Run validation before opening the final test')
                if step5_locked is None:
                    step5_locked=step5_selected.value
                    checkpoint_step5()  # persist the choice before evaluating the held-out test
                # UI LOGIC: Keep the selection control disabled after a test choice is locked.
                step5_selected.disabled=True
                # CORE LOGIC: STEP 8 — Evaluate the locked test only if completed test predictions are absent.
                # Input: locked choice=Age decay, completed test predictions=None, pre-test refit mask excludes its last two trade dates.
                # Output: Fit Base, Reliability and Age decay on original refit rows and predict the same Test rows; existing test predictions skip fitting.
                if step5_test_predictions is None:
                    step5_test_predictions,_=run_comparison(step5_frame,LGB_PARAMS,'Test',step5_locked,progress=report_step5)
                    # CACHEING LOGIC: Persist completed locked-test predictions with the unchanged choice.
                    checkpoint_step5()
                # PLOTTING LOGIC: Draw already computed locked-test results; remaining rendering/error text is display-only.
                step5_figure=validation_figure(step5_frame,step5_test_predictions,step5_locked,'Locked test')
            else:step5_figure=validation_figure(step5_frame)
        step5_detail.value='Rendering the four-panel dashboard...'
        with BytesIO() as buffer:
            step5_figure.savefig(buffer,format='png',dpi=110);step5_image.value=buffer.getvalue()
        step5_save.disabled=False
        step5_progress.bar_style='success';step5_rounds.bar_style='success'
        step5_detail.value=escape(f'Complete. Full experiment saved to {step5_last_checkpoint}') if action!='preview' else 'Issuer preview only; full experiment results are unchanged.'
        step5_status.value='Finished. All four panels are ready to share; no large tables printed.'
    except KeyboardInterrupt:
        step5_status.value='Interrupted. Any fully built feature frame remains in this kernel; partial model results are not saved.'
        step5_progress.bar_style='warning';step5_rounds.bar_style='warning'
    except Exception as error:
        step5_status.value=escape(f'{type(error).__name__}: {error}. Completed in-memory results are retained; Export features retries saving.')
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
        # CORE LOGIC: STEP 9 — Preserve finished/locked state in the controls after an action ends.
        # Input: completed validation exists, step5_locked=Age decay, saved_lock_guard=False.
        # Output: Choice and Run validation are disabled; the existing test predictions are reused on a later test click.
        step5_selected.disabled=step5_predictions is None or step5_locked is not None
        step5_validate.disabled=step5_saved_lock_guard or step5_locked is not None or step5_predictions is not None
        step5_test.disabled=step5_saved_lock_guard or step5_predictions is None


# UI LOGIC: Define export_step5; calling is explicit, not triggered by this declaration.
def export_step5(_=None):
    # UI LOGIC: Export is an explicit save retry of completed objects, never a model rerun.
    if step5_frame is None:
        step5_status.value='Build all features before exporting.';return
    try:
        folder=checkpoint_step5()
        step5_status.value=escape(f'Saved features, experiment settings and available predictions to {folder}')
    except Exception as error:
        step5_status.value=escape(f'Save failed: {error}. Results remain in memory; previous snapshots are unchanged.')


# PLOTTING LOGIC: Define save_step5; calling is explicit, not triggered by this declaration.
def save_step5(_=None):
    # PLOTTING LOGIC: Write the complete current figure at 180 dpi without changing feature/prediction files.
    folder=Path('outputs/quote_quality_step5');folder.mkdir(parents=True,exist_ok=True)
    step5_figure.savefig(folder/'dashboard.png',dpi=180);step5_status.value=f'Saved {folder}/dashboard.png'

# PLOTTING LOGIC: Define show_step5_choice; calling is explicit, not triggered by this declaration.
def show_step5_choice(change=None):
    global step5_figure
    # CORE LOGIC: STEP 1 — Permit display-only validation choice changes only before locking.
    # Input: step5_busy=False, completed Validation exists, step5_locked=Age decay.
    # Output: Return without changing the locked choice; unlocked completed Validation can be redrawn.
    if step5_busy or step5_predictions is None or step5_locked is not None:return
    # PLOTTING LOGIC: Redraw the completed selected validation; no feature building or fit occurs.
    step5_figure=validation_figure(step5_frame,step5_predictions,step5_selected.value,'Validation')
    with BytesIO() as buffer:
        step5_figure.savefig(buffer,format='png',dpi=110);step5_image.value=buffer.getvalue()
    step5_save.disabled=False


# UI LOGIC: Bind the existing explicit actions once and display the compact dashboard.
step5_callbacks=[(step5_preview,lambda _:run_step5('preview')),(step5_build,lambda _:run_step5('build')),
    (step5_validate,lambda _:run_step5('validate')),(step5_test,lambda _:run_step5('test')),
    (step5_save,save_step5),(step5_export,export_step5)]
for button,callback in step5_callbacks:button.on_click(callback)
step5_selected.observe(show_step5_choice,names='value')
step5_dashboard=widgets.VBox([step5_issuer,widgets.HBox([step5_preview,step5_build,step5_validate]),widgets.HBox([step5_selected,step5_test,step5_export,step5_save]),step5_status,step5_progress,step5_rounds,step5_detail,step5_elapsed,step5_image])
display(step5_dashboard)
# CORE LOGIC: STEP 11 — Restore a completed validation choice from retained kernel state.
# Input: completed MAEs rank Age decay first; prior Choice=Reliability, no lock.
# Output: Choice remains Reliability if present; a saved Age decay lock takes precedence.
if step5_predictions is not None:
    order=step5_predictions.groupby('model').abs_error_bps.mean().sort_values().index.tolist()
    step5_selected.options=order
    choice=step5_locked or step5_previous_choice
    step5_selected.value=choice if choice in order else order[0]
    # PLOTTING LOGIC: Render the restored validation; no new model is trained.
    step5_figure=validation_figure(step5_frame,step5_predictions,step5_selected.value,'Validation')
    step5_status.value='Completed validation retained. No feature build or training was repeated.'
# UI LOGIC: Display the retained full frame and update export/display availability.
elif step5_frame is not None:
    step5_figure=validation_figure(step5_frame)
    step5_status.value='Completed full feature frame retained. No feature build was repeated.'
else:
    run_step5('preview')
if step5_frame is not None:
    with BytesIO() as buffer:
        step5_figure.savefig(buffer,format='png',dpi=110);step5_image.value=buffer.getvalue()
    step5_save.disabled=False
    step5_export.disabled=False
    # CORE LOGIC: STEP 12 — Respect completion/lock guards when initializing the rebuilt controls.
    # Input: step5_predictions exists and step5_saved_lock_guard=True.
    # Output: Run validation and Run locked test stay disabled; no reset clears the persisted exposure state.
    step5_selected.disabled=step5_predictions is None or step5_locked is not None
    step5_validate.disabled=step5_saved_lock_guard or step5_predictions is not None or step5_locked is not None
    step5_test.disabled=step5_saved_lock_guard or step5_predictions is None
# CORE LOGIC: STEP 13 — Enforce the saved experiment guard across build, validation and test actions.
# Input: new kernel recovered locked_choice=Age decay from verified metadata.
# Output: Build, Run validation and Run locked test are disabled; only existing saved validation review is permitted.
if step5_saved_lock_guard:
    step5_build.disabled=True;step5_validate.disabled=True;step5_test.disabled=True
    step5_status.value='Saved experiment is locked or unverifiable. Use the saved-validation helper; restarting the kernel does not unlock the test.'


# %% [markdown]
# ## Continue completed validation — run only this cell
#
# Keep the existing idle Step5 kernel. Run the cell below directly; no code needs to
# be pasted or added. It reuses completed results and opens the finite research panel.
# It never loads raw quotes, runs the original validation again, or opens locked test.
#
# Start with **Age decay → Review losses**, then **Review rule effects → Freeze cases**.
# Next build movement features and inspect support before either one-model Fit.
# Repeated execution keeps the existing research session, including added predictions.
# The last cell reloads the lightweight saved-review/case helpers to apply fixes in place.
# A fresh kernel can review saved validation; movement needs the original event cache
# or a matching saved movement sidecar. Missing/corrupt results show a message.

# %% 4. Continue completed validation research
# CORE LOGIC: STEP 1 — Refuse to rebuild a panel while either experiment is running.
# Input: step5_busy=False, step5_research_busy=True, existing session=R.
# Output: RuntimeError before importing/opening a new panel; R and its results remain intact.
# Trick: check the session object too, so a missing namespace flag cannot hide active work.
step5_existing_research = globals().get('step5_research')
if (globals().get('step5_busy', False) or globals().get('step5_research_busy', False)
        or getattr(step5_existing_research, 'busy', False)):
    raise RuntimeError('Step5 is running. Wait for completion; keep the existing kernel.')
# UI LOGIC: This cell is independent of the loader and original controls cells.
from html import escape as step5_research_escape
from IPython.display import display, HTML
# SETUP LOGIC: Refresh review/case helpers only after the busy guard, without reloading core/cache/model code.
# Trick: Existing research callbacks import these functions on each click; retained session objects need no rebuild.
import importlib as step5_review_importlib
import quote_quality_saved as step5_saved_review
import quote_quality_diagnostics as step5_case_review
step5_review_importlib.reload(step5_saved_review)
step5_review_importlib.reload(step5_case_review)
# UI LOGIC: Opening or redisplaying controls is explicit; imports do not run an analysis.
from quote_quality_research import show_research
# UI LOGIC: Preserve an existing session, including its sidecar and incremental predictions.
try:
    if step5_existing_research is None:
        step5_research = show_research(globals())
    else:
        display(step5_existing_research.dashboard)
except (FileNotFoundError, ValueError, OSError) as error:
    # UI LOGIC: A missing/unverifiable snapshot never starts a replacement fit or raw load.
    display(HTML('<b>Research panel could not open.</b> ' + step5_research_escape(str(error))
        + '<br>Keep the current kernel and completed results. No feature build or fit was started.'))
