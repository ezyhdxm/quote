# %% [markdown]
# # Step 4 — same-dealer bid/ask candidate pairing
# Run three cells. Loading is included; output is one shareable dashboard.

# %% 1. Load data
from pathlib import Path
import numpy as np
import pandas as pd
from io import BytesIO
from textwrap import fill
from time import perf_counter
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import ipywidgets as widgets
from IPython.display import display

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


from quote_quality_core import (KEYS, event_history, pair_snapshots, pair_features,
    pair_policy_masks, pair_policy_comparison, pair_extreme_sources, representative_issuers)
from quote_quality_cache import prepare_quote_events
from quote_quality_population import population_tables, fixed_case_manifest, case_options

# %% [markdown]
# ## Decisions
# Compare A fresh pairs, B size/time eligible slots with original candidates, and C matched candidates on the identical B slots.
# signed gap = bid spread - ask spread. A negative value is crossed; zero is locked.
# Candidate ranges classify none / some / all combinations crossing. They are not probabilities.
# A common positive raw quantity is a matching condition, not proof of execution comparability.

# %% 2. Pairing settings and one six-panel figure
AGE_MIN = 30
SYNC_MIN = 1
GRID = '5min'


def pairing_figure(result, dealer, bond, day, age, sync):
    raw,e,p,f=result['raw'],result['events'],result['pairs'],result['features']
    own=raw.loc[raw.firm.eq(dealer)]
    selected=p.loc[p.firm.eq(dealer)]
    fresh=p.loc[p.fresh_pair]; picked=selected.loc[selected.fresh_pair]
    fig=Figure(figsize=(17,12),facecolor='white')
    axes=fig.subplots(3,2); a,b,c,d,ax5,ax6=axes.flat
    fig.subplots_adjust(top=.85,bottom=.14,hspace=.6,wspace=.28)
    fig.suptitle(f"{result['issuer']} | {bond} | {day:%Y-%m-%d} ET\nSelected dealer {dealer} | age <= {age} min | size-match time gap <= {sync} min",fontsize=16,y=.98)
    fig.text(.5,.91,f"{len(own):,} selected raw rows | {len(p):,} dealer-grid slots (at least one side observed) | {len(fresh):,} fresh complete pairs",ha='center',fontsize=11)
    for side,color,marker in [('bid','#277F8E','o'),('ask','#C9563D','x')]:
        z=own.loc[own.side.eq(side)]
        a.scatter(z.quote_timestamp_ET,z.s,color=color,marker=marker,s=14,alpha=.65,label=side)
    a.set_title('Selected dealer: all raw bid / ask candidates');a.set_ylabel('Benchmark spread (bps)')
    b.axhline(0,color='#777777',lw=.8)
    for state,color in [('None','#277F8E'),('Some','#BA8C27'),('All','#C9563D')]:
        z=picked.loc[picked['cross'].eq(state)]
        b.vlines(z.time,z.gap_low,z.gap_high,color=color,alpha=.65,lw=1.5,label=state+' crossing')
        b.scatter(z.time,z.gap_center,color=color,s=10)
    m=selected.loc[selected.size_time_pair]
    b.scatter(m.time,m.matched_gap,marker='x',s=22,color='#8560A5',label='Matched positive-size gap')
    b.set_title('Selected dealer: candidate gap range');b.set_ylabel('bid - ask (bps)')
    if picked.empty:b.text(.5,.5,'No fresh complete bid/ask pair',transform=b.transAxes,ha='center')
    policies=[('A fresh\noriginal',fresh,'cross'),
              ('B match slots\noriginal',p.loc[p.size_time_pair],'cross'),
              ('C same B slots\nmatched',p.loc[p.size_time_pair],'cross_matched')]
    bottom=np.zeros(3)
    for state,color in [('None','#277F8E'),('Some','#BA8C27'),('All','#C9563D')]:
        values=[z[field].eq(state).mean() if len(z) else 0 for _,z,field in policies]
        c.bar(range(3),values,bottom=bottom,color=color,label=state);bottom+=values
    for i,(_,z,_) in enumerate(policies):c.text(i,1.03,f'n={len(z):,}' if len(z) else 'Unassessed',ha='center',fontsize=9)
    c.set_xticks(range(3),[label for label,_,_ in policies]);c.set_ylim(0,1.18);c.set_ylabel('Fraction of dealer-grid pairs')
    c.set_title('All dealers: A→B selection; B→C matching effect')
    total=p.groupby('time').size().reindex(f.index,fill_value=0)
    complete=p.loc[p.complete].groupby('time').size().reindex(f.index,fill_value=0)
    for values,label,color in [(total,'At least one side','#AAAAAA'),(complete,'Both complete','#555555'),(f.n_pair,'Both fresh','#277F8E'),(f.n_size_time_pair,'Size + short gap','#8560A5')]:
        d.step(f.index,values,where='post',label=label,color=color)
    d.set_title('All dealers: pairing coverage cost');d.set_ylabel('Dealers');d.set_ylim(bottom=0)
    for state,color in [('None','#277F8E'),('Some','#BA8C27'),('All','#C9563D')]:
        z=fresh.loc[fresh['cross'].eq(state)]
        ax5.scatter(z.time_gap,z.gap_center,s=13,alpha=.5,color=color,label=state)
    ax5.axvline(sync,color='#777777',ls='--',lw=.8);ax5.axhline(0,color='#777777',lw=.8)
    extremes=result['extreme_sources']
    for label,field,ascending,offset in [('Low gap boundary','gap_low',True,15),('High gap boundary','gap_high',False,-25)]:
        if fresh.empty:continue
        row=fresh.sort_values(field,ascending=ascending,kind='stable').iloc[0]
        z=extremes.loc[extremes.extreme.eq(label)&extremes.firm.eq(row.firm)&extremes.time.eq(row.time)&extremes.boundary_candidate]
        tags=[]
        for side in ['bid','ask']:
            source=z.loc[z.side.eq(side)].drop_duplicates(['spread','quantity'])
            values=[f'{r.spread:g} / q={r.quantity}' if isinstance(r.spread,(float,int)) else f'{r.spread} / q={r.quantity}' for r in source.itertuples()]
            tags.append(side+' '+', '.join(values[:2])+(' …' if len(values)>2 else ''))
        ax5.scatter([row.time_gap],[row[field]],marker='^' if ascending else 'v',color='#202020',s=45,zorder=5)
        ax5.annotate(f'{row.firm} {row[field]:.1f} bps\n'+ '\n'.join(tags),(row.time_gap,row[field]),
                     xytext=(5,offset),textcoords='offset points',fontsize=7,va='bottom' if ascending else 'top')
    ax5.set_title('All dealers: ± gap boundaries and raw source quantities');ax5.set_xlabel('Absolute bid/ask timestamp gap (min)');ax5.set_ylabel('Center / boundary gap (bps)')
    b_mid=p.loc[p.size_time_pair].groupby('time').mid.mean().reindex(f.index)
    for values,label,color in [(f.pair_mid,'A fresh / original','#277F8E'),
                              (b_mid,'B match slots / original','#BA8C27'),
                              (f.size_time_mid,'C same slots / matched','#8560A5')]:
        ax6.plot(f.index,values,color=color,label=label,drawstyle='steps-post')
    ax6.set_title('All dealers: midpoint change on identical B/C slots');ax6.set_ylabel('Spread midpoint description (bps)')
    for ax in [a,b,d,ax6]:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=6,tz=day.tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=day.tz));ax.set_xlabel('ET')
        if len(f):ax.set_xlim(f.index.min()-pd.Timedelta(minutes=1),f.index.max()+pd.Timedelta(minutes=1))
    for ax in axes.flat:
        ax.grid(alpha=.15);h,l=ax.get_legend_handles_labels()
        if h:ax.legend(fontsize=8,ncol=2,loc='best')
    fig.text(.08,.085,'Decision: keep single-side features everywhere; add paired descriptors only with coverage, time-gap and candidate/size ambiguity.',fontsize=11)
    summary=result['policy_comparison']
    selection=summary.iloc[1].gap_mean_bps-summary.iloc[0].gap_mean_bps
    matching=summary.iloc[2].gap_mean_bps-summary.iloc[1].gap_mean_bps
    mid_matching=summary.iloc[2].mid_mean_bps-summary.iloc[1].mid_mean_bps
    fig.text(.08,.05,f'Slot means: gap A→B {selection:+.2f} bps (selection); B→C {matching:+.2f} bps and midpoint {mid_matching:+.2f} bps (same slots). Raw culprit rows are in step4_result.',fontsize=10)
    fig.text(.08,.025,'Zero / negative spreads and zero quantities are retained. Positive raw-size matching does not establish economic comparability or an executable market.',fontsize=10,color='#555555')
    return fig

# %% [markdown]
# ## Use and share
# The issuer labels mean previously reviewed cases, not clean/dirty issuers.
# Cases with observed bid and ask on the same dealer/bond/day are promoted first; all choices remain available.
# `step4_result['pairs']` retains missing sides, incomplete pairs, raw-size matches and crossing classes.
# Apply changes explicitly. Dealer changes only redraw cached all-dealer bond/day state.
# Freeze 18 cases explicitly to run three fixed local cleaning probes; default opening does not run that population scan.
# `policy_comparison` separates slot selection from matching; `extreme_sources` keeps original bid/ask quantity provenance.
# Save PNG exports all six panels. No DataFrame is printed.

# %% 3. Case controls
if 'step4_controls' in globals():
    for box in step4_controls:box.unobserve(refresh_step4,names='value')
    step4_case.unobserve(choose_step4_case,names='value')
    step4_freeze.on_click(freeze_step4_cases,remove=True)
    step4_apply.on_click(apply_step4,remove=True)
    step4_save.on_click(save_step4,remove=True)
    step4_dashboard.close()
# Rerunning this cell reuses explicitly scoped caches only for the same loaded source.
if globals().get('step4_cache_source') is not bcq_df:
    step4_issuer_cache={};step4_pair_cache={};step4_cache_source=bcq_df
    step4_population=None;step4_manifest=None;step4_quote_source=bcq_df
step4_navigation=step4_quote_source[['ISSUER']+KEYS].dropna().drop_duplicates().copy()
step4_navigation['day']=step4_navigation.quote_timestamp_ET.dt.normalize()
step4_issuer=widgets.Dropdown(options=representative_issuers(step4_quote_source),description='Issuer:',layout=widgets.Layout(width='850px'))
step4_bond=widgets.Dropdown(description='Bond:');step4_dealer=widgets.Dropdown(description='Dealer:');step4_day=widgets.Dropdown(description='ET day:')
step4_age=widgets.Dropdown(options=[10,30,60],value=AGE_MIN,description='Age min:')
step4_sync=widgets.Dropdown(options=[0,1,5],value=SYNC_MIN,description='Gap min:')
step4_apply=widgets.Button(description='Apply case',icon='check',button_style='primary')
step4_freeze=widgets.Button(description='Freeze 18 cases',icon='snowflake-o')
step4_case=widgets.Dropdown(options=[('Cases not frozen',None)] if step4_manifest is None else case_options(step4_manifest),
    description='Fixed case:',disabled=step4_manifest is None,layout=widgets.Layout(width='1050px'))
step4_save=widgets.Button(description='Save all PNG',icon='download');step4_status=widgets.HTML()
step4_image=widgets.Image(format='png',layout=widgets.Layout(width='100%',max_width='1600px'))
step4_result=None;step4_busy=False;step4_applied=None


def paint_step4():
    global step4_figure
    if step4_result is None:return
    step4_figure=pairing_figure(step4_result,step4_dealer.value,step4_result['bond'],
        step4_result['day'],step4_result['age'],step4_result['sync'])
    with BytesIO() as buffer:
        step4_figure.savefig(buffer,format='png',dpi=110);step4_image.value=buffer.getvalue()


def refresh_step4(change=None):
    global step4_busy
    if step4_busy:return
    owner=change['owner'] if change else None
    if owner is step4_dealer and step4_result is not None and step4_applied[:3]==(step4_issuer.value,step4_bond.value,step4_day.value):
        paint_step4()
        step4_status.value='Dealer redrawn from cached all-dealer bond/day state. Apply pending age/time changes separately.'
        return
    if owner in [step4_age,step4_sync]:
        step4_image.value=b'';step4_save.disabled=True
        step4_status.value='Settings pending. Apply case derives new age/time masks from cached state.'
        return
    step4_busy=True
    try:
        eligible=step4_navigation.loc[step4_navigation.ISSUER.eq(step4_issuer.value)]
        reset=owner is step4_issuer
        # Dealer never restricts day options or the all-dealer bond/day cache.
        for box,col in [(step4_bond,'cusip'),(step4_day,'day'),(step4_dealer,'firm')]:
            cases=eligible.groupby(['cusip','firm','day'],observed=True).side.nunique().rename('sides').reset_index()
            paired=cases.loc[cases.sides.eq(2)].groupby(col,observed=True).size()
            counts=eligible.groupby(col,observed=True).size()
            order=pd.DataFrame({'paired_cases':paired,'events':counts}).fillna(0).sort_values(['paired_cases','events'],ascending=False,kind='stable')
            old=box.value;options=order.index.tolist()
            box.options=[(f'{v:%Y-%m-%d}' if col=='day' else str(v),v) for v in options]
            box.value=old if old in options and not reset else (options[0] if options else None)
            reset |= owner is box
            eligible=eligible.loc[eligible[col].eq(box.value)]
        step4_image.value=b'';step4_save.disabled=True
        step4_status.value='Case pending. Apply case builds each issuer and all-dealer bond/day state once.'
    finally:step4_busy=False
    if change is None:apply_step4()


def apply_step4(_=None):
    global step4_result,step4_busy,step4_applied,step4_figure
    if step4_busy:return
    step4_busy=True;started=perf_counter();step4_save.disabled=True;step4_image.value=b''
    try:
        key=(step4_issuer.value,step4_bond.value,step4_day.value,GRID)
        if any(v is None for v in key[:3]):
            step4_result=None;step4_save.disabled=True
            step4_figure=Figure(figsize=(12,5));ax=step4_figure.subplots();ax.axis('off');ax.text(.5,.5,'No keyed events',ha='center')
            with BytesIO() as buffer:
                step4_figure.savefig(buffer,format='png',dpi=110);step4_image.value=buffer.getvalue()
            step4_status.value='No keyed events in this case.'
            return
        event_reused=key[0] in step4_issuer_cache
        if not event_reused:
            step4_status.value='Building issuer events...'
            step4_issuer_cache[key[0]]=event_history(step4_quote_source.loc[step4_quote_source.ISSUER.eq(key[0])])
        history=step4_issuer_cache[key[0]]
        events_done=perf_counter()
        pair_reused=key in step4_pair_cache
        if not pair_reused:
            step4_status.value='Building all-dealer bond/day state...'
            all_events=history['events'];e=all_events.loc[all_events.cusip.eq(key[1])&all_events.day.eq(key[2])]
            start,end=e.quote_timestamp_ET.min(),e.quote_timestamp_ET.max()
            times=pd.date_range(start.ceil(GRID),end.floor(GRID),freq=GRID).union(pd.DatetimeIndex([start,end]))
            r=history['raw'];raw=r.loc[r.cusip.eq(key[1])&r.quote_timestamp_ET.dt.normalize().eq(key[2])]
            step4_pair_cache[key]=dict(raw=raw,events=e,times=times,pairs=pair_snapshots(e,times))
        cached=step4_pair_cache[key]
        state_done=perf_counter()
        p=pair_policy_masks(cached['pairs'],step4_age.value,step4_sync.value)
        f=pair_features(p,cached['times'])
        step4_result=dict(issuer=key[0],bond=key[1],day=key[2],age=step4_age.value,sync=step4_sync.value,
            raw=cached['raw'],events=cached['events'],pairs=p,features=f,
            policy_comparison=pair_policy_comparison(p),extreme_sources=pair_extreme_sources(cached['raw'],p,1))
        step4_applied=(key[0],key[1],key[2],step4_age.value,step4_sync.value)
        calculated=perf_counter();paint_step4();step4_save.disabled=False
        step4_result['timings']=dict(calculate_s=calculated-started,render_s=perf_counter()-calculated,
            event_s=events_done-started,pair_state_s=state_done-events_done,masks_features_s=calculated-state_done,
            event_cache_reused=event_reused,pair_cache_reused=pair_reused)
        step4_status.value=(f'All-dealer bond/day slots. Events {events_done-started:.2f}s; state {state_done-events_done:.2f}s; masks/features {calculated-state_done:.2f}s; render {perf_counter()-calculated:.2f}s. '
            f'Event cache {"reused" if event_reused else "built"}; state cache {"reused" if pair_reused else "built"}. '
            'A→B selects slots; B→C matches quantity on the same slots. Raw ± sources are in step4_result.')
    finally:step4_busy=False


def choose_step4_case(change=None):
    global step4_busy
    if step4_busy or step4_manifest is None or step4_case.value is None:return
    row=step4_manifest.loc[step4_manifest.case_id.eq(step4_case.value)].iloc[0]
    step4_busy=True
    try:
        step4_issuer.value=row.ISSUER
        eligible=step4_navigation.loc[step4_navigation.ISSUER.eq(row.ISSUER)]
        for box,col,value in [(step4_bond,'cusip',row.cusip),(step4_day,'day',row.day),(step4_dealer,'firm',row.firm)]:
            options=eligible.groupby(col,observed=True).size().sort_values(ascending=False,kind='stable').index.tolist()
            box.options=[(f'{v:%Y-%m-%d}' if col=='day' else str(v),v) for v in options]
            box.value=value
            eligible=eligible.loc[eligible[col].eq(value)]
    finally:step4_busy=False
    apply_step4()
    step4_result['fixed_case']=row.to_dict()
    step4_status.value+=f' Frozen {row.selection} case; manifest side={row.side}. {row.selection_reason}.'


def freeze_step4_cases(_=None):
    global step4_population,step4_manifest,step4_navigation,step4_quote_source,step4_busy
    if step4_busy:return
    if step4_manifest is not None:
        step4_status.value=f'The existing {len(step4_manifest)} frozen cases are retained for this loaded source. Choose a case from the fixed list.'
        return
    step4_busy=True;step4_freeze.disabled=True;started=perf_counter()
    try:
        step4_status.value='Preparing the full traded-universe population, then three fixed local rule probes per bond/side/day...'
        if step4_population is None:
            prepared=prepare_quote_events(bcq_df)
            step4_population=population_tables(data_ig,bcq_df,event_cache=prepared)
        if step4_manifest is None:
            step4_manifest=fixed_case_manifest(step4_population)
        step4_quote_source=step4_population['raw']
        step4_navigation=step4_quote_source[['ISSUER']+KEYS].dropna().drop_duplicates().copy()
        step4_navigation['day']=step4_navigation.quote_timestamp_ET.dt.normalize()
        # The population event cache is shared. Raw values are normalized once per
        # issuer for plotting, without rebuilding event histories or candidate sets.
        prepared_events=step4_population['event_cache']['events']
        by_bond=step4_population['universe'].set_index('cusip').ISSUER
        event_groups=prepared_events.groupby(prepared_events.cusip.map(by_bond),observed=True).indices
        for issuer,indices in step4_quote_source.groupby('ISSUER',observed=True).indices.items():
            raw=step4_quote_source.iloc[indices][KEYS+['spread','quantity']].copy()
            raw['s']=pd.to_numeric(raw.spread,errors='coerce').replace([np.inf,-np.inf],np.nan)
            e=prepared_events.iloc[event_groups.get(issuer,[])]
            step4_issuer_cache[issuer]=dict(raw=raw,events=e,unkeyed=len(raw)-int(e.rows.sum()))
        step4_issuer.options=representative_issuers(step4_quote_source)
        options=case_options(step4_manifest)
        step4_case.options=options;step4_case.disabled=step4_manifest.empty
        step4_case.value=options[0][1] if options else None
        step4_status.value=(f'Frozen {len(step4_manifest)} distinct cases in {perf_counter()-started:.2f}s. '
            'Random seed 2026; typical strata; high impact uses measured cleaning effects at three local queries. '
            'This is descriptive evidence, not prediction gain. Full Global/SECTOR/dealer tables remain in step4_population.')
    finally:step4_busy=False;step4_freeze.disabled=False
    if not step4_manifest.empty:choose_step4_case()


def save_step4(_=None):
    folder=Path('outputs/quote_quality_step4');folder.mkdir(parents=True,exist_ok=True)
    r=step4_result
    name='_'.join(str(v) for v in [r['issuer'],r['bond'],step4_dealer.value,r['day'],r['age'],r['sync']])
    path=folder/(''.join(c if c.isalnum() else '_' for c in name)[:220]+'.png')
    step4_figure.savefig(path,dpi=180);step4_status.value=f'Saved: {path}'

step4_controls=[step4_issuer,step4_bond,step4_day,step4_dealer,step4_age,step4_sync]
for box in step4_controls:box.observe(refresh_step4,names='value')
step4_case.observe(choose_step4_case,names='value');step4_freeze.on_click(freeze_step4_cases)
step4_apply.on_click(apply_step4);step4_save.on_click(save_step4)
step4_dashboard=widgets.VBox([widgets.HBox([step4_freeze,step4_case]),step4_issuer,widgets.HBox([step4_bond,step4_day,step4_dealer]),widgets.HBox([step4_age,step4_sync,step4_apply,step4_save]),step4_status,step4_image])
display(step4_dashboard)
refresh_step4()
