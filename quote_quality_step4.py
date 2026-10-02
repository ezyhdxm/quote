# %% [markdown]
# # Step 4 — same-dealer bid/ask candidate pairing
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


from quote_quality_core import (event_history, pair_snapshots, pair_features, representative_issuers)

# %% [markdown]
# ## Decisions
# Compare observed signed gaps, exact-time pairs, and positive raw-size matches within a short time gap.
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
    policies=[('All fresh',fresh,'cross'),('Exact time',fresh.loc[fresh.same_time],'cross'),('Size + short gap',p.loc[p.size_time_pair],'cross_matched')]
    bottom=np.zeros(3)
    for state,color in [('None','#277F8E'),('Some','#BA8C27'),('All','#C9563D')]:
        values=[z[field].eq(state).mean() if len(z) else 0 for _,z,field in policies]
        c.bar(range(3),values,bottom=bottom,color=color,label=state);bottom+=values
    for i,(_,z,_) in enumerate(policies):c.text(i,1.03,f'n={len(z):,}' if len(z) else 'Unassessed',ha='center',fontsize=9)
    c.set_xticks(range(3),[label for label,_,_ in policies]);c.set_ylim(0,1.18);c.set_ylabel('Fraction of dealer-grid pairs')
    c.set_title('All dealers: crossing under three pair policies')
    total=p.groupby('time').size().reindex(f.index,fill_value=0)
    complete=p.loc[p.complete].groupby('time').size().reindex(f.index,fill_value=0)
    for values,label,color in [(total,'At least one side','#AAAAAA'),(complete,'Both complete','#555555'),(f.n_pair,'Both fresh','#277F8E'),(f.n_size_time_pair,'Size + short gap','#8560A5')]:
        d.step(f.index,values,where='post',label=label,color=color)
    d.set_title('All dealers: pairing coverage cost');d.set_ylabel('Dealers');d.set_ylim(bottom=0)
    for state,color in [('None','#277F8E'),('Some','#BA8C27'),('All','#C9563D')]:
        z=fresh.loc[fresh['cross'].eq(state)]
        ax5.scatter(z.time_gap,z.gap_center,s=13,alpha=.5,color=color,label=state)
    ax5.axvline(sync,color='#777777',ls='--',lw=.8);ax5.axhline(0,color='#777777',lw=.8)
    ax5.set_title('All dealers: crossing versus time mismatch');ax5.set_xlabel('Absolute bid/ask timestamp gap (min)');ax5.set_ylabel('Observed center gap (bps)')
    for col,label,color in [('pair_mid','All fresh paired centers','#277F8E'),('size_time_mid','Size + short gap','#8560A5')]:
        ax6.plot(f.index,f[col],color=color,label=label,drawstyle='steps-post')
    ax6.set_title('All dealers: midpoint sensitivity to pairing');ax6.set_ylabel('Spread midpoint description (bps)')
    for ax in [a,b,d,ax6]:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3,maxticks=6,tz=day.tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=day.tz));ax.set_xlabel('ET')
        if len(f):ax.set_xlim(f.index.min()-pd.Timedelta(minutes=1),f.index.max()+pd.Timedelta(minutes=1))
    for ax in axes.flat:
        ax.grid(alpha=.15);h,l=ax.get_legend_handles_labels()
        if h:ax.legend(fontsize=8,ncol=2,loc='best')
    fig.text(.08,.085,'Decision: keep single-side features everywhere; add paired descriptors only with coverage, time-gap and candidate/size ambiguity.',fontsize=11)
    fig.text(.08,.05,'Each dealer contributes once per grid time. Policy populations overlap; a lower crossing rate can reflect dropped pairs. No candidate is selected merely to avoid crossing.',fontsize=10)
    fig.text(.08,.025,'Zero / negative spreads and zero quantities are retained. Positive raw-size matching does not establish economic comparability or an executable market.',fontsize=10,color='#555555')
    return fig

# %% [markdown]
# ## Use and share
# The issuer labels mean previously reviewed cases, not clean/dirty issuers.
# Cases with observed bid and ask on the same dealer/bond/day are promoted first; all choices remain available.
# `step4_result['pairs']` retains missing sides, incomplete pairs, raw-size matches and crossing classes.
# Save PNG exports all six panels. No DataFrame is printed.

# %% 3. Case controls
if 'step4_controls' in globals():
    for box in step4_controls:box.unobserve(refresh_step4,names='value')
    step4_save.on_click(save_step4,remove=True)
    step4_dashboard.close()
step4_issuer=widgets.Dropdown(options=representative_issuers(bcq_df),description='Issuer:',layout=widgets.Layout(width='850px'))
step4_bond=widgets.Dropdown(description='Bond:');step4_dealer=widgets.Dropdown(description='Dealer:');step4_day=widgets.Dropdown(description='ET day:')
step4_age=widgets.Dropdown(options=[10,30,60],value=AGE_MIN,description='Age min:')
step4_sync=widgets.Dropdown(options=[0,1,5],value=SYNC_MIN,description='Gap min:')
step4_save=widgets.Button(description='Save all PNG',icon='download');step4_status=widgets.HTML()
step4_image=widgets.Image(format='png',layout=widgets.Layout(width='100%',max_width='1600px'))
step4_result=None;step4_busy=False


def refresh_step4(change=None):
    global step4_result,step4_busy,step4_figure
    if step4_busy:return
    step4_busy=True
    try:
        step4_status.value='Building same-dealer pair comparisons...'
        if step4_result is None or step4_result['issuer']!=step4_issuer.value:
            step4_result=event_history(bcq_df.loc[bcq_df.ISSUER.eq(step4_issuer.value)])
            step4_result['issuer']=step4_issuer.value
        all_events=step4_result['events'];eligible=all_events
        reset=change is not None and change['owner'] is step4_issuer
        for box,col in [(step4_bond,'cusip'),(step4_dealer,'firm'),(step4_day,'day')]:
            # Promote supported two-sided cases, while retaining all one-sided cases.
            cases=eligible.groupby(['cusip','firm','day'],observed=True).side.nunique().rename('sides').reset_index()
            paired=cases.loc[cases.sides.eq(2)].groupby(col,observed=True).size()
            counts=eligible.groupby(col,observed=True).size()
            order=pd.DataFrame({'paired_cases':paired,'events':counts}).fillna(0).sort_values(['paired_cases','events'],ascending=False,kind='stable')
            counts=counts.reindex(order.index)
            old=box.value;options=counts.index.tolist()
            box.options=[(f'{v:%Y-%m-%d}' if col=='day' else str(v),v) for v in options]
            box.value=old if old in options and not reset else (options[0] if options else None)
            reset |= change is not None and change['owner'] is box
            eligible=eligible.loc[eligible[col].eq(box.value)]
        if eligible.empty:
            step4_figure=Figure(figsize=(12,5));ax=step4_figure.subplots();ax.axis('off');ax.text(.5,.5,'No keyed events',ha='center')
        else:
            e=all_events.loc[all_events.cusip.eq(step4_bond.value)&all_events.day.eq(step4_day.value)]
            start,end=e.quote_timestamp_ET.min(),e.quote_timestamp_ET.max()
            times=pd.date_range(start.ceil(GRID),end.floor(GRID),freq=GRID).union(pd.DatetimeIndex([start,end]))
            p=pair_snapshots(e,times,step4_age.value,step4_sync.value)
            f=pair_features(p,times)
            r=step4_result['raw'];raw=r.loc[r.cusip.eq(step4_bond.value)&r.quote_timestamp_ET.dt.normalize().eq(step4_day.value)]
            case=dict(issuer=step4_issuer.value,raw=raw,events=e,pairs=p,features=f)
            step4_result.update(pairs=p,features=f)
            step4_figure=pairing_figure(case,step4_dealer.value,step4_bond.value,step4_day.value,step4_age.value,step4_sync.value)
        with BytesIO() as buffer:
            step4_figure.savefig(buffer,format='png',dpi=110);step4_image.value=buffer.getvalue()
        step4_status.value='All six panels share this bond/day. Raw observations remain available in step4_result.'
    finally:step4_busy=False


def save_step4(_=None):
    folder=Path('outputs/quote_quality_step4');folder.mkdir(parents=True,exist_ok=True)
    name='_'.join(str(v) for v in [step4_issuer.value,step4_bond.value,step4_dealer.value,step4_day.value,step4_age.value,step4_sync.value])
    path=folder/(''.join(c if c.isalnum() else '_' for c in name)[:220]+'.png')
    step4_figure.savefig(path,dpi=180);step4_status.value=f'Saved: {path}'

step4_controls=[step4_issuer,step4_bond,step4_dealer,step4_day,step4_age,step4_sync]
for box in step4_controls:box.observe(refresh_step4,names='value')
step4_save.on_click(save_step4)
step4_dashboard=widgets.VBox([step4_issuer,widgets.HBox([step4_bond,step4_dealer,step4_day]),widgets.HBox([step4_age,step4_sync,step4_save]),step4_status,step4_image])
display(step4_dashboard)
refresh_step4()
