# SETUP LOGIC: 模块定义与已有依赖；导入本身不表示计算或训练已完成。
# %% [markdown]
# # Step 4 — same-dealer bid/ask candidate pairing
# Run three cells. Loading is included; output is one shareable dashboard.

# %% 1. Load data
# SETUP LOGIC: 现成loader、数值、绘图和控件依赖。
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

# CONFIGURATION LOGIC: 原数据路径与loader配置。
PIPELINE_CSV = Path("data/pipeline/data_pipeline.csv_20260506")
BENCHMARK_CSV = Path("data/pipeline/DailyCloseUSTBenchmarks.csv_20260506")
RAW_QUOTES_FILE = Path("data/bondcliq/quotes_pretrade_260301_260401_Wells_quotes2.parquet")
DATA_IG_CACHE = Path("data/pipeline/data_ig.parquet")

# SETUP LOGIC: to_ny_datetime：函数接口；计算阶段见内部CORE标记。
def to_ny_datetime(series):
    # CORE LOGIC: STEP 1 — 统一纽约时间
    # Input: series=["2026-03-02 15:00+00:00"].
    # Output: [2026-03-02 10:00-05:00].
    # Trick: 无时区localize，有时区convert；不把纽约钟面误当UTC。
    values = pd.to_datetime(series, errors="coerce", format="mixed")
    if values.dt.tz is None:
        return values.dt.tz_localize("America/New_York")
    return values.dt.tz_convert("America/New_York")

# CACHEING LOGIC: 读取现成trade缓存；缺失时才运行原loader。
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

# FILE IO LOGIC: 读取原quote文件。
bcq_df = pd.read_parquet(RAW_QUOTES_FILE)
# CORE LOGIC: STEP 1 — 固定traded-bond universe与known time
# Input: trade CUSIP=[A]; quotes=[(A,15:00 UTC),(B,15:00 UTC)] on 03-02.
# Output: 仅A保留，ET timestamp=10:00-05:00。
# Trick: 过滤按交易universe，不按spread、quantity或crossing。
bcq_df = bcq_df.loc[bcq_df["cusip"].isin(data_ig["CUSIP"])].copy()
bcq_df["quote_timestamp_ET"] = pd.to_datetime(
    bcq_df["quote_timestamp_UTC"], utc=True, errors="coerce", format="mixed",
).dt.tz_convert("America/New_York")
# CORE LOGIC: STEP 2 — 附加展示issuer
# Input: trade mapping A→I; quote cusip=A.
# Output: quote.ISSUER=I；quote数值不变。
cusip_issuer = (data_ig[["ISSUER", "CUSIP"]].dropna()
                .drop_duplicates("CUSIP").set_index("CUSIP")["ISSUER"])
bcq_df["ISSUER"] = bcq_df["cusip"].map(cusip_issuer)


# SETUP LOGIC: 共享事件/配对/群体函数接口。
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
# CONFIGURATION LOGIC: 默认age=30分钟、两侧同步间隔=1分钟、网格5分钟。
AGE_MIN = 30
SYNC_MIN = 1
GRID = '5min'


# SETUP LOGIC: pairing_figure：函数接口；计算阶段见内部CORE标记。
def pairing_figure(result, dealer, bond, day, age, sync):
    # CORE LOGIC: STEP 1 — 固定所选dealer及A fresh母集
    # Input: p含D1 fresh=True、D2 fresh=False；dealer=D1；raw两dealer各1行。
    # Output: own=D1 raw；selected=D1 slots；fresh=picked=D1 slot。
    # Trick: fresh仍从全dealer母集选；不能用所选dealer替代群体分母。
    raw,e,p,f=result['raw'],result['events'],result['pairs'],result['features']
    own=raw.loc[raw.firm.eq(dealer)]
    selected=p.loc[p.firm.eq(dealer)]
    fresh=p.loc[p.fresh_pair]; picked=selected.loc[selected.fresh_pair]
    # PLOTTING LOGIC: 六子图布局与原始候选、gap和同slot对照可视化。
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
    # CORE LOGIC: STEP 2 — 分离slot选择效应和quantity匹配效应
    # Input: p有3个fresh slots，size_time_pair=[True,True,False]；cross=[None,Some,All]；cross_matched前两项=[None,None].
    # Output: A=3个original；B=前2个original；C=相同前2个matched。
    # Trick: B和C行完全相同，只切换候选规则；A→B才含选择效应。
    policies=[('A fresh\noriginal',fresh,'cross'),
              ('B match slots\noriginal',p.loc[p.size_time_pair],'cross'),
              ('C same B slots\nmatched',p.loc[p.size_time_pair],'cross_matched')]
    # PLOTTING LOGIC: 绘各crossing类占比；这不是各候选组合的发生概率。
    bottom=np.zeros(3)
    for state,color in [('None','#277F8E'),('Some','#BA8C27'),('All','#C9563D')]:
        # CORE LOGIC: STEP 3 — 计算各政策同slot crossing占比
        # Input: state=None；A.cross=[None,Some,All]；B.cross=[None,Some]；C.cross_matched=[None,None].
        # Output: values=[1/3,1/2,1]；每项是该政策dealer-grid slots占比；空政策返回绘图占位0并标Unassessed。
        # Trick: C用matched分类但行与B相同；不把候选组合数当分母。
        values=[z[field].eq(state).mean() if len(z) else 0 for _,z,field in policies]
        # PLOTTING LOGIC: 把已算比例画成堆叠柱，bottom仅确定柱位置。
        c.bar(range(3),values,bottom=bottom,color=color,label=state);bottom+=values
    for i,(_,z,_) in enumerate(policies):c.text(i,1.03,f'n={len(z):,}' if len(z) else 'Unassessed',ha='center',fontsize=9)
    c.set_xticks(range(3),[label for label,_,_ in policies]);c.set_ylim(0,1.18);c.set_ylabel('Fraction of dealer-grid pairs')
    c.set_title('All dealers: A→B selection; B→C matching effect')
    # CORE LOGIC: STEP 4 — 同一时间网格计算观测和完整覆盖
    # Input: t1两个slots: complete=[True,False]; t2没有slot；f.index=[t1,t2].
    # Output: total=[2,0]; complete=[1,0].
    # Trick: reindex(fill_value=0)只补计数；不把缺失spread填成0。
    total=p.groupby('time').size().reindex(f.index,fill_value=0)
    complete=p.loc[p.complete].groupby('time').size().reindex(f.index,fill_value=0)
    # PLOTTING LOGIC: 绘覆盖及时间间隔；极端点标签保留原quote数量来源。
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
        # CORE LOGIC: STEP 5 — 查某个正/负gap边界的原始来源
        # Input: fresh gap_low: D1@t1=-8,D2@t2=-4；field=gap_low,ascending=True；extremes含D1的bid58/ask66.
        # Output: row=D1@t1；z为该极低边界的bid58与ask66原始来源。
        # Trick: stable排序保证相同边界的选择顺序可重复；不自动删负gap。
        row=fresh.sort_values(field,ascending=ascending,kind='stable').iloc[0]
        z=extremes.loc[extremes.extreme.eq(label)&extremes.firm.eq(row.firm)&extremes.time.eq(row.time)&extremes.boundary_candidate]
        # PLOTTING LOGIC: 边界来源做短标签，完整来源仍保留在result中。
        tags=[]
        for side in ['bid','ask']:
            source=z.loc[z.side.eq(side)].drop_duplicates(['spread','quantity'])
            values=[f'{r.spread:g} / q={r.quantity}' if isinstance(r.spread,(float,int)) else f'{r.spread} / q={r.quantity}' for r in source.itertuples()]
            tags.append(side+' '+', '.join(values[:2])+(' …' if len(values)>2 else ''))
        ax5.scatter([row.time_gap],[row[field]],marker='^' if ascending else 'v',color='#202020',s=45,zorder=5)
        ax5.annotate(f'{row.firm} {row[field]:.1f} bps\n'+ '\n'.join(tags),(row.time_gap,row[field]),
                     xytext=(5,offset),textcoords='offset points',fontsize=7,va='bottom' if ascending else 'top')
    ax5.set_title('All dealers: ± gap boundaries and raw source quantities');ax5.set_xlabel('Absolute bid/ask timestamp gap (min)');ax5.set_ylabel('Center / boundary gap (bps)')
    # CORE LOGIC: STEP 6 — 在B的相同slot计算原始midpoint
    # Input: t1两条size_time slots mid=[60,64]，另有不合格mid80；f.index=[t1,t2].
    # Output: b_mid=[62,NaN]；80不进B分母，无匹配的t2为未知。
    # Trick: B→C比较只改变候选匹配，不改变slot样本。
    b_mid=p.loc[p.size_time_pair].groupby('time').mid.mean().reindex(f.index)
    # PLOTTING LOGIC: 共用坐标绘A/B/C摘要，展示完整六面板和口径。
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
    # CORE LOGIC: STEP 7 — 量化两类效应
    # Input: summary A/B/C gap_mean=[4,6,5], mid_mean=[60,62,63].
    # Output: selection=+2 bps; matching=-1 bps; mid_matching=+1 bps。
    # Trick: 三个数使用既有同slot汇总；不误把A→C混合效应称为quantity贡献。
    summary=result['policy_comparison']
    selection=summary.iloc[1].gap_mean_bps-summary.iloc[0].gap_mean_bps
    matching=summary.iloc[2].gap_mean_bps-summary.iloc[1].gap_mean_bps
    mid_matching=summary.iloc[2].mid_mean_bps-summary.iloc[1].mid_mean_bps
    # PLOTTING LOGIC: 打印图注并返回Figure，不输出大表。
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
# UI LOGIC: 重跑控件cell时解除旧回调，避免一次点击触发多次计算。
if 'step4_controls' in globals():
    for box in step4_controls:box.unobserve(refresh_step4,names='value')
    step4_case.unobserve(choose_step4_case,names='value')
    step4_freeze.on_click(freeze_step4_cases,remove=True)
    step4_apply.on_click(apply_step4,remove=True)
    step4_save.on_click(save_step4,remove=True)
    step4_dashboard.close()
# Rerunning this cell reuses explicitly scoped caches only for the same loaded source.
# CACHEING LOGIC: 仅同loaded source复用issuer和bond/day缓存。
if globals().get('step4_cache_source') is not bcq_df:
    step4_issuer_cache={};step4_pair_cache={};step4_cache_source=bcq_df
    step4_population=None;step4_manifest=None;step4_quote_source=bcq_df
# NAVIGATION LOGIC: 唯一事件键用于菜单；菜单选择需Apply才成为结果。
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


# PLOTTING LOGIC: 重绘已Apply结果；dealer切换不重建all-dealer状态。
def paint_step4():
    global step4_figure
    if step4_result is None:return
    step4_figure=pairing_figure(step4_result,step4_dealer.value,step4_result['bond'],
        step4_result['day'],step4_result['age'],step4_result['sync'])
    with BytesIO() as buffer:
        step4_figure.savefig(buffer,format='png',dpi=110);step4_image.value=buffer.getvalue()


# NAVIGATION LOGIC: 级联菜单优先有双边的案例，所有选项保留；只更新导航。
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


# UI LOGIC: Apply的busy防重入、空案例提示及完成状态。
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
        # CACHEING LOGIC: 每issuer事件历史只建一次，all-dealer状态按bond/day/grid缓存。
        event_reused=key[0] in step4_issuer_cache
        if not event_reused:
            step4_status.value='Building issuer events...'
            step4_issuer_cache[key[0]]=event_history(step4_quote_source.loc[step4_quote_source.ISSUER.eq(key[0])])
        history=step4_issuer_cache[key[0]]
        events_done=perf_counter()
        pair_reused=key in step4_pair_cache
        if not pair_reused:
            step4_status.value='Building all-dealer bond/day state...'
            # CORE LOGIC: STEP 1 — 生成本bond/day含首末点的查询网格
            # Input: A当日事件10:01和10:12，另有B事件；GRID=5min.
            # Output: e仅A；times=[10:01,10:05,10:10,10:12]；raw仅A当日。
            # Trick: ceil/floor给内部网格，union补真实首末时间；不捏造全天观察。
            all_events=history['events'];e=all_events.loc[all_events.cusip.eq(key[1])&all_events.day.eq(key[2])]
            start,end=e.quote_timestamp_ET.min(),e.quote_timestamp_ET.max()
            times=pd.date_range(start.ceil(GRID),end.floor(GRID),freq=GRID).union(pd.DatetimeIndex([start,end]))
            r=history['raw'];raw=r.loc[r.cusip.eq(key[1])&r.quote_timestamp_ET.dt.normalize().eq(key[2])]
            # CACHEING LOGIC: 保存一次all-dealer pair_snapshots，age/sync变化复用此状态。
            step4_pair_cache[key]=dict(raw=raw,events=e,times=times,pairs=pair_snapshots(e,times))
        cached=step4_pair_cache[key]
        state_done=perf_counter()
        # CORE LOGIC: STEP 2 — 在同一缓存状态重新派生规则和摘要
        # Input: 缓存t的D1 bid62、ask60，各age5min，time_gap0，共同quantity2；age=30,sync=1.
        # Output: p fresh_pair=True,size_time_pair=True,gap_center=2,mid=61；f n_pair=1；policy_comparison和原始来源随结果保存。
        # Trick: 原始候选不丢；规则只派生资格mask，不就地删除历史。
        p=pair_policy_masks(cached['pairs'],step4_age.value,step4_sync.value)
        f=pair_features(p,cached['times'])
        step4_result=dict(issuer=key[0],bond=key[1],day=key[2],age=step4_age.value,sync=step4_sync.value,
            raw=cached['raw'],events=cached['events'],pairs=p,features=f,
            policy_comparison=pair_policy_comparison(p),extreme_sources=pair_extreme_sources(cached['raw'],p,1))
        # UI LOGIC: 记录已Apply键、绘图、timings和缓存复用状态。
        step4_applied=(key[0],key[1],key[2],step4_age.value,step4_sync.value)
        calculated=perf_counter();paint_step4();step4_save.disabled=False
        step4_result['timings']=dict(calculate_s=calculated-started,render_s=perf_counter()-calculated,
            event_s=events_done-started,pair_state_s=state_done-events_done,masks_features_s=calculated-state_done,
            event_cache_reused=event_reused,pair_cache_reused=pair_reused)
        step4_status.value=(f'All-dealer bond/day slots. Events {events_done-started:.2f}s; state {state_done-events_done:.2f}s; masks/features {calculated-state_done:.2f}s; render {perf_counter()-calculated:.2f}s. '
            f'Event cache {"reused" if event_reused else "built"}; state cache {"reused" if pair_reused else "built"}. '
            'A→B selects slots; B→C matches quantity on the same slots. Raw ± sources are in step4_result.')
    finally:step4_busy=False


# NAVIGATION LOGIC: 使用已冻结案例的issuer/bond/dealer/day；选后显式调用Apply。
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


# UI LOGIC: 冻结按钮防重入；存在名单则保留；仅显式点击才全群体扫描。
def freeze_step4_cases(_=None):
    global step4_population,step4_manifest,step4_navigation,step4_quote_source,step4_busy
    if step4_busy:return
    if step4_manifest is not None:
        step4_status.value=f'The existing {len(step4_manifest)} frozen cases are retained for this loaded source. Choose a case from the fixed list.'
        return
    step4_busy=True;step4_freeze.disabled=True;started=perf_counter()
    try:
        step4_status.value='Preparing the full traded-universe population, then three fixed local rule probes per bond/side/day...'
        # CACHEING LOGIC: 共享群体事件与固定名单；已算source不再次聚合。
        if step4_population is None:
            prepared=prepare_quote_events(bcq_df)
            step4_population=population_tables(data_ig,bcq_df,event_cache=prepared)
        if step4_manifest is None:
            step4_manifest=fixed_case_manifest(step4_population)
        # NAVIGATION LOGIC: 用群体raw建立选择器键；不影响事件表。
        step4_quote_source=step4_population['raw']
        step4_navigation=step4_quote_source[['ISSUER']+KEYS].dropna().drop_duplicates().copy()
        step4_navigation['day']=step4_navigation.quote_timestamp_ET.dt.normalize()
        # The population event cache is shared. Raw values are normalized once per
        # issuer for plotting, without rebuilding event histories or candidate sets.
        # CACHEING LOGIC: 按issuer索引共享事件表，原始spread仅标准化供绘图；不重建candidate sets。
        prepared_events=step4_population['event_cache']['events']
        by_bond=step4_population['universe'].set_index('cusip').ISSUER
        event_groups=prepared_events.groupby(prepared_events.cusip.map(by_bond),observed=True).indices
        for issuer,indices in step4_quote_source.groupby('ISSUER',observed=True).indices.items():
            raw=step4_quote_source.iloc[indices][KEYS+['spread','quantity']].copy()
            raw['s']=pd.to_numeric(raw.spread,errors='coerce').replace([np.inf,-np.inf],np.nan)
            e=prepared_events.iloc[event_groups.get(issuer,[])]
            step4_issuer_cache[issuer]=dict(raw=raw,events=e,unkeyed=len(raw)-int(e.rows.sum()))
        # UI LOGIC: 填冻结案例菜单并显示有限扫描完成状态。
        step4_issuer.options=representative_issuers(step4_quote_source)
        options=case_options(step4_manifest)
        step4_case.options=options;step4_case.disabled=step4_manifest.empty
        step4_case.value=options[0][1] if options else None
        step4_status.value=(f'Frozen {len(step4_manifest)} distinct cases in {perf_counter()-started:.2f}s. '
            'Random seed 2026; typical strata; high impact uses measured cleaning effects at three local queries. '
            'This is descriptive evidence, not prediction gain. Full Global/SECTOR/dealer tables remain in step4_population.')
    finally:step4_busy=False;step4_freeze.disabled=False
    if not step4_manifest.empty:choose_step4_case()


# FILE IO LOGIC: 保存所有六面板PNG；安全化文件名。
def save_step4(_=None):
    folder=Path('outputs/quote_quality_step4');folder.mkdir(parents=True,exist_ok=True)
    r=step4_result
    name='_'.join(str(v) for v in [r['issuer'],r['bond'],step4_dealer.value,r['day'],r['age'],r['sync']])
    path=folder/(''.join(c if c.isalnum() else '_' for c in name)[:220]+'.png')
    step4_figure.savefig(path,dpi=180);step4_status.value=f'Saved: {path}'

# UI LOGIC: 绑定控件和展示；首次Apply行为保持原样。
step4_controls=[step4_issuer,step4_bond,step4_day,step4_dealer,step4_age,step4_sync]
for box in step4_controls:box.observe(refresh_step4,names='value')
step4_case.observe(choose_step4_case,names='value');step4_freeze.on_click(freeze_step4_cases)
step4_apply.on_click(apply_step4);step4_save.on_click(save_step4)
step4_dashboard=widgets.VBox([widgets.HBox([step4_freeze,step4_case]),step4_issuer,widgets.HBox([step4_bond,step4_day,step4_dealer]),widgets.HBox([step4_age,step4_sync,step4_apply,step4_save]),step4_status,step4_image])
display(step4_dashboard)
refresh_step4()
