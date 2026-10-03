# SETUP LOGIC: 模块定义与已有依赖；导入本身不表示计算或训练已完成。
# %% [markdown]
# # Step 3 — quote cleaning choices and point-in-time feature drafts
# Run three cells. Choose issuer and dealer / bond / side / ET day.
# All four questions appear together; optional detail views enlarge one section.
# The shown rules are hypotheses for later chronological validation, not a fitted cleaner.

# %% 1. Load the same data and traded-bond universe
# SETUP LOGIC: 现成loader、数组、绘图和控件依赖。
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
from time import perf_counter
from quote_quality_core import event_history, side_features_fast
from quote_quality_cache import prepare_quote_events

# CONFIGURATION LOGIC: 沿用原路径；本次注释不改变数据来源。
PIPELINE_CSV = Path("data/pipeline/data_pipeline.csv_20260506")
BENCHMARK_CSV = Path("data/pipeline/DailyCloseUSTBenchmarks.csv_20260506")
RAW_QUOTES_FILE = Path("data/bondcliq/quotes_pretrade_260301_260401_Wells_quotes2.parquet")
DATA_IG_CACHE = Path("data/pipeline/data_ig.parquet")

# SETUP LOGIC: to_ny_datetime：函数接口；计算阶段见内部CORE标记。
def to_ny_datetime(series):
    # CORE LOGIC: STEP 1 — 统一纽约时间
    # Input: series=["2026-03-02 15:00:00+00:00"].
    # Output: [2026-03-02 10:00:00-05:00].
    # Trick: 有时区用convert；无时区按本地纽约时间localize，不误当UTC。
    values = pd.to_datetime(series, errors="coerce", format="mixed")
    if values.dt.tz is None:
        return values.dt.tz_localize("America/New_York")
    return values.dt.tz_convert("America/New_York")

# CACHEING LOGIC: 有现成trade Parquet就读取；否则调用原loader并保存缓存。
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

# FILE IO LOGIC: 读取现有quote文件；不在这里做新模型训练。
bcq_df = pd.read_parquet(RAW_QUOTES_FILE)
# CORE LOGIC: STEP 1 — 固定traded-bond universe并转换quote时间
# Input: data_ig.CUSIP=[A]; quotes=[(A,15:00 UTC),(Z,15:00 UTC)] on 2026-03-02.
# Output: 仅A保留，quote_timestamp_ET=10:00-05:00；Z不在universe。
# Trick: UTC必须显式声明；isin保留universe内所有quote行，不按spread符号过滤。
bcq_df = bcq_df.loc[bcq_df["cusip"].isin(data_ig["CUSIP"])].copy()
bcq_df["quote_timestamp_ET"] = pd.to_datetime(
    bcq_df["quote_timestamp_UTC"], utc=True, errors="coerce", format="mixed",
).dt.tz_convert("America/New_York")
# CORE LOGIC: STEP 2 — 给展示行附issuer
# Input: data_ig=[(CUSIP=A,ISSUER=I)]; quote.cusip=[A].
# Output: quote.ISSUER=[I].
# Trick: drop_duplicates沿用现成loader映射；这里不是新增因果issuer因子的prefix映射。
cusip_issuer = (data_ig[["ISSUER", "CUSIP"]].dropna()
                .drop_duplicates("CUSIP").set_index("CUSIP")["ISSUER"])
bcq_df["ISSUER"] = bcq_df["cusip"].map(cusip_issuer)

# %% [markdown]
# ## Four questions, four uses
# 1. Candidates: does one scalar hide ambiguity or manufacture momentum?
# 2. Quantity: is conditional aggregation supported, or just sparse / ambiguous?
# 3. Age: compare no expiry, age decay and a maximum message age.
# 4. Influence: compare dealer-equal aggregation with candidate clipping / dealer downweighting.
# `step3_result` retains raw rows, events, as-of dealer slots and the selected case's features.
# `step3_features` is a draft on a five-minute research grid, not a full trade-level feature job.
# Event groups are atomic at their exact timestamps. All calculations use quotes at or before t.
# First observed change ages are unknown; no fill across ET days. Center is descriptive, not executable.

# %% 2. Event sets, as-of comparisons and figures
# CONFIGURATION LOGIC: 事件键、固定30min窗口和研究阈值；只声明，不自动清洗。
KEYS = ["firm", "cusip", "side", "quote_timestamp_ET"]
SERIES = KEYS[:3]
GRID = "5min"
HISTORY_GAP_MIN = 60  # Reset change history, not an asserted quote expiry.
LOOKBACK_MIN = 30
DEFAULT_AGE_MIN = 30
MIN_PEERS = 3
CLIP_FLOOR_BPS = 10.0
MAD_MULTIPLIER = 4.0
issuer_labels = bcq_df["ISSUER"].astype("string").fillna("[Missing issuer]")
VIEWS = ["1 Candidates", "2 Quantity", "3 Age", "4 Influence"]
ALL_VIEWS = "All four"

# Event construction is shared across issuers. Source rows remain in bcq_df;
# only the narrow diagnostic columns travel through the case interface.
# CACHEING LOGIC: 同一loaded对象复用事件表；仅source对象改变才重新prepare。
if globals().get("step3_source") is not bcq_df:
    step3_prepared = prepare_quote_events(bcq_df)
    step3_source = bcq_df
    step3_raw = bcq_df[KEYS + ["spread", "quantity", "ISSUER"]].copy()
    # CORE LOGIC: STEP 3 — 保留重复和spread有限性标记
    # Input: raw spreads=[0,-2,+inf], quantities=[0,2,None]，三行互不重复。
    # Output: repeat=[False,False,False]; s=[0,-2,NaN]; q=[0,2,NaN].
    # Trick: 零/负spread保留；仅非有限数成为NaN用于可用性描述。
    step3_raw["repeat"] = bcq_df.duplicated()
    step3_raw["s"] = pd.to_numeric(step3_raw.spread, errors="coerce").replace([np.inf, -np.inf], np.nan)
    step3_raw["q"] = pd.to_numeric(step3_raw.quantity, errors="coerce")
    # CORE LOGIC: STEP 4 — 明确quantity类别
    # Input: quantity=[None,0,2,-1].
    # Output: qkind=[Missing,Zero,Positive,Other]; 初始qtag同类别名。
    # Trick: np.select按条件先后匹配；0不当已知正数量。
    step3_raw["qkind"] = np.select(
        [step3_raw.quantity.isna(), step3_raw.q.eq(0), np.isfinite(step3_raw.q) & step3_raw.q.gt(0)],
        ["Missing", "Zero", "Positive"], default="Other")
    step3_raw["qtag"] = step3_raw.qkind.astype(str)
    # CORE LOGIC: STEP 5 — 保留具体数量标签与不完整标记
    # Input: q=[2,-1,0]; qkind=[Positive,Other,Zero]; s=[60,NaN,0].
    # Output: qtag=[q=2.0,Other:-1,Zero]; bad=[False,True,False].
    # Trick: repr(float(v))避免用显示四舍五入合并不同正数量。
    positive = step3_raw.qkind.eq("Positive")
    step3_raw.loc[positive, "qtag"] = step3_raw.loc[positive, "q"].map(lambda v: "q=" + repr(float(v)))
    other = step3_raw.qkind.eq("Other")
    step3_raw.loc[other, "qtag"] = "Other:" + step3_raw.loc[other, "quantity"].astype(str)
    step3_raw["bad"] = step3_raw.s.isna()
    # CACHEING LOGIC: 重置此数据source的局部案例缓存；不更改原始rows。
    step3_case_cache = {}
    step3_population = None
    step3_manifest = None


# SETUP LOGIC: issuer_choices：函数接口；计算阶段见内部CORE标记。
def issuer_choices():
    """Rank the shared event table; navigation labels are descriptive only."""
    # CORE LOGIC: STEP 1 — 汇总导航所需事件特征
    # Input: I有两事件：candidate_count=[1,2], quantity_set=[(q=1.0),(q=1.0,q=2.0)].
    # Output: multi=[False,True]; size_multi=[False,True]; I: events=2,multi=1,size_multi=1.
    # Trick: 事件计数与原始重复行数不同；sum布尔值只计事件。
    g = step3_prepared["events"].copy()
    g["issuer"] = g.cusip.map(cusip_issuer).astype("string").fillna("[Missing issuer]")
    g["multi"] = g.candidate_count.gt(1)
    g["size_multi"] = g["multi"] & g.quantity_set.map(lambda tags: sum(t.startswith("q=") for t in tags) > 1)
    scores = g.groupby("issuer", observed=True).agg(
        events=("candidate_count", "size"), multi=("multi", "sum"), size_multi=("size_multi", "sum"),
        days=("day", "nunique"), dealers=("firm", "nunique"), gap=("gap", "median"))
    # CORE LOGIC: STEP 2 — 支持门槛与导航比例
    # Input: I:events=200,multi=20,days=3,dealers=4; J:events=10,multi=5,days=1,dealers=1.
    # Output: I rate=0.10且进入supported/multi；J不进入supported。
    # Trick: 这些门槛只选导航代表，不删训练数据或认定异常。
    scores["rate"] = scores["multi"] / scores["events"]
    supported = scores.loc[scores.events.ge(100) & scores.days.ge(2) & scores.dealers.ge(2)]
    multi = supported.loc[supported.multi.ge(5)]
    # NAVIGATION LOGIC: 保留已读案例标签，再按指标挑导航入口；不作为模型特征或发生率估计。
    promoted = {}
    for prefix in ["SKY GROUP", "EASTERN GAS", "DUKE ENERGY", "IBM", "EXPAND ENERGY", "HPS CORPORATE", "COMCAST", "MITSUBISHI UFJ"]:
        for name in sorted(issuer_labels.unique()):
            if name.upper().startswith(prefix):
                promoted[name] = "Reviewed case"
                break
    for tag, pool, field, ascending in [
        ("Multi", multi, "rate", False), ("Wide", multi, "gap", False),
        ("Quantity", multi.loc[multi.size_multi.ge(5)], "size_multi", False),
        ("Low synchronous multi", supported, "rate", True)]:
        for name in pool.sort_values([field, "events"], ascending=[ascending, False]).index[:2]:
            promoted.setdefault(name, tag)
    order = list(promoted) + [n for n in sorted(issuer_labels.unique()) if n not in promoted]
    return [(f"[{promoted[n]}] {n}" if n in promoted else n, n) for n in order]


# SETUP LOGIC: asof_features：函数接口；计算阶段见内部CORE标记。
def asof_features(events, times, age_min=DEFAULT_AGE_MIN):
    """Fast shared summaries plus case slots and fixed-lag composition diagnostics."""
    # CORE LOGIC: STEP 1 — 合并当前与固定滞后查询
    # Input: times=[10:30,10:35,10:30] ET同日；LOOKBACK_MIN=30.
    # Output: requested=[10:30,10:35]; lagged=[10:00,10:05]; queries=[10:00,10:05,10:30,10:35].
    # Trick: sort/unique/union先去重；同一时间计算一次再映射回用户请求。
    from quote_quality_core import side_features_fast
    requested = pd.DatetimeIndex(times).sort_values().unique()
    lagged = requested - pd.Timedelta(minutes=LOOKBACK_MIN)
    queries = requested.union(lagged)
    features = side_features_fast(events, queries, age_min)
    base = pd.DataFrame({"time": queries})
    parts = []
    # CORE LOGIC: STEP 2 — 每dealer向后as-of且不跨日
    # Input: dealer D事件09:59=60、10:31=70；query10:30和次日10:00.
    # Output: 10:30匹配09:59；次日匹配被day条件排除；10:31不会用于10:30。
    # Trick: 先找最新记录再看完整性，不能先删不完整记录而回退旧价。
    for firm, history in events.groupby("firm", observed=True):
        merged = pd.merge_asof(base, history.sort_values(KEYS[-1]), left_on="time", right_on=KEYS[-1], direction="backward")
        merged = merged.loc[merged[KEYS[-1]].notna() & merged.time.dt.normalize().eq(merged.day)].copy()
        if merged.empty:
            continue
        # CORE LOGIC: STEP 3 — 计算两种age并保留观测历史
        # Input: query10:30，message10:25,last_change10:10,history_start10:00.
        # Output: message_age_min=5; spread_set_change_age_min=20; observed_history_min=30；加入parts。
        # Trick: 首次last_change=NaT时change age为NaN，不能填成message age。
        merged[["complete", "change_age_unknown"]] = merged[["complete", "change_age_unknown"]].astype(bool)
        merged["message_age_min"] = (merged.time - merged[KEYS[-1]]).dt.total_seconds() / 60
        merged["spread_set_change_age_min"] = (merged.time - merged.last_change).dt.total_seconds() / 60
        merged["observed_history_min"] = (merged.time - merged.history_start).dt.total_seconds() / 60
        parts.append(merged)
    # CORE LOGIC: STEP 4 — 建立查询×dealer矩阵和默认输出
    # Input: queries=[10:00,10:30], dealers=[D1,D2]；parts含这两个dealer记录。
    # Output: center/age形状(2,2)，初值NaN；conditions为-1；peer统计初始0/NaN。
    # Trick: NaN表示未观测；-1是内部条件编码，不能解释为经济数量。
    slots = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(events.columns) + ["time", "message_age_min", "spread_set_change_age_min", "observed_history_min"])
    n, dealers = len(queries), pd.Index(events.firm.drop_duplicates())
    d = len(dealers)
    center = np.full((n, d), np.nan)
    age = np.full((n, d), np.nan)
    conditions = np.full((n, d), -1, dtype=int)
    slots["peer_center"], slots["peer_radius"], slots["peer_residual"] = np.nan, np.nan, np.nan
    slots["n_peers"], slots["clipped_candidate_count"] = 0, 0
    slots["clipped_center"], slots["dealer_weight"] = slots["center"], 1.0
    # CORE LOGIC: STEP 5 — 把完整slot散射到矩阵
    # Input: slots=[(time10:00,D1,complete=True,center60,age0,q=(2),count1)].
    # Output: center[0,0]=60; age[0,0]=0; conditions[0,0]=0; fresh[0,0]=True.
    # Trick: get_indexer产整数坐标；factorize编码(quantity_set,candidate_count)，只比较同一批编码是否相同。
    if len(slots):
        ti, di = queries.get_indexer(slots.time), dealers.get_indexer(slots.firm)
        complete = slots.complete.to_numpy(dtype=bool)
        center[ti[complete], di[complete]] = slots.loc[complete, "center"]
        age[ti[complete], di[complete]] = slots.loc[complete, "message_age_min"]
        codes, _ = pd.factorize(pd.Series(list(zip(slots.quantity_set, slots.candidate_count))))
        conditions[ti[complete], di[complete]] = codes[complete]
        fresh = np.isfinite(center) & (age <= age_min)
        # Loop over dealers, with all query times handled in arrays. No per-slot
        # DataFrame slicing/iterrows; peers always exclude the target dealer.
        # CORE LOGIC: STEP 6 — 同行支持排除本dealer
        # Input: fresh一行=[True,True,True,True]，当前j=0，own_rows=[0].
        # Output: peer_mask=[False,True,True,True]; counts=[3]; supported=[True].
        # Trick: peer_mask[:,j]=False是在副本操作，不改变全局fresh；少于3同行跳过影响评估。
        for j in range(d):
            own_rows = np.flatnonzero((di == j) & complete)
            if not len(own_rows):
                continue
            own_time = ti[own_rows]
            peer_mask = fresh[own_time].copy()
            peer_mask[:, j] = False
            counts = peer_mask.sum(axis=1)
            slots.loc[own_rows, "n_peers"] = counts
            supported = counts >= MIN_PEERS
            # CORE LOGIC: STEP 7 — 稳健同行中心与半径
            # Input: 3个fresh peers=[59,60,61], own center=80, MIN_PEERS=3.
            # Output: ref=60,MAD=1,radius=max(10,4×1.4826)=10,residual=20.
            # Trick: ref[:,None]沿dealer维广播；MAD是价差绝对偏离，不是variance。
            if not supported.any():
                continue
            idx = own_rows[supported]
            peers = np.where(peer_mask[supported], center[own_time[supported]], np.nan)
            ref = np.nanmedian(peers, axis=1)
            radius = np.maximum(CLIP_FLOOR_BPS, MAD_MULTIPLIER * 1.4826 * np.nanmedian(np.abs(peers - ref[:, None]), axis=1))
            residual = slots.loc[idx, "center"].to_numpy() - ref
            # CORE LOGIC: STEP 8 — 候选限幅与dealer软权重
            # Input: candidate_set=(50,80), ref=60,radius=10,residual=5.
            # Output: middle_low=50,middle_high=80; clipped_center=(50+70)/2=60; weight=1.
            # Trick: 单调clip后median等于两个中位次序统计量clip后的均值，不需explode全部候选；零residual默认weight1。
            candidate_sets = slots.loc[idx, "spread_set"]
            middle_low = candidate_sets.map(lambda x: x[(len(x) - 1) // 2]).to_numpy()
            middle_high = candidate_sets.map(lambda x: x[len(x) // 2]).to_numpy()
            clipped = (np.clip(middle_low, ref - radius, ref + radius) + np.clip(middle_high, ref - radius, ref + radius)) / 2
            weights = np.minimum(1, np.divide(radius, np.abs(residual), out=np.ones(len(idx)), where=residual != 0))
            slots.loc[idx, ["peer_center", "peer_radius", "peer_residual", "clipped_center", "dealer_weight"]] = np.column_stack([ref, radius, residual, clipped, weights])
        # CORE LOGIC: STEP 9 — 计受影响候选且恢复slot索引
        # Input: slot index7:spread_set=(50,80),peer_center=60,peer_radius=10,n_peers=3.
        # Output: index7 clipped_candidate_count=1（80越界，50在边界）。
        # Trick: explode复制原slot索引；groupby(level=0)+reindex按原slot对齐，不按候选次序赋值。
        supported = slots.n_peers.ge(MIN_PEERS)
        if supported.any():
            candidates = slots.loc[supported, ["spread_set", "peer_center", "peer_radius"]].explode("spread_set")
            clipped = (pd.to_numeric(candidates.spread_set) - candidates.peer_center).abs().gt(candidates.peer_radius)
            slots.loc[supported, "clipped_candidate_count"] = clipped.groupby(level=0).sum().reindex(slots.index[supported]).to_numpy()
    # CORE LOGIC: STEP 10 — 对齐当前/滞后聚合和共同dealer
    # Input: requested=[10:30], lagged=[10:00]; now中心[D1=61,D2=NaN],past=[60,62].
    # Output: now_present=[True,False],past_present=[True,True],common=[True,False],common_n=1.
    # Trick: set_axis(requested)把10:00摘要与对应10:30同索引比较，不让pandas按原timestamp相减而全NaN。
    now = features.reindex(requested).copy()
    before = features.reindex(lagged).set_axis(requested)
    ni, pi = queries.get_indexer(requested), queries.get_indexer(lagged)
    same_day = np.asarray(requested.normalize() == lagged.normalize())
    now_present, past_present = np.isfinite(center[ni]), np.isfinite(center[pi])
    common = now_present & past_present & same_day[:, None]
    common_n = common.sum(axis=1)
    # CORE LOGIC: STEP 11 — 分开真实价格变化与组成变化
    # Input: now D1=61,past D1=60,D2=62；两端同日且均有报价。
    # Output: composition_changed=True,aggregate_delta=61−61=0,center_delta_30m=NaN,aggregate_delta_30m=0.
    # Trick: 旧guarded列遇dealer/quantity组成变化置未知；不把组成变化当价格动量。
    eligible = now.n_dealers.gt(0) & before.n_dealers.gt(0) & same_day
    composition_changed = (now_present != past_present).any(axis=1) | ((conditions[ni] != conditions[pi]) & common).any(axis=1)
    now["composition_changed_30m"] = pd.Series(composition_changed, index=requested).astype(float).where(eligible)
    aggregate_delta = (now.center_equal - before.center_equal).where(eligible)
    # Preserve the old guarded feature; the new diagnostic retains common dealer
    # changes even when quantity support/candidate count changes.
    now["center_delta_30m"] = aggregate_delta.where(~composition_changed)
    now["aggregate_delta_30m"] = aggregate_delta
    # CORE LOGIC: STEP 12 — 共同dealer差分与两端留存
    # Input: now=[61,NaN],past=[60,62],common=[True,False],now_n=1,past_n=2,条件D1不变.
    # Output: common_delta=1; n_common=1; current_retention=1; past_retention=0.5; condition_changed_fraction=0.
    # Trick: np.divide指定out=NaN和where>0，无共同dealer时保持未知；不是nansum后的假0变化。
    common_difference = np.where(common, center[ni] - center[pi], np.nan)
    now["common_dealer_delta_30m"] = np.divide(np.nansum(common_difference, axis=1), common_n, out=np.full(len(requested), np.nan), where=common_n > 0)
    now["n_common_dealers_30m"] = common_n
    now["common_retention_30m"] = np.divide(common_n, now.n_dealers.to_numpy(), out=np.full(len(requested), np.nan), where=now.n_dealers.to_numpy() > 0)
    now["past_common_retention_30m"] = np.divide(common_n, before.n_dealers.to_numpy(), out=np.full(len(requested), np.nan), where=before.n_dealers.to_numpy() > 0)
    changed_n = ((conditions[ni] != conditions[pi]) & common).sum(axis=1)
    now["common_condition_changed_fraction_30m"] = np.divide(changed_n, common_n, out=np.full(len(requested), np.nan), where=common_n > 0)
    return slots.loc[slots.time.isin(requested)].reset_index(drop=True), now


# SETUP LOGIC: quantity局部诊断接口；不改变原始报价或TRACE口径。
def quantity_evidence(rows):
    """Same-event quantity contrasts, without equating units to TRACE size."""
    # CORE LOGIC: STEP 1 — 同事件同quantity取distinct spread摘要
    # Input: 同timestamp t, q=2 的spreads=[60,60,64].
    # Output: cells=(t,q=2,lo=60,hi=64,center=62,count=2).
    # Trick: 中心用unique后的median；重复60不把中心拉成60。
    finite = rows.loc[rows["s"].notna()]
    cells = finite.groupby([KEYS[-1], "qtag"], observed=True).agg(
        lo=("s", "min"), hi=("s", "max"), center=("s", "median"), count=("s", "nunique"),
    ).reset_index()
    # Median of distinct spreads, never frequency-weighted by repeated rows.
    cells["center"] = finite.groupby([KEYS[-1], "qtag"], observed=True)["s"].agg(lambda v: np.median(v.unique())).to_numpy()
    # CORE LOGIC: STEP 2 — 只在完整且至少两档正quantity的事件比较条件
    # Input: t1={(q2,s60),(q3,s64)}; t2={(q0,s60),(q3,s64)}，均完整.
    # Output: eligible[t1]=True,t2=False；positive仅t1两档；counts[t1]=2。
    # Trick: 这是条件效应可评估集合，不是删除t2原始事件。
    eligible = rows.groupby(KEYS[-1])["qkind"].agg(lambda v: v.eq("Positive").all())
    complete = rows.groupby(KEYS[-1])["s"].agg(lambda v: v.notna().all())
    positive = cells.loc[cells[KEYS[-1]].isin(eligible.index[eligible & complete])]
    counts = positive.groupby(KEYS[-1])["qtag"].nunique()
    positive = positive.loc[positive[KEYS[-1]].isin(counts.index[counts.ge(2)])]
    # Remove any event where a positive size still maps to multiple spreads.
    # CORE LOGIC: STEP 3 — 每档单价时去共同事件中心
    # Input: t1:q2→60,q3→64；t2:q2→[60,62],q3→64.
    # Output: unique仅t1；within_event_residual=[−2,+2]，t2无可辨认的一档一价对照。
    # Trick: transform(median)广播同事件中心62给两个quantity行。
    unique = positive.groupby(KEYS[-1])["count"].max().eq(1)
    unique = positive.loc[positive[KEYS[-1]].isin(unique.index[unique])].copy()
    unique["within_event_residual"] = unique["center"] - unique.groupby(KEYS[-1])["center"].transform("median")
    return cells, unique


# SETUP LOGIC: research_figure：函数接口；计算阶段见内部CORE标记。
def research_figure(result, dealer, bond, side, day, view, qtag, age_min, canvas=None):
    """One shared case; all four decisions in a 2x2 dashboard by default."""
    # CORE LOGIC: STEP 1 — 固定绘图案例的同券同侧同日母集
    # Input: raw=[(A,bid,03-02,D1),(A,bid,03-02,D2),(B,bid,03-02,D1)], dealer=D1,bond=A.
    # Output: peers_raw含A的D1/D2；rows只含A的D1；e为D1对应事件，own为D1 slots。
    # Trick: dealer只筛展示自己的行，不能筛掉对照同行。
    raw = result["raw"]
    mask = raw["cusip"].eq(bond) & raw["side"].eq(side) & raw[KEYS[-1]].dt.normalize().eq(day)
    peers_raw = raw.loc[mask]
    rows = peers_raw.loc[peers_raw["firm"].eq(dealer)]
    g = result["events"]
    e = g.loc[g["firm"].eq(dealer) & g["cusip"].eq(bond) & g["side"].eq(side) & g["day"].eq(day)]
    slots, f = result["slots"], result["features"]
    own = slots.loc[slots["firm"].eq(dealer)] if len(slots) else slots
    # PLOTTING LOGIC: 构造标题、同屏子图和可选局部canvas；不重新拟合或改变处理规则。
    summary = (f"Selected dealer: {len(rows):,} raw rows | {len(e):,} events | "
               f"{int(e['candidate_count'].gt(1).sum()):,} multi-spread | "
               f"{int(e['repeats'].sum()):,} repeats | {int(e['bad'].sum()):,} nonfinite spreads")
    if view == ALL_VIEWS:
        fig = Figure(figsize=(20, 16), facecolor="white")
        fig.suptitle(f"{result['issuer']} | {bond} | {side} | {day:%Y-%m-%d} ET\n"
                     f"Dealer {dealer} | quantity {qtag} | age hypothesis {age_min} min", fontsize=18, y=0.993)
        fig.text(0.5, 0.947, summary, ha="center", fontsize=12)
        for k, question in enumerate(VIEWS):
            research_figure(result, dealer, bond, side, day, question, qtag, age_min,
                            canvas=(fig, 0.5 * (k % 2), 0.485 if k < 2 else 0.03, 0.5, 0.445))
        fig.text(0.5, 0.008, "Raw records retained. Candidate ranges are not uncertainty bands; no quote identity or executable price is inferred.",
                 ha="center", fontsize=10, color="#555555")
        return fig
    compact = canvas is not None
    fig, x0, y0, width, height = canvas if compact else (Figure(figsize=(16, 9.5), facecolor="white"), 0, 0, 1, 1)
    grid = fig.add_gridspec(2, 2, left=x0 + width * 0.085, right=x0 + width * 0.97,
                           top=y0 + height * (0.85 if compact else 0.77),
                           bottom=y0 + height * (0.33 if compact else 0.27), hspace=0.68, wspace=0.30)
    a, b, c = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    if compact:
        fig.text(x0 + width * 0.5, y0 + height * 0.98, view, ha="center", va="top", fontsize=15, fontweight="bold")
    else:
        fig.suptitle(f"{result['issuer']} | {bond} | {side}\n{view} | dealer {dealer} | {day:%Y-%m-%d} ET", fontsize=16, y=0.98)
        fig.text(0.5, 0.86, summary, ha="center", fontsize=11)
    if view == VIEWS[0]:
        a.scatter(rows[KEYS[-1]], rows["s"], s=13, alpha=0.5, color="#277F8E", label="Raw candidates")
        a.scatter(e[KEYS[-1]], e["center"], s=16, color="#C9563D", marker="x", label="Distinct-spread median")
        a.set_title("Raw levels and their scalar summary")
        b.scatter(e[KEYS[-1]], e["gap"], s=17, color="#8560A5")
        b.set_title("Same-time candidate gap"); b.set_ylabel("max - min (bps)")
        for col, label, color in [("aggregate_delta_30m", "Original aggregate", "#BBBBBB"),
                                  ("common_dealer_delta_30m", "Common dealers", "#277F8E"),
                                  ("center_delta_30m", "Condition guarded", "#8560A5")]:
            c.plot(f.index, f[col], color=color, label=label, drawstyle="steps-post", lw=1.2)
        c.set_title("Fixed 30min change: common vs full roster"); c.set_ylabel("bps / 30 minutes")
        # CORE LOGIC: STEP 2 — 检查摘要是否真是原候选
        # Input: 事件1候选[60,64]→center62,nearest_gap2；事件2候选[60]→center60,gap0.
        # Output: finite.sum=2,unquoted=1；一半事件的median不是可直接引用的quote。
        # Trick: 阈值1e-9仅容忍浮点误差，不是价格清洗阈值。
        finite = e["center"].notna()
        unquoted = int(e.loc[finite, "center_nearest_gap"].gt(1e-9).sum())
        # PLOTTING LOGIC: 把已有诊断数值写进图注，显示量不改变特征。
        # CORE LOGIC: STEP 3 — 将候选和组成统计写入诊断说明
        # Input: unquoted=1,finite=[True,True],gap=[4,0],condition_changed=[False,True],history_break=[True,False]；common_n=[1,3],retention=[1,.5],changed_fraction=[0,1]。
        # Output: 图注显示未报价median 1/2，median gap=2.00bps，变化1/1，median common N=2.0，median retention=75.0%，mean changed=50.0%。
        # Trick: retention用median，changed fraction用mean；两者分母含义不能混称。
        notes = [f"OBSERVED: median is not a quoted candidate at {unquoted}/{int(finite.sum())} events; median candidate gap {e['gap'].median():.2f} bps.",
                 f"CHECK: {int(e['condition_changed'].sum())}/{int((~e['history_break']).sum())} comparable transitions change quantity support or count. Bounds are order statistics, not tracked streams.",
                 f"USE: common dealers median N={f['n_common_dealers_30m'].median():.1f}; current retention={f['common_retention_30m'].median():.1%}; changed conditions={f['common_condition_changed_fraction_30m'].mean():.1%}. Difference is composition sensitivity, not a causal decomposition. Adjacent-event changes remain in events."]
    # PLOTTING LOGIC: quantity图布局和原始散点；核心条件对照已在quantity_evidence计算。
    elif view == VIEWS[1]:
        tags = sorted(rows["qtag"].unique())
        if len(tags) <= 6:
            colors = ["#277F8E", "#C9563D", "#8560A5", "#BA8C27", "#3076B5", "#65754C"]
            for tag, color in zip(tags, colors):
                chosen = rows.loc[rows["qtag"].eq(tag)]
                label = "Zero (unspecified)" if tag == "Zero" else tag
                a.scatter(chosen[KEYS[-1]], chosen["s"], s=18, color=color, alpha=0.65, label=label)
        else:
            a.scatter(rows[KEYS[-1]], rows["s"], s=12, color="#BBBBBB", alpha=0.5, label=f"All {len(tags)} conditions")
            selected = rows.loc[rows["qtag"].eq(qtag)]
            a.scatter(selected[KEYS[-1]], selected["s"], s=20, color="#277F8E", label=f"Selected: {qtag}")
        a.set_title("Levels by raw quantity condition (units unknown)")
        # CACHEING LOGIC: 同dealer/bond/side/day的quantity诊断只算一次，换选项重画。
        quantity_key = (dealer, bond, side, day)
        diagnostics = result.setdefault("quantity_evidence_cache", {})
        if quantity_key not in diagnostics:
            diagnostics[quantity_key] = quantity_evidence(rows)
        cells, contrasts = diagnostics[quantity_key]
        # PLOTTING LOGIC: 只高亮所选quantity的已算cells/contrasts，空对照显示Unassessed。
        selected_cells = cells.loc[cells["qtag"].eq(qtag)]
        b.scatter(selected_cells[KEYS[-1]], selected_cells["count"], s=20, color="#8560A5")
        b.set_title(f"Spreads per event: {qtag}"); b.set_ylabel("count"); b.set_ylim(bottom=0)
        b.yaxis.set_major_locator(MaxNLocator(integer=True))
        selected_contrasts = contrasts.loc[contrasts["qtag"].eq(qtag)]
        c.set_title(f"Within-event contrast: {qtag}"); c.set_ylabel("bps vs event median")
        if selected_contrasts.empty:
            c.text(0.5, 0.5, "No eligible positive-size contrasts\nNot evidence of zero size effect", transform=c.transAxes, ha="center", va="center", fontsize=10)
            c.set_yticks([])
        else:
            c.scatter(selected_contrasts[KEYS[-1]], selected_contrasts["within_event_residual"], s=20, color="#277F8E")
            c.axhline(0, color="#999999", lw=0.7)
        # CORE LOGIC: STEP 4 — 统计选中quantity仍多价的事件
        # Input: selected_cells.count=[1,2,3],qtag=q=2.0；selected_contrasts有1条已确认完整对照。
        # Output: 图注显示2/3 finite q=2.0 events remain multi-spread，1 eligible contrast；不是2/3 raw rows。
        # Trick: count已经是同事件distinct spread数，重复行不会加大此发生比例。
        notes = [f"OBSERVED: {int(selected_cells['count'].gt(1).sum())}/{len(selected_cells)} finite {qtag} events remain multi-spread; {len(selected_contrasts)} eligible contrasts.",
                 "ELIGIBLE: complete events, >=2 positive raw quantities, one spread per quantity. Zero/missing size cannot distinguish economic conditions.",
                 "USE: conditional summaries only where mapping repeats with coverage. Else retain pooled candidate ambiguity; no forced size curve."]
    # PLOTTING LOGIC: 绘已算age规则摘要及覆盖，再单独计算覆盖代价。
    elif view == VIEWS[2]:
        for col, label, color in [("center_equal", "Dealer equal / no expiry", "#555555"),
                                  ("center_decay", f"Half-life {age_min}m", "#277F8E"),
                                  ("center_max_age", f"Max age {age_min}m", "#C9563D")]:
            a.plot(f.index, f[col], label=label, color=color, lw=1.4, drawstyle="steps-post")
        a.set_title("How age rules change the aggregate")
        if len(own):
            b.plot(own["time"], own["message_age_min"], color="#277F8E", label="Message age")
            b.plot(own["time"], own["spread_set_change_age_min"], color="#8560A5", label="Set-change age (known)")
        b.set_title("Selected dealer: refresh vs change"); b.set_ylabel("minutes")
        max_age = own[["message_age_min", "spread_set_change_age_min"]].max().max() if len(own) else 0
        b.set_ylim(0, max(1, max_age * 1.05) if pd.notna(max_age) else 1)
        for col, label, color, style in [("n_dealers", "Complete", "#555555", "-"),
                                         ("n_fresh_dealers", f"Within {age_min}m", "#C9563D", "-"),
                                         ("decay_effective_dealers", "Decay effective N", "#277F8E", "--"),
                                         ("n_common_dealers_30m", "Common with t-30min", "#8560A5", ":")]:
            c.plot(f.index, f[col], color=color, label=label, ls=style, drawstyle="steps-post")
        c.set_title("Coverage and weight concentration"); c.set_ylabel("dealers / effective N")
        c.yaxis.set_major_locator(MaxNLocator(integer=True)); c.set_ylim(bottom=0)
        # CORE LOGIC: STEP 5 — 量化年龄规则覆盖代价
        # Input: equal=[60,61],max_age=[60,NaN],decay=[60,62]; own.change_age_unknown=[True,False].
        # Output: covered=[True,True]（sum=2）,lost=1,unknown=1,max_effect=1 bps。
        # Trick: 覆盖损失与数值改变量分开；缺失max_age不填成0报价。
        covered = f["center_equal"].notna()
        lost = int((covered & f["center_max_age"].isna()).sum())
        unknown = int(own["change_age_unknown"].sum()) if len(own) else 0
        max_effect = (f["center_decay"] - f["center_equal"]).abs().max()
        # PLOTTING LOGIC: 把年龄/覆盖结果写成说明；后续分支只绘已有同行支持结果。
        notes = [f"OBSERVED: max-age loses {lost}/{int(covered.sum())} covered grid times; max decay-vs-equal shift {max_effect:.2f} bps. Change age unknown: {unknown}/{len(own)} slots.",
                 "CHECK: low message age can coexist with an unchanged level. Changing dealer contributions can move the aggregate without market repricing.",
                 "USE: retain both ages, coverage and effective N=(sum w)^2/sum(w^2). These age rules remain hypotheses, not calibrated expiry."]
    else:
        other = peers_raw.loc[peers_raw["firm"].ne(dealer)]
        a.scatter(other[KEYS[-1]], other["s"], s=9, color="#AAAAAA", alpha=0.35, label="Other dealers: raw")
        a.scatter(rows[KEYS[-1]], rows["s"], s=12, color="#277F8E", alpha=0.5, label="Selected dealer: raw")
        for col, label, color in [("center_equal", "Dealer equal", "#333333"), ("center_candidate_clip", "Candidate clip", "#C9563D"), ("center_dealer_downweight", "Dealer downweight", "#8560A5")]:
            a.plot(f.index, f[col], label=label, color=color, lw=1.3, drawstyle="steps-post")
        a.set_title("Raw range and alternative aggregates")
        # CORE LOGIC: STEP 6 — 区分有同行支持和未评估
        # Input: n_peer_supported=[0,2]；第一个时点只存在baseline fallback。
        # Output: assessed=[False,True]；第一时点不纳入已评估零影响。
        assessed = f["n_peer_supported"].gt(0)
        # PLOTTING LOGIC: 设置影响图坐标与分支；数值差分见下一块。
        b.set_title("Impact only where peers support a test"); b.set_ylabel("bps vs baseline")
        if assessed.any():
            for col, label, color in [("center_candidate_clip", "Candidate clip", "#C9563D"), ("center_dealer_downweight", "Dealer downweight", "#8560A5")]:
                # CORE LOGIC: STEP 7 — 仅在有支持的同时间样本上算变化
                # Input: center_equal=[60,62],center_candidate_clip=[60,61],assessed=[False,True]，三者index相同。
                # Output: delta=[NaN,-1] bps；第一行fallback相等不伪装成实测0。
                # Trick: Series相减按时间index对齐；where把未评估行留缺口。
                delta = (f[col] - f["center_equal"]).where(assessed)
                # PLOTTING LOGIC: 画已经过资格mask的变化曲线；不连接未知区间。
                b.plot(f.index, delta, color=color, label=label, marker=".", ms=3, drawstyle="steps-post")
            b.axhline(0, color="#888888", lw=0.6)
        else:
            b.text(0.5, 0.5, "NOT ASSESSED\nInsufficient fresh peers\nBaseline fallback is not a measured zero", transform=b.transAxes, ha="center", va="center", fontsize=9.5)
            b.set_yticks([])
        # CORE LOGIC: STEP 8 — 选完整且有足够同行的残差母集
        # Input: own rows=(complete=True,n_peers=3),(True,2),(False,4)；MIN_PEERS=3。
        # Output: comparable为前两行；supported仅第一行；不完整第三行不作为可比中心。
        comparable = own.loc[own["complete"]] if len(own) else own
        supported = comparable.loc[comparable["n_peers"].ge(MIN_PEERS)] if len(comparable) else comparable
        # PLOTTING LOGIC: 绘原候选相对同行的区间，保留无支持缺口。
        c.set_title("Selected candidates vs other dealers"); c.set_ylabel("bps vs peer median")
        if len(supported):
            # NaN entries preserve gaps instead of connecting unsupported times.
            valid_ref = own["peer_center"].notna() & own["complete"]
            c.vlines(own.loc[valid_ref, "time"], own.loc[valid_ref, "lo"] - own.loc[valid_ref, "peer_center"],
                     own.loc[valid_ref, "hi"] - own.loc[valid_ref, "peer_center"], color="#277F8E", alpha=0.5, label="Candidate min-max")
            c.scatter(own["time"], own["peer_residual"], s=14, color="#277F8E", label="Center residual")
            c.plot(own["time"], own["peer_radius"], color="#C9563D", ls="--", lw=1, label="Illustrative threshold")
            c.plot(own["time"], -own["peer_radius"], color="#C9563D", ls="--", lw=1)
        else:
            c.text(0.5, 0.5, f"Need >= {MIN_PEERS} fresh other dealers\nNo candidate residual can be assessed", transform=c.transAxes, ha="center", va="center", fontsize=9.5)
            c.set_yticks([])
        # CORE LOGIC: STEP 9 — 区分候选变化与中心变化
        # Input: assessed=[True,True],n_clipped_dealers=[1,1],n_changed_centers=[0,1].
        # Output: affected=2,centers=1；两次clip仅一次改变聚合中心。
        # Trick: 只在真实supported状态计数，fallback不算已评估的零影响。
        affected = int(f.loc[assessed, "n_clipped_dealers"].gt(0).sum())
        centers = int(f.loc[assessed, "n_changed_centers"].gt(0).sum())
        # PLOTTING LOGIC: 支持/阈值图注与时间轴格式；四个问题共用原结果。
        notes = [f"SUPPORT: selected dealer {len(supported)}/{len(comparable)} complete slots; any dealer assessed {int(assessed.sum())}/{int(f['n_dealers'].gt(0).sum())} covered times. Candidate/center changes: {affected}/{centers} of {int(assessed.sum())} assessed times.",
                 f"RULE: >=3 other dealers aged <= {age_min}m; radius=max(10bps, 4 x 1.4826 x peer MAD). Unsupported comparisons stay unassessed.",
                 "USE: separate no support, assessed/no change, and assessed/change. Peer agreement is not truth; clipping candidates can leave their median unchanged."]
    a.set_ylabel("Benchmark spread (bps)")
    if len(rows):
        left, right = rows[KEYS[-1]].min(), rows[KEYS[-1]].max()
        if len(f):
            left, right = min(left, f.index.min()), max(right, f.index.max())
        pad = max((right - left) / 50, pd.Timedelta(minutes=1))
        for ax in [a, b, c]:
            ax.set_xlim(max(day, left - pad), min(day + pd.DateOffset(days=1), right + pad))
    for ax in [a, b, c]:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6, tz=day.tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=day.tz))
        ax.set_xlabel("ET", fontsize=9)
        ax.set_title(ax.get_title(), fontsize=11)
        ax.tick_params(labelsize=9)
        ax.grid(alpha=0.15)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=7.5 if compact else 8, loc="best", ncol=min(3 if ax is a else 2, len(handles)))
    for y, note in zip([0.235, 0.15, 0.065] if compact else [0.18, 0.12, 0.06], notes):
        fig.text(x0 + width * 0.085, y0 + height * y, fill(note, width=105 if compact else 175), fontsize=9.5 if compact else 10, va="top")
    return fig

# %% [markdown]
# ## Controls and outputs
# Representative cases lead the issuer list; all issuers remain selectable.
# Bond/side/day selectors favour multi-spread support, then event count.
# Apply combines selector changes. Display dealer, quantity and layout reuse the applied state.
# Freeze fixed cases selects random / typical / measured impact cases from three local probes per observed unit.
# Default All four is one 2x2 dashboard. Detail layouts preserve the case; Save all 4 PNG always exports all sections.
# Changing layout preserves the case. Quantity can be selected from every observed raw condition.
# The Age setting is both the decay half-life and the max-age hypothesis; for Influence it limits peer age.
# `step3_result['events']`: event-time median, gap, condition flags, guarded delta and observed changes in (t-30m, t].
# `step3_features`: same-day levels, coverage, ages, ambiguity, guarded change and common-dealer fixed-30min diagnostics.
# The fixed-horizon change is missing when dealer roster / quantity support / candidate count differs at the endpoints.
# Common-dealer change remains observable after a condition change, with N, retention and changed-condition fraction.
# Its difference from the original aggregate is a composition sensitivity check, not a causal decomposition.
# Unknown change age stays NaN. Absent or incomplete current quotes do not fall back to a previous valid quote.
# Full-day plots / case rankings are retrospective displays; neither is a feature or a claim of predictive success.

# %% 3. Choose a case; compare rules; save one PNG
# UI LOGIC: 旧控件解绑与初始化；未Apply选项不是结果。
if "step3_controls" in globals():
    for control in step3_controls:
        control.unobserve_all("value")
    step3_save.on_click(save_step3, remove=True)
    if "step3_apply" in globals():
        step3_apply.on_click(refresh_step3, remove=True)
    if "step3_freeze" in globals():
        step3_freeze.on_click(freeze_step3_cases, remove=True)
    step3_dashboard.close()
step3_issuer = widgets.Dropdown(options=issuer_choices(), description="Issuer:", layout=widgets.Layout(width="850px"))
step3_view = widgets.Dropdown(options=[ALL_VIEWS] + VIEWS, value=ALL_VIEWS, description="Layout:")
step3_dealer = widgets.Dropdown(description="Dealer:", layout=widgets.Layout(width="340px"))
step3_bond, step3_side, step3_day = widgets.Dropdown(description="Bond:"), widgets.Dropdown(description="Side:"), widgets.Dropdown(description="ET day:")
step3_quantity = widgets.Dropdown(description="Quantity:", layout=widgets.Layout(width="340px"))
step3_age = widgets.Dropdown(options=[10, 30, 60], value=DEFAULT_AGE_MIN, description="Age (min):")
step3_apply = widgets.Button(description="Apply / Refresh", icon="refresh")
step3_freeze = widgets.Button(description="Freeze fixed cases", icon="list")
step3_case = widgets.Dropdown(options=[("Issuer drilldown", None)], description="Case:", layout=widgets.Layout(width="1000px"))
step3_save = widgets.Button(description="Save all 4 PNG", icon="download", disabled=True)
step3_status, step3_case_status = widgets.HTML(), widgets.HTML()
step3_image = widgets.Image(format="png", layout=widgets.Layout(width="100%", max_width="1500px"))
step3_result, step3_features, step3_figure, step3_busy, step3_cache_key = None, pd.DataFrame(), None, False, None


# CACHEING LOGIC: 缓存键包括issuer/bond/side/day/age；dealer/quantity/layout不进入计算键。
def step3_key():
    return (step3_issuer.value, step3_bond.value, step3_side.value, step3_day.value, step3_age.value)


# CACHEING LOGIC: 仅切已有共享事件表，保留issuer子表用于反复绘图。
def step3_issuer_result():
    global step3_result
    if step3_result is None or step3_result["issuer"] != step3_issuer.value:
        labels = step3_prepared["events"].cusip.map(cusip_issuer).astype("string").fillna("[Missing issuer]")
        raw = step3_raw.loc[issuer_labels.eq(step3_issuer.value)]
        valid = raw[KEYS].notna().all(axis=1)
        for key in SERIES:
            valid &= raw[key].astype("string").str.strip().ne("").fillna(False)
        step3_result = dict(raw=raw, events=step3_prepared["events"].loc[labels.eq(step3_issuer.value)],
                            unkeyed=int((~valid).sum()), issuer=step3_issuer.value, quantity_evidence_cache={})


# NAVIGATION LOGIC: 同步导航选项；不创建事件或运行as-of。
def sync_step3_selectors(reset=False):
    """Cheap event-table navigation; never constructs events or as-of features."""
    step3_issuer_result()
    eligible = step3_result["events"].assign(multi=lambda x: x.candidate_count.gt(1))
    # Select the bond/side/day first, so changing display dealer cannot change
    # the query universe or force another as-of calculation.
    for box, column in [(step3_bond, "cusip"), (step3_side, "side"), (step3_day, "day"), (step3_dealer, "firm")]:
        ranked = eligible.groupby(column, observed=True)["multi"].agg(["sum", "size"]).sort_values(["sum", "size"], ascending=False, kind="stable")
        old, options = box.value, list(ranked.index)
        box.options = [(f"{v:%Y-%m-%d}" if column == "day" else str(v), v) for v in options]
        box.value = old if not reset and old in options else (options[0] if options else None)
        eligible = eligible.loc[eligible[column].eq(box.value)]
    raw = step3_result["raw"]
    if len(eligible):
        own = raw.loc[raw.firm.eq(step3_dealer.value) & raw.cusip.eq(step3_bond.value) & raw.side.eq(step3_side.value) & raw[KEYS[-1]].dt.normalize().eq(step3_day.value)]
        tags = own.qtag.value_counts().index.tolist()
        options = [t for t in tags if t.startswith("q=")] + [t for t in tags if not t.startswith("q=")]
        old = step3_quantity.value
        step3_quantity.options = options
        step3_quantity.value = old if old in options and not reset else (options[0] if options else None)
    else:
        step3_quantity.options = []
    return eligible


# PLOTTING LOGIC: 仅绘已Apply的缓存结果；未Apply的选项不会冒充结果。
def draw_step3():
    global step3_figure, step3_features
    step3_quantity.layout.display = "" if step3_view.value in [ALL_VIEWS, VIEWS[1]] else "none"
    step3_age.layout.display = "" if step3_view.value in [ALL_VIEWS] + VIEWS[2:] else "none"
    if step3_key() != step3_cache_key or step3_result is None or "features" not in step3_result:
        step3_features = pd.DataFrame()
        step3_image.value = b""
        step3_save.disabled = True
        step3_status.value = "Selection pending. Set bond / side / ET day / age, then Apply / Refresh. Dealer, quantity and layout only redraw an applied case."
        return
    step3_features = step3_result["features"].assign(cusip=step3_bond.value, side=step3_side.value)
    step3_figure = research_figure(step3_result, step3_dealer.value, step3_bond.value, step3_side.value, step3_day.value, step3_view.value, step3_quantity.value, step3_age.value)
    with BytesIO() as buffer:
        step3_figure.savefig(buffer, format="png", dpi=110, facecolor="white")
        step3_image.value = buffer.getvalue()
    step3_save.disabled = False
    step3_status.value = f"Shared event table; cached bond/side/day state. As-of + detail calculation {step3_result['asof_seconds']:.3f}s. All four sections save together; raw source stays in bcq_df."


# UI LOGIC: 控件回调busy guard，结束后恢复响应。
def selection_changed_step3(change=None):
    global step3_busy
    if step3_busy:
        return
    step3_busy = True
    try:
        sync_step3_selectors(reset=change is not None and change["owner"] is step3_issuer)
        draw_step3()
    finally:
        step3_busy = False


# UI LOGIC: Apply回调、空案例展示和异常后busy恢复。
def refresh_step3(change=None):
    global step3_result, step3_features, step3_figure, step3_busy, step3_cache_key
    if step3_busy:
        return
    step3_busy = True
    step3_apply.disabled, step3_save.disabled = True, True
    step3_image.value = b""
    try:
        eligible = sync_step3_selectors()
        if eligible.empty:
            step3_features = pd.DataFrame()
            step3_figure = Figure(figsize=(12, 5), facecolor="white")
            ax = step3_figure.subplots(); ax.axis("off")
            ax.text(.5, .5, f"No keyed events for this selection.\nUnkeyed rows: {step3_result['unkeyed']:,}", ha="center", va="center")
            with BytesIO() as buffer:
                step3_figure.savefig(buffer, format="png", dpi=110)
                step3_image.value = buffer.getvalue()
            step3_status.value = "No keyed events; no numeric aggregation was run."
            return
        # CACHEING LOGIC: 命中同bond/side/day/age时复用已有slot结果。
        key = step3_key()
        if key not in step3_case_cache:
            step3_status.value = "Building this bond / side / ET day state once..."
            started = perf_counter()
            # CORE LOGIC: STEP 1 — 仅构建本次bond/side/day查询网格
            # Input: A/bid当日报价首10:01、末10:12；GRID=5min.
            # Output: times=[10:01,10:05,10:10,10:12]；只对本案例调用asof_features。
            # Trick: union补首末真实时间；该网格不是全天覆盖分母。
            e = step3_result["events"]
            history = e.loc[e.cusip.eq(step3_bond.value) & e.side.eq(step3_side.value) & e.day.eq(step3_day.value)]
            today = history[KEYS[-1]]
            times = pd.date_range(today.min().ceil(GRID), today.max().floor(GRID), freq=GRID).union(pd.DatetimeIndex([today.min(), today.max()]))
            slots, features = asof_features(history, times, step3_age.value)
            # CACHEING LOGIC: 保存局部计算及时长；随后draw只重画，不重算事件。
            step3_case_cache[key] = dict(slots=slots, features=features, asof_seconds=perf_counter() - started)
        step3_result.update(step3_case_cache[key])
        step3_cache_key = key
        draw_step3()
    finally:
        step3_apply.disabled = False
        step3_busy = False


# NAVIGATION LOGIC: 用户显式冻结时建立群体与案例；已冻结名单保留。
def freeze_step3_cases(_=None):
    global step3_population, step3_manifest, step3_busy
    if step3_busy:
        return
    step3_busy = True
    step3_freeze.disabled = True
    try:
        from quote_quality_population import population_tables, fixed_case_manifest, case_options
        step3_case_status.value = "Freezing cases from observed bond/side/days; three local rule probes per unit..."
        if step3_population is None:
            step3_population = population_tables(data_ig, bcq_df, event_cache=step3_prepared)
        if step3_manifest is None:
            step3_manifest = fixed_case_manifest(step3_population)
        step3_case.options = [("Issuer drilldown", None)] + case_options(step3_manifest)
        counts = step3_manifest.selection.value_counts()
        impacts = step3_population["impact_table"]
        step3_case_status.value = (f"Fixed cases: random {int(counts.get('Random', 0))}, typical {int(counts.get('Typical', 0))}, measured high impact {int(counts.get('High impact', 0))}. "
                                  f"Impact selection used {len(impacts):,} observed bond/side/days and {int(impacts.impact_queries.sum()):,} first/middle/last local queries. Cases and probes do not estimate full-day or market occurrence rates.")
    finally:
        step3_freeze.disabled = False
        step3_busy = False


# NAVIGATION LOGIC: 应用固定名单中的同一案例和显示选项。
def choose_fixed_step3(change=None):
    global step3_busy
    if step3_busy or step3_case.value is None:
        return
    row = step3_manifest.loc[step3_manifest.case_id.eq(step3_case.value)].iloc[0]
    step3_busy = True
    try:
        source_issuer = bcq_df.loc[bcq_df.cusip.eq(row.cusip), "ISSUER"].astype("string").fillna("[Missing issuer]").iloc[0]
        step3_issuer.value = source_issuer
        sync_step3_selectors(reset=True)
        for box, value in [(step3_bond, row.cusip), (step3_side, row.side), (step3_day, row.day), (step3_dealer, row.firm)]:
            if value not in [v for _, v in box.options]:
                raise ValueError("Fixed case is absent from this loaded issuer selection")
            box.value = value
            sync_step3_selectors()
    finally:
        step3_busy = False
    refresh_step3()


# FILE IO LOGIC: 导出同屏完整PNG；不打印大表或改变feature。
def save_step3(change=None):
    if step3_busy or step3_save.disabled or step3_key() != step3_cache_key:
        return
    folder = Path("outputs/quote_quality_step3"); folder.mkdir(parents=True, exist_ok=True)
    name = "_".join(str(v) for v in [step3_issuer.value, step3_bond.value, step3_dealer.value, step3_side.value, step3_day.value, ALL_VIEWS, step3_quantity.value, step3_age.value])
    name = "".join(c if c.isalnum() else "_" for c in name)[:220]
    path = folder / f"{name}.png"
    export = research_figure(step3_result, step3_dealer.value, step3_bond.value, step3_side.value, step3_day.value, ALL_VIEWS, step3_quantity.value, step3_age.value)
    export.savefig(path, dpi=180, facecolor="white")
    step3_status.value = f"Saved: {path}"

# UI LOGIC: 绑定回调并展示；保持原Apply行为。
step3_controls = [step3_issuer, step3_view, step3_dealer, step3_bond, step3_side, step3_day, step3_quantity, step3_age, step3_case]
for box in [step3_issuer, step3_dealer, step3_bond, step3_side, step3_day, step3_age]:
    box.observe(selection_changed_step3, names="value")
for box in [step3_view, step3_quantity]:
    box.observe(selection_changed_step3, names="value")
step3_case.observe(choose_fixed_step3, names="value")
step3_apply.on_click(refresh_step3)
step3_freeze.on_click(freeze_step3_cases)
step3_save.on_click(save_step3)
step3_dashboard = widgets.VBox([step3_issuer, widgets.HBox([step3_freeze, step3_apply]), step3_case, step3_case_status, step3_view,
                               widgets.HBox([step3_bond, step3_side, step3_day]), widgets.HBox([step3_dealer, step3_quantity, step3_age, step3_save]), step3_status, step3_image])
display(step3_dashboard)
refresh_step3()
