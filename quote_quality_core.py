# SETUP LOGIC: 模块说明及导入；导入本身不加载原始数据或运行训练
"""Shared calculations for the Step 4/5 research notebooks; no data loading or plotting on import."""
import numpy as np
import pandas as pd
from time import perf_counter

# SETUP LOGIC: 固定字段/阈值/特征字典声明；没有观察值筛选或数值估计
KEYS = ["firm", "cusip", "side", "quote_timestamp_ET"]
SERIES = KEYS[:3]
HISTORY_GAP_MIN = 60
LOOKBACK_MIN = 30
DEFAULT_AGE_MIN = 30
MIN_PEERS = 3
CLIP_FLOOR_BPS = 10.0
MAD_MULTIPLIER = 4.0

# SETUP LOGIC: to_ny_datetime 的函数签名与既有 docstring；不改动说明字符串
def to_ny_datetime(series):
    # CORE LOGIC: STEP 1 — 解析并统一纽约时区
    # Input: ['2026-03-02T15:00:00Z','bad']
    # Output: [2026-03-02 10:00:00-05:00,NaT]
    # Trick: coerce 将不可解析时间标为 NaT；naive 时间按纽约本地解释，aware 时间转换而非重新贴标签
    values = pd.to_datetime(series, errors="coerce", format="mixed")
    if values.dt.tz is None:
        return values.dt.tz_localize("America/New_York")
    return values.dt.tz_convert("America/New_York")


# SETUP LOGIC: event_history 的函数签名与既有 docstring；不改动说明字符串
def event_history(raw, progress=None, keep_raw=True):
    """Distinct sets and prefix-only changes; optional progress(stage, done, total, detail)."""
    # PROGRESS LOGIC: 阶段计时及可选进度通知，不改变事件值
    started = perf_counter()
    if progress is not None:
        progress('events', None, None, f'Normalizing {len(raw):,} quote rows')
    # Duplicate semantics still use the complete source row, while event work needs
    # only six columns. Extra source metadata no longer travels through groupby.
    # CORE LOGIC: STEP 1 — 保留全原始行重复语义，识别有键事件
    # Input: raw两行均为(A,X,bid,10:00,100,2)且元数据相同，另行firm=' '
    # Output: repeat=[False,True,False]；valid=[True,True,False]
    # Trick: duplicated检查全source行；strip仅检查键是否空，原键不改写
    repeats = raw.duplicated()
    q = raw.copy() if keep_raw else raw[KEYS + ['spread', 'quantity']].copy()
    valid = q[KEYS].notna().all(axis=1)
    for key in SERIES:
        valid &= q[key].astype("string").str.strip().ne("").fillna(False)
    q["repeat"] = repeats
    # CORE LOGIC: STEP 2 — spread数值化和quantity类别化
    # Input: spread=[0,-5,inf,'bad']；quantity=[0,2,None,-1]
    # Output: s=[0,-5,NaN,NaN]；qkind=['Zero','Positive','Missing','Other']
    # Trick: 仅nonfinite spread变NaN；零/负spread和未知quantity仍保留
    q["s"] = pd.to_numeric(q["spread"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    q["q"] = pd.to_numeric(q["quantity"], errors="coerce")
    q["qkind"] = np.select([q["quantity"].isna(), q["q"].eq(0), np.isfinite(q["q"]) & q["q"].gt(0)],
                            ["Missing", "Zero", "Positive"], default="Other")
    # CORE LOGIC: STEP 3 — 构造可比较quantity标签
    # Input: quantity为object列=[2,0,None,-1]
    # Output: qtag=['q=2.0','Zero','Missing','Other:-1']
    # Trick: 正quantity用float repr固定标签；未知和非法数量不会假装为0
    q["qtag"] = q["qkind"].astype(str)
    positive = q["qkind"].eq("Positive")
    q.loc[positive, "qtag"] = q.loc[positive, "q"].map(lambda v: "q=" + repr(float(v)))
    other = q["qkind"].eq("Other")
    q.loc[other, "qtag"] = "Other:" + q.loc[other, "quantity"].astype(str)
    # CORE LOGIC: STEP 4 — 保留不完整候选及有效键原行
    # Input: s=[100,NaN]；qtag=['q=2.0','Missing']；valid=[True,True]
    # Output: pair=[(100,'q=2.0'),('Nonfinite','Missing')]；usable仍有2行
    # Trick: Nonfinite哨兵保留原事件不完整来源，不能先删坏行再聚合
    q["bad"] = q["s"].isna()
    q["pair"] = list(zip(q["s"].fillna("Nonfinite"), q["qtag"]))
    usable = q.loc[valid]
    # PROGRESS LOGIC: 记录归一化耗时，报告接下来聚合的原行数
    normalized = perf_counter()
    if progress is not None:
        progress('events', None, None, f'Aggregating {len(usable):,} keyed quote rows')
    # CORE LOGIC: STEP 5 — 同dealer/bond/side/known-time聚合distinct候选
    # Input: A/X/bid/10:00行：(100,2),(110,2),(100,2),(NaN,None)
    # Output: rows=4,repeats=1,bad=1,spread_set=(100,110),quantity_set=('Missing','q=2.0'),candidate_count=2
    # Trick: distinct集合不以重复行加权；stable排序使各series时间连续
    g = usable.groupby(KEYS, observed=True, sort=False).agg(
        rows=("s", "size"), repeats=("repeat", "sum"), bad=("bad", "sum"),
        spread_set=("s", lambda v: tuple(sorted(v.dropna().unique()))),
        quantity_set=("qtag", lambda v: tuple(sorted(v.unique()))),
        pair_set=("pair", lambda v: frozenset(v)),
    ).reset_index().sort_values(SERIES + [KEYS[-1]], kind="stable").reset_index(drop=True)
    g["candidate_count"] = g["spread_set"].map(len)
    # PROGRESS LOGIC: 记录聚合耗时，通知开始历史计算
    aggregated = perf_counter()
    if progress is not None:
        progress('events', None, None, f'Building history for {len(g):,} events')
    # CORE LOGIC: STEP 6 — 候选范围、候选中位数和完整状态
    # Input: spread_set=(100,110),bad=0；另事件spread_set=(),bad=1
    # Output: lo=100,hi=110,center=105,gap=10,nearest_gap=5,complete=True；空事件center=NaN,complete=False
    # Trick: 偶数候选中位数可以不是原quote；只要同事件有bad就不完整
    for col, fn in [("lo", min), ("hi", max), ("center", np.median)]:
        g[col] = g["spread_set"].map(lambda v: float(fn(v)) if v else np.nan)
    g["gap"] = g["hi"] - g["lo"]
    g["center_nearest_gap"] = [min(abs(s - c) for s in ss) if ss else np.nan
                               for ss, c in zip(g["spread_set"], g["center"])]
    g["complete"] = g["bad"].eq(0) & g["candidate_count"].gt(0)
    g["day"] = g[KEYS[-1]].dt.normalize()
    # CORE LOGIC: STEP 7 — 取同dealer/bond/side/ET日的上一事件
    # Input: A/X/bid当天10:00完整100→10:10完整105；次日10:00完整106
    # Output: 10:10 interval_min=10,continuous=True；次日interval=NaN,history_break=True
    # Trick: groupby shift按原index对齐；跨日、首条、不完整或>60分钟都断历史
    group = g.groupby(SERIES + ["day"], sort=False, observed=True)
    prev = group[["spread_set", "quantity_set", "pair_set", "complete", "candidate_count", "center", KEYS[-1]]].shift()
    g["interval_min"] = (g[KEYS[-1]] - prev[KEYS[-1]]).dt.total_seconds() / 60
    continuous = g["interval_min"].le(HISTORY_GAP_MIN) & g["complete"] & prev["complete"].eq(True)
    g["history_break"] = ~continuous
    # CORE LOGIC: STEP 8 — 区分刷新、spread变化与条件变化
    # Input: 上一候选(100),quantity=('q=2.0',),center=100；当前(105),quantity=('q=3.0',),center=105
    # Output: spread_changed=True,pair_refresh=False,condition_changed=True,center_delta=5,guarded_delta=NaN
    # Trick: guarded仅在数量集合及candidate_count不变时评估；signed delta保留方向
    g["spread_changed"] = continuous & g["spread_set"].ne(prev["spread_set"])
    g["pair_refresh"] = continuous & g["pair_set"].eq(prev["pair_set"])
    g["condition_changed"] = continuous & (g["quantity_set"].ne(prev["quantity_set"]) | g["candidate_count"].ne(prev["candidate_count"]))
    g["center_delta"] = (g["center"] - prev["center"]).where(continuous)
    g["guarded_delta"] = g["center_delta"].where(~g["condition_changed"])
    # Bounds are order statistics, not tracked quote identities.
    # CORE LOGIC: STEP 9 — 计算条件稳定边界变化及可观察ABA
    # Input: 同日完整候选100→105→100，数量均2且间隔10分钟
    # Output: 第三条lo_delta=-5,hi_delta=-5,observed_aba=True
    # Trick: 边界是order statistic而非跟踪某条quote；ABA还要求两次连续历史
    for bound in ["lo", "hi"]:
        g[f"{bound}_delta"] = group[bound].diff().where(continuous & ~g["condition_changed"])
    two_back = group["pair_set"].shift(2)
    g["observed_aba"] = continuous & group["history_break"].shift().eq(False) & g["pair_set"].eq(two_back) & g["pair_set"].ne(prev["pair_set"])
    # Break on day/series changes, incomplete observations, or a long gap.
    # CORE LOGIC: STEP 10 — 历史分段内前缀change age
    # Input: 10:00首条100→10:05变105→10:10刷新105
    # Output: history_start均10:00；last_change=[NaT,10:05,10:05]；change_age=[NaN,0,5]
    # Trick: cumsum隔开断点，ffill仅向后传播已观察变化；首条未知不是age=0
    segment = g["history_break"].cumsum()
    g["history_start"] = g.groupby(segment)[KEYS[-1]].transform("first")
    g["last_change"] = g[KEYS[-1]].where(g["spread_changed"]).groupby(segment).ffill()
    g["change_age_min"] = (g[KEYS[-1]] - g["last_change"]).dt.total_seconds() / 60
    g["change_age_unknown"] = g["last_change"].isna()
    # Rows are already sorted by series and time, so each history segment is a
    # contiguous slice. Read arrays once and write once instead of constructing
    # pandas indexers/Series for every (often singleton) segment.
    # CORE LOGIC: STEP 11 — 一次提取纳秒数组及连续segment边界
    # Input: history_break=[True,False,True]；spread_changed=[False,True,False]
    # Output: starts=[0,2],stops=[2,3],changed=[0,1,0],changes_30m初始[0,0,0]
    # Trick: as_unit('ns')避免微秒精度将未来事件提前；segment切片无需逐组g.loc
    event_times = g[KEYS[-1]].array.as_unit("ns").asi8
    changed = g["spread_changed"].to_numpy(dtype=np.int64)
    starts = np.flatnonzero(g["history_break"].to_numpy())
    stops = np.r_[starts[1:], len(g)]
    changes_30m = np.zeros(len(g), dtype=np.int64)
    # PROGRESS LOGIC: 显示历史segment计数起点，不改变特征
    if progress is not None:
        progress('events', 0, len(starts), 'Counting changes within history segments')
    # CORE LOGIC: STEP 12 — 每segment计算严格左开30分钟变化计数
    # Input: 同segment事件10:00,10:05,10:35；changed=[0,1,1]
    # Output: 10:35 changes_30m=1，恰好10:05被排除；若末条10:34:59.999999999则计数2
    # Trick: searchsorted(side='right')排除恰好t-30m；prefix差保留当前变化；singleton直接维持0
    for completed, (start, stop) in enumerate(zip(starts, stops), 1):
        # A singleton starts with history_break=True, hence spread_changed=False.
        if stop-start > 1:
            times = event_times[start:stop]
            counts = np.r_[0, changed[start:stop].cumsum()]
            left = np.searchsorted(times, times - LOOKBACK_MIN * 60 * 10**9, side="right")
            changes_30m[start:stop] = counts[1:] - counts[left]
        # PROGRESS LOGIC: 按segment数量节流报告进度
        if progress is not None and (completed % max(1, (len(starts) + 99) // 100) == 0 or completed == len(starts)):
            progress('events', completed, len(starts), f'History segments {completed:,}/{len(starts):,}')
    # CORE LOGIC: STEP 13 — 一次回写并返回事件/无键计数
    # Input: g为X在10:00/10:05/10:35的3事件，changes_30m=[0,1,1]；valid=[True,True,True,False]；keep_raw=False；started/normalized/aggregated/当前时钟=0/1/2/3秒
    # Output: result.events这3行changes_30m=[0,1,1]，unkeyed=1，timings={normalize_s:1,aggregate_s:1,history_s:1}，没有raw键
    # Trick: 按已排序位置整列写回；timings是观察元数据，不参与特征
    g["changes_30m"] = changes_30m
    result = {"events": g, "unkeyed": int((~valid).sum()),
              "timings": {'normalize_s': normalized-started,
                          'aggregate_s': aggregated-normalized,
                          'history_s': perf_counter()-aggregated}}
    if keep_raw:
        result['raw'] = q
    return result


# SETUP LOGIC: prepare_quote_events 的函数签名与既有 docstring；不改动说明字符串
def prepare_quote_events(quotes, progress=None):
    """Reusable narrow event table. Rebuild explicitly after changing source quotes.

    It retains incomplete latest messages; an incomplete message is observed state,
    not an absent quote. No hidden cache can silently reuse a changed DataFrame.
    """
    # CORE LOGIC: STEP 1 — 复用窄事件接口，不保留宽raw副本
    # Input: raw=[{firm:'A',cusip:'X',side:'bid',quote_timestamp_ET:'2026-03-02 10:00-05:00',spread:NaN,quantity:2}]
    # Output: events的一行为firm=A,cusip=X,side=bid,time=10:00,rows=1,bad=1,candidate_count=0,center=NaN,complete=False,changes_30m=0；unkeyed=0；没有raw键，timings只记录运行耗时
    # Trick: 不完整最新消息仍是状态，不能视作无quote
    return event_history(quotes, progress=progress, keep_raw=False)




# SETUP LOGIC: representative_issuers 的函数签名与既有 docstring；不改动说明字符串
def representative_issuers(quotes):
    """Navigation only: observed case support, never a clean/dirty issuer label."""
    # CORE LOGIC: STEP 1 — 统计可导航issuer的同事件多价证据
    # Input: IBM有10:00候选100/110及10:01候选105；ACME有20个单价事件
    # Output: IBM scores.events=2,multi=1,gap=10；ACME events=20,multi=0,gap=0
    # Trick: 同事件范围不等于时间变化；这些分数仅选导航，不是clean/dirty标签
    x = quotes.dropna(subset=KEYS + ['ISSUER']).copy()
    x['s'] = pd.to_numeric(x['spread'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    g = x.groupby(['ISSUER'] + KEYS, observed=True)['s'].agg(['min', 'max', 'nunique']).reset_index()
    g['multi'], g['gap'] = g['nunique'].gt(1), g['max'] - g['min']
    scores = g.groupby('ISSUER', observed=True).agg(events=('multi','size'), multi=('multi','sum'), gap=('gap','max'))
    # CORE LOGIC: STEP 2 — 优先既审阅名称及有足够事件的高指标issuer
    # Input: scores=[{ISSUER:'IBM',events:2,multi:1,gap:10},{ISSUER:'ACME',events:20,multi:0,gap:0}]
    # Output: names=['IBM','ACME']；dropdown=[('[Reviewed case] IBM','IBM'),('[High multi] ACME','ACME')]
    # Trick: Reviewed优先标签不被setdefault覆盖；其余issuer仍可下钻
    promoted = {}
    for prefix in ['HPS CORPORATE','EXPAND ENERGY','EASTERN GAS','SKY GROUP','DUKE ENERGY','IBM','COMCAST','MITSUBISHI UFJ','KKR GROUP']:
        found = sorted(n for n in scores.index if str(n).upper().startswith(prefix))
        if found: promoted[found[0]] = 'Reviewed case'
    for metric in ['multi','gap','events']:
        for name in scores.loc[scores.events.ge(20)].sort_values(metric,ascending=False).index[:2]:
            promoted.setdefault(name, 'High ' + metric)
    names = list(promoted) + [n for n in sorted(scores.index) if n not in promoted]
    return [(f'[{promoted[n]}] {n}' if n in promoted else str(n), n) for n in names]


# SETUP LOGIC: pair_snapshots 的函数签名与既有 docstring；不改动说明字符串
def pair_snapshots(events, times, age_min=30, sync_min=1, allow_exact=True):
    """One bond; latest bid/ask per dealer, same ET day. Cartesian ranges stay set-valued."""
    # CORE LOGIC: STEP 1 — 确认单bond输入并去重排序查询时间
    # Input: events只含X；times=[10:05,10:00,10:05]，age_min=30,sync_min=1
    # Output: times=[10:00,10:05]；若events含X和Y则ValueError
    # Trick: 去重只减少计算，后续调用方仍可映射回所有原trade rows
    if age_min <= 0 or sync_min < 0: raise ValueError('age_min must be positive and sync_min nonnegative')
    times = pd.DatetimeIndex(times).sort_values().unique()
    if events['cusip'].nunique() > 1:
        raise ValueError('pair_snapshots expects one bond')
    rows = []
    fields = ['firm','side',KEYS[-1],'day','complete','center','lo','hi','candidate_count','pair_set']
    # CORE LOGIC: STEP 2 — 每dealer/side向后取最新消息且限定同ET日
    # Input: A bid 09:59=100、10:04=NaN；查询10:05
    # Output: 最新bid取10:04完整=False,age=1；不会退回09:59完整100
    # Trick: merge_asof先选择最新，再检查完整；allow_exact=False排除相同known-time
    for (firm, side), h in events.groupby(['firm','side'], observed=True):
        if side not in ['bid','ask']: continue
        joined = pd.merge_asof(pd.DataFrame({'time':times}), h[fields].sort_values(KEYS[-1]),
                               left_on='time', right_on=KEYS[-1], direction='backward', allow_exact_matches=allow_exact)
        joined = joined.loc[joined[KEYS[-1]].notna() & joined.time.dt.normalize().eq(joined.day)].copy()
        joined['age'] = (joined.time - joined[KEYS[-1]]).dt.total_seconds()/60
        rows.append(joined)
    # CORE LOGIC: STEP 3 — 按查询time/dealer外连接bid和ask
    # Input: slots三行(time='10:05',firm='A',side='bid',center=100)、(10:05,B,bid,105)、(10:05,B,ask,102)
    # Output: p两行(time=10:05,firm=A,center_bid=100,center_ask=NaN)、(time=10:05,firm=B,center_bid=105,center_ask=102)
    # Trick: outer保留单边状态；不同dealer不能相互拼成一对
    fields += ['time','age']
    slots = pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=fields)
    b = slots.loc[slots.side.eq('bid')].drop(columns='side')
    a = slots.loc[slots.side.eq('ask')].drop(columns='side')
    p = b.merge(a,on=['time','firm'],how='outer',suffixes=('_bid','_ask'))
    # CORE LOGIC: STEP 4 — 定义完整双边、age和同步距离
    # Input: A bid10:04完整100、ask10:03完整102；B缺ask
    # Output: A both=True,complete=True,max_age=2,time_gap=1,same_time=False；B both=False
    # Trick: time_gap取绝对分钟；锁价/负spread不参与是否完整的过滤
    p['both'] = p[f'{KEYS[-1]}_bid'].notna() & p[f'{KEYS[-1]}_ask'].notna()
    p['complete'] = p.complete_bid.eq(True) & p.complete_ask.eq(True)
    p['max_age'] = p[['age_bid','age_ask']].max(axis=1).where(p.both)
    p['time_gap'] = (pd.to_datetime(p[f'{KEYS[-1]}_bid']) - pd.to_datetime(p[f'{KEYS[-1]}_ask'])).abs().dt.total_seconds()/60
    p['same_time'] = p.both & p['time_gap'].eq(0)
    # CORE LOGIC: STEP 5 — 为未评估pair保持NaN及0支持
    # Input: B仅bid100没有ask
    # Output: B gap_low/high/center/mid均NaN，positive_matches=0,size_state='Unassessed'
    # Trick: 空支持不能用0填价格差；计数0表示确实无匹配
    for name in ['gap_low','gap_high','gap_center','mid','mid_range','matched_gap_low','matched_gap_high','matched_gap','matched_mid']:
        p[name] = np.nan
    p['positive_matches'] = 0
    p['size_state'] = 'Unassessed'
    # CORE LOGIC: STEP 6 — signed bid减ask及候选笛卡尔边界
    # Input: bid候选100/110，ask候选105/115，center为105/110
    # Output: gap_low=-15,gap_high=5,gap_center=-5,mid=107.5,mid_range=10
    # Trick: 有符号gap用于crossing；range计算保留所有不同候选而不删负值
    p['gap_low'] = (p.lo_bid-p.hi_ask).where(p.complete)
    p['gap_high'] = (p.hi_bid-p.lo_ask).where(p.complete)
    p['gap_center'] = (p.center_bid-p.center_ask).where(p.complete)
    p['mid'] = ((p.center_bid+p.center_ask)/2).where(p.complete)
    p['mid_range'] = ((p.hi_bid-p.lo_bid+p.hi_ask-p.lo_ask)/2).where(p.complete)
    # Repeated snapshots reuse the same candidate-set match, instead of expanding pairs.
    # CACHEING LOGIC: 以不可变bid/ask候选集合为key复用相同size匹配，避免反复展开
    cache = {}
    matched_columns = ['size_state','positive_matches','matched_gap_low','matched_gap_high','matched_gap','matched_mid']
    matches = []
    # CORE LOGIC: STEP 7 — 将每个完整pair候选按raw quantity标签分组
    # Input: bid={(100,'q=2.0'),(110,'q=2.0'),(120,'Zero')}；ask={(105,'q=2.0')}
    # Output: bid['q=2.0']=[100,110],bid['Zero']=[120]；ask['q=2.0']=[105]
    # Trick: 仅按标签收集，零/未知quantity依旧保留在原candidate集合
    for r in p.loc[p.complete].itertuples():
        key = (r.pair_set_bid, r.pair_set_ask)
        if key not in cache:
            bid, ask = {}, {}
            for pairs, dest in [(r.pair_set_bid,bid),(r.pair_set_ask,ask)]:
                for spread, tag in pairs: dest.setdefault(tag,[]).append(spread)
            # CORE LOGIC: STEP 8 — 标记共享正quantity与匹配状态
            # Input: bid标签['q=2.0','Zero']，ask标签['q=2.0']；另bid q=2,ask q=3
            # Output: 第一对shared=['q=2.0'],state='Shared positive'；第二对'Positive unmatched'
            # Trick: Unknown/mixed与正数量不相配分开；没有shared不等于原quantity全部未知
            shared = [tag for tag in set(bid)&set(ask) if tag.startswith('q=')]
            only_positive = all(tag.startswith('q=') for tag in list(bid)+list(ask))
            state = 'Shared positive' if shared else ('Positive unmatched' if only_positive else 'Unknown / mixed')
            values = [state,0,np.nan,np.nan,np.nan,np.nan]
            # CORE LOGIC: STEP 9 — 每个共享quantity内求边界、中位数gap和mid
            # Input: bid q=2候选100/110；ask q=2候选105；仅共享q=2
            # Output: positive_matches=1,matched_gap_low=-5,high=5,gap=0,mid=105
            # Trick: 先每个q内保留多价、取median，再在共享q间equal mean；quantity不作权重
            if shared:
                values = [state,len(shared), min(min(bid[q])-max(ask[q]) for q in shared),
                    max(max(bid[q])-min(ask[q]) for q in shared),
                    np.mean([np.median(bid[q])-np.median(ask[q]) for q in shared]),
                    np.mean([(np.median(bid[q])+np.median(ask[q]))/2 for q in shared])]
            # CACHEING LOGIC: 写入并读取候选集合匹配memo；同状态重复grid不会再次计算
            cache[key] = values
        matches.append(cache[key])
    # CORE LOGIC: STEP 10 — 按原pair index回写匹配结果
    # Input: 完整pair index=[1,3]；matching结果gap=[0,2]
    # Output: p.loc[[1,3],'matched_gap']=[0,2]；其余不完整行仍NaN
    # Trick: 构造matches DataFrame带相同index，防pandas自动对齐错位
    if matches:
        p.loc[p.complete,matched_columns] = pd.DataFrame(matches,index=p.index[p.complete],columns=matched_columns)
    # CORE LOGIC: STEP 11 — 分类原集合及同quantity集合crossing
    # Input: gap范围[0,5]、[-5,-1]、[-5,5]；另不完整pair
    # Output: cross依次None,All,Some,Unassessed；matched无正quantity同样Unassessed
    # Trick: low>=0含locked=0；high<0才All；不是看到负mid就自动删除
    for suffix,lo,hi,eligible in [('', 'gap_low','gap_high',p.complete),
                                  ('_matched','matched_gap_low','matched_gap_high',p.positive_matches.gt(0))]:
        p['cross'+suffix] = 'Unassessed'
        p.loc[eligible,'cross'+suffix] = np.select([p.loc[eligible,lo].ge(0),p.loc[eligible,hi].lt(0)],['None','All'],default='Some')
    # CORE LOGIC: STEP 12 — 稳定排序并派生政策eligibility
    # Input: 10:05 A完整,max_age=2,time_gap=1,positive_matches=1；age=30,sync=1
    # Output: A fresh_pair=True,size_time_pair=True，pair返回按time/firm排序
    # Trick: 只派生mask，原候选和crossing结论不删除
    p = p.sort_values(['time','firm'],kind='stable').reset_index(drop=True)
    return pair_policy_masks(p, age_min, sync_min)


# SETUP LOGIC: pair_policy_masks 的函数签名与既有 docstring；不改动说明字符串
def pair_policy_masks(pairs, age_min=30, sync_min=1):
    """Derive age/sync eligibility from cached all-dealer state, without as-of work."""
    # CORE LOGIC: STEP 1 — 用已有pair状态生成fresh及size-time masks
    # Input: A complete=True,max_age=30,time_gap=1,positive_matches=1；B age=31
    # Output: age_min=30,sync_min=1时A两个mask都True，B两个mask都False
    # Trick: <=包含恰好阈值；copy保护原缓存，切阈值不重做asof
    if age_min <= 0 or sync_min < 0:
        raise ValueError('age_min must be positive and sync_min nonnegative')
    p = pairs.copy()
    p['fresh_pair'] = p.complete & p.max_age.le(age_min)
    p['size_time_pair'] = p.fresh_pair & p.time_gap.le(sync_min) & p.positive_matches.gt(0)
    return p


# SETUP LOGIC: pair_policy_comparison 的函数签名与既有 docstring；不改动说明字符串
def pair_policy_comparison(pairs):
    """A→B isolates slot selection; B→C isolates candidate matching on identical slots.

    Slots are all-dealer bond/day grid observations, not independent trades. B is
    eligible for size/time matching but uses its original candidate sets. C uses
    shared positive raw-quantity candidates on exactly those same B slots.
    """
    # CORE LOGIC: STEP 1 — A/B/C在原槽位与同B槽位匹配上作比较
    # Input: pairs两行：firm=A,fresh=True,size_time=False,gap=-10,mid=95,cross=All；firm=B,fresh=True,size_time=True,gap=4,mid=103,cross=Some,matched_gap=2,matched_mid=99,cross_matched=None
    # Output: A n_slots=2,n_dealers=2,gap_mean=-3,mid_mean=99,cross(None/Some/All)=(0,.5,.5)；B n=1,dealers=1,gap=4,mid=103,cross=(0,1,0)；C n=1,dealers=1,gap=2,mid=99,cross=(1,0,0)
    # Trick: A→B差7是选择效应，B→C差-2是quantity匹配作用；slot不是独立trade
    rows = []
    for label, mask, cross, gap, mid in [
        ('A fresh / original', pairs.fresh_pair, 'cross', 'gap_center', 'mid'),
        ('B match slots / original', pairs.size_time_pair, 'cross', 'gap_center', 'mid'),
        ('C same slots / matched', pairs.size_time_pair, 'cross_matched', 'matched_gap', 'matched_mid')]:
        z = pairs.loc[mask]
        rows.append(dict(policy=label, n_slots=len(z), n_dealers=z.firm.nunique(),
            gap_mean_bps=z[gap].mean(), mid_mean_bps=z[mid].mean(),
            **{state.lower()+'_fraction': z[cross].eq(state).mean() for state in ['None', 'Some', 'All']}))
    # CORE LOGIC: STEP 2 — 以policy为index返回小比较表
    # Input: rows三完整records，字段顺序(policy,n_slots,n_dealers,gap_mean_bps,mid_mean_bps,none_fraction,some_fraction,all_fraction)：('A fresh / original',2,2,-3,99,0,.5,.5)，('B match slots / original',1,1,4,103,0,1,0)，('C same slots / matched',1,1,2,99,1,0,0)
    # Output: index=['A fresh / original','B match slots / original','C same slots / matched']，其余7列依次为[2,2,-3,99,0,.5,.5]、[1,1,4,103,0,1,0]、[1,1,2,99,1,0,0]
    # Trick: B与C由同一size_time_pair mask选取，计数必相同
    return pd.DataFrame(rows).set_index('policy')


# SETUP LOGIC: pair_extreme_sources 的函数签名与既有 docstring；不改动说明字符串
def pair_extreme_sources(raw, pairs, limit_each=2):
    """Locate both signed gap extremes in original bid/ask spread and quantity rows.

    Repeated grid observations of one candidate-state pair count once here. Boundary
    flags identify the raw candidates creating the extrema; other raw candidates in
    those messages remain visible. This diagnoses provenance without deleting rows.
    """
    # SETUP LOGIC: 固定原始来源诊断列名
    columns = ['extreme', 'time', 'firm', 'side', 'quote_timestamp_ET', 'spread',
               'quantity', 'boundary_candidate', 'gap_bound_bps']
    # CORE LOGIC: STEP 1 — 限定单bond并去除重复grid同消息pair
    # Input: 同firm pair在10:05和10:06都引用bid10:04/ask10:03；limit_each=2
    # Output: distinct仅保留该消息pair一次；limit_each=0返回同列空表
    # Trick: 这里只去重诊断观察，不删除quote或改变feature权重
    if 'cusip' in raw and raw.cusip.nunique() > 1:
        raise ValueError('pair_extreme_sources expects raw rows for one bond')
    if limit_each < 1:
        return pd.DataFrame(columns=columns)
    distinct = pairs.loc[pairs.fresh_pair].drop_duplicates(
        ['firm', 'quote_timestamp_ET_bid', 'quote_timestamp_ET_ask'])
    # CORE LOGIC: STEP 2 — 分别选择最负/最正gap边界及其原side消息
    # Input: A gap_low=-15,high=5；B gap_low=-2,high=10；limit_each=1
    # Output: Low选A，bid取lo/ask取hi；High选B，bid取hi/ask取lo
    # Trick: signed两端分开查，不只看异常负值；同事件其它候选也会进入来源表
    sources = []
    for label, field, ascending, bounds in [
        ('Low gap boundary', 'gap_low', True, {'bid': 'lo', 'ask': 'hi'}),
        ('High gap boundary', 'gap_high', False, {'bid': 'hi', 'ask': 'lo'})]:
        selected = distinct.sort_values(field, ascending=ascending, kind='stable').head(limit_each)
        for row in selected.itertuples():
            for side in ['bid', 'ask']:
                stamp = getattr(row, 'quote_timestamp_ET_'+side)
                z = raw.loc[raw.firm.eq(row.firm) & raw.side.eq(side) & raw.quote_timestamp_ET.eq(stamp)]
                for source in z.itertuples():
                    # CORE LOGIC: STEP 3 — 标记真正形成边界的原spread/quantity候选
                    # Input: 选A Low；bid候选100(q=2)、110(q=3)，ask候选105(q=2)、115(q=3)
                    # Output: 所有4原行输出；bid100和ask115 boundary_candidate=True,gap_bound=-15
                    # Trick: 数值化仅用于边界比较，输出保留原spread/quantity；无自动删除
                    value = pd.to_numeric(pd.Series([source.spread]), errors='coerce').iloc[0]
                    sources.append(dict(extreme=label, time=row.time, firm=row.firm, side=side,
                        quote_timestamp_ET=stamp, spread=source.spread, quantity=source.quantity,
                        boundary_candidate=bool(np.isfinite(value) and value == getattr(row, bounds[side]+'_'+side)),
                        gap_bound_bps=getattr(row, field)))
    # CORE LOGIC: STEP 4 — 按固定schema返回原始来源小表
    # Input: sources=[{extreme:'Low gap boundary',time:'2026-03-02 10:05-05:00',firm:'A',side:'bid',quote_timestamp_ET:'2026-03-02 10:04-05:00',spread:100,quantity:2,boundary_candidate:True,gap_bound_bps:-15}]
    # Output: 一行DataFrame，按columns顺序为['Low gap boundary','2026-03-02 10:05-05:00','A','bid','2026-03-02 10:04-05:00',100,2,True,-15]
    # Trick: columns固定使没有来源时也得到稳定空schema
    return pd.DataFrame(sources, columns=columns)


# SETUP LOGIC: pair_features 的函数签名与既有 docstring；不改动说明字符串
def pair_features(pairs, times):
    """Dealer-equal observed gaps; unknown quantities never become matched-size pairs."""
    # SETUP LOGIC: 固定去重查询时间及pair特征schema
    index = pd.DatetimeIndex(times).sort_values().unique()
    cols = ['n_pair','n_size_time_pair','pair_gap','pair_gap_low','pair_gap_high','pair_mid',
            'pair_mid_range','pair_cross_some','pair_cross_all','pair_time_gap','pair_unknown_size',
            'size_time_gap','size_time_mid','size_time_cross_some','size_time_cross_all']
    # CORE LOGIC: STEP 1 — 初始化未知pair值及0计数
    # Input: times=[10:00,10:05]，当前没有任何pair
    # Output: 10:00/10:05两行n_pair=n_size_time_pair=0；pair_gap/low/high/mid/mid_range/cross_some/cross_all/time_gap/unknown_size和size_time_gap/mid/cross_some/cross_all均NaN
    # Trick: 0计数与未知价格差有不同含义，不能把缺价格差填0
    f = pd.DataFrame(np.nan,index=index,columns=cols)
    f[['n_pair','n_size_time_pair']] = 0
    # CORE LOGIC: STEP 2 — 用政策mask选择slots并编码crossing
    # Input: 10:05 fresh A cross=None、fresh B cross=All；第三slot不fresh
    # Output: fresh q包含A/B，cross_some=[0,0],cross_all=[0,1]
    # Trick: 一次dealer slot一票；size-time另用cross_matched，不偷偷换原样本
    for eligible,prefix,cross_field,gap_field,mid_field in [
        ('fresh_pair','pair','cross','gap_center','mid'),
        ('size_time_pair','size_time','cross_matched','matched_gap','matched_mid')]:
        q = pairs.loc[pairs[eligible]].copy()
        if q.empty: continue
        q['cross_some'] = q[cross_field].eq('Some').astype(float)
        q['cross_all'] = q[cross_field].eq('All').astype(float)
        # CORE LOGIC: STEP 3 — 各time计数并确定输出列映射
        # Input: 10:05 q有A/B，10:06 q有A
        # Output: n_pair(10:05)=2，n_pair(10:06)=1；pair_gap读取gap_center
        # Trick: counts.index和f.loc显式对齐，避免把group结果按位置写到错误查询
        g = q.groupby('time',sort=False)
        counts = g.size()
        count_column = 'n_pair' if prefix=='pair' else 'n_size_time_pair'
        f.loc[counts.index,count_column] = counts
        mapping = {prefix+'_gap':gap_field,prefix+'_mid':mid_field,
                   prefix+'_cross_some':'cross_some',prefix+'_cross_all':'cross_all'}
        # CORE LOGIC: STEP 4 — 统计无正quantity匹配、边界及时间差
        # Input: 10:05 A positive_matches=1/time_gap=0，B matches=0/time_gap=2
        # Output: pair_unknown_size=.5,pair_time_gap=1；保留gap_low/high/mid_range
        # Trick: unknown_size表示无common正quantity，包括正quantity不匹配，不能解释成全size缺失
        if prefix=='pair':
            q['unknown_size'] = q.positive_matches.eq(0).astype(float)
            g = q.groupby('time',sort=False)
            mapping.update(pair_gap_low='gap_low',pair_gap_high='gap_high',
                           pair_mid_range='mid_range',pair_unknown_size='unknown_size')
            f.loc[counts.index,'pair_time_gap'] = g.time_gap.median()
        # CORE LOGIC: STEP 5 — 输出每dealer同权的pair摘要
        # Input: 10:05 A gap=-4、B gap=2，cross_all=[0,1]
        # Output: pair_gap=-1,pair_cross_all=.5；f.index.name='time'
        # Trick: 均值按dealer行而非raw重复或quantity加权；无eligibility时间仍NaN
        for output,source in mapping.items(): f.loc[counts.index,output] = g[source].mean()
    f.index.name='time'
    return f


# SETUP LOGIC: empty_quote_features 的函数签名与既有 docstring；不改动说明字符串
def empty_quote_features(times):
    """The same zero-count/NaN schema as real as-of calculations, with no dealer work."""
    # SETUP LOGIC: 声明和真实计算相同的side输出列与查询index
    index = pd.DatetimeIndex(times).sort_values().unique()
    sides = ['center_equal','n_dealers','n_fresh_dealers','n_incomplete','center_decay',
             'center_max_age','dispersion_bps','mean_candidate_gap','multi_fraction',
             'zero_quantity_fraction','unknown_quantity_fraction','median_message_age_min',
             'median_change_age_min','unknown_change_age_fraction','center_lower','center_upper',
             'max_decay_weight_share','decay_effective_dealers','center_candidate_clip',
             'center_dealer_downweight','n_peer_supported','n_clipped_dealers','n_changed_centers']
    # CORE LOGIC: STEP 1 — 空状态仍保留整数0支持和NaN数值
    # Input: index=['2026-03-02 10:00-05:00','2026-03-02 10:05-05:00']，没有dealer状态
    # Output: 每time的bid/ask n_dealers,n_fresh_dealers,n_incomplete,n_peer_supported,n_clipped_dealers,n_changed_centers均int 0；center_equal/decay/max_age、dispersion和数量/年龄/候选范围数值均NaN
    # Trick: dtype同side_features_fast；无quote与incomplete当前消息不同，后者必须走真实路径
    counts = ['n_dealers','n_fresh_dealers','n_incomplete','n_peer_supported','n_clipped_dealers','n_changed_centers']
    pieces = []
    for side in ['bid', 'ask']:
        f = pd.DataFrame(np.nan, index=index, columns=sides)
        # Match side_features_fast's integer count dtypes as well as its values.
        for count in counts:
            f[count] = np.zeros(len(index), dtype=int)
        pieces.append(f.add_prefix('bcq_'+side+'_'))
    # CORE LOGIC: STEP 2 — 组合双side和pair空schema
    # Input: times=[10:00,10:05]；bid/ask在各time有0支持和NaN中心；empty_pairs有fresh_pair/size_time_pair列且0行
    # Output: 每time的bcq_n_pair=0,bcq_n_size_time_pair=0,bcq_has_quote=0.0；bcq_pair_gap/low/high/mid/mid_range/cross_some/cross_all/time_gap/unknown_size与bcq_size_time_gap/mid/cross_some/cross_all均NaN
    # Trick: 列名prefix与非空路径一致，便于安全整表回写
    empty_pairs = pd.DataFrame(columns=['fresh_pair', 'size_time_pair'])
    pieces.append(pair_features(empty_pairs, index).add_prefix('bcq_'))
    result = pd.concat(pieces, axis=1)
    result['bcq_has_quote'] = 0.0
    result.index.name = 'time'
    return result


# SETUP LOGIC: build_quote_features 的函数签名与既有 docstring；不改动说明字符串
def build_quote_features(quotes, queries, age_min=30, sync_min=1, allow_exact=True, progress=None, event_cache=None):
    """Preserve every query. Optional prepare_quote_events result avoids repeated event work.

    Timings and skipped no-state query counts are attached to result.attrs. Empty
    bond/day and before-first queries skip both side matrices and pair snapshots;
    latest incomplete messages still use the full path to preserve n_incomplete.
    """
    # CORE LOGIC: STEP 1 — 固定查询身份与合法age/sync
    # Input: queries row_id=[7,7]或time含NaT；age_min=0
    # Output: 抛ValueError；row_id=[7,8]且time正常才继续
    # Trick: many_to_one时间计算可以复用，但row_id必须唯一来恢复全原行
    if queries.row_id.duplicated().any() or queries[['row_id','cusip','time']].isna().any().any():
        raise ValueError('Queries require unique row_id and nonmissing cusip/time')
    if age_min <= 0 or sync_min < 0:
        raise ValueError('age_min must be positive and sync_min nonnegative')
    # SETUP LOGIC: 记录阶段时钟及缓存复用标志；建立按CUSIP查询分组
    started = perf_counter()
    cache_reused = event_cache is not None
    query_groups=queries.groupby('cusip',sort=False,observed=True)
    # PROGRESS LOGIC: 显示原始trade query数与不同查询bond数
    if progress is not None:
        progress('features', 0, query_groups.ngroups, f'{len(queries):,} trade queries across {query_groups.ngroups:,} bonds')
    # CORE LOGIC: STEP 2 — 空query立即返回，不加载或构建事件
    # Input: queries=pd.DataFrame(columns=['row_id','cusip','time'])，event_cache=None，started=0秒且当前时钟=.1秒
    # Output: 同3列0行DataFrame；attrs.quote_feature_timings={event_prepare_s:0.0,asof_s:0.0,total_s:.1,no_state_unique_queries:0,event_cache_reused:False}
    # Trick: 这是空query路径，不意味着非空query的latest incomplete可跳过
    if queries.empty:
        result = queries.copy()
        result.attrs['quote_feature_timings'] = dict(event_prepare_s=0.0, asof_s=0.0,
            total_s=perf_counter()-started, no_state_unique_queries=0, event_cache_reused=cache_reused)
        return result
    # CORE LOGIC: STEP 3 — 仅无已有事件时准备相关bond的事件
    # Input: quotes包含X/Y；queries仅X；event_cache=None
    # Output: 仅X raw传prepare_quote_events；若提供events cache则直接用其events
    # Trick: 已有event_cache避免再次遍历raw；cache正确性由caller绑定source
    if event_cache is None:
        relevant = quotes.loc[quotes.cusip.isin(queries.cusip.unique())]
        event_cache = prepare_quote_events(relevant, progress=progress)
    all_events = event_cache['events']
    # SETUP LOGIC: 事件准备计时及输出容器；按bond建立事件分组
    prepared = perf_counter()
    output=[]
    grouped=all_events.groupby('cusip',observed=True)
    skipped = 0
    # PROGRESS LOGIC: 逐bond更新feature进度；分母是queries的distinct CUSIP
    for completed,(bond,q) in enumerate(query_groups, 1):
        if progress is not None:
            progress('features', completed-1, query_groups.ngroups, f'Bond {bond}: {len(q):,} trade queries')
        # CORE LOGIC: STEP 4 — 识别同日首条前/无事件的廉价路径
        # Input: X3/2首条10:00；queries为3/2 09:59,10:00及3/3 10:00；allow_exact=True
        # Output: active=[False,True,False]，active_times=[3/2 10:00]，skipped增加2；09:59和次日query所有支持计数0、中心NaN
        # Trick: first按ET day map到查询index；latest incomplete在首条之后仍active，不能误判无quote
        events=grouped.get_group(bond) if bond in grouped.groups else all_events.iloc[:0]
        times=pd.DatetimeIndex(q.time).sort_values().unique()
        first = events.groupby('day', observed=True)[KEYS[-1]].min()
        first_at_query = pd.Series(times.normalize(), index=times).map(first)
        active = first_at_query.notna() & (first_at_query.le(times) if allow_exact else first_at_query.lt(times))
        active_times = times[active.to_numpy()]
        skipped += int((~active).sum())
        f = empty_quote_features(times)
        # CORE LOGIC: STEP 5 — 仅active times计算同bond双side及paired状态
        # Input: X的A在10:00 bid100(q=2)/ask105(q=2)；queries=[09:59,10:00]，age=30,sync=1,exact=True
        # Output: 10:00 bid/ask center_equal=100/105，n_dealers=1/1，n_pair=n_size_time_pair=1，pair_gap=size_time_gap=-5；09:59上述支持计数0、价格差NaN
        # Trick: 同一次distinct time计算覆盖重复trade queries；不改14base或target
        if len(active_times):
            pieces=[]
            for side in ['bid','ask']:
                side_f=side_features_fast(events.loc[events.side.eq(side)],active_times,age_min,allow_exact)
                pieces.append(side_f.add_prefix(f'bcq_{side}_'))
            p=pair_snapshots(events,active_times,age_min,sync_min,allow_exact)
            pieces.append(pair_features(p,active_times).add_prefix('bcq_'))
            # CORE LOGIC: STEP 6 — 把活跃行写入完整query-time schema
            # Input: f在09:59/10:00均bcq_has_quote=0、bid_n=0、bid_center=NaN；active_f@10:00 bid_n=1,ask_n=0,bid_center=100
            # Output: f@10:00 bcq_has_quote=1,bid_n=1,bid_center=100；f@09:59 bcq_has_quote=0,bid_n=0,bid_center=NaN
            # Trick: f.loc[active_times]与active_f按同DatetimeIndex/列标签对齐
            active_f=pd.concat(pieces,axis=1)
            active_f['bcq_has_quote']=(active_f.bcq_bid_n_dealers.add(active_f.bcq_ask_n_dealers).gt(0)).astype(float)
            f.loc[active_times] = active_f
        # CORE LOGIC: STEP 7 — many-to-one回接原trade查询身份
        # Input: q=[{row_id:7,cusip:'X',time:'10:00'},{row_id:8,cusip:'X',time:'10:00'}]；示例f仅列bcq_has_quote，index10:00值1
        # Output: joined=[{row_id:7,cusip:'X',time:'10:00',bcq_has_quote:1},{row_id:8,cusip:'X',time:'10:00',bcq_has_quote:1}]
        # Trick: 时间去重只省计算，不丢任何trade；validate约束每time唯一feature
        joined=q[['row_id','cusip','time']].merge(f,left_on='time',right_index=True,how='left',validate='many_to_one')
        output.append(joined)
        # PROGRESS LOGIC: 报告该bond已完成的原始trade query数量
        if progress is not None:
            progress('features', completed, query_groups.ngroups, f'Finished bond {bond}: {len(q):,} trade queries')
    # CORE LOGIC: STEP 8 — 恢复原row_id顺序并附阶段耗时
    # Input: queries顺序row_id=[8,7]；output示例两行[{row_id:7,cusip:'X',time:'10:00',bcq_has_quote:1},{row_id:8,cusip:'Y',time:'10:00',bcq_has_quote:0}]；started=0,prepared=1,当前时钟=2,skipped=1,cache_reused=True,event_cache.timings={}
    # Output: result两行依次为(8,'Y','10:00',0)、(7,'X','10:00',1)；attrs.quote_feature_timings={event_prepare_s:1,asof_s:1,total_s:2,no_state_unique_queries:1,event_cache_reused:True}
    # Trick: set_index/reindex以row_id恢复身份，不能靠concat的分组顺序
    result = pd.concat(output,ignore_index=True).set_index('row_id').reindex(queries.row_id).reset_index() if output else queries.copy()
    result.attrs['quote_feature_timings'] = dict(event_prepare_s=prepared-started,
        asof_s=perf_counter()-prepared, total_s=perf_counter()-started,
        no_state_unique_queries=skipped, event_cache_reused=cache_reused,
        **event_cache.get('timings', {}))
    return result


# SETUP LOGIC: side_features_fast 的函数签名与既有 docstring；不改动说明字符串
def side_features_fast(events, times, age_min=30, allow_exact=True):
    """Vectorized query-by-dealer summaries; same definitions as Step 3, without momentum."""
    # CORE LOGIC: STEP 1 — 建立query×dealer空矩阵和纳秒查询轴
    # Input: times=[10:05,10:00,10:05]；events仅A/B两个dealer
    # Output: n=2,d=2；各数值矩阵shape=(2,2)，present/valid全False
    # Trick: 去重排序只用于计算；as_unit('ns')防微秒舍入造成lookahead
    if age_min <= 0: raise ValueError('age_min must be positive')
    times=pd.DatetimeIndex(times).sort_values().unique()
    groups=list(events.groupby('firm',observed=True)); n,d=len(times),len(groups)
    names=['center','lo','hi','gap','count','age','change_age','zero','unknown','mid_low','mid_high']
    a={k:np.full((n,d),np.nan) for k in names}
    present=np.zeros((n,d),bool); valid=np.zeros((n,d),bool)
    query_ns=times.as_unit('ns').asi8; days=times.normalize().asi8
    # CORE LOGIC: STEP 2 — 按dealer向后取最新消息并检查ET日/完整性
    # Input: A10:00完整100、10:04不完整；query10:05
    # Output: idx指10:04，present=True,valid=False；不会退回10:00
    # Trick: searchsorted right允许exact，left拒exact；idx=-1先clamp防越界，再由present屏蔽
    for j,(_,h) in enumerate(groups):
        h=h.sort_values(KEYS[-1]); stamp=h[KEYS[-1]].array.as_unit('ns').asi8
        idx=np.searchsorted(stamp,query_ns,side='right' if allow_exact else 'left')-1
        z=h.iloc[np.maximum(idx,0)]
        present[:,j]=(idx>=0)&(z.day.array.as_unit(times.unit).asi8==days)
        valid[:,j]=present[:,j]&z.complete.to_numpy(dtype=bool)
        # CORE LOGIC: STEP 3 — 取最新候选统计与message/change年龄
        # Input: A10:00完整center105,lo100,hi110,last_change09:55；query10:05
        # Output: center=105,lo=100,hi=110,age=5,change_age=10
        # Trick: last_change NaT保持NaN；float纳秒换算只求age，asof选择已用整数ns完成
        for key,col in [('center','center'),('lo','lo'),('hi','hi'),('gap','gap'),('count','candidate_count')]:
            a[key][:,j]=z[col].to_numpy()
        a['age'][:,j]=(query_ns-stamp[np.maximum(idx,0)])/60e9
        last=z.last_change.array.as_unit('ns').asi8
        a['change_age'][:,j]=np.where(z.last_change.notna(),(query_ns-last.astype(float))/60e9,np.nan)
        # CORE LOGIC: STEP 4 — 保留数量未知状态及偶数候选两个中间值
        # Input: A候选100/110，quantity_set=('Zero','Missing')
        # Output: zero=True,unknown=True,mid_low=100,mid_high=110
        # Trick: mid_low/high允许逐候选clip后精确重构median，不能仅clip原median
        a['zero'][:,j]=z.quantity_set.map(lambda x:'Zero' in x).to_numpy()
        a['unknown'][:,j]=z.quantity_set.map(lambda x:any(t=='Missing' or t.startswith('Other:') for t in x)).to_numpy()
        for key,offset in [('mid_low',-1),('mid_high',0)]:
            values=h.spread_set.map(lambda x:x[(len(x)+offset)//2] if x else np.nan).to_numpy()
            a[key][:,j]=values[np.maximum(idx,0)]
    # CORE LOGIC: STEP 5 — 按完整valid屏蔽数值并生成fresh masks
    # Input: A latest incomplete；B完整age=30；age_min=30
    # Output: A各数值NaN；B valid=True,fresh=True；count=1
    # Trick: present与valid分开，n_incomplete保留不完整支持计数；<=含阈值边界
    for k in a: a[k][~valid]=np.nan
    count=valid.sum(axis=1); fresh=valid&(a['age']<=age_min)
    # SETUP LOGIC: mean 的函数签名与既有 docstring；不改动说明字符串
    def mean(x,mask=valid):
        # CORE LOGIC: STEP 1 — 仅支持dealer等权均值，空支持为NaN
        # Input: x=[[100,NaN],[NaN,NaN]],mask=[[True,False],[False,False]]
        # Output: [100,NaN]
        # Trick: nansum空行是0，但where mask.sum>0阻止把未知输出为0
        return np.divide(np.nansum(np.where(mask,x,np.nan),axis=1),mask.sum(axis=1),out=np.full(n,np.nan),where=mask.sum(axis=1)>0)
    # SETUP LOGIC: median 的函数签名与既有 docstring；不改动说明字符串
    def median(x):
        # CORE LOGIC: STEP 1 — 仅有finite支持的行计算中位数
        # Input: x=[[100,110],[NaN,NaN]]
        # Output: [105,NaN]
        # Trick: have先找finite支持，避免对全NaN行nanmedian发警告
        result=np.full(n,np.nan); have=np.isfinite(x).any(axis=1)
        if have.any(): result[have]=np.nanmedian(x[have],axis=1)
        return result
    # CORE LOGIC: STEP 6 — 等dealer中心和完整/新鲜/不完整计数
    # Input: A完整center100 age5；B完整center110 age35；C当前不完整
    # Output: center_equal=105,n_dealers=2,n_fresh_dealers=1,n_incomplete=1
    # Trick: 等dealer权重不使用quantity或重复候选数量
    f=pd.DataFrame(index=times)
    f['center_equal']=mean(a['center']); f['n_dealers']=count
    f['n_fresh_dealers']=fresh.sum(axis=1); f['n_incomplete']=(present&~valid).sum(axis=1)
    # CORE LOGIC: STEP 7 — 计算数值稳定half-life age权重
    # Input: Acenter100 age0；Bcenter110 age30；age_min=30
    # Output: weights=[1,.5]，center_decay=103.333333，center_max_age=105
    # Trick: 先减最小age再exp2防极旧消息共同underflow；比值等价于未平移指数
    min_age=np.min(np.where(valid,a['age'],np.inf),axis=1) if d else np.full(n,np.inf)
    weights=np.where(valid,np.exp2(-(a['age']-min_age[:,None])/age_min),0)
    weight_sum=weights.sum(axis=1)
    f['center_decay']=np.divide(np.nansum(a['center']*weights,axis=1),weight_sum,out=np.full(n,np.nan),where=weight_sum>0)
    f['center_max_age']=mean(a['center'],fresh)
    # CORE LOGIC: STEP 8 — 给出候选/数量/消息年龄可靠性摘要
    # Input: Acenter100 gap0 age5；Bcenter110 gap10 age15，各1 dealer
    # Output: dispersion=5,mean_candidate_gap=5,multi_fraction=.5,median_message_age=10
    # Trick: dispersion围绕等dealercenter；零/未知quantity率是状态feature，非过滤条件
    f['dispersion_bps']=np.sqrt(mean((a['center']-f.center_equal.to_numpy()[:,None])**2))
    f['mean_candidate_gap']=mean(a['gap']); f['multi_fraction']=mean(a['count']>1)
    f['zero_quantity_fraction']=mean(a['zero']); f['unknown_quantity_fraction']=mean(a['unknown'])
    f['median_message_age_min']=median(a['age']); f['median_change_age_min']=median(a['change_age'])
    f['unknown_change_age_fraction']=mean(~np.isfinite(a['change_age']))
    # CORE LOGIC: STEP 9 — 范围与有效age权重支持
    # Input: A lo=95 hi=105 weight1；B lo=105 hi=115 weight.5
    # Output: center_lower=100,center_upper=110,max_weight_share=2/3,effective_dealers=1.8
    # Trick: effective=(sum w)^2/sum(w^2)，原dealer数量与有效权重支持分开
    f['center_lower']=mean(a['lo']); f['center_upper']=mean(a['hi'])
    f['max_decay_weight_share']=np.divide(weights.max(axis=1) if d else np.zeros(n),weight_sum,out=np.full(n,np.nan),where=weight_sum>0)
    f['decay_effective_dealers']=np.divide(weight_sum**2,(weights**2).sum(axis=1),out=np.full(n,np.nan),where=weight_sum>0)
    # CORE LOGIC: STEP 10 — 初始化leave-one-dealer-out规则状态
    # Input: 一query的valid=[True,True,True,True]、A/B/C/D centers=[100,101,102,200]
    # Output: clipped=[[100,101,102,200]],dw=[[1,1,1,1]]；supported/clipped_any/center_changed各[[False,False,False,False]]
    # Trick: 初始化是无peer支持时的原值fallback，不代表规则已被验证
    clipped=a['center'].copy(); dw=np.where(valid,1.0,0.0)
    supported=np.zeros((n,d),bool); clipped_any=supported.copy(); center_changed=supported.copy()
    # CORE LOGIC: STEP 11 — 仅有至少3个其它fresh dealer时建立peer参考
    # Input: 被评估D=200，其它fresh中心100/101/102
    # Output: D ok=True,ref=101,MAD=1,radius=max(10,4×1.4826)=10
    # Trick: peer_mask[:,j]=False严格排除自身；不足3 peers继续保留原值
    for j in range(d):
        peer_mask=fresh.copy(); peer_mask[:,j]=False
        ok=valid[:,j]&(peer_mask.sum(axis=1)>=MIN_PEERS)
        if not ok.any(): continue
        peers=np.where(peer_mask[ok],a['center'][ok],np.nan)
        ref=np.nanmedian(peers,axis=1)
        radius=np.maximum(CLIP_FLOOR_BPS,MAD_MULTIPLIER*1.4826*np.nanmedian(abs(peers-ref[:,None]),axis=1))
        # CORE LOGIC: STEP 12 — 候选中位数clip和dealer residual降权
        # Input: D候选190/210(center200)，peer ref101,radius10
        # Output: D clipped median=111；residual99,dw=10/99；supported=True
        # Trick: 对中间两候选各clip再平均；降权以原median偏离衡量，保留完整dealer
        clipped[ok,j]=(np.clip(a['mid_low'][ok,j],ref-radius,ref+radius)+np.clip(a['mid_high'][ok,j],ref-radius,ref+radius))/2
        residual=abs(a['center'][ok,j]-ref)
        dw[ok,j]=np.minimum(1,np.divide(radius,residual,out=np.ones(len(ref)),where=residual>0))
        supported[ok,j]=True
        clipped_any[ok,j]=(a['lo'][ok,j]<ref-radius)|(a['hi'][ok,j]>ref+radius)
        center_changed[ok,j]=~np.isclose(clipped[ok,j],a['center'][ok,j],rtol=0,atol=1e-9)
    # CORE LOGIC: STEP 13 — 合成规则中心及实际支持/改变计数
    # Input: A/B/C/D均fresh且完整；center=[100,101,102,200]，clipped=[100,101,102,111]，dw=[1,1,1,10/99]，supported=[True,True,True,True]，clipped_any=center_changed=[False,False,False,True]
    # Output: center_candidate_clip=103.5，center_dealer_downweight≈104.224756，n_peer_supported=4,n_clipped_dealers=1,n_changed_centers=1
    # Trick: unsupported原值在总中心保留；changed用atol1e-9且rtol0避免浮点伪改变
    f['center_candidate_clip']=mean(clipped)
    f['center_dealer_downweight']=np.divide(np.nansum(a['center']*dw,axis=1),dw.sum(axis=1),out=np.full(n,np.nan),where=dw.sum(axis=1)>0)
    f['n_peer_supported']=supported.sum(axis=1); f['n_clipped_dealers']=clipped_any.sum(axis=1); f['n_changed_centers']=center_changed.sum(axis=1)
    f.index.name='time'
    return f


# SETUP LOGIC: 固定字段/阈值/特征字典声明；没有观察值筛选或数值估计
BASE_FEATURES = ['D_CPP_BM_SPREAD','COUPON','D_CDX_TRADE','MEAN_ISSUER_SPREAD_DEV',
                 'NUM_OF_ISSUER_TRADES_SINCE_PREV','PREV_BM_SPREAD','PREV_QUANTITY',
                 'PREV_TRADE_TYPE','QUANTITY','TRADE_TYPE','YRS_TO_MATURITY','BM_YIELD_STD',
                 'D_SHORT_TO_BM_SPREAD','PREV_BM_SPREAD_STD_GROUP_BY_TYPE']
BASE_CAT_FEATURES = ['PREV_TRADE_TYPE','TRADE_TYPE']


# SETUP LOGIC: model_versions 的函数签名与既有 docstring；不改动说明字符串
def model_versions(frame):
    """Shared anchor/target/base; replace only quote levels when comparing cleaning rules."""
    # CORE LOGIC: STEP 1 — 保持BASE14，仅构造quote相对同一anchor的levels与可靠性列
    # Input: PREV_BM_SPREAD=1.00百分比，bid center_equal=105bps,n_dealers=2
    # Output: bcq_bid_anchor_gap=5bps；Quote levels列=BASE14+双side gap/count+has_quote
    # Trick: 100只做percent→bps；signed anchor_gap是当前位置，不是时间movement
    x=frame.copy()
    levels=[]; reliability=[]
    for side in ['bid','ask']:
        prefix=f'bcq_{side}_'
        gap=prefix+'anchor_gap'
        x[gap]=x[prefix+'center_equal']-100*x.PREV_BM_SPREAD
        # CORE LOGIC: STEP 2 — 建立Base/Quote levels/Reliability的固定特征spec
        # Input: 双side levels各gap/count，reliability各9列，has_quote=1
        # Output: Base仍14列；Quote levels19列；Reliability37列
        # Trick: 所有版本共用同frame、target和anchor；spec是列名定义而非额外训练
        levels += [gap,prefix+'n_dealers']
        reliability += [prefix+k for k in ['mean_candidate_gap','dispersion_bps','median_message_age_min',
            'median_change_age_min','unknown_change_age_fraction','zero_quantity_fraction',
            'unknown_quantity_fraction','decay_effective_dealers','n_peer_supported']]
    levels += ['bcq_has_quote']
    common=BASE_FEATURES+levels+reliability
    specs={'Base':(BASE_FEATURES,{}),'Quote levels':(BASE_FEATURES+levels,{}),'Reliability':(common,{})}
    # CORE LOGIC: STEP 3 — 政策版本只替换levels；Max age同步替换覆盖计数
    # Input: PREV=1.00，bidcenter_decay=103、fresh_n=0；原center_equal105,n=2
    # Output: Age decay替换anchor_gap为3；Max age用fresh_n=0且has_quote按fresh双side
    # Trick: 只改替换feature，不删目标行、不改模型params
    for label,column in [('Age decay','center_decay'),('Max age','center_max_age'),
                         ('Candidate clip','center_candidate_clip'),('Dealer downweight','center_dealer_downweight')]:
        replacements={f'bcq_{side}_anchor_gap':x[f'bcq_{side}_{column}']-100*x.PREV_BM_SPREAD for side in ['bid','ask']}
        if label=='Max age':
            replacements.update({f'bcq_{side}_n_dealers':x[f'bcq_{side}_n_fresh_dealers'] for side in ['bid','ask']})
            replacements['bcq_has_quote']=(x.bcq_bid_n_fresh_dealers+x.bcq_ask_n_fresh_dealers).gt(0).astype(float)
        specs[label]=(common,replacements)
    # CORE LOGIC: STEP 4 — 给Reliability加13个paired增量列
    # Input: common已有BASE14+5levels+18reliability；paired字段suffix明确为n_pair,n_size_time_pair,pair_gap,pair_gap_low,pair_gap_high,pair_mid_range,pair_cross_some,pair_cross_all,pair_time_gap,pair_unknown_size,size_time_gap,size_time_cross_some,size_time_cross_all
    # Output: specs['Paired']=(common+这13个加bcq_前缀字段,{})共50列；返回的x仍是原frame copy及已算的双side anchor_gap，未替换任何target/anchor值
    # Trick: pair_mid/size_time_mid是诊断而非这13入模列；未评估值保留NaN
    paired=['bcq_'+k for k in ['n_pair','n_size_time_pair','pair_gap','pair_gap_low','pair_gap_high',
                              'pair_mid_range','pair_cross_some','pair_cross_all','pair_time_gap',
                              'pair_unknown_size','size_time_gap','size_time_cross_some','size_time_cross_all']]
    specs['Paired']= (common+paired,{})
    return x,specs


# SETUP LOGIC: chronological_split 的函数签名与既有 docstring；不改动说明字符串
def chronological_split(frame, quote_end, val_days=5, test_days=5, embargo_days=2, min_train_days=10):
    """One fixed final test block. Pilot defaults reflect the short quote file, not tuning."""
    # CORE LOGIC: STEP 1 — 按真实trade ET日和quote end确认固定实验窗口
    # Input: 35个日序号1..35；quote_end第35日；val=5,test=5,embargo=2,min_train=10
    # Output: v=25,t=30；不足22个日期时按同配置抛ValueError
    # Trick: 只用<=quote_end的日期；embargo按观察日期索引而非日历天
    if min(val_days,test_days,min_train_days) < 1 or embargo_days < 0 or pd.isna(quote_end):
        raise ValueError('Need positive train/validation/test lengths, nonnegative embargo and a quote end date')
    dates=pd.DatetimeIndex(frame.time.dt.normalize().unique()).sort_values()
    end=to_ny_datetime(pd.Series([quote_end])).iloc[0].normalize()
    dates=dates[dates<=end]
    v=len(dates)-val_days-test_days; t=len(dates)-test_days
    if v-embargo_days < min_train_days:
        raise ValueError(f'Need >= {min_train_days+embargo_days+val_days+test_days} distinct trade dates through quote end; found {len(dates)}. Set explicit research window lengths.')
    # CORE LOGIC: STEP 2 — 产生Train/Validation/Test和最终refit masks
    # Input: 日期1..35，v=25,t=30，embargo=2
    # Output: Train1..23，Validation26..30，Test31..35；24/25 Outside；refit1..28
    # Trick: 原frame每行按其ET日映射；test前29/30 embargo不进入refit
    day=frame.time.dt.normalize()
    split=pd.Series('Outside experiment',index=frame.index)
    split.loc[day.isin(dates[:v-embargo_days])]='Train'
    split.loc[day.isin(dates[v:t])]='Validation'
    split.loc[day.isin(dates[t:])]='Test'
    # Final refit uses all earlier data except the same embargo before test.
    refit=day.isin(dates[:t-embargo_days])
    return split,refit


# SETUP LOGIC: run_comparison 的函数签名与既有 docstring；不改动说明字符串
def run_comparison(frame, params, stage='Validation', selected=None, progress=None, versions=None):
    """Same target rows and BASE14; optional finite Validation versions include Base.

    None retains the eight-version comparison. Test uses its locked selected
    version and existing references, so an explicit versions list is rejected.
    """
    # SETUP LOGIC: 延迟导入训练库，导入不执行fit
    import lightgbm as lgb
    # CORE LOGIC: STEP 1 — 限制阶段及Test不能指定validation版本网格
    # Input: stage='Test',versions=['Base','Age decay']
    # Output: ValueError；stage='Validation',versions=None则允许默认8spec
    # Trick: Test沿已锁selected路径，不能在Test试选多个候选
    if stage not in ['Validation','Test']: raise ValueError('Unknown evaluation stage')
    if stage=='Test' and versions is not None:
        raise ValueError('versions is only supported for Validation; Test uses the locked selected version')
    x,specs=model_versions(frame)
    # CORE LOGIC: STEP 2 — 读取有限validation版本列表并要求非空/字符串/唯一
    # Input: versions=['Base','Age decay']；或[]、'Age decay'、['Base','Base']
    # Output: 第一组requested保持顺序；其它三组ValueError
    # Trick: 避免字符串误展开为字符；None仍保留既有8版本行为
    if versions is not None:
        try:
            requested=list(versions) if not isinstance(versions,str) else []
        except TypeError:
            raise ValueError('versions must be a nonempty list of model names including Base') from None
        if not requested or any(not isinstance(name,str) for name in requested):
            raise ValueError('versions must be a nonempty list of model names including Base')
        if len(requested)!=len(set(requested)):
            raise ValueError('versions must contain unique model names')
        # CORE LOGIC: STEP 3 — 仅保留已知且含Base的请求顺序
        # Input: requested=['Age decay','Base']；另requested=['Direction','Base']
        # Output: spec顺序Age decay→Base；未知Direction报错；缺Base也报错
        # Trick: 这里不增加新版本定义，只在已有spec字典上过滤
        unknown=[name for name in requested if name not in specs]
        if unknown:
            raise ValueError('Unknown model versions: '+', '.join(unknown))
        if 'Base' not in requested:
            raise ValueError('versions must include Base')
        specs={name:specs[name] for name in requested}
    # CORE LOGIC: STEP 4 — 锁定Test所需baseline/reference/selected
    # Input: selected='Age decay'；另selected='Reliability'
    # Output: 前者只Base/Reliability/Age decay；后者只Base/Quote levels/Reliability
    # Trick: dict.fromkeys去重保序；reference由既定版本链确定而非看Test表现
    if stage=='Test':
        if selected not in specs: raise ValueError('Choose a validation-selected version before testing')
        reference='Base' if selected in ['Base','Quote levels'] else ('Quote levels' if selected=='Reliability' else 'Reliability')
        specs={k:specs[k] for k in dict.fromkeys(['Base',reference,selected])}
    # CORE LOGIC: STEP 5 — 同样本finite target/anchor及时间先后/覆盖约束
    # Input: Train20行time早于Validation5行；D/PREV/BM有限且两组至少1 covered quote
    # Output: train/evaluate masks均保留所有符合原target口径行；无覆盖或时间重叠报错
    # Trick: 只按既有target有效性筛选，不因quote缺失删除；时间检查防训练晚于评估
    eligible=np.isfinite(x[['D_BM_SPREAD','PREV_BM_SPREAD','BM_SPREAD']]).all(axis=1)
    train=eligible & (x['split'].eq('Train') if stage=='Validation' else x['refit_train'])
    evaluate=eligible & x['split'].eq(stage)
    if train.sum()<20 or evaluate.sum()==0: raise ValueError('Insufficient common training or evaluation target rows')
    if not x.loc[train,'time'].max() < x.loc[evaluate,'time'].min(): raise ValueError('Training must precede evaluation')
    if not x.loc[train,'bcq_has_quote'].gt(0).any() or not x.loc[evaluate,'bcq_has_quote'].gt(0).any():
        raise ValueError('Quote increment is unassessed: training or evaluation has no covered trades. Inspect the split/coverage dashboard.')
    # SETUP LOGIC: 存放各版本预测和模型对象；开始按已选spec遍历
    predictions=[]; models={}
    for completed,(name,(columns,replacements)) in enumerate(specs.items()):
        # PROGRESS LOGIC: 显示当前将准备的模型和模型总数
        if progress is not None:
            progress('models', completed, len(specs), f'Preparing model {completed+1}/{len(specs)}: {name}')
        # CORE LOGIC: STEP 6 — 应用该版本替换且category域仅由Train确定
        # Input: Train TRADE_TYPE=['B','S']，Validation有新值'X'；Age decay gap=3
        # Output: z categorical域仅B/S，Validation X变NaN；数值nonfinite变NaN；只该版本level被替换
        # Trick: 不从Validation学习categories；frame copy保护其他版本共享原值
        z=x[columns].copy()
        for col,value in replacements.items(): z[col]=value
        for col in columns:
            if col in BASE_CAT_FEATURES:
                categories=pd.Index(z.loc[train,col].dropna().unique())
                z[col]=pd.Categorical(z[col],categories=categories)
            else: z[col]=pd.to_numeric(z[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
        # SETUP LOGIC: 用未改变params初始化模型与fit回调容器
        model=lgb.LGBMRegressor(**params)
        fit_kwargs={}
        # PROGRESS LOGIC: 注册iteration进度回调，不改变fit样本或目标
        if progress is not None:
            progress('fit', 0, model.n_estimators, f'{name}: training')
            # SETUP LOGIC: report_iteration 的函数签名与既有 docstring；不改动说明字符串
            def report_iteration(env):
                # PROGRESS LOGIC: 每10轮或最后一轮报告LightGBM iteration；只读env.iteration等进度字段
                done=env.iteration-env.begin_iteration+1
                total=env.end_iteration-env.begin_iteration
                if done % 10 == 0 or done == total:
                    progress('fit', done, total, f'{name}: iteration {done:,}/{total:,}')
            report_iteration.order=20
            report_iteration.before_iteration=False
            fit_kwargs['callbacks']=[report_iteration]
        # CORE LOGIC: STEP 7 — 按相同原目标D_BM_SPREAD拟合当前spec
        # Input: Train20行，目标D_BM_SPREAD=[0.01,0.02]重复10次；categorical=['PREV_TRADE_TYPE','TRADE_TYPE']
        # Output: 一个按params拟合的LGBMRegressor供本版本预测
        # Trick: 调用仅使用train mask的z/target；callbacks只报告轮数
        model.fit(z.loc[train],x.loc[train,'D_BM_SPREAD'],categorical_feature=BASE_CAT_FEATURES,**fit_kwargs)
        # PROGRESS LOGIC: 显示当前版本开始预测原evaluate行
        if progress is not None:
            progress('models', completed, len(specs), f'Predicting {stage.lower()} rows: {name}')
        # CORE LOGIC: STEP 8 — 把预测delta加固定anchor并形成bps误差
        # Input: model预测delta=.02，PREV_BM_SPREAD=1.00，BM_SPREAD=1.01
        # Output: pred_spread=1.02,error_bps=1,abs_error_bps=1；每原evaluate row_id保留
        # Trick: 模型目标和anchor仍是百分比单位，只有error×100换bps
        pred=model.predict(z.loc[evaluate])+x.loc[evaluate,'PREV_BM_SPREAD'].to_numpy()
        out=x.loc[evaluate,['row_id','time','TRADE_TYPE','QUANTITY','bcq_has_quote','bcq_n_pair','bcq_n_size_time_pair']].copy()
        out['model']=name;out['stage']=stage;out['pred_spread']=pred
        out['error_bps']=(pred-x.loc[evaluate,'BM_SPREAD'].to_numpy())*100
        out['abs_error_bps']=abs(out.error_bps)
        # SETUP LOGIC: 记录Train计数并收集本版本结果/模型
        out['train_n']=int(train.sum())
        predictions.append(out); models[name]=model
        # PROGRESS LOGIC: 报告该模型完成，不触发额外fit
        if progress is not None:
            progress('models', completed+1, len(specs), f'Finished model {completed+1}/{len(specs)}: {name}')
    # CORE LOGIC: STEP 9 — 按版本拼接原评估row的预测结果
    # Input: predictions两DataFrame仅示例一evaluate行：Base={row_id:7,time:'10:00',TRADE_TYPE:'B',QUANTITY:2,bcq_has_quote:1,bcq_n_pair:0,bcq_n_size_time_pair:0,model:'Base',stage:'Validation',pred_spread:1.02,error_bps:1,abs_error_bps:1,train_n:20}；Age decay同键行pred_spread=1.01,error_bps=abs_error_bps=0,model='Age decay'；models两值均为LGBMRegressor(n_estimators=12,random_state=2026)已fit实例
    # Output: 2行按Base→Age decay顺序，row_id都7，pred_spread=[1.02,1.01],error_bps=abs_error_bps=[1,0],train_n=[20,20]，其余明示键值不变；返回models keys=['Base','Age decay']及原两个模型实例
    # Trick: 每版本同一评估样本，结果按spec运行顺序拼接
    return pd.concat(predictions,ignore_index=True),models
