# SETUP LOGIC: 模块说明及导入；导入本身不加载原始数据或运行训练
"""Causal direction sidecar from checked narrow events; no raw quote aggregation.

Only Train/Validation rows are returned. This module does not fit models, read
targets, inspect test losses, or change the existing fourteen base features.
"""
from time import perf_counter

import numpy as np
import pandas as pd

from quote_quality_core import to_ny_datetime


# SETUP LOGIC: 固定字段/阈值/特征字典声明；没有观察值筛选或数值估计
_SIDES = ("bid", "ask")
_MOVE_SUFFIXES = (
    "mean_bps", "median_bps", "guarded_mean_bps", "up_fraction",
    "down_fraction", "flat_fraction", "common_n", "current_n", "lookback_n",
    "retention", "condition_changed_fraction", "mean_support_age_min",
    "max_support_age_min", "guarded_common_n",
)
# SETUP LOGIC: 固定字段/阈值/特征字典声明；没有观察值筛选或数值估计
_ISSUER_SUFFIXES = (
    "mean_bps", "median_bps", "up_fraction", "down_fraction", "flat_fraction",
    "other_bond_n", "common_dealer_n", "dispersion_bps",
    "mean_support_age_min", "max_support_age_min",
)
# SETUP LOGIC: 固定字段/阈值/特征字典声明；没有观察值筛选或数值估计
DIRECTION_FEATURES = [f"bcq_{s}_move_{v}" for s in _SIDES for v in _MOVE_SUFFIXES]
ISSUER_FEATURES = [f"bcq_{s}_issuer_move_{v}" for s in _SIDES for v in _ISSUER_SUFFIXES]
_MOVE_COL = {name: i for i, name in enumerate(_MOVE_SUFFIXES)}
_COUNT_SUFFIXES = {"common_n", "current_n", "lookback_n", "guarded_common_n",
                   "other_bond_n", "common_dealer_n"}
_MINUTE_NS = 60 * 10**9
_QUERY_BLOCK = 512
_NEVER = np.iinfo(np.int64).max


# SETUP LOGIC: _mean 的函数签名与既有 docstring；不改动说明字符串
def _mean(values, n):
    # CORE LOGIC: STEP 1 — 按已有支持计数求均值，空行保留NaN
    # Input: values=[[2,NaN],[NaN,NaN]],n=[1,0]
    # Output: [2,NaN]
    # Trick: nansum第二行虽是0，where=n>0仍禁止把无支持均值变0
    return np.divide(np.nansum(values, axis=1), n,
                     out=np.full(len(n), np.nan), where=n > 0)


# SETUP LOGIC: _median 的函数签名与既有 docstring；不改动说明字符串
def _median(values, n):
    # CORE LOGIC: STEP 1 — 只对有支持的行求median
    # Input: values=[[2,6],[NaN,NaN]],n=[2,0]
    # Output: [4,NaN]
    # Trick: 先用n>0选择行，避免全NaN行nanmedian警告和伪0
    result = np.full(len(n), np.nan)
    ok = n > 0
    if ok.any():
        result[ok] = np.nanmedian(values[ok], axis=1)
    return result


# SETUP LOGIC: _maximum 的函数签名与既有 docstring；不改动说明字符串
def _maximum(values, n):
    # CORE LOGIC: STEP 1 — 只对有支持的行求最大值
    # Input: values=[[2,6],[NaN,NaN]],n=[2,0]
    # Output: [6,NaN]
    # Trick: 支持为0的行仍未知；不能把空最大age设为0
    result = np.full(len(n), np.nan)
    ok = n > 0
    if ok.any():
        result[ok] = np.nanmax(values[ok], axis=1)
    return result


# SETUP LOGIC: _empty_move 的函数签名与既有 docstring；不改动说明字符串
def _empty_move(n):
    # CORE LOGIC: STEP 1 — 构造稳定无支持direction schema
    # Input: n=1，当前没有共同完整fresh dealer
    # Output: 唯一行按mean_bps,median_bps,guarded_mean_bps,up_fraction,down_fraction,flat_fraction,common_n,current_n,lookback_n,retention,condition_changed_fraction,mean_support_age_min,max_support_age_min,guarded_common_n顺序为[NaN,NaN,NaN,NaN,NaN,NaN,0,0,0,NaN,NaN,NaN,NaN,0]
    # Trick: 计数0是已知无支持；movement/fractions/age未知不能填0
    result = np.full((n, len(_MOVE_SUFFIXES)), np.nan)
    for col in _COUNT_SUFFIXES.intersection(_MOVE_COL):
        result[:, _MOVE_COL[col]] = 0
    return result


# SETUP LOGIC: _issuer_records 的函数签名与既有 docstring；不改动说明字符串
def _issuer_records(queries):
    """First known label and first observed conflict, never whole-sample backfill.

    Missing/unknown labels before the first known label do not establish a
    mapping. After that point, any different or unknown observation makes the
    mapping unreliable from its own timestamp onward. A later conflict cannot
    retroactively invalidate an earlier query.
    """
    # CORE LOGIC: STEP 1 — 归一化issuer缺失标签并保留查询身份
    # Input: queries三行X：09:00 'Unknown'，10:00 ' I '，11:00 'J'
    # Output: label=[NA,'I','J']，work保留bond及各已知时间整数ns
    # Trick: 这里仅标签strip；CUSIP和query time不改键；Unknown不建立归属
    labels = queries["ISSUER"].astype("string").str.strip()
    labels = labels.mask(labels.str.lower().isin(["", "unknown", "none", "nan", "<na>"]))
    work = pd.DataFrame({"bond": queries.cusip, "ns": queries._ns, "label": labels})
    records = {}
    # CORE LOGIC: STEP 2 — 只从最早非缺失且同刻无冲突标签建立映射
    # Input: X09:00 Unknown、10:00 I、11:00 J；Y10:00同时I/Unknown
    # Output: X first=10:00,label='I'；Y不建立record
    # Trick: 同一时刻未知/多标签表示当时不能确定；NaT哨兵先排除
    for bond, g in work.loc[work.ns.ne(np.iinfo(np.int64).min)].groupby("bond", sort=False, observed=True):
        known = g.loc[g.label.notna()]
        if known.empty:
            continue
        first = int(known.ns.min())
        at_first = g.loc[g.ns.eq(first), "label"]
        if at_first.isna().any() or at_first.nunique() != 1:
            continue
        # CORE LOGIC: STEP 3 — 记录首次已观察冲突时间，不回溯推翻过去
        # Input: X10:00 I、10:30 I、11:00 J；Z10:00 I之后无冲突
        # Output: records['X']=('I',10:00ns,11:00ns)，Z冲突时间为int64最大值
        # Trick: 后续query只有first<=t<conflict时归属有效；11:00冲突不会改变10:30结果
        label = str(at_first.iloc[0])
        conflict = g.loc[g.ns.ge(first) & (g.label.isna() | g.label.ne(label).fillna(True)), "ns"]
        records[bond] = (label, first, int(conflict.min()) if len(conflict) else _NEVER)
    return records


# SETUP LOGIC: _event_index 的函数签名与既有 docstring；不改动说明字符串
def _event_index(events, queries, needed_bonds, lookback_ns, age_ns):
    """One narrow index, restricted to queried ET days and possible fresh times."""
    # CORE LOGIC: STEP 1 — 以整数ns和ET日过滤有效queries，空查询快速结束
    # Input: queries只有time=NaT；events有X10:00
    # Output: (pd.DataFrame()共0行0列,{},set())；events中X10:00没有被借给NaT query
    # Trick: NaT的int64最小值不是时间0；实际query不存在时无需扫描dealer
    stamp = to_ny_datetime(events.quote_timestamp_ET).array.as_unit("ns").asi8
    day = to_ny_datetime(events.day).array.as_unit("ns").asi8
    valid_queries = queries.loc[queries._ns.ne(np.iinfo(np.int64).min)]
    if valid_queries.empty:
        return pd.DataFrame(), {}, set()
    # CORE LOGIC: STEP 2 — 只保留可支持窗口的日期、bond、side、过去事件
    # Input: query X3/2 10:30，needed_bonds={'X'},lookback=30,age=30；事件X09:00/10:00/11:00,Y10:00
    # Output: narrow仅X10:00；index键(X,bid,3/2dayns)指向行0
    # Trick: 全局下界t_min-lookback-age排除不可能fresh的旧消息；上界t_max防未来，dealer细节稍后按每query选择
    valid = (np.isin(day, valid_queries._day.unique()) &
             (stamp >= int(valid_queries._ns.min()) - lookback_ns - age_ns) &
             (stamp <= int(valid_queries._ns.max())) & events.cusip.isin(needed_bonds).to_numpy() &
             events.side.isin(_SIDES).to_numpy())
    cols = ["firm", "cusip", "side", "complete", "center", "quantity_set", "candidate_count"]
    narrow = events.loc[valid, cols].reset_index(drop=True)
    narrow["_ns"], narrow["_day"] = stamp[valid], day[valid]
    index = narrow.groupby(["cusip", "side", "_day"], observed=True, sort=False).indices
    return narrow, index, set(narrow._day.unique())


# SETUP LOGIC: _dealer_states 的函数签名与既有 docstring；不改动说明字符串
def _dealer_states(narrow, positions):
    # CORE LOGIC: STEP 1 — 从一次bond/side/day切片排序每dealer事件
    # Input: A消息输入顺序10:01中心102、10:00中心100；B10:00中心105
    # Output: A排序time=[10:00ns,10:01ns]；相同A/known-time出现两事件则ValueError
    # Trick: iloc使用groupby.indices的相对位置；stable排序保留键对齐，严格去掉重复事件歧义
    states = []
    g = narrow.iloc[positions]
    for indices in g.groupby("firm", observed=True, sort=False).indices.values():
        h = g.iloc[indices]
        order = np.argsort(h._ns.to_numpy(), kind="stable")
        h = h.iloc[order]
        times = h._ns.to_numpy(dtype="int64")
        if len(times) > 1 and (np.diff(times) <= 0).any():
            raise ValueError("Event cache must have one event per firm/bond/side/known time")
        # CORE LOGIC: STEP 2 — 量化候选条件标签并打包dealer状态数组
        # Input: A三事件quantity_set=[('q=2.0',),('q=3.0',),('q=2.0',)]，count=[1,1,2]
        # Output: quantity codes=[0,1,0]，candidate_count=[1,1,2]，与time/center逐行对齐
        # Trick: factorize仅用于集合相等判断，不把code当quantity大小；不删0/未知size候选
        states.append((times, h.complete.fillna(False).to_numpy(dtype=bool),
                       h.center.to_numpy(dtype=float),
                       pd.factorize(h.quantity_set, sort=False)[0],
                       h.candidate_count.to_numpy(dtype=float)))
    return states


# SETUP LOGIC: _movement_at 的函数签名与既有 docstring；不改动说明字符串
def _movement_at(states, times, day_ns, lookback_ns, age_ns, allow_exact):
    """One bond/side/day, all query times in a bounded block, one dealer per vote."""
    # CORE LOGIC: STEP 1 — 无dealer状态直接返回0支持/NaN
    # Input: times=[10:30ns,10:35ns]，states=[]
    # Output: 两行common_n/current_n/lookback_n/guarded_common_n=0，方向和支持age均NaN
    # Trick: 无quote捷径不能用于存在但latest incomplete的状态，后者仍选择最新消息
    if not states:
        return _empty_move(len(times)), np.full(len(times), np.nan), np.full(len(times), np.nan)
    # CORE LOGIC: STEP 2 — 建立query×dealer方向和支持age矩阵
    # Input: times有2个query，states有A/B共2 dealer
    # Output: delta/guarded/ages/guarded_ages shape=(2,2)全NaN，current_n/old_n/changed_n=[0,0]
    # Trick: 列固定代表同dealer，后续共同样本判断不会把不同roster中心相减
    n = len(times)
    delta = np.full((n, len(states)), np.nan)
    guarded = delta.copy()
    ages = delta.copy()
    guarded_ages = delta.copy()
    current_n, old_n = np.zeros(n), np.zeros(n)
    changed_n = np.zeros(n)
    # CORE LOGIC: STEP 3 — 固定30分钟起点且要求同ET日
    # Input: query 3/3 00:10，lookback=30min，day_ns=3/3 00:00
    # Output: previous=3/2 23:40ns，same_day=False；allow_exact=True使用right
    # Trick: elapsed窗口使用整数ns；越过ET午夜时起点不借前日quote
    previous = times - lookback_ns
    same_day = previous >= day_ns
    side = "right" if allow_exact else "left"
    # CORE LOGIC: STEP 4 — 两端分别向后选同dealer最新known-time
    # Input: A消息10:00完整100、10:29完整105、10:30不完整；query10:30，lookback10:00
    # Output: allow_exact=True now指10:30,old指10:00；False now指10:29且old=-1
    # Trick: right-1是<=，left-1是<；idx=-1先clamp防越界，稍后mask禁止偷取第0条
    for dealer, (stamp, complete, center, quantity, count) in enumerate(states):
        now = np.searchsorted(stamp, times, side=side) - 1
        old = np.searchsorted(stamp, previous, side=side) - 1
        ni, oi = np.maximum(now, 0), np.maximum(old, 0)
        now_age, old_age = times - stamp[ni], previous - stamp[oi]
        # CORE LOGIC: STEP 5 — 最新完整、finite、fresh且同日才构成共同dealer
        # Input: A端点10:00完整100、10:30不完整；B两端完整，较大age=30分钟；age_min=30
        # Output: A current_n贡献0、lookback_n贡献1但common=False；B两端均支持common=True
        # Trick: 必须先asof最新，再check complete；不允许退回A旧完整消息，<=保留恰好fresh阈值
        valid_now = (now >= 0) & complete[ni] & np.isfinite(center[ni]) & (now_age >= 0) & (now_age <= age_ns)
        valid_old = same_day & (old >= 0) & complete[oi] & np.isfinite(center[oi]) & (old_age >= 0) & (old_age <= age_ns)
        current_n += valid_now
        old_n += valid_old
        common = valid_now & valid_old
        # CORE LOGIC: STEP 6 — 保留signed观察方向，并标记quantity/count条件变化
        # Input: A center100→104，quantity code0→1,count1→1；B center100→98，quantity/count不变
        # Output: delta=[4,-2]，guarded=[NaN,-2]，changed_n=1；ages各取两端较大message age
        # Trick: guard不是删quote；它明确区分可比条件的movement与条件改变后的观察差
        changed = (quantity[ni] != quantity[oi]) | (count[ni] != count[oi])
        value = center[ni] - center[oi]
        delta[:, dealer] = np.where(common, value, np.nan)
        guarded[:, dealer] = np.where(common & ~changed, value, np.nan)
        age = np.maximum(now_age, old_age) / _MINUTE_NS
        ages[:, dealer] = np.where(common, age, np.nan)
        guarded_ages[:, dealer] = np.where(common & ~changed, age, np.nan)
        changed_n += common & changed
    # CORE LOGIC: STEP 7 — 在同一dealer集合上汇总方向mean/median及guarded mean
    # Input: delta=[4,-2]，guarded=[NaN,-2]
    # Output: common_n=2,guarded_n=1,mean_bps=1,median_bps=1,guarded_mean_bps=-2
    # Trick: 观察mean与guarded mean用各自实际支持分母，不能把被guard掉值当0
    common_n = np.isfinite(delta).sum(axis=1)
    guarded_n = np.isfinite(guarded).sum(axis=1)
    result = _empty_move(n)
    result[:, _MOVE_COL["mean_bps"]] = _mean(delta, common_n)
    result[:, _MOVE_COL["median_bps"]] = _median(delta, common_n)
    result[:, _MOVE_COL["guarded_mean_bps"]] = _mean(guarded, guarded_n)
    # CORE LOGIC: STEP 8 — 同共同dealer样本给出up/down/flat与支持计数
    # Input: delta=[4,-2,0]，current_n=4,lookback_n=3，guarded有2个有效值
    # Output: up/down/flat各1/3；common_n=3,current_n=4,lookback_n=3,guarded_common_n=2
    # Trick: 正/负指spread升降，不是价格涨跌；exact 0单列为flat，无支持fractions仍NaN
    for name, values in [("up_fraction", delta > 0), ("down_fraction", delta < 0), ("flat_fraction", delta == 0)]:
        result[:, _MOVE_COL[name]] = np.divide(values.sum(axis=1), common_n,
            out=np.full(n, np.nan), where=common_n > 0)
    for name, values in [("common_n", common_n), ("current_n", current_n),
                         ("lookback_n", old_n), ("guarded_common_n", guarded_n)]:
        result[:, _MOVE_COL[name]] = values
    # CORE LOGIC: STEP 9 — roster留存、条件变化率和支持age
    # Input: common_n=3,current_n=4,lookback_n=3,changed_n=1；三dealer最大端点age=[0,5,10]
    # Output: retention=.75,condition_changed_fraction=1/3,mean_age=5,max_age=10
    # Trick: retention分母max两个fresh roster大小；age基于共同dealer两端较大值，guarded age另返回用于issuer
    result[:, _MOVE_COL["retention"]] = np.divide(common_n, np.maximum(current_n, old_n),
        out=np.full(n, np.nan), where=common_n > 0)
    result[:, _MOVE_COL["condition_changed_fraction"]] = np.divide(changed_n, common_n,
        out=np.full(n, np.nan), where=common_n > 0)
    result[:, _MOVE_COL["mean_support_age_min"]] = _mean(ages, common_n)
    result[:, _MOVE_COL["max_support_age_min"]] = _maximum(ages, common_n)
    return result, _mean(guarded_ages, guarded_n), _maximum(guarded_ages, guarded_n)


# SETUP LOGIC: _issuer_at 的函数签名与既有 docstring；不改动说明字符串
def _issuer_at(values, dealer_n, mean_age, max_age):
    """Target bond already excluded; each finite donor bond supplies one vote."""
    # CORE LOGIC: STEP 1 — 按其它bond有效方向计数，dealer数仅为支持信息
    # Input: target已排除；values一行=[2,6,NaN]，dealer_n=[3,1,5]
    # Output: other_bond_n=2,common_dealer_n=4；supported=True
    # Trick: 缺方向bond的5 dealer不计入支持；一个bond始终只一票
    valid = np.isfinite(values)
    n = valid.sum(axis=1)
    result = np.full((len(values), len(_ISSUER_SUFFIXES)), np.nan)
    result[:, 5], result[:, 6] = n, np.where(valid, dealer_n, 0).sum(axis=1)
    supported = n >= 2
    # CORE LOGIC: STEP 2 — 至少两个其它bond时计算方向和分歧比例
    # Input: values=[2,6,NaN]，另行仅[2,NaN,NaN]
    # Output: 首行mean=4,median=4,up=1,down=0,flat=0；次行mean/median/up/down/flat/dispersion/mean_age/max_age均NaN，但other_bond_n=1、common_dealer_n保留该唯一bond实际支持
    # Trick: 不按dealer_n或quantity加权；min2限制是一条预声明支持门槛
    if supported.any():
        v, nn = values[supported], n[supported]
        avg = _mean(v, nn)
        result[supported, 0] = avg
        result[supported, 1] = _median(v, nn)
        for col, flag in [(2, v > 0), (3, v < 0), (4, v == 0)]:
            result[supported, col] = flag.sum(axis=1) / nn
        # CORE LOGIC: STEP 3 — equal-bond dispersion与支持age
        # Input: 两donor方向2/6，各bond guarded-dealer mean age1/5,max age2/8
        # Output: dispersion=2,mean_support_age=3,max_support_age=8
        # Trick: std分母为bond数、ddof=0；age mean先各bond均值再equal bond mean，不由大dealer bond主导
        result[supported, 7] = np.sqrt(np.nansum((v - avg[:, None]) ** 2, axis=1) / nn)
        result[supported, 8] = _mean(np.where(valid[supported], mean_age[supported], np.nan), nn)
        result[supported, 9] = _maximum(np.where(valid[supported], max_age[supported], np.nan), nn)
    return result


# SETUP LOGIC: _dictionary 的函数签名与既有 docstring；不改动说明字符串
def _dictionary(include_issuer):
    # SETUP LOGIC: 固定direction feature的文字定义，不计算观察数据
    definitions = {
        "mean_bps": "Equal-dealer mean of current minus lookback event candidate medians",
        "median_bps": "Median of common-dealer current minus lookback candidate medians",
        "guarded_mean_bps": "Equal-dealer mean change where quantity_set and candidate_count match at endpoints",
        "up_fraction": "Fraction of supported votes with a strictly positive spread change",
        "down_fraction": "Fraction of supported votes with a strictly negative spread change",
        "flat_fraction": "Fraction of supported votes with exactly zero spread change",
        "common_n": "Dealer count complete and fresh at both endpoints",
        "current_n": "Complete fresh dealer count at query time",
        # SETUP LOGIC: 继续固定支持/retention/age文字定义，既有字符串原样保留
        "lookback_n": "Complete fresh dealer count at window start on the same ET day",
        "guarded_common_n": "Common dealer count with unchanged quantity_set and candidate_count",
        "retention": "common_n / max(current_n, lookback_n); unknown without any common dealer",
        "condition_changed_fraction": "Common dealers with changed quantity_set or candidate_count / common_n",
        "mean_support_age_min": "Mean of the larger of the two endpoint message ages",
        "max_support_age_min": "Maximum message age across both endpoints and supported votes",
    }
    # SETUP LOGIC: 为每个side/列附单位和固定support说明
    result = {}
    for side in _SIDES:
        for name in _MOVE_SUFFIXES:
            unit = "count" if name in _COUNT_SUFFIXES else "minutes" if name.endswith("_min") else "bps" if name.endswith("_bps") else "fraction"
            result[f"bcq_{side}_move_{name}"] = {"unit": unit, "definition": definitions[name],
                "support": "Same bond/side/dealer, latest complete fresh events, same ET day; no automatic spread/quantity filtering"}
        # SETUP LOGIC: 为issuer列附文字定义与明确min2/排自身限制
        if include_issuer:
            for name in _ISSUER_SUFFIXES:
                unit = "count" if name in _COUNT_SUFFIXES else "minutes" if name.endswith("_min") else "bps" if name.endswith("_bps") else "fraction"
                definition = ({"other_bond_n": "Number of other bonds with a guarded common-dealer direction",
                    "common_dealer_n": "Sum of guarded common-dealer support across other bonds; not a price weight",
                    "dispersion_bps": "Population standard deviation of other-bond mean directions",
                    "mean_bps": "Equal-bond mean of other bonds' guarded common-dealer mean directions",
                    "median_bps": "Median of other bonds' guarded common-dealer mean directions",
                    # SETUP LOGIC: 继续固定issuer age文字说明；这些是metadata而非方向计算
                    "mean_support_age_min": "Equal-bond mean of donor guarded-dealer mean maximum endpoint ages",
                    "max_support_age_min": "Maximum endpoint age among donor guarded dealers"}.get(name, definitions.get(name, name)))
                # SETUP LOGIC: 组装issuer字典项并返回metadata
                result[f"bcq_{side}_issuer_move_{name}"] = {"unit": unit, "definition": definition,
                    "support": "Target CUSIP excluded; one bond per vote; >=2 other bonds for non-count values; prefix-known stable issuer at window start through query"}
    return result


# SETUP LOGIC: build_movement_features 的函数签名与既有 docstring；不改动说明字符串
def build_movement_features(frame, event_cache, *, lookback_min=30, age_min=30,
                            allow_exact=True, include_issuer=True, progress=None):
    """Build direction features only for Train/Validation from a live event cache.

    Query order/row_id are retained; Test and other split rows are explicitly
    excluded. Latest incomplete messages block old valid state. Exact matches
    apply to quote known-times only. Issuer mapping uses Train/Validation frame
    observations known by the query; donors must already be reliably mapped at
    the window start and remain so through the query. Quote-level medians are
    never pooled across bonds. The sole supported research window is 30 minutes.

    Each event is indexed once. Dealer arrays are reused within an ET day, and
    issuer query blocks share all donor asof calculations before excluding each
    target CUSIP. Memory for issuer query matrices is bounded to 512 times per
    block. No raw input, event aggregation, model fit or disk cache build occurs.
    """
    # SETUP LOGIC: 开始计时，保持已有函数说明/签名
    started = perf_counter()
    # CORE LOGIC: STEP 1 — 限制固定窗口、required frame列及live events输入
    # Input: lookback=10或age=0；另一输入frame缺split；event_cache=None
    # Output: 分别ValueError；合法窗口只30min，age有限正数，缓存须已有events DataFrame
    # Trick: 此模块没有raw参数，输入缺cache时不能偷偷触发39M聚合
    if lookback_min != 30 or not np.isfinite(age_min) or age_min <= 0:
        raise ValueError("This research uses fixed lookback_min=30 and positive finite age_min")
    required = {"row_id", "cusip", "time", "split"} | ({"ISSUER"} if include_issuer else set())
    if not required.issubset(frame.columns):
        raise ValueError("Movement frame missing columns: " + ", ".join(sorted(required - set(frame.columns))))
    if not isinstance(event_cache, dict) or not isinstance(event_cache.get("events"), pd.DataFrame):
        raise ValueError("Provide the existing checked step5_event_cache; movement does not build events")
    # CORE LOGIC: STEP 2 — 检查事件schema并明确只保留Train/Validation query
    # Input: frame row_id=[7,8,9]，split=['Train','Validation','Test']，已有完整events列
    # Output: q保留row_id7/8，Test9排除；非唯一row_id或缺事件字段报错
    # Trick: split筛选作用于查询身份；不会使用Test row建立issuer映射或新增Test features
    events = event_cache["events"]
    event_columns = {"firm", "cusip", "side", "quote_timestamp_ET", "day", "complete", "center", "quantity_set", "candidate_count"}
    if not event_columns.issubset(events.columns):
        raise ValueError("Movement event cache missing columns: " + ", ".join(sorted(event_columns - set(events.columns))))
    allowed = frame.split.isin(["Train", "Validation"])
    q = frame.loc[allowed, list(required)].copy().reset_index(drop=True)
    if q.row_id.isna().any() or q.row_id.duplicated().any():
        raise ValueError("Train/Validation row_id must be nonmissing and unique")
    # CORE LOGIC: STEP 3 — 统一时间ns但不改变既有CUSIP身份
    # Input: frame cusip=' X '，time='2026-03-02T15:30:00Z'，age/lookback=30
    # Output: cusip仍' X '，time=纽约10:30，_ns为相同瞬间，_day为纽约午夜ns
    # Trick: CUSIP不strip以免变成另一quoted bond；时间选择不用float微秒
    q["time"] = to_ny_datetime(q.time)
    # Match the existing Step5 exact CUSIP keys; do not silently normalize a
    # different identifier into a quoted bond or change sidecar join keys.
    q["cusip"] = q.cusip.astype("string")
    q["_ns"] = q.time.array.as_unit("ns").asi8
    q["_day"] = q.time.dt.normalize().array.as_unit("ns").asi8
    lookback_ns, age_ns = int(lookback_min * _MINUTE_NS), int(age_min * _MINUTE_NS)
    records = _issuer_records(q) if include_issuer else {}
    q["_issuer"] = None
    # CORE LOGIC: STEP 4 — 将target query映射到当时已知稳定issuer
    # Input: X record=('I',10:00ns,11:00ns)；queries09:59/10:30/11:00
    # Output: _issuer=[None,'I',None]
    # Trick: positions来自reset_index后的groupby.indices；mask只看prefix截止，未来冲突不回填过去
    for bond, positions in q.groupby("cusip", observed=True, sort=False).indices.items():
        record = records.get(bond)
        if record is not None:
            label, first, conflict = record
            positions = np.asarray(positions)
            valid = (q._ns.to_numpy()[positions] >= first) & (q._ns.to_numpy()[positions] < conflict)
            q.loc[positions[valid], "_issuer"] = label
    # PROGRESS LOGIC: 报告即将建立窄events索引；不做任何raw聚合
    if progress is not None:
        progress("movement_index", None, None, f"Indexing existing events for {len(q):,} Train/Validation queries; no raw aggregation")
    # CORE LOGIC: STEP 5 — 使用live窄events建立所需查询索引
    # Input: q只含X/Y3/2的Train/Validation，records额含同issuer Z
    # Output: 索引保留可用X/Y/Z历史状态，available_days仅有实际可支持查询的ET日
    # Trick: 这里调用独立索引函数而非prepare/events聚合；donor不是来自未来Test label
    narrow, index, available_days = _event_index(events, q, set(q.cusip.dropna()) | set(records), lookback_ns, age_ns)
    # SETUP LOGIC: 记录索引阶段完成时间
    indexed = perf_counter()
    # CORE LOGIC: STEP 6 — 初始化声明features并将无支持计数设0
    # Input: q=[{row_id:7,cusip:'X',time:'10:00'},{row_id:8,cusip:'X',time:'10:30'}]，include_issuer=True
    # Output: output shape=(2,48)，两行都相同：双side move_common_n/current_n/lookback_n/guarded_common_n及issuer_move_other_bond_n/common_dealer_n共12列0；其余36列NaN
    # Trick: 覆盖图应看guarded_mean/issuer_mean的finite值，不把0计数non-null当有效feature
    names = DIRECTION_FEATURES + (ISSUER_FEATURES if include_issuer else [])
    output = np.full((len(q), len(names)), np.nan)
    for i, name in enumerate(names):
        if any(name.endswith("_" + suffix) for suffix in _COUNT_SUFFIXES):
            output[:, i] = 0
    # CORE LOGIC: STEP 7 — 一次按已知issuer组织可候选donor bond
    # Input: records={'X':('I',10:00ns,11:00ns),'Y':('I',09:00ns,maxint)}
    # Output: issuer_bonds={'I':['X','Y']}
    # Trick: 此处只建立索引；每query仍会按first/conflict检查，不等于把未来归属提前
    issuer_bonds = {}
    for bond, record in records.items():
        issuer_bonds.setdefault(record[0], []).append(bond)
    # Unknown mappings require only their own bond/day; they never cause an
    # issuer-wide donor scan. Invalid-time rows retain the empty schema.
    # CORE LOGIC: STEP 8 — 有可靠issuer按issuer/day共享查询，其它只算自身bond/day
    # Input: q位置0/1/2/3分别为X/I/3月2日10:30、Y/I/3月2日10:30、Z/None/3月2日10:30、X/I/NaT
    # Output: groups=[(3月2日ET午夜ns,'I',array([0,1])),(同日ET午夜ns,None,array([2]))]；位置3没有进入组且输出支持为0、方向NaN
    # Trick: unknown不能引发全issuer扫描；groups只按ET日排序以便state缓存释放
    valid = q._ns.ne(np.iinfo(np.int64).min) & q.cusip.notna() & q.cusip.str.strip().ne("")
    known = valid & q._issuer.notna() if include_issuer else pd.Series(False, index=q.index)
    groups = []
    for (day, label), positions in q.loc[known].groupby(["_day", "_issuer"], observed=True, sort=False).groups.items():
        groups.append((int(day), label, np.asarray(positions)))
    for (day, bond), positions in q.loc[valid & ~known].groupby(["_day", "cusip"], observed=True, sort=False).groups.items():
        groups.append((int(day), None, np.asarray(positions)))
    groups.sort(key=lambda item: item[0])
    # CACHEING LOGIC: 初始化单ET日dealer-array缓存；换日释放，不保存巨大全query矩阵
    cached_day, states_cache = None, {}
    # CORE LOGIC: STEP 9 — 同日查询稳定排序并去重为512时间块
    # Input: 同I3/2 positions=[row7@10:35,row8@10:30,row9@10:30]
    # Output: 排序后[8,9,7]，unique=[10:30ns,10:35ns]；同time只计算一次
    # Trick: 同一天复用states_cache，日期变化清空；无available_day直接保留空输出
    for done, (day, label, positions) in enumerate(groups, 1):
        if day != cached_day:
            cached_day, states_cache = day, {}
        if day in available_days:
            order = np.argsort(q._ns.to_numpy()[positions], kind="stable")
            positions = positions[order]
            ns = q._ns.to_numpy()[positions]
            unique = np.unique(ns)
            for start in range(0, len(unique), _QUERY_BLOCK):
                times = unique[start:start + _QUERY_BLOCK]
                # PROGRESS LOGIC: 每512 unique-query块开始前更新具体时间块范围，避免大issuer长时间无状态
                if progress is not None:
                    progress("movement_features", done - 1, len(groups),
                        f"Query group {done:,}/{len(groups):,}: {label or 'unmapped bond'}; unique times {start + 1:,}–{start + len(times):,}/{len(unique):,}")
                # CORE LOGIC: STEP 10 — 把重复target rows映射回当前时间块，并仅选可能已知donor
                # Input: ns排序=[10:30,10:30,10:35]，块含两个时间；X/Y为target，Z在10:00已知I
                # Output: rows含全部3条；row_times=[0,0,1]；bonds并集X/Y/Z且bond_index逐列固定
                # Trick: left取首个重复行，right包含末个重复行；donor粗筛后还需逐time检查prefix，不凭块末信息回填早query
                a = np.searchsorted(ns, times[0], side="left")
                b = np.searchsorted(ns, times[-1], side="right")
                rows = positions[a:b]
                row_times = np.searchsorted(times, q._ns.to_numpy()[rows])
                targets = q.cusip.to_numpy()[rows]
                donors = [] if label is None else [bond for bond in issuer_bonds.get(label, [])
                    if records[bond][1] <= times[-1] - lookback_ns and records[bond][2] > times[0]]
                bonds = sorted(set(donors) | set(targets))
                bond_index = {bond: i for i, bond in enumerate(bonds)}
                # CORE LOGIC: STEP 11 — 为当前块建立donor方向矩阵并跳过不存在的bond/side/day
                # Input: times=[10:30ns,10:35ns],bonds=['X','Y','Z'],side='bid'；index仅有(X,bid,当天),(Y,bid,当天)
                # Output: donor_values/donor_age/donor_max_age初始[[NaN,NaN,NaN],[NaN,NaN,NaN]],donor_n=[[0,0,0],[0,0,0]]；Z列不计算asof，保留NaN及0支持
                # Trick: 矩阵列恒为bond；bid/ask分开处理，donor投票不会跨side拼接
                for side_i, side in enumerate(_SIDES):
                    donor_values = np.full((len(times), len(bonds)), np.nan)
                    donor_n = np.zeros_like(donor_values)
                    donor_age = donor_values.copy()
                    donor_max_age = donor_values.copy()
                    for bond_i, bond in enumerate(bonds):
                        key = (bond, side, day)
                        if key not in index:
                            continue
                        # CACHEING LOGIC: 已有bond/side/day数组直接复用；第一次仅从narrow index取dealer状态
                        if key not in states_cache:
                            states_cache[key] = _dealer_states(narrow, index[key])
                        # CORE LOGIC: STEP 12 — 计算当前bond共同dealer方向并按query位置写自身特征
                        # Input: X共同dealer方向+4；rows里X两条都query10:30，Y一条query10:35
                        # Output: 两条X row均写mean=4/guarded_mean=4；Y row不被X结果覆盖
                        # Trick: own按CUSIP筛位置，row_times映射shared计算轴；计算节省不能丢重复trade rows
                        movement, mean_age, max_age = _movement_at(states_cache[key], times, day, lookback_ns, age_ns, allow_exact)
                        own = np.flatnonzero(targets == bond)
                        if len(own):
                            offset = side_i * len(_MOVE_SUFFIXES)
                            output[rows[own], offset:offset + len(_MOVE_SUFFIXES)] = movement[row_times[own]]
                        # CORE LOGIC: STEP 13 — donor必须在窗口起点已知且到query未冲突
                        # Input: Y labelI首次10:01；query10:30与10:35，窗口起点10:00/10:05
                        # Output: mapped=[False,True]；第一Y vote=NaN，第二用Y guarded_mean及guarded dealer支持
                        # Trick: 起点归属未知不能用query末已知标签追溯；冲突必须strictly>query才允许
                        record = records.get(bond) if label is not None else None
                        if record is not None and record[0] == label:
                            mapped = (record[1] <= times - lookback_ns) & (record[2] > times)
                            donor_values[:, bond_i] = np.where(mapped, movement[:, _MOVE_COL["guarded_mean_bps"]], np.nan)
                            donor_n[:, bond_i] = movement[:, _MOVE_COL["guarded_common_n"]]
                            donor_age[:, bond_i], donor_max_age[:, bond_i] = mean_age, max_age
                    # CORE LOGIC: STEP 14 — 每target完整排除自身bond，再合成issuer方向及支持
                    # Input: donor_values@10:30=[X100,Y2,Z6]，target=X，Y/Z dealer_n=3/1
                    # Output: 排后[NaN,2,6]；issuer mean=4,other_bond_n=2,common_dealer_n=4；写回X row
                    # Trick: copy防删除X改变Y target的原共享矩阵；自身从mean/median/age/dispersion/支持计数全部排除
                    if label is not None:
                        # Share donor snapshots, then remove the target bond from
                        # every vote, count, dispersion and support-age statistic.
                        for bond in set(targets):
                            own = np.flatnonzero(targets == bond)
                            ti = row_times[own]
                            values = donor_values[ti].copy()
                            values[:, bond_index[bond]] = np.nan
                            issuer = _issuer_at(values, donor_n[ti], donor_age[ti], donor_max_age[ti])
                            offset = len(DIRECTION_FEATURES) + side_i * len(_ISSUER_SUFFIXES)
                            output[rows[own], offset:offset + len(_ISSUER_SUFFIXES)] = issuer
        # PROGRESS LOGIC: 按query组数量报告完成；计数不改变feature结果
        if progress is not None and (done == len(groups) or done % max(1, len(groups) // 100) == 0):
            progress("movement_features", done, len(groups), f"Completed {done:,}/{len(groups):,} issuer/day or unmapped bond/day query blocks")
    # SETUP LOGIC: 记录方向计算完成时间
    computed = perf_counter()
    # CORE LOGIC: STEP 15 — 回接原Train/Validation身份及附加features
    # Input: include_issuer=False，q=[{row_id:8,cusip:'X',time:'10:30'},{row_id:7,cusip:'Y',time:'10:30'}]；按14个_MOVE_SUFFIXES顺序，output的bid向量为[4,4,4,1,0,0,1,1,1,1,0,0,0,1]与[-2,-2,-2,0,1,0,1,1,1,1,0,0,0,1]，两row ask向量都是[NaN,NaN,NaN,NaN,NaN,NaN,0,0,0,NaN,NaN,NaN,NaN,0]
    # Output: 2行31列DataFrame；前3列分别(8,'X','10:30')、(7,'Y','10:30')；每行后28列依次是明示的bid14向量+ask14向量，顺序不按row_id大小重排
    # Trick: q和output都reset RangeIndex，concat(axis=1)按同index对齐而非time，避免重复时间串行
    result = pd.concat([q[["row_id", "cusip", "time"]], pd.DataFrame(output, columns=names)], axis=1)
    # SETUP LOGIC: 记录固定窗口、asof、条件guard、issuer因果/min2规则配置，字面定义不改计算
    config = dict(lookback_min=30.0, age_min=float(age_min), allow_exact=bool(allow_exact),
        include_issuer=bool(include_issuer), same_et_day=True,
        direction_support="same dealer; latest complete and fresh at both endpoints",
        condition_guard="quantity_set and candidate_count unchanged at endpoints",
        retention_denominator="max(current_n, lookback_n)", issuer_min_other_bonds=2,
        issuer_vote="one bond; guarded common-dealer mean change",
        issuer_mapping_source="Train/Validation frame ISSUER prefix only",
        issuer_mapping_rule="target known at query; donor known at window start and no observed conflict through query; unknown/conflicted excluded")
    # Deferred import keeps movement import independent of modelling code and
    # uses the same experiment binding as the incremental-fit consumer.
    # SETUP LOGIC: 延迟导入共有frame_fingerprint，避免movement和model模块循环导入
    from quote_quality_incremental import frame_fingerprint
    # CORE LOGIC: STEP 16 — 绑定原frame与event cache身份，记录字典及配置
    # Input: frame三行(row_id,split)=(7,Train),(8,Validation),(9,Test)；event_cache.cache_key='checked-events-001'；started=0,indexed=1,computed=2,当前时钟=3秒；age/lookback=30,exact=True,include_issuer=True
    # Output: attrs.event_cache_key='checked-events-001',excluded_rows=1,included_splits=['Train','Validation']；movement_config.lookback_min=age_min=30且allow_exact=True；timings={index_s:1,features_s:1,fingerprint_s:1,total_s:3}；source_frame_sha256为原7/8两行全部字段的SHA256，feature_dictionary有28direction+20issuer定义
    # Trick: hash仅实验身份校验，不作为预测输入；Test-only输入返回空result并hash=None；timings不改特征
    fingerprint = frame_fingerprint(frame) if len(q) else None
    result.attrs.update(movement_config=config, event_cache_key=event_cache.get("cache_key"),
        source_frame_sha256=fingerprint, feature_dictionary=_dictionary(include_issuer),
        excluded_rows=int((~allowed).sum()), included_splits=["Train", "Validation"],
        timings=dict(index_s=indexed - started, features_s=computed - indexed,
                     fingerprint_s=perf_counter() - computed, total_s=perf_counter() - started))
    # PROGRESS LOGIC: 报告sidecar完成和Test排除状态
    if progress is not None:
        progress("movement_features", len(groups), len(groups), f"Movement sidecar ready: {len(result):,} rows; Test excluded")
    # CORE LOGIC: STEP 17 — 返回已计算同样本的独立sidecar
    # Input: result唯一row的身份为(row_id=8,cusip='X',time='2026-03-02 10:30-05:00')；bid14向量=[4,4,4,1,0,0,1,1,1,1,0,0,0,1]，ask14=[NaN,NaN,NaN,NaN,NaN,NaN,0,0,0,NaN,NaN,NaN,NaN,0]，每side issuer10=[NaN,NaN,NaN,NaN,NaN,0,0,NaN,NaN,NaN]；attrs.event_cache_key='events-001',excluded_rows=1,movement_config.lookback_min=30
    # Output: 原同一DataFrame对象，row_id=8、51列(3身份+明示48数值)以及attrs的'events-001'/1/30原值不变；不添加Test row_id9，也不改成另一查询time
    # Trick: 不合并改写原frame、不fit、不打开Test；调用者以row_id验证并保存
    return result
