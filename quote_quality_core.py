"""Shared calculations for the Step 4/5 research notebooks; no data loading or plotting on import."""
import numpy as np
import pandas as pd
from time import perf_counter

KEYS = ["firm", "cusip", "side", "quote_timestamp_ET"]
SERIES = KEYS[:3]
HISTORY_GAP_MIN = 60
LOOKBACK_MIN = 30
DEFAULT_AGE_MIN = 30
MIN_PEERS = 3
CLIP_FLOOR_BPS = 10.0
MAD_MULTIPLIER = 4.0

def to_ny_datetime(series):
    values = pd.to_datetime(series, errors="coerce", format="mixed")
    if values.dt.tz is None:
        return values.dt.tz_localize("America/New_York")
    return values.dt.tz_convert("America/New_York")


def event_history(raw, progress=None, keep_raw=True):
    """Distinct sets and prefix-only changes; optional progress(stage, done, total, detail)."""
    started = perf_counter()
    if progress is not None:
        progress('events', None, None, f'Normalizing {len(raw):,} quote rows')
    # Duplicate semantics still use the complete source row, while event work needs
    # only six columns. Extra source metadata no longer travels through groupby.
    repeats = raw.duplicated()
    q = raw.copy() if keep_raw else raw[KEYS + ['spread', 'quantity']].copy()
    valid = q[KEYS].notna().all(axis=1)
    for key in SERIES:
        valid &= q[key].astype("string").str.strip().ne("").fillna(False)
    q["repeat"] = repeats
    q["s"] = pd.to_numeric(q["spread"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    q["q"] = pd.to_numeric(q["quantity"], errors="coerce")
    q["qkind"] = np.select([q["quantity"].isna(), q["q"].eq(0), np.isfinite(q["q"]) & q["q"].gt(0)],
                            ["Missing", "Zero", "Positive"], default="Other")
    q["qtag"] = q["qkind"].astype(str)
    positive = q["qkind"].eq("Positive")
    q.loc[positive, "qtag"] = q.loc[positive, "q"].map(lambda v: "q=" + repr(float(v)))
    other = q["qkind"].eq("Other")
    q.loc[other, "qtag"] = "Other:" + q.loc[other, "quantity"].astype(str)
    q["bad"] = q["s"].isna()
    q["pair"] = list(zip(q["s"].fillna("Nonfinite"), q["qtag"]))
    usable = q.loc[valid]
    normalized = perf_counter()
    if progress is not None:
        progress('events', None, None, f'Aggregating {len(usable):,} keyed quote rows')
    g = usable.groupby(KEYS, observed=True, sort=False).agg(
        rows=("s", "size"), repeats=("repeat", "sum"), bad=("bad", "sum"),
        spread_set=("s", lambda v: tuple(sorted(v.dropna().unique()))),
        quantity_set=("qtag", lambda v: tuple(sorted(v.unique()))),
        pair_set=("pair", lambda v: frozenset(v)),
    ).reset_index().sort_values(SERIES + [KEYS[-1]], kind="stable").reset_index(drop=True)
    g["candidate_count"] = g["spread_set"].map(len)
    aggregated = perf_counter()
    if progress is not None:
        progress('events', None, None, f'Building history for {len(g):,} events')
    for col, fn in [("lo", min), ("hi", max), ("center", np.median)]:
        g[col] = g["spread_set"].map(lambda v: float(fn(v)) if v else np.nan)
    g["gap"] = g["hi"] - g["lo"]
    g["center_nearest_gap"] = [min(abs(s - c) for s in ss) if ss else np.nan
                               for ss, c in zip(g["spread_set"], g["center"])]
    g["complete"] = g["bad"].eq(0) & g["candidate_count"].gt(0)
    g["day"] = g[KEYS[-1]].dt.normalize()
    group = g.groupby(SERIES + ["day"], sort=False, observed=True)
    prev = group[["spread_set", "quantity_set", "pair_set", "complete", "candidate_count", "center", KEYS[-1]]].shift()
    g["interval_min"] = (g[KEYS[-1]] - prev[KEYS[-1]]).dt.total_seconds() / 60
    continuous = g["interval_min"].le(HISTORY_GAP_MIN) & g["complete"] & prev["complete"].eq(True)
    g["history_break"] = ~continuous
    g["spread_changed"] = continuous & g["spread_set"].ne(prev["spread_set"])
    g["pair_refresh"] = continuous & g["pair_set"].eq(prev["pair_set"])
    g["condition_changed"] = continuous & (g["quantity_set"].ne(prev["quantity_set"]) | g["candidate_count"].ne(prev["candidate_count"]))
    g["center_delta"] = (g["center"] - prev["center"]).where(continuous)
    g["guarded_delta"] = g["center_delta"].where(~g["condition_changed"])
    # Bounds are order statistics, not tracked quote identities.
    for bound in ["lo", "hi"]:
        g[f"{bound}_delta"] = group[bound].diff().where(continuous & ~g["condition_changed"])
    two_back = group["pair_set"].shift(2)
    g["observed_aba"] = continuous & group["history_break"].shift().eq(False) & g["pair_set"].eq(two_back) & g["pair_set"].ne(prev["pair_set"])
    # Break on day/series changes, incomplete observations, or a long gap.
    segment = g["history_break"].cumsum()
    g["history_start"] = g.groupby(segment)[KEYS[-1]].transform("first")
    g["last_change"] = g[KEYS[-1]].where(g["spread_changed"]).groupby(segment).ffill()
    g["change_age_min"] = (g[KEYS[-1]] - g["last_change"]).dt.total_seconds() / 60
    g["change_age_unknown"] = g["last_change"].isna()
    # Rows are already sorted by series and time, so each history segment is a
    # contiguous slice. Read arrays once and write once instead of constructing
    # pandas indexers/Series for every (often singleton) segment.
    event_times = g[KEYS[-1]].array.as_unit("ns").asi8
    changed = g["spread_changed"].to_numpy(dtype=np.int64)
    starts = np.flatnonzero(g["history_break"].to_numpy())
    stops = np.r_[starts[1:], len(g)]
    changes_30m = np.zeros(len(g), dtype=np.int64)
    if progress is not None:
        progress('events', 0, len(starts), 'Counting changes within history segments')
    for completed, (start, stop) in enumerate(zip(starts, stops), 1):
        # A singleton starts with history_break=True, hence spread_changed=False.
        if stop-start > 1:
            times = event_times[start:stop]
            counts = np.r_[0, changed[start:stop].cumsum()]
            left = np.searchsorted(times, times - LOOKBACK_MIN * 60 * 10**9, side="right")
            changes_30m[start:stop] = counts[1:] - counts[left]
        if progress is not None and (completed % max(1, (len(starts) + 99) // 100) == 0 or completed == len(starts)):
            progress('events', completed, len(starts), f'History segments {completed:,}/{len(starts):,}')
    g["changes_30m"] = changes_30m
    result = {"events": g, "unkeyed": int((~valid).sum()),
              "timings": {'normalize_s': normalized-started,
                          'aggregate_s': aggregated-normalized,
                          'history_s': perf_counter()-aggregated}}
    if keep_raw:
        result['raw'] = q
    return result


def prepare_quote_events(quotes, progress=None):
    """Reusable narrow event table. Rebuild explicitly after changing source quotes.

    It retains incomplete latest messages; an incomplete message is observed state,
    not an absent quote. No hidden cache can silently reuse a changed DataFrame.
    """
    return event_history(quotes, progress=progress, keep_raw=False)




def representative_issuers(quotes):
    """Navigation only: observed case support, never a clean/dirty issuer label."""
    x = quotes.dropna(subset=KEYS + ['ISSUER']).copy()
    x['s'] = pd.to_numeric(x['spread'], errors='coerce').replace([np.inf, -np.inf], np.nan)
    g = x.groupby(['ISSUER'] + KEYS, observed=True)['s'].agg(['min', 'max', 'nunique']).reset_index()
    g['multi'], g['gap'] = g['nunique'].gt(1), g['max'] - g['min']
    scores = g.groupby('ISSUER', observed=True).agg(events=('multi','size'), multi=('multi','sum'), gap=('gap','max'))
    promoted = {}
    for prefix in ['HPS CORPORATE','EXPAND ENERGY','EASTERN GAS','SKY GROUP','DUKE ENERGY','IBM','COMCAST','MITSUBISHI UFJ','KKR GROUP']:
        found = sorted(n for n in scores.index if str(n).upper().startswith(prefix))
        if found: promoted[found[0]] = 'Reviewed case'
    for metric in ['multi','gap','events']:
        for name in scores.loc[scores.events.ge(20)].sort_values(metric,ascending=False).index[:2]:
            promoted.setdefault(name, 'High ' + metric)
    names = list(promoted) + [n for n in sorted(scores.index) if n not in promoted]
    return [(f'[{promoted[n]}] {n}' if n in promoted else str(n), n) for n in names]


def pair_snapshots(events, times, age_min=30, sync_min=1, allow_exact=True):
    """One bond; latest bid/ask per dealer, same ET day. Cartesian ranges stay set-valued."""
    if age_min <= 0 or sync_min < 0: raise ValueError('age_min must be positive and sync_min nonnegative')
    times = pd.DatetimeIndex(times).sort_values().unique()
    if events['cusip'].nunique() > 1:
        raise ValueError('pair_snapshots expects one bond')
    rows = []
    fields = ['firm','side',KEYS[-1],'day','complete','center','lo','hi','candidate_count','pair_set']
    for (firm, side), h in events.groupby(['firm','side'], observed=True):
        if side not in ['bid','ask']: continue
        joined = pd.merge_asof(pd.DataFrame({'time':times}), h[fields].sort_values(KEYS[-1]),
                               left_on='time', right_on=KEYS[-1], direction='backward', allow_exact_matches=allow_exact)
        joined = joined.loc[joined[KEYS[-1]].notna() & joined.time.dt.normalize().eq(joined.day)].copy()
        joined['age'] = (joined.time - joined[KEYS[-1]]).dt.total_seconds()/60
        rows.append(joined)
    fields += ['time','age']
    slots = pd.concat(rows,ignore_index=True) if rows else pd.DataFrame(columns=fields)
    b = slots.loc[slots.side.eq('bid')].drop(columns='side')
    a = slots.loc[slots.side.eq('ask')].drop(columns='side')
    p = b.merge(a,on=['time','firm'],how='outer',suffixes=('_bid','_ask'))
    p['both'] = p[f'{KEYS[-1]}_bid'].notna() & p[f'{KEYS[-1]}_ask'].notna()
    p['complete'] = p.complete_bid.eq(True) & p.complete_ask.eq(True)
    p['max_age'] = p[['age_bid','age_ask']].max(axis=1).where(p.both)
    p['time_gap'] = (pd.to_datetime(p[f'{KEYS[-1]}_bid']) - pd.to_datetime(p[f'{KEYS[-1]}_ask'])).abs().dt.total_seconds()/60
    p['same_time'] = p.both & p['time_gap'].eq(0)
    for name in ['gap_low','gap_high','gap_center','mid','mid_range','matched_gap_low','matched_gap_high','matched_gap','matched_mid']:
        p[name] = np.nan
    p['positive_matches'] = 0
    p['size_state'] = 'Unassessed'
    p['gap_low'] = (p.lo_bid-p.hi_ask).where(p.complete)
    p['gap_high'] = (p.hi_bid-p.lo_ask).where(p.complete)
    p['gap_center'] = (p.center_bid-p.center_ask).where(p.complete)
    p['mid'] = ((p.center_bid+p.center_ask)/2).where(p.complete)
    p['mid_range'] = ((p.hi_bid-p.lo_bid+p.hi_ask-p.lo_ask)/2).where(p.complete)
    # Repeated snapshots reuse the same candidate-set match, instead of expanding pairs.
    cache = {}
    matched_columns = ['size_state','positive_matches','matched_gap_low','matched_gap_high','matched_gap','matched_mid']
    matches = []
    for r in p.loc[p.complete].itertuples():
        key = (r.pair_set_bid, r.pair_set_ask)
        if key not in cache:
            bid, ask = {}, {}
            for pairs, dest in [(r.pair_set_bid,bid),(r.pair_set_ask,ask)]:
                for spread, tag in pairs: dest.setdefault(tag,[]).append(spread)
            shared = [tag for tag in set(bid)&set(ask) if tag.startswith('q=')]
            only_positive = all(tag.startswith('q=') for tag in list(bid)+list(ask))
            state = 'Shared positive' if shared else ('Positive unmatched' if only_positive else 'Unknown / mixed')
            values = [state,0,np.nan,np.nan,np.nan,np.nan]
            if shared:
                values = [state,len(shared), min(min(bid[q])-max(ask[q]) for q in shared),
                    max(max(bid[q])-min(ask[q]) for q in shared),
                    np.mean([np.median(bid[q])-np.median(ask[q]) for q in shared]),
                    np.mean([(np.median(bid[q])+np.median(ask[q]))/2 for q in shared])]
            cache[key] = values
        matches.append(cache[key])
    if matches:
        p.loc[p.complete,matched_columns] = pd.DataFrame(matches,index=p.index[p.complete],columns=matched_columns)
    for suffix,lo,hi,eligible in [('', 'gap_low','gap_high',p.complete),
                                  ('_matched','matched_gap_low','matched_gap_high',p.positive_matches.gt(0))]:
        p['cross'+suffix] = 'Unassessed'
        p.loc[eligible,'cross'+suffix] = np.select([p.loc[eligible,lo].ge(0),p.loc[eligible,hi].lt(0)],['None','All'],default='Some')
    p = p.sort_values(['time','firm'],kind='stable').reset_index(drop=True)
    return pair_policy_masks(p, age_min, sync_min)


def pair_policy_masks(pairs, age_min=30, sync_min=1):
    """Derive age/sync eligibility from cached all-dealer state, without as-of work."""
    if age_min <= 0 or sync_min < 0:
        raise ValueError('age_min must be positive and sync_min nonnegative')
    p = pairs.copy()
    p['fresh_pair'] = p.complete & p.max_age.le(age_min)
    p['size_time_pair'] = p.fresh_pair & p.time_gap.le(sync_min) & p.positive_matches.gt(0)
    return p


def pair_policy_comparison(pairs):
    """A→B isolates slot selection; B→C isolates candidate matching on identical slots.

    Slots are all-dealer bond/day grid observations, not independent trades. B is
    eligible for size/time matching but uses its original candidate sets. C uses
    shared positive raw-quantity candidates on exactly those same B slots.
    """
    rows = []
    for label, mask, cross, gap, mid in [
        ('A fresh / original', pairs.fresh_pair, 'cross', 'gap_center', 'mid'),
        ('B match slots / original', pairs.size_time_pair, 'cross', 'gap_center', 'mid'),
        ('C same slots / matched', pairs.size_time_pair, 'cross_matched', 'matched_gap', 'matched_mid')]:
        z = pairs.loc[mask]
        rows.append(dict(policy=label, n_slots=len(z), n_dealers=z.firm.nunique(),
            gap_mean_bps=z[gap].mean(), mid_mean_bps=z[mid].mean(),
            **{state.lower()+'_fraction': z[cross].eq(state).mean() for state in ['None', 'Some', 'All']}))
    return pd.DataFrame(rows).set_index('policy')


def pair_extreme_sources(raw, pairs, limit_each=2):
    """Locate both signed gap extremes in original bid/ask spread and quantity rows.

    Repeated grid observations of one candidate-state pair count once here. Boundary
    flags identify the raw candidates creating the extrema; other raw candidates in
    those messages remain visible. This diagnoses provenance without deleting rows.
    """
    columns = ['extreme', 'time', 'firm', 'side', 'quote_timestamp_ET', 'spread',
               'quantity', 'boundary_candidate', 'gap_bound_bps']
    if 'cusip' in raw and raw.cusip.nunique() > 1:
        raise ValueError('pair_extreme_sources expects raw rows for one bond')
    if limit_each < 1:
        return pd.DataFrame(columns=columns)
    distinct = pairs.loc[pairs.fresh_pair].drop_duplicates(
        ['firm', 'quote_timestamp_ET_bid', 'quote_timestamp_ET_ask'])
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
                    value = pd.to_numeric(pd.Series([source.spread]), errors='coerce').iloc[0]
                    sources.append(dict(extreme=label, time=row.time, firm=row.firm, side=side,
                        quote_timestamp_ET=stamp, spread=source.spread, quantity=source.quantity,
                        boundary_candidate=bool(np.isfinite(value) and value == getattr(row, bounds[side]+'_'+side)),
                        gap_bound_bps=getattr(row, field)))
    return pd.DataFrame(sources, columns=columns)


def pair_features(pairs, times):
    """Dealer-equal observed gaps; unknown quantities never become matched-size pairs."""
    index = pd.DatetimeIndex(times).sort_values().unique()
    cols = ['n_pair','n_size_time_pair','pair_gap','pair_gap_low','pair_gap_high','pair_mid',
            'pair_mid_range','pair_cross_some','pair_cross_all','pair_time_gap','pair_unknown_size',
            'size_time_gap','size_time_mid','size_time_cross_some','size_time_cross_all']
    f = pd.DataFrame(np.nan,index=index,columns=cols)
    f[['n_pair','n_size_time_pair']] = 0
    for eligible,prefix,cross_field,gap_field,mid_field in [
        ('fresh_pair','pair','cross','gap_center','mid'),
        ('size_time_pair','size_time','cross_matched','matched_gap','matched_mid')]:
        q = pairs.loc[pairs[eligible]].copy()
        if q.empty: continue
        q['cross_some'] = q[cross_field].eq('Some').astype(float)
        q['cross_all'] = q[cross_field].eq('All').astype(float)
        g = q.groupby('time',sort=False)
        counts = g.size()
        count_column = 'n_pair' if prefix=='pair' else 'n_size_time_pair'
        f.loc[counts.index,count_column] = counts
        mapping = {prefix+'_gap':gap_field,prefix+'_mid':mid_field,
                   prefix+'_cross_some':'cross_some',prefix+'_cross_all':'cross_all'}
        if prefix=='pair':
            q['unknown_size'] = q.positive_matches.eq(0).astype(float)
            g = q.groupby('time',sort=False)
            mapping.update(pair_gap_low='gap_low',pair_gap_high='gap_high',
                           pair_mid_range='mid_range',pair_unknown_size='unknown_size')
            f.loc[counts.index,'pair_time_gap'] = g.time_gap.median()
        for output,source in mapping.items(): f.loc[counts.index,output] = g[source].mean()
    f.index.name='time'
    return f


def empty_quote_features(times):
    """The same zero-count/NaN schema as real as-of calculations, with no dealer work."""
    index = pd.DatetimeIndex(times).sort_values().unique()
    sides = ['center_equal','n_dealers','n_fresh_dealers','n_incomplete','center_decay',
             'center_max_age','dispersion_bps','mean_candidate_gap','multi_fraction',
             'zero_quantity_fraction','unknown_quantity_fraction','median_message_age_min',
             'median_change_age_min','unknown_change_age_fraction','center_lower','center_upper',
             'max_decay_weight_share','decay_effective_dealers','center_candidate_clip',
             'center_dealer_downweight','n_peer_supported','n_clipped_dealers','n_changed_centers']
    counts = ['n_dealers','n_fresh_dealers','n_incomplete','n_peer_supported','n_clipped_dealers','n_changed_centers']
    pieces = []
    for side in ['bid', 'ask']:
        f = pd.DataFrame(np.nan, index=index, columns=sides)
        # Match side_features_fast's integer count dtypes as well as its values.
        for count in counts:
            f[count] = np.zeros(len(index), dtype=int)
        pieces.append(f.add_prefix('bcq_'+side+'_'))
    empty_pairs = pd.DataFrame(columns=['fresh_pair', 'size_time_pair'])
    pieces.append(pair_features(empty_pairs, index).add_prefix('bcq_'))
    result = pd.concat(pieces, axis=1)
    result['bcq_has_quote'] = 0.0
    result.index.name = 'time'
    return result


def build_quote_features(quotes, queries, age_min=30, sync_min=1, allow_exact=True, progress=None, event_cache=None):
    """Preserve every query. Optional prepare_quote_events result avoids repeated event work.

    Timings and skipped no-state query counts are attached to result.attrs. Empty
    bond/day and before-first queries skip both side matrices and pair snapshots;
    latest incomplete messages still use the full path to preserve n_incomplete.
    """
    if queries.row_id.duplicated().any() or queries[['row_id','cusip','time']].isna().any().any():
        raise ValueError('Queries require unique row_id and nonmissing cusip/time')
    if age_min <= 0 or sync_min < 0:
        raise ValueError('age_min must be positive and sync_min nonnegative')
    started = perf_counter()
    cache_reused = event_cache is not None
    query_groups=queries.groupby('cusip',sort=False,observed=True)
    if progress is not None:
        progress('features', 0, query_groups.ngroups, f'{len(queries):,} trade queries across {query_groups.ngroups:,} bonds')
    if queries.empty:
        result = queries.copy()
        result.attrs['quote_feature_timings'] = dict(event_prepare_s=0.0, asof_s=0.0,
            total_s=perf_counter()-started, no_state_unique_queries=0, event_cache_reused=cache_reused)
        return result
    if event_cache is None:
        relevant = quotes.loc[quotes.cusip.isin(queries.cusip.unique())]
        event_cache = prepare_quote_events(relevant, progress=progress)
    all_events = event_cache['events']
    prepared = perf_counter()
    output=[]
    grouped=all_events.groupby('cusip',observed=True)
    skipped = 0
    for completed,(bond,q) in enumerate(query_groups, 1):
        if progress is not None:
            progress('features', completed-1, query_groups.ngroups, f'Bond {bond}: {len(q):,} trade queries')
        events=grouped.get_group(bond) if bond in grouped.groups else all_events.iloc[:0]
        times=pd.DatetimeIndex(q.time).sort_values().unique()
        first = events.groupby('day', observed=True)[KEYS[-1]].min()
        first_at_query = pd.Series(times.normalize(), index=times).map(first)
        active = first_at_query.notna() & (first_at_query.le(times) if allow_exact else first_at_query.lt(times))
        active_times = times[active.to_numpy()]
        skipped += int((~active).sum())
        f = empty_quote_features(times)
        if len(active_times):
            pieces=[]
            for side in ['bid','ask']:
                side_f=side_features_fast(events.loc[events.side.eq(side)],active_times,age_min,allow_exact)
                pieces.append(side_f.add_prefix(f'bcq_{side}_'))
            p=pair_snapshots(events,active_times,age_min,sync_min,allow_exact)
            pieces.append(pair_features(p,active_times).add_prefix('bcq_'))
            active_f=pd.concat(pieces,axis=1)
            active_f['bcq_has_quote']=(active_f.bcq_bid_n_dealers.add(active_f.bcq_ask_n_dealers).gt(0)).astype(float)
            f.loc[active_times] = active_f
        joined=q[['row_id','cusip','time']].merge(f,left_on='time',right_index=True,how='left',validate='many_to_one')
        output.append(joined)
        if progress is not None:
            progress('features', completed, query_groups.ngroups, f'Finished bond {bond}: {len(q):,} trade queries')
    result = pd.concat(output,ignore_index=True).set_index('row_id').reindex(queries.row_id).reset_index() if output else queries.copy()
    result.attrs['quote_feature_timings'] = dict(event_prepare_s=prepared-started,
        asof_s=perf_counter()-prepared, total_s=perf_counter()-started,
        no_state_unique_queries=skipped, event_cache_reused=cache_reused,
        **event_cache.get('timings', {}))
    return result


def side_features_fast(events, times, age_min=30, allow_exact=True):
    """Vectorized query-by-dealer summaries; same definitions as Step 3, without momentum."""
    if age_min <= 0: raise ValueError('age_min must be positive')
    times=pd.DatetimeIndex(times).sort_values().unique()
    groups=list(events.groupby('firm',observed=True)); n,d=len(times),len(groups)
    names=['center','lo','hi','gap','count','age','change_age','zero','unknown','mid_low','mid_high']
    a={k:np.full((n,d),np.nan) for k in names}
    present=np.zeros((n,d),bool); valid=np.zeros((n,d),bool)
    query_ns=times.as_unit('ns').asi8; days=times.normalize().asi8
    for j,(_,h) in enumerate(groups):
        h=h.sort_values(KEYS[-1]); stamp=h[KEYS[-1]].array.as_unit('ns').asi8
        idx=np.searchsorted(stamp,query_ns,side='right' if allow_exact else 'left')-1
        z=h.iloc[np.maximum(idx,0)]
        present[:,j]=(idx>=0)&(z.day.array.as_unit(times.unit).asi8==days)
        valid[:,j]=present[:,j]&z.complete.to_numpy(dtype=bool)
        for key,col in [('center','center'),('lo','lo'),('hi','hi'),('gap','gap'),('count','candidate_count')]:
            a[key][:,j]=z[col].to_numpy()
        a['age'][:,j]=(query_ns-stamp[np.maximum(idx,0)])/60e9
        last=z.last_change.array.as_unit('ns').asi8
        a['change_age'][:,j]=np.where(z.last_change.notna(),(query_ns-last.astype(float))/60e9,np.nan)
        a['zero'][:,j]=z.quantity_set.map(lambda x:'Zero' in x).to_numpy()
        a['unknown'][:,j]=z.quantity_set.map(lambda x:any(t=='Missing' or t.startswith('Other:') for t in x)).to_numpy()
        for key,offset in [('mid_low',-1),('mid_high',0)]:
            values=h.spread_set.map(lambda x:x[(len(x)+offset)//2] if x else np.nan).to_numpy()
            a[key][:,j]=values[np.maximum(idx,0)]
    for k in a: a[k][~valid]=np.nan
    count=valid.sum(axis=1); fresh=valid&(a['age']<=age_min)
    def mean(x,mask=valid):
        return np.divide(np.nansum(np.where(mask,x,np.nan),axis=1),mask.sum(axis=1),out=np.full(n,np.nan),where=mask.sum(axis=1)>0)
    def median(x):
        result=np.full(n,np.nan); have=np.isfinite(x).any(axis=1)
        if have.any(): result[have]=np.nanmedian(x[have],axis=1)
        return result
    f=pd.DataFrame(index=times)
    f['center_equal']=mean(a['center']); f['n_dealers']=count
    f['n_fresh_dealers']=fresh.sum(axis=1); f['n_incomplete']=(present&~valid).sum(axis=1)
    min_age=np.min(np.where(valid,a['age'],np.inf),axis=1) if d else np.full(n,np.inf)
    weights=np.where(valid,np.exp2(-(a['age']-min_age[:,None])/age_min),0)
    weight_sum=weights.sum(axis=1)
    f['center_decay']=np.divide(np.nansum(a['center']*weights,axis=1),weight_sum,out=np.full(n,np.nan),where=weight_sum>0)
    f['center_max_age']=mean(a['center'],fresh)
    f['dispersion_bps']=np.sqrt(mean((a['center']-f.center_equal.to_numpy()[:,None])**2))
    f['mean_candidate_gap']=mean(a['gap']); f['multi_fraction']=mean(a['count']>1)
    f['zero_quantity_fraction']=mean(a['zero']); f['unknown_quantity_fraction']=mean(a['unknown'])
    f['median_message_age_min']=median(a['age']); f['median_change_age_min']=median(a['change_age'])
    f['unknown_change_age_fraction']=mean(~np.isfinite(a['change_age']))
    f['center_lower']=mean(a['lo']); f['center_upper']=mean(a['hi'])
    f['max_decay_weight_share']=np.divide(weights.max(axis=1) if d else np.zeros(n),weight_sum,out=np.full(n,np.nan),where=weight_sum>0)
    f['decay_effective_dealers']=np.divide(weight_sum**2,(weights**2).sum(axis=1),out=np.full(n,np.nan),where=weight_sum>0)
    clipped=a['center'].copy(); dw=np.where(valid,1.0,0.0)
    supported=np.zeros((n,d),bool); clipped_any=supported.copy(); center_changed=supported.copy()
    for j in range(d):
        peer_mask=fresh.copy(); peer_mask[:,j]=False
        ok=valid[:,j]&(peer_mask.sum(axis=1)>=MIN_PEERS)
        if not ok.any(): continue
        peers=np.where(peer_mask[ok],a['center'][ok],np.nan)
        ref=np.nanmedian(peers,axis=1)
        radius=np.maximum(CLIP_FLOOR_BPS,MAD_MULTIPLIER*1.4826*np.nanmedian(abs(peers-ref[:,None]),axis=1))
        clipped[ok,j]=(np.clip(a['mid_low'][ok,j],ref-radius,ref+radius)+np.clip(a['mid_high'][ok,j],ref-radius,ref+radius))/2
        residual=abs(a['center'][ok,j]-ref)
        dw[ok,j]=np.minimum(1,np.divide(radius,residual,out=np.ones(len(ref)),where=residual>0))
        supported[ok,j]=True
        clipped_any[ok,j]=(a['lo'][ok,j]<ref-radius)|(a['hi'][ok,j]>ref+radius)
        center_changed[ok,j]=~np.isclose(clipped[ok,j],a['center'][ok,j],rtol=0,atol=1e-9)
    f['center_candidate_clip']=mean(clipped)
    f['center_dealer_downweight']=np.divide(np.nansum(a['center']*dw,axis=1),dw.sum(axis=1),out=np.full(n,np.nan),where=dw.sum(axis=1)>0)
    f['n_peer_supported']=supported.sum(axis=1); f['n_clipped_dealers']=clipped_any.sum(axis=1); f['n_changed_centers']=center_changed.sum(axis=1)
    f.index.name='time'
    return f


BASE_FEATURES = ['D_CPP_BM_SPREAD','COUPON','D_CDX_TRADE','MEAN_ISSUER_SPREAD_DEV',
                 'NUM_OF_ISSUER_TRADES_SINCE_PREV','PREV_BM_SPREAD','PREV_QUANTITY',
                 'PREV_TRADE_TYPE','QUANTITY','TRADE_TYPE','YRS_TO_MATURITY','BM_YIELD_STD',
                 'D_SHORT_TO_BM_SPREAD','PREV_BM_SPREAD_STD_GROUP_BY_TYPE']
BASE_CAT_FEATURES = ['PREV_TRADE_TYPE','TRADE_TYPE']


def model_versions(frame):
    """Shared anchor/target/base; replace only quote levels when comparing cleaning rules."""
    x=frame.copy()
    levels=[]; reliability=[]
    for side in ['bid','ask']:
        prefix=f'bcq_{side}_'
        gap=prefix+'anchor_gap'
        x[gap]=x[prefix+'center_equal']-100*x.PREV_BM_SPREAD
        levels += [gap,prefix+'n_dealers']
        reliability += [prefix+k for k in ['mean_candidate_gap','dispersion_bps','median_message_age_min',
            'median_change_age_min','unknown_change_age_fraction','zero_quantity_fraction',
            'unknown_quantity_fraction','decay_effective_dealers','n_peer_supported']]
    levels += ['bcq_has_quote']
    common=BASE_FEATURES+levels+reliability
    specs={'Base':(BASE_FEATURES,{}),'Quote levels':(BASE_FEATURES+levels,{}),'Reliability':(common,{})}
    for label,column in [('Age decay','center_decay'),('Max age','center_max_age'),
                         ('Candidate clip','center_candidate_clip'),('Dealer downweight','center_dealer_downweight')]:
        replacements={f'bcq_{side}_anchor_gap':x[f'bcq_{side}_{column}']-100*x.PREV_BM_SPREAD for side in ['bid','ask']}
        if label=='Max age':
            replacements.update({f'bcq_{side}_n_dealers':x[f'bcq_{side}_n_fresh_dealers'] for side in ['bid','ask']})
            replacements['bcq_has_quote']=(x.bcq_bid_n_fresh_dealers+x.bcq_ask_n_fresh_dealers).gt(0).astype(float)
        specs[label]=(common,replacements)
    paired=['bcq_'+k for k in ['n_pair','n_size_time_pair','pair_gap','pair_gap_low','pair_gap_high',
                              'pair_mid_range','pair_cross_some','pair_cross_all','pair_time_gap',
                              'pair_unknown_size','size_time_gap','size_time_cross_some','size_time_cross_all']]
    specs['Paired']= (common+paired,{})
    return x,specs


def chronological_split(frame, quote_end, val_days=5, test_days=5, embargo_days=2, min_train_days=10):
    """One fixed final test block. Pilot defaults reflect the short quote file, not tuning."""
    if min(val_days,test_days,min_train_days) < 1 or embargo_days < 0 or pd.isna(quote_end):
        raise ValueError('Need positive train/validation/test lengths, nonnegative embargo and a quote end date')
    dates=pd.DatetimeIndex(frame.time.dt.normalize().unique()).sort_values()
    end=to_ny_datetime(pd.Series([quote_end])).iloc[0].normalize()
    dates=dates[dates<=end]
    v=len(dates)-val_days-test_days; t=len(dates)-test_days
    if v-embargo_days < min_train_days:
        raise ValueError(f'Need >= {min_train_days+embargo_days+val_days+test_days} distinct trade dates through quote end; found {len(dates)}. Set explicit research window lengths.')
    day=frame.time.dt.normalize()
    split=pd.Series('Outside experiment',index=frame.index)
    split.loc[day.isin(dates[:v-embargo_days])]='Train'
    split.loc[day.isin(dates[v:t])]='Validation'
    split.loc[day.isin(dates[t:])]='Test'
    # Final refit uses all earlier data except the same embargo before test.
    refit=day.isin(dates[:t-embargo_days])
    return split,refit


def run_comparison(frame, params, stage='Validation', selected=None, progress=None):
    """Predict D_BM_SPREAD + PREV_BM_SPREAD; same target rows for every version."""
    import lightgbm as lgb
    if stage not in ['Validation','Test']: raise ValueError('Unknown evaluation stage')
    x,specs=model_versions(frame)
    if stage=='Test':
        if selected not in specs: raise ValueError('Choose a validation-selected version before testing')
        reference='Base' if selected in ['Base','Quote levels'] else ('Quote levels' if selected=='Reliability' else 'Reliability')
        specs={k:specs[k] for k in dict.fromkeys(['Base',reference,selected])}
    eligible=np.isfinite(x[['D_BM_SPREAD','PREV_BM_SPREAD','BM_SPREAD']]).all(axis=1)
    train=eligible & (x['split'].eq('Train') if stage=='Validation' else x['refit_train'])
    evaluate=eligible & x['split'].eq(stage)
    if train.sum()<20 or evaluate.sum()==0: raise ValueError('Insufficient common training or evaluation target rows')
    if not x.loc[train,'time'].max() < x.loc[evaluate,'time'].min(): raise ValueError('Training must precede evaluation')
    if not x.loc[train,'bcq_has_quote'].gt(0).any() or not x.loc[evaluate,'bcq_has_quote'].gt(0).any():
        raise ValueError('Quote increment is unassessed: training or evaluation has no covered trades. Inspect the split/coverage dashboard.')
    predictions=[]; models={}
    for completed,(name,(columns,replacements)) in enumerate(specs.items()):
        if progress is not None:
            progress('models', completed, len(specs), f'Preparing model {completed+1}/{len(specs)}: {name}')
        z=x[columns].copy()
        for col,value in replacements.items(): z[col]=value
        for col in columns:
            if col in BASE_CAT_FEATURES:
                categories=pd.Index(z.loc[train,col].dropna().unique())
                z[col]=pd.Categorical(z[col],categories=categories)
            else: z[col]=pd.to_numeric(z[col],errors='coerce').replace([np.inf,-np.inf],np.nan)
        model=lgb.LGBMRegressor(**params)
        fit_kwargs={}
        if progress is not None:
            progress('fit', 0, model.n_estimators, f'{name}: training')
            def report_iteration(env):
                done=env.iteration-env.begin_iteration+1
                total=env.end_iteration-env.begin_iteration
                if done % 10 == 0 or done == total:
                    progress('fit', done, total, f'{name}: iteration {done:,}/{total:,}')
            report_iteration.order=20
            report_iteration.before_iteration=False
            fit_kwargs['callbacks']=[report_iteration]
        model.fit(z.loc[train],x.loc[train,'D_BM_SPREAD'],categorical_feature=BASE_CAT_FEATURES,**fit_kwargs)
        if progress is not None:
            progress('models', completed, len(specs), f'Predicting {stage.lower()} rows: {name}')
        pred=model.predict(z.loc[evaluate])+x.loc[evaluate,'PREV_BM_SPREAD'].to_numpy()
        out=x.loc[evaluate,['row_id','time','TRADE_TYPE','QUANTITY','bcq_has_quote','bcq_n_pair','bcq_n_size_time_pair']].copy()
        out['model']=name;out['stage']=stage;out['pred_spread']=pred
        out['error_bps']=(pred-x.loc[evaluate,'BM_SPREAD'].to_numpy())*100
        out['abs_error_bps']=abs(out.error_bps)
        out['train_n']=int(train.sum())
        predictions.append(out); models[name]=model
        if progress is not None:
            progress('models', completed+1, len(specs), f'Finished model {completed+1}/{len(specs)}: {name}')
    return pd.concat(predictions,ignore_index=True),models
