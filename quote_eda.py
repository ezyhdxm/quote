# %% [markdown]
# # Quote EDA — choose an issuer, inspect the cleaned quotes
# Run three cells in order, then switch issuer in the dropdown. `data_ig` is your
# three-month trade universe; `bcq_df` contains the shorter quote history.
# Goal: obtain a usable dealer consensus and check whether bonds move together.
# The tabs answer: **what was removed**, **what is the cleaned quote**, and
# **is there enough joint movement**. No large tables are printed.

# %% 1. Settings and inputs
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display
import ipywidgets as widgets

C = {
    "ISSUER": None, "START": None, "END": None,
    "QUOTES_PATH": None, "TRADES_PATH": None, "BOND_INFO_PATH": None,
    "TRADE_UNIVERSE_START": None, "TRADE_UNIVERSE_END": None,
    "GRID": "5min", "MAX_AGE": "30min", "SESSION": ("08:00", "17:00"),
    "Q_NAIVE_TZ": "America/New_York", "T_NAIVE_TZ": "America/New_York",
    "QUOTE_SPREAD_MULTIPLIER": 1., "TRADE_SPREAD_MULTIPLIER": 100.,
    "QUOTE_SEQUENCE_COL": None, "TIE_TOL_BPS": 2.,
    "NONPOSITIVE_QUANTITY_IS_MISSING": True, "ZERO_SPREAD_IS_MISSING": False,
    "PAIR_MAX_GAP": "5min", "PAIR_ALLOW_UNKNOWN_SIZE": True,
    "ZERO_PAIR_GAP_BPS": 10., "PEER_Z": 4., "NOISE_FLOOR_BPS": 2.,
    "MIN_PEER_DEALERS": 3, "MAX_ANALYSIS_BONDS": 40,
    "CHANGE_LAG": "30min", "ANALYSIS_SIDE": "bid",
    "MIN_MATCHED_DEALERS": 2, "MIN_COMMON_OBS": 30,
    "MAX_RAW_MARKERS": 300, "SHOW_PLOTS": True, "SHOW_WIDGETS": True,
    "USE_DEMO": False, **globals().get("EDA_OVERRIDES", {}),
}
Q_COL = {"cusip": "cusip", "firm": "firm", "side": "side",
         "event_time": "quote_timestamp_ET", "spread": "spread", "quantity": "quantity"}
T_COL = {"cusip": "CUSIP", "time": "EFFECTIVE_DATETIME_TS", "spread": "BM_SPREAD",
         "quantity": "QUANTITY", "type": "EFF_SIDE"}
M_COL = {"cusip": "CUSIP", "issuer": "ISSUER", "tenor": "YRS_TO_MATURITY"}
if C["USE_DEMO"]:
    from demo_data import make_demo
    bcq_df, data_ig, bond_info_df = make_demo()
for setting, name in [("QUOTES_PATH", "bcq_df"), ("TRADES_PATH", "data_ig"), ("BOND_INFO_PATH", "bond_info_df")]:
    if C[setting]:
        globals()[name] = pd.read_parquet(C[setting])

# %% [markdown]
# ## Prepare once
# Keep only CUSIPs traded in the supplied three-month table. Existing `.isin`
# filtering is fine. Quote START/END does not change this universe.
# Availability is assumed to equal event time. Quantities stay in source units.

# %% 2. Normalize inputs and create issuer choices
# One timezone helper is shared by quotes, trades and date settings.
def to_et(values, zone):
    dt = pd.to_datetime(values, errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(zone, ambiguous="NaT", nonexistent="NaT")
    return dt.dt.tz_convert("America/New_York")

if Q_COL["event_time"] not in bcq_df and "quote_timestamp_UTC" in bcq_df:
    bcq_df = bcq_df.assign(quote_timestamp_ET=pd.to_datetime(
        bcq_df["quote_timestamp_UTC"], utc=True, errors="coerce").dt.tz_convert("America/New_York"))
q = bcq_df.reindex(columns=list(Q_COL.values())).rename(columns={v: k for k, v in Q_COL.items()}).copy()
t = data_ig.reindex(columns=list(T_COL.values())).rename(columns={v: k for k, v in T_COL.items()}).copy()
for frame in [q, t]:
    frame["cusip"] = frame["cusip"].astype("string").str.strip().replace("", pd.NA)
    for column in ["spread", "quantity"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
q["spread"] *= C["QUOTE_SPREAD_MULTIPLIER"]
t["spread"] *= C["TRADE_SPREAD_MULTIPLIER"]
q["event_time"] = to_et(q["event_time"], C["Q_NAIVE_TZ"])
t["time"] = to_et(t["time"], C["T_NAIVE_TZ"])
q["firm"] = q["firm"].astype("string").str.strip().replace("", pd.NA)
q["side"] = q["side"].astype("string").str.strip().str.lower()
t = t.dropna(subset=["cusip", "time"])
for setting, comparison in [("TRADE_UNIVERSE_START", "ge"), ("TRADE_UNIVERSE_END", "le")]:
    if C[setting]:
        bound = to_et(pd.Series([C[setting]]), C["T_NAIVE_TZ"]).iloc[0]
        t = t.loc[getattr(t["time"], comparison)(bound)]
trade_universe_cusips = pd.Index(t["cusip"].unique(), name="cusip")
q = q.loc[q["cusip"].isin(trade_universe_cusips)].copy()
q["input_row"] = np.arange(len(q))
q["sequence"] = pd.to_numeric(bcq_df.loc[q.index, C["QUOTE_SEQUENCE_COL"]], errors="coerce") if C["QUOTE_SEQUENCE_COL"] else np.nan
q["quantity_raw"] = q["quantity"]
if C["NONPOSITIVE_QUANTITY_IS_MISSING"]:
    q["quantity"] = q["quantity"].where(q["quantity"].gt(0))
# Metadata priority: security master, quotes, then trades. Metadata does not clean spreads.
meta_parts = []
for source, cusip_col in [(data_ig, T_COL["cusip"]), (bcq_df, Q_COL["cusip"]), (globals().get("bond_info_df"), M_COL["cusip"])]:
    if source is not None and M_COL["issuer"] in source:
        part = source.reindex(columns=[cusip_col, M_COL["issuer"], M_COL["tenor"]]).copy()
        part.columns = ["cusip", "issuer", "tenor"]
        meta_parts.append(part)
meta = pd.concat(meta_parts, ignore_index=True).dropna(subset=["cusip", "issuer"])
meta["cusip"] = meta["cusip"].astype("string").str.strip()
meta["issuer"] = meta["issuer"].astype("string").str.strip()
meta["tenor"] = pd.to_numeric(meta["tenor"], errors="coerce")
meta = meta.drop_duplicates("cusip", keep="last").set_index("cusip")
q["issuer"] = q["cusip"].map(meta["issuer"])
issuer_counts = q.groupby("issuer").size().sort_values(ascending=False)
if issuer_counts.empty:
    raise ValueError("No traded CUSIPs have both quotes and issuer metadata.")
trade_counts = t.groupby("cusip").size()

# %% [markdown]
# ## Cleaning rules and reading the charts
# 1. Remove exact duplicate messages. At the same timestamp, use a supplied feed
#    sequence; otherwise take the median only if spreads agree within 2 bps.
#    Larger conflicts invalidate that latest state; never fall back to an older quote.
# 2. Carry the latest event strictly before each grid point, only for MAX_AGE.
# 3. For close-time same-dealer bid/ask with compatible or unknown size: a single
#    zero opposite a spread at least 10 bps away is suspect. Exclude that zero leg.
#    Other crossed pairs exclude both legs because the bad side is unidentified.
# 4. Remove cross-dealer outliers only with at least three same-side dealers.
#    Sign alone never invalidates a spread. One-sided quotes remain usable.
# 5. For plotting, prefer the same set of noncrossed paired dealers on both sides.
#    If independent side medians cross without such pairs, withhold both medians.
# Raw records remain inspectable. These are research heuristics, not vendor truth.
# Solid curves are cleaned snapshot medians. Optional dotted EWMA is display only,
# resets after missing points/overnight, and never fills a gap or changes analysis.

# %% 3. Issuer dashboard — dropdown, three tabs, no dataframe dumps
# A callback is necessary for the dropdown; all issuer work lives in this one function.
def inspect_issuer(issuer):
    raw = q.loc[q["issuer"].eq(issuer)].copy()
    bonds = raw.groupby("cusip").size().sort_values(ascending=False).index
    if C["MAX_ANALYSIS_BONDS"] is not None:
        bonds = bonds[:C["MAX_ANALYSIS_BONDS"]]
    raw = raw.loc[raw["cusip"].isin(bonds)].copy()
    start = to_et(pd.Series([C["START"]]), C["Q_NAIVE_TZ"]).iloc[0] if C["START"] else raw["event_time"].min().floor(C["GRID"])
    end = to_et(pd.Series([C["END"]]), C["Q_NAIVE_TZ"]).iloc[0] if C["END"] else raw["event_time"].max().ceil(C["GRID"]) + pd.Timedelta(C["GRID"])
    full_grid = pd.date_range(start, end, freq=C["GRID"])
    grid = full_grid[(full_grid.dayofweek < 5)]
    grid = grid[grid.indexer_between_time(*C["SESSION"])]
    if grid.empty:
        raise ValueError("No weekday session points in START/END.")
    key = ["cusip", "firm", "side"]
    valid = raw[key + ["event_time"]].notna().all(axis=1) & raw["side"].isin(["bid", "ask"])
    messages = raw.loc[valid & raw["event_time"].le(end)].sort_values(["event_time", "input_row"])
    payload = key + ["event_time", "spread", "quantity_raw"]
    duplicate_count = int(messages.duplicated(payload).sum())
    messages = messages.drop_duplicates(payload, keep="last")
    batch_key = key + ["event_time"]
    batch = messages.groupby(batch_key, sort=False)
    replay = batch.agg(spread=("spread", "median"), quantity=("quantity", "first"),
        low=("spread", "min"), high=("spread", "max"), rows=("spread", "size"), finite=("spread", "count"))
    replay["quantity"] = replay["quantity"].where(batch["quantity"].nunique(dropna=False).eq(1))
    replay["tie_conflict"] = replay["rows"].gt(1) & ((replay["high"] - replay["low"]).gt(C["TIE_TOL_BPS"]) | replay["finite"].ne(replay["rows"]))
    if C["QUOTE_SEQUENCE_COL"]:
        winners = messages.loc[messages["sequence"].eq(batch["sequence"].transform("max"))]
        sequenced = winners.drop_duplicates(batch_key, keep="last").set_index(batch_key)
        covered = batch["sequence"].count().eq(replay["rows"]) & winners.groupby(batch_key).size().reindex(replay.index).eq(1)
        seq_keys = replay.index[covered]
        replay.loc[seq_keys, ["spread", "quantity"]] = sequenced.reindex(seq_keys)[["spread", "quantity"]]
        replay.loc[seq_keys, "tie_conflict"] = False
    replay.loc[replay["tie_conflict"], "spread"] = np.nan
    replay = replay.reset_index().sort_values("event_time", kind="stable")
    prev = replay.groupby(key)["spread"].shift()
    changed = ~(replay["spread"].eq(prev) | (replay["spread"].isna() & prev.isna())) | replay.groupby(key).cumcount().eq(0)
    replay["last_change"] = replay["event_time"].where(changed).groupby([replay[k] for k in key]).ffill()
    chunks = [pd.merge_asof(pd.DataFrame({"time": grid}), frame, left_on="time", right_on="event_time",
              direction="backward", allow_exact_matches=False).dropna(subset=["event_time"])
              for _, frame in replay.groupby(key, sort=False)]
    if not chunks:
        raise ValueError("No valid quote keys in the chosen issuer/window.")
    states = pd.concat(chunks, ignore_index=True)
    if states.empty:
        raise ValueError("No events precede the selected snapshots.")
    states["age_min"] = (states["time"] - states["event_time"]).dt.total_seconds() / 60
    states["change_age_min"] = (states["time"] - states["last_change"]).dt.total_seconds() / 60
    states["reason"] = np.select([states["age_min"].gt(pd.Timedelta(C["MAX_AGE"]).total_seconds()/60),
        states["tie_conflict"].eq(True), states["spread"].isna(), states["spread"].eq(0) & C["ZERO_SPREAD_IS_MISSING"]],
        ["expired", "timestamp conflict", "nonfinite", "explicit zero policy"], default="kept")
    fresh = states.loc[states["reason"].eq("kept")].copy()
    pk = ["time", "cusip", "firm"]
    pair = fresh.loc[fresh["side"].eq("bid")].merge(fresh.loc[fresh["side"].eq("ask")], on=pk, suffixes=("_bid", "_ask"))
    pair["unknown_size"] = pair["quantity_bid"].isna() | pair["quantity_ask"].isna()
    pair["same_size"] = pair["quantity_bid"].gt(0) & pair["quantity_bid"].eq(pair["quantity_ask"])
    gap = (pair["event_time_bid"] - pair["event_time_ask"]).abs()
    valid_sizes = (pair["quantity_bid"].gt(0) | pair["quantity_bid"].isna()) & (pair["quantity_ask"].gt(0) | pair["quantity_ask"].isna())
    pair["eligible"] = gap.le(pd.Timedelta(C["PAIR_MAX_GAP"])) & valid_sizes & (pair["same_size"] | (pair["unknown_size"] & C["PAIR_ALLOW_UNKNOWN_SIZE"]))
    pair["crossed"] = pair["eligible"] & pair["spread_bid"].lt(pair["spread_ask"])
    pair["zero_bid"] = pair["eligible"] & pair["spread_bid"].eq(0) & pair["spread_ask"].abs().ge(C["ZERO_PAIR_GAP_BPS"])
    pair["zero_ask"] = pair["eligible"] & pair["spread_ask"].eq(0) & pair["spread_bid"].abs().ge(C["ZERO_PAIR_GAP_BPS"])
    for side in ["bid", "ask"]:
        flags = pair[pk].copy()
        flags["side"] = side
        flags["pair_reason"] = np.select([pair[f"zero_{side}"], pair["crossed"] & ~(pair["zero_bid"] | pair["zero_ask"])],
                                         ["contextual zero", "crossed dealer pair"], default="kept")
        flagged = flags.loc[flags["pair_reason"].ne("kept")].set_index(pk + ["side"])["pair_reason"]
        lookup = pd.MultiIndex.from_frame(states[pk + ["side"]]).map(flagged)
        states.loc[states["reason"].eq("kept") & pd.notna(lookup), "reason"] = lookup[pd.notna(lookup) & states["reason"].eq("kept").to_numpy()]
    candidates = states.loc[states["reason"].eq("kept")].copy()
    gk = ["time", "cusip", "side"]
    center = candidates.groupby(gk)["spread"].transform("median")
    residual = candidates["spread"] - center
    mad = residual.abs().groupby([candidates[k] for k in gk]).transform("median")
    candidates["peer_n"] = candidates.groupby(gk)["firm"].transform("nunique")
    candidates["peer_flag"] = candidates["peer_n"].ge(C["MIN_PEER_DEALERS"]) & residual.abs().gt(C["PEER_Z"] * (1.4826*mad).clip(lower=C["NOISE_FLOOR_BPS"]))
    states.loc[candidates.index[candidates["peer_flag"]], "reason"] = "peer outlier"
    clean = candidates.loc[~candidates["peer_flag"]].copy()
    consensus = clean.groupby(gk).agg(median=("spread", "median"), n=("firm", "nunique")).reset_index()
    levels = consensus.pivot(index=["time", "cusip"], columns="side", values="median").reindex(columns=["bid", "ask"])
    good_keys = pd.MultiIndex.from_frame(clean[pk + ["side"]])
    pair["clean"] = pair["eligible"] & ~pair["crossed"] & pd.MultiIndex.from_frame(pair[pk].assign(side="bid")).isin(good_keys) & pd.MultiIndex.from_frame(pair[pk].assign(side="ask")).isin(good_keys)
    paired_levels = pair.loc[pair["clean"]].groupby(["time", "cusip"])[["spread_bid", "spread_ask"]].median().rename(columns={"spread_bid":"bid", "spread_ask":"ask"})
    levels.update(paired_levels)
    aggregate_cross = levels["bid"].lt(levels["ask"])
    levels.loc[aggregate_cross, ["bid", "ask"]] = np.nan
    coverage = pd.DataFrame(index=bonds).join(meta[["tenor"]]).assign(n_trades=trade_counts.reindex(bonds, fill_value=0))
    coverage["window_trades"] = t.loc[t["time"].between(start, end)].groupby("cusip").size().reindex(bonds, fill_value=0)
    for side in ["bid", "ask"]:
        counts = clean.loc[clean["side"].eq(side)].groupby(["time", "cusip"])["firm"].nunique().unstack("cusip").reindex(index=grid, columns=bonds).fillna(0)
        coverage[f"{side}_fresh"] = counts.gt(0).mean()
        coverage[f"{side}_multi"] = counts.ge(2).mean()
    # Same dealer at both endpoints: no fill and no replacement by a new dealer.
    fields = ["time", "cusip", "firm", "side", "spread", "quantity"]
    old = clean[fields].copy()
    lag = pd.Timedelta(C["CHANGE_LAG"])
    old["time"] += lag
    matched = clean[fields].merge(old, on=["time", "cusip", "firm", "side"], suffixes=("", "_old"))
    matched = matched.loc[matched["time"].dt.normalize().eq((matched["time"]-lag).dt.normalize())].copy()
    matched["delta"] = matched["spread"] - matched["spread_old"]
    matched["moved"] = matched["delta"].abs().gt(1e-10)
    movement = matched.groupby(gk).agg(delta=("delta", "median"), n=("firm", "nunique"), moved_fraction=("moved", "mean"))
    movement["moved_median"] = matched.loc[matched["moved"]].groupby(gk)["delta"].median()
    movement["nonzero_delta"] = matched.loc[matched["spread"].ne(0) & matched["spread_old"].ne(0)].groupby(gk)["delta"].median()
    nz_n = matched.loc[matched["spread"].ne(0) & matched["spread_old"].ne(0)].groupby(gk)["firm"].nunique()
    movement["nonzero_delta"] = movement["nonzero_delta"].where(nz_n.ge(C["MIN_MATCHED_DEALERS"]))
    movement.loc[movement["n"].lt(C["MIN_MATCHED_DEALERS"]), ["delta", "moved_median"]] = np.nan
    selected = movement.reset_index().loc[lambda x:x["side"].eq(C["ANALYSIS_SIDE"])]
    changes = selected.pivot(index="time", columns="cusip", values="delta").reindex(index=grid, columns=bonds)
    changes_nonzero = selected.pivot(index="time", columns="cusip", values="nonzero_delta").reindex(index=grid, columns=bonds)
    overlap = changes.notna().astype("int64").T @ changes.notna().astype("int64")
    correlation = changes.corr(min_periods=C["MIN_COMMON_OBS"])
    correlation_nonzero = changes_nonzero.corr(min_periods=C["MIN_COMMON_OBS"])
    pca_cols = changes.columns[(changes.count().ge(C["MIN_COMMON_OBS"]) & changes.std().gt(0))]
    pca_cols = changes[pca_cols].count().sort_values(ascending=False).head(6).index
    complete = changes[pca_cols].dropna()
    complete = complete.loc[:, complete.std(ddof=0).gt(0)]
    variance, loadings = pd.Series(dtype=float), pd.Series(dtype=float)
    if len(complete) >= C["MIN_COMMON_OBS"] and complete.shape[1] >= 3:
        z = (complete-complete.mean())/complete.std(ddof=0)
        _, singular, vectors = np.linalg.svd(z.to_numpy(), full_matrices=False)
        variance = pd.Series(singular**2/(singular**2).sum(), index=[f"PC{i+1}" for i in range(len(singular))])
        loadings = pd.Series(vectors[0], index=complete.columns)
        loadings *= 1 if loadings.sum() >= 0 else -1
    # Tab 1: actionable quality information, instead of large audit tables.
    quality = make_subplots(rows=2, cols=2, subplot_titles=["Kept / withheld snapshot records", "Plotted quote coverage by day / bond",
                             "Fresh message vs unchanged-spread age", "30min movement: participation vs median suppression"])
    reasons = states["reason"].value_counts()
    reason_plot = reasons.drop("kept",errors="ignore").copy()
    reason_plot["kept: >=3 peers"] = int(clean["peer_n"].ge(C["MIN_PEER_DEALERS"]).sum())
    reason_plot["kept: <3 peers (unassessed)"] = int(clean["peer_n"].lt(C["MIN_PEER_DEALERS"]).sum())
    reason_plot = reason_plot.loc[reason_plot.gt(0)]
    quality.add_trace(go.Bar(x=reason_plot.index, y=reason_plot.values, name="snapshot records"), row=1,col=1)
    quality.update_yaxes(type="log",title_text="records (log scale)",row=1,col=1)
    order = coverage.sort_values("tenor").index
    available = levels.notna().any(axis=1).unstack("cusip").reindex(index=grid,columns=bonds).eq(True)
    daily_coverage = available.groupby(available.index.date).mean().T.reindex(order)
    quality.add_trace(go.Heatmap(z=daily_coverage,x=[str(day) for day in daily_coverage.columns],y=order,
        zmin=0,zmax=1, colorscale="Blues", showscale=False,
        hovertemplate="%{y}<br>%{x}<br>session snapshots with a plotted side: %{z:.1%}<extra></extra>"), row=1,col=2)
    for column, label in [("age_min","message age"),("change_age_min","spread-change age")]:
        quality.add_trace(go.Histogram(x=clean[column].clip(upper=120), histnorm="probability", xbins=dict(start=0,end=125,size=5), opacity=.55, name=label), row=2,col=1)
    activity = selected.groupby("cusip").agg(moved=("moved_fraction","mean"), zero_median=("delta",lambda x:x.dropna().eq(0).mean())).reindex(order)
    for column,label in [("moved","fraction of matched dealers moving"),("zero_median","fraction of medians equal to zero")]:
        quality.add_trace(go.Bar(x=order,y=activity[column],name=label),row=2,col=2)
    quality.update_yaxes(range=[0,1],row=2,col=2)
    quality.update_xaxes(title_text="minutes; 120 includes older",row=2,col=1)
    quality.update_layout(title=f"{issuer}: cleaning and coverage",height=780,template="plotly_white",barmode="group",legend=dict(orientation="h",y=-.2),margin=dict(b=140))
    # Tab 2: liquid/sparse comparisons, clean traces first; raw points are hidden.
    pool = coverage.sort_values("tenor").copy()
    pool["band"] = pd.qcut(np.arange(len(pool)), min(4,len(pool)), labels=False)
    panels = []
    for _, band in pool.groupby("band"):
        liquid = band["n_trades"].idxmax()
        other = band.drop(index=liquid)
        sparse = other["n_trades"].idxmin() if len(other) else None
        panels.extend([liquid,sparse])
    titles = [f"{b} | {coverage.loc[b,'tenor']:.1f}y | trades U/Q={coverage.loc[b,'n_trades']}/{coverage.loc[b,'window_trades']}" if b else "No second bond" for b in panels]
    explorer = make_subplots(rows=len(panels)//2,cols=2,subplot_titles=titles,vertical_spacing=.07)
    for panel,bond in enumerate(panels):
        if bond is None:
            continue
        row,col = panel//2+1,panel%2+1
        bond_levels = levels.xs(bond,level="cusip").reindex(full_grid) if bond in levels.index.get_level_values("cusip") else pd.DataFrame(index=full_grid,columns=["bid","ask"])
        trends = pd.DataFrame(index=full_grid)
        for side,color in [("bid","#2467a5"),("ask","#d47c27")]:
            y = bond_levels[side]
            block = y.isna().cumsum()
            trends[side] = y.groupby([y.index.date,block]).transform(lambda a:a.ewm(span=3,adjust=False).mean()).where(y.notna())
        trends.loc[trends["bid"].lt(trends["ask"]), ["bid","ask"]] = np.nan
        for side,color in [("bid","#2467a5"),("ask","#d47c27")]:
            explorer.add_trace(go.Scatter(x=full_grid.tz_localize(None),y=bond_levels[side],mode="lines",line=dict(color=color,shape="hv"),name=f"clean {side}",legendgroup=side,showlegend=panel==0,connectgaps=False),row=row,col=col)
            explorer.add_trace(go.Scatter(x=full_grid.tz_localize(None),y=trends[side],mode="lines",line=dict(color=color,dash="dot"),name=f"display trend {side}",meta="trend",visible=False,showlegend=panel==0,connectgaps=False),row=row,col=col)
        points = raw.loc[raw["cusip"].eq(bond) & raw["event_time"].between(start,end)]
        if len(points)>C["MAX_RAW_MARKERS"]:
            points = points.iloc[np.linspace(0,len(points)-1,C["MAX_RAW_MARKERS"]).astype(int)]
        explorer.add_trace(go.Scattergl(x=points["event_time"].dt.tz_localize(None),y=points["spread"],mode="markers",marker=dict(color="#999999",size=4),customdata=points[["firm","side","quantity_raw"]],hovertemplate="%{x}<br>%{y:.2f} bps<br>%{customdata[0]} %{customdata[1]} size=%{customdata[2]}<extra></extra>",name="raw messages",meta="raw",visible=False,showlegend=panel==0),row=row,col=col)
        trades = t.loc[t["cusip"].eq(bond) & t["time"].between(start,end)]
        for kind,color in [("B","#238b45"),("S","#c62828"),("D","#4f70a0")]:
            tt = trades.loc[trades["type"].eq(kind)]
            scale = max(1e-12,tt["quantity"].quantile(.95))
            sizes = 5+8*np.sqrt(tt["quantity"].fillna(0).clip(0,scale)/scale)
            explorer.add_trace(go.Scattergl(x=tt["time"].dt.tz_localize(None),y=tt["spread"],mode="markers",marker=dict(color=color,size=sizes),customdata=tt[["quantity"]],hovertemplate="%{x}<br>%{y:.2f} bps<br>size=%{customdata[0]}<extra></extra>",name=f"trade {kind}",legendgroup=kind,showlegend=panel==0),row=row,col=col)
    explorer.update_yaxes(title_text="bps")
    explorer.update_xaxes(matches="x",title_text="ET")
    explorer.update_layout(title=f"{issuer}: cleaned dealer consensus — gaps mean unavailable",height=280*(len(panels)//2)+180,template="plotly_white",legend=dict(orientation="h",y=-.12),margin=dict(t=90,b=110))
    # Tab 3: correlation with observation counts in hover, plus PCA loadings.
    movement_fig = make_subplots(rows=2,cols=2,subplot_titles=["Matched-dealer change correlation (hover: n)","Exact-zero endpoint exclusion sensitivity", "PCA explained variance", "PC1 loading versus tenor"])
    for column,matrix in [(1,correlation),(2,correlation_nonzero)]:
        obs = changes if column==1 else changes_nonzero
        n = obs.notna().astype("int64").T @ obs.notna().astype("int64")
        matrix = matrix.reindex(index=order,columns=order)
        movement_fig.add_trace(go.Heatmap(z=matrix,x=order,y=order,zmin=-1,zmax=1,colorscale="RdBu",showscale=False,customdata=n.reindex(index=order,columns=order),hovertemplate="%{y} / %{x}<br>corr=%{z:.2f}<br>n=%{customdata}<extra></extra>"),row=1,col=column)
    movement_fig.add_trace(go.Bar(x=variance.index,y=variance.values,name="variance fraction"),row=2,col=1)
    movement_fig.add_trace(go.Scatter(x=coverage["tenor"].reindex(loadings.index),y=loadings,mode="markers",text=loadings.index,hovertemplate="%{text}<br>tenor=%{x:.2f}y<br>loading=%{y:.3f}<extra></extra>",name="PC1 loading"),row=2,col=2)
    pca_note = f"PCA: {len(complete):,} complete rows, {complete.shape[1]} bonds" if len(variance) else f"PCA unavailable: {len(complete):,} complete rows / {complete.shape[1]} varying bonds; need >= {C['MIN_COMMON_OBS']} rows and 3 bonds"
    movement_fig.update_layout(title=f"{issuer}: {C['CHANGE_LAG']} {C['ANALYSIS_SIDE']} changes<br><sup>{pca_note}</sup>",height=820,template="plotly_white",margin=dict(t=100),showlegend=False)
    movement_fig.update_yaxes(range=[0,1],row=2,col=1)
    movement_fig.update_xaxes(title_text="years to maturity",row=2,col=2)
    audit = {"duplicates":duplicate_count,"conflicting_batches":int(replay["tie_conflict"].sum()),"suppressed_consensus_crossings":int(aggregate_cross.sum()),"reasons":reasons}
    return {"issuer":issuer,"raw":raw,"replay":replay,"states":states,"clean":clean,"pairs":pair,"levels":levels,"coverage":coverage,
            "matched":matched,"movement":movement,"changes":changes,"overlap":overlap,"correlation":correlation,"correlation_nonzero":correlation_nonzero,
            "pca_variance":variance,"pca_loadings":loadings,"audit":audit,"figures":[quality,explorer,movement_fig],"pca_note":pca_note}

issuer_box = widgets.Dropdown(options=[(f"{name} ({count:,} messages)",name) for name,count in issuer_counts.items()],
    value=C["ISSUER"] if C["ISSUER"] is not None else issuer_counts.index[0],description="Issuer:",layout=widgets.Layout(width="650px"))
raw_box = widgets.Checkbox(value=False,description="Show raw messages (audit)")
trend_box = widgets.Checkbox(value=False,description="Show display-only smooth trend")
status = widgets.Output()
outputs = [widgets.Output() for _ in range(3)]
tabs = widgets.Tab(children=outputs)
for i,label in enumerate(["1 · Cleaning / coverage","2 · Cleaned quotes","3 · Co-movement"]):
    tabs.set_title(i,label)

# Small UI wrapper: computation runs only when issuer changes, not on display toggles.
def refresh(change=None):
    global result
    needs_analysis = change is None or change.get("owner") is issuer_box or globals().get("result") is None
    with status:
        status.clear_output(wait=True)
        try:
            if needs_analysis:
                result = inspect_issuer(issuer_box.value)
        except Exception as error:
            result = None
            for out in outputs:
                out.clear_output(wait=False)
            print(f"Cannot render {issuer_box.value}: {error}")
            raise
        for trace in result["figures"][1].data:
            if trace.meta in ["raw","trend"]:
                trace.visible = raw_box.value if trace.meta=="raw" else trend_box.value
        a = result["audit"]
        print(f"{result['issuer']} | {len(result['coverage'])} bonds | {len(result['clean']):,} clean dealer snapshots | "
              f"duplicates {a['duplicates']:,}; conflicting batches {a['conflicting_batches']:,}; withheld consensus crossings {a['suppressed_consensus_crossings']:,}")
        print("Read: cleaning reasons -> cleaned curves -> correlation / n / loadings. Grey raw points are audit only.")
    if C["SHOW_PLOTS"]:
        for out,fig in zip(outputs,result["figures"]):
            with out:
                out.clear_output(wait=True)
                fig.show()

issuer_box.observe(refresh,names="value")
raw_box.observe(refresh,names="value")
trend_box.observe(refresh,names="value")
if C["SHOW_WIDGETS"]:
    display(widgets.VBox([issuer_box,widgets.HBox([raw_box,trend_box]),status,tabs]))
refresh()
