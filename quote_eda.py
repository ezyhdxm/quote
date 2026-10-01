# %% [markdown]
# # BondCliq quote EDA: latest dealer states and issuer co-movement
# Run in the same kernel as `bcq_df` and `data_ig`. Optional `bond_info_df`
# supplies CUSIP / ISSUER / YRS_TO_MATURITY when trade metadata is incomplete.
# Edit cell 1, then run the cells in order. All spread plots use **bps**.
# This is descriptive EDA; no issuer factor is fitted here.

# %% 1. Inputs and settings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display

EDA_CONFIG = {
    "QUOTES_PATH": None, "TRADES_PATH": None, "BOND_INFO_PATH": None,
    "TRADE_UNIVERSE_START": None, "TRADE_UNIVERSE_END": None,  # None uses supplied 3-month data_ig
    "ISSUER": None,                    # None selects the issuer with most quoted bonds
    "START": None, "END": None,        # ET, e.g. "2026-03-02 08:00"
    "GRID": "5min", "MAX_AGE": "30min",
    "SESSION": ("08:00", "17:00"),     # weekdays; not a market holiday calendar
    "Q_NAIVE_TZ": "America/New_York",  # screenshot: quote_timestamp_ET
    "T_NAIVE_TZ": "America/New_York",
    "QUOTE_SPREAD_MULTIPLIER": 1.0,     # screenshot: spread already in bps
    "TRADE_SPREAD_MULTIPLIER": 100.0,   # screenshot: BM_SPREAD * 100
    "QUOTE_SEQUENCE_COL": None,        # optional feed sequence for timestamp ties
    "NONPOSITIVE_QUANTITY_IS_MISSING": True,  # keeps quantity_raw for audit
    "ZERO_SPREAD_IS_MISSING": False,    # zero/negative spreads are valid by default
    "PAIR_MAX_GAP": "5min", "PAIR_ALLOW_UNKNOWN_SIZE": True,
    "NOISE_FLOOR_BPS": 2.0, "PEER_Z": 4.0, "MIN_PEER_DEALERS": 3,
    "CHANGE_LAG": "30min", "ANALYSIS_SIDE": "bid",
    "MIN_MATCHED_DEALERS": 2, "MIN_COMMON_OBS": 30,
    "MAX_ANALYSIS_BONDS": 40, "MAX_RAW_MARKERS": 2000,
    "SHOW_PLOTS": True, "USE_DEMO": False,
    **globals().get("EDA_OVERRIDES", {}),  # optional overrides for %run -i / checks
}
C = EDA_CONFIG
Q_COL = {"cusip": "cusip", "firm": "firm", "side": "side",
         "event_time": "quote_timestamp_ET", "spread": "spread",
         "quantity": "quantity"}
T_COL = {"cusip": "CUSIP", "time": "EFFECTIVE_DATETIME_TS",
         "spread": "BM_SPREAD", "quantity": "QUANTITY", "type": "EFF_SIDE"}
M_COL = {"cusip": "CUSIP", "issuer": "ISSUER", "tenor": "YRS_TO_MATURITY"}

if C["USE_DEMO"]:
    from demo_data import make_demo
    bcq_df, data_ig, bond_info_df = make_demo()
if C["QUOTES_PATH"] is not None:
    bcq_df = pd.read_parquet(C["QUOTES_PATH"])  # already trade-CUSIP-filtered quotes are also fine
if C["TRADES_PATH"] is not None:
    data_ig = pd.read_parquet(C["TRADES_PATH"])
if C["BOND_INFO_PATH"] is not None:
    bond_info_df = pd.read_parquet(C["BOND_INFO_PATH"])
if "bcq_df" not in globals():
    raise ValueError("Load bcq_df first, or set USE_DEMO=True in cell 1.")
if Q_COL["event_time"] not in bcq_df and "quote_timestamp_UTC" in bcq_df:
    bcq_df = bcq_df.assign(quote_timestamp_ET=pd.to_datetime(
        bcq_df["quote_timestamp_UTC"], utc=True, errors="coerce").dt.tz_convert("America/New_York"))
if "data_ig" not in globals():
    raise ValueError("Load the three-month data_ig trade table, or set TRADES_PATH / USE_DEMO.")
trade_input = data_ig
metadata_input = globals().get("bond_info_df")
figures = {}

# %% [markdown]
# ## Define the three-month trade universe, then audit quotes
# Invalid latest records replace earlier records without reviving an old quote.
# Nonpositive size is unknown by default; quantity_raw preserves the input.
# Finite zero and negative spreads are valid by default. Peer checks depend on
# cross-dealer disagreement, not sign. Zero exclusion is an optional sensitivity.
# Timestamp ties use QUOTE_SEQUENCE_COL, or the last input row when unavailable.
# Naive times use the configured source zone; aware times are converted to ET.
# Replay assumes availability at event time; no separate known-time input.

# %% 2. Normalize columns, time zones and bond metadata
def to_et(values, naive_tz):
    dt = pd.to_datetime(values, errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(naive_tz, ambiguous="NaT", nonexistent="NaT")
    return dt.dt.tz_convert("America/New_York")

q = bcq_df[[col for col in Q_COL.values() if col in bcq_df]].rename(
    columns={v: k for k, v in Q_COL.items()}).copy()
t = trade_input[[col for col in T_COL.values() if col in trade_input]].rename(
    columns={v: k for k, v in T_COL.items()}).copy()
for df in (q, t):
    df["cusip"] = df["cusip"].astype("string").str.strip().replace("", pd.NA)
    df["quantity"] = pd.to_numeric(df.get("quantity", np.nan), errors="coerce")
    df["spread"] = pd.to_numeric(df.get("spread", np.nan), errors="coerce")
    df["spread"] = df["spread"].replace([np.inf, -np.inf], np.nan)
q["spread"] *= C["QUOTE_SPREAD_MULTIPLIER"]
t["spread"] *= C["TRADE_SPREAD_MULTIPLIER"]
q["firm"] = q["firm"].astype("string").str.strip().replace("", pd.NA)
q["side"] = q["side"].astype("string").str.strip().str.lower()
q["event_time"] = to_et(q["event_time"], C["Q_NAIVE_TZ"])
t["time"] = to_et(t["time"], C["T_NAIVE_TZ"])
t["type"] = t.get("type", pd.Series("?", index=t.index)).astype("string")
q["input_row"] = np.arange(len(q))
q["sequence"] = (pd.to_numeric(bcq_df[C["QUOTE_SEQUENCE_COL"]], errors="coerce").fillna(q["input_row"])
                 if C["QUOTE_SEQUENCE_COL"] else q["input_row"])
q["quantity_raw"] = q["quantity"]
q["spread_valid"] = q["spread"].notna() & (~q["spread"].eq(0) if C["ZERO_SPREAD_IS_MISSING"] else True)
key = ["cusip", "firm", "side"]
valid_key = q[key + ["event_time"]].notna().all(axis=1)
valid_key &= q["side"].isin(["bid", "ask"])
audit = pd.Series({
    "input_quote_rows": len(q), "invalid_keys_or_times": int((~valid_key).sum()),
    "nonfinite_spread_rows": int(q["spread"].isna().sum()),
    "zero_spread_rows": int(q["spread"].eq(0).sum()),
    "negative_spread_rows": int(q["spread"].lt(0).sum()),
    "missing_quantity_rows": int(q["quantity"].isna().sum()),
    "nonpositive_quantity_rows": int(q["quantity"].le(0).sum()),
}, name="rows")
t = t.dropna(subset=["cusip", "time"]).sort_values("time", kind="stable")
if t.empty:
    raise ValueError("The trade universe is empty: supply your three-month data_ig table.")
universe_start = to_et(pd.Series([C["TRADE_UNIVERSE_START"]]), C["T_NAIVE_TZ"]).iloc[0] if C["TRADE_UNIVERSE_START"] else t["time"].min()
universe_end = to_et(pd.Series([C["TRADE_UNIVERSE_END"]]), C["T_NAIVE_TZ"]).iloc[0] if C["TRADE_UNIVERSE_END"] else t["time"].max()
if pd.isna(universe_start) or pd.isna(universe_end) or universe_start > universe_end:
    raise ValueError("Invalid TRADE_UNIVERSE_START / TRADE_UNIVERSE_END.")
t = t.loc[t["time"].between(universe_start, universe_end)].copy()
trade_universe_cusips = pd.Index(t["cusip"].unique(), name="cusip")
if trade_universe_cusips.empty:
    raise ValueError("No trades fall inside the selected trade-universe window.")
in_trade_universe = q["cusip"].isin(trade_universe_cusips)
audit["quotes_outside_trade_universe"] = (~in_trade_universe).sum()
audit["trade_universe_bonds"] = len(trade_universe_cusips)
audit["retained_valid_quote_rows"] = (valid_key & in_trade_universe).sum()
q = q.loc[valid_key & in_trade_universe].sort_values(["event_time", "sequence", "input_row"], kind="stable")
payload = key + ["event_time", "spread", "quantity"]
audit["exact_duplicate_rows"] = q.duplicated(payload, keep="last").sum()
conflicts = q.drop_duplicates(key + ["event_time", "spread", "quantity"]).groupby(key + ["event_time"], observed=True).size()
audit["conflicting_timestamp_groups"] = conflicts.gt(1).sum()
q = q.drop_duplicates(payload, keep="last").reset_index(drop=True)
timestamp_conflict_sample = q.loc[q.duplicated(key + ["event_time"], keep=False)].head(20)
if C["NONPOSITIVE_QUANTITY_IS_MISSING"]:
    q["quantity"] = q["quantity"].where(q["quantity"].gt(0))
audit["quantity_unknown_after_normalization"] = q["quantity"].isna().sum()
# Structural metadata only; trade fallback is for EDA grouping, not feature replay.
meta_parts = []
if metadata_input is not None:
    meta_parts.append(metadata_input.reindex(columns=list(M_COL.values())).assign(_priority=2))
if M_COL["issuer"] in bcq_df:
    quote_meta = pd.DataFrame({M_COL["cusip"]: bcq_df[Q_COL["cusip"]],
                               M_COL["issuer"]: bcq_df[M_COL["issuer"]],
                               M_COL["tenor"]: bcq_df.get(M_COL["tenor"], np.nan)})
    meta_parts.append(quote_meta.drop_duplicates().assign(_priority=1))
if "CUSIP" in trade_input and "ISSUER" in trade_input:
    fallback = trade_input.reindex(columns=list(M_COL.values())).copy()
    fallback["_time"] = to_et(trade_input[T_COL["time"]], C["T_NAIVE_TZ"])
    meta_parts.append(fallback.sort_values("_time").drop(columns="_time").assign(_priority=0))
if not meta_parts:
    raise ValueError("Supply bond_info_df with CUSIP/ISSUER, or ISSUER in quotes/trades.")
meta = pd.concat(meta_parts, ignore_index=True).rename(columns={v: k for k, v in M_COL.items()})
meta["cusip"] = meta["cusip"].astype("string").str.strip().replace("", pd.NA)
meta["issuer"] = meta["issuer"].astype("string").str.strip().replace("", pd.NA)
meta["tenor"] = pd.to_numeric(meta["tenor"], errors="coerce")
# Static metadata, then quote metadata, then latest trade fallback.
meta = (meta.dropna(subset=["cusip", "issuer"]).sort_values("_priority", kind="stable")
        .groupby("cusip", sort=False).last()[["issuer", "tenor"]])
unmapped_quote_cusips = sorted(set(q["cusip"].dropna()) - set(meta.index))
display(audit.to_frame())
print(f"Trade universe: {universe_start} to {universe_end}; {len(trade_universe_cusips):,} traded bonds.")
print("Supplied data_ig defines the three-month universe; quote START/END is independent. Replay uses event time.")
print(f"Policies: nonpositive size -> unknown={C['NONPOSITIVE_QUANTITY_IS_MISSING']}; exclude zero spread={C['ZERO_SPREAD_IS_MISSING']}.")
if len(timestamp_conflict_sample):
    print("Timestamp tie examples; use a feed sequence or investigate concurrent size slots.")
    display(timestamp_conflict_sample)
if unmapped_quote_cusips:
    print(f"{len(unmapped_quote_cusips)} quoted CUSIPs have no issuer metadata; supply bond_info_df.")
    display(pd.DataFrame({"unmapped_cusip": unmapped_quote_cusips[:30]}))

# %% [markdown]
# ## Select an issuer within the three-month traded-bond universe
# Set ISSUER in cell 1 using the table below. A bond must have a trade in data_ig
# (or its explicit universe window), but need not trade during the quote window.
# A 40-bond default cap limits memory. Set MAX_ANALYSIS_BONDS=None to use every bond.

# %% 3. Coverage universe and analysis window
quote_summary = q.groupby("cusip").agg(
    n_quotes=("spread", "size"), n_dealers=("firm", "nunique"))
trade_summary = t.groupby("cusip").agg(n_trades=("spread", "size"))
universe = trade_universe_cusips.sort_values()
bond_summary = pd.DataFrame(index=universe).join(meta).join(quote_summary).join(trade_summary)
for col in ["n_quotes", "n_dealers", "n_trades"]:
    bond_summary[col] = bond_summary[col].fillna(0).astype(int)
bond_summary["has_quotes"] = bond_summary["n_quotes"].gt(0)
issuer_summary = bond_summary.groupby("issuer").agg(
    bonds=("n_quotes", "size"), quoted_bonds=("has_quotes", "sum"),
    quotes=("n_quotes", "sum"), trades=("n_trades", "sum"))
issuer_summary = issuer_summary.loc[issuer_summary["quoted_bonds"].gt(0)].sort_values(["quoted_bonds", "quotes"], ascending=False)
if issuer_summary.empty:
    raise ValueError("No traded bonds have both quotes and issuer metadata in the selected universe.")
display(issuer_summary.head(25))
issuer = C["ISSUER"] if C["ISSUER"] is not None else issuer_summary.index[0]
issuer_bonds = bond_summary.loc[bond_summary["issuer"].eq(issuer)].copy()
if issuer_bonds.empty:
    raise ValueError(f"Issuer {issuer!r} not found. Choose a name from issuer_summary.")
analysis_cusips = issuer_bonds.sort_values("n_quotes", ascending=False).index.tolist()
if C["MAX_ANALYSIS_BONDS"] is not None:
    analysis_cusips = analysis_cusips[:C["MAX_ANALYSIS_BONDS"]]
iq = q.loc[q["cusip"].isin(analysis_cusips)].copy()
if iq.empty:
    raise ValueError(f"No valid quote messages for {issuer!r}.")
step = pd.Timedelta(C["GRID"])
max_age = pd.Timedelta(C["MAX_AGE"])
start = pd.Timestamp(C["START"]) if C["START"] else iq["event_time"].min().floor(C["GRID"])
end = pd.Timestamp(C["END"]) if C["END"] else iq["event_time"].max().ceil(C["GRID"]) + step
start = start.tz_localize("America/New_York") if start.tzinfo is None else start.tz_convert("America/New_York")
end = end.tz_localize("America/New_York") if end.tzinfo is None else end.tz_convert("America/New_York")
full_grid = pd.date_range(start.floor(C["GRID"]), end.floor(C["GRID"]), freq=C["GRID"])
grid = full_grid[(full_grid >= start) & (full_grid <= end) & (full_grid.dayofweek < 5)]
grid = grid[grid.indexer_between_time(*C["SESSION"])]
if len(grid) == 0:
    raise ValueError("No weekday session snapshots in START/END.")
iq = iq.loc[iq["event_time"].le(end)].copy()  # keeps pre-START history for warm start
it = t.loc[t["cusip"].isin(analysis_cusips) & t["time"].between(start, end)].copy()
window_q = iq.loc[iq["event_time"].between(start, end)].copy()
coverage = issuer_bonds.loc[analysis_cusips].copy()
coverage["window_quotes"] = window_q.groupby("cusip").size().reindex(coverage.index, fill_value=0)
coverage["window_trades"] = it.groupby("cusip").size().reindex(coverage.index, fill_value=0)
print(f"{issuer} | {len(analysis_cusips)}/{len(issuer_bonds)} bonds | {len(grid):,} snapshots")
print(f"Window: {start} to {end}; all axis times are ET.")

# %% [markdown]
# ## Latest messages, strictly before each snapshot
# One record per CUSIP / firm / side. An unchanged message refreshes message age
# but not value-change age. Stale and invalid latest messages cannot revive an
# older valid quote. No forward fill across expiry, overnight or weekend gaps.
# This is a latest-message view, not a reconstructed executable order book.

# %% 4. Build dealer states and peer-noise flags
state_messages = iq.copy()
previous = state_messages.groupby(key, sort=False)[["spread", "quantity"]].shift()
same_spread = state_messages["spread"].eq(previous["spread"]) | (state_messages["spread"].isna() & previous["spread"].isna())
same_qty = state_messages["quantity"].eq(previous["quantity"]) | (state_messages["quantity"].isna() & previous["quantity"].isna())
state_messages["value_changed"] = ~(same_spread & same_qty) | state_messages.groupby(key).cumcount().eq(0)
state_messages["spread_changed"] = ~same_spread | state_messages.groupby(key).cumcount().eq(0)
state_messages["last_change_time"] = state_messages["event_time"].where(state_messages["value_changed"])
state_messages["last_change_time"] = state_messages.groupby(key, sort=False)["last_change_time"].ffill()
state_messages["last_spread_change_time"] = state_messages["event_time"].where(state_messages["spread_changed"])
state_messages["last_spread_change_time"] = state_messages.groupby(key, sort=False)["last_spread_change_time"].ffill()
iq = iq.join(state_messages[["value_changed", "spread_changed", "last_change_time", "last_spread_change_time"]])
iq["gap_min"] = iq.groupby(key)["event_time"].diff().dt.total_seconds() / 60
parts = []
for _, messages in state_messages.groupby(key, sort=False):
    parts.append(pd.merge_asof(
        pd.DataFrame({"time": grid}),
        messages.sort_values("event_time", kind="stable"),
        left_on="time", right_on="event_time", direction="backward",
        allow_exact_matches=False,
    ).dropna(subset=["event_time"]))
states = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
if states.empty:
    raise ValueError("No quote records are known before the chosen snapshots.")
# merge_asof can promote booleans to object when early snapshots have no match.
states["spread_valid"] = states["spread_valid"].astype(bool)
states["age_min"] = (states["time"] - states["event_time"]).dt.total_seconds() / 60
states["value_age_min"] = (states["time"] - states["last_change_time"]).dt.total_seconds() / 60
states["spread_age_min"] = (states["time"] - states["last_spread_change_time"]).dt.total_seconds() / 60
states["clock_fresh"] = states["age_min"].between(0, max_age.total_seconds() / 60)
states["fresh"] = states["clock_fresh"] & states["spread"].notna()
states["usable"] = states["fresh"] & states["spread_valid"]
active_raw = states.loc[states["fresh"]].copy()
active = states.loc[states["usable"]].copy()
gkey = ["time", "cusip", "side"]
active["peer_center"] = active.groupby(gkey)["spread"].transform("median")
active["peer_residual"] = active["spread"] - active["peer_center"]
active["peer_mad"] = active["peer_residual"].abs().groupby([active[k] for k in gkey]).transform("median")
active["peer_n"] = active.groupby(gkey)["firm"].transform("nunique")
active["peer_assessed"] = active["peer_n"].ge(C["MIN_PEER_DEALERS"])
active["peer_z"] = active["peer_residual"] / (1.4826 * active["peer_mad"]).clip(lower=C["NOISE_FLOOR_BPS"])
active["peer_flag"] = active["peer_assessed"] & active["peer_z"].abs().gt(C["PEER_Z"])
active["age_weight"] = np.exp(-active["age_min"] / (max_age.total_seconds() / 60))
clean_active = active.loc[~active["peer_flag"]].copy()
active["spread_region"] = np.select([active["spread"].eq(0), active["spread"].lt(0)],
                                    ["zero", "negative"], default="positive")
spread_sign_diagnostics = active.groupby("spread_region").agg(
    states=("spread", "size"), bonds=("cusip", "nunique"),
    peer_assessed_fraction=("peer_assessed", "mean"), peer_flag_fraction=("peer_flag", "mean"))
display(spread_sign_diagnostics)
raw_side_stats = active_raw.groupby(gkey).agg(
    median=("spread", "median"), p25=("spread", lambda x: x.quantile(.25)),
    p75=("spread", lambda x: x.quantile(.75)), n_dealers=("firm", "nunique"))
side_stats = clean_active.groupby(gkey).agg(
    median=("spread", "median"), p25=("spread", lambda x: x.quantile(.25)),
    p75=("spread", lambda x: x.quantile(.75)), n_dealers=("firm", "nunique"),
    age_min=("age_min", "median"), flagged_fraction=("peer_flag", "mean"))
side_stats["median_before_peer_filter"] = active.groupby(gkey)["spread"].median()
side_stats["n_before_peer_filter"] = active.groupby(gkey)["firm"].nunique()
side_stats["flagged_fraction"] = active.groupby(gkey)["peer_flag"].mean()
side_stats["peer_assessed_fraction"] = active.groupby(gkey)["peer_assessed"].mean()
side_stats["effective_dealers_age"] = (
    clean_active.groupby(gkey)["age_weight"].sum() ** 2
    / clean_active["age_weight"].pow(2).groupby([clean_active[k] for k in gkey]).sum())
snapshot_funnel = pd.DataFrame([
    {"stage": label, "records": len(frame), "bonds": frame["cusip"].nunique(),
     "bond_side_snapshots": frame[gkey].drop_duplicates().shape[0]}
    for label, frame in [("cached latest states", states),
                          ("fresh clock", states.loc[states["clock_fresh"]]),
                          ("finite spread, including zero", active_raw),
                          ("spread validity policy", active), ("after peer filter", clean_active)]
]).set_index("stage")
snapshot_funnel["fraction_of_previous"] = snapshot_funnel["records"].div(snapshot_funnel["records"].shift()).replace([np.inf, -np.inf], np.nan)
display(snapshot_funnel)
print("Solid curves apply the spread policy and peer filter; dotted curves retain all finite fresh spreads.")
print("Not peer-flagged does not imply validated: inspect peer_assessed_fraction.")

# %% [markdown]
# ## Pair bid/ask within a dealer before using a midpoint
# Spread width = bid spread minus ask spread. Only fresh pairs with close
# timestamps qualify. Known quantities must match; unknown-size pairs are
# explicitly unverified and allowed by PAIR_ALLOW_UNKNOWN_SIZE.
# Crossed eligible pairs stay in the audit; they do not produce a usable midpoint.
# Missing quantity on both sides means size is unknown, not verified equal.

# %% 5. Pairing and coverage diagnostics
pair_fields = ["time", "cusip", "firm", "spread", "event_time", "quantity", "peer_flag"]
bid = active.loc[active["side"].eq("bid"), pair_fields]
ask = active.loc[active["side"].eq("ask"), pair_fields]
pairs = bid.merge(ask, on=["time", "cusip", "firm"], suffixes=("_bid", "_ask"))
pairs["time_gap_min"] = (pairs["event_time_bid"] - pairs["event_time_ask"]).abs().dt.total_seconds() / 60
pairs["both_sizes_unknown"] = pairs["quantity_bid"].isna() & pairs["quantity_ask"].isna()
pairs["unknown_size"] = pairs["quantity_bid"].isna() | pairs["quantity_ask"].isna()
pairs["known_same_size"] = pairs["quantity_bid"].gt(0) & pairs["quantity_ask"].gt(0) & pairs["quantity_bid"].eq(pairs["quantity_ask"])
pairs["invalid_size"] = (pairs["quantity_bid"].le(0) | pairs["quantity_ask"].le(0))
pairs["size_status"] = np.select(
    [pairs["invalid_size"], pairs["both_sizes_unknown"], pairs["unknown_size"], pairs["known_same_size"]],
    ["invalid_nonpositive", "unknown_both", "unknown_one", "known_same"], default="known_mismatch")
pairs["time_compatible"] = pairs["time_gap_min"].le(pd.Timedelta(C["PAIR_MAX_GAP"]).total_seconds() / 60)
pairs["eligible_known_size"] = pairs["time_compatible"] & pairs["known_same_size"]
pairs["eligible_unknown_size"] = pairs["time_compatible"] & pairs["unknown_size"] & ~pairs["invalid_size"] & C["PAIR_ALLOW_UNKNOWN_SIZE"]
pairs["eligible"] = pairs["eligible_known_size"] | pairs["eligible_unknown_size"]
pairs["width_bps"] = pairs["spread_bid"] - pairs["spread_ask"]
pairs["crossed"] = pairs["eligible"] & pairs["width_bps"].lt(0)
pairs["peer_clean"] = pairs["peer_flag_bid"].eq(False) & pairs["peer_flag_ask"].eq(False)
pairs["raw_mid"] = (pairs["spread_bid"] + pairs["spread_ask"]) / 2
pairs["mid"] = pairs["raw_mid"].where(pairs["eligible"] & ~pairs["crossed"] & pairs["peer_clean"])
pairs["mid_known_size"] = pairs["mid"].where(pairs["eligible_known_size"])
pairs["mid_unknown_size"] = pairs["mid"].where(pairs["eligible_unknown_size"])
pair_stats = pairs.loc[pairs["eligible"]].groupby(["time", "cusip"]).agg(
    paired_dealers=("firm", "nunique"), crossed_fraction=("crossed", "mean"),
    width_bps=("width_bps", "median"), mid=("mid", "median"))
for side in ["bid", "ask"]:
    raw_counts = active_raw.loc[active_raw["side"].eq(side)].groupby(["time", "cusip"])["firm"].nunique()
    coverage[f"{side}_raw_fresh_fraction"] = raw_counts.gt(0).groupby(level="cusip").sum().reindex(coverage.index, fill_value=0) / len(grid)
    counts = clean_active.loc[clean_active["side"].eq(side)].groupby(["time", "cusip"])["firm"].nunique()
    coverage[f"{side}_fresh_fraction"] = counts.gt(0).groupby(level="cusip").sum().reindex(coverage.index, fill_value=0) / len(grid)
    coverage[f"{side}_2dealer_fraction"] = counts.ge(2).groupby(level="cusip").sum().reindex(coverage.index, fill_value=0) / len(grid)
mid_counts = pairs.loc[pairs["mid"].notna()].groupby(["time", "cusip"])["firm"].nunique()
coverage["paired_mid_fraction"] = mid_counts.gt(0).groupby(level="cusip").sum().reindex(coverage.index, fill_value=0) / len(grid)
quality_summary = pd.Series({
    "raw_fresh_records": len(active_raw), "valid_spread_records": len(active),
    "clean_records": len(clean_active), "peer_flag_fraction": active["peer_flag"].mean(),
    "peer_assessed_fraction": active["peer_assessed"].mean(),
    "both_sides_present_pairs": len(pairs), "eligible_pairs": pairs["eligible"].sum(),
    "crossed_eligible_fraction": pairs.loc[pairs["eligible"], "crossed"].mean(),
    "eligible_known_size_pairs": pairs["eligible_known_size"].sum(),
    "eligible_unknown_size_pairs": pairs["eligible_unknown_size"].sum(),
    "eligible_size_unknown_fraction": pairs.loc[pairs["eligible"], "unknown_size"].mean(),
}, name="value")
pair_funnel = pd.Series({"both valid-spread sides present": len(pairs),
                         "timestamps compatible": int(pairs["time_compatible"].sum()),
                         "eligible, including unverified size": int(pairs["eligible"].sum()),
                         "eligible noncrossed": int((pairs["eligible"] & ~pairs["crossed"]).sum()),
                         "after peer filter, usable midpoint": int(pairs["mid"].notna().sum())}, name="pairs").to_frame()
display(quality_summary.to_frame())
display(pair_funnel)
display(pairs.groupby("size_status").agg(pairs=("firm", "size"), eligible=("eligible", "sum")))
display(coverage.sort_values(["n_trades", "bid_fresh_fraction"], ascending=[True, False]))
# Directly inspect anomalies; never silently blank crossed medians.
display(pairs.loc[pairs["crossed"]].head(20))

# %% [markdown]
# ## Bond explorer: liquid versus sparse, ordered by tenor
# Each tenor band compares the most traded bond with a different sparsely traded
# bond having quote coverage, ranked by three-month trade count. Grey triangles are thinned
# raw messages; solid lines/IQRs use valid, peer-filtered equal-dealer states.
# Dotted lines retain raw fresh-state medians, including zero. Orange crosses
# flag peer anomalies and any optional zero-policy exclusions. Gaps mean no usable coverage.

# %% 6. Multi-bond quote/trade explorer
pool = coverage.loc[coverage["window_quotes"].gt(0) & coverage["tenor"].notna()].copy()
if pool.empty:
    pool = coverage.loc[coverage["window_quotes"].gt(0)].copy()
pool = pool.sort_values("tenor", na_position="last")
n_rows = min(4, len(pool))
panels = []
if n_rows:
    pool["tenor_band"] = pd.qcut(pool["tenor"].rank(method="first", na_option="bottom"), n_rows, labels=False)
    for _, band in pool.groupby("tenor_band", sort=True):
        liquid = band.sort_values(["n_trades", "window_quotes"], ascending=False).index[0]
        other = band.drop(index=liquid)
        covered = other.loc[other[["bid_fresh_fraction", "ask_fresh_fraction"]].max(axis=1).ge(.1)]
        candidate = covered if len(covered) else other
        sparse = candidate.sort_values(["n_trades", "window_quotes"], ascending=[True, False]).index[0] if len(candidate) else None
        panels.extend([(liquid, "most traded"), (sparse, "sparse / quoted")])
    titles = [f"{cusip} | {role} | {coverage.loc[cusip, 'tenor']:.1f}y | U/Q={coverage.loc[cusip, 'n_trades']}/{coverage.loc[cusip, 'window_trades']}"
              if cusip else "No second quoted bond in this tenor band" for cusip, role in panels]
    fig = make_subplots(rows=n_rows, cols=2, shared_xaxes=True, subplot_titles=titles,
                        vertical_spacing=.07, horizontal_spacing=.06)
    color = {"bid": "#2467a5", "ask": "#d47c27"}
    fill = {"bid": "rgba(36,103,165,.13)", "ask": "rgba(212,124,39,.13)"}
    for panel, (cusip, _) in enumerate(panels):
        row, col = panel // 2 + 1, panel % 2 + 1
        if cusip is None:
            continue
        for side in ["bid", "ask"]:
            if len(side_stats):
                ss = side_stats.reset_index().query("cusip == @cusip and side == @side").set_index("time").reindex(full_grid)
            else:
                ss = pd.DataFrame(index=full_grid, columns=["p25", "p75", "median"])
            x = ss.index.tz_localize(None)
            fig.add_trace(go.Scatter(x=x, y=ss["p25"], mode="lines", line_width=0,
                                    showlegend=False, hoverinfo="skip", connectgaps=False), row=row, col=col)
            fig.add_trace(go.Scatter(x=x, y=ss["p75"], mode="lines", line_width=0,
                                    fill="tonexty", fillcolor=fill[side], name=f"{side} dealer IQR",
                                    legendgroup=side, showlegend=panel == 0, connectgaps=False), row=row, col=col)
            fig.add_trace(go.Scatter(x=x, y=ss["median"], mode="lines", line_color=color[side],
                                    name=f"{side} clean median", legendgroup=side,
                                    showlegend=panel == 0, connectgaps=False), row=row, col=col)
            raw_ss = raw_side_stats.reset_index().query("cusip == @cusip and side == @side").set_index("time").reindex(full_grid)
            fig.add_trace(go.Scatter(x=x, y=raw_ss["median"], mode="lines",
                                    line=dict(color=color[side], dash="dot", width=1), opacity=.35,
                                    name=f"{side} raw median", legendgroup=f"raw {side}",
                                    showlegend=panel == 0, connectgaps=False), row=row, col=col)
        raw_all = window_q.loc[window_q["cusip"].eq(cusip) & window_q["spread"].notna()].copy()
        raw_all["plot_bin"] = raw_all["event_time"].dt.floor("15min")
        raw = raw_all.drop_duplicates(["firm", "side", "plot_bin"], keep="last")
        # Keep extreme/zero examples even when a later refresh overwrites the plot bin.
        extremes = pd.concat([raw_all.loc[raw_all["spread"].le(0)],
                              raw_all.sort_values("spread").groupby("side").head(10),
                              raw_all.sort_values("spread").groupby("side").tail(10)]).drop_duplicates("input_row")
        reserve = max(1, C["MAX_RAW_MARKERS"] // 4)
        if len(extremes) > reserve:
            extremes = extremes.iloc[np.linspace(0, len(extremes) - 1, reserve).astype(int)]
        raw = raw.loc[~raw["input_row"].isin(extremes["input_row"])]
        remaining = C["MAX_RAW_MARKERS"] - len(extremes)
        if len(raw) > remaining:
            raw = raw.iloc[np.linspace(0, len(raw) - 1, remaining).astype(int)]
        raw = pd.concat([raw, extremes]).sort_values("event_time")
        fig.add_trace(go.Scattergl(
            x=raw["event_time"].dt.tz_localize(None), y=raw["spread"], mode="markers",
            marker=dict(size=4, color="rgba(100,105,115,.4)", symbol=np.where(raw["side"].eq("bid"), "triangle-up", "triangle-down")),
            customdata=raw[["firm", "side", "quantity_raw"]].to_numpy(),
            hovertemplate="%{x}<br>%{y:.2f} bps<br>%{customdata[0]} %{customdata[1]}<br>quantity=%{customdata[2]}<extra></extra>",
            name="raw dealer messages", showlegend=panel == 0), row=row, col=col)
        bad = pd.concat([active.loc[active["cusip"].eq(cusip) & active["peer_flag"]],
                         active_raw.loc[active_raw["cusip"].eq(cusip) & ~active_raw["spread_valid"]]], ignore_index=True)
        if len(bad) > C["MAX_RAW_MARKERS"]:
            bad = bad.iloc[np.linspace(0, len(bad) - 1, C["MAX_RAW_MARKERS"]).astype(int)]
        fig.add_trace(go.Scattergl(x=bad["time"].dt.tz_localize(None), y=bad["spread"], mode="markers",
                                  marker=dict(size=6, symbol="x", color="#d59615"), name="excluded anomaly (raw)",
                                  showlegend=panel == 0), row=row, col=col)
        tr = it.loc[it["cusip"].eq(cusip)]
        for trade_type, trade_color in {"B": "#238b45", "S": "#c62828", "D": "#4f70a0", "?": "#777777"}.items():
            tt = tr.loc[tr["type"].eq(trade_type)]
            size_scale = max(1e-12, tt["quantity"].clip(lower=0).quantile(.95))
            size = 5 + 12 * np.sqrt(tt["quantity"].fillna(0).clip(0, size_scale) / size_scale)
            fig.add_trace(go.Scattergl(x=tt["time"].dt.tz_localize(None), y=tt["spread"], mode="markers",
                                      marker=dict(size=size, color=trade_color, opacity=.8),
                                      customdata=tt[["quantity"]].to_numpy(),
                                      hovertemplate="%{x}<br>%{y:.2f} bps<br>quantity=%{customdata[0]}<extra></extra>",
                                      name=f"trade {trade_type}", legendgroup=f"trade {trade_type}",
                                      showlegend=panel == 0), row=row, col=col)
    fig.update_xaxes(matches="x", title_text="ET")
    fig.update_yaxes(title_text="Spread (bps)")
    fig.update_layout(title=dict(text=f"{issuer}: latest dealer quotes and trades", y=.99), height=300 * n_rows + 180,
                      template="plotly_white", margin=dict(t=160), legend=dict(orientation="h", y=1.13))
    figures["bond_explorer"] = fig
    if C["SHOW_PLOTS"]:
        fig.show()

# %% [markdown]
# ## Dealer activity, refreshes and disagreement
# Event share describes feed activity. Fresh-slot share counts each dealer once
# per bond/side/snapshot, so 20 messages are not 20 independent observations.
# A recently refreshed but numerically unchanged quote can still be stale
# economically. Peer flags are descriptive, not a calibrated error probability.

# %% 7. Dealer diagnostics
events = iq.loc[iq["event_time"].between(start, end)]
dealer_stats = events.groupby("firm").agg(
    events=("spread", "size"), median_gap_min=("gap_min", "median"),
    changed_fraction=("value_changed", "mean"), spread_changed_fraction=("spread_changed", "mean"),
    median_quantity=("quantity", "median"))
dealer_stats = dealer_stats.reindex(pd.Index(sorted(set(events["firm"]) | set(active["firm"])), name="firm"))
dealer_stats["events"] = dealer_stats["events"].fillna(0)
dealer_stats["event_share"] = dealer_stats["events"] / max(1, dealer_stats["events"].sum())
dealer_stats["fresh_slot_share"] = active.groupby("firm").size().reindex(dealer_stats.index, fill_value=0) / max(1, len(active))
dealer_stats["peer_flag_fraction"] = active.groupby("firm")["peer_flag"].mean()
dealer_stats["peer_assessed_fraction"] = active.groupby("firm")["peer_assessed"].mean()
dealer_stats["median_peer_residual_bps"] = active.groupby("firm")["peer_residual"].median()
dealer_stats["missing_quantity_fraction"] = events["quantity"].isna().groupby(events["firm"]).mean()
dealer_stats = dealer_stats.sort_values("events", ascending=False)
display(dealer_stats)
fig = make_subplots(rows=2, cols=2, subplot_titles=[
    "Update share versus fresh dealer-slot share", "Message age versus spread-change age (valid records)",
    "Dealer residual to same-side peer median", "Peer residual versus quote size (descriptive)"])
top = dealer_stats.head(15)
for column, label in [("event_share", "message share"), ("fresh_slot_share", "fresh-slot share")]:
    fig.add_trace(go.Bar(x=top.index, y=top[column], name=label), row=1, col=1)
bins = np.linspace(0, 120, 25)
for column, label in [("age_min", "message age"), ("spread_age_min", "spread-change age")]:
    count, edges = np.histogram(active[column].clip(upper=119.99).dropna(), bins=bins)
    fig.add_trace(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=count / max(1, count.sum()), name=label), row=1, col=2)
fig.add_trace(go.Bar(x=top.index, y=top["median_peer_residual_bps"], name="dealer residual"), row=2, col=1)
size_residuals = active.loc[active["quantity"].gt(0)].copy()
if len(size_residuals):
    size_residuals["log_size_bin"] = pd.cut(np.log10(size_residuals["quantity"]), bins=8)
    size_stats = size_residuals.groupby(["side", "log_size_bin"], observed=True).agg(
        quantity=("quantity", "median"), residual=("peer_residual", "median"), n=("spread", "size"))
    for side in ["bid", "ask"]:
        sub = size_stats.loc[size_stats.index.get_level_values("side") == side]
        fig.add_trace(go.Scatter(x=sub["quantity"], y=sub["residual"], mode="lines+markers",
                                customdata=sub[["n"]].to_numpy(), name=f"size residual {side}"), row=2, col=2)
fig.update_xaxes(title_text="minutes; last bin includes >=120", row=1, col=2)
fig.update_xaxes(type="log", title_text="quoted quantity (source units)", row=2, col=2)
fig.update_yaxes(title_text="bps", row=2)
fig.update_layout(title=f"{issuer}: dealer and refresh diagnostics", height=750, template="plotly_white", barmode="group")
figures["dealer_diagnostics"] = fig
if C["SHOW_PLOTS"]:
    fig.show()

# %% [markdown]
# ## Do quotes provide coverage beyond trades?
# A crossed fraction is computed only on same-dealer, time/size-compatible pairs.
# Inspect mismatched pair rows separately; side-consensus medians are never
# forced to form a non-crossed synthetic market.

# %% 8. Pair quality and sparse-bond coverage
fig = make_subplots(rows=2, cols=2, subplot_titles=[
    "Eligible dealer-pair width, including crossings", "Crossed eligible pairs over time",
    "Bonds with >=2 fresh dealers on a side", "Fresh quote coverage versus universe trade count"])
eligible = pairs.loc[pairs["eligible"]]
width_time = eligible.groupby("time")["width_bps"].median().reindex(full_grid)
cross_time = eligible.groupby("time")["crossed"].mean().reindex(full_grid)
fig.add_trace(go.Scatter(x=full_grid.tz_localize(None), y=width_time, mode="lines", name="median signed width", connectgaps=False), row=1, col=1)
fig.add_trace(go.Scatter(x=full_grid.tz_localize(None), y=cross_time, mode="lines", name="crossed fraction", connectgaps=False), row=1, col=2)
for side in ["bid", "ask"]:
    counts = clean_active.loc[clean_active["side"].eq(side)].groupby(["time", "cusip"])["firm"].nunique()
    n_bonds = counts.ge(2).groupby(level="time").sum().reindex(grid, fill_value=0).reindex(full_grid)
    fig.add_trace(go.Scatter(x=full_grid.tz_localize(None), y=n_bonds, mode="lines", name=f"covered bonds {side}", connectgaps=False), row=2, col=1)
fig.add_trace(go.Scatter(
    x=coverage["n_trades"] + 1, y=coverage[["bid_fresh_fraction", "ask_fresh_fraction"]].max(axis=1),
    mode="markers", marker=dict(size=10, color=coverage["tenor"].fillna(0), colorscale="Viridis", showscale=True,
                                colorbar=dict(title="tenor", len=.4, y=.22)),
    customdata=np.column_stack([coverage.index, coverage["window_quotes"], coverage["n_trades"], coverage["window_trades"]]),
    hovertemplate="%{customdata[0]}<br>fresh coverage=%{y:.1%}<br>quotes=%{customdata[1]}<br>universe trades=%{customdata[2]}<br>quote-window trades=%{customdata[3]}<extra></extra>",
    name="bond coverage"), row=2, col=2)
fig.update_xaxes(type="log", title_text="trade-universe count + 1", row=2, col=2)
fig.update_yaxes(title_text="coverage fraction", range=[0, 1.05], row=2, col=2)
fig.update_yaxes(title_text="bps", row=1, col=1)
fig.update_layout(title=f"{issuer}: pair quality and sparse-bond coverage", height=750, template="plotly_white")
figures["pair_quality"] = fig
if C["SHOW_PLOTS"]:
    fig.show()

# %% [markdown]
# ## Issuer co-movement: matched dealers, with size sensitivity checks
# At t and t-lag, match the same bond / firm / side. Require freshness at both
# ends and stay within the same ET session day. Remove peer flags at either end,
# then take the median dealer change per bond, without requiring size equality.
# Report known-equal-size and unknown-size subsets as separate sensitivity checks.
# Also compare with a subset excluding exact zero at either endpoint; keep negatives.
# This removes shifts caused solely by dealers entering/leaving the sample.
# Report overlap counts beside correlations; avoid interpreting level correlations.
# PCA uses complete rows for a small covered subset; no zero-fill or forward-fill.
# These full-sample correlations/PCA are descriptive, not online LGBM features.

# %% 9. Matched changes, correlation and exploratory PCA
lag = pd.Timedelta(C["CHANGE_LAG"])
fields = ["time", "cusip", "firm", "side", "spread", "quantity", "peer_flag"]
old = active[fields].copy()
old["time"] += lag
matched = active[fields].merge(old, on=["time", "cusip", "firm", "side"], suffixes=("", "_old"))
matched_both_ends = len(matched)
matched = matched.loc[matched["time"].dt.normalize().eq((matched["time"] - lag).dt.normalize())].copy()
matched["known_same_size"] = matched["quantity"].gt(0) & matched["quantity_old"].gt(0) & matched["quantity"].eq(matched["quantity_old"])
matched["unknown_size"] = matched["quantity"].isna() | matched["quantity_old"].isna()
matched["known_size_changed"] = matched["quantity"].gt(0) & matched["quantity_old"].gt(0) & ~matched["known_same_size"]
matched["change_bps"] = matched["spread"] - matched["spread_old"]
matched_clean = matched.loc[matched["peer_flag"].eq(False) & matched["peer_flag_old"].eq(False)].copy()
movement = matched.groupby(gkey).agg(change_before_peer_filter=("change_bps", "median"), n_before_peer_filter=("firm", "nunique"))
movement = movement.join(matched_clean.groupby(gkey).agg(change_bps=("change_bps", "median"), n_matched=("firm", "nunique")))
stable = matched_clean.loc[matched_clean["known_same_size"]]
unknown = matched_clean.loc[matched_clean["unknown_size"]]
nonzero = matched_clean.loc[matched_clean["spread"].ne(0) & matched_clean["spread_old"].ne(0)]
movement = movement.join(stable.groupby(gkey).agg(change_stable_size=("change_bps", "median"), n_stable_size=("firm", "nunique")))
movement = movement.join(unknown.groupby(gkey).agg(change_unknown_size=("change_bps", "median"), n_unknown_size=("firm", "nunique")))
movement = movement.join(nonzero.groupby(gkey).agg(change_excluding_zero=("change_bps", "median"), n_excluding_zero=("firm", "nunique")))
for value, count in [("change_bps", "n_matched"), ("change_before_peer_filter", "n_before_peer_filter"),
                     ("change_stable_size", "n_stable_size"), ("change_unknown_size", "n_unknown_size"),
                     ("change_excluding_zero", "n_excluding_zero")]:
    movement[value] = movement[value].where(movement[count].ge(C["MIN_MATCHED_DEALERS"]))
supported = matched_clean.loc[matched_clean.groupby(gkey)["firm"].transform("nunique").ge(C["MIN_MATCHED_DEALERS"])]
movement_funnel = pd.DataFrame([
    {"stage": label, "dealer_pairs": len(frame), "bond_side_snapshots": frame[gkey].drop_duplicates().shape[0]}
    for label, frame in [("matched, same session day", matched), ("main: after peer filter, all sizes", matched_clean),
                          ("main: minimum matched dealers", supported),
                          ("sensitivity: known stable size", stable), ("sensitivity: unknown size", unknown),
                          ("sensitivity: exclude exact zero endpoints", nonzero)]
]).set_index("stage")
print(f"Matched endpoints before same-day restriction: {matched_both_ends:,} dealer pairs.")
display(movement_funnel)
chosen = movement.reset_index().loc[lambda x: x["side"].eq(C["ANALYSIS_SIDE"])]
change_views = {}
for name, value in [("main", "change_bps"), ("before_peer_filter", "change_before_peer_filter"),
                     ("stable_size", "change_stable_size"), ("unknown_size", "change_unknown_size"),
                     ("excluding_zero", "change_excluding_zero")]:
    change_views[name] = chosen.pivot(index="time", columns="cusip", values=value).reindex(grid)
ordered = coverage.sort_values("tenor", na_position="last").index.intersection(change_views["main"].columns, sort=False)
change_views = {name: frame.reindex(columns=ordered) for name, frame in change_views.items()}
changes = change_views["main"]
changes_stable_size = change_views["stable_size"]
changes_unknown_size = change_views["unknown_size"]
changes_excluding_zero = change_views["excluding_zero"]
comovement_coverage = pd.DataFrame({"main_observations": changes.count(),
    "stable_size_observations": changes_stable_size.count(), "unknown_size_observations": changes_unknown_size.count(),
    "excluding_zero_observations": changes_excluding_zero.count(),
    "nonzero_main_changes": (changes.abs().gt(1e-10) & changes.notna()).sum(), "std_bps": changes.std()})
display(comovement_coverage)
observed = changes.notna().astype("int64")
overlap = observed.T @ observed
correlation = changes.corr(min_periods=C["MIN_COMMON_OBS"])
correlation_before_peer_filter = change_views["before_peer_filter"].corr(min_periods=C["MIN_COMMON_OBS"])
correlation_stable_size = changes_stable_size.corr(min_periods=C["MIN_COMMON_OBS"])
correlation_unknown_size = changes_unknown_size.corr(min_periods=C["MIN_COMMON_OBS"])
overlap_excluding_zero = changes_excluding_zero.notna().astype("int64").T @ changes_excluding_zero.notna().astype("int64")
correlation_excluding_zero = changes_excluding_zero.corr(min_periods=C["MIN_COMMON_OBS"])
display(overlap)
display(correlation)
print("Compare correlation_stable_size / correlation_unknown_size / correlation_before_peer_filter separately.")
print("Exact-zero sensitivity (negative spreads retained): compare correlation_excluding_zero and overlap_excluding_zero.")
display(correlation_excluding_zero)
pca_columns = changes.count().sort_values(ascending=False).loc[lambda x: x.ge(C["MIN_COMMON_OBS"])].head(6).index
complete = changes[pca_columns].dropna()
complete = complete.loc[:, complete.std(ddof=0).gt(0)]
pca_variance = pd.Series(dtype=float, name="explained_variance_fraction")
if len(complete) >= C["MIN_COMMON_OBS"] and complete.shape[1] >= 3:
    z = (complete - complete.mean()) / complete.std(ddof=0)
    _, singular, _ = np.linalg.svd(z.to_numpy(), full_matrices=False)
    pca_variance = pd.Series(singular ** 2 / (singular ** 2).sum(),
                             index=[f"PC{k+1}" for k in range(len(singular))], name="explained_variance_fraction")
    print(f"PCA: {len(complete)} complete snapshots, {complete.shape[1]} bonds: {list(complete.columns)}")
    display(pca_variance.to_frame())
else:
    print(f"PCA skipped: {len(complete)} complete rows, {complete.shape[1]} nonconstant bonds; need >=3 bonds and >={C['MIN_COMMON_OBS']} rows.")
fig = make_subplots(rows=2, cols=2, subplot_titles=[
    f"Matched-dealer {C['CHANGE_LAG']} {C['ANALYSIS_SIDE']} changes (bps)", "Pairwise change correlation",
    "Number of shared observations", "PCA on complete standardized changes (subset)"])
for cusip in changes.count().sort_values(ascending=False).head(8).index:
    line = changes[cusip].reindex(full_grid)
    fig.add_trace(go.Scatter(x=full_grid.tz_localize(None), y=line, mode="lines", name=cusip,
                            connectgaps=False, opacity=.75), row=1, col=1)
fig.add_trace(go.Heatmap(z=correlation.to_numpy(), x=correlation.columns, y=correlation.index,
                        zmin=-1, zmax=1, colorscale="RdBu", showscale=False,
                        customdata=overlap.to_numpy(), hovertemplate="%{y} / %{x}<br>corr=%{z:.2f}<br>n=%{customdata}<extra></extra>"), row=1, col=2)
fig.add_trace(go.Heatmap(z=overlap.to_numpy(), x=overlap.columns, y=overlap.index,
                        colorscale="Blues", showscale=False), row=2, col=1)
fig.add_trace(go.Bar(x=pca_variance.index, y=pca_variance, name="PCA variance"), row=2, col=2)
fig.update_yaxes(range=[0, 1], title_text="explained fraction", row=2, col=2)
fig.update_layout(title=f"{issuer}: issuer co-movement diagnostics", height=850, template="plotly_white")
figures["co_movement"] = fig
if C["SHOW_PLOTS"]:
    fig.show()

# %% [markdown]
# ## What to inspect next
# * Does a three-month-traded sparse bond have quotes when the short window has no trades?
# * Are common changes visible on both bid and ask, and after removing peer flags?
# * Do results persist at MAX_AGE=10/30/60min and CHANGE_LAG=15/30/60min?
# * Are similar moves independent dealer evidence or a single dealer's batch refresh?
# * Do unknown/mismatched sizes or asynchronously updated sides dominate?
# Inspect `snapshot_funnel`, `pair_funnel`, `movement_funnel`, `comovement_coverage`,
# plus `coverage`, `dealer_stats`, `pairs`, `active_raw`, `clean_active`, and `overlap`.
# Good co-movement is evidence to prototype an online issuer state, not proof of
# predictive value. That requires a later chronological downstream evaluation.
