"""Small population tables and deterministic cases for the research notebooks.

Raw records remain unchanged. Counts of rows, events and dealer/bond/side/day
units are kept separate; quote absence is measured against the traded universe.
"""
# SETUP LOGIC: STEP 1 — Import shared keys and declare displayed states; seed is fixed at 2026
from hashlib import sha256

import numpy as np
import pandas as pd

from quote_quality_core import KEYS, SERIES, side_features_fast

KINDS = ["Positive", "Zero", "Missing", "Other"]
QKINDS = ["Same positive", "Different positive", "Contains zero", "Missing / other"]
CASE_SEED = 2026


# SETUP LOGIC: Declare _labels; existing arguments and docstring are preserved
def _labels(values, missing):
    # CORE LOGIC: STEP 1 — Normalize a label without inventing metadata
    # Input: values=[' Acme ', '', None], missing='Unknown'
    # Output: ['Acme', 'Unknown', 'Unknown']
    return values.astype("string").str.strip().replace("", pd.NA).fillna(missing)


# SETUP LOGIC: Declare build_bond_universe; existing arguments and docstring are preserved
def build_bond_universe(trades):
    """One CUSIP per supplied three-month traded universe, including no quotes.

    SECTOR is assigned only when exactly one distinct non-null label exists.
    This never chooses an arbitrary first SECTOR from conflicting records.
    """
    # CORE LOGIC: STEP 1 — Build the traded CUSIP index from narrow metadata
    # Input: trades CUSIP=['X','X','Y','',None], ISSUER=['A','A','B','C','D'], SECTOR=['Energy','Energy','Utilities','Energy','Energy']
    # Output: u index=['X','Y']; narrow t retains the first three records; no-quote Y is retained
    # Trick: Only identifier validity filters the universe; no quote condition enters this index.
    metadata_columns = ["CUSIP"] + [c for c in ["ISSUER", "SECTOR"] if c in trades]
    t = trades.loc[trades["CUSIP"].notna(), metadata_columns].copy()
    t["CUSIP"] = t["CUSIP"].astype("string").str.strip()
    t = t.loc[t["CUSIP"].ne("")]
    u = pd.DataFrame({"cusip": sorted(t["CUSIP"].unique())}).set_index("cusip")
    for source, missing, conflict in [("SECTOR", "Unknown", "Conflicting"),
                                      ("ISSUER", "[Missing issuer]", "[Conflicting issuer]")]:
        if source in t:
            # CORE LOGIC: STEP 2 — Resolve labels using the number of distinct non-null values
            # Input: X SECTOR=['Energy','Utilities']; Y SECTOR=[None]
            # Output: X='Conflicting', Y='Unknown'; trade_rows X=2, Y=1
            # Trick: nunique ignores nulls; first is accepted only when the distinct count is at most one.
            labels = t[source].astype("string").str.strip().replace("", pd.NA)
            n = labels.groupby(t["CUSIP"]).nunique()
            first = labels.groupby(t["CUSIP"]).first()
            u[source] = first.reindex(u.index).fillna(missing).mask(n.reindex(u.index).gt(1), conflict)
        else:
            u[source] = missing
    u["trade_rows"] = t.groupby("CUSIP").size().reindex(u.index, fill_value=0)
    return u.reset_index()


# SETUP LOGIC: Declare attach_quote_metadata; existing arguments and docstring are preserved
def attach_quote_metadata(quotes, universe):
    """Annotate quote rows within the universe without filtering by quote value."""
    # CORE LOGIC: STEP 1 — Attach universe metadata while preserving quote values
    # Input: universe={X:Energy,Y:Utilities}; quotes=[(X,spread=0,q=0),(Z,spread=-5,q=None)]
    # Output: X row remains with Energy; Z lies outside the traded universe
    q = quotes.loc[quotes["cusip"].isin(universe["cusip"])].copy()
    metadata = universe.set_index("cusip")
    for col in ["ISSUER", "SECTOR"]:
        q[col] = q["cusip"].map(metadata[col])
    return q


# SETUP LOGIC: Declare quantity_rows; existing arguments and docstring are preserved
def quantity_rows(raw):
    # CORE LOGIC: STEP 1 — Keep raw quantity categories and their numeric view
    # Input: quantity=[0,2,None,-1,'bad',float('inf')]
    # Output: quantity_kind=['Zero','Positive','Missing','Other','Other','Other']
    # Trick: A true null is Missing; invalid non-null values are Other, and no row is deleted.
    q = raw.copy()
    q["ISSUER"] = _labels(q["ISSUER"], "[Missing issuer]")
    q["firm"] = _labels(q["firm"], "[Missing dealer]")
    values = pd.to_numeric(q["quantity"], errors="coerce").to_numpy(float, na_value=np.nan)
    finite = np.isfinite(values)
    q["quantity_numeric"] = values
    q["quantity_kind"] = np.select(
        [q["quantity"].isna(), finite & (values == 0), finite & (values > 0)],
        ["Missing", "Zero", "Positive"], default="Other")
    return q


# SETUP LOGIC: Declare _same_positive_multi; existing arguments and docstring are preserved
def _same_positive_multi(pairs):
    # CORE LOGIC: STEP 1 — Check ambiguity within the same positive size
    # Input: pairs=[(95,'q=2.0'),(98,'q=2.0'),(100,'Zero')]
    # Output: True; q=2.0 maps to two finite spreads
    # Trick: Different positive sizes alone do not establish same-size ambiguity.
    sizes = {}
    for spread, tag in pairs:
        if tag.startswith("q=") and isinstance(spread, (int, float, np.number)) and np.isfinite(spread):
            sizes.setdefault(tag, set()).add(float(spread))
    return any(len(v) > 1 for v in sizes.values())


# SETUP LOGIC: Declare _event_table; existing arguments and docstring are preserved
def _event_table(events, universe):
    # CORE LOGIC: STEP 1 — Map metadata and flag simultaneous candidate ambiguity
    # Input: u=(cusip=X,ISSUER=A,SECTOR=Energy); event=(cusip=X,candidate_count=2,pair_set=((95,'q=2.0'),(98,'q=2.0')),quantity_set=('q=2.0',))
    # Output: event ISSUER=A, SECTOR=Energy, multi=True, same_positive_multi=True, has_zero=False, all_zero=False
    g = events.copy()
    meta = universe.set_index("cusip")
    for col in ["ISSUER", "SECTOR"]:
        g[col] = g["cusip"].map(meta[col])
    g["multi"] = g["candidate_count"].ge(2)
    g["same_positive_multi"] = g["pair_set"].map(_same_positive_multi).astype(bool)
    g["has_zero"] = g["quantity_set"].map(lambda x: "Zero" in x).astype(bool)
    g["all_zero"] = g["quantity_set"].map(lambda x: x == ("Zero",)).astype(bool)
    # CORE LOGIC: STEP 2 — Classify quantity conditions with explicit precedence
    # Input: quantity_set=('q=2.0','Zero','Missing'), candidate_count=2, gap=3, bad=0
    # Output: has_unknown=True, n_quantity=1, qclass='Missing / other', range_bps=3
    # Trick: Unknown overrides zero, which overrides different positive sizes; zero spread is not zero quantity.
    g["has_unknown"] = g["quantity_set"].map(lambda x: any(v == "Missing" or v.startswith("Other:") for v in x)).astype(bool)
    g["n_quantity"] = g["quantity_set"].map(lambda x: sum(v.startswith("q=") for v in x))
    g["qclass"] = np.select([g["has_unknown"], g["has_zero"], g["n_quantity"].gt(1)],
                            [QKINDS[3], QKINDS[2], QKINDS[1]], default=QKINDS[0])
    # Backward compatible names used by the Step 2 figures.
    g["n_spreads"], g["range_bps"], g["bad_spreads"] = g["candidate_count"], g["gap"], g["bad"]
    return g


# SETUP LOGIC: Declare _summary_row; existing arguments and docstring are preserved
def _summary_row(q, g, u, all_events, trade_days):
    # CORE LOGIC: STEP 1 — Separate event-weighted and unit-weighted denominators
    # Input: A/X/bid/Mar2 has two events with multi=[True,False]; raw quantity_kind=['Positive','Zero','Missing']
    # Output: n=2, m=1; unit mean=0.5, max=True; raw_counts Positive=1, Zero=1, Missing=1, Other=0
    n, m = len(g), int(g["multi"].sum())
    days = g.groupby(SERIES + ["day"], observed=True)["multi"].agg(["mean", "max"])
    bond_days = g.groupby(["cusip", "day"], observed=True)["multi"].max()
    raw_counts = q["quantity_kind"].value_counts().reindex(KINDS, fill_value=0)
    # SETUP LOGIC: Declare equal_rate; existing arguments and docstring are preserved
    def equal_rate(col):
        # CORE LOGIC: STEP 1 — Average entity rates with equal entity weight
        # Input: issuer A: 10 multi events; issuer B: 1 single-spread event
        # Output: equal_rate('ISSUER')=0.5; event-weighted rate would be 10/11
        # Trick: The mean of per-issuer means prevents a busy issuer from receiving ten times the weight.
        return float(g.groupby(col, observed=True)["multi"].mean().mean()) if n else np.nan
    # CORE LOGIC: STEP 2 — Measure observed bond-day coverage only in the quote-file window
    # Input: quoted X on Mar 2; traded X and Y on Mar 2 plus X on Jan 1; window is March
    # Output: quoted_bonds=1; target_days=2; covered_trade_days=1; Jan 1 remains outside-window
    quoted_bonds = q["cusip"].nunique()
    observed = set(zip(q["cusip"], q["quote_timestamp_ET"].dt.normalize()))
    universe_days = trade_days.loc[trade_days["cusip"].isin(u["cusip"])]
    target_days = universe_days.loc[universe_days["in_quote_window"]]
    covered_trade_days = sum(key in observed for key in target_days[["cusip", "day"]].itertuples(index=False, name=None))
    # CORE LOGIC: STEP 3 — Record the universe and coverage denominators
    # Input: u=[X,Y]; target_days=[(X,Mar2),(Y,Mar2)], covered_trade_days=1, Jan1 outside-window
    # Output: traded_bonds=2, quoted_bonds=1, no_quote_bonds=1, bond_coverage=0.5; traded_bond_days=2; three_month days=3
    r = dict(traded_bonds=len(u), quoted_bonds=quoted_bonds, no_quote_bonds=len(u) - quoted_bonds,
             bond_coverage=quoted_bonds / len(u) if len(u) else np.nan,
             traded_bond_days=len(target_days), quote_covered_traded_bond_days=covered_trade_days,
             three_month_traded_bond_days=len(universe_days),
             outside_file_window_trade_days=int(((~universe_days["in_quote_window"]) & universe_days["quote_window_known"]).sum()),
             unknown_quote_window_trade_days=int((~universe_days["quote_window_known"]).sum()),
             raw_rows=len(q), unkeyed_rows=len(q) - int(g["rows"].sum()), events=n, multi_events=m,
             observed_issuers=g["ISSUER"].nunique(), observed_dealers=g["firm"].nunique(),
             # CORE LOGIC: STEP 4 — Record event and dealer-unit counts independently
             # Input: 2 events, multi=[True,False], both in one dealer/bond/side/day; 3 keyed raw rows
             # Output: raw_rows=3, events=2, multi_events=1, multi_event_rate=0.5, unit_equal_multi_rate=0.5
             traded_issuers=u["ISSUER"].nunique(),
             multi_event_rate=m / n if n else np.nan,
             dealer_bond_side_days=len(days), affected_dealer_bond_side_days=int(days["max"].sum()),
             unit_equal_multi_rate=float(days["mean"].mean()),
             affected_unit_rate=float(days["max"].mean()),
             observed_bond_days=len(bond_days), affected_bond_days=int(bond_days.sum()),
             issuer_equal_multi_rate=equal_rate("ISSUER"), dealer_equal_multi_rate=equal_rate("firm"),
             same_positive_multi_events=int(g["same_positive_multi"].sum()),
             # CORE LOGIC: STEP 5 — Retain quantity ambiguity, missing conditions and history evidence
             # Input: one of 2 events has same-size ambiguity; one contains unknown quantity; one is incomplete; 1 repeated raw row
             # Output: same_positive_multi_events=1, unknown_quantity_events=1, incomplete_events=1, repeats=1
             # Trick: History counts describe recorded transitions; they do not certify a cleaning rule or forecast gain.
             unknown_quantity_events=int(g["has_unknown"].sum()), zero_quantity_events=int(g["has_zero"].sum()),
             keyed_quote_bonds=g["cusip"].nunique(), no_keyed_event_bonds=len(u) - g["cusip"].nunique(),
             incomplete_events=int((~g["complete"]).sum()), repeats=int(q["_source_repeat"].sum()),
             continuous_event_transitions=int((~g["history_break"]).sum()),
             unchanged_pair_refresh_events=int(g["pair_refresh"].sum()),
             changed_spread_events=int(g["spread_changed"].sum()),
             changed_condition_events=int(g["condition_changed"].sum()),
             share_of_all_events=n / max(all_events, 1))
    for state in QKINDS:
        # CORE LOGIC: STEP 6 — Add event composition and raw-row shares to the summary
        # Input: qclass=['Same positive','Missing / other']; raw quantities=[2,0,None]
        # Output: quantity_event_same_positive=1; raw_positive_rows=1; raw_positive_share=1/3
        r["quantity_event_" + state.lower().replace(" / ", "_").replace(" ", "_")] = int(g["qclass"].eq(state).sum())
    for k in KINDS:
        r[f"raw_{k.lower()}_rows"] = int(raw_counts[k])
        r[f"raw_{k.lower()}_share"] = raw_counts[k] / len(q) if len(q) else np.nan
    return r


# SETUP LOGIC: Declare population_tables; existing arguments and docstring are preserved
def population_tables(trades, quotes, event_cache=None):
    """Build once; dashboards slice these tables without recomputing histories.

    Dealer coverage denominators are every traded bond, not dealer's observed
    bonds. Equal issuer/dealer rates include quote-observed groups only, whose
    counts are explicit; no-quote bonds remain in universe coverage denominators.
    """
    # CORE LOGIC: STEP 1 — Build the full-path universe and retained raw quantity rows
    # Input: trades=[(CUSIP=X,ISSUER=A,SECTOR=Energy),(CUSIP=Y,ISSUER=B,SECTOR=Utilities)]; quotes=[(firm=D,cusip=X,side=bid,ET=Mar2 10:00,spread=95,quantity=0)]
    # Output: u=[X,Y]; raw and q each contain the D/X row; q quantity_kind='Zero', _source_repeat=False
    u = build_bond_universe(trades)
    raw = attach_quote_metadata(quotes, u)
    q = quantity_rows(raw)
    q["_source_repeat"] = raw.duplicated().to_numpy()
    # CACHEING LOGIC: STEP 2 — Reuse supplied prepared events or construct the shared event cache once
    if event_cache is None:
        from quote_quality_cache import prepare_quote_events
        event_cache = prepare_quote_events(raw)
    # CORE LOGIC: STEP 2 — Annotate events and normalize trade dates to ET
    # Input: one cached X event; naive trade time='2026-03-02 10:00'
    # Output: event receives issuer/sector; trade day=2026-03-02 00:00 America/New_York
    g = _event_table(event_cache["events"], u)
    timestamps = pd.to_datetime(trades["EFFECTIVE_DATETIME_TS"], errors="coerce")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize("America/New_York")
    else:
        timestamps = timestamps.dt.tz_convert("America/New_York")
    td = pd.DataFrame({"cusip": trades["CUSIP"], "day": timestamps.dt.normalize()}).dropna().drop_duplicates()
    quote_dates = quotes["quote_timestamp_ET"].dropna().dt.normalize()
    td["in_quote_window"] = (td["day"].between(quote_dates.min(), quote_dates.max())
                             # CORE LOGIC: STEP 3 — Mark quote-window coverage and create the Global summary
                             # Input: trade days Jan1 and Mar2; quote dates Mar1-Mar31
                             # Output: in_quote_window=[False,True]; Global keeps both days in three-month counts
                             # Trick: Days outside the file are reported separately, not treated as evidence of absent March quotes.
                             if len(quote_dates) else False)
    td["quote_window_known"] = bool(len(quote_dates))
    tables = {}
    tables["Global"] = pd.DataFrame([_summary_row(q, g, u, len(g), td)], index=["Global"])
    # Group indices avoid rescanning the full quote/event table for every group.
    for scope, col in [("SECTOR", "SECTOR"), ("Issuer", "ISSUER"), ("Dealer", "firm")]:
        qi, gi = q.groupby(col, observed=True).indices, g.groupby(col, observed=True).indices
        ui = u.groupby(col, observed=True).indices if col != "firm" else {}
        # CORE LOGIC: STEP 4 — Reuse group indices to select complete scope populations
        # Input: issuer A has X quotes/events; issuer B has traded Y but no quotes
        # Output: names include A and B; B gets empty quote/event slices and a nonempty traded-universe slice
        # Trick: Group indices select rows once; a dealer retains the full traded-universe denominator.
        names = sorted(set(qi) | set(ui))
        rows = []
        for name in names:
            qq = q.iloc[qi.get(name, [])]
            gg = g.iloc[gi.get(name, [])]
            uu = u if col == "firm" else u.iloc[ui.get(name, [])]
            rows.append(_summary_row(qq, gg, uu, len(g), td))
        if scope == "Dealer" and not names:
            names = ["[No observed dealer]"]
            # CORE LOGIC: STEP 5 — Keep the no-dealer placeholder and return reusable populations
            # Input: u=[(X,A,Energy),(Y,B,Utilities)]; q and g have zero rows; supplied event_cache={'events':empty,'unkeyed':0}
            # Output: tables['Dealer'] row '[No observed dealer]' has traded_bonds=2, no_quote_bonds=2, events=0; returned event_cache is the supplied object
            rows = [_summary_row(q.iloc[:0], g.iloc[:0], u, len(g), td)]
        tables[scope] = pd.DataFrame(rows, index=pd.Index(names, name=col))
    return {"universe": u, "raw": raw, "quantity": q, "events": g, "tables": tables,
            "event_cache": event_cache, "unkeyed": event_cache["unkeyed"], "trade_days": td,
            "case_manifest": None, "case_candidates": None, "impact_table": None}


# SETUP LOGIC: Declare quantity_population_tables; existing arguments and docstring are preserved
def quantity_population_tables(trades, quotes, progress=None):
    """Step 1 descriptive summaries without histories, event caches or set objects.

    All group reductions use numeric columns and pandas built-ins. The only
    event/quantity secondary groupby checks whether the same positive raw size
    has two distinct finite spreads. Refresh/change history is not evaluated.
    """
    # SETUP LOGIC: Declare report; existing arguments and docstring are preserved
    def report(done, detail):
        # UI LOGIC: STEP 1 — Forward bounded phase progress without calculating histories
        if progress is not None:
            progress("quantity_population", done, 6, detail)

    # CORE LOGIC: STEP 1 — Build narrow quantity data while retaining raw source repeats
    # Input: traded X,Y; two identical X q=0 rows; Y has no quote
    # Output: u has 2 bonds; q has 2 rows; _source_repeat=[False,True]; raw_zero_rows=[True,True]
    # Trick: Repeat identity uses every loaded raw field; working summaries copy only needed columns.
    report(0, f"Reading narrow metadata for {len(trades):,} traded rows")
    u = build_bond_universe(trades)
    raw = attach_quote_metadata(quotes, u)
    metadata = u.set_index("cusip")
    # The raw source stays available once; quantity and aggregation work are narrow.
    q = quantity_rows(raw[KEYS + ["quantity", "ISSUER", "SECTOR"]])
    q["_source_repeat"] = raw.duplicated().to_numpy()
    for kind in KINDS:
        q[f"raw_{kind.lower()}_rows"] = q["quantity_kind"].eq(kind)
    # CORE LOGIC: STEP 2 — Validate exact event keys without filtering spread or quantity states
    # Input: rows have firm=['A','',None], cusip='X', side='bid', ET timestamps all present
    # Output: valid=[True,False,False]; quantity rows remain in raw summaries
    q["day"] = q[KEYS[-1]].dt.normalize()
    report(1, f"Classified quantity for {len(q):,} raw quote rows; grouping exact timestamps")
    valid = raw[KEYS].notna().all(axis=1)
    for key in SERIES:
        valid &= raw[key].astype("string").str.strip().ne("").fillna(False)
    z = raw.loc[valid, KEYS].copy()
    # CORE LOGIC: STEP 3 — Create finite-spread and quantity flags for keyed rows
    # Input: spread=[95,float('inf')], quantity=[2,None] at the same exact event
    # Output: s=[95,NaN], positive_q=[2.0,NaN], unknown=[False,True], bad=[False,True]; one event_id
    # Trick: Positive size keys use the numeric float view, matching the existing qtag semantics.
    z["s"] = pd.to_numeric(raw.loc[valid, "spread"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    z["positive_q"] = q.loc[valid, "quantity_numeric"].where(q.loc[valid, "quantity_kind"].eq("Positive"))
    z["zero"] = q.loc[valid, "quantity_kind"].eq("Zero")
    z["unknown"] = q.loc[valid, "quantity_kind"].isin(["Missing", "Other"])
    z["bad"] = z["s"].isna()
    z["repeat"] = q.loc[valid, "_source_repeat"]
    z["event_id"] = z.groupby(KEYS, observed=True, sort=False).ngroup()
    # CORE LOGIC: STEP 4 — Aggregate each exact timestamp with built-in reductions
    # Input: one event has (spread,q)=[(95,2),(98,2),(98,2)] with repeat flags [False,False,True]
    # Output: rows=3, repeats=1, n_spreads=2, lo=95, hi=98, n_quantity=1, has_zero=False
    stats = z.groupby("event_id", observed=True, sort=False).agg(
        rows=("s", "size"), repeats=("repeat", "sum"), bad=("bad", "sum"),
        n_spreads=("s", "nunique"), lo=("s", "min"), hi=("s", "max"),
        n_quantity=("positive_q", "nunique"), has_zero=("zero", "max"),
        all_zero=("zero", "min"), has_unknown=("unknown", "max"))
    # CORE LOGIC: STEP 5 — Detect same-positive-size multi-price events precisely
    # Input: event 0: (95,2),(98,2); event 1: (95,2),(98,5)
    # Output: same_positive_multi={0:True,1:False}; one row per event after the key/stats join
    # Trick: Reduce by (event_id, positive_q), then broadcast any matching size ambiguity back onto event IDs.
    same = z.loc[z["positive_q"].notna() & z["s"].notna()].groupby(
        ["event_id", "positive_q"], observed=True, sort=False)["s"].nunique().gt(1)
    same = same.groupby(level="event_id", sort=False).any()
    stats["same_positive_multi"] = same.reindex(stats.index, fill_value=False)
    g = z[["event_id"] + KEYS].drop_duplicates("event_id").set_index("event_id").join(stats).reset_index(drop=True)
    # CORE LOGIC: STEP 6 — Derive complete-event and candidate-range flags
    # Input: event n_spreads=2, bad=0, lo=95, hi=98, ET timestamp Mar2 10:00
    # Output: candidate_count=2, multi=True, complete=True, incomplete=False, gap=range_bps=3; day=Mar2 midnight
    g["candidate_count"] = g["n_spreads"]
    g["multi"] = g["n_spreads"].ge(2)
    g["complete"] = g["bad"].eq(0) & g["n_spreads"].gt(0)
    g["incomplete"] = ~g["complete"]
    g["bad_spreads"], g["gap"] = g["bad"], g["hi"] - g["lo"]
    g["range_bps"] = g["gap"]
    g["day"] = g[KEYS[-1]].dt.normalize()
    for col in ["ISSUER", "SECTOR"]:
        g[col] = g["cusip"].map(metadata[col])
    # CORE LOGIC: STEP 7 — Classify event conditions without building change history
    # Input: event has_unknown=True, has_zero=True, n_quantity=2
    # Output: qclass='Missing / other'; quantity_event_missing_other=True; other one-hot states False
    # Trick: Missing conditions take priority; these flags preserve candidates rather than remove them.
    g["qclass"] = np.select([g["has_unknown"].to_numpy(bool), g["has_zero"].to_numpy(bool), g["n_quantity"].gt(1)],
                            [QKINDS[3], QKINDS[2], QKINDS[1]], default=QKINDS[0])
    for state in QKINDS:
        g["quantity_event_" + state.lower().replace(" / ", "_").replace(" ", "_")] = g["qclass"].eq(state)
    report(2, f"Reduced {len(z):,} keyed rows to {len(g):,} events; no change history or cache job")
    # CORE LOGIC: STEP 8 — Build equal-weight dealer/bond/side/day units and traded-day metadata
    # Input: one unit has 3 events with multi=[True,False,True]; two X trades share Mar2
    # Output: unit events=3, multi=2, affected=True, rate=2/3; traded X/Mar2 deduplicates to one row
    units = g.groupby(SERIES + ["day"], observed=True, sort=False).agg(
        events=("multi", "size"), multi=("multi", "sum"), affected=("multi", "max"),
        ISSUER=("ISSUER", "first"), SECTOR=("SECTOR", "first")).reset_index()
    units["rate"] = units["multi"].div(units["events"])
    timestamps = pd.to_datetime(trades["EFFECTIVE_DATETIME_TS"], errors="coerce")
    timestamps = (timestamps.dt.tz_localize("America/New_York") if timestamps.dt.tz is None
                  else timestamps.dt.tz_convert("America/New_York"))
    td = pd.DataFrame({"cusip": trades["CUSIP"], "day": timestamps.dt.normalize()}).dropna().drop_duplicates()
    td = td.loc[td["cusip"].isin(u["cusip"])]
    # CORE LOGIC: STEP 9 — Restrict coverage evidence to the quote-file date interval
    # Input: quote dates Mar1-Mar31; traded dates Jan1 and Mar2
    # Output: in_quote_window=[False,True], outside=[True,False], unknown=[False,False]
    # Trick: The bond universe stays three months; only coverage evidence uses file-window trade days.
    for col in ["ISSUER", "SECTOR"]:
        td[col] = td["cusip"].map(metadata[col])
    quote_dates = quotes[KEYS[-1]].dropna().dt.normalize()
    td["in_quote_window"] = td["day"].between(quote_dates.min(), quote_dates.max()) if len(quote_dates) else False
    td["quote_window_known"] = bool(len(quote_dates))
    td["outside_file_window_trade_days"] = ~td["in_quote_window"] & td["quote_window_known"]
    td["unknown_quote_window_trade_days"] = ~td["quote_window_known"]
    # One quote/trade-day merge. Dealer denominators are scalar universe counts,
    # so no dealer x traded-day cartesian product is constructed.
    # CORE LOGIC: STEP 10 — Join quote/traded days once and deduplicate global bond-days
    # Input: A and B both quote X/Mar2; trades contain one X/Mar2 and one Y/Mar2
    # Output: covered has 2 dealer rows; covered_bonds has 1 X/Mar2 row
    # Trick: many_to_one validates trade-day uniqueness and avoids a dealer-by-traded-day Cartesian product.
    quote_days = q[["firm", "cusip", "day"]].dropna(subset=["cusip", "day"]).drop_duplicates()
    covered = quote_days.merge(td.loc[td["in_quote_window"]], on=["cusip", "day"], how="inner", validate="many_to_one")
    covered_bonds = covered.drop_duplicates(["cusip", "day"])
    report(3, f"Joined quote/trade coverage once for {len(td):,} traded bond-days")

    # SETUP LOGIC: Declare grouped; existing arguments and docstring are preserved
    def grouped(frame, col):
        # CORE LOGIC: STEP 1 — Broadcast a Global key on the existing row index
        # Input: frame indices=[7,9], col=None
        # Output: grouped(frame,None) contains one Global group with row indices [7,9]
        # Trick: An aligned Series broadcasts the Global label without adding columns or changing row order.
        key = frame[col] if col else pd.Series("Global", index=frame.index, dtype="string")
        return frame.groupby(key, observed=True, sort=False)

    # UI LOGIC: STEP 11 — Report the vectorized-summary phase; history fields remain explicitly unevaluated
    report(4, "Building vectorized Global / SECTOR / issuer / dealer summaries")
    # CORE LOGIC: STEP 11 — Include all universe groups in the summary index
    # Input: u SECTOR=['Energy','Utilities']; q contains only Energy quotes
    # Output: SECTOR index includes Energy and Utilities; absent raw counts will be filled with 0
    # Trick: No-quote sectors and issuers enter through universe labels, not through observed quote labels.
    history_fields = ["continuous_event_transitions", "unchanged_pair_refresh_events",
                      "changed_spread_events", "changed_condition_events"]
    tables = {}
    for scope, col in [("Global", None), ("SECTOR", "SECTOR"), ("Issuer", "ISSUER"), ("Dealer", "firm")]:
        names = (["Global"] if col is None else
                 sorted(q[col].unique()) if col == "firm" else sorted(u[col].unique()))
        if scope == "Dealer" and not names:
            names = ["[No observed dealer]"]
        index = pd.Index(names, name=col)
        # CORE LOGIC: STEP 12 — Reduce raw-row categories per scope in one aggregation
        # Input: Energy has quantities [2,0,None]; Utilities has no quotes
        # Output: Energy raw_rows=3, positive=1, zero=1, missing=1; Utilities raw_rows=0
        raw_aggs = dict(raw_rows=("quantity_kind", "size"), quoted_bonds=("cusip", "nunique"),
                        repeats=("_source_repeat", "sum"))
        raw_aggs.update({f"raw_{k.lower()}_rows": (f"raw_{k.lower()}_rows", "sum") for k in KINDS})
        summary = grouped(q, col).agg(**raw_aggs).reindex(index, fill_value=0)
        # CORE LOGIC: STEP 13 — Declare event reductions without row-level Python histories
        # Input: two events: multi=[True,False], rows=[2,1], same_positive_multi=[True,False]
        # Output: event reducers count events=2, multi_events=1, keyed_rows=3, same_positive_multi_events=1
        event_aggs = dict(events=("multi", "size"), multi_events=("multi", "sum"),
            keyed_rows=("rows", "sum"), observed_issuers=("ISSUER", "nunique"),
            observed_dealers=("firm", "nunique"), keyed_quote_bonds=("cusip", "nunique"),
            same_positive_multi_events=("same_positive_multi", "sum"),
            unknown_quantity_events=("has_unknown", "sum"), zero_quantity_events=("has_zero", "sum"),
            incomplete_events=("incomplete", "sum"))
        event_aggs.update({"quantity_event_" + k.lower().replace(" / ", "_").replace(" ", "_"):
                          ("quantity_event_" + k.lower().replace(" / ", "_").replace(" ", "_"), "sum") for k in QKINDS})
        # CORE LOGIC: STEP 14 — Join event and daily-unit reductions
        # Input: one scope has two units, each rate=[1.0,0.0], affected=[True,False]
        # Output: dealer_bond_side_days=2, affected_dealer_bond_side_days=1, unit_equal_multi_rate=0.5
        summary = summary.join(grouped(g, col).agg(**event_aggs).reindex(index, fill_value=0))
        daily = grouped(units, col).agg(dealer_bond_side_days=("rate", "size"),
            affected_dealer_bond_side_days=("affected", "sum"), unit_equal_multi_rate=("rate", "mean"),
            affected_unit_rate=("affected", "mean")).reindex(index)
        daily[["dealer_bond_side_days", "affected_dealer_bond_side_days"]] = daily[["dealer_bond_side_days", "affected_dealer_bond_side_days"]].fillna(0)
        summary = summary.join(daily)
        # CORE LOGIC: STEP 15 — Collapse affected dealer units to affected bond-days
        # Input: X/Mar2 has two dealers with multi flags True,False; Y/Mar2 has False
        # Output: observed_bond_days=2, affected_bond_days=1
        # Trick: max means any simultaneous ambiguity affects that bond-day, regardless of dealer count.
        bdkeys = ([col] if col else []) + ["cusip", "day"]
        bond_days = g.groupby(bdkeys, observed=True, sort=False)["multi"].max().reset_index()
        summary = summary.join(grouped(bond_days, col).agg(observed_bond_days=("multi", "size"),
            affected_bond_days=("multi", "sum")).reindex(index, fill_value=0))
        # CORE LOGIC: STEP 16 — Average issuer/dealer means with equal entity weight
        # Input: in Energy, issuer A has 10/10 multi events and issuer B has 0/1
        # Output: issuer_equal_multi_rate=0.5; event rate=10/11
        # Trick: A first groupby computes each entity mean; the second mean gives every observed entity equal weight.
        for equal_col, output in [("ISSUER", "issuer_equal_multi_rate"), ("firm", "dealer_equal_multi_rate")]:
            if col is None:
                summary[output] = g.groupby(equal_col, observed=True)["multi"].mean().mean()
            elif col == equal_col:
                summary[output] = grouped(g, col)["multi"].mean()
            else:
                equal = g.groupby([col, equal_col], observed=True, sort=False)["multi"].mean()
                summary[output] = equal.groupby(level=0, observed=True, sort=False).mean()
        # CORE LOGIC: STEP 17 — Use scalar universe denominators for every observed dealer
        # Input: u contains X,Y with two March traded-days and one Jan outside-window day; dealer A quotes X only
        # Output: A traded_bonds=2, traded_bond_days=2, three_month_traded_bond_days=3, outside_file_window_trade_days=1
        if col == "firm":
            summary["traded_bonds"] = len(u)
            summary["traded_issuers"] = u["ISSUER"].nunique()
            summary["three_month_traded_bond_days"] = len(td)
            summary["traded_bond_days"] = int(td["in_quote_window"].sum())
            summary["outside_file_window_trade_days"] = int(td["outside_file_window_trade_days"].sum())
            summary["unknown_quote_window_trade_days"] = int(td["unknown_quote_window_trade_days"].sum())
            coverage = covered.groupby("firm", observed=True, sort=False).size()
        # CORE LOGIC: STEP 18 — Attach scope-specific universe and deduplicated traded-day counts
        # Input: Energy has X and Utilities has Y; only X/Mar2 is quote-covered
        # Output: Energy traded_bonds=1, Utilities traded_bonds=1; coverage counts are Energy=1, Utilities=0
        else:
            universe_stats = grouped(u, col).agg(traded_bonds=("cusip", "size"), traded_issuers=("ISSUER", "nunique"))
            summary = summary.join(universe_stats.reindex(index, fill_value=0))
            trade_stats = grouped(td, col).agg(three_month_traded_bond_days=("cusip", "size"),
                traded_bond_days=("in_quote_window", "sum"), outside_file_window_trade_days=("outside_file_window_trade_days", "sum"),
                unknown_quote_window_trade_days=("unknown_quote_window_trade_days", "sum"))
            summary = summary.join(trade_stats.reindex(index, fill_value=0))
            coverage = grouped(covered_bonds, col).size()
        # CORE LOGIC: STEP 19 — Compute absence and rate fields with explicit zero denominators
        # Input: scope traded_bonds=2, quoted_bonds=1, keyed_quote_bonds=1, raw_rows=3, keyed_rows=2, events=1, multi_events=1
        # Output: no_quote_bonds=1, no_keyed_event_bonds=1, unkeyed_rows=1, bond_coverage=0.5, multi_event_rate=1.0
        # Trick: Empty denominators become NaN; an unevaluated rate is not a measured zero.
        summary["quote_covered_traded_bond_days"] = coverage.reindex(index, fill_value=0)
        summary["no_quote_bonds"] = summary["traded_bonds"] - summary["quoted_bonds"]
        summary["no_keyed_event_bonds"] = summary["traded_bonds"] - summary["keyed_quote_bonds"]
        summary["unkeyed_rows"] = summary["raw_rows"] - summary.pop("keyed_rows")
        summary["bond_coverage"] = summary["quoted_bonds"].div(summary["traded_bonds"].replace(0, np.nan))
        summary["multi_event_rate"] = summary["multi_events"].div(summary["events"].replace(0, np.nan))
        summary["share_of_all_events"] = summary["events"] / max(len(g), 1)
        # CORE LOGIC: STEP 20 — Return quantity-only summaries with history unavailable
        # Input: 3 raw rows, one zero quantity row; no history was computed
        # Output: raw_zero_share=1/3; refresh/change fields=NaN; event_cache=None; population_kind='quantity_only'
        # Trick: NaN records unassessed histories and summary_html displays that status explicitly.
        for kind in KINDS:
            summary[f"raw_{kind.lower()}_share"] = summary[f"raw_{kind.lower()}_rows"].div(summary["raw_rows"].replace(0, np.nan))
        summary[history_fields] = np.nan
        tables[scope] = summary
    report(6, f"Ready: {len(q):,} raw rows / {len(g):,} exact events; history not evaluated")
    return dict(universe=u, raw=raw, quantity=q, events=g, tables=tables, event_cache=None,
                unkeyed=int((~valid).sum()), trade_days=td, case_manifest=None,
                case_candidates=None, impact_table=None, population_kind="quantity_only")


# SETUP LOGIC: Declare scope_selection; existing arguments and docstring are preserved
def scope_selection(population, scope="Global", value=None):
    # CORE LOGIC: STEP 1 — Select a scope without weakening dealer coverage denominators
    # Input: u=[(X,A,Energy),(Y,B,Utilities)]; q/g each contain only (firm=D,cusip=X); scope='Dealer', value='D'
    # Output: selected raw/events each contain D/X; selected universe remains both X and Y
    q, g, u = population["quantity"], population["events"], population["universe"]
    col = {"SECTOR": "SECTOR", "Issuer": "ISSUER", "Dealer": "firm"}.get(scope)
    if col is not None:
        q, g = q.loc[q[col].eq(value)], g.loc[g[col].eq(value)]
        if col != "firm":
            u = u.loc[u[col].eq(value)]
    # CORE LOGIC: STEP 2 — Return the selected summary and navigation label
    # Input: scope='SECTOR', value='Energy'; selected q/g contain X; u contains X; tables['SECTOR'].loc['Energy'] has traded_bonds=1, quoted_bonds=1, events=2
    # Output: result issuer='SECTOR: Energy', scope='SECTOR', value='Energy'; summary includes traded_bonds=1, quoted_bonds=1, events=2
    label = "Global" if scope == "Global" else f"{scope}: {value}"
    summary = population["tables"][scope].loc["Global" if scope == "Global" else value]
    return {"raw": q, "groups": g, "universe": u, "summary": summary,
            "issuer": label, "scope": scope, "value": value}


# SETUP LOGIC: Declare case_candidates; existing arguments and docstring are preserved
def case_candidates(events):
    """Dealer/bond/side/day units; quote-absent universe stays in summary only."""
    # CORE LOGIC: STEP 1 — Summarize observed dealer/bond/side/day navigation units
    # Input: A/X/bid/Mar2 has multi=[True,False], has_unknown=[False,True], pair_refresh=[False,True], same_positive_multi=[True,False]; no ask events
    # Output: one A/X/bid/Mar2 row: events=2, multi=1, multi_rate=0.5, unknown_rate=0.5, refresh_rate=0.5, one_sided_day=True
    columns = SERIES + ["day"]
    c = events.groupby(columns, observed=True, sort=True).agg(
        ISSUER=("ISSUER", "first"), SECTOR=("SECTOR", "first"), events=("multi", "size"),
        multi_rate=("multi", "mean"), unknown_rate=("has_unknown", "mean"),
        refresh_rate=("pair_refresh", "mean"), multi=("multi", "sum"),
        same_positive_multi=("same_positive_multi", "sum")).reset_index()
    both = events.groupby(["firm", "cusip", "day"], observed=True)["side"].nunique()
    c["one_sided_day"] = [both.loc[(r.firm, r.cusip, r.day)] == 1 for r in c.itertuples()]
    # CORE LOGIC: STEP 2 — Assign activity strata and row-order-independent identities
    # Input: A/X/bid/2026-03-02 00:00:00-05:00 has 2 events; seed=2026
    # Output: activity='Sparse (1-3)'; case_id='A|X|bid|2026-03-02 00:00:00-05:00'; stable_hash=a4e8c5c3cc97d4d0d9c9e5d3ce426fdf04eb73b81602fc02afa8e7598bf9cab7
    # Trick: Hash ordering is stable under source-row permutations; it is navigation sampling, not a population estimator.
    c["activity"] = pd.cut(c["events"], [0, 3, 30, np.inf], labels=["Sparse (1-3)", "Medium (4-30)", "Active (>30)"])
    c["case_id"] = ["|".join([str(v) for v in row]) for row in c[columns].itertuples(index=False, name=None)]
    c["stable_hash"] = c["case_id"].map(lambda v: sha256(f"{CASE_SEED}|{v}".encode()).hexdigest())
    return c


# SETUP LOGIC: Declare measure_rule_impacts; existing arguments and docstring are preserved
def measure_rule_impacts(events):
    """Actual rule changes at three fixed local queries per bond/side/ET day.

    First, middle and last observed event timestamps are selected once. This is
    an exhaustive observed bond/side/day panel with a sparse query denominator,
    not a trade-weighted estimate or a claim about prediction improvement.
    """
    # CORE LOGIC: STEP 1 — Use fixed first/middle/last local queries for actual rule effects
    # Input: X/bid/Mar2 timestamps=[10:00,10:01,10:02,10:03,10:04]
    # Output: times=[10:00,10:02,10:04]; centers are evaluated on the same three queries
    # Trick: With only one or two timestamps, unique positions prevent duplicate queries.
    rows = []
    for (bond, side, day), g in events.groupby(["cusip", "side", "day"], observed=True, sort=True):
        timestamps = pd.DatetimeIndex(g[KEYS[-1]].drop_duplicates().sort_values())
        times = timestamps[np.unique([0, len(timestamps) // 2, len(timestamps) - 1])]
        f = side_features_fast(g, times, age_min=30)
        # CORE LOGIC: STEP 2 — Measure center differences and retained dealer slots on the same queries
        # Input: equal=[10,20,30], clip=[10,18,30], downweight=[10,20,30], max_age=[10,20,29]; n_dealers=[2,2,2], n_fresh_dealers=[2,1,1], n_peer_supported=[0,1,0]
        # Output: maximum changes per query=[0,2,1]; slots=6, fresh=4, supported=1
        changes = pd.concat([(f[v] - f["center_equal"]).abs() for v in
                             ["center_candidate_clip", "center_dealer_downweight", "center_max_age"]], axis=1)
        slots = float(f["n_dealers"].sum())
        fresh = float(f["n_fresh_dealers"].sum())
        supported = int(f["n_peer_supported"].gt(0).sum())
        # CORE LOGIC: STEP 3 — Report local center/coverage impact with its query denominator
        # Input: bond='X',side='bid',day=Mar2; times has 3 entries; per-query changes=[0,2,1], slots=6, fresh=4, supported=1
        # Output: X/bid/Mar2 row: impact_queries=3, peer_supported_queries=1, max_center_effect_bps=2, mean_center_effect_bps=1, max_age_lost_slots=2, coverage_loss_fraction=1/3
        # Trick: These are fixed local probes, not prediction improvement or trade-weighted population loss.
        rows.append(dict(cusip=bond, side=side, day=day, impact_queries=len(times),
                         peer_supported_queries=supported,
                         max_center_effect_bps=float(changes.max(axis=1).max()),
                         mean_center_effect_bps=float(changes.max(axis=1).mean()),
                         baseline_dealer_slots=slots, fresh_dealer_slots=fresh,
                         max_age_lost_slots=slots - fresh,
                         coverage_loss_fraction=(slots - fresh) / slots if slots else 0.0))
    return pd.DataFrame(rows, columns=["cusip", "side", "day", "impact_queries", "peer_supported_queries",
        "max_center_effect_bps", "mean_center_effect_bps", "baseline_dealer_slots", "fresh_dealer_slots",
        "max_age_lost_slots", "coverage_loss_fraction"])


# SETUP LOGIC: Declare _pick; existing arguments and docstring are preserved
def _pick(pool, used, count, required=None):
    # CORE LOGIC: STEP 1 — Exclude already-used cases and count existing entity representation
    # Input: used cases=[A/I/X,A/I/Y]; pool also contains A/I/Z and B/J/W
    # Output: dealer_counts A=2, issuer_counts I=2; remaining excludes X and Y
    selected, dealer_counts, issuer_counts = [], {}, {}
    for _, row in used.iterrows():
        dealer_counts[row["firm"]] = dealer_counts.get(row["firm"], 0) + 1
        issuer_counts[row["ISSUER"]] = issuer_counts.get(row["ISSUER"], 0) + 1
    remaining = pool.loc[~pool["case_id"].isin(used["case_id"])]
    # SETUP LOGIC: Declare take; existing arguments and docstring are preserved
    def take(row, cap=True):
        # CORE LOGIC: STEP 1 — Apply the soft entity cap while taking a distinct case
        # Input: selected=[], dealer_counts={'A':2}, issuer_counts={'I':2}, row=(case_id='Z',firm='A',ISSUER='I'), cap=True
        # Output: False with counts unchanged; the same row with cap=False appends Z and sets both counts to 3
        # Trick: The cap is relaxed only in the fallback pass, after sector/activity balancing cannot fill the requested count.
        if row["case_id"] in selected:
            return False
        if cap and (dealer_counts.get(row["firm"], 0) >= 2 or issuer_counts.get(row["ISSUER"], 0) >= 2):
            return False
        selected.append(row["case_id"])
        dealer_counts[row["firm"]] = dealer_counts.get(row["firm"], 0) + 1
        issuer_counts[row["ISSUER"]] = issuer_counts.get(row["ISSUER"], 0) + 1
        return True
    # CORE LOGIC: STEP 2 — Satisfy rare-support requirements before balanced traversal
    # Input: remaining=[(case_id=W,firm=B,ISSUER=J,SECTOR=Energy,activity=Sparse,unknown_rate=1)]; selected=[]; both counts zero; required unknown_rate>0
    # Output: selected=['W']; strata contains one ('Energy','Sparse') group with W
    for predicate in required or []:
        for _, row in remaining.loc[predicate(remaining)].iterrows():
            if take(row):
                break
    strata = list(remaining.groupby(["SECTOR", "activity"], observed=True, sort=True))
    # Round-robin sector/activity strata; each stratum retains the supplied order.
    # CORE LOGIC: STEP 3 — Traverse strata round-robin, then relax caps only if needed
    # Input: Energy stratum=[E1,E2], Utilities=[U1,U2], count=2, none capped
    # Output: selection=[E1,U1]; count limit returns immediately
    # Trick: Positions alternate across strata while preserving the incoming order within each stratum.
    for cap in [True, False]:
        for position in range(max((len(g) for _, g in strata), default=0)):
            for _, group in strata:
                if len(selected) >= count:
                    return remaining.set_index("case_id").loc[selected].reset_index()
                if position < len(group):
                    take(group.iloc[position], cap)
    return remaining.set_index("case_id").loc[selected[:count]].reset_index()


# SETUP LOGIC: Declare fixed_case_manifest; existing arguments and docstring are preserved
def fixed_case_manifest(population, impacts=None, n_each=6, impact_source=None):
    """Freeze random/typical/measured-impact cases; stable across input row order.

    Soft dealer/issuer cap of two is relaxed only to fill a small available pool.
    Random cases deliberately include sparse, one-sided and unknown-size units
    where available. High-impact order uses measured center/coverage changes.
    """
    # CORE LOGIC: STEP 1 — Choose supplied impacts or clearly labeled local probes
    # Input: events has zero rows with the expected event schema; impacts is a supplied zero-row table with bond/side/day and effect columns
    # Output: empty manifest includes typed selection, selection_reason, impact_source, case_number and cap_relaxed columns; population stores it
    c = case_candidates(population["events"])
    local_queries = impacts is None
    impacts = measure_rule_impacts(population["events"]) if local_queries else impacts
    impact_source = impact_source or ("first/middle/last local queries" if local_queries else impacts.attrs.get("query_source", "supplied common queries"))
    if c.empty:
        result = c.assign(selection=pd.Series(dtype=str), selection_reason=pd.Series(dtype=str),
                          impact_source=pd.Series(dtype=str), case_number=pd.Series(dtype=int),
                          cap_relaxed=pd.Series(dtype=bool))
        population["case_manifest"], population["case_candidates"], population["impact_table"] = result, c, impacts
        return result
    # CORE LOGIC: STEP 2 — Select stable random cases with sparse/one-sided/unknown support
    # Input: n_each=1; c has only A/X/bid/Mar2: SECTOR=Energy, activity='Sparse (1-3)', events=2, one_sided_day=True, unknown_rate=0.5; matching supplied impact center=0/loss=0
    # Output: Random contains A/X/bid/Mar2 once; used now contains that case ID
    c = c.merge(impacts, on=["cusip", "side", "day"], how="left", validate="many_to_one")
    used = c.iloc[:0].copy()
    random = _pick(c.sort_values("stable_hash"), used, n_each,
                   [lambda x: x["events"].le(3), lambda x: x["one_sided_day"], lambda x: x["unknown_rate"].gt(0)])
    random["selection"] = "Random"
    random["selection_reason"] = "seed 2026; sector/activity strata; sparse/one-sided/unknown support"
    used = pd.concat([used, random], ignore_index=True)
    # CORE LOGIC: STEP 3 — Choose typical cases using within-stratum medians
    # Input: one Energy/Sparse stratum has cases T0,T1,T2,R with multi_rate=[0,0.5,1,0.5], unknown_rate=refresh_rate=0; used={R}, n_each=1
    # Output: median profile=[0.5,0,0]; typical_distance=[0.5,0,0.5,0]; Typical selects T1 after excluding R
    # Trick: transform broadcasts the stratum median back to each row; already-selected random IDs are excluded.
    metrics = ["multi_rate", "unknown_rate", "refresh_rate"]
    med = c.groupby(["SECTOR", "activity"], observed=True)[metrics].transform("median")
    c["typical_distance"] = (c[metrics] - med).abs().sum(axis=1)
    typical = _pick(c.sort_values(["typical_distance", "stable_hash"], kind="stable"), used, n_each)
    typical["selection"] = "Typical"
    typical["selection_reason"] = "nearest within-stratum median ambiguity/quantity/refresh profile"
    used = pd.concat([used, typical], ignore_index=True)
    # CORE LOGIC: STEP 4 — Rank high impact by measured center and coverage changes
    # Input: remaining cases A: center=0, loss=0; B: center=2, loss=0.2; C: center=1, loss=0.1
    # Output: A excluded; B score=2.0, C score=1.0; B precedes C
    # Trick: Percentile ranks combine effects with different units; raw candidate gap alone cannot qualify a high-impact case.
    measured = c.loc[c["max_center_effect_bps"].gt(1e-9) | c["max_age_lost_slots"].gt(0)].copy()
    measured["impact_score"] = measured["max_center_effect_bps"].rank(pct=True) + measured["coverage_loss_fraction"].rank(pct=True)
    impact = _pick(measured.sort_values(["impact_score", "max_center_effect_bps", "stable_hash"],
                                       ascending=[False, False, True], kind="stable"), used, n_each)
    impact["selection"] = "High impact"
    impact["selection_reason"] = "actual rule center/coverage effect; " + impact_source
    # CORE LOGIC: STEP 5 — Freeze case numbering and report cap relaxation
    # Input: Random count=6, Typical count=6, High impact count=2; dealer A appears 3 times
    # Output: case_number=1..14; A cases cap_relaxed=True; same result is cached in population
    # Trick: A lack of remaining measured effects may yield fewer than 18 cases; no zero-effect high cases are invented.
    result = pd.concat([random, typical, impact], ignore_index=True)
    result["impact_source"] = impact_source
    result["case_number"] = np.arange(1, len(result) + 1)
    result["cap_relaxed"] = result["firm"].map(result["firm"].value_counts()).gt(2) | result["ISSUER"].map(result["ISSUER"].value_counts()).gt(2)
    population["case_manifest"], population["case_candidates"], population["impact_table"] = result, c, impacts
    return result


# SETUP LOGIC: Declare scope_options; existing arguments and docstring are preserved
def scope_options(population, scope):
    # UI LOGIC: STEP 1 — List scope groups already present in the small summary tables
    return ["Global"] if scope == "Global" else population["tables"][scope].index.tolist()


# SETUP LOGIC: Declare case_options; existing arguments and docstring are preserved
def case_options(manifest):
    """A compact common navigation label for the Step 2/3/4 dashboards."""
    # UI LOGIC: STEP 1 — Format stable case labels for Step 2/3/4 navigation
    return [(f"{r.case_number:02}. {r.selection} | {r.ISSUER} | {r.cusip} | {r.side} | {r.day.date()} | {r.firm}", r.case_id)
            for r in manifest.itertuples()]


# SETUP LOGIC: Declare summary_html; existing arguments and docstring are preserved
def summary_html(result):
    """Short decision-oriented text; detailed small tables remain in memory."""
    # UI LOGIC: STEP 1 — Render compact coverage, rate denominators and unassessed history text
    s = result["summary"]
    rate = lambda v: f"{v:.1%}" if pd.notna(v) else "N/A"
    history = (f"Unchanged pair refresh={int(s.unchanged_pair_refresh_events):,}, spread changes={int(s.changed_spread_events):,} "
               f"of {int(s.continuous_event_transitions):,} continuous transitions. "
               if pd.notna(s.continuous_event_transitions) else
               "Refresh/change history was not evaluated in Step 1. ")
    return (
        f"<b>{result['issuer']}</b>: quotes on {int(s.quoted_bonds):,}/{int(s.traded_bonds):,} traded bonds "
        f"({rate(s.bond_coverage)}); no quote={int(s.no_quote_bonds):,}; keyed-event coverage={int(s.keyed_quote_bonds):,}/{int(s.traded_bonds):,}. "
        f"Quote-file-window traded bond-days covered={int(s.quote_covered_traded_bond_days):,}/{int(s.traded_bond_days):,}; "
        f"outside quote-file-window trade days={int(s.outside_file_window_trade_days):,}; unknown window={int(s.unknown_quote_window_trade_days):,}. "
        f"Raw rows={int(s.raw_rows):,}; events={int(s.events):,}; dealer/bond/side/day={int(s.dealer_bond_side_days):,}. "
        f"Multi={int(s.multi_events):,}/{int(s.events):,} ({rate(s.multi_event_rate)}); "
        f"issuer equal={rate(s.issuer_equal_multi_rate)} (N={int(s.observed_issuers):,}), dealer equal={rate(s.dealer_equal_multi_rate)} (N={int(s.observed_dealers):,}), "
        f"unit equal={rate(s.unit_equal_multi_rate)}. "
        f"Same positive quantity still multi={int(s.same_positive_multi_events):,}/{int(s.events):,}; "
        f"affected bond-days={int(s.affected_bond_days):,}/{int(s.observed_bond_days):,}. "
        + history +
        "<br>Decision: preserve unknown/zero quantity and multi-price candidates; use coverage and ambiguity "
        "features before any size rule. Equal rates describe quote-observed groups; no-quote bonds stay in universe coverage. "
        "Dealer coverage uses the full traded universe. These counts do not establish prediction gain.")
