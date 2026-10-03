"""Small population tables and deterministic cases for the research notebooks.

Raw records remain unchanged. Counts of rows, events and dealer/bond/side/day
units are kept separate; quote absence is measured against the traded universe.
"""
from hashlib import sha256

import numpy as np
import pandas as pd

from quote_quality_core import KEYS, SERIES, side_features_fast

KINDS = ["Positive", "Zero", "Missing", "Other"]
QKINDS = ["Same positive", "Different positive", "Contains zero", "Missing / other"]
CASE_SEED = 2026


def _labels(values, missing):
    return values.astype("string").str.strip().replace("", pd.NA).fillna(missing)


def build_bond_universe(trades):
    """One CUSIP per supplied three-month traded universe, including no quotes.

    SECTOR is assigned only when exactly one distinct non-null label exists.
    This never chooses an arbitrary first SECTOR from conflicting records.
    """
    t = trades.loc[trades["CUSIP"].notna()].copy()
    t["CUSIP"] = t["CUSIP"].astype("string").str.strip()
    t = t.loc[t["CUSIP"].ne("")]
    u = pd.DataFrame({"cusip": sorted(t["CUSIP"].unique())}).set_index("cusip")
    for source, missing, conflict in [("SECTOR", "Unknown", "Conflicting"),
                                      ("ISSUER", "[Missing issuer]", "[Conflicting issuer]")]:
        if source in t:
            labels = t[source].astype("string").str.strip().replace("", pd.NA)
            n = labels.groupby(t["CUSIP"]).nunique()
            first = labels.groupby(t["CUSIP"]).first()
            u[source] = first.reindex(u.index).fillna(missing).mask(n.reindex(u.index).gt(1), conflict)
        else:
            u[source] = missing
    u["trade_rows"] = t.groupby("CUSIP").size().reindex(u.index, fill_value=0)
    return u.reset_index()


def attach_quote_metadata(quotes, universe):
    """Annotate quote rows within the universe without filtering by quote value."""
    q = quotes.loc[quotes["cusip"].isin(universe["cusip"])].copy()
    metadata = universe.set_index("cusip")
    for col in ["ISSUER", "SECTOR"]:
        q[col] = q["cusip"].map(metadata[col])
    return q


def quantity_rows(raw):
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


def _same_positive_multi(pairs):
    sizes = {}
    for spread, tag in pairs:
        if tag.startswith("q=") and isinstance(spread, (int, float, np.number)) and np.isfinite(spread):
            sizes.setdefault(tag, set()).add(float(spread))
    return any(len(v) > 1 for v in sizes.values())


def _event_table(events, universe):
    g = events.copy()
    meta = universe.set_index("cusip")
    for col in ["ISSUER", "SECTOR"]:
        g[col] = g["cusip"].map(meta[col])
    g["multi"] = g["candidate_count"].ge(2)
    g["same_positive_multi"] = g["pair_set"].map(_same_positive_multi).astype(bool)
    g["has_zero"] = g["quantity_set"].map(lambda x: "Zero" in x).astype(bool)
    g["all_zero"] = g["quantity_set"].map(lambda x: x == ("Zero",)).astype(bool)
    g["has_unknown"] = g["quantity_set"].map(lambda x: any(v == "Missing" or v.startswith("Other:") for v in x)).astype(bool)
    g["n_quantity"] = g["quantity_set"].map(lambda x: sum(v.startswith("q=") for v in x))
    g["qclass"] = np.select([g["has_unknown"], g["has_zero"], g["n_quantity"].gt(1)],
                            [QKINDS[3], QKINDS[2], QKINDS[1]], default=QKINDS[0])
    # Backward compatible names used by the Step 2 figures.
    g["n_spreads"], g["range_bps"], g["bad_spreads"] = g["candidate_count"], g["gap"], g["bad"]
    return g


def _summary_row(q, g, u, all_events, trade_days):
    n, m = len(g), int(g["multi"].sum())
    days = g.groupby(SERIES + ["day"], observed=True)["multi"].agg(["mean", "max"])
    bond_days = g.groupby(["cusip", "day"], observed=True)["multi"].max()
    raw_counts = q["quantity_kind"].value_counts().reindex(KINDS, fill_value=0)
    def equal_rate(col):
        return float(g.groupby(col, observed=True)["multi"].mean().mean()) if n else np.nan
    quoted_bonds = q["cusip"].nunique()
    observed = set(zip(q["cusip"], q["quote_timestamp_ET"].dt.normalize()))
    universe_days = trade_days.loc[trade_days["cusip"].isin(u["cusip"])]
    target_days = universe_days.loc[universe_days["in_quote_window"]]
    covered_trade_days = sum(key in observed for key in target_days[["cusip", "day"]].itertuples(index=False, name=None))
    r = dict(traded_bonds=len(u), quoted_bonds=quoted_bonds, no_quote_bonds=len(u) - quoted_bonds,
             bond_coverage=quoted_bonds / len(u) if len(u) else np.nan,
             traded_bond_days=len(target_days), quote_covered_traded_bond_days=covered_trade_days,
             three_month_traded_bond_days=len(universe_days),
             outside_file_window_trade_days=int(((~universe_days["in_quote_window"]) & universe_days["quote_window_known"]).sum()),
             unknown_quote_window_trade_days=int((~universe_days["quote_window_known"]).sum()),
             raw_rows=len(q), unkeyed_rows=len(q) - int(g["rows"].sum()), events=n, multi_events=m,
             observed_issuers=g["ISSUER"].nunique(), observed_dealers=g["firm"].nunique(),
             traded_issuers=u["ISSUER"].nunique(),
             multi_event_rate=m / n if n else np.nan,
             dealer_bond_side_days=len(days), affected_dealer_bond_side_days=int(days["max"].sum()),
             unit_equal_multi_rate=float(days["mean"].mean()),
             affected_unit_rate=float(days["max"].mean()),
             observed_bond_days=len(bond_days), affected_bond_days=int(bond_days.sum()),
             issuer_equal_multi_rate=equal_rate("ISSUER"), dealer_equal_multi_rate=equal_rate("firm"),
             same_positive_multi_events=int(g["same_positive_multi"].sum()),
             unknown_quantity_events=int(g["has_unknown"].sum()), zero_quantity_events=int(g["has_zero"].sum()),
             keyed_quote_bonds=g["cusip"].nunique(), no_keyed_event_bonds=len(u) - g["cusip"].nunique(),
             incomplete_events=int((~g["complete"]).sum()), repeats=int(q["_source_repeat"].sum()),
             continuous_event_transitions=int((~g["history_break"]).sum()),
             unchanged_pair_refresh_events=int(g["pair_refresh"].sum()),
             changed_spread_events=int(g["spread_changed"].sum()),
             changed_condition_events=int(g["condition_changed"].sum()),
             share_of_all_events=n / max(all_events, 1))
    for state in QKINDS:
        r["quantity_event_" + state.lower().replace(" / ", "_").replace(" ", "_")] = int(g["qclass"].eq(state).sum())
    for k in KINDS:
        r[f"raw_{k.lower()}_rows"] = int(raw_counts[k])
        r[f"raw_{k.lower()}_share"] = raw_counts[k] / len(q) if len(q) else np.nan
    return r


def population_tables(trades, quotes, event_cache=None):
    """Build once; dashboards slice these tables without recomputing histories.

    Dealer coverage denominators are every traded bond, not dealer's observed
    bonds. Equal issuer/dealer rates include quote-observed groups only, whose
    counts are explicit; no-quote bonds remain in universe coverage denominators.
    """
    u = build_bond_universe(trades)
    raw = attach_quote_metadata(quotes, u)
    q = quantity_rows(raw)
    q["_source_repeat"] = raw.duplicated().to_numpy()
    if event_cache is None:
        from quote_quality_cache import prepare_quote_events
        event_cache = prepare_quote_events(raw)
    g = _event_table(event_cache["events"], u)
    timestamps = pd.to_datetime(trades["EFFECTIVE_DATETIME_TS"], errors="coerce")
    if timestamps.dt.tz is None:
        timestamps = timestamps.dt.tz_localize("America/New_York")
    else:
        timestamps = timestamps.dt.tz_convert("America/New_York")
    td = pd.DataFrame({"cusip": trades["CUSIP"], "day": timestamps.dt.normalize()}).dropna().drop_duplicates()
    quote_dates = quotes["quote_timestamp_ET"].dropna().dt.normalize()
    td["in_quote_window"] = (td["day"].between(quote_dates.min(), quote_dates.max())
                             if len(quote_dates) else False)
    td["quote_window_known"] = bool(len(quote_dates))
    tables = {}
    tables["Global"] = pd.DataFrame([_summary_row(q, g, u, len(g), td)], index=["Global"])
    # Group indices avoid rescanning the full quote/event table for every group.
    for scope, col in [("SECTOR", "SECTOR"), ("Issuer", "ISSUER"), ("Dealer", "firm")]:
        qi, gi = q.groupby(col, observed=True).indices, g.groupby(col, observed=True).indices
        ui = u.groupby(col, observed=True).indices if col != "firm" else {}
        names = sorted(set(qi) | set(ui))
        rows = []
        for name in names:
            qq = q.iloc[qi.get(name, [])]
            gg = g.iloc[gi.get(name, [])]
            uu = u if col == "firm" else u.iloc[ui.get(name, [])]
            rows.append(_summary_row(qq, gg, uu, len(g), td))
        if scope == "Dealer" and not names:
            names = ["[No observed dealer]"]
            rows = [_summary_row(q.iloc[:0], g.iloc[:0], u, len(g), td)]
        tables[scope] = pd.DataFrame(rows, index=pd.Index(names, name=col))
    return {"universe": u, "raw": raw, "quantity": q, "events": g, "tables": tables,
            "event_cache": event_cache, "unkeyed": event_cache["unkeyed"], "trade_days": td,
            "case_manifest": None, "case_candidates": None, "impact_table": None}


def scope_selection(population, scope="Global", value=None):
    q, g, u = population["quantity"], population["events"], population["universe"]
    col = {"SECTOR": "SECTOR", "Issuer": "ISSUER", "Dealer": "firm"}.get(scope)
    if col is not None:
        q, g = q.loc[q[col].eq(value)], g.loc[g[col].eq(value)]
        if col != "firm":
            u = u.loc[u[col].eq(value)]
    label = "Global" if scope == "Global" else f"{scope}: {value}"
    summary = population["tables"][scope].loc["Global" if scope == "Global" else value]
    return {"raw": q, "groups": g, "universe": u, "summary": summary,
            "issuer": label, "scope": scope, "value": value}


def case_candidates(events):
    """Dealer/bond/side/day units; quote-absent universe stays in summary only."""
    columns = SERIES + ["day"]
    c = events.groupby(columns, observed=True, sort=True).agg(
        ISSUER=("ISSUER", "first"), SECTOR=("SECTOR", "first"), events=("multi", "size"),
        multi_rate=("multi", "mean"), unknown_rate=("has_unknown", "mean"),
        refresh_rate=("pair_refresh", "mean"), multi=("multi", "sum"),
        same_positive_multi=("same_positive_multi", "sum")).reset_index()
    both = events.groupby(["firm", "cusip", "day"], observed=True)["side"].nunique()
    c["one_sided_day"] = [both.loc[(r.firm, r.cusip, r.day)] == 1 for r in c.itertuples()]
    c["activity"] = pd.cut(c["events"], [0, 3, 30, np.inf], labels=["Sparse (1-3)", "Medium (4-30)", "Active (>30)"])
    c["case_id"] = ["|".join([str(v) for v in row]) for row in c[columns].itertuples(index=False, name=None)]
    c["stable_hash"] = c["case_id"].map(lambda v: sha256(f"{CASE_SEED}|{v}".encode()).hexdigest())
    return c


def measure_rule_impacts(events):
    """Actual rule changes at three fixed local queries per bond/side/ET day.

    First, middle and last observed event timestamps are selected once. This is
    an exhaustive observed bond/side/day panel with a sparse query denominator,
    not a trade-weighted estimate or a claim about prediction improvement.
    """
    rows = []
    for (bond, side, day), g in events.groupby(["cusip", "side", "day"], observed=True, sort=True):
        timestamps = pd.DatetimeIndex(g[KEYS[-1]].drop_duplicates().sort_values())
        times = timestamps[np.unique([0, len(timestamps) // 2, len(timestamps) - 1])]
        f = side_features_fast(g, times, age_min=30)
        changes = pd.concat([(f[v] - f["center_equal"]).abs() for v in
                             ["center_candidate_clip", "center_dealer_downweight", "center_max_age"]], axis=1)
        slots = float(f["n_dealers"].sum())
        fresh = float(f["n_fresh_dealers"].sum())
        supported = int(f["n_peer_supported"].gt(0).sum())
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


def _pick(pool, used, count, required=None):
    selected, dealer_counts, issuer_counts = [], {}, {}
    for _, row in used.iterrows():
        dealer_counts[row["firm"]] = dealer_counts.get(row["firm"], 0) + 1
        issuer_counts[row["ISSUER"]] = issuer_counts.get(row["ISSUER"], 0) + 1
    remaining = pool.loc[~pool["case_id"].isin(used["case_id"])]
    def take(row, cap=True):
        if row["case_id"] in selected:
            return False
        if cap and (dealer_counts.get(row["firm"], 0) >= 2 or issuer_counts.get(row["ISSUER"], 0) >= 2):
            return False
        selected.append(row["case_id"])
        dealer_counts[row["firm"]] = dealer_counts.get(row["firm"], 0) + 1
        issuer_counts[row["ISSUER"]] = issuer_counts.get(row["ISSUER"], 0) + 1
        return True
    for predicate in required or []:
        for _, row in remaining.loc[predicate(remaining)].iterrows():
            if take(row):
                break
    strata = list(remaining.groupby(["SECTOR", "activity"], observed=True, sort=True))
    # Round-robin sector/activity strata; each stratum retains the supplied order.
    for cap in [True, False]:
        for position in range(max((len(g) for _, g in strata), default=0)):
            for _, group in strata:
                if len(selected) >= count:
                    return remaining.set_index("case_id").loc[selected].reset_index()
                if position < len(group):
                    take(group.iloc[position], cap)
    return remaining.set_index("case_id").loc[selected[:count]].reset_index()


def fixed_case_manifest(population, impacts=None, n_each=6, impact_source=None):
    """Freeze random/typical/measured-impact cases; stable across input row order.

    Soft dealer/issuer cap of two is relaxed only to fill a small available pool.
    Random cases deliberately include sparse, one-sided and unknown-size units
    where available. High-impact order uses measured center/coverage changes.
    """
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
    c = c.merge(impacts, on=["cusip", "side", "day"], how="left", validate="many_to_one")
    used = c.iloc[:0].copy()
    random = _pick(c.sort_values("stable_hash"), used, n_each,
                   [lambda x: x["events"].le(3), lambda x: x["one_sided_day"], lambda x: x["unknown_rate"].gt(0)])
    random["selection"] = "Random"
    random["selection_reason"] = "seed 2026; sector/activity strata; sparse/one-sided/unknown support"
    used = pd.concat([used, random], ignore_index=True)
    metrics = ["multi_rate", "unknown_rate", "refresh_rate"]
    med = c.groupby(["SECTOR", "activity"], observed=True)[metrics].transform("median")
    c["typical_distance"] = (c[metrics] - med).abs().sum(axis=1)
    typical = _pick(c.sort_values(["typical_distance", "stable_hash"], kind="stable"), used, n_each)
    typical["selection"] = "Typical"
    typical["selection_reason"] = "nearest within-stratum median ambiguity/quantity/refresh profile"
    used = pd.concat([used, typical], ignore_index=True)
    measured = c.loc[c["max_center_effect_bps"].gt(1e-9) | c["max_age_lost_slots"].gt(0)].copy()
    measured["impact_score"] = measured["max_center_effect_bps"].rank(pct=True) + measured["coverage_loss_fraction"].rank(pct=True)
    impact = _pick(measured.sort_values(["impact_score", "max_center_effect_bps", "stable_hash"],
                                       ascending=[False, False, True], kind="stable"), used, n_each)
    impact["selection"] = "High impact"
    impact["selection_reason"] = "actual rule center/coverage effect; " + impact_source
    result = pd.concat([random, typical, impact], ignore_index=True)
    result["impact_source"] = impact_source
    result["case_number"] = np.arange(1, len(result) + 1)
    result["cap_relaxed"] = result["firm"].map(result["firm"].value_counts()).gt(2) | result["ISSUER"].map(result["ISSUER"].value_counts()).gt(2)
    population["case_manifest"], population["case_candidates"], population["impact_table"] = result, c, impacts
    return result


def scope_options(population, scope):
    return ["Global"] if scope == "Global" else population["tables"][scope].index.tolist()


def case_options(manifest):
    """A compact common navigation label for the Step 2/3/4 dashboards."""
    return [(f"{r.case_number:02}. {r.selection} | {r.ISSUER} | {r.cusip} | {r.side} | {r.day.date()} | {r.firm}", r.case_id)
            for r in manifest.itertuples()]


def summary_html(result):
    """Short decision-oriented text; detailed small tables remain in memory."""
    s = result["summary"]
    rate = lambda v: f"{v:.1%}" if pd.notna(v) else "N/A"
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
        f"Unchanged pair refresh={int(s.unchanged_pair_refresh_events):,}, spread changes={int(s.changed_spread_events):,} "
        f"of {int(s.continuous_event_transitions):,} continuous transitions. "
        "<br>Decision: preserve unknown/zero quantity and multi-price candidates; use coverage and ambiguity "
        "features before any size rule. Equal rates describe quote-observed groups; no-quote bonds stay in universe coverage. "
        "Dealer coverage uses the full traded universe. These counts do not establish prediction gain.")
