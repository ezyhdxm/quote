# %% [markdown]
# # Step 3 — quote cleaning choices and point-in-time feature drafts
# Run three cells. Choose issuer and dealer / bond / side / ET day.
# All four questions appear together; optional detail views enlarge one section.
# The shown rules are hypotheses for later chronological validation, not a fitted cleaner.

# %% 1. Load the same data and traded-bond universe
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
if globals().get("step3_source") is not bcq_df:
    step3_prepared = prepare_quote_events(bcq_df)
    step3_source = bcq_df
    step3_raw = bcq_df[KEYS + ["spread", "quantity", "ISSUER"]].copy()
    step3_raw["repeat"] = bcq_df.duplicated()
    step3_raw["s"] = pd.to_numeric(step3_raw.spread, errors="coerce").replace([np.inf, -np.inf], np.nan)
    step3_raw["q"] = pd.to_numeric(step3_raw.quantity, errors="coerce")
    step3_raw["qkind"] = np.select(
        [step3_raw.quantity.isna(), step3_raw.q.eq(0), np.isfinite(step3_raw.q) & step3_raw.q.gt(0)],
        ["Missing", "Zero", "Positive"], default="Other")
    step3_raw["qtag"] = step3_raw.qkind.astype(str)
    positive = step3_raw.qkind.eq("Positive")
    step3_raw.loc[positive, "qtag"] = step3_raw.loc[positive, "q"].map(lambda v: "q=" + repr(float(v)))
    other = step3_raw.qkind.eq("Other")
    step3_raw.loc[other, "qtag"] = "Other:" + step3_raw.loc[other, "quantity"].astype(str)
    step3_raw["bad"] = step3_raw.s.isna()
    step3_case_cache = {}
    step3_population = None
    step3_manifest = None


def issuer_choices():
    """Rank the shared event table; navigation labels are descriptive only."""
    g = step3_prepared["events"].copy()
    g["issuer"] = g.cusip.map(cusip_issuer).astype("string").fillna("[Missing issuer]")
    g["multi"] = g.candidate_count.gt(1)
    g["size_multi"] = g["multi"] & g.quantity_set.map(lambda tags: sum(t.startswith("q=") for t in tags) > 1)
    scores = g.groupby("issuer", observed=True).agg(
        events=("candidate_count", "size"), multi=("multi", "sum"), size_multi=("size_multi", "sum"),
        days=("day", "nunique"), dealers=("firm", "nunique"), gap=("gap", "median"))
    scores["rate"] = scores["multi"] / scores["events"]
    supported = scores.loc[scores.events.ge(100) & scores.days.ge(2) & scores.dealers.ge(2)]
    multi = supported.loc[supported.multi.ge(5)]
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


def asof_features(events, times, age_min=DEFAULT_AGE_MIN):
    """Fast shared summaries plus case slots and fixed-lag composition diagnostics."""
    from quote_quality_core import side_features_fast
    requested = pd.DatetimeIndex(times).sort_values().unique()
    lagged = requested - pd.Timedelta(minutes=LOOKBACK_MIN)
    queries = requested.union(lagged)
    features = side_features_fast(events, queries, age_min)
    base = pd.DataFrame({"time": queries})
    parts = []
    for firm, history in events.groupby("firm", observed=True):
        merged = pd.merge_asof(base, history.sort_values(KEYS[-1]), left_on="time", right_on=KEYS[-1], direction="backward")
        merged = merged.loc[merged[KEYS[-1]].notna() & merged.time.dt.normalize().eq(merged.day)].copy()
        if merged.empty:
            continue
        merged[["complete", "change_age_unknown"]] = merged[["complete", "change_age_unknown"]].astype(bool)
        merged["message_age_min"] = (merged.time - merged[KEYS[-1]]).dt.total_seconds() / 60
        merged["spread_set_change_age_min"] = (merged.time - merged.last_change).dt.total_seconds() / 60
        merged["observed_history_min"] = (merged.time - merged.history_start).dt.total_seconds() / 60
        parts.append(merged)
    slots = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(events.columns) + ["time", "message_age_min", "spread_set_change_age_min", "observed_history_min"])
    n, dealers = len(queries), pd.Index(events.firm.drop_duplicates())
    d = len(dealers)
    center = np.full((n, d), np.nan)
    age = np.full((n, d), np.nan)
    conditions = np.full((n, d), -1, dtype=int)
    slots["peer_center"], slots["peer_radius"], slots["peer_residual"] = np.nan, np.nan, np.nan
    slots["n_peers"], slots["clipped_candidate_count"] = 0, 0
    slots["clipped_center"], slots["dealer_weight"] = slots["center"], 1.0
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
            if not supported.any():
                continue
            idx = own_rows[supported]
            peers = np.where(peer_mask[supported], center[own_time[supported]], np.nan)
            ref = np.nanmedian(peers, axis=1)
            radius = np.maximum(CLIP_FLOOR_BPS, MAD_MULTIPLIER * 1.4826 * np.nanmedian(np.abs(peers - ref[:, None]), axis=1))
            residual = slots.loc[idx, "center"].to_numpy() - ref
            candidate_sets = slots.loc[idx, "spread_set"]
            middle_low = candidate_sets.map(lambda x: x[(len(x) - 1) // 2]).to_numpy()
            middle_high = candidate_sets.map(lambda x: x[len(x) // 2]).to_numpy()
            clipped = (np.clip(middle_low, ref - radius, ref + radius) + np.clip(middle_high, ref - radius, ref + radius)) / 2
            weights = np.minimum(1, np.divide(radius, np.abs(residual), out=np.ones(len(idx)), where=residual != 0))
            slots.loc[idx, ["peer_center", "peer_radius", "peer_residual", "clipped_center", "dealer_weight"]] = np.column_stack([ref, radius, residual, clipped, weights])
        supported = slots.n_peers.ge(MIN_PEERS)
        if supported.any():
            candidates = slots.loc[supported, ["spread_set", "peer_center", "peer_radius"]].explode("spread_set")
            clipped = (pd.to_numeric(candidates.spread_set) - candidates.peer_center).abs().gt(candidates.peer_radius)
            slots.loc[supported, "clipped_candidate_count"] = clipped.groupby(level=0).sum().reindex(slots.index[supported]).to_numpy()
    now = features.reindex(requested).copy()
    before = features.reindex(lagged).set_axis(requested)
    ni, pi = queries.get_indexer(requested), queries.get_indexer(lagged)
    same_day = np.asarray(requested.normalize() == lagged.normalize())
    now_present, past_present = np.isfinite(center[ni]), np.isfinite(center[pi])
    common = now_present & past_present & same_day[:, None]
    common_n = common.sum(axis=1)
    eligible = now.n_dealers.gt(0) & before.n_dealers.gt(0) & same_day
    composition_changed = (now_present != past_present).any(axis=1) | ((conditions[ni] != conditions[pi]) & common).any(axis=1)
    now["composition_changed_30m"] = pd.Series(composition_changed, index=requested).astype(float).where(eligible)
    aggregate_delta = (now.center_equal - before.center_equal).where(eligible)
    # Preserve the old guarded feature; the new diagnostic retains common dealer
    # changes even when quantity support/candidate count changes.
    now["center_delta_30m"] = aggregate_delta.where(~composition_changed)
    now["aggregate_delta_30m"] = aggregate_delta
    common_difference = np.where(common, center[ni] - center[pi], np.nan)
    now["common_dealer_delta_30m"] = np.divide(np.nansum(common_difference, axis=1), common_n, out=np.full(len(requested), np.nan), where=common_n > 0)
    now["n_common_dealers_30m"] = common_n
    now["common_retention_30m"] = np.divide(common_n, now.n_dealers.to_numpy(), out=np.full(len(requested), np.nan), where=now.n_dealers.to_numpy() > 0)
    now["past_common_retention_30m"] = np.divide(common_n, before.n_dealers.to_numpy(), out=np.full(len(requested), np.nan), where=before.n_dealers.to_numpy() > 0)
    changed_n = ((conditions[ni] != conditions[pi]) & common).sum(axis=1)
    now["common_condition_changed_fraction_30m"] = np.divide(changed_n, common_n, out=np.full(len(requested), np.nan), where=common_n > 0)
    return slots.loc[slots.time.isin(requested)].reset_index(drop=True), now


def quantity_evidence(rows):
    """Same-event quantity contrasts, without equating units to TRACE size."""
    finite = rows.loc[rows["s"].notna()]
    cells = finite.groupby([KEYS[-1], "qtag"], observed=True).agg(
        lo=("s", "min"), hi=("s", "max"), center=("s", "median"), count=("s", "nunique"),
    ).reset_index()
    # Median of distinct spreads, never frequency-weighted by repeated rows.
    cells["center"] = finite.groupby([KEYS[-1], "qtag"], observed=True)["s"].agg(lambda v: np.median(v.unique())).to_numpy()
    eligible = rows.groupby(KEYS[-1])["qkind"].agg(lambda v: v.eq("Positive").all())
    complete = rows.groupby(KEYS[-1])["s"].agg(lambda v: v.notna().all())
    positive = cells.loc[cells[KEYS[-1]].isin(eligible.index[eligible & complete])]
    counts = positive.groupby(KEYS[-1])["qtag"].nunique()
    positive = positive.loc[positive[KEYS[-1]].isin(counts.index[counts.ge(2)])]
    # Remove any event where a positive size still maps to multiple spreads.
    unique = positive.groupby(KEYS[-1])["count"].max().eq(1)
    unique = positive.loc[positive[KEYS[-1]].isin(unique.index[unique])].copy()
    unique["within_event_residual"] = unique["center"] - unique.groupby(KEYS[-1])["center"].transform("median")
    return cells, unique


def research_figure(result, dealer, bond, side, day, view, qtag, age_min, canvas=None):
    """One shared case; all four decisions in a 2x2 dashboard by default."""
    raw = result["raw"]
    mask = raw["cusip"].eq(bond) & raw["side"].eq(side) & raw[KEYS[-1]].dt.normalize().eq(day)
    peers_raw = raw.loc[mask]
    rows = peers_raw.loc[peers_raw["firm"].eq(dealer)]
    g = result["events"]
    e = g.loc[g["firm"].eq(dealer) & g["cusip"].eq(bond) & g["side"].eq(side) & g["day"].eq(day)]
    slots, f = result["slots"], result["features"]
    own = slots.loc[slots["firm"].eq(dealer)] if len(slots) else slots
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
        finite = e["center"].notna()
        unquoted = int(e.loc[finite, "center_nearest_gap"].gt(1e-9).sum())
        notes = [f"OBSERVED: median is not a quoted candidate at {unquoted}/{int(finite.sum())} events; median candidate gap {e['gap'].median():.2f} bps.",
                 f"CHECK: {int(e['condition_changed'].sum())}/{int((~e['history_break']).sum())} comparable transitions change quantity support or count. Bounds are order statistics, not tracked streams.",
                 f"USE: common dealers median N={f['n_common_dealers_30m'].median():.1f}; current retention={f['common_retention_30m'].median():.1%}; changed conditions={f['common_condition_changed_fraction_30m'].mean():.1%}. Difference is composition sensitivity, not a causal decomposition. Adjacent-event changes remain in events."]
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
        quantity_key = (dealer, bond, side, day)
        diagnostics = result.setdefault("quantity_evidence_cache", {})
        if quantity_key not in diagnostics:
            diagnostics[quantity_key] = quantity_evidence(rows)
        cells, contrasts = diagnostics[quantity_key]
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
        notes = [f"OBSERVED: {int(selected_cells['count'].gt(1).sum())}/{len(selected_cells)} finite {qtag} events remain multi-spread; {len(selected_contrasts)} eligible contrasts.",
                 "ELIGIBLE: complete events, >=2 positive raw quantities, one spread per quantity. Zero/missing size cannot distinguish economic conditions.",
                 "USE: conditional summaries only where mapping repeats with coverage. Else retain pooled candidate ambiguity; no forced size curve."]
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
        covered = f["center_equal"].notna()
        lost = int((covered & f["center_max_age"].isna()).sum())
        unknown = int(own["change_age_unknown"].sum()) if len(own) else 0
        max_effect = (f["center_decay"] - f["center_equal"]).abs().max()
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
        assessed = f["n_peer_supported"].gt(0)
        b.set_title("Impact only where peers support a test"); b.set_ylabel("bps vs baseline")
        if assessed.any():
            for col, label, color in [("center_candidate_clip", "Candidate clip", "#C9563D"), ("center_dealer_downweight", "Dealer downweight", "#8560A5")]:
                delta = (f[col] - f["center_equal"]).where(assessed)
                b.plot(f.index, delta, color=color, label=label, marker=".", ms=3, drawstyle="steps-post")
            b.axhline(0, color="#888888", lw=0.6)
        else:
            b.text(0.5, 0.5, "NOT ASSESSED\nInsufficient fresh peers\nBaseline fallback is not a measured zero", transform=b.transAxes, ha="center", va="center", fontsize=9.5)
            b.set_yticks([])
        comparable = own.loc[own["complete"]] if len(own) else own
        supported = comparable.loc[comparable["n_peers"].ge(MIN_PEERS)] if len(comparable) else comparable
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
        affected = int(f.loc[assessed, "n_clipped_dealers"].gt(0).sum())
        centers = int(f.loc[assessed, "n_changed_centers"].gt(0).sum())
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


def step3_key():
    return (step3_issuer.value, step3_bond.value, step3_side.value, step3_day.value, step3_age.value)


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
        key = step3_key()
        if key not in step3_case_cache:
            step3_status.value = "Building this bond / side / ET day state once..."
            started = perf_counter()
            e = step3_result["events"]
            history = e.loc[e.cusip.eq(step3_bond.value) & e.side.eq(step3_side.value) & e.day.eq(step3_day.value)]
            today = history[KEYS[-1]]
            times = pd.date_range(today.min().ceil(GRID), today.max().floor(GRID), freq=GRID).union(pd.DatetimeIndex([today.min(), today.max()]))
            slots, features = asof_features(history, times, step3_age.value)
            step3_case_cache[key] = dict(slots=slots, features=features, asof_seconds=perf_counter() - started)
        step3_result.update(step3_case_cache[key])
        step3_cache_key = key
        draw_step3()
    finally:
        step3_apply.disabled = False
        step3_busy = False


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
