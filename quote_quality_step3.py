# %% [markdown]
# # Step 3 — quote cleaning choices and point-in-time feature drafts
# Run three cells. Choose issuer, research question, dealer / bond / side / ET day.
# Each view compares a proposed rule and explains its feature use. No rows are deleted.
# The shown rules are hypotheses for later chronological validation, not a fitted cleaner.

# %% 1. Load the same data and traded-bond universe
from pathlib import Path
import numpy as np
import pandas as pd
from io import BytesIO
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import ipywidgets as widgets
from IPython.display import display

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


def issuer_choices():
    """Lightweight global ranking; detailed event histories are built on demand."""
    x = bcq_df[KEYS + ["spread", "quantity"]].assign(issuer=issuer_labels)
    x["spread"] = pd.to_numeric(x["spread"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    size = pd.to_numeric(x["quantity"], errors="coerce")
    x["positive_q"] = size.where(np.isfinite(size) & size.gt(0))
    g = x.dropna(subset=KEYS).groupby(["issuer"] + KEYS, observed=True).agg(
        lo=("spread", "min"), hi=("spread", "max"), n=("spread", "nunique"), nq=("positive_q", "nunique"),
    ).reset_index()
    g["multi"], g["size_multi"] = g["n"].gt(1), g["n"].gt(1) & g["nq"].gt(1)
    g["gap"] = (g["hi"] - g["lo"]).where(g["multi"])
    g["day"] = g["quote_timestamp_ET"].dt.normalize()
    scores = g.groupby("issuer", observed=True).agg(
        events=("n", "size"), multi=("multi", "sum"), size_multi=("size_multi", "sum"),
        days=("day", "nunique"), dealers=("firm", "nunique"), gap=("gap", "median"),
    )
    scores["rate"] = scores["multi"] / scores["events"]
    supported = scores.loc[scores["events"].ge(100) & scores["days"].ge(2) & scores["dealers"].ge(2)]
    multi = supported.loc[supported["multi"].ge(5)]
    promoted = {}
    # Previously inspected cases remain easy to find, when present in this load.
    for prefix, reason in [("SKY GROUP", "Prior multi"), ("EASTERN GAS", "Prior multi"),
                           ("DUKE ENERGY", "Prior quantity"), ("IBM", "Prior same size"),
                           ("EXPAND ENERGY", "Prior wide"), ("HPS CORPORATE", "Prior wide"),
                           ("COMCAST", "Prior control"), ("MITSUBISHI UFJ", "Prior control")]:
        for name in sorted(issuer_labels.unique()):
            if name.upper().startswith(prefix):
                promoted[name] = reason
                break
    for tag, pool, field, ascending in [
        ("Multi", multi, "rate", False), ("Wide", multi, "gap", False),
        ("Quantity", multi.loc[multi["size_multi"].ge(5)], "size_multi", False),
        ("Control", supported, "rate", True),
    ]:
        for name in pool.sort_values([field, "events"], ascending=[ascending, False]).index[:2]:
            promoted.setdefault(name, tag)
    order = list(promoted) + [n for n in sorted(issuer_labels.unique()) if n not in promoted]
    return [(f"[{promoted[n]}] {n}" if n in promoted else n, n) for n in order]


def event_history(raw):
    """Distinct numeric spread / raw-size-condition sets, with prefix-only changes."""
    q = raw.copy()
    valid = q[KEYS].notna().all(axis=1)
    for key in SERIES:
        valid &= q[key].astype("string").str.strip().ne("").fillna(False)
    q["repeat"] = q.duplicated()
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
    g = usable.groupby(KEYS, observed=True, sort=False).agg(
        rows=("s", "size"), repeats=("repeat", "sum"), bad=("bad", "sum"),
        spread_set=("s", lambda v: tuple(sorted(v.dropna().unique()))),
        quantity_set=("qtag", lambda v: tuple(sorted(v.unique()))),
        pair_set=("pair", lambda v: frozenset(v)),
    ).reset_index().sort_values(SERIES + [KEYS[-1]], kind="stable").reset_index(drop=True)
    g["candidate_count"] = g["spread_set"].map(len)
    for col, fn in [("lo", min), ("hi", max), ("center", np.median)]:
        g[col] = g["spread_set"].map(lambda v: float(fn(v)) if v else np.nan)
    g["gap"] = g["hi"] - g["lo"]
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
    two_back = group["pair_set"].shift(2)
    g["observed_aba"] = continuous & group["history_break"].shift().eq(False) & g["pair_set"].eq(two_back) & g["pair_set"].ne(prev["pair_set"])
    # Break on day/series changes, incomplete observations, or a long gap.
    segment = g["history_break"].cumsum()
    g["history_start"] = g.groupby(segment)[KEYS[-1]].transform("first")
    g["last_change"] = g[KEYS[-1]].where(g["spread_changed"]).groupby(segment).ffill()
    g["change_age_min"] = (g[KEYS[-1]] - g["last_change"]).dt.total_seconds() / 60
    g["change_age_unknown"] = g["last_change"].isna()
    g["changes_30m"] = 0
    for _, idx in g.groupby(segment).groups.items():
        times = g.loc[idx, KEYS[-1]].array.as_unit("ns").asi8
        counts = np.r_[0, g.loc[idx, "spread_changed"].to_numpy().cumsum()]
        left = np.searchsorted(times, times - LOOKBACK_MIN * 60 * 10**9, side="right")
        g.loc[idx, "changes_30m"] = counts[1:] - counts[left]
    return {"raw": q, "events": g, "unkeyed": int((~valid).sum())}


def asof_features(events, times, age_min=DEFAULT_AGE_MIN):
    """One bond/side. Same-day as-of, dealer-capped comparisons; no future peers."""
    requested = pd.DatetimeIndex(times).sort_values().unique()
    # Fixed-horizon changes cannot depend on how many query rows the caller asks for.
    lagged = requested - pd.Timedelta(minutes=LOOKBACK_MIN)
    times = requested.union(lagged)
    base = pd.DataFrame({"time": times})
    parts = []
    for firm, history in events.groupby("firm", observed=True):
        merged = pd.merge_asof(base, history.sort_values(KEYS[-1]), left_on="time", right_on=KEYS[-1], direction="backward")
        merged = merged.loc[merged[KEYS[-1]].notna() & merged["time"].dt.normalize().eq(merged["day"])].copy()
        if merged.empty:
            continue
        merged[["complete", "change_age_unknown"]] = merged[["complete", "change_age_unknown"]].astype(bool)
        merged["message_age_min"] = (merged["time"] - merged[KEYS[-1]]).dt.total_seconds() / 60
        merged["spread_set_change_age_min"] = (merged["time"] - merged["last_change"]).dt.total_seconds() / 60
        merged["observed_history_min"] = (merged["time"] - merged["history_start"]).dt.total_seconds() / 60
        parts.append(merged)
    slots = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    columns = ["center_equal", "center_decay", "center_max_age", "center_candidate_clip", "center_dealer_downweight",
               "n_dealers", "n_fresh_dealers", "n_incomplete", "dispersion_bps", "mean_candidate_gap",
               "multi_fraction", "zero_quantity_fraction", "unknown_quantity_fraction", "median_message_age_min", "median_change_age_min", "unknown_change_age_fraction",
               "max_decay_weight_share", "n_peer_supported", "n_clipped_dealers"]
    features = pd.DataFrame(index=times, columns=columns, dtype=float)
    features.index.name = "time"
    features[["n_dealers", "n_fresh_dealers", "n_incomplete", "n_peer_supported", "n_clipped_dealers"]] = 0
    if slots.empty:
        features["center_delta_30m"], features["composition_changed_30m"] = np.nan, np.nan
        return slots, features.reindex(requested)
    composition = {}
    slots["peer_center"], slots["peer_radius"], slots["peer_residual"] = np.nan, np.nan, np.nan
    slots["n_peers"] = 0
    slots["clipped_center"], slots["dealer_weight"] = slots["center"], 1.0
    for t, idx in slots.groupby("time", sort=False).groups.items():
        block = slots.loc[idx]
        valid = block.loc[block["complete"]]
        fresh = valid.loc[valid["message_age_min"].le(age_min)]
        for i, row in valid.iterrows():
            peers = fresh.loc[fresh["firm"].ne(row["firm"]), "center"]
            slots.loc[i, "n_peers"] = len(peers)
            if len(peers) < MIN_PEERS:
                continue
            ref = peers.median()
            radius = max(CLIP_FLOOR_BPS, MAD_MULTIPLIER * 1.4826 * (peers - ref).abs().median())
            residual = row["center"] - ref
            clipped = np.median(np.clip(row["spread_set"], ref - radius, ref + radius))
            slots.loc[i, ["peer_center", "peer_radius", "peer_residual", "clipped_center", "dealer_weight"]] = [
                ref, radius, residual, clipped, min(1.0, radius / abs(residual)) if residual else 1.0,
            ]
        features.loc[t, ["n_dealers", "n_fresh_dealers", "n_incomplete"]] = [len(valid), len(fresh), int(block["complete"].eq(False).sum())]
        if valid.empty:
            continue
        v = slots.loc[valid.index]
        composition[t] = tuple(sorted((str(r.firm), r.quantity_set, r.candidate_count) for r in v.itertuples()))
        # Relative ages preserve normalized weights and avoid all-weight underflow.
        weights = np.exp2(-(v["message_age_min"] - v["message_age_min"].min()) / age_min)
        supported = v["n_peers"].ge(MIN_PEERS)
        changed = supported & ~np.isclose(v["clipped_center"], v["center"])
        features.loc[t] = [v["center"].mean(), np.average(v["center"], weights=weights),
            fresh["center"].mean(), v["clipped_center"].mean(), np.average(v["center"], weights=v["dealer_weight"]),
            len(v), len(fresh), int(block["complete"].eq(False).sum()), v["center"].std(ddof=0), v["gap"].mean(),
            v["candidate_count"].gt(1).mean(), v["quantity_set"].map(lambda tags: "Zero" in tags).mean(),
            v["quantity_set"].map(lambda tags: any(tag == "Missing" or tag.startswith("Other:") for tag in tags)).mean(),
            v["message_age_min"].median(), v["spread_set_change_age_min"].dropna().median() if v["spread_set_change_age_min"].notna().any() else np.nan,
            v["change_age_unknown"].mean(), weights.max() / weights.sum(), int(supported.sum()), int(changed.sum())]
    now = features.reindex(requested).copy()
    before = features.reindex(lagged).set_axis(requested)
    eligible = now["n_dealers"].gt(0) & before["n_dealers"].gt(0) & (requested.normalize() == lagged.normalize())
    changed = pd.Series([composition.get(t) != composition.get(old) for t, old in zip(requested, lagged)], index=requested)
    now["composition_changed_30m"] = changed.astype(float).where(eligible)
    now["center_delta_30m"] = (now["center_equal"] - before["center_equal"]).where(eligible & ~changed)
    return slots.loc[slots["time"].isin(requested)].reset_index(drop=True), now


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


def research_figure(result, dealer, bond, side, day, view, qtag, age_min):
    """One decision per screenshot; always retain the full observed spread range."""
    raw = result["raw"]
    mask = raw["cusip"].eq(bond) & raw["side"].eq(side) & raw[KEYS[-1]].dt.normalize().eq(day)
    peers_raw = raw.loc[mask]
    rows = peers_raw.loc[peers_raw["firm"].eq(dealer)]
    g = result["events"]
    e = g.loc[g["firm"].eq(dealer) & g["cusip"].eq(bond) & g["side"].eq(side) & g["day"].eq(day)]
    slots, f = result["slots"], result["features"]
    own = slots.loc[slots["firm"].eq(dealer)] if len(slots) else slots
    fig = Figure(figsize=(16, 9.5), facecolor="white")
    grid = fig.add_gridspec(2, 2, left=0.08, right=0.97, top=0.77, bottom=0.27, hspace=0.65, wspace=0.24)
    a, b, c = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    fig.suptitle(f"{result['issuer']} | {bond} | {side}\n{view} | dealer {dealer} | {day:%Y-%m-%d} ET", fontsize=16, y=0.98)
    fig.text(0.5, 0.86, f"{len(rows):,} raw rows | {len(e):,} exact-time events | {int(e['candidate_count'].gt(1).sum()):,} multi-spread | {int(e['repeats'].sum()):,} extra identical rows | {int(e['bad'].sum()):,} nonfinite spreads", ha="center", fontsize=11)
    time_axes = [a, b, c]
    if view == VIEWS[0]:
        a.scatter(rows[KEYS[-1]], rows["s"], s=15, alpha=0.45, color="#277F8E", label="All raw candidates")
        a.scatter(e[KEYS[-1]], e["center"], s=17, color="#C9563D", marker="x", label="One scalar: median of distinct spreads")
        a.set_title("Does a single scalar hide multiple contemporaneous levels?")
        b.scatter(e[KEYS[-1]], e["gap"], s=18, color="#8560A5")
        b.set_title("Same-time candidate gap"); b.set_ylabel("max - min (bps)")
        c.scatter(e[KEYS[-1]], e["center_delta"], s=24, color="#999999", label="All adjacent center changes")
        c.scatter(e[KEYS[-1]], e["guarded_delta"], s=20, color="#277F8E", marker="x", label="Same quantity set + candidate count")
        c.set_title("Which summary changes survive the condition check?"); c.set_ylabel("bps / adjacent event")
        transitions = int((~e["history_break"]).sum())
        hidden = int(e["condition_changed"].sum())
        notes = [f"OBSERVED: {hidden}/{transitions} comparable-history transitions change quantity support or candidate count; {int(e['observed_aba'].sum())} A/B/A returns seen by the third event.",
                 "RULE TO TEST: retain median + gap + count; withhold condition-switch momentum. Equal support still does not establish quote identity.",
                 "FEATURE USE: candidate ambiguity and guarded summary changes. Decide usefulness on later dates, not by curve smoothness."]
    elif view == VIEWS[1]:
        selected = rows.loc[rows["qtag"].eq(qtag)]
        a.scatter(rows[KEYS[-1]], rows["s"], s=15, color="#BBBBBB", alpha=0.5, label="All quantities")
        a.scatter(selected[KEYS[-1]], selected["s"], s=22, color="#277F8E", label=f"Selected: {qtag}")
        a.set_title("Does the selected raw quantity isolate one spread at each timestamp?")
        cells, contrasts = quantity_evidence(rows)
        selected_cells = cells.loc[cells["qtag"].eq(qtag)]
        b.scatter(selected_cells[KEYS[-1]], selected_cells["count"], s=22, color="#8560A5")
        b.set_title("Distinct spreads at the selected quantity"); b.set_ylabel("count")
        b.set_ylim(bottom=0)
        b.yaxis.set_major_locator(MaxNLocator(integer=True))
        selected_contrasts = contrasts.loc[contrasts["qtag"].eq(qtag)]
        c.scatter(selected_contrasts[KEYS[-1]], selected_contrasts["within_event_residual"], s=24, color="#277F8E")
        c.axhline(0, color="#999999", lw=0.7)
        c.set_title("Within-event quantity contrast (eligible events only)"); c.set_ylabel("spread - event median (bps)")
        if selected_contrasts.empty:
            c.text(0.5, 0.65, "No eligible contrasts for this quantity", transform=c.transAxes, ha="center")
        ambiguous = int(selected_cells["count"].gt(1).sum())
        notes = [f"OBSERVED: {ambiguous}/{len(selected_cells)} finite selected-quantity events still have multiple spreads; {len(selected_contrasts)} eligible same-time contrasts.",
                 "ELIGIBLE: complete, all-positive events with >=2 quantities and one spread per quantity. Size units / execution remain unknown.",
                 "RULE / FEATURE: test quantity-conditioned summaries only with repeated coverage; otherwise use pooled summaries + quantity state."]
    elif view == VIEWS[2]:
        for column, label, color in [("center_equal", "No expiry / dealer equal", "#555555"),
                                     ("center_decay", f"Message-age half-life {age_min}m", "#277F8E"),
                                     ("center_max_age", f"Message age <= {age_min}m", "#C9563D")]:
            a.plot(f.index, f[column], label=label, color=color, lw=1.4, drawstyle="steps-post")
        a.set_title("Would age treatment materially change the same-side aggregate?")
        if len(own):
            b.plot(own["time"], own["message_age_min"], color="#277F8E", label="Message age")
            b.plot(own["time"], own["spread_set_change_age_min"], color="#8560A5", label="Observed set-change age (known only)")
        b.set_title("Selected dealer: refresh is not necessarily repricing"); b.set_ylabel("minutes")
        max_age = own[["message_age_min", "spread_set_change_age_min"]].max().max() if len(own) else 0
        b.set_ylim(0, max(1, max_age * 1.05) if pd.notna(max_age) else 1)
        c.plot(f.index, f["n_dealers"], color="#555555", label="Complete latest observations")
        c.plot(f.index, f["n_fresh_dealers"], color="#C9563D", label=f"Within {age_min}m")
        c.set_title("Coverage cost of the age rule"); c.set_ylabel("dealers")
        c.yaxis.set_major_locator(MaxNLocator(integer=True))
        comparable = f["center_equal"].notna()
        lost = int((comparable & f["center_max_age"].isna()).sum())
        unknown = int(own["change_age_unknown"].sum()) if len(own) else 0
        notes = [f"OBSERVED: max-age rule leaves {lost}/{int(comparable.sum())} otherwise-covered grid times without a level; selected dealer change age unknown at {unknown}/{len(own)} slots.",
                 f"RULES TO TEST: no expiry, half-life={age_min}m, max-age={age_min}m. These illustrative settings are not calibrated cleaning thresholds.",
                 "FEATURE USE: retain both ages + coverage. First observations / gaps do not initialize an unknown change age to zero."]
    else:
        a.scatter(peers_raw.loc[peers_raw["firm"].ne(dealer), KEYS[-1]], peers_raw.loc[peers_raw["firm"].ne(dealer), "s"], s=9, color="#BBBBBB", alpha=0.3, label="Other dealers: raw")
        a.scatter(rows[KEYS[-1]], rows["s"], s=15, color="#277F8E", alpha=0.5, label="Selected dealer: raw")
        for column, label, color in [("center_equal", "Dealer equal", "#333333"), ("center_candidate_clip", "Candidate clipping", "#C9563D"), ("center_dealer_downweight", "Dealer downweight", "#8560A5")]:
            a.plot(f.index, f[column], label=label, color=color, lw=1.3, drawstyle="steps-post")
        a.set_title("Full raw range and counterfactual aggregates — no deletion")
        b.plot(f.index, f["center_candidate_clip"] - f["center_equal"], color="#C9563D", label="Candidate clip - baseline")
        b.plot(f.index, f["center_dealer_downweight"] - f["center_equal"], color="#8560A5", label="Dealer downweight - baseline")
        b.set_title("How much does the proposed rule change the level?"); b.set_ylabel("bps")
        if len(own):
            c.scatter(own["time"], own["peer_residual"], s=18, color="#277F8E", label="Selected center - other-dealer median")
            c.plot(own["time"], own["peer_radius"], color="#C9563D", ls="--", lw=1, label="Illustrative clipping threshold")
            c.plot(own["time"], -own["peer_radius"], color="#C9563D", ls="--", lw=1)
        c.set_title("Leave-one-dealer-out reference; unsupported = blank"); c.set_ylabel("bps")
        supported = int(own["n_peers"].ge(MIN_PEERS).sum()) if len(own) else 0
        if not supported:
            c.text(0.5, 0.5, f"Need >= {MIN_PEERS} fresh other dealers\nNo residual can be assessed", transform=c.transAxes, ha="center", va="center")
            c.set_yticks([])
        affected = int(f["n_clipped_dealers"].gt(0).sum())
        notes = [f"OBSERVED: selected dealer has >= {MIN_PEERS} eligible peers at {supported}/{len(own)} slots; clipping changes an aggregate contributor at {affected}/{len(f)} grid times.",
                 f"RULE: other peers <= {age_min}m; radius=max({CLIP_FLOOR_BPS:g}bps, {MAD_MULTIPLIER:g} x 1.4826 x peer MAD). Insufficient peers => keep baseline.",
                 "FEATURE USE: retain level, age, ambiguity and supported residuals. Reference is unconditional on unknown size; thresholds are not uncertainty bands or truth."]
    a.set_ylabel("Benchmark spread (bps)")
    if len(rows):
        left, right = rows[KEYS[-1]].min(), rows[KEYS[-1]].max()
        if len(f) and view in VIEWS[2:]:
            left, right = min(left, f.index.min()), max(right, f.index.max())
        pad = max((right - left) / 50, pd.Timedelta(minutes=1))
        for ax in time_axes:
            ax.set_xlim(max(day, left - pad), min(day + pd.DateOffset(days=1), right + pad))
    for ax in time_axes:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6, tz=day.tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=day.tz))
        ax.set_xlabel("ET")
        ax.grid(alpha=0.15)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=8, loc="best", ncol=min(3, len(handles)))
    for y, note in zip([0.18, 0.13, 0.08], notes):
        fig.text(0.08, y, note, fontsize=10, wrap=True)
    fig.text(0.08, 0.025, "Research comparisons only. Raw candidates retained; no identity reconstruction, crossing cleanup, future-confirmed flags or fitted quality score.", fontsize=9, color="#666666")
    return fig

# %% [markdown]
# ## Controls and outputs
# Representative cases lead the issuer list; all issuers remain selectable.
# Default dealer/bond/day order favours multi-spread support, then event count.
# Switching question preserves the case. Quantity can be selected from every observed raw condition.
# The Age setting is both the decay half-life and the max-age hypothesis; for Influence it limits peer age.
# `step3_result['events']`: event-time median, gap, condition flags, guarded delta and observed changes in (t-30m, t].
# `step3_features`: same-day as-of levels, coverage, ages, ambiguity and guarded fixed-30min aggregate changes.
# The fixed-horizon change is missing when dealer roster / quantity support / candidate count differs at the endpoints.
# Unknown change age stays NaN. Absent or incomplete current quotes do not fall back to a previous valid quote.
# Full-day plots / case rankings are retrospective displays; neither is a feature or a claim of predictive success.

# %% 3. Choose a case; compare rules; save one PNG
if "step3_controls" in globals():
    for control in step3_controls:
        control.unobserve(refresh_step3, names="value")
    step3_save.on_click(save_step3, remove=True)
    step3_dashboard.close()
step3_issuer = widgets.Dropdown(options=issuer_choices(), description="Issuer:", layout=widgets.Layout(width="850px"))
step3_view = widgets.ToggleButtons(options=VIEWS, description="Question:")
step3_dealer = widgets.Dropdown(description="Dealer:", layout=widgets.Layout(width="340px"))
step3_bond, step3_side, step3_day = widgets.Dropdown(description="Bond:"), widgets.Dropdown(description="Side:"), widgets.Dropdown(description="ET day:")
step3_quantity = widgets.Dropdown(description="Quantity:", layout=widgets.Layout(width="340px"))
step3_age = widgets.Dropdown(options=[10, 30, 60], value=DEFAULT_AGE_MIN, description="Age (min):")
step3_save = widgets.Button(description="Save PNG", icon="download")
step3_status = widgets.HTML()
step3_image = widgets.Image(format="png", layout=widgets.Layout(width="100%", max_width="1500px"))
step3_result, step3_features, step3_figure, step3_busy, step3_cache_key = None, None, None, False, None


def refresh_step3(change=None):
    global step3_result, step3_features, step3_figure, step3_busy, step3_cache_key
    if step3_busy:
        return
    step3_busy = True
    try:
        step3_status.value = "Building the selected research case..."
        if step3_result is None or step3_result["issuer"] != step3_issuer.value:
            raw = bcq_df.loc[issuer_labels.eq(step3_issuer.value)].copy()
            step3_result = event_history(raw)
            step3_result["issuer"] = step3_issuer.value
            step3_cache_key = None
        eligible = step3_result["events"].copy()
        eligible["multi"] = eligible["candidate_count"].gt(1)
        reset = change is not None and change["owner"] is step3_issuer
        for box, column in [(step3_dealer, "firm"), (step3_bond, "cusip"), (step3_side, "side"), (step3_day, "day")]:
            ranked = eligible.groupby(column, observed=True)["multi"].agg(["sum", "size"]).sort_values(["sum", "size"], ascending=False, kind="stable")
            old = box.value
            options = list(ranked.index)
            box.options = [(f"{v:%Y-%m-%d}" if column == "day" else str(v), v) for v in options]
            box.value = old if not reset and old in options else (options[0] if options else None)
            reset |= change is not None and change["owner"] is box
            eligible = eligible.loc[eligible[column].eq(box.value)]
        step3_quantity.layout.display = "" if step3_view.value == VIEWS[1] else "none"
        step3_age.layout.display = "" if step3_view.value in VIEWS[2:] else "none"
        if eligible.empty:
            step3_features = pd.DataFrame()
            step3_figure = Figure(figsize=(12, 5), facecolor="white")
            ax = step3_figure.subplots(); ax.axis("off")
            ax.text(0.5, 0.5, f"No keyed events for this selection.\nUnkeyed rows: {step3_result['unkeyed']:,}", ha="center", va="center")
        else:
            raw = step3_result["raw"]
            case = raw.loc[raw["firm"].eq(step3_dealer.value) & raw["cusip"].eq(step3_bond.value) & raw["side"].eq(step3_side.value) & raw[KEYS[-1]].dt.normalize().eq(step3_day.value)]
            tags = case["qtag"].value_counts().index.tolist()
            positive_tags = [t for t in tags if t.startswith("q=")]
            options = positive_tags + [t for t in tags if t not in positive_tags]
            old = step3_quantity.value
            step3_quantity.options = options
            step3_quantity.value = old if old in options and not reset else options[0]
            key = (step3_issuer.value, step3_bond.value, step3_side.value, step3_day.value, step3_age.value)
            if key != step3_cache_key:
                e = step3_result["events"]
                history = e.loc[e["cusip"].eq(step3_bond.value) & e["side"].eq(step3_side.value)]
                today = history.loc[history["day"].eq(step3_day.value), KEYS[-1]]
                times = pd.date_range(today.min().ceil(GRID), today.max().floor(GRID), freq=GRID).union(pd.DatetimeIndex([today.min(), today.max()]))
                slots, features = asof_features(history, times, step3_age.value)
                step3_result.update(slots=slots, features=features)
                step3_cache_key = key
            step3_features = step3_result["features"].assign(cusip=step3_bond.value, side=step3_side.value)
            step3_figure = research_figure(step3_result, step3_dealer.value, step3_bond.value, step3_side.value, step3_day.value, step3_view.value, step3_quantity.value, step3_age.value)
        with BytesIO() as buffer:
            step3_figure.savefig(buffer, format="png", dpi=110, facecolor="white")
            step3_image.value = buffer.getvalue()
        step3_status.value = "Raw records retained. Features and event history are in step3_features / step3_result; no tables printed."
    finally:
        step3_busy = False


def save_step3(change=None):
    folder = Path("outputs/quote_quality_step3")
    folder.mkdir(parents=True, exist_ok=True)
    name = "_".join(str(v) for v in [step3_issuer.value, step3_bond.value, step3_dealer.value, step3_side.value, step3_day.value, step3_view.value, step3_quantity.value, step3_age.value])
    name = "".join(c if c.isalnum() else "_" for c in name)[:220]
    path = folder / f"{name}.png"
    step3_figure.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    step3_status.value = f"Saved: {path}"

step3_controls = [step3_issuer, step3_view, step3_dealer, step3_bond, step3_side, step3_day, step3_quantity, step3_age]
for box in step3_controls:
    box.observe(refresh_step3, names="value")
step3_save.on_click(save_step3)
step3_dashboard = widgets.VBox([step3_issuer, step3_view, widgets.HBox([step3_dealer, step3_bond, step3_side]), widgets.HBox([step3_day, step3_quantity, step3_age, step3_save]), step3_status, step3_image])
display(step3_dashboard)
refresh_step3()
