# %% [markdown]
# # Step 2 — repeated content and same-timestamp multi-spread quotes
# Run the three code cells, then choose issuer and Overview / Case.
# Each group is dealer × bond × side × exact event timestamp. Raw records stay
# intact; no latest/median selection, expiry, crossing removal or smoothing.

# %% 1. Load data — same paths and cache logic as step 1
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import ipywidgets as widgets
from IPython.display import display, clear_output

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
# ## What the figures mean
# A multi-spread group contains >=2 distinct finite spreads. Exact repeated
# content is counted across all loaded columns, without removing rows.
# Groups containing a nonfinite/unparseable spread are incomplete, even if they
# also contain two finite candidates. Quantity categories are mutually exclusive:
# Missing/other first, then contains zero, then different/same positive values.
# Zero quantity is not zero spread. Positive quantity stays in original units.

# %% 2. Analyze one issuer and define the overview and case figures
KEYS = ["firm", "cusip", "side", "quote_timestamp_ET"]
QKINDS = ["Same positive", "Different positive", "Contains zero", "Missing / other"]
QCOLORS = ["#287D8E", "#7656A8", "#D98B20", "#A25367"]
DEALERS_PER_PAGE = 8
REP_MIN_GROUPS = 100
REP_MIN_DAYS = 2
REP_MIN_MULTI = 5
REP_PER_THEME = 2
REP_MAX_CONTROL_RATE = 0.01
issuer_labels = bcq_df["ISSUER"].astype("string").fillna("[Missing issuer]")

def group_quotes(raw, keys=KEYS, count_repeats=True):
    q = raw[keys + ["spread", "quantity"]].copy()
    valid_key = q[keys].notna().all(axis=1)
    for column in keys[:-1]:
        valid_key &= q[column].astype("string").str.strip().ne("").fillna(False)
    spread = pd.to_numeric(q["spread"], errors="coerce").to_numpy(float, na_value=np.nan)
    quantity = pd.to_numeric(q["quantity"], errors="coerce").to_numpy(float, na_value=np.nan)
    q["s"] = np.where(np.isfinite(spread), spread, np.nan)
    q["q"] = quantity
    q["qkind"] = np.select(
        [q["quantity"].isna(), np.isfinite(quantity) & (quantity == 0),
         np.isfinite(quantity) & (quantity > 0)],
        ["Missing", "Zero", "Positive"], default="Other",
    )
    q["positive_q"] = q["q"].where(q["qkind"].eq("Positive"))
    q["zero"] = q["qkind"].eq("Zero")
    q["unknown"] = q["qkind"].isin(["Missing", "Other"])
    q["bad_spread"] = q["s"].isna()
    # Ranking needs only narrow group statistics; exact content repeats are
    # checked across every loaded field only for the selected issuer.
    q["duplicate"] = raw.duplicated().to_numpy() if count_repeats else False
    groups = q.loc[valid_key].groupby(keys, sort=False, observed=True).agg(
        rows=("s", "size"), repeats=("duplicate", "sum"),
        n_spreads=("s", "nunique"), lo=("s", "min"), hi=("s", "max"),
        n_quantity=("positive_q", "nunique"), has_zero=("zero", "any"),
        all_zero=("zero", "all"), has_unknown=("unknown", "any"),
        bad_spreads=("bad_spread", "sum"),
    ).reset_index()
    groups["range_bps"] = groups["hi"] - groups["lo"]
    groups["multi"] = groups["n_spreads"].ge(2)
    groups["qclass"] = np.select(
        [groups["has_unknown"], groups["has_zero"], groups["n_quantity"].gt(1)],
        [QKINDS[3], QKINDS[2], QKINDS[1]], default=QKINDS[0],
    )
    groups["day"] = groups["quote_timestamp_ET"].dt.normalize()
    return {"quotes": q.loc[valid_key] if count_repeats else None, "groups": groups,
            "unkeyed_rows": int((~valid_key).sum()), "repeated_rows": int(q["duplicate"].sum()),
            "bad_spread_rows": int(q["bad_spread"].sum())}

def analyze_issuer(issuer):
    raw = bcq_df.loc[issuer_labels.eq(issuer)].copy()
    result = group_quotes(raw)
    groups = result["groups"]
    daily = groups.groupby(KEYS[:3] + ["day"], observed=True)["multi"].agg(["mean", "max"])
    result.update(issuer=issuer, raw=raw, day_balanced_rate=daily["mean"].mean(),
                  affected_day_rate=daily["max"].mean(), day_count=len(daily))
    return result

def representative_issuers():
    # One vectorized pass, not 1,700 full dashboard runs. Discard global quote
    # details after constructing the small issuer summary; keep every issuer.
    narrow = bcq_df[KEYS + ["spread", "quantity"]].assign(issuer=issuer_labels.to_numpy())
    g = group_quotes(narrow, ["issuer"] + KEYS, count_repeats=False)["groups"]
    g["incomplete"] = g["bad_spreads"].gt(0)
    for category, column in zip(QKINDS, ["same_multi", "different_multi", "zero_multi", "unknown_multi"]):
        g[column] = g["multi"] & g["qclass"].eq(category)
    summary = g.groupby("issuer", observed=True).agg(
        groups=("multi", "size"), multi=("multi", "sum"), dealers=("firm", "nunique"),
        days=("day", "nunique"), bonds=("cusip", "nunique"), incomplete=("incomplete", "sum"),
        same_multi=("same_multi", "sum"), different_multi=("different_multi", "sum"),
        zero_multi=("zero_multi", "sum"), unknown_multi=("unknown_multi", "sum"),
    ).reindex(sorted(issuer_labels.unique()), fill_value=0)
    daily = g.groupby(["issuer"] + KEYS[:3] + ["day"], observed=True)["multi"].agg(["mean", "max"])
    summary["balanced_rate"] = daily["mean"].groupby(level="issuer").mean()
    summary["affected_days"] = daily["max"].groupby(level="issuer").mean()
    summary["range_p90"] = g.loc[g["multi"]].groupby("issuer")["range_bps"].quantile(0.9)
    supported = (summary["groups"].ge(REP_MIN_GROUPS) & summary["days"].ge(REP_MIN_DAYS)
                 & summary["dealers"].ge(2))
    fallback = not supported.any()
    pool = summary.loc[supported if not fallback else summary["groups"].gt(0)]
    min_multi = 1 if fallback else REP_MIN_MULTI
    multi_pool = pool.loc[pool["multi"].ge(min_multi)]
    themes = [
        ("Broad multi", multi_pool, "affected_days", False),
        ("Wide multi", multi_pool, "range_p90", False),
    ]
    for tag, column in [("Different quantity", "different_multi"), ("Zero quantity", "zero_multi"),
                        ("Same quantity", "same_multi"), ("Unknown quantity", "unknown_multi")]:
        candidates = multi_pool.loc[multi_pool[column].ge(min_multi)].copy()
        candidates["theme_share"] = candidates[column] / candidates["multi"]
        themes.append((tag, candidates, "theme_share", False))
    themes.append(("Active control", pool.loc[pool["incomplete"].eq(0) & pool["balanced_rate"].le(REP_MAX_CONTROL_RATE)], "balanced_rate", True))
    rankings = [(tag, candidates.sort_values([metric, "groups"], ascending=[ascending, False], kind="stable").index)
                for tag, candidates, metric, ascending in themes]
    promoted = {}
    for _ in range(REP_PER_THEME):  # Interleave themes; avoid duplicate issuers.
        for tag, order in rankings:
            next_issuer = next((issuer for issuer in order if issuer not in promoted), None)
            if next_issuer is not None:
                promoted[next_issuer] = tag + ("; limited sample" if fallback else "")
    order = list(promoted) + [issuer for issuer in summary.index if issuer not in promoted]
    options = [(f"[{promoted[issuer]}] {issuer} | multi={int(summary.loc[issuer, 'multi']):,}/{int(summary.loc[issuer, 'groups']):,}"
                if issuer in promoted else issuer, issuer) for issuer in order]
    summary["front_reason"] = pd.Series(promoted)
    return options, summary

def overview_figure(result, page=1):
    g = result["groups"]
    dealer = g.groupby("firm", observed=True)["multi"].agg(["sum", "count"])
    dealer = dealer.sort_values("count", ascending=False, kind="stable")
    shown = dealer.iloc[(page - 1) * DEALERS_PER_PAGE:page * DEALERS_PER_PAGE]
    by_q = g.groupby("qclass", observed=True)["multi"].agg(["sum", "count"]).reindex(QKINDS, fill_value=0)
    fig, axes = plt.subplots(1, 3, figsize=(17, 7), facecolor="white")
    fig.subplots_adjust(left=0.16, right=0.98, top=0.72, bottom=0.26, wspace=0.52)
    times = result["raw"]["quote_timestamp_ET"].dropna()
    dates = f"{times.min():%Y-%m-%d} to {times.max():%Y-%m-%d} ET" if len(times) else "No valid dates"
    fig.suptitle(f"{result['issuer']}\nStep 2: repeated content or multiple spreads at the same timestamp?", fontsize=17, y=0.99)
    fig.text(0.5, 0.86, f"{dates} | raw rows={len(result['raw']):,} | repeated rows={result['repeated_rows']:,} (all loaded fields)", ha="center", fontsize=11)
    n, m = len(g), int(g["multi"].sum())
    rate = f"{m / n:.1%}" if n else "N/A"
    balanced = f"{result['day_balanced_rate']:.1%}" if result["day_count"] else "N/A"
    affected = f"{result['affected_day_rate']:.1%}" if result["day_count"] else "N/A"
    fig.text(0.5, 0.82, f"Multi-spread={m:,}/{n:,} groups ({rate}) | day-balanced rate={balanced} | affected days={affected} of {result['day_count']:,}", ha="center", fontsize=11)
    fig.text(0.5, 0.78, f"Incomplete spreads: {result['bad_spread_rows']:,} rows; {int(g['bad_spreads'].gt(0).sum()):,} keyed groups | unkeyed rows={result['unkeyed_rows']:,}", ha="center", fontsize=10, color="#8A3B4B")
    a, b, c = axes
    a.barh(np.arange(len(shown)), shown["sum"].div(shown["count"]) * 100, color=QCOLORS[0])
    labels = [f"{str(firm)[:22]}\n{int(row['sum']):,}/{int(row['count']):,} ({row['sum'] / row['count']:.1%})" for firm, row in shown.iterrows()]
    a.set_yticks(np.arange(len(shown)), labels, fontsize=10)
    a.invert_yaxis()
    if shown.empty:
        a.text(0.5, 0.5, "No keyed groups", ha="center", transform=a.transAxes)
    a.set_title(f"Which dealers have multi-spread groups?\nPage {page}; {len(dealer)} dealers", fontsize=11)
    b.barh(np.arange(4), by_q["sum"].div(by_q["count"].replace(0, np.nan)) * 100, color=QCOLORS)
    b.set_yticks(np.arange(4), [f"{k}\n{int(r['sum']):,}/{int(r['count']):,}" + (f" ({r['sum'] / r['count']:.1%})" if r['count'] else " (unobserved)") for k, r in by_q.iterrows()], fontsize=10)
    b.set_ylim(3.6, -0.6)
    b.set_title("Does quantity distinguish the cases?\nAll issuer dealers, independent of page", fontsize=11)
    for axis in [a, b]:
        axis.set_xlim(0, 100)
        axis.xaxis.set_major_formatter(PercentFormatter(100))
        axis.set_xlabel("Multi-spread groups / all groups")
    ranges = g.loc[g["multi"], "range_bps"].sort_values().to_numpy()
    if len(ranges):
        c.step(ranges, np.arange(1, len(ranges) + 1) / len(ranges) * 100, where="post", color=QCOLORS[1])
        c.set_xscale("log")
        c.text(0.98, 0.08, f"N={len(ranges):,}\nMedian={np.median(ranges):g} bps\nMax={ranges[-1]:g} bps", ha="right", transform=c.transAxes, fontsize=10)
    else:
        c.text(0.5, 0.5, "No same-timestamp multi-spread groups", ha="center", wrap=True, transform=c.transAxes)
        c.set_xticks([])
    c.set_title("How far apart are the candidates?\nECDF; all multi-spread groups", fontsize=11)
    c.set_xlabel("Spread range (bps; log scale)")
    c.set_ylim(0, 105)
    c.yaxis.set_major_formatter(PercentFormatter(100))
    fig.text(0.16, 0.10, "Group = dealer / bond / side / exact ET timestamp. All keyed groups are denominators, including incomplete groups.\nDay-balanced = mean of daily multi-group fractions; affected days = any multi group per dealer-bond-side-day.\nQuantity: missing/other takes priority, then contains zero, then different/same positive values. Zero size is not zero spread.", fontsize=10)
    fig.text(0.16, 0.035, "Raw candidates retained. An incomplete group is unassessed beyond its observed finite values; multi-spread does not prove corruption.", fontsize=10, color="#8A3B4B")
    return fig

def case_figure(result, event, minutes=15, value_page=1):
    q, t = result["quotes"], event["quote_timestamp_ET"]
    mask = q["firm"].eq(event["firm"]) & q["cusip"].eq(event["cusip"]) & q["side"].eq(event["side"])
    mask &= q["quote_timestamp_ET"].dt.normalize().eq(t.normalize())
    mask &= q["quote_timestamp_ET"].between(t - pd.Timedelta(minutes=minutes), t + pd.Timedelta(minutes=minutes))
    local = q.loc[mask].sort_values("quote_timestamp_ET", kind="stable")
    selected = local.loc[local["quote_timestamp_ET"].eq(t)]
    fig, (a, b) = plt.subplots(2, 1, sharex=True, figsize=(14, 8), facecolor="white")
    fig.subplots_adjust(left=0.09, right=0.71, top=0.79, bottom=0.17, hspace=0.15)
    fig.suptitle(f"{result['issuer']} | {event['cusip']} | {event['side']}\nDealer {event['firm']} | event {t.isoformat()} ET", fontsize=14, y=0.98)
    zero_kind = "all zero" if event["all_zero"] else "zero / positive mix"
    category = zero_kind if event["qclass"] == "Contains zero" else event["qclass"]
    fig.text(0.09, 0.85, f"Event: {int(event['rows'])} raw rows, {int(event['repeats'])} repeated rows, {int(event['n_spreads'])} finite spreads\nQuantity: {category} | nonfinite spreads={int(event['bad_spreads'])} | local raw rows={len(local):,}", fontsize=11)
    for kind, color, marker in zip(["Positive", "Zero", "Missing", "Other"],
                                   ["#287D8E", "#D98B20", "#A5ADB8", "#A25367"], ["o", "x", "^", "s"]):
        rows = local.loc[local["qkind"].eq(kind)]
        a.scatter(rows["quote_timestamp_ET"], rows["s"], c=color, marker=marker, s=28, label=kind, alpha=0.65)
        if kind in ["Positive", "Zero"]:
            b.scatter(rows["quote_timestamp_ET"], rows["q"], c=color, marker=marker, s=28, alpha=0.65)
    a.scatter(selected["quote_timestamp_ET"], selected["s"], s=120, facecolors="none", edgecolors="#222222", linewidths=1.3)
    for axis in [a, b]:
        axis.axvline(t, color="#555555", ls=":", lw=1)
        axis.grid(alpha=0.15)
        axis.set_xlim(max(t.normalize(), t - pd.Timedelta(minutes=minutes)),
                      min(t.normalize() + pd.DateOffset(days=1), t + pd.Timedelta(minutes=minutes)))
    a.set_ylabel("Benchmark spread (bps)")
    a.legend(loc="lower left", bbox_to_anchor=(1.02, 0.02), ncol=2, fontsize=9)
    b.set_ylabel("Raw quantity\n(unit unconfirmed)")
    b.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=t.tz))
    b.set_xlabel(f"Event time ET; +/-{minutes} min within the same day; no connecting lines")
    # Paginated event values keep exact candidates visible without a large table.
    candidates = selected[["spread", "quantity"]].copy()
    candidates["spread"] = candidates["spread"].astype(str)
    candidates["quantity"] = candidates["quantity"].astype(str)
    values = candidates.groupby(["spread", "quantity"], dropna=False, sort=False).size()
    lines = [f"spread={s}; q={size}; rows={count}" for (s, size), count in values.items()]
    start = (value_page - 1) * 8
    a.text(1.02, 0.98, "Exact event candidates (raw units)\n" + "\n".join(lines[start:start + 8]) +
           f"\nValues {start + 1}-{min(start + 8, len(lines))} of {len(lines)}", transform=a.transAxes, va="top", fontsize=10)
    counts = local["qkind"].value_counts()
    b.text(1.02, 0.98, "Quantity rows in this window\n" + "\n".join(f"{k}: {counts.get(k, 0):,}" for k in ["Positive", "Zero", "Missing", "Other"]) +
           "\nMissing/other: no numeric position", transform=b.transAxes, va="top", fontsize=10)
    explanation = "Different positive quantities accompany these candidates; conditions may differ." if category == "Different positive" else "Quantity does not uniquely distinguish the observed price candidates."
    if not event["multi"]:
        explanation = "No observed multi-spread at this timestamp; incomplete values remain unassessed." if event["bad_spreads"] else "Single-spread control; identical content may still be repeated."
    fig.text(0.09, 0.07, f"Observed: finite spread range={event['range_bps']:g} bps. {explanation}\nUnknown: execution, missing conditions and candidate identity. Missing/other quantity appears by symbol in the spread panel, not as numeric zero.", fontsize=10)
    return fig

# %% [markdown]
# ## Select issuer; inspect cases; save the displayed figure
# Overview includes every keyed group. Case selection offers multi-spread by
# quantity category, complete single-spread controls and incomplete groups.
# Dealer/bond/side/date are selectable. Events paginate 50 at a time;
# every eligible event is accessible and overview statistics are never sampled.
# Candidate values paginate eight per screenshot. All rows remain in step2_result.
# Representative issuers lead the dropdown, interleaving supported multi-spread
# patterns and active controls; the remaining issuers follow alphabetically.

# %% 3. Dropdown controls and PNG export
issuer_options, issuer_summary = representative_issuers()
issuer_box = widgets.Dropdown(options=issuer_options, description="Issuer:", layout=widgets.Layout(width="850px"))
if not issuer_box.options:
    raise ValueError("No quote rows remain in the supplied three-month traded-bond universe.")
view_box = widgets.ToggleButtons(options=["Overview", "Case"], description="View:")
page_box = widgets.Dropdown(options=[1], description="Dealer page:")
case_type = widgets.Dropdown(options=["All multi-spread"] + QKINDS + ["Single-spread control", "Incomplete spread"], description="Case type:", layout=widgets.Layout(width="360px"))
dealer_box = widgets.Dropdown(description="Dealer:", layout=widgets.Layout(width="360px"))
bond_box = widgets.Dropdown(description="Bond:")
side_box = widgets.Dropdown(description="Side:")
date_box = widgets.Dropdown(description="ET date:")
event_box = widgets.Dropdown(description="Event:", layout=widgets.Layout(width="650px"))
event_page = widgets.Dropdown(options=[1], description="Event page:")
window_box = widgets.IntSlider(value=15, min=1, max=60, description="+/- min:", continuous_update=False)
values_page = widgets.Dropdown(options=[1], description="Values page:")
save_button = widgets.Button(description="Save PNG", icon="download")
save_status, plot_output = widgets.HTML(), widgets.Output()
case_controls = widgets.VBox([case_type, widgets.HBox([dealer_box, bond_box, side_box]),
                              widgets.HBox([date_box, window_box, values_page]), widgets.HBox([event_page, event_box])])
step2_result, current_figure, busy = None, None, False

def refresh_step2(change=None):
    global step2_result, current_figure, busy
    if busy:
        return
    busy = True
    try:
        if current_figure is not None:
            plt.close(current_figure)
        if step2_result is None or step2_result["issuer"] != issuer_box.value:
            step2_result = analyze_issuer(issuer_box.value)
            page_box.options = range(1, max(1, (step2_result["groups"]["firm"].nunique() + 7) // 8) + 1)
            page_box.value = 1
        g = step2_result["groups"]
        page_box.layout.display = "" if view_box.value == "Overview" else "none"
        case_controls.layout.display = "none" if view_box.value == "Overview" else ""
        if view_box.value == "Overview":
            current_figure = overview_figure(step2_result, page_box.value)
        else:
            if case_type.value == "Single-spread control":
                eligible = g.loc[g["n_spreads"].eq(1) & g["bad_spreads"].eq(0)]
            elif case_type.value == "Incomplete spread":
                eligible = g.loc[g["bad_spreads"].gt(0)]
            else:
                eligible = g.loc[g["multi"]]
                if case_type.value in QKINDS:
                    eligible = eligible.loc[eligible["qclass"].eq(case_type.value)]
            # Updating upstream controls resets downstream choices, without nested callbacks.
            chain = [(dealer_box, "firm"), (bond_box, "cusip"), (side_box, "side"), (date_box, "day")]
            reset = change is not None and change["owner"] in [issuer_box, case_type]
            for box, column in chain:
                options = sorted(eligible[column].unique())
                previous = box.value
                box.options = [(str(x.date()) if column == "day" else str(x), x) for x in options]
                box.value = previous if not reset and previous in options else (options[0] if options else None)
                reset |= change is not None and change["owner"] is box
                eligible = eligible.loc[eligible[column].eq(box.value)]
            eligible = eligible.sort_values("quote_timestamp_ET", kind="stable")
            prior_page = event_page.value
            event_page.options = range(1, max(1, (len(eligible) + 49) // 50) + 1)
            event_page.value = prior_page if not reset and prior_page in event_page.options else 1
            sample = eligible.iloc[(event_page.value - 1) * 50:event_page.value * 50]
            previous = event_box.value
            event_box.options = [(f"{r['quote_timestamp_ET'].isoformat()} | {r['n_spreads']} spreads | {r['range_bps']:g} bps", i) for i, r in sample.iterrows()]
            event_box.value = previous if not reset and previous in sample.index else (sample.index[0] if len(sample) else None)
            if event_box.value is None:
                current_figure, ax = plt.subplots(figsize=(12, 5))
                ax.axis("off")
                ax.text(0.5, 0.5, f"{issuer_box.value}\nNo groups match: {case_type.value}\nChoose another case type or issuer.", ha="center", va="center", fontsize=14)
            else:
                event = g.loc[event_box.value]
                rows = step2_result["quotes"]
                mask = rows["quote_timestamp_ET"].eq(event["quote_timestamp_ET"])
                for key in KEYS[:3]:
                    mask &= rows[key].eq(event[key])
                n_values = rows.loc[mask, ["spread", "quantity"]].astype(str).drop_duplicates().shape[0]
                prior = values_page.value
                values_page.options = range(1, max(1, (n_values + 7) // 8) + 1)
                values_page.value = prior if prior in values_page.options and change is not None and change["owner"] is values_page else 1
                current_figure = case_figure(step2_result, event, window_box.value, values_page.value)
                current_figure.text(0.09, 0.015, f"Event selector page {event_page.value}/{len(event_page.options)}: {len(sample)} of {len(eligible):,} eligible groups for this dealer/bond/side/day. Every group is accessible.", fontsize=10)
        save_status.value = ""
        with plot_output:
            clear_output(wait=True)
            display(current_figure)
        plt.close(current_figure)
    finally:
        busy = False

def save_step2(change=None):
    folder = Path("outputs/quote_quality_step2")
    folder.mkdir(parents=True, exist_ok=True)
    name = "".join(c if c.isalnum() else "_" for c in str(issuer_box.value))[:60]
    suffix = f"overview_{page_box.value}" if view_box.value == "Overview" else f"case_{event_box.value}_values_{values_page.value}"
    path = folder / f"{name}_{suffix}.png"
    current_figure.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    save_status.value = f"Saved: {path}"

for box in [issuer_box, view_box, page_box, case_type, dealer_box, bond_box, side_box, date_box, event_page, event_box, window_box, values_page]:
    box.observe(refresh_step2, names="value")
save_button.on_click(save_step2)
display(widgets.VBox([issuer_box, widgets.HBox([view_box, page_box, save_button]), case_controls, save_status, plot_output]))
refresh_step2()
