# %% [markdown]
# # Step 2 — repeated content and same-timestamp multi-spread quotes
# Run the three code cells. Overview asks **where same-side multi-spread events occur**;
# Case shows **which spreads and quantities coexist at one timestamp**.
# An event = dealer × bond × side × exact timestamp. Two bids at the same time
# are two candidates, not a time-series jump or a bid/ask crossing.
# Start with Global; scope changes apply once. Freeze 18 cases once for stable navigation.

# %% 1. Load data — same paths and cache logic as step 1
from pathlib import Path
import numpy as np
import pandas as pd
from io import BytesIO
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
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
bcq_df["quote_timestamp_ET"] = pd.to_datetime(
    bcq_df["quote_timestamp_UTC"], utc=True, errors="coerce", format="mixed",
).dt.tz_convert("America/New_York")
from quote_quality_population import (population_tables, scope_selection, scope_options, summary_html,
                                      fixed_case_manifest)
quality_population = population_tables(data_ig, bcq_df)
bcq_df = quality_population["raw"]

# %% [markdown]
# ## Read the figures
# - **Dealer panel:** multi-spread events / all events for that dealer.
# - **Quantity panel:** composition of the issuer's multi-spread events; shares sum to 100%.
#   All-zero and zero/positive mixtures are shown separately. This does not establish size effects.
# - **Gap panel:** distribution of max spread minus min spread **at the same timestamp**.
# - **Case:** full-day raw spread points plus separate candidates at the selected timestamp.
#   Candidate row numbers are display positions, not quote identities tracked through time.
# Each event counts once. Repeated content means extra identical rows across all loaded fields;
# it does not include unchanged quotes at different timestamps. Incomplete spreads stay unassessed.
# Quantity units remain unknown; zero quantity is not zero spread. Raw records are retained.

# %% 2. Analyze one issuer and define the overview and case figures
KEYS = ["firm", "cusip", "side", "quote_timestamp_ET"]
QKINDS = ["Same positive", "Different positive", "Contains zero", "Missing / other"]
QCOLORS = ["#287D8E", "#7656A8", "#D98B20", "#A25367"]
DEALERS_PER_PAGE = 8
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

def analyze_scope(scope="Global", value=None):
    selection = scope_selection(quality_population, scope, value)
    classified, groups = selection["raw"], selection["groups"]
    raw = quality_population["raw"].loc[classified.index]
    q = raw[KEYS + ["spread", "quantity"]].copy()
    q["s"] = pd.to_numeric(q["spread"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    q["q"] = pd.to_numeric(q["quantity"], errors="coerce")
    q["qkind"] = classified["quantity_kind"]
    valid = q[KEYS].notna().all(axis=1)
    for key in KEYS[:-1]:
        valid &= q[key].astype("string").str.strip().ne("").fillna(False)
    summary = selection["summary"]
    return dict(selection, raw=raw, quotes=q.loc[valid], unkeyed_rows=int((~valid).sum()),
                repeated_rows=int(raw.duplicated().sum()),
                bad_spread_rows=int(q["s"].isna().sum()),
                day_balanced_rate=summary.unit_equal_multi_rate,
                affected_day_rate=summary.affected_unit_rate, day_count=int(summary.dealer_bond_side_days))

def analyze_issuer(issuer):
    return analyze_scope("Issuer", issuer)

def representative_issuers():
    # All traded issuers, including quote-absent ones; fixed cases are separate.
    summary = quality_population["tables"]["Issuer"].copy()
    summary["groups"], summary["multi"] = summary.events, summary.multi_events
    summary["balanced_rate"] = summary.unit_equal_multi_rate
    summary["affected_days"] = summary.affected_unit_rate
    summary["front_reason"] = ""
    return [(str(name), name) for name in summary.index], summary


def overview_figure(result, page=1):
    g = result["groups"]
    multi = g.loc[g["multi"]]
    n, m = len(g), len(multi)
    dealer = g.groupby("firm", observed=True)["multi"].agg(["sum", "count"])
    dealer = dealer.sort_values(["sum", "count"], ascending=False, kind="stable")
    shown = dealer.iloc[(page - 1) * DEALERS_PER_PAGE:page * DEALERS_PER_PAGE]
    # Composition among multi-spread events, not the old within-category rates.
    qlabels = ["Same positive", "Different positive", "All zero", "Zero + positive", "Missing / other"]
    qtypes = multi["qclass"].where(~multi["has_zero"], "Zero + positive")
    qtypes = qtypes.mask(multi["all_zero"], "All zero").mask(multi["has_unknown"], "Missing / other")
    qcounts = qtypes.value_counts().reindex(qlabels, fill_value=0)
    fig = Figure(figsize=(17, 7.5), facecolor="white")
    a, b, c = fig.subplots(1, 3)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.62, bottom=0.25, wspace=0.62)
    times = result["raw"]["quote_timestamp_ET"].dropna()
    dates = f"{times.min():%Y-%m-%d} to {times.max():%Y-%m-%d} ET" if len(times) else "No valid dates"
    fig.suptitle(f"{result['issuer']}\nStep 2 | Multiple spreads on the same side, at the same time", fontsize=17, y=0.98)
    rate = f"{m / n:.2%}" if n else "N/A"
    fig.text(0.5, 0.84, f"{m:,} of {n:,} events ({rate}) have multiple spreads", ha="center", fontsize=15, weight="bold")
    fig.text(0.5, 0.79, f"{dates} | {len(result['raw']):,} raw rows | {result['repeated_rows']:,} extra identical rows", ha="center", fontsize=11)
    summary = result["summary"]
    fig.text(0.5, 0.74, f"Quotes: {int(summary.quoted_bonds):,}/{int(summary.traded_bonds):,} traded bonds | no quote={int(summary.no_quote_bonds):,} | affected bond-days={int(summary.affected_bond_days):,}/{int(summary.observed_bond_days):,}", ha="center", fontsize=10)
    a.barh(np.arange(len(shown)), shown["sum"].div(shown["count"]) * 100, color=QCOLORS[0])
    labels = [f"{firm}\n{int(r['sum']):,}/{int(r['count']):,} ({r['sum'] / r['count']:.2%})" for firm, r in shown.iterrows()]
    a.set_yticks(np.arange(len(shown)), labels, fontsize=9)
    a.invert_yaxis()
    a.set_title(f"1. Which dealers?\nMost multi-spread events first | page {page}", fontsize=12)
    a.set_xlabel("Multi-spread / this dealer's events")
    if shown.empty:
        a.text(0.5, 0.5, "No keyed events", ha="center", transform=a.transAxes)
    b.barh(np.arange(5), qcounts / m * 100 if m else np.zeros(5),
           color=[QCOLORS[0], QCOLORS[1], QCOLORS[2], "#E9BA70", QCOLORS[3]])
    b.set_yticks(np.arange(5), [f"{k}\n{v:,}/{m:,} ({v / m:.1%})" if m else f"{k}\nN/A: no multi-spread events" for k, v in qcounts.items()], fontsize=9)
    b.set_ylim(4.6, -0.6)
    b.set_title("2. What quantities accompany them?\nComposition of all scope multi-spread events", fontsize=12)
    b.set_xlabel("Share of scope multi-spread events")
    for axis in [a, b]:
        axis.set_xlim(0, 100)
        axis.xaxis.set_major_formatter(PercentFormatter(100))
        axis.grid(axis="x", alpha=0.15)
        axis.set_axisbelow(True)
    ranges = multi["range_bps"].sort_values().to_numpy()
    if m:
        c.step(ranges, np.arange(1, m + 1) / m * 100, where="post", color=QCOLORS[1])
        c.set_xscale("log")
        c.text(0.98, 0.06, f"{m:,} events\nMedian gap = {np.median(ranges):g} bps\nMax gap = {ranges[-1]:g} bps", ha="right", transform=c.transAxes, fontsize=10)
    else:
        c.text(0.5, 0.5, "No observed multi-spread events", ha="center", wrap=True, transform=c.transAxes)
        c.set_xticks([])
    c.set_title("3. How far apart at that instant?\nCumulative distribution of candidate gaps", fontsize=12)
    c.set_xlabel("Same-time max - min spread (bps; log)")
    c.set_ylabel("Events with a gap at or below x")
    c.set_ylim(0, 105)
    c.yaxis.set_major_formatter(PercentFormatter(100))
    incomplete = int(g["bad_spreads"].gt(0).sum())
    fig.text(0.14, 0.15, "Event = same dealer / bond / side / exact timestamp. Each event counts once.\nMiddle panel: all multi-spread events are the denominator. Right panel: candidate disagreement, not a move over time.", fontsize=11)
    fig.text(0.14, 0.07, f"Incomplete spreads: {result['bad_spread_rows']:,} rows in {incomplete:,} keyed groups; unkeyed rows={result['unkeyed_rows']:,}.\nIncomplete values remain unassessed. Multiple spreads, zero quantity and a large gap do not by themselves prove corruption.", fontsize=10, color="#8A3B4B")
    return fig


def case_figure(result, event, minutes=None, value_page=1):
    q, t = result["quotes"], event["quote_timestamp_ET"]
    mask = q["firm"].eq(event["firm"]) & q["cusip"].eq(event["cusip"]) & q["side"].eq(event["side"])
    mask &= q["quote_timestamp_ET"].dt.normalize().eq(t.normalize())
    day = q.loc[mask].sort_values("quote_timestamp_ET", kind="stable")
    local = day if minutes is None else day.loc[day["quote_timestamp_ET"].between(t - pd.Timedelta(minutes=minutes), t + pd.Timedelta(minutes=minutes))]
    selected = day.loc[day["quote_timestamp_ET"].eq(t)]
    # Separate candidate rows avoid overplotting two zero quantities at one point.
    # Row positions are labels only; neither candidate identities nor size ordering.
    candidates = selected.assign(spread_text=selected["spread"].astype(str), quantity_text=selected["quantity"].astype(str))
    values = candidates.groupby(["spread_text", "quantity_text"], sort=False, dropna=False).agg(
        rows=("s", "size"), s=("s", "first"), qkind=("qkind", "first"),
    ).reset_index()
    start = (value_page - 1) * 8
    shown = values.iloc[start:start + 8]
    fig = Figure(figsize=(16, 8), facecolor="white")
    a, b = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.65, 1]})
    fig.subplots_adjust(left=0.08, right=0.94, top=0.75, bottom=0.29, wspace=0.40)
    fig.suptitle(f"{result['issuer']} | {event['cusip']} | {event['side']}\nDealer {event['firm']} | {t.isoformat()}", fontsize=15, y=0.98)
    gap = f"{event['range_bps']:g} bps" if event["n_spreads"] else "unassessed"
    fig.text(0.5, 0.84, f"Selected instant: {int(event['n_spreads'])} distinct finite spreads | gap = {gap} | {int(event['rows'])} raw rows", ha="center", fontsize=13, weight="bold")
    colors = dict(zip(["Positive", "Zero", "Missing", "Other"], [QCOLORS[0], QCOLORS[2], "#8895A7", QCOLORS[3]]))
    markers = dict(zip(colors, ["o", "x", "^", "s"]))
    for kind in colors:
        rows = local.loc[local["qkind"].eq(kind)]
        if len(rows):
            a.scatter(rows["quote_timestamp_ET"], rows["s"], c=colors[kind], marker=markers[kind], s=28, label=f"Quantity: {kind.lower()}", alpha=0.65)
    a.scatter(selected["quote_timestamp_ET"], selected["s"], s=110, facecolors="none", edgecolors="#222222", linewidths=1.3)
    a.axvline(t, color="#555555", ls=":", lw=1)
    if minutes is None:
        left, right = day["quote_timestamp_ET"].min(), day["quote_timestamp_ET"].max()
        padding = max((right - left) / 30, pd.Timedelta(minutes=1))
        left, right = left - padding, right + padding
    else:
        left, right = t - pd.Timedelta(minutes=minutes), t + pd.Timedelta(minutes=minutes)
    a.set_xlim(max(t.normalize(), left), min(t.normalize() + pd.DateOffset(days=1), right))
    a.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6, tz=t.tz))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=t.tz))
    a.set_ylabel("Benchmark spread (bps)")
    window = "Full day" if minutes is None else f"+/- {minutes} min"
    n_times = local["quote_timestamp_ET"].nunique()
    a.set_title(f"{window}: {len(local):,} raw rows at {n_times:,} timestamps\nCircles identify the selected instant", fontsize=12)
    a.set_xlabel(f"{t:%Y-%m-%d} ET | points at actual timestamps")
    fig.legend(*a.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.08, 0.225), ncol=4, fontsize=9, borderaxespad=0)
    a.grid(alpha=0.15)
    b.set_yticks(np.arange(len(shown)), [f"#{start + i + 1}  q={row.quantity_text}\n{row.rows} raw row(s)" for i, row in enumerate(shown.itertuples())], fontsize=10)
    for i, row in enumerate(shown.itertuples()):
        if pd.notna(row.s):
            b.scatter(row.s, i, color=colors[row.qkind], marker=markers[row.qkind], s=75)
            b.annotate(f"spread={row.spread_text}", (row.s, i), xytext=(6, 9), textcoords="offset points", fontsize=10)
        else:
            b.text(0.5, i, f"spread={row.spread_text} (not finite)", transform=b.get_yaxis_transform(), ha="center", fontsize=10)
    b.set_ylim(len(shown) - 0.3, -0.7)
    b.margins(x=0.35)
    b.set_xlabel("Benchmark spread (bps) | separate event scale")
    b.set_title(f"Candidates at the selected instant only\nValues {start + 1}-{min(start + 8, len(values))} of {len(values)}", fontsize=12)
    b.grid(axis="x", alpha=0.15)
    if event["all_zero"]:
        meaning = "All quantities are zero: size information cannot distinguish these candidates."
    elif event["qclass"] == "Same positive":
        meaning = "The same positive quantity accompanies different spreads; other conditions remain unknown."
    elif event["qclass"] == "Different positive":
        ambiguous_size = selected.groupby("q")["s"].nunique().gt(1).any()
        meaning = ("Different quantities occur, but a quantity still maps to multiple spreads." if ambiguous_size else
                   "Different positive quantities accompany the spreads; a size effect is not yet established.")
    else:
        meaning = "Zero / missing / other quantities leave quote conditions partly unknown."
    if not event["multi"]:
        meaning = "No observed multi-spread here; incomplete spread values remain unassessed." if event["bad_spreads"] else "One observed spread at this timestamp; this does not validate the quote's economic meaning."
    context = ("Only one timestamp in this window: persistence cannot be assessed." if n_times <= 1 else
               "Inspect whether multiple levels recur; the plot does not assign identities across timestamps.")
    fig.text(0.08, 0.135, meaning + "\n" + context, fontsize=11)
    fig.text(0.08, 0.055, f"Same-side candidates coexist; their gap is not a time-series jump or a bid/ask width. Quantity is in raw, unconfirmed units.\nSelected event: {int(event['repeats'])} extra identical rows; {int(event['bad_spreads'])} nonfinite spreads. Row labels are display positions only.", fontsize=10, color="#6A4F39")
    return fig

# %% [markdown]
# ## Select issuer; inspect cases; save the displayed figure
# Overview includes every keyed group. Case selection offers multi-spread by
# quantity category, complete single-spread observations and incomplete groups.
# Dealer/bond/side/date are selectable. Events paginate 50 at a time;
# every eligible event is accessible and overview statistics are never sampled.
# Candidate values paginate eight per screenshot. Full day is the default; switch to a local window to zoom.
# All rows remain in step2_result. Optional counting details are collapsed below the controls.
# Representative issuers lead the dropdown, interleaving supported multi-spread
# patterns and active controls; the remaining issuers follow alphabetically.

# %% 3. Dropdown controls and PNG export
# Detach old observers when rerunning this cell in the same kernel.
if "step2_controls" in globals():
    for control in step2_controls:
        control.unobserve(refresh_step2, names="value")
    save_button.on_click(save_step2, remove=True)
    scope_box.unobserve(scope_changed, names="value")
    issuer_box.unobserve(change_issuer, names="value")
    apply_button.on_click(apply_scope, remove=True)
    freeze_button.on_click(freeze_cases, remove=True)
    apply_case_button.on_click(apply_fixed_case, remove=True)
    step2_dashboard.close()
issuer_options, issuer_summary = representative_issuers()
scope_box = widgets.Dropdown(options=["Global", "SECTOR", "Dealer", "Issuer"], description="Scope:")
value_box = widgets.Dropdown(options=["Global"], description="Group:", layout=widgets.Layout(width="650px"))
apply_button = widgets.Button(description="Apply", button_style="primary")
freeze_button = widgets.Button(description="Freeze 18 cases", icon="thumb-tack")
fixed_case_box = widgets.Dropdown(options=[("Freeze once: random 6 / typical 6 / measured impact 6", None)],
                                 description="Fixed case:", layout=widgets.Layout(width="900px"))
apply_case_button = widgets.Button(description="Open case")
case_manifest_status = widgets.HTML()
case_manifest = quality_population["case_manifest"]
applied_scope, applied_value = "Global", "Global"
issuer_box = widgets.Dropdown(options=issuer_options, description="Issuer:", layout=widgets.Layout(width="850px"))
if not issuer_box.options:
    raise ValueError("No traded bonds remain in the supplied universe.")
view_box = widgets.ToggleButtons(options=["Overview", "Case"], description="View:")
page_box = widgets.Dropdown(options=[1], description="Dealer page:")
case_type = widgets.Dropdown(options=["All multi-spread"] + QKINDS + ["Single-spread observation", "Incomplete spread"], description="Case type:", layout=widgets.Layout(width="360px"))
dealer_box = widgets.Dropdown(description="Dealer:", layout=widgets.Layout(width="360px"))
bond_box = widgets.Dropdown(description="Bond:")
side_box = widgets.Dropdown(description="Side:")
date_box = widgets.Dropdown(description="ET date:")
event_box = widgets.Dropdown(description="Event:", layout=widgets.Layout(width="650px"))
event_page = widgets.Dropdown(options=[1], description="Event page:")
window_mode = widgets.ToggleButtons(options=["Full day", "Local window"], description="Window:")
window_box = widgets.IntSlider(value=15, min=1, max=60, description="+/- min:", continuous_update=False)
values_page = widgets.Dropdown(options=[1], description="Values page:")
save_button = widgets.Button(description="Save PNG", icon="download")
save_status, details = widgets.HTML(), widgets.HTML()
plot_output = widgets.Image(format="png", layout=widgets.Layout(width="100%", max_width="1500px"))
case_controls = widgets.VBox([case_type, widgets.HBox([dealer_box, bond_box, side_box]),
                              widgets.HBox([date_box, values_page]), widgets.HBox([window_mode, window_box]), widgets.HBox([event_page, event_box])])
step2_result, current_figure, busy = None, None, False

def refresh_step2(change=None):
    global step2_result, current_figure, busy
    if busy:
        return
    busy = True
    try:
        if step2_result is None or step2_result["scope"] != applied_scope or step2_result["value"] != applied_value:
            step2_result = analyze_scope(applied_scope, applied_value)
            page_box.options = range(1, max(1, (step2_result["groups"]["firm"].nunique() + 7) // 8) + 1)
            page_box.value = 1
        g = step2_result["groups"]
        window_box.layout.display = "" if window_mode.value == "Local window" else "none"
        page_box.layout.display = "" if view_box.value == "Overview" else "none"
        case_controls.layout.display = "none" if view_box.value == "Overview" else ""
        if view_box.value == "Overview":
            current_figure = overview_figure(step2_result, page_box.value)
        else:
            if case_type.value == "All events":
                eligible = g
            elif case_type.value == "Single-spread observation":
                eligible = g.loc[g["n_spreads"].eq(1) & g["bad_spreads"].eq(0)]
            elif case_type.value == "Incomplete spread":
                eligible = g.loc[g["bad_spreads"].gt(0)]
            else:
                eligible = g.loc[g["multi"]]
                if case_type.value in QKINDS:
                    eligible = eligible.loc[eligible["qclass"].eq(case_type.value)]
            # Updating upstream controls resets downstream choices, without nested callbacks.
            chain = [(dealer_box, "firm"), (bond_box, "cusip"), (side_box, "side"), (date_box, "day")]
            reset = change is not None and change["owner"] in [issuer_box, apply_button, apply_case_button, case_type]
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
                current_figure = Figure(figsize=(12, 5), facecolor="white")
                ax = current_figure.subplots()
                ax.axis("off")
                ax.text(0.5, 0.5, f"{step2_result['issuer']}\nNo groups match: {case_type.value}\nChoose another case type or issuer.", ha="center", va="center", fontsize=14)
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
                current_figure = case_figure(step2_result, event, window_box.value if window_mode.value == "Local window" else None, values_page.value)
                current_figure.text(0.09, 0.015, f"Event page {event_page.value}/{len(event_page.options)} | {len(eligible):,} eligible events for this dealer / bond / side / day | all events accessible", fontsize=10)
        save_status.value = ""
        r = step2_result
        balanced = f"{r['day_balanced_rate']:.2%}" if r["day_count"] else "N/A"
        affected = f"{r['affected_day_rate']:.2%}" if r["day_count"] else "N/A"
        details.value = (
            summary_html(r) + "<details><summary>Counting details and interpretation</summary>"
            f"<p>Day-balanced multi-spread rate: {balanced}. Dealer-bond-side-days with any multi-spread: {affected} "
            f"of {r['day_count']:,}. These are dealer/bond/side/day units, not calendar days.</p>"
            "<p>Day-balanced = mean of each unit's multi-spread fraction. "
            "All keyed events, including incomplete ones, stay in denominators. "
            "Unknown quantities take priority over zero, then different/same positive quantities. "
            "Extra identical rows match all loaded fields; unchanged quotes at later timestamps are separate events.</p>"
            "<p>Fixed cases use seed 2026 and sector/activity strata, with a soft dealer/issuer cap of two. "
            "High impact uses actual center or 30-minute cutoff coverage changes at first/middle/last local queries. "
            "Cases are navigation; full population counts never use this sample. No observed multi-spread does not certify a clean issuer.</p></details>"
        )
        # An unmanaged Figure rendered into ONE image widget avoids both inline
        # auto-display and explicit-display paths emitting the same figure twice.
        with BytesIO() as buffer:
            current_figure.savefig(buffer, format="png", dpi=110, facecolor="white")
            plot_output.value = buffer.getvalue()
    finally:
        busy = False

def save_step2(change=None):
    folder = Path("outputs/quote_quality_step2")
    folder.mkdir(parents=True, exist_ok=True)
    name = "".join(c if c.isalnum() else "_" for c in str(step2_result["issuer"]))[:60]
    suffix = f"overview_{page_box.value}" if view_box.value == "Overview" else f"case_{event_box.value}_values_{values_page.value}_{'day' if window_mode.value == 'Full day' else str(window_box.value) + 'min'}"
    path = folder / f"{name}_{suffix}.png"
    current_figure.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    save_status.value = f"Saved: {path}"

def scope_changed(change=None):
    value_box.options = scope_options(quality_population, scope_box.value)
    value_box.layout.display = "none" if scope_box.value in ["Global", "Issuer"] else ""
    issuer_box.layout.display = "" if scope_box.value == "Issuer" else "none"
    if scope_box.value == "Issuer" and issuer_box.value in value_box.options:
        value_box.value = issuer_box.value

def apply_scope(change=None):
    global applied_scope, applied_value, step2_result
    applied_scope, applied_value = scope_box.value, value_box.value
    step2_result = None
    refresh_step2({"owner": apply_button})

def change_issuer(change=None):
    if busy:
        return
    scope_box.value = "Issuer"
    value_box.value = issuer_box.value
    apply_scope()

def freeze_cases(change=None):
    global case_manifest
    freeze_button.disabled = True
    case_manifest_status.value = "Measuring rule center/coverage changes at first/middle/last local queries per observed bond/side/day..."
    try:
        if quality_population["case_manifest"] is None:
            case_manifest = fixed_case_manifest(quality_population)
        else:
            case_manifest = quality_population["case_manifest"]
        fixed_case_box.options = [(f"{r.case_number:02}. {r.selection} | {r.ISSUER} | {r.cusip} | {r.side} | {r.day.date()} | {r.firm}", r.case_id)
                                  for r in case_manifest.itertuples()]
        fixed_case_box.value = case_manifest.case_id.iloc[0] if len(case_manifest) else None
        impact = quality_population["impact_table"]
        case_manifest_status.value = (f"Frozen {len(case_manifest)} cases; exhaustive {len(impact):,} observed bond/side/day groups, "
                                      f"{int(impact.impact_queries.sum()):,} local queries. High impact cases may be fewer than 6 when no remaining measured effect exists. "
                                      "Counts remain full population; query effects are descriptive and are not prediction gain.")
    finally:
        freeze_button.disabled = False

def apply_fixed_case(change=None):
    global applied_scope, applied_value, step2_result, busy
    if case_manifest is None or fixed_case_box.value is None:
        return
    row = case_manifest.set_index("case_id").loc[fixed_case_box.value]
    busy = True
    try:
        scope_box.value = "Issuer"
        issuer_box.value = row.ISSUER
        value_box.value = row.ISSUER
        applied_scope, applied_value = "Issuer", row.ISSUER
        step2_result = analyze_scope(applied_scope, applied_value)
        view_box.value = "Case"
        case_type.value = "All events"
        for box, col in [(dealer_box, "firm"), (bond_box, "cusip"), (side_box, "side"), (date_box, "day")]:
            box.options = [(str(row[col]), row[col])]
            box.value = row[col]
    finally:
        busy = False
    refresh_step2()
    case_manifest_status.value = f"Case {row.case_number}: {row.selection_reason}; events={int(row.events):,}. " +         "Cleaning/feature decision: preserve candidate/quantity ambiguity and check the measured center/coverage cost."

# All-events navigation exposes sparse and one-sided controls selected by the manifest.
case_type.options = ["All events"] + list(case_type.options)
step2_controls = [view_box, page_box, case_type, dealer_box, bond_box, side_box, date_box, event_page, event_box, window_mode, window_box, values_page]
for box in step2_controls:
    box.observe(refresh_step2, names="value")
scope_box.observe(scope_changed, names="value")
issuer_box.observe(change_issuer, names="value")
apply_button.on_click(apply_scope)
freeze_button.on_click(freeze_cases)
apply_case_button.on_click(apply_fixed_case)
save_button.on_click(save_step2)
step2_dashboard = widgets.VBox([widgets.HBox([scope_box, value_box, apply_button]), issuer_box,
    widgets.HBox([freeze_button, apply_case_button]), fixed_case_box, case_manifest_status,
    widgets.HBox([view_box, page_box, save_button]), case_controls, details, save_status, plot_output])
display(step2_dashboard)
scope_changed()
refresh_step2()
