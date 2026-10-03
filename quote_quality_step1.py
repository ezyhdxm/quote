# %% [markdown]
# # Step 1 — understand quote quantity
# Run three code cells. Start with Global; choose SECTOR, dealer or issuer and Apply.
# TRACE is already clean; benchmark-spread multipliers are already correct.
# This step counts raw quote rows, including repeats. Quantity stays in source
# units: positive, zero, missing and other are separate. No quote cleaning runs.

# %% 1. Load data — paths and cache logic from your notebook
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import ipywidgets as widgets
from IPython.display import display, clear_output

PIPELINE_CSV = Path("data/pipeline/data_pipeline.csv_20260506")
BENCHMARK_CSV = Path("data/pipeline/DailyCloseUSTBenchmarks.csv_20260506")
RAW_QUOTES_FILE = Path("data/bondcliq/quotes_pretrade_260301_260401_Wells_quotes2.parquet")
DATA_IG_CACHE = Path("data/pipeline/data_ig.parquet")

def to_ny_datetime(series):
    values = pd.to_datetime(series, errors="coerce")
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
    bcq_df["quote_timestamp_UTC"], utc=True,
).dt.tz_convert("America/New_York")
from quote_quality_population import (population_tables, scope_selection, scope_options, summary_html)
quality_population = population_tables(data_ig, bcq_df)
bcq_df = quality_population["raw"]

# %% [markdown]
# ## Read the two panels
# Left: each dealer's quantity categories, divided by **all that dealer's raw
# rows in this scope**. Right: **all positive rows in this scope**, regardless
# of the dealer page. Both quote sides are included. Zero's meaning and the
# possible MM unit remain unconfirmed. True nulls are missing; negative values,
# infinities and unparseable non-null values (including blank strings) are other.

# %% 2. Classify quantity and define one figure
KINDS = ["Positive", "Zero", "Missing", "Other"]
COLORS = ["#287D8E", "#E5A43C", "#A5ADB8", "#BD5367"]
DEALERS_PER_PAGE = 8
quantity_data = quality_population["quantity"]

def quantity_figure(issuer=None, page=1, scope="Issuer", value=None):
    selection = scope_selection(quality_population, scope, issuer if value is None else value)
    issuer = selection["issuer"] if scope != "Issuer" else issuer
    q = selection["raw"]
    counts = pd.crosstab(q["firm"], q["quantity_kind"]).reindex(columns=KINDS, fill_value=0)
    order = counts.sum(axis=1).sort_values(ascending=False, kind="stable").index
    counts = counts.reindex(order)
    pages = max(1, (len(counts) + DEALERS_PER_PAGE - 1) // DEALERS_PER_PAGE)
    page = min(max(1, page), pages)
    shown = counts.iloc[(page - 1) * DEALERS_PER_PAGE:page * DEALERS_PER_PAGE]
    shares = shown.div(shown.sum(axis=1), axis=0) * 100
    positive = q.loc[q["quantity_kind"].eq("Positive"), "quantity_numeric"]
    totals = q["quantity_kind"].value_counts().reindex(KINDS, fill_value=0)
    with plt.rc_context({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}):
        fig, (left, right) = plt.subplots(1, 2, figsize=(15, 7), facecolor="white")
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.23, top=0.66, wspace=0.30)
    fig.suptitle(f"{issuer}\nStep 1: what does quote quantity contain?", fontsize=17, y=0.99)
    dates = q["quote_timestamp_ET"].dropna()
    date_label = f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d} ET" if len(dates) else "No quote dates"
    stats = " | ".join(f"{kind}: {totals[kind]:,} ({totals[kind] / max(len(q), 1):.1%})" for kind in KINDS)
    fig.text(0.5, 0.87, f"{date_label} | {len(q):,} raw rows | {len(counts)} dealers | both sides", ha="center", fontsize=12)
    fig.text(0.5, 0.83, stats, ha="center", fontsize=11)
    summary = selection["summary"]
    fig.text(0.5, 0.78, f"Traded bonds with quotes: {int(summary.quoted_bonds):,}/{int(summary.traded_bonds):,}; no quote: {int(summary.no_quote_bonds):,} | {int(summary.events):,} events | {int(summary.dealer_bond_side_days):,} dealer/bond/side/days", ha="center", fontsize=10)
    offset = np.zeros(len(shown))
    for kind, color in zip(KINDS, COLORS):
        widths = shares[kind].to_numpy()
        left.barh(np.arange(len(shown)), widths, left=offset, color=color, label=kind)
        for y, (start, width) in enumerate(zip(offset, widths)):
            if width >= 10:
                left.text(start + width / 2, y, f"{width:.0f}%", ha="center", va="center", fontsize=10)
        offset += widths
    labels = [f"{firm if len(firm) <= 25 else firm[:15] + '...' + firm[-6:]}\nN={int(shown.loc[firm].sum()):,}" for firm in shown.index]
    left.set_yticks(np.arange(len(shown)), labels, fontsize=10)
    left.invert_yaxis()
    left.set_xlim(0, 100)
    left.xaxis.set_major_formatter(PercentFormatter(100))
    left.set_xlabel("Share of this dealer's raw quote rows")
    left.set_title(f"Which dealers report zero or missing quantity?\nDealer page {page}/{pages}; showing {len(shown)}/{len(counts)}", fontsize=12, pad=12)
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False, fontsize=10)
    right.set_title(f"Which positive values occur?\nAll scope dealers; N={len(positive):,} positive rows", fontsize=12, pad=12)
    frequencies = positive.value_counts()
    if positive.empty:
        right.text(0.5, 0.5, "No positive quantity in this scope", ha="center", va="center", transform=right.transAxes)
        right.set_xticks([])
        right.set_yticks([])
    else:
        if len(frequencies) <= 20:
            distribution = frequencies.sort_index() / len(positive) * 100
            right.bar(np.arange(len(distribution)), distribution, color=COLORS[0])
            right.set_xticks(np.arange(len(distribution)), [f"{x:g}" for x in distribution.index], rotation=45, ha="right", fontsize=10)
            right.set_xlabel("Raw quantity value (unit unconfirmed)")
        else:
            right.hist(np.log10(positive), bins=30, weights=np.full(len(positive), 100 / len(positive)), color=COLORS[0], edgecolor="white")
            right.set_xlabel("log10(raw quantity) — full range, unit unconfirmed")
        right.yaxis.set_major_formatter(PercentFormatter(100))
        right.set_ylabel("Share of all positive rows")
        right.set_ylim(0, right.get_ylim()[1] * 1.35)
        common = frequencies.head(4)
        lines = ["Most common raw values:"] + [f"{value:g}: {count / len(positive):.1%} (N={count:,})" for value, count in common.items()]
        right.text(0.98, 0.98, "\n".join(lines), ha="right", va="top", transform=right.transAxes, fontsize=10,
                   bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    fig.text(0.17, 0.06, "Zero remains zero; missing means a null input. Other = negative, nonfinite or unparseable non-null quantity.\nRaw-row counts include repeated quotes. No deduplication, spread filtering, size conversion or trade-size matching.", fontsize=10, color="#444444")
    return {"issuer": issuer, "page": page, "counts": counts, "totals": totals,
            "positive_values": positive, "page_dealers": list(shown.index), "figure": fig,
            "scope": scope, "value": selection["value"], "summary": summary}

# %% [markdown]
# ## Apply a scope, drill down to issuer, and save a screenshot
# Run this cell once. Dealer paging changes only the left panel. Save PNG exports
# exactly the current figure to `outputs/quote_quality_step1/`. Detailed counts
# are available in `quantity_result`, but no tables are displayed automatically.

# %% 3. Scope, issuer drill-down, dealer pages and full PNG export
if "quantity_dashboard" in globals():
    page_box.unobserve(refresh_quantity, names="value")
    scope_box.unobserve(scope_changed, names="value")
    issuer_box.unobserve(change_issuer, names="value")
    apply_button.on_click(apply_quantity, remove=True)
    save_button.on_click(save_quantity, remove=True)
    quantity_dashboard.close()
scope_box = widgets.Dropdown(options=["Global", "SECTOR", "Dealer", "Issuer"], description="Scope:")
value_box = widgets.Dropdown(options=["Global"], description="Group:", layout=widgets.Layout(width="650px"))
issuer_options = scope_options(quality_population, "Issuer")
issuer_box = widgets.Dropdown(options=issuer_options, description="Issuer:", layout=widgets.Layout(width="650px"))
page_box = widgets.Dropdown(options=[1], description="Dealer page:")
apply_button = widgets.Button(description="Apply", button_style="primary")
save_button = widgets.Button(description="Save PNG", icon="download")
save_status, population_details = widgets.HTML(), widgets.HTML()
plot_output = widgets.Output()
quantity_result, applied_scope, applied_value, quantity_busy = None, "Global", "Global", False

def refresh_quantity(change=None):
    global quantity_result
    if quantity_busy:
        return
    if quantity_result is not None:
        plt.close(quantity_result["figure"])
    quantity_result = quantity_figure(applied_value, page_box.value, scope=applied_scope)
    save_status.value = ""
    population_details.value = summary_html(scope_selection(quality_population, applied_scope, applied_value))
    with plot_output:
        clear_output(wait=True)
        display(quantity_result["figure"])
    plt.close(quantity_result["figure"])

def apply_quantity(change=None):
    global applied_scope, applied_value, quantity_busy
    quantity_busy = True
    try:
        applied_scope, applied_value = scope_box.value, value_box.value
        selection = scope_selection(quality_population, applied_scope, applied_value)
        n = selection["raw"]["firm"].nunique()
        page_box.options = range(1, max(1, (n + DEALERS_PER_PAGE - 1) // DEALERS_PER_PAGE) + 1)
        page_box.value = 1
    finally:
        quantity_busy = False
    refresh_quantity()

def scope_changed(change=None):
    value_box.options = scope_options(quality_population, scope_box.value)
    value_box.layout.display = "none" if scope_box.value in ["Global", "Issuer"] else ""
    issuer_box.layout.display = "" if scope_box.value == "Issuer" else "none"
    if scope_box.value == "Issuer" and issuer_box.value in value_box.options:
        value_box.value = issuer_box.value

def change_issuer(change=None):
    # The retained issuer dropdown immediately drills down; scope/group selection
    # stays pending until Apply so large-table work cannot repeat during changes.
    scope_box.value = "Issuer"
    value_box.value = issuer_box.value
    apply_quantity()

def save_quantity(change=None):
    folder = Path("outputs/quote_quality_step1")
    folder.mkdir(parents=True, exist_ok=True)
    name = "".join(c if c.isalnum() else "_" for c in str(quantity_result["issuer"]))[:80]
    path = folder / f"{name}_dealer_page_{quantity_result['page']}.png"
    quantity_result["figure"].savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
    save_status.value = f"Saved: {path}"

page_box.observe(refresh_quantity, names="value")
scope_box.observe(scope_changed, names="value")
issuer_box.observe(change_issuer, names="value")
apply_button.on_click(apply_quantity)
save_button.on_click(save_quantity)
quantity_dashboard = widgets.VBox([widgets.HBox([scope_box, value_box, apply_button]), issuer_box,
                      widgets.HBox([page_box, save_button]), population_details, save_status, plot_output])
display(quantity_dashboard)
scope_changed()
apply_quantity()
