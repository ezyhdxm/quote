# Data quality research — step 2

Open [quote_quality_step2.ipynb](quote_quality_step2.ipynb) and run its **three code cells**.
The matching [quote_quality_step2.py](quote_quality_step2.py) has VS Code cells.
The loading cell includes the same TRACE cache / pipeline and BondCliQ parquet paths
as step 1. Run from your existing project root containing `data/` and `data.py`.

Select **issuer**, then **Overview** or **Case**:

- **Overview:** dealer multi-spread rates, rates by quantity category, and the
  full distribution of within-group spread ranges. Numerators and denominators
  are labelled. Dealer pages change only the dealer panel.
- **Case:** choose case type, dealer, bond, side and ET date. Select an event and
  inspect raw spread and quantity points in a configurable same-day window.
  Every eligible event is accessible through pages of 50; exact candidate
  values paginate eight per figure.
- **Save PNG:** export the displayed figure to `outputs/quote_quality_step2/`.
  Detailed raw rows and group counts remain in `step2_result`; no large tables
  are printed.

A group is **dealer / bond / side / exact ET timestamp**, without time rounding.
Multi-spread means at least two distinct finite spreads. Repeated content is
counted across all loaded fields, without deleting rows. Nonfinite spread rows,
incomplete groups and rows missing grouping keys are reported separately.
Incomplete groups remain in the rate denominators; the multi-spread rate counts
observed finite candidates and does not certify the other groups as clean.
The day-balanced rate averages within-day group rates across
dealer-bond-side-days; affected-day coverage measures days with any multi-spread.

Quantity categories are mutually exclusive: missing/other first, then contains
zero, then different/same positive values. Units and economic meaning remain
unconfirmed. Missing/other quantity is not plotted at numeric zero.
Spreads retain the confirmed units and multiplier; zero/negative values stay.
No latest/median selection, quote-state reconstruction, expiry, outlier removal,
crossing cleaning, smoothing or feature generation runs here.
The notebook contains no private data or saved real-data outputs.

---

# Data quality research — step 1

Open [quote_quality_step1.ipynb](quote_quality_step1.ipynb) and run its **three code cells**.
The matching [quote_quality_step1.py](quote_quality_step1.py) has VS Code cells.
Run from your existing project root containing `data/` and the local `data.py` loader.

Cell 1 includes your photographed loading code and paths: read `data_ig.parquet`
when present; otherwise call `data.load_merged_prints` with thresholds
`15 / 0.005 / 0.2` and cache the result. Then read the Wells quotes parquet,
restrict it to CUSIPs in the supplied three-month TRACE table, convert UTC to ET,
and map issuer from TRACE. Unrelated model/feature imports are not required.

Select an **issuer** after loading. One screenshot-friendly figure shows:

- Left: positive / zero / missing / other quantity shares for each dealer,
  using every raw row for that issuer as the dealer-specific denominator.
- Right: the positive raw quantity distribution across **all** that issuer's
  dealers. Small discrete distributions use exact-value bars; larger ones use
  a clearly labelled log10 histogram covering the entire positive range.

Dealer pages contain eight dealers. Paging changes only the left panel.
**Save PNG** exports the current figure to `outputs/quote_quality_step1/`.
No large DataFrames are printed; `quantity_result` holds the counts and figure.

Quantity is **not** converted to MM/par. Zero remains separate from missing.
Only true nulls count as missing; negative, infinite and unparseable non-null
values (including blank strings) count as other. Numeric strings are read as
numbers for classification; the original `quantity` values remain intact.
Repeated rows and concurrent quotes remain in the counts. Missing issuer/dealer
labels remain visible. No spread cleaning, deduplication, crossing analysis,
expiry, smoothing or feature generation runs in this notebook. TRACE is already
clean and the confirmed spread multipliers are not changed.

Install `requirements.txt` in the notebook environment if needed. The local
`data.load_merged_prints` dependency is needed only when the TRACE cache is absent.
The repository contains no private input data or executed real-data outputs.

---

The earlier EDA below is a separate notebook; it is not run by step 1.

# Interactive quote EDA

Open [quote_eda.ipynb](quote_eda.ipynb) in the kernel containing `bcq_df` and
`data_ig`. Run **three cells once**, then select an issuer from the dropdown.
The matching [quote_eda.py](quote_eda.py) also works with `%run -i quote_eda.py`.

```python
%pip install -r requirements.txt
```

The interface uses ipywidgets and Plotly. Use a Jupyter/VS Code notebook with
widget rendering enabled. It contains three tabs, not printed dataframe dumps:

| Tab | What to read |
|---|---|
| Cleaning / coverage | Why snapshots were withheld; retained records with/without enough peers; daily plotted-quote coverage by bond; refresh age versus spread-change age; how often dealers move while the median stays zero |
| Cleaned quotes | Liquid/sparse bond comparisons across tenor; clean bid/ask consensus and trades; optional raw audit points and display-only smooth trend |
| Co-movement | Matched-dealer change correlation with overlap counts in hover; exact-zero exclusion sensitivity; PCA variance and PC1 loadings versus tenor |

Raw points and the smooth trend are off initially. Display toggles do not rerun
analysis. Issuer selection rebuilds that issuer's dashboard. An unsuccessful
selection clears the old charts instead of displaying a previous issuer's result.

## Purpose and inputs

The goal is a usable dealer consensus and evidence of shared issuer movements.
This notebook is descriptive research, not a feature-engineering framework.

- `data_ig`: the supplied **three-month trade universe**, with `CUSIP` and
  `EFFECTIVE_DATETIME_TS`. Trade plots additionally use `BM_SPREAD`, `QUANTITY`,
  and `EFF_SIDE`. Defaults convert trade spreads from percentage points to bps.
- `bcq_df`: shorter quote history, with `cusip`, `firm`, `side`,
  `quote_timestamp_ET`, `spread`, and optional `quantity`. UTC is accepted when
  the ET column is absent. Quote spreads default to bps.
- Metadata: `CUSIP`, `ISSUER`, `YRS_TO_MATURITY` in optional `bond_info_df`, or
  issuer/tenor fields in quotes/trades. Security master has highest priority.

Your existing `bcq_df.cusip.isin(data_ig.CUSIP)` restriction is correct for this
scope. The notebook also applies it. A bond with no trade in the entire universe
is excluded; a traded bond with no trade in the shorter quote window remains.
`TRADE_UNIVERSE_START/END` optionally restrict a longer supplied trade history.
Quote `START/END` controls plots independently. Quantities stay in source units.
Availability is assumed to equal **event time**, with strictly earlier events
used at each snapshot. There is no known-time input.

The settings and column maps are in cell 1. Optional `QUOTES_PATH`, `TRADES_PATH`,
and `BOND_INFO_PATH` load parquet files. `EDA_OVERRIDES` takes priority over the
literal settings; clear it when you want those settings to control the run.
`USE_DEMO=True` runs synthetic inputs without private data.

## Exactly how quotes are cleaned

| Step | Decision | Why |
|---|---|---|
| Exact duplicates | Keep one copy | Repeated rows must not increase dealer weight |
| Same dealer/bond/side/event-time batch | Use a supplied numeric feed sequence if present for every row with a unique latest sequence; otherwise median only if spread range <= `TIE_TOL_BPS` (2 bps) and every spread is finite | Avoid selecting an arbitrary last row from conflicting quotes |
| Conflicting latest batch | Invalidate the latest state; do not resurrect an older quote | Feed ambiguity is unresolved evidence, not a fresh quote |
| Freshness | Latest event strictly before snapshot; expire after `MAX_AGE` (30min) | An old quote is unavailable, even if its value looks plausible |
| Contextual zero | With close-time, size-compatible same-dealer sides, flag the zero leg if the opposite spread is at least `ZERO_PAIR_GAP_BPS` (10 bps) away | Detect a plausible missing-side placeholder without deleting all zero/negative spreads |
| Other crossed dealer pairs | Remove both legs when the bad leg is unidentified | Under the convention `bid spread >= ask spread`, that pair is inconsistent |
| Peer outliers | At least 3 remaining same-side dealers; residual > `PEER_Z` times MAD scale, floored at 2 bps | Use contemporaneous quote evidence rather than future trades |
| Plot consensus | Prefer medians from the same noncrossed paired dealer set; independent side medians that cross are withheld | Different dealer sets must not draw a synthetic crossed market |

These thresholds are explicit research heuristics. Zero/negative sign by itself
is never an error. Near-zero or negative consistent quotes remain usable.
A contextual-zero threshold can also reject a genuine unusually wide quote;
inspect raw records and vary the threshold. Set `ZERO_PAIR_GAP_BPS=np.inf` to
turn this particular heuristic off. `ZERO_SPREAD_IS_MISSING=True` is a separate,
optional blanket-zero sensitivity and defaults to **False**.

Nonpositive quantity becomes unknown by default; `quantity_raw` is retained.
Pairing verifies equal positive sizes or explicitly allows unknown size with
`PAIR_ALLOW_UNKNOWN_SIZE=True`. Known unequal/invalid sizes do not verify a pair.
One-sided quotes can remain useful for side analysis. A peer test with fewer than
three dealers is unassessed, so a retained record is not automatically validated.

The cleaning tab counts **snapshot records**, not independent messages. A bad
message carried for several snapshots is therefore counted several times.
The compact status line separately reports duplicate/conflicting message batches.
All raw data and exclusion reasons are retained in `result`; no data is exported.

## Gaps, smoothness and crossing

The default solid line is a **step curve**, because quoted values remain constant
between events. Missing/expired/withheld observations stay missing; curves never
join across unavailable slots or overnight. A gap indicates lack of supported
coverage, not a plotting interpolation problem. Inspect the coverage heatmap
and cleaning counts to understand it.

The optional dotted trend is a causal three-snapshot EWMA, calculated separately
within each uninterrupted day/segment. It resets at gaps, never fills missing
points, and withholds any crossed smoothed pair. It is display only: correlations
and PCA still use actual cleaned dealer changes. No clamp, side swap, interpolation
or smoothing is used to manufacture usable observations.

Raw audit points are hidden initially, so rejected zeros do not compress the
clean plot's vertical range. Enable them to inspect original dealer/side/quantity
values. Trade marker sizes are scaled relative to their panel's source quantities.

## Co-movement

Match the same bond/dealer/side at `t` and `t-CHANGE_LAG`, require freshness and
cleanliness at both ends, and stay within one ET session day. The main change is
the median matched-dealer change with at least two dealers. Size equality is not
a main filter. An exact-zero-endpoint-excluded view provides a sensitivity.

The quality tab also displays the fraction of matched dealers that moved and
the fraction of supported medians equal to zero. This makes median suppression
visible rather than interpreting every unchanged median as no market movement.
`result['movement']['moved_median']` retains the changed-dealer-only alternative;
it conditions on moving dealers and is not substituted for the main statistic.

Correlation hover gives pairwise observation counts. Blank correlations mean
insufficient overlap or zero variance. PCA uses at most six covered varying bonds
and complete standardized rows, without imputation. Its title explains when
sample support is insufficient. PC1 loadings are sign-oriented for display;
loadings describe standardized bond changes, not direct bps sensitivities.
Overlapping lagged changes are not independent observations. Shared dealer or
market-wide effects can also produce issuer co-movement.

## Code organization and checks

Cell 1: settings. Cell 2: normalize once and construct the issuer choices.
Cell 3: one issuer-analysis callback and a short UI refresh wrapper. They are
needed so dropdown changes update the charts without rerunning the input cells.
The source is about half the previous version's length.

For optional inspection, the current selection is in `result`, including
`raw`, `replay`, `states.reason`, `clean`, `pairs`, `levels`, `coverage`,
`matched`, `movement`, `changes`, `correlation`, `pca_loadings`, and `figures`.
These objects are not automatically printed.

```bash
python -m unittest discover -s tests -v
```

Synthetic checks cover traded-bond scope, event-time causality, expiry, timestamp
conflicts/feed sequence, contextual zeros, legitimate zero/negative values,
crossings, unknown size, peer filtering, timezone conversion, dropdown switching,
and display-only smoothing. No private market data is included.
