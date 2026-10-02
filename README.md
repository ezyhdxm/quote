# Data quality research — step 3

Open [quote_quality_step3.ipynb](quote_quality_step3.ipynb) and run its **three code cells**.
It is standalone: the loading cell includes the same TRACE cache / pipeline,
BondCliQ parquet, three-month traded-bond universe and ET conversion as steps 1–2.
The matching [quote_quality_step3.py](quote_quality_step3.py) has VS Code cells.
Run from the existing project root containing `data/` and `data.py`.

Choose **issuer → dealer / bond / side / ET day**. **All four** is the default: a single 2×2 dashboard with three small plots per section and one shared case header.
Previously inspected issuers lead the dropdown when present; automatically ranked
multi-spread, wide-gap, quantity and active comparison cases follow. Every issuer
remains available. Case ranking is retrospective selection, not an online feature.
Optional detail layouts enlarge one section without changing the case. Every observed quantity condition remains selectable. **Save all 4 PNG** always exports the full dashboard, including when a detail layout is displayed. Send that PNG to share all settings and results together.

| View | What it compares | Decision / feature use |
|---|---|---|
| Candidates | All raw candidates versus their distinct-spread median; adjacent changes with/without a quantity-support and candidate-count guard | Retain ambiguity; avoid treating a change in quote conditions as ordinary momentum |
| Quantity | All raw points and one selected raw-size condition; within-size multi-spread counts; eligible within-event contrasts | Decide whether conditional aggregation has enough repeated support |
| Age | Dealer-equal no-expiry mean, exponential age decay and a maximum-message-age filter | Compare level changes against lost coverage; retain message age and observed set-change age separately |
| Influence | Dealer-equal aggregation, candidate clipping and dealer downweighting | Quantify how proposed influence control changes the result; never label the peer median as truth |

Every chart includes observed counts, the rule being tested and the feature use.
Raw observations are retained. No large DataFrames, smoothing, crossing cleanup,
quote-ID inference or model training run here. Empty / unsupported references are
labelled instead of treated as zero deviation. Figures use one PNG widget and cell reruns detach old callbacks. All sections use the same time range. Quantity shows all conditions in separate colors when there are at most six; otherwise it shows all points as background and highlights the selected condition.

## Calculation choices to inspect before using the drafts

- Events are `firm / cusip / side / exact timestamp`. Duplicate rows do not change
  candidate sets or price weights. Zero and negative spreads remain; raw quantity
  zero, missing and other states remain distinct. Positive quantity is never
  converted to TRACE par or declared executable.
- The event center is the median of **distinct finite spreads**, a descriptive
  value that need not be a quoted candidate. Incomplete events are visible in
  the research plots but do not enter numerical as-of aggregation. A later
  incomplete event blocks that dealer's contribution; it does not silently
  fall back to the preceding valid quote.
- Adjacent changes are assessed only within an ET day, with two complete events
  and a gap at most `HISTORY_GAP_MIN=60`. `guarded_delta` additionally requires the
  same quantity-condition set and candidate count. This is a comparability
  convention, not identification of the same quote. A/B/A is marked at the
  **third** observed event; the earlier observations are never rewritten.
- Change history starts unknown, not at age zero. The age remains missing until
  an observed set change in a continuous complete history. Day boundaries,
  long gaps and incomplete events restart this history; observed history length
  is retained. `changes_30m` counts observed changes in **(t−30m, t]** within that
  history segment, so a short history is not a fully observed 30-minute window.
- `asof_features(events, times, age_min=30)` expects one bond/side and aware query
  times. It uses the latest event at or before each query, only within the same
  ET day. Equal-time events are processed as one atomic quote set. Taking the
  latest observed set is a feature convention; it does not infer withdrawal of
  candidates omitted from the next event.
- Baseline aggregation gives one total vote per dealer: mean of dealer event
  centers. Message volume and candidate multiplicity cannot increase a dealer's
  total vote. The exploratory Age setting (10/30/60 minutes) is both the
  message-age half-life and the maximum-age hypothesis, in separate comparisons.
- Influence references exclude the target dealer, require at least three other
  complete dealer observations within the selected age limit, and use their
  center median. The illustrative radius is `max(10 bps, 4 × 1.4826 × peer MAD)`.
  One comparison clips each finite candidate to that reference interval, then
  takes its median; the other keeps the dealer center but weights it by
  `min(1, radius / abs(center − peer_median))`. Without enough peers both retain
  the baseline numerically, while the impact chart leaves those comparisons blank and labels them **NOT ASSESSED** when none are available. Candidate changes and median changes are counted separately. Raw records are untouched. Unknown size conditions can explain
  apparent disagreement; these are uncalibrated alternatives, not corruption tests.
- Quantity contrasts use complete all-positive events with at least two quantities
  and exactly one spread per quantity. The contrast subtracts the event's median
  across those quantities. Quantity graphs summarize the selected day and are
  **diagnostics**, not features computed with future information.

## Background results — no automatic table output

`step3_result` retains raw rows, event histories, selected bond/side as-of dealer
slots and comparison features. `step3_features` contains the current bond/side
research case on a five-minute query grid plus its first/last observed timestamps;
it is not an all-bond trade-level feature export. `asof_features` also accepts
arbitrary query times for later integration. No private data is committed.

| Draft columns / group | Meaning |
|---|---|
| `center_equal`, `center_decay`, `center_max_age` | No-expiry, message-age-weighted and age-limited mean dealer centers, in bps |
| `center_candidate_clip`, `center_dealer_downweight` | Two counterfactual influence-control aggregates, in bps |
| `n_dealers`, `n_fresh_dealers`, `n_incomplete` | Complete, age-eligible and incomplete latest dealer observations |
| `dispersion_bps`, `mean_candidate_gap`, `multi_fraction` | Cross-dealer spread dispersion versus within-dealer candidate ambiguity; not posterior uncertainty |
| `zero_quantity_fraction`, `unknown_quantity_fraction` | Shares of contributing dealers whose latest sets contain zero or missing/other quantity |
| `median_message_age_min`, `median_change_age_min`, `unknown_change_age_fraction` | Age and its observation limitation; median change age uses only known ages |
| `max_decay_weight_share`, `decay_effective_dealers` | Largest normalized age-decay weight, and effective N = (sum w)^2 / sum(w^2); these describe contribution concentration, not independent-source counts |
| `center_lower`, `center_upper` | Dealer-equal mean of candidate minima / maxima; descriptive sensitivity bounds, not executable prices or uncertainty bands |
| `n_peer_supported`, `n_clipped_dealers`, `n_changed_centers` | Supported dealer count, dealers with any clipped candidate, and dealers whose median changes; unsupported fallback is not evidence that a rule worked |
| `center_delta_30m`, `composition_changed_30m` | Fixed 30-minute endpoint comparison, independent of query batch; delta is withheld if dealer roster, quantity support or candidate count changes, or history is unavailable |

Per-event `center_nearest_gap` measures distance from the median to the nearest observed candidate. `lo_delta` and `hi_delta` compare candidate extrema under the same condition checks as `guarded_delta`; they do not track quote identities. Per-event `center_delta`, `guarded_delta`, `condition_changed`, `observed_aba`,
`changes_30m`, `change_age_unknown` and `history_start` are in the event history.
The fixed-horizon aggregate delta describes observed summaries, not matched quote
identities; its ages and ambiguity must be considered with it.

This implements the Step 3 research comparisons and feature draft. The revised dashboard distinguishes persistent multi-level ambiguity, supported quantity conditions, contribution changes, and insufficient peers. Out-of-time predictive validation remains pending; reviewed cases do not establish a generally valid cleaner.
A more stable-looking line is not evidence that a rule improves the LGBM task.

---

# Data quality research — step 2

Open [quote_quality_step2.ipynb](quote_quality_step2.ipynb) and run its **three code cells**.
The matching [quote_quality_step2.py](quote_quality_step2.py) has VS Code cells.
The loading cell includes the same TRACE cache / pipeline and BondCliQ parquet paths
as step 1. Run from your existing project root containing `data/` and `data.py`.

Select **issuer**, then **Overview** or **Case**:

Representative issuers are automatically placed first. The labelled front section
interleaves broad multi-spread coverage, wide candidate ranges, different / zero /
same / unknown quantity cases, and active low-multi-spread controls (up to two
distinct issuers per theme). Labels show the reason and multi-group count / total
group count. All remaining issuers follow alphabetically; none are removed.

Defaults require at least 100 keyed groups, two ET dates and two dealers; anomaly
themes require at least five multi-spread groups. Controls have no incomplete
spread groups and at most a 1% day-balanced multi-spread rate. If no issuer meets
the support requirements, the fallback is explicitly labelled **limited sample**.
These constants are at the start of cell 2. Ranking runs one vectorized pass over
narrow quote fields, rather than running every issuer dashboard; selected-issuer
plots are still built on demand. `issuer_summary` holds the ranking evidence
without automatically printing a table. The front section is research case
selection, not a random sample or a dealer/issuer quality score.

- **Overview:** dealer multi-spread rates (most multi-spread events first),
  **quantity composition among multi-spread events**, and the cumulative distribution
  of same-time candidate gaps. The middle panel sums to 100%; it splits all-zero
  from zero/positive mixtures. Its denominator is all issuer multi-spread events,
  not all events in that quantity category. Dealer pages change only the left panel.
- **Case:** choose case type, dealer, bond, side and ET date. Select an event and
  inspect a **full day** of raw spread points, or switch to a local +/- minute window.
  The right panel shows each selected-time spread/quantity combination separately,
  labelled with raw quantity and row count. Two q=0 candidates remain separately
  visible. Row numbers are display positions, not tracked quote identities.
  Every eligible event is accessible through pages of 50; candidate values
  paginate eight per figure. No candidates are joined into an inferred curve.
- **Save PNG:** export the displayed figure to `outputs/quote_quality_step2/`.
  Detailed raw rows and group counts remain in `step2_result`; no large tables
  are printed.

A group is **dealer / bond / side / exact ET timestamp**, without time rounding.
Multi-spread means at least two distinct finite spreads. Repeated content is
counted across all loaded fields, without deleting rows. Nonfinite spread rows,
incomplete groups and rows missing grouping keys are reported separately.
Incomplete groups remain in the rate denominators; the multi-spread rate counts
observed finite candidates and does not certify the other groups as clean.
The collapsed **Counting details and interpretation** panel retains the day-balanced
rate (average group rate across dealer-bond-side-days) and affected-day coverage
(units with any multi-spread). These units are not calendar days. Extra identical
rows do not count unchanged quotes at later timestamps.

The charts render into one PNG image widget, using unmanaged Matplotlib figures;
there is no second inline figure display. Rerunning the controls cell detaches
its old observers. When replacing an older notebook, restart the kernel and run
all three cells once to clear the old dashboard and its saved outputs.

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
