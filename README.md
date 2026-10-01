# Quote EDA

Open **[quote_eda.ipynb](quote_eda.ipynb)** in the same Jupyter kernel where `bcq_df`
and `data_ig` are loaded. The matching [quote_eda.py](quote_eda.py) uses VS Code
`# %%` cells. Nine sequential code cells; no model-training or feature-generation framework.

```python
# Install requirements in your notebook environment once.
%pip install -r requirements.txt
```

Edit the settings in cell 1, then run the cells in order. `ISSUER=None` selects
the issuer with the most quoted bonds. To switch issuers or change the analysis
window, edit cell 1 and rerun. Interactive Plotly figures support hover, zoom and
legend toggles; no widgets extension is required.

You can also load complete parquet inputs directly in cell 1 with `QUOTES_PATH`,
`TRADES_PATH`, and `BOND_INFO_PATH`. Do **not** prefilter quotes with
`bcq_df.cusip.isin(data_ig.CUSIP)`: that removes quote-only bonds before the EDA
can see them. Reload the full quote input if your earlier loader did this.

To run inside your existing notebook without copying its data into another kernel:

```python
EDA_OVERRIDES = {"ISSUER": None}  # or your issuer's exact name
%run -i quote_eda.py
```

`EDA_OVERRIDES` takes priority over settings in cell 1. Remove/reset it when
you want cell 1's literal settings to control the run.

To try synthetic data first, set `USE_DEMO=True`. Keep `demo_data.py` beside the
notebook. The demo includes a quote-only bond, unequal dealer update rates,
one-sided quotes, zero spikes, zero-coded missing sizes, crossings and an ET
daylight-saving transition. Demo quantities use small source units.
The committed notebook has no outputs.

## Inputs

The defaults match the photographed notebook. Adjust `Q_COL`, `T_COL`, `M_COL`
in cell 1 if your names differ.

| Input | Required columns | Optional columns |
|---|---|---|
| `bcq_df` | `cusip`, `firm`, `side`, `quote_timestamp_ET`, `spread` | `quantity`, configured receipt/sequence columns, `ISSUER`, `YRS_TO_MATURITY` |
| `data_ig` | `CUSIP`, `EFFECTIVE_DATETIME_TS` | `BM_SPREAD`, `QUANTITY`, `EFF_SIDE`, `ISSUER`, `YRS_TO_MATURITY` |
| `bond_info_df` | `CUSIP`, `ISSUER` | `YRS_TO_MATURITY` |

If ET is absent, `quote_timestamp_UTC` is accepted and converted to ET. `data_ig`
may be absent for quote-only analysis. Metadata priority is static security
master (`bond_info_df`), quote metadata, then latest trade metadata. Trade
metadata is only a descriptive grouping fallback. Unmapped quote CUSIPs are
reported instead of silently assigned to an issuer.

`spread` defaults to bps; `BM_SPREAD` defaults to percentage points and is
multiplied by 100, as in the screenshot. Confirm the vendor's spread definition
and benchmark before comparing quote and trade levels. Preserve CUSIPs as strings.
Naive timestamps are localized to the configured source time zone; aware
timestamps are converted to ET. Ambiguous/nonexistent naive DST timestamps are
reported as invalid rather than guessed. Plots explicitly display ET.
Quantity stays in **source units**; no dollar/par conversion is assumed. Trade
marker sizes are scaled relative to their panel's quantities.

## Placeholder policies

These defaults address the missing-value patterns in the supplied screenshots;
confirm them against your feed's definitions. Raw messages remain available.

| Setting | Default behavior | How to change it |
|---|---|---|
| `NONPOSITIVE_QUANTITY_IS_MISSING=True` | Zero/negative quantity becomes unknown; `quantity_raw` retains the numeric input | Set `False` to keep it; nonpositive sizes still cannot verify a usable pair |
| `ZERO_SPREAD_IS_MISSING=True` | Exact zero is excluded from consensus, pairs and changes; raw points/dotted curves retain it | Set `False` if exact zero is a valid spread |
| `PAIR_ALLOW_UNKNOWN_SIZE=True` | Timestamp-compatible unknown-size bid/ask pairs are allowed but explicitly unverified | Set `False` to require known equal positive sizes |

Finite negative spreads are retained. An invalid latest spread replaces the old
state: the replay does not revive an earlier valid quote. Compare `size_status`,
`mid_known_size` and `mid_unknown_size`; two missing sizes never count as equal.

## What changes from the existing explorer

- Quotes are not filtered against past or future trades. Raw spikes, zero and
  negative spreads stay visible. Exact duplicates, timestamp conflicts and
  malformed keys/times are audited.
- Every grid point uses the latest message **strictly before** that point for
  each `(cusip, firm, side)`. Each dealer contributes once to each bond/side
  median, irrespective of message count. Quotes expire after `MAX_AGE`.
- Message age, time since the spread changed, and time since either spread or
  normalized quantity changed are separate. Repeated refreshes do not imply new
  economic information.
- The bond comparison includes sparse and quote-only bonds. It selects pairs
  by tenor, trade count and quote coverage, not realized spread outliers.
- Midpoints use bid/ask from the **same dealer**, with compatible timestamps
  and size status. Known mismatches are rejected; unknown-size eligibility is
  configurable. Crossed eligible pairs are reported before peer filtering;
  they do not produce usable midpoints. Equal-dealer side medians are never
  forced into a synthetic non-crossed market.
- Peer MAD flags are based on contemporaneous same-bond/same-side dealers.
  Solid consensus curves and the main co-movement calculation remove flags.
  Dotted curves show raw fresh-state medians. Before-filter results and coverage
  are retained. Fewer than three dealers means the peer test is not assessed.
- Co-movement uses quote **changes** matched on dealer/bond/side at both ends,
  with freshness and same ET session day. The main view includes every size;
  verified stable positive size and unknown size are separate sensitivities.
  This prevents widespread missing size from erasing the main sample, but
  size changes can still influence quoted spreads. Correlations include overlap
  counts. PCA uses complete standardized change rows for up to six covered
  bonds, without imputation.
- Snapshot, bid/ask-pair and matched-change funnels show where observations are
  lost. Per-bond coverage distinguishes insufficient observations from constant
  changes that cannot yield a correlation.

## Outputs

| Object / figure | Question it answers |
|---|---|
| `audit`, `unmapped_quote_cusips` | Which input issues or missing metadata need attention? |
| `coverage`, bond explorer | Do sparse bonds have useful fresh multi-dealer quotes? |
| `states`, `active_raw`, `active`, `clean_active`, `side_stats`, `raw_side_stats` | What was known, which states survived each filter, and how old/dispersed were they? |
| `dealer_stats`, dealer diagnostics | Are activity, refresh rates, bias or size dominating the picture? |
| `pairs`, `pair_stats`, pair-quality figure | Are sides asynchronous, size-mismatched or crossed? |
| `matched`, `movement`, `changes` | Do the same dealers move multiple bonds together? |
| `snapshot_funnel`, `pair_funnel`, `movement_funnel`, `comovement_coverage` | How many observations survive, and why can a result be missing? |
| `correlation`, `overlap`, `correlation_before_peer_filter` | Is the main all-size co-movement supported, and sensitive to peer filtering? |
| `changes_stable_size`, `changes_unknown_size`, `correlation_stable_size`, `correlation_unknown_size` | Does the result depend on verified stable sizes or unverified sizes? |
| `pca_variance`, co-movement figure | Is there a shared low-dimensional pattern in the covered subset? |
| `figures` | Plotly figure objects for optional local inspection/export |

There is no automatic data or figure export. The repository contains generated
code, documentation and synthetic examples.

## Interpretation and scope

Start with coverage and dealer diagnostics, then inspect bond plots and matched
changes. Compare `MAX_AGE=10/30/60min`, `CHANGE_LAG=15/30/60min`, and both
`ANALYSIS_SIDE='bid'/'ask'`. Repeated overlapping 30-minute changes are descriptive
observations, not independent samples for inference.

This is a **latest-message view**, assuming one current message per dealer,
bond and side. It is not an executable book reconstruction: cancellations,
withdrawals, quote type and concurrent size ladders require their feed semantics.
If multiple simultaneous size slots exist, preserve a stable slot identifier
and reduce within dealer before cross-dealer aggregation. Do not interpret
latest-message quantities as a full size curve. Missing size on both sides
means unknown size, not verified equality.

With no receipt/availability timestamp, causal replay assumes event-time
availability. Set `QUOTE_AVAILABLE_COL` to a receipt column when available.
Set `QUOTE_SEQUENCE_COL` for numeric feed sequence when available; ties otherwise
use the last input row. This cannot reconstruct simultaneous size slots.
Late older events are audited and do not overwrite a newer known state. Records
with event time after their availability time remain in the raw audit and are
excluded from state replay.
The session grid uses weekdays, not a holiday calendar. The bond cap uses
full-input quote counts for descriptive sample selection and reports its subset.
Full-sample correlations and PCA are EDA summaries, not causal features.

Peer consensus can be wrong when dealers share an error. High co-movement can
reflect one dealer's batch repricing or a common vendor source. A large first PC
does not establish a credit factor or downstream predictive value. Positive EDA
results justify prototyping an online issuer state and evaluating it later on
chronological prediction data.

## Checks

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic inputs and exercise snapshot causality, availability,
expiry, invalid latest records, dealer balancing, pair eligibility, quote-only
metadata, time zones, feed sequence, zero placeholders, unknown sizes, changing
sizes and matched-dealer changes. No market dataset is included.
