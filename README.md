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

To run inside your existing notebook without copying its data into another kernel:

```python
EDA_OVERRIDES = {"ISSUER": None}  # or your issuer's exact name
%run -i quote_eda.py
```

`EDA_OVERRIDES` takes priority over settings in cell 1. Remove/reset it when
you want cell 1's literal settings to control the run.

To try synthetic data first, set `USE_DEMO=True`. Keep `demo_data.py` beside the
notebook. The demo includes a quote-only bond, unequal dealer update rates,
one-sided quotes, zero spikes, crossings and an ET daylight-saving transition.
The committed notebook has no outputs.

## Inputs

The defaults match the photographed notebook. Adjust `Q_COL`, `T_COL`, `M_COL`
in cell 1 if your names differ.

| Input | Required columns | Optional columns |
|---|---|---|
| `bcq_df` | `cusip`, `firm`, `side`, `quote_timestamp_ET`, `spread` | `quantity`, configured receipt timestamp |
| `data_ig` | `CUSIP`, `EFFECTIVE_DATETIME_TS` | `BM_SPREAD`, `QUANTITY`, `EFF_SIDE`, `ISSUER`, `YRS_TO_MATURITY` |
| `bond_info_df` | `CUSIP`, `ISSUER` | `YRS_TO_MATURITY` |

`data_ig` may be absent for quote-only analysis. Supply `bond_info_df` in that
case. Metadata inferred from trades is a fallback for descriptive grouping only.
For bonds with no trades, use a static security master; unmapped quote CUSIPs are
reported instead of silently assigned to an issuer.

`spread` defaults to bps; `BM_SPREAD` defaults to percentage points and is
multiplied by 100, as in the screenshot. Confirm the vendor's spread definition
and benchmark before comparing quote and trade levels. Preserve CUSIPs as strings.
Naive timestamps are localized to the configured source time zone; aware
timestamps are converted to ET. Ambiguous/nonexistent naive DST timestamps are
reported as invalid rather than guessed. Plots explicitly display ET.

## What changes from the existing explorer

- Quotes are not filtered against past or future trades. Raw spikes, zero and
  negative spreads stay visible. Exact duplicates and malformed keys/times are
  audited; invalid latest spreads replace earlier valid spreads.
- Every grid point uses the latest message **strictly before** that point for
  each `(cusip, firm, side)`. Each dealer contributes once to each bond/side
  median, irrespective of message count. Quotes expire after `MAX_AGE`.
- Message age and time since the spread/quantity last changed are separate.
  Repeated unchanged refreshes do not imply new economic information.
- The bond comparison includes sparse and quote-only bonds. It selects pairs
  by tenor, trade count and quote coverage, not realized spread outliers.
- Midpoints use bid/ask from the **same dealer**, with compatible timestamps
  and sizes. Crossed eligible pairs are reported and remain in side plots;
  they do not produce usable midpoints. Equal-dealer side medians are never
  forced into a synthetic non-crossed market.
- Peer MAD flags are based on contemporaneous same-bond/same-side dealers.
  The main consensus retains them. A separate sensitivity calculation removes
  flagged states; its lower dealer coverage is explicit.
- Co-movement uses quote **changes** matched on dealer/side and stable quoted
  size at both ends. Correlations include overlap counts. PCA uses complete
  standardized change rows for up to six covered bonds, without imputation.

## Outputs

| Object / figure | Question it answers |
|---|---|
| `audit`, `unmapped_quote_cusips` | Which input issues or missing metadata need attention? |
| `coverage`, bond explorer | Do sparse bonds have useful fresh multi-dealer quotes? |
| `states`, `active`, `side_stats` | What was known before each snapshot, and how old/dispersed was it? |
| `dealer_stats`, dealer diagnostics | Are activity, refresh rates, bias or size dominating the picture? |
| `pairs`, `pair_stats`, pair-quality figure | Are sides asynchronous, size-mismatched or crossed? |
| `matched`, `movement`, `changes` | Do the same dealers move multiple bonds together? |
| `correlation`, `overlap`, `correlation_without_flags` | Is co-movement supported by sufficient aligned observations? |
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
Timestamp ties use the last input row; sort by feed sequence upstream if needed.
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
metadata, time zones and matched-dealer changes. No market dataset is included.
