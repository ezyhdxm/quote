"""Finite research diagnostics from completed Step5 features and validation only.

No quote loading, event construction, training, or held-out prediction access.
State cuts are declared here, never estimated from validation errors or values.
"""
# SETUP LOGIC: STEP 1 — Import finite validation-only diagnostics dependencies
from hashlib import sha256
from pathlib import Path
from uuid import uuid4
import json

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

from quote_quality_core import to_ny_datetime
from quote_quality_saved import direct_reference, validation_rows


# SETUP LOGIC: Declare research_reference; existing arguments and docstring are preserved
def research_reference(selected):
    # CORE LOGIC: STEP 1 — Choose the required direct comparison model
    # Input: selected='Age decay + Issuer'
    # Output: reference='Age decay'; the established Age decay model uses its saved direct reference
    # Trick: New Direction/Issuer variants cannot use a weaker baseline to overstate their incremental gain.
    return 'Age decay' if selected in ['Age decay + Direction', 'Age decay + Issuer'] else direct_reference(selected)


# SETUP LOGIC: Declare _numeric; existing arguments and docstring are preserved
def _numeric(rows, column):
    # CORE LOGIC: STEP 1 — Convert numeric features and keep missing support unavailable
    # Input: rows[column]=[2,'bad',float('inf')]; a second requested column is absent
    # Output: first numeric series=[2,NaN,NaN]; absent column is all NaN on the original row index
    if column not in rows:
        return pd.Series(np.nan, index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors='coerce').replace([np.inf, -np.inf], np.nan)


# SETUP LOGIC: Declare _labels; existing arguments and docstring are preserved
def _labels(values, missing='Unknown'):
    # CORE LOGIC: STEP 1 — Normalize target-row labels
    # Input: values=[' Energy ', '', None], missing='Unknown'
    # Output: ['Energy','Unknown','Unknown']
    return values.astype('string').str.strip().replace('', pd.NA).fillna(missing)


# SETUP LOGIC: Declare _bins; existing arguments and docstring are preserved
def _bins(values, boundaries, labels):
    # CORE LOGIC: STEP 1 — Bin fixed nonnegative values without fitting cutoffs
    # Input: values=[0,5,5.1,30,31,-1,None], boundaries=[5,30], labels=['0-5','5-30','>30']
    # Output: ['0-5','0-5','5-30','5-30','>30','Unknown','Unknown']
    # Trick: right=True includes each upper boundary; negative or unavailable values remain Unknown.
    valid = values.where(values.ge(0))
    return pd.cut(valid, [-np.inf] + list(boundaries) + [np.inf], labels=labels,
                  right=True).astype('string').fillna('Unknown')


# SETUP LOGIC: Declare _states; existing arguments and docstring are preserved
def _states(rows):
    """Return exhaustive, mutually exclusive states within each family."""
    # SETUP LOGIC: STEP 1 — Initialize the helper interface
    states, definitions = {}, []
    # SETUP LOGIC: Declare add; existing arguments and docstring are preserved
    def add(name, values, columns, definition):
        # CORE LOGIC: STEP 1 — Register aligned states and whether all required columns exist
        # Input: name='quote_state', values=['Two-sided',None], columns=['bcq_bid_n_dealers','bcq_ask_n_dealers'], definition='side counts'; ask column absent
        # Output: states['quote_state']=['Two-sided','Unknown']; definitions row has available=False, cut_source='predeclared fixed cuts'
        states[name] = values.astype('string').fillna('Unknown')
        definitions.append(dict(family=name, columns=columns, definition=definition,
                                available=all(c in rows for c in columns), cut_source='predeclared fixed cuts'))
    # CORE LOGIC: STEP 1 — Distinguish complete quote states and maximum side support
    # Input: bid=[0,1,2,None], ask=[0,0,4,1]
    # Output: quote_state=['No quote','One-sided','Two-sided','Unknown']; dealer_support=['0','1','4+','Unknown']
    # Trick: Both side counts must be known and nonnegative; maximum support is not a distinct-dealer union.
    bid, ask = [_numeric(rows, f'bcq_{s}_n_dealers') for s in ['bid', 'ask']]
    quote = pd.Series('Unknown', index=rows.index)
    known = bid.notna() & ask.notna() & bid.ge(0) & ask.ge(0)
    quote.loc[known] = np.select([(bid.gt(0) & ask.gt(0))[known],
                                 (bid.gt(0) | ask.gt(0))[known]], ['Two-sided', 'One-sided'], default='No quote')
    add('quote_state', quote, ['bcq_bid_n_dealers', 'bcq_ask_n_dealers'],
        'Complete same-day dealer counts: both sides >0 / one side >0 / neither. Missing counts are Unknown.')
    support = pd.concat([bid, ask], axis=1).max(axis=1).where(known)
    add('dealer_support', _bins(support, [0, 1, 3], ['0', '1', '2-3', '4+']),
        ['bcq_bid_n_dealers', 'bcq_ask_n_dealers'], 'Maximum of bid/ask complete-dealer counts; not a distinct-dealer union.')
    # CORE LOGIC: STEP 2 — Identify observed peer support on either side
    # Input: peer counts bid=[0,1,None], ask=[0,0,0]
    # Output: peer_support=['No supported side','Any supported side','Unknown']
    # Trick: Missing support is not a measured zero effect; positive support requires fresh-peer evidence in saved features.
    peers = pd.concat([_numeric(rows, f'bcq_{s}_n_peer_supported') for s in ['bid', 'ask']], axis=1)
    peer = pd.Series('Unknown', index=rows.index)
    pknown = peers.notna().all(axis=1) & peers.ge(0).all(axis=1)
    peer.loc[pknown] = np.where(peers.loc[pknown].gt(0).any(axis=1), 'Any supported side', 'No supported side')
    add('peer_support', peer, ['bcq_bid_n_peer_supported', 'bcq_ask_n_peer_supported'],
        'At least one dealer has >=3 other fresh peers on either side; lack of support is not a measured zero effect.')
    for side in ['bid', 'ask']:
        for kind in ['message', 'change']:
            # CORE LOGIC: STEP 3 — Assign fixed message/change-age and disagreement states
            # Input: bid message age=[5,31,None], bid mean_candidate_gap=[0,2.1,11]
            # Output: bid_message_age=['0-5','>30','Unknown']; bid_candidate_gap=['0','2-10','>10']
            # Trick: The same predeclared age and bps boundaries apply to both sides; validation values do not fit thresholds.
            column = f'bcq_{side}_median_{kind}_age_min'
            add(f'{side}_{kind}_age', _bins(_numeric(rows, column), [5, 30], ['0-5', '5-30', '>30']), [column],
                'Dealer-median age in minutes: [0,5], (5,30], >30; negative, nonfinite or missing = Unknown.')
        for short, suffix in [('candidate_gap', 'mean_candidate_gap'), ('dispersion', 'dispersion_bps')]:
            column = f'bcq_{side}_{suffix}'
            add(f'{side}_{short}', _bins(_numeric(rows, column), [0, 2, 10], ['0', '0-2', '2-10', '>10']), [column],
                'bps: zero, (0,2], (2,10], >10; negative/nonfinite/missing = Unknown. Fixed descriptive cuts, not tuned thresholds.')
        for kind in ['zero', 'unknown']:
            column = f'bcq_{side}_{kind}_quantity_fraction'
            # CORE LOGIC: STEP 4 — Flag presence of zero or unknown quantities without deleting candidates
            # Input: bcq_bid_unknown_quantity_fraction=[0,0.5,None,1.2]
            # Output: bid_unknown_quantity=['Absent','Present','Unknown','Unknown']
            # Trick: Only fractions in [0,1] are assessed; missing side support stays Unknown.
            fraction = _numeric(rows, column)
            state = pd.Series('Unknown', index=rows.index)
            good = fraction.between(0, 1)
            state.loc[good] = np.where(fraction.loc[good].gt(0), 'Present', 'Absent')
            add(f'{side}_{kind}_quantity', state, [column],
                'Any latest complete dealer candidates contain this quantity state (>0 fraction); no side support remains Unknown.')
    pair, matched = _numeric(rows, 'bcq_n_pair'), _numeric(rows, 'bcq_n_size_time_pair')
    state = pd.Series('Unknown', index=rows.index)
    # CORE LOGIC: STEP 5 — Record pairing support without claiming a matching gain
    # Input: bcq_n_pair=[0,2,2,1], bcq_n_size_time_pair=[0,0,1,2]
    # Output: paired_support=['No fresh pair','Fresh pair only','Size-time pair','Unknown']
    # Trick: A matched count larger than its fresh-pair count is invalid; gain still requires same-sample A/B/C analysis.
    good = pair.ge(0) & matched.ge(0) & matched.le(pair)
    state.loc[good] = np.select([matched.loc[good].gt(0), pair.loc[good].gt(0)],
                              ['Size-time pair', 'Fresh pair only'], default='No fresh pair')
    add('paired_support', state, ['bcq_n_pair', 'bcq_n_size_time_pair'],
        'Any size-time pair / fresh pair without size-time support / neither. This is support, not matching gain.')
    issuer = _numeric(rows, 'NUM_OF_ISSUER_TRADES_SINCE_PREV')
    state = pd.Series('Unknown', index=rows.index)
    # CORE LOGIC: STEP 6 — Use the existing issuer count and optional saved anchor timestamp
    # Input: NUM_OF_ISSUER_TRADES_SINCE_PREV=[0,2,None]; current=10:05 ET, previous=10:00 ET
    # Output: issuer_trade_count=['0','>0','Unknown']; saved anchor_age='0-5'
    # Trick: The count is an existing BASE feature, not a rebuilt cross-bond history or learned issuer factor.
    state.loc[issuer.ge(0)] = np.where(issuer.loc[issuer.ge(0)].gt(0), '>0', '0')
    add('issuer_trade_count', state, ['NUM_OF_ISSUER_TRADES_SINCE_PREV'],
        'Existing BASE issuer-trades count only; does not reconstruct issuer history or a latent factor.')
    if 'PREV_EFFECTIVE_DATETIME_TS' in rows:
        age = (to_ny_datetime(rows.time) - to_ny_datetime(rows.PREV_EFFECTIVE_DATETIME_TS)).dt.total_seconds() / 60
        add('anchor_age', _bins(age, [5, 30], ['0-5', '5-30', '>30']), ['PREV_EFFECTIVE_DATETIME_TS'],
            'Current target time minus saved previous-trade timestamp, minutes; negative/nonfinite/missing = Unknown.')
    sector = _labels(rows['SECTOR']) if 'SECTOR' in rows else pd.Series('Unknown', index=rows.index)
    # CORE LOGIC: STEP 7 — Use target-row sector and fixed TRACE trade-size categories
    # Input: SECTOR=[' Energy ',None,None]; QUANTITY=[50000,500000,2000000]
    # Output: SECTOR=['Energy','Unknown','Unknown']; trade_size=['<=100K','100K-1MM','>=1MM']
    add('SECTOR', sector, ['SECTOR'], 'Target trade-row SECTOR; no whole-window CUSIP mapping or added model feature.')
    quantity = _numeric(rows, 'QUANTITY')
    size = pd.Series('Unknown', index=rows.index)
    good = quantity.ge(0)
    size.loc[good] = np.select([quantity.loc[good].le(100_000), quantity.loc[good].ge(1_000_000)],
                              ['<=100K', '>=1MM'], default='100K-1MM')
    add('trade_size', size, ['QUANTITY'], 'TRACE par quantity: [0,100K], (100K,1MM), >=1MM; invalid/missing = Unknown.')
    # CORE LOGIC: STEP 8 — Keep existing target trade type and return exhaustive family definitions
    # Input: TRADE_TYPE=['B',None]; all registered families have a state for each row
    # Output: trade_type=['B','Unknown']; return states plus a definitions table
    # Trick: Families overlap with one another, but each family partitions its own validation rows exactly once.
    types = _labels(rows.TRADE_TYPE) if 'TRADE_TYPE' in rows else pd.Series('Unknown', index=rows.index)
    add('trade_type', types, ['TRADE_TYPE'], 'Existing target trade type; no inference from quote side.')
    return states, pd.DataFrame(definitions)


# SETUP LOGIC: Declare _metrics; existing arguments and docstring are preserved
def _metrics(rows, overall_n):
    # CORE LOGIC: STEP 1 — Compute paired mean loss contributions and model-specific tails
    # Input: selected_error=[1,13], reference_error=[2,12], loss_delta=[-1,1], overall_n=4; both rows share one day
    # Output: n=2, dates=1, delta_mae=0, total_loss_delta=0, contribution_bps=0, share=0.5; delta_p95=0.9, delta_gt10=0
    # Trick: P95 is each model error quantile, not the quantile of paired differences; contribution uses overall N.
    selected, reference = rows.selected_error, rows.reference_error
    return dict(n=len(rows), dates=rows.day.nunique(), selected_mae=selected.mean(), reference_mae=reference.mean(),
                delta_mae=rows.loss_delta.mean(), total_loss_delta=rows.loss_delta.sum(),
                contribution_bps=rows.loss_delta.sum() / overall_n, share=len(rows) / overall_n,
                selected_p95=selected.quantile(.95), reference_p95=reference.quantile(.95),
                delta_p95=selected.quantile(.95) - reference.quantile(.95),
                selected_gt10=selected.gt(10).mean(), reference_gt10=reference.gt(10).mean(),
                delta_gt10=selected.gt(10).mean() - reference.gt(10).mean())


# SETUP LOGIC: Declare diagnostics; existing arguments and docstring are preserved
def diagnostics(frame, predictions, selected='Age decay', reference=None):
    """Same-row state loss diagnostics. Contributions add only within one family."""
    # CORE LOGIC: STEP 1 — Require saved validation rows, the chosen model and its direct reference
    # Input: completed frame/predictions match on row_id=7, time=Mar2 10:00 ET, target=5; selected='Age decay + Direction', explicit reference='Base'
    # Output: ValueError: Age decay + Direction must compare directly with Age decay
    # Trick: validation_rows first verifies row IDs, times and targets; no test rows are requested.
    rows, wide = validation_rows(frame, predictions)
    expected = research_reference(selected)
    if selected in ['Age decay + Direction', 'Age decay + Issuer'] and reference not in [None, expected]:
        raise ValueError(f'{selected} must compare directly with Age decay')
    reference = expected if reference is None else reference
    if selected not in wide or reference not in wide:
        raise ValueError(f'Need saved {selected} and {reference} validation predictions')
    rows = rows.copy()
    # CORE LOGIC: STEP 2 — Attach same-row model losses and aligned diagnostic states
    # Input: saved selected errors=[1,3], reference errors=[2,2] on row_ids=[7,9]
    # Output: loss_delta=[-1,1] on those same row IDs; n=2; groups/daily initialized empty
    rows['day'] = to_ny_datetime(rows.time).dt.normalize()
    rows['selected_error'], rows['reference_error'] = wide[selected], wide[reference]
    rows['loss_delta'] = rows.selected_error - rows.reference_error
    states, definitions = _states(rows)
    n = len(rows)
    groups, daily = [], []
    for family, state in states.items():
        # CORE LOGIC: STEP 3 — Aggregate every state and date within each family
        # Input: quote_state=['No quote','Two-sided'], loss_delta=[-1,1], overall_n=2 on one date
        # Output: two quote_state groups, each n=1 with contributions -0.5 and +0.5 bps; family contributions sum to 0
        rows[family] = state
        for label, group in rows.groupby(family, observed=True, sort=True):
            groups.append(dict(family=family, state=label, **_metrics(group, n)))
        for (label, day), group in rows.groupby([family, 'day'], observed=True, sort=True):
            daily.append(dict(family=family, state=label, day=day, **_metrics(group, n)))
    by_day = pd.DataFrame([dict(day=day, **_metrics(group, n)) for day, group in rows.groupby('day', sort=True)])
    sums, counts = by_day.total_loss_delta.to_numpy(), by_day.n.to_numpy()
    # CORE LOGIC: STEP 4 — Evaluate fixed-prediction date sensitivity and final summary
    # Input: two ET dates each have one row with loss_delta=[-1,3]
    # Output: overall delta=1; day_equal_delta=1; leave-one-date-out values=[3,-1], lodo_min=-1, lodo_max=3
    # Trick: LODO removes a date from fixed predictions; it is neither refitting nor a confidence interval.
    lodo = (sums.sum() - sums) / (n - counts) if len(by_day) > 1 else np.array([])
    overall = pd.Series(_metrics(rows, n))
    overall['day_equal_delta'] = by_day.delta_mae.mean()
    overall['lodo_min'] = lodo.min() if len(lodo) else np.nan
    overall['lodo_max'] = lodo.max() if len(lodo) else np.nan
    return dict(selected=selected, reference=reference, n=n, rows=rows,
                overall=overall, groups=pd.DataFrame(groups), daily=pd.DataFrame(daily),
                # CORE LOGIC: STEP 5 — Return complete validation diagnostics with provenance and limits
                # Input: selected='Age decay', reference='Reliability'; validated rows (row_id=7, selected_error=1, reference_error=2) and (row_id=9, selected_error=3, reference_error=2) on Mar2
                # Output: result n=2; rows loss_delta=[-1,1]; overall delta_mae=0; overall_daily Mar2 n=2/delta_mae=0; source says no fitting
                overall_daily=by_day, definitions=definitions,
                source='saved validation predictions + completed feature frame; no fitting',
                limitations='Families overlap: do not sum across families. P95 differences are model-tail differences, not P95 of paired loss. LODO is fixed-prediction sensitivity, not a confidence interval or refit.')


# SETUP LOGIC: Declare figure; existing arguments and docstring are preserved
def figure(result):
    """A bounded 2x3 view; complete numbers remain in the export tables."""
    # PLOTTING LOGIC: STEP 1 — Draw the bounded 2x3 state dashboard; all computed groups remain in result
    fig = Figure(figsize=(19, 13), facecolor='white')
    axes = fig.subplots(2, 3).flat
    fig.subplots_adjust(left=.15, right=.97, bottom=.11, top=.84, wspace=.65, hspace=.5)
    fig.suptitle(f"Saved validation | {result['selected']} minus {result['reference']}\nFinite state diagnostics; identical targets; no training", fontsize=17)
    o = result['overall']
    fig.text(.5, .91, f"n={result['n']:,} | {int(o.dates)} dates | MAE delta={o.delta_mae:+.4f} bps | date-equal={o.day_equal_delta:+.4f} | LODO [{o.lodo_min:+.4f}, {o.lodo_max:+.4f}]", ha='center', fontsize=11)
    groups = result['groups']
    panels = [(['quote_state', 'dealer_support', 'peer_support'], 'Quote and dealer support'),
              (['bid_message_age', 'bid_change_age', 'ask_message_age', 'ask_change_age'], 'Message / observed-change age'),
              (['bid_candidate_gap', 'ask_candidate_gap', 'bid_dispersion', 'ask_dispersion'], 'Candidate / dealer disagreement'),
              (['paired_support', 'bid_zero_quantity', 'ask_zero_quantity', 'bid_unknown_quantity', 'ask_unknown_quantity'], 'Quantity / pairing support'),
              (['SECTOR', 'trade_size', 'trade_type', 'issuer_trade_count', 'anchor_age'], 'Largest absolute loss contributions')]
    # PLOTTING LOGIC: STEP 2 — Show at most 16 states per panel ordered by absolute contribution before label ordering
    for ax, (families, title) in zip(axes, panels):
        table = groups.loc[groups.family.isin(families)].copy()
        available = len(table)
        if len(table) > 16:
            table = table.assign(order=table.contribution_bps.abs()).sort_values(['order', 'family', 'state'], ascending=[False, True, True], kind='stable').head(16)
        table = table.sort_values(['family', 'state'], kind='stable')
        y = np.arange(len(table))
        bars = ax.barh(y, table.contribution_bps, color=np.where(table.contribution_bps.le(0), '#287D8E', '#BD5367'))
        for bar, worse in zip(bars, table.delta_p95.gt(0)):
            if worse: bar.set_hatch('///')
        ax.set_yticks(y, [f'{r.family}: {r.state} (n={r.n:,})' for r in table.itertuples()], fontsize=8)
        ax.invert_yaxis(); ax.axvline(0, color='#777', lw=.8)
        shown = f'\nTop {len(table)}/{available} by |contribution|' if len(table) < available else ''
        ax.set_title(title + shown + '\nSum of paired loss delta / overall N', fontsize=11)
        ax.set_xlabel('Contribution to overall MAE delta (bps)'); ax.grid(axis='x', alpha=.15)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax = list(fig.axes)[-1]
    daily = result['overall_daily']
    ax.plot(daily.day, daily.delta_mae, marker='o', color='#287D8E', label='Daily MAE delta')
    ax.set_xticks(daily.day)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d', tz='America/New_York'))
    ax.axhline(0, color='#777', lw=.8); ax.tick_params(axis='x', rotation=30)
    ax.set_title('Fixed predictions by ET trade date'); ax.set_ylabel('bps; negative improves'); ax.grid(alpha=.15)
    fig.text(.05, .055, 'Decision: separate available support, selection costs and loss contributions before adding a rule or feature. Missing support is explicit.', fontsize=11)
    fig.text(.05, .025, 'Fixed cuts; families overlap. If >16 states, display the 16 largest |contributions| only; full groups are unchanged. Hatching = worse P95. Few dates do not establish stability.', fontsize=9)
    return fig


# SETUP LOGIC: Declare sector_tail_figure; existing arguments and docstring are preserved
def sector_tail_figure(result):
    # PLOTTING LOGIC: STEP 1 — Require the direct Age decay reference and render all sector/size/type tails
    """SECTOR / size / type mean and tail costs, readable without opening tables."""
    if result['selected'] in ['Age decay + Direction', 'Age decay + Issuer'] and result['reference'] != 'Age decay':
        raise ValueError('New research models must compare directly with Age decay')
    groups = result['groups']
    sectors = groups.loc[groups.family.eq('SECTOR')].sort_values(['n', 'state'], ascending=[False, True], kind='stable')
    fig = Figure(figsize=(19, max(13, 8 + .3 * len(sectors))), facecolor='white')
    a, b, c, d, e, f = fig.subplots(2, 3).flat
    fig.subplots_adjust(left=.15, right=.97, bottom=.12, top=.83, hspace=.55, wspace=.55)
    fig.suptitle(f"Saved validation | {result['selected']} minus {result['reference']}\nSECTOR, size and trade-type loss / tail costs; identical target rows", fontsize=17)
    o = result['overall']
    fig.text(.5, .915, f"n={result['n']:,} | {int(o.dates)} ET dates | MAE delta={o.delta_mae:+.4f} bps | P95 delta={o.delta_p95:+.4f} bps | >10bps share delta={o.delta_gt10:+.2%}", ha='center', fontsize=11)
    fig.text(.5, .88, f"Date-equal MAE delta={o.day_equal_delta:+.4f} bps | leave-one-date-out [{o.lodo_min:+.4f}, {o.lodo_max:+.4f}] bps; fixed-prediction sensitivity", ha='center', fontsize=10)
    # SETUP LOGIC: Declare labels; existing arguments and docstring are preserved
    def labels(table):
        # PLOTTING LOGIC: STEP 1 — Format the existing figure or compact count labels
        return [f'{r.state} (n={r.n:,}{"*" if r.n < 200 else ""}, d={r.dates})' for r in table.itertuples()]
    # PLOTTING LOGIC: STEP 2 — Render sector mean, P95 and >10bps-share panels with every sector retained
    for ax, column, title, unit in [(a, 'delta_mae', 'SECTOR mean loss delta', 'bps'),
                                   (b, 'delta_p95', 'SECTOR model P95 delta', 'bps'),
                                   (c, 'delta_gt10', 'SECTOR >10bps error-share delta', 'percentage points')]:
        y = np.arange(len(sectors))
        values = sectors[column] * (100 if column == 'delta_gt10' else 1)
        ax.barh(y, values, color=np.where(values.le(0), '#287D8E', '#BD5367'))
        for position, value in zip(y, values):
            ax.text(.98, position, f'{value:+.4f}', transform=ax.get_yaxis_transform(),
                    ha='right', va='center', fontsize=7, bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=.5))
        ax.set_yticks(y, labels(sectors), fontsize=8); ax.invert_yaxis()
        ax.axvline(0, color='#777', lw=.8); ax.grid(axis='x', alpha=.15)
        ax.set_title(title, fontsize=12); ax.set_xlabel(unit + '; negative improves')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4)); ax.tick_params(axis='x', labelsize=8)
    for ax, family, title in [(d, 'trade_size', 'TRACE par size'), (e, 'trade_type', 'Target trade type')]:
        table = groups.loc[groups.family.eq(family)].sort_values('state', kind='stable')
        y = np.arange(len(table))
        ax.barh(y - .18, table.delta_mae, height=.35, color='#287D8E', label='MAE delta')
        ax.barh(y + .18, table.delta_p95, height=.35, color='#BA8C27', label='Model P95 delta')
        ax.set_yticks(y, labels(table), fontsize=9); ax.invert_yaxis()
        ax.axvline(0, color='#777', lw=.8); ax.grid(axis='x', alpha=.15)
        ax.set_title(title + ': mean and tail', fontsize=12); ax.set_xlabel('bps; negative improves')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4)); ax.tick_params(axis='x', labelsize=8)
        ax.legend(fontsize=9)
    daily = result['overall_daily']
    f.plot(daily.day, daily.delta_mae, marker='o', color='#287D8E', label='Daily MAE delta')
    f.axhline(0, color='#777', lw=.8); f.grid(alpha=.15); f.set_xticks(daily.day)
    f.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d', tz='America/New_York'))
    f.set_title('Overall loss delta by ET date'); f.set_ylabel('bps; negative improves')
    f.tick_params(axis='x', rotation=30)
    fig.text(.05, .065, 'Decision: retain meaningful mean improvement while recording sector / size / type tail costs; no sector-specific deletion follows from this chart.', fontsize=11)
    fig.text(.05, .035, '* n<200 is a descriptive small-group reminder, not an eligibility threshold. P95 is each model\'s tail quantile; it is not the P95 of paired loss differences.', fontsize=10)
    fig.text(.05, .015, 'Only saved Validation predictions are used. No test predictions, fitting or event rebuilding; few dates and small groups do not establish stable future gains.', fontsize=10)
    return fig


# SETUP LOGIC: Declare show_validation_diagnostics; existing arguments and docstring are preserved
def show_validation_diagnostics(frame=None, predictions=None, selected='Age decay', reference=None,
                                folder='outputs/quote_quality_step5', output_folder='outputs/quote_quality_diagnostics',
                                parquet=False):
    """Review live completed objects or saved validation; save a fresh full PNG and tables."""
    # CACHEING LOGIC: STEP 1 — Load only saved validation when live completed objects are not supplied
    if (frame is None) != (predictions is None):
        raise ValueError('Supply both completed frame and validation predictions, or neither')
    if frame is None:
        from quote_quality_saved import load_saved_validation
        frame, predictions, _ = load_saved_validation(folder)
    # CORE LOGIC: STEP 1 — Compute validation diagnostics from completed objects without fitting
    # Input: completed matching rows (row_id=7,time=Mar2 10:00 ET,target=5) and (row_id=9,time=Mar2 11:00 ET,target=5); Age decay errors=[1,3], Reliability errors=[2,2]
    # Output: result selected='Age decay', reference='Reliability', n=2, paired loss_delta=[-1,1], overall delta_mae=0
    result = diagnostics(frame, predictions, selected, reference)
    # FILE IO LOGIC: STEP 3 — Create a fresh UUID export directory and save complete numerical tables and metadata
    target = Path(output_folder) / uuid4().hex
    target.mkdir(parents=True, exist_ok=False)
    for name in ['groups', 'daily', 'overall_daily', 'definitions']:
        result[name].to_csv(target / f'{name}.csv', index=False)
        if parquet:
            result[name].to_parquet(target / f'{name}.parquet', index=False)
    result['overall'].to_csv(target / 'overall.csv', header=['value'])
    metadata = {k: result[k] for k in ['selected', 'reference', 'n', 'source', 'limitations']}
    (target / 'diagnostics.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    # PLOTTING LOGIC: STEP 4 — Export both full PNG figures and display the state dashboard
    path = target / 'diagnostics.png'
    figure(result).savefig(path, dpi=160)
    tail_path = target / 'sector_tails.png'
    sector_tail_figure(result).savefig(tail_path, dpi=160)
    result['output_folder'], result['png_path'], result['tail_png_path'] = target, path, tail_path
    from IPython.display import display, Image
    display(Image(filename=str(path)))
    return result


# SETUP LOGIC: Declare fixed_trade_case_manifest; existing arguments and docstring are preserved
def fixed_trade_case_manifest(frame, impacts, n_each=6, existing=None, seed=2026,
                              manifest_path=None, quote_start=None, quote_end=None):
    """Freeze bond/side/day navigation from saved trade queries; never reads raw quotes.

    No dealer identity exists in aggregate model features: firm stays missing and
    the local notebook must choose it from that bond/day's existing source state.
    An existing manifest is kept, never silently replaced. Cases are not rates.
    """
    # CACHEING LOGIC: Preserve an existing in-memory manifest instead of resampling.
    if existing is not None:
        return existing.copy()
    # FILE IO LOGIC: Read an existing CSV without rewriting the frozen file.
    path = Path(manifest_path) if manifest_path is not None else None
    if path is not None and path.is_file():
        kept = pd.read_csv(path)
        # CORE LOGIC: STEP 1 — Restore each frozen case date in New York time.
        # Input: day=['2026-03-01T00:00:00-05:00','2026-03-20T00:00:00-04:00','2026-03-02'].
        # Output: day=[2026-03-01 00:00-05:00,2026-03-20 00:00-04:00,2026-03-02 00:00-05:00], all America/New_York.
        # Trick: Scalar parsing avoids mixed DST offsets producing object dtype; naive dates remain local wall time.
        if 'day' in kept:
            kept['day'] = kept.day.map(lambda value: to_ny_datetime(pd.Series([value])).iloc[0])
        # CACHEING LOGIC: Return the frozen identities and local provenance without resampling.
        kept.attrs['source'] = 'existing frozen manifest; not overwritten'
        return kept
    # CORE LOGIC: STEP 2 — Validate completed query keys and measured impact uniqueness
    # Input: frame row_id=[7,7] with cusip/time columns; n_each=6
    # Output: ValueError for non-unique completed row IDs; no raw quotes or training are accessed
    if n_each < 1:
        raise ValueError('n_each must be positive')
    required = {'row_id', 'cusip', 'time'}
    if not required.issubset(frame) or frame.row_id.duplicated().any():
        raise ValueError('Need completed features with unique row_id and cusip/time')
    impact_required = {'cusip', 'side', 'day', 'max_center_effect_bps', 'coverage_loss_fraction'}
    if not impact_required.issubset(impacts):
        raise ValueError('Need saved trade-query impact keys and measured center / coverage effects')
    # CORE LOGIC: STEP 3 — Prepare narrow saved query features and impact provenance
    # Input: impacts=[(cusip=X,side=bid,day=Mar2 10:00 ET,max_center_effect_bps=2,coverage_loss_fraction=0.1)]; frame row=(row_id=7,cusip=X,time=Mar2 10:00 ET,bcq_bid_n_dealers=2)
    # Output: measured day=Mar2 00:00 ET; copied rows retain row_id=7,cusip=X,time and bcq_bid_n_dealers=2; extra model columns are not copied
    # Trick: Duplicate bond/side/day impacts fail before the join, preventing accidental multiplication of case rows.
    measured = impacts.copy()
    measured['day'] = to_ny_datetime(measured.day).dt.normalize()
    if measured.duplicated(['cusip', 'side', 'day']).any():
        raise ValueError('Impact table must have one row per bond/side/day')
    source = impacts.attrs.get('query_source', 'supplied trade-query impacts; provenance not independently verified')
    columns = ['row_id', 'cusip', 'time', 'ISSUER', 'SECTOR'] + [f'bcq_{side}_{suffix}'
        for side in ['bid', 'ask'] for suffix in ['n_dealers', 'mean_candidate_gap', 'unknown_quantity_fraction']]
    rows = frame[[c for c in columns if c in frame]].copy()
    rows['day'] = to_ny_datetime(rows.time).dt.normalize()
    # CORE LOGIC: STEP 4 — Validate the stated quote-file date interval.
    # Input: quote_start='2026-03-01T00:00-05:00', quote_end='2026-04-01T00:00-04:00'.
    # Output: start=2026-03-01 00:00-05:00, end=2026-04-01 00:00-04:00; malformed or reversed bounds reject.
    # Trick: Parse bounds separately across DST; naive values remain New York wall time. This window only bounds navigation.
    start = quote_start if quote_start is not None else impacts.attrs.get('start', measured.day.min())
    end = quote_end if quote_end is not None else impacts.attrs.get('end', measured.day.max())
    start = to_ny_datetime(pd.Series([start])).dt.normalize().iloc[0]
    end = to_ny_datetime(pd.Series([end])).dt.normalize().iloc[0]
    if pd.isna(start) or pd.isna(end):
        raise ValueError('Need quote-file dates or a nonempty supplied impact window')
    if start > end:
        raise ValueError('Invalid quote-file window')
    # CORE LOGIC: STEP 5 — Retain eligible navigation dates and normalize blank metadata.
    # Input: rows=[(X,2026-01-01,ISSUER='A',SECTOR='Energy'),(X,2026-03-02,ISSUER=' ',SECTOR='Energy')]; window=Mar1-Apr1 ET.
    # Output: rows=[(X,2026-03-02,ISSUER=NA,SECTOR='Energy')].
    # Trick: A missing CUSIP cannot identify a local case; empty metadata stays missing until grouped display labels.
    rows = rows.loc[rows.day.between(start, end) & rows.cusip.notna()].copy()
    for col, missing in [('ISSUER', '[Missing issuer]'), ('SECTOR', 'Unknown')]:
        rows[col] = rows[col].astype('string').str.strip().replace('', pd.NA) if col in rows else pd.NA
    # CORE LOGIC: STEP 6 — Create per-side known-support indicators from saved counts
    # Input: one bond/day has bid n_dealers=[0,2,None]
    # Output: quote_present=[0.0,1.0,NaN], support_known=[True,True,False], dealer_count=[0,2,NaN]
    parts = []
    for side in ['bid', 'ask']:
        z = rows[['cusip', 'day', 'ISSUER', 'SECTOR']].copy()
        count = _numeric(rows, f'bcq_{side}_n_dealers')
        z['quote_present'] = count.gt(0).astype(float).where(count.ge(0))
        z['support_known'] = count.ge(0)
        z['dealer_count'] = count.where(count.ge(0))
        z['candidate_gap'] = _numeric(rows, f'bcq_{side}_mean_candidate_gap')
        z['unknown_quantity'] = _numeric(rows, f'bcq_{side}_unknown_quantity_fraction')
        g = z.groupby(['cusip', 'day'], observed=True, sort=True)
        # CORE LOGIC: STEP 7 — Reduce trade-query support and resolve conflicting metadata
        # Input: X/Mar2 has quote_present=[0,1,NaN], issuer=['A','A','A'], sector=['Energy','Utilities',None]
        # Output: trade_queries=3, known_support_queries=2, quote_query_fraction=0.5, ISSUER='A', SECTOR='Conflicting'
        # Trick: Unknown counts do not enter the known-support mean; its denominator is reported separately from total queries.
        c = g.agg(trade_queries=('quote_present', 'size'), known_support_queries=('support_known', 'sum'),
                  quote_query_fraction=('quote_present', 'mean'),
                  mean_dealer_count=('dealer_count', 'mean'), mean_candidate_gap=('candidate_gap', 'mean'),
                  unknown_quantity_fraction=('unknown_quantity', 'mean')).reset_index()
        for col, missing, conflict in [('ISSUER', '[Missing issuer]', '[Conflicting issuer]'), ('SECTOR', 'Unknown', 'Conflicting')]:
            values, distinct = g[col].first(), g[col].nunique()
            c[col] = values.fillna(missing).mask(distinct.gt(1), conflict).to_numpy()
        c['side'] = side
        parts.append(c)
    # CORE LOGIC: STEP 8 — Assign stable bond/side/day case IDs and measured effects
    # Input: X/bid/Mar2 has 3 saved eligible trade queries, seed=2026, one matching center-effect=2 bps impact
    # Output: activity='Sparse'; case_id='X|bid|2026-03-02T00:00:00-05:00'; hash=386cef0ea54386623fd4f439eb08442f0ef687ea79beeeafeb8ea651017b56fb; joined center-effect=2
    # Trick: Identities do not depend on source row order or validation errors.
    candidates = pd.concat(parts, ignore_index=True)
    candidates['activity'] = pd.cut(candidates.trade_queries, [0, 3, 30, np.inf], labels=['Sparse', 'Medium', 'Active']).astype('string')
    candidates['case_id'] = ['|'.join([str(r.cusip), r.side, r.day.isoformat()]) for r in candidates.itertuples()]
    candidates['stable_hash'] = candidates.case_id.map(lambda key: sha256(f'{seed}|{key}'.encode()).hexdigest())
    candidates = candidates.merge(measured, on=['cusip', 'side', 'day'], how='left', validate='one_to_one')
    # CORE LOGIC: STEP 9 — Rank typical cases by normalized distance to stratum medians
    # Input: one stratum support metric=[0,0.5,1] and other metrics constant
    # Output: median broadcasts as [0.5,0.5,0.5]; scale=1; distance=[0.5,0,0.5]
    # Trick: transform keeps original row alignment; constant metric ranges become 1 and unavailable distances contribute 0.
    metrics = ['quote_query_fraction', 'mean_dealer_count', 'mean_candidate_gap', 'unknown_quantity_fraction']
    med = candidates.groupby(['SECTOR', 'activity'], observed=True)[metrics].transform('median')
    distance = (candidates[metrics] - med).abs()
    strata_metrics = candidates.groupby(['SECTOR', 'activity'], observed=True)[metrics]
    scale = strata_metrics.transform('max') - strata_metrics.transform('min')
    candidates['typical_distance'] = distance.div(scale.replace(0, 1)).fillna(0).sum(axis=1)
    used, dealer_issuer_counts = set(), {}
    # SETUP LOGIC: Declare pick; existing arguments and docstring are preserved
    def pick(pool, requirements=()):
        # SETUP LOGIC: STEP 1 — Initialize a local selected-index list for the balanced picker
        selected = []
        # SETUP LOGIC: Declare take; existing arguments and docstring are preserved
        def take(row, enforce=True):
            # CORE LOGIC: STEP 1 — Take distinct cases with a soft issuer cap
            # Input: used={'X|bid|Mar2'}, issuer A already appears twice; next row is new A case; enforce=True
            # Output: take returns False without changing used; enforce=False can accept it
            # Trick: Aggregate saved features lack dealer IDs, so only the issuer cap is assessable here.
            if row.case_id in used or (enforce and dealer_issuer_counts.get(row.ISSUER, 0) >= 2):
                return False
            selected.append(row.Index); used.add(row.case_id)
            dealer_issuer_counts[row.ISSUER] = dealer_issuer_counts.get(row.ISSUER, 0) + 1
            return True
        # CORE LOGIC: STEP 1 — Reserve rare support states before sector/activity traversal
        # Input: requirements include quote_query_fraction==0; candidate NoQuote1 is eligible and unused
        # Output: NoQuote1 is selected once; strata are formed from sector and saved-query activity
        for mask in requirements:
            for row in pool.loc[mask(pool)].itertuples():
                if len(selected) < n_each and take(row): break
        strata = list(pool.groupby(['SECTOR', 'activity'], observed=True, sort=True))
        # CORE LOGIC: STEP 2 — Traverse strata until filled or no additional distinct case remains
        # Input: Energy candidates=[E1,E2], Utilities=[U1,U2], n_each=2; all unused and uncapped
        # Output: selected=[E1,U1]; returns two copied candidate rows
        # Trick: A fallback pass relaxes the issuer cap; lack of progress stops an undersized pool without repeats.
        for enforce in [True, False]:
            while len(selected) < n_each:
                before = len(selected)
                for _, group in strata:
                    for row in group.itertuples():
                        if take(row, enforce): break
                    if len(selected) == n_each: break
                if len(selected) == before: break
        return pool.loc[selected].copy()
    # CORE LOGIC: STEP 10 — Freeze seeded random and within-stratum typical samples
    # Input: n_each=1; candidates has one row X/bid/Mar2 (SECTOR=Energy, activity=Sparse, quote_query_fraction=0, trade_queries=2, unknown_quantity_fraction=0)
    # Output: Random=[X/bid/Mar2]; Typical=[] because the only case ID is already used
    random = pick(candidates.sort_values('stable_hash'), [lambda x: x.quote_query_fraction.eq(0),
                  lambda x: x.trade_queries.le(3), lambda x: x.unknown_quantity_fraction.gt(0)])
    random['selection'] = 'Random'; random['selection_reason'] = 'seeded sector/activity strata; no-quote/sparse/unknown support included when available'
    typical = pick(candidates.sort_values(['typical_distance', 'stable_hash'], kind='stable'))
    typical['selection'] = 'Typical'; typical['selection_reason'] = 'nearest within-stratum median saved query support/ambiguity profile'
    # CORE LOGIC: STEP 11 — Select high impact by actual center or coverage effects
    # Input: n_each=2, used={A,D,E,F}; remaining quoted B and C have center_effect=[2,1], coverage_loss=[0.2,0.1], distinct issuers; A has quote_fraction=0
    # Output: High impact=[B,C] with impact_score=[2,1]; A excluded despite any recorded gap
    # Trick: Raw gap cannot qualify a high case, and a missing measured effect is not replaced with zero.
    high = candidates.loc[candidates.quote_query_fraction.gt(0) &
        (candidates.max_center_effect_bps.gt(1e-9) | candidates.coverage_loss_fraction.gt(0))].copy()
    high['impact_score'] = high.max_center_effect_bps.rank(pct=True) + high.coverage_loss_fraction.rank(pct=True)
    high = pick(high.sort_values(['impact_score', 'stable_hash'], ascending=[False, True], kind='stable'))
    high['selection'] = 'High impact'; high['selection_reason'] = 'measured rule center / coverage effect on supplied queries'
    # CORE LOGIC: STEP 12 — Record the navigation-only source and missing dealer identity
    # Input: manifest contains 6 Random, 6 Typical, 2 High impact cases from supplied trade queries
    # Output: case_number=1..14; firm is NA; source columns identify aggregate-query support and navigation-only scope
    manifest = pd.concat([random, typical, high], ignore_index=True)
    manifest['case_number'] = np.arange(1, len(manifest) + 1)
    manifest['firm'] = pd.NA
    manifest['dealer_source'] = 'not retained in aggregate features; choose locally from cached bond/day'
    manifest['quote_support_source'] = 'fraction of supplied queries with known side count; missing count is unassessed'
    manifest['impact_source'] = source
    manifest['candidate_source'] = 'supplied eligible trade-query frame within stated window'
    manifest['scope_flag'] = 'navigation sample only; no population-rate claim'
    # CORE LOGIC: STEP 13 — Report soft-cap relaxation and preserve window/provenance flags
    # Input: issuer A occurs 3 times; explicit quote dates Mar1-Mar31; seed=2026
    # Output: A cases issuer_cap_relaxed=True; window_source='explicit quote dates'; attrs population_rate=False
    # Trick: Neither sample counts nor query fractions should be presented as population event rates.
    manifest['issuer_cap_relaxed'] = manifest.ISSUER.map(manifest.ISSUER.value_counts()).gt(2)
    manifest['window_start'], manifest['window_end'] = start, end
    manifest['window_source'] = 'explicit quote dates' if quote_start is not None and quote_end is not None else 'impact metadata or supplied impact-day range'
    manifest.attrs.update(seed=seed, source=source, population_rate=False, dealer_cap='not assessable without dealer-level state')
    # FILE IO LOGIC: Persist only a newly created manifest with exclusive file creation.
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            manifest.to_csv(stream, index=False)
    # CORE LOGIC: STEP 14 — Return the frozen navigation manifest
    # Input: manifest rows: (case_number=1,selection='Random',cusip='X',side='bid',day=Mar2,ISSUER='A',SECTOR='Energy',firm=NA)
    # Output: return the same one-row manifest after optional exclusive CSV export
    return manifest
