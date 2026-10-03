"""Read existing validation results and slice target-trade SECTOR. Never trains."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def direct_reference(selected):
    return 'Base' if selected in ['Base', 'Quote levels'] else ('Quote levels' if selected == 'Reliability' else 'Reliability')


def validation_rows(frame, predictions):
    """Reject stale joins and unequal cohorts before drawing any improvement."""
    frame_required = {'row_id', 'time', 'split', 'BM_SPREAD'}
    if not frame_required.issubset(frame):
        raise ValueError('Missing saved feature columns: ' + ', '.join(sorted(frame_required - set(frame))))
    if frame.row_id.duplicated().any():
        raise ValueError('Feature frame row_id must be unique')
    required = {'row_id', 'model', 'stage', 'time', 'abs_error_bps', 'pred_spread', 'error_bps'}
    if not required.issubset(predictions):
        raise ValueError('Missing validation prediction columns: ' + ', '.join(sorted(required - set(predictions))))
    if predictions.empty or not predictions.stage.eq('Validation').all():
        raise ValueError('This viewer accepts saved Validation only; it never opens test predictions')
    if predictions.duplicated(['row_id', 'model']).any():
        raise ValueError('Duplicate model / row_id predictions')
    if not np.isfinite(predictions.abs_error_bps).all() or predictions.abs_error_bps.lt(0).any():
        raise ValueError('Invalid validation absolute errors')
    wide = predictions.pivot(index='row_id', columns='model', values='abs_error_bps')
    if wide.isna().any().any():
        raise ValueError('Every model must evaluate exactly the same target rows')
    ids = wide.index
    if not ids.isin(frame.row_id).all():
        raise ValueError('Prediction row_id is absent from the saved feature frame')
    rows = frame.set_index('row_id').loc[ids].copy()
    if 'split' in rows and not rows['split'].eq('Validation').all():
        raise ValueError('Predictions do not join to Validation rows')
    if 'split' in frame and set(frame.loc[frame['split'].eq('Validation'), 'row_id']) != set(ids):
        raise ValueError('Saved predictions omit validation target rows')
    joined = predictions.merge(frame[['row_id', 'time']].rename(columns={'time': 'frame_time'}), on='row_id', validate='many_to_one')
    if not pd.to_datetime(joined.time).eq(pd.to_datetime(joined.frame_time)).all():
        raise ValueError('Saved prediction / feature timestamps differ')
    if {'pred_spread', 'error_bps'}.issubset(predictions) and 'BM_SPREAD' in rows:
        truth = predictions.row_id.map(rows.BM_SPREAD)
        if not np.allclose((predictions.pred_spread - truth) * 100, predictions.error_bps, rtol=1e-9, atol=1e-7):
            raise ValueError('Saved prediction errors disagree with the target or bps multiplier')
        if not np.allclose(predictions.error_bps.abs(), predictions.abs_error_bps, rtol=1e-9, atol=1e-7):
            raise ValueError('Saved absolute errors disagree with signed errors')
    return rows, wide


def sector_diagnostics(frame, predictions, selected='Quote levels'):
    rows, wide = validation_rows(frame, predictions)
    if 'SECTOR' not in rows:
        raise ValueError('Target-trade SECTOR is missing from model_features; do not substitute a CUSIP mapping')
    reference = direct_reference(selected)
    if selected not in wide or reference not in wide:
        raise ValueError(f'Need saved {selected} and {reference} predictions')
    # Row-level target metadata; no whole-window CUSIP sector mapping here.
    rows['sector'] = rows.SECTOR.astype('string').str.strip().replace('', pd.NA).fillna('Unknown')
    rows['day'] = pd.to_datetime(rows.time).dt.normalize()
    rows['selected_error'] = wide[selected]
    rows['reference_error'] = wide[reference]
    rows['loss_delta'] = rows.selected_error - rows.reference_error
    groups = rows.groupby('sector', observed=True, sort=True)
    summary = groups.agg(n=('loss_delta', 'size'), dates=('day', 'nunique'),
                         selected_mae=('selected_error', 'mean'), reference_mae=('reference_error', 'mean'),
                         delta_mae=('loss_delta', 'mean'))
    summary['delta_p95'] = groups.selected_error.quantile(.95) - groups.reference_error.quantile(.95)
    summary['delta_gt10'] = groups.selected_error.apply(lambda x: x.gt(10).mean()) - groups.reference_error.apply(lambda x: x.gt(10).mean())
    daily = rows.groupby(['sector', 'day'], observed=True).loss_delta.mean().unstack('day')
    daily_n = rows.groupby(['sector', 'day'], observed=True).size().unstack('day', fill_value=0)
    summary['day_equal_delta'] = daily.mean(axis=1)
    summary['worse_dates'] = daily.gt(0).sum(axis=1)
    coverage = pd.DataFrame(index=summary.index)
    for label, column in [('Any quote', 'bcq_has_quote'), ('Fresh pair', 'bcq_n_pair'), ('Size-time pair', 'bcq_n_size_time_pair')]:
        if column in rows:
            coverage[label] = rows[column].gt(0).groupby(rows.sector).mean()
    masks = {'All validation': pd.Series(True, index=rows.index)}
    if 'bcq_has_quote' in rows:
        masks.update({'No quote': rows.bcq_has_quote.eq(0), 'Covered': rows.bcq_has_quote.gt(0)})
    if 'QUANTITY' in rows:
        q = pd.to_numeric(rows.QUANTITY, errors='coerce')
        masks.update({'<=100K par': q.le(100_000), '>=1MM par': q.ge(1_000_000)})
    slices = pd.DataFrame([{'slice': name, 'n': int(mask.sum()), 'delta_mae': rows.loc[mask, 'loss_delta'].mean()}
                           for name, mask in masks.items()]).set_index('slice')
    all_daily = rows.groupby('day').loss_delta.mean()
    leave_one = [rows.loc[rows.day.ne(day), 'loss_delta'].mean() for day in all_daily.index] if len(all_daily) > 1 else []
    return dict(selected=selected, reference=reference, summary=summary, daily=daily, daily_n=daily_n,
                coverage=coverage, slices=slices, n=len(rows), date_n=len(all_daily),
                delta=float(rows.loss_delta.mean()), day_equal_delta=float(all_daily.mean()),
                leave_one_date_range=(min(leave_one), max(leave_one)) if leave_one else (np.nan, np.nan))


def sector_figure(result):
    summary = result['summary'].sort_values('n', ascending=False)
    names = summary.index
    fig = Figure(figsize=(18, max(11, 6 + len(names) * .3)), facecolor='white')
    a, b, c, d = fig.subplots(2, 2).flat
    fig.subplots_adjust(top=.85, bottom=.15, wspace=.42, hspace=.45, left=.16)
    fig.suptitle(f"Saved validation | {result['selected']} minus {result['reference']}\nTarget-trade SECTOR; identical target rows; no fitting", fontsize=16)
    fig.text(.5, .90, f"n={result['n']:,} | {result['date_n']} ET trade dates | MAE delta={result['delta']:+.4f} bps | date-equal delta={result['day_equal_delta']:+.4f} bps", ha='center')
    y = np.arange(len(names))
    labels = [f"{s} (n={int(summary.loc[s, 'n']):,}, d={int(summary.loc[s, 'dates'])})" for s in names]
    a.barh(y, summary.delta_mae, color=np.where(summary.delta_mae.le(0), '#277F8E', '#C9563D'))
    a.set_yticks(y, labels); a.invert_yaxis(); a.axvline(0, color='#777777')
    a.set_title('Same-row sector MAE delta'); a.set_xlabel('bps; negative improves')
    coverage = result['coverage'].reindex(names)
    for j, name in enumerate(coverage):
        b.barh(y + (j - (len(coverage.columns) - 1) / 2) * .22, coverage[name], height=.22, label=name)
    b.set_yticks(y, names); b.invert_yaxis(); b.set_xlim(0, 1.05); b.legend(fontsize=8)
    b.set_title('Availability / all sector validation trades'); b.set_xlabel('Fraction; no target rows removed')
    daily = result['daily'].reindex(names)
    values = daily.to_numpy(dtype=float)
    bound = max(float(np.nanmax(np.abs(values))), .001)
    im = c.imshow(np.ma.masked_invalid(values), cmap='RdBu_r', vmin=-bound, vmax=bound, aspect='auto')
    c.set_yticks(y, names); c.set_xticks(np.arange(len(daily.columns)), [x.strftime('%m-%d') for x in daily.columns], rotation=25)
    c.set_title('Sector × ET date delta; blank = unassessed')
    if values.size <= 120:
        for i in range(len(names)):
            for j, day in enumerate(daily.columns):
                if np.isfinite(values[i, j]):
                    c.text(j, i, f"{values[i,j]:+.3f}\nn={int(result['daily_n'].loc[names[i], day]):,}", ha='center', va='center', fontsize=7)
    fig.colorbar(im, ax=c, label='MAE delta (bps)', fraction=.04)
    d.barh(y, summary.delta_p95, color='#8560A5'); d.set_yticks(y, names); d.invert_yaxis(); d.axvline(0, color='#777777')
    d.set_title('Sector pooled P95 absolute-error delta'); d.set_xlabel('bps; separate from mean loss')
    lo, hi = result['leave_one_date_range']
    fig.text(.08, .085, f"Dropping each date in turn: overall delta range [{lo:+.4f}, {hi:+.4f}] bps. This is a sensitivity diagnostic, not a confidence interval.", fontsize=10)
    slices = result['slices']
    fig.text(.08, .055, ' | '.join(f"{name}: {r.delta_mae:+.4f} bps (n={int(r.n):,})" for name, r in slices.iterrows()), fontsize=9)
    fig.text(.08, .025, 'Decision: check sector/date/size losses before locking a validation choice. SECTOR is descriptive metadata, not an added BASE feature.', fontsize=10)
    return fig


def load_saved_validation(folder='outputs/quote_quality_step5'):
    folder = Path(folder)
    manifest = json.loads((folder / 'experiment.json').read_text())
    for filename, expected in manifest.get('validation_files_sha256', {}).items():
        if filename not in ['model_features.parquet', 'validation_predictions.parquet']:
            raise ValueError('Invalid validation manifest filename')
        if hashlib.sha256((folder / filename).read_bytes()).hexdigest() != expected:
            raise ValueError('Saved validation file differs from its manifest: ' + filename)
    frame = pd.read_parquet(folder / 'model_features.parquet')
    predictions = pd.read_parquet(folder / 'validation_predictions.parquet')
    rows, _ = validation_rows(frame, predictions)
    expected_dates = manifest.get('split_dates', {}).get('Validation')
    if expected_dates is not None and sorted(pd.to_datetime(rows.time).dt.strftime('%Y-%m-%d').unique()) != expected_dates:
        raise ValueError('Saved validation dates differ from experiment.json')
    return frame, predictions, manifest


def show_validation_sectors(frame=None, predictions=None, folder='outputs/quote_quality_step5', selected='Quote levels'):
    """Safe add-on for an old live kernel: only reads existing results and draws."""
    if frame is None or predictions is None:
        frame, predictions, _ = load_saved_validation(folder)
    result = sector_diagnostics(frame, predictions, selected)
    fig = sector_figure(result)
    output = Path(folder); output.mkdir(parents=True, exist_ok=True)
    slug = selected.lower().replace(' ', '_')
    path = output / f'validation_sector_{slug}.png'
    fig.savefig(path, dpi=180)
    from IPython.display import display, Image
    display(Image(filename=str(path)))
    return result
