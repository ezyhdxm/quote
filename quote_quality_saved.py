"""Read existing validation results and slice target-trade SECTOR. Never trains."""
# SETUP LOGIC: Import serialization, tables and figures without loading research data.
from pathlib import Path
import hashlib
import json
import os
import tempfile
from uuid import uuid4
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


# CACHEING LOGIC: Define _file_sha256; calling is explicit, not triggered by this declaration.
def _file_sha256(path):
    # CACHEING LOGIC: Read a file in 1 MiB blocks; return a SHA256 digest without opening Parquet data.
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


# SETUP LOGIC: Define save_step5_checkpoint; calling is explicit, not triggered by this declaration.
def save_step5_checkpoint(frame, predictions=None, test_predictions=None, metadata=None,
                          folder='outputs/quote_quality_step5'):
    """Save a new immutable snapshot, then atomically publish latest.json.

    Neither flat exports nor earlier snapshots are rewritten. A failed write
    leaves the previous latest pointer usable and propagates its exception.
    No models are run and no test predictions are read or evaluated here.
    """
    # CORE LOGIC: STEP 1 — Require a completed, uniquely keyed feature frame.
    # Input: frame has row_id=[7,7], time=[10:00,10:01 ET], split=Validation and BM_SPREAD=[0.60,0.62].
    # Output: ValueError for duplicate row 7; no snapshot is written.
    if frame is None:
        raise ValueError('A completed feature frame is required before saving a checkpoint')
    if not {'row_id', 'time', 'split', 'BM_SPREAD'}.issubset(frame):
        raise ValueError('The checkpoint needs the complete Step5 feature frame')
    if frame.row_id.duplicated().any():
        raise ValueError('Feature frame row_id must be unique')
    # CORE LOGIC: STEP 2 — Validate completed validation before marking it available.
    # Input: predictions contain row 7 Validation, target=0.60, pred_spread=0.61, signed error=1 bps.
    # Output: available=True and row-level validation passes; predictions=None sets available=False.
    available = predictions is not None and not predictions.empty
    if available:
        validation_rows(frame, predictions)
    # SETUP LOGIC: Retain caller metadata; supply only fixed interface defaults for missing fields.
    from quote_quality_core import BASE_FEATURES, to_ny_datetime
    manifest = dict(metadata or {})
    for name, value in [('base_features', list(BASE_FEATURES)), ('target', 'D_BM_SPREAD'),
                        ('anchor', 'PREV_BM_SPREAD'), ('error_multiplier', 100),
                        ('quote_spread_unit', 'bps'), ('allow_exact_quotes', True),
                        ('locked_choice', None)]:
        manifest.setdefault(name, value)
    # CORE LOGIC: STEP 3 — Record availability and actual ET dates for each split.
    # Input: available=True; Validation rows at 2026-03-19 10:00 ET and 2026-03-20 10:00 ET; test_predictions=None.
    # Output: validation_available=True, test_available=False; Validation split_dates=['2026-03-19','2026-03-20'].
    # Trick: Date extraction uses positional arrays so duplicate DataFrame index labels do not mix split dates.
    manifest['validation_available'] = available
    manifest['test_available'] = test_predictions is not None and not test_predictions.empty
    et_time = to_ny_datetime(frame.time)
    split_days = pd.DataFrame({'split': frame['split'].to_numpy(),
                               'date': et_time.dt.strftime('%Y-%m-%d').to_numpy()})
    manifest['split_dates'] = {
        str(label): sorted(group.date.dropna().unique().tolist())
        for label, group in split_days.groupby('split', observed=True)}
    # CACHEING LOGIC: Write a fresh UUID snapshot; partial attempts never overwrite existing files.
    folder = Path(folder)
    snapshot = folder / 'snapshots' / uuid4().hex
    snapshot.mkdir(parents=True, exist_ok=False)
    # Each file is new, inside this attempt's unique directory. An interruption
    # can leave an unreferenced partial snapshot, never a partially updated save.
    frame.to_parquet(snapshot / 'model_features.parquet', index=False)
    files = ['model_features.parquet']
    if available:
        predictions.to_parquet(snapshot / 'validation_predictions.parquet', index=False)
        files.append('validation_predictions.parquet')
    if manifest['test_available']:
        test_predictions.to_parquet(snapshot / 'test_predictions.parquet', index=False)
        files.append('test_predictions.parquet')
    # CACHEING LOGIC: Record separate validation file checksums and atomically publish latest only after all writes succeed.
    manifest['files_sha256'] = {name: _file_sha256(snapshot / name) for name in files}
    manifest['validation_files_sha256'] = {
        name: manifest['files_sha256'][name] for name in files
        if name in ['model_features.parquet', 'validation_predictions.parquet']}
    manifest['checkpoint_schema'] = 1
    manifest_path = snapshot / 'experiment.json'
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    pointer = dict(schema=1, snapshot=snapshot.relative_to(folder).as_posix(),
                   experiment_sha256=_file_sha256(manifest_path))
    # Trick: only the final replace changes latest.json; a disk error leaves the previous pointer intact and propagates.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, prefix='.latest-', suffix='.json',
                                         mode='w', encoding='utf-8', delete=False) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(pointer, temporary, indent=2)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(folder / 'latest.json')
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return snapshot


# CACHEING LOGIC: Define _saved_validation_location; calling is explicit, not triggered by this declaration.
def _saved_validation_location(folder):
    """Resolve only a snapshot inside the requested folder; old exports still work."""
    # CACHEING LOGIC: Prefer safe latest.json snapshots, with legacy flat exports as the no-pointer fallback.
    folder = Path(folder)
    pointer_path = folder / 'latest.json'
    if not pointer_path.is_file():
        return folder, None
    pointer = json.loads(pointer_path.read_text(encoding='utf-8'))
    if not isinstance(pointer, dict):
        raise ValueError('Saved Step5 latest.json has an invalid snapshot pointer')
    relative = pointer.get('snapshot')
    if pointer.get('schema') != 1 or not isinstance(relative, str) or not relative:
        raise ValueError('Saved Step5 latest.json has an invalid snapshot pointer')
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Saved Step5 snapshot must stay inside its output folder')
    # Trick: resolved path containment also rejects symlinks escaping the requested output folder.
    snapshot = (folder / path).resolve()
    root = folder.resolve()
    if snapshot == root or not snapshot.is_relative_to(root):
        raise ValueError('Saved Step5 snapshot must stay inside its output folder')
    expected = pointer.get('experiment_sha256')
    if not isinstance(expected, str) or not expected:
        raise ValueError('Saved Step5 snapshot is missing its experiment checksum')
    return snapshot, expected


# SETUP LOGIC: Define direct_reference; calling is explicit, not triggered by this declaration.
def direct_reference(selected):
    # CORE LOGIC: STEP 1 — Select the fixed direct comparison baseline.
    # Input: selected='Quote levels', 'Reliability', or 'Age decay'.
    # Output: references='Base', 'Quote levels', and 'Reliability', respectively.
    return 'Base' if selected in ['Base', 'Quote levels'] else ('Quote levels' if selected == 'Reliability' else 'Reliability')


# SETUP LOGIC: Define validation_rows; calling is explicit, not triggered by this declaration.
def validation_rows(frame, predictions):
    """Reject stale joins and unequal cohorts before drawing any improvement."""
    # CORE LOGIC: STEP 1 — Require the original target metadata and unique source rows.
    # Input: frame has row 7, time=2026-03-19 10:00 ET, split=Validation, BM_SPREAD=0.60.
    # Output: Required frame fields pass; duplicate row 7 rejects before joining.
    frame_required = {'row_id', 'time', 'split', 'BM_SPREAD'}
    if not frame_required.issubset(frame):
        raise ValueError('Missing saved feature columns: ' + ', '.join(sorted(frame_required - set(frame))))
    if frame.row_id.duplicated().any():
        raise ValueError('Feature frame row_id must be unique')
    # CORE LOGIC: STEP 2 — Accept only complete validation predictions with the error interface.
    # Input: row 7 model=Age decay, stage=Validation, time=2026-03-19 10:00 ET, pred_spread=0.61, error_bps=1, abs_error_bps=1.
    # Output: All seven required prediction fields pass; stage=Test rejects.
    required = {'row_id', 'model', 'stage', 'time', 'abs_error_bps', 'pred_spread', 'error_bps'}
    if not required.issubset(predictions):
        raise ValueError('Missing validation prediction columns: ' + ', '.join(sorted(required - set(predictions))))
    if predictions.empty or not predictions.stage.eq('Validation').all():
        raise ValueError('This viewer accepts saved Validation only; it never opens test predictions')
    # CORE LOGIC: STEP 3 — Reject duplicate model votes and invalid absolute errors.
    # Input: (row_id,model) pairs=[(7,Base),(7,Age decay)], abs_error_bps=[1,0.8].
    # Output: The pairs pass and pivot to one row/two model columns; abs_error=-1 or a duplicate pair rejects.
    if predictions.duplicated(['row_id', 'model']).any():
        raise ValueError('Duplicate model / row_id predictions')
    if not np.isfinite(predictions.abs_error_bps).all() or predictions.abs_error_bps.lt(0).any():
        raise ValueError('Invalid validation absolute errors')
    wide = predictions.pivot(index='row_id', columns='model', values='abs_error_bps')
    # CORE LOGIC: STEP 4 — Require the same existing target rows for every model.
    # Input: wide row 7=[Base:1,Age decay:0.8], row 8=[Base:2,Age decay:1.9]; frame has rows 7,8.
    # Output: Both rows are kept; one missing model/row or row 9 absent from frame rejects.
    if wide.isna().any().any():
        raise ValueError('Every model must evaluate exactly the same target rows')
    ids = wide.index
    if not ids.isin(frame.row_id).all():
        raise ValueError('Prediction row_id is absent from the saved feature frame')
    rows = frame.set_index('row_id').loc[ids].copy()
    # CORE LOGIC: STEP 5 — Match the full Validation cohort, including no-quote targets.
    # Input: frame Validation ids=[7,8], bcq_has_quote=[1,0]; predicted ids=[7,8].
    # Output: Both 7 and no-quote row 8 remain; predicted ids=[7] reject.
    if 'split' in rows and not rows['split'].eq('Validation').all():
        raise ValueError('Predictions do not join to Validation rows')
    if 'split' in frame and set(frame.loc[frame['split'].eq('Validation'), 'row_id']) != set(ids):
        raise ValueError('Saved predictions omit validation target rows')
    # CORE LOGIC: STEP 6 — Join by row_id and verify exact target timestamps.
    # Input: frame row 7 timestamp=2026-03-19 10:00:00 ET; prediction row 7 timestamp=10:00:01 ET.
    # Output: ValueError for the one-second mismatch; matching timestamps pass.
    # Trick: Time alone is not a unique target key; two trades at 10:00 remain distinct through row_id.
    joined = predictions.merge(frame[['row_id', 'time']].rename(columns={'time': 'frame_time'}), on='row_id', validate='many_to_one')
    if not pd.to_datetime(joined.time).eq(pd.to_datetime(joined.frame_time)).all():
        raise ValueError('Saved prediction / feature timestamps differ')
    # CORE LOGIC: STEP 7 — Recompute signed and absolute bps errors against the saved truth.
    # Input: row 7 target=0.60, pred_spread=0.61, error_bps=1, abs_error_bps=1.
    # Output: Signed (0.61-0.60)*100=1 and abs(1)=1 pass; stored error=0.01 rejects.
    if {'pred_spread', 'error_bps'}.issubset(predictions) and 'BM_SPREAD' in rows:
        truth = predictions.row_id.map(rows.BM_SPREAD)
        if not np.allclose((predictions.pred_spread - truth) * 100, predictions.error_bps, rtol=1e-9, atol=1e-7):
            raise ValueError('Saved prediction errors disagree with the target or bps multiplier')
        if not np.allclose(predictions.error_bps.abs(), predictions.abs_error_bps, rtol=1e-9, atol=1e-7):
            raise ValueError('Saved absolute errors disagree with signed errors')
    return rows, wide


# SETUP LOGIC: Define sector_diagnostics; calling is explicit, not triggered by this declaration.
def sector_diagnostics(frame, predictions, selected='Quote levels'):
    # CORE LOGIC: STEP 1 — Validate the common cohort and select an existing direct reference.
    # Input: selected=Age decay, saved models include Age decay/Reliability, target row 7 has SECTOR=Energy.
    # Output: The pair is Age decay minus Reliability; missing target-trade SECTOR rejects.
    rows, wide = validation_rows(frame, predictions)
    if 'SECTOR' not in rows:
        raise ValueError('Target-trade SECTOR is missing from model_features; do not substitute a CUSIP mapping')
    reference = direct_reference(selected)
    if selected not in wide or reference not in wide:
        raise ValueError(f'Need saved {selected} and {reference} predictions')
    # Row-level target metadata; no whole-window CUSIP sector mapping here.
    # CORE LOGIC: STEP 2 — Keep target-row SECTOR and compute per-target loss differences.
    # Input: row 7 SECTOR='', ET date=2026-03-19, selected absolute error=1, reference=1.5.
    # Output: sector='Unknown', day=2026-03-19 ET, loss_delta=-0.5 bps.
    # Trick: No full-period CUSIP mapping backfills a missing target sector.
    rows['sector'] = rows.SECTOR.astype('string').str.strip().replace('', pd.NA).fillna('Unknown')
    from quote_quality_core import to_ny_datetime
    rows['day'] = to_ny_datetime(rows.time).dt.normalize()
    rows['selected_error'] = wide[selected]
    rows['reference_error'] = wide[reference]
    rows['loss_delta'] = rows.selected_error - rows.reference_error
    # CORE LOGIC: STEP 3 — Summarize same-row sector mean and tail differences.
    # Input: Energy selected errors=[1,2], reference=[1,3], on 2 dates.
    # Output: n=2, selected_mae=1.5, reference_mae=2, delta_mae=-0.5, delta_p95=-0.95, delta_gt10=0.
    groups = rows.groupby('sector', observed=True, sort=True)
    summary = groups.agg(n=('loss_delta', 'size'), dates=('day', 'nunique'),
                         selected_mae=('selected_error', 'mean'), reference_mae=('reference_error', 'mean'),
                         delta_mae=('loss_delta', 'mean'))
    summary['delta_p95'] = groups.selected_error.quantile(.95) - groups.reference_error.quantile(.95)
    summary['delta_gt10'] = groups.selected_error.apply(lambda x: x.gt(10).mean()) - groups.reference_error.apply(lambda x: x.gt(10).mean())
    # CORE LOGIC: STEP 4 — Retain sector/date cells, support counts and date-equal effects.
    # Input: Energy daily loss deltas on 03-19/03-20 are -1 and +0.5, one trade each.
    # Output: daily=[-1,+0.5], daily_n=[1,1], day_equal_delta=-0.25, worse_dates=1.
    daily = rows.groupby(['sector', 'day'], observed=True).loss_delta.mean().unstack('day')
    daily_n = rows.groupby(['sector', 'day'], observed=True).size().unstack('day', fill_value=0)
    summary['day_equal_delta'] = daily.mean(axis=1)
    summary['worse_dates'] = daily.gt(0).sum(axis=1)
    coverage = pd.DataFrame(index=summary.index)
    # CORE LOGIC: STEP 5 — Report availability over all sector validation trades.
    # Input: Energy bcq_has_quote=[1,0], bcq_n_pair=[1,0], bcq_n_size_time_pair=[0,0].
    # Output: Coverage is Any quote=0.5, Fresh pair=0.5, Size-time pair=0; no target row is removed.
    for label, column in [('Any quote', 'bcq_has_quote'), ('Fresh pair', 'bcq_n_pair'), ('Size-time pair', 'bcq_n_size_time_pair')]:
        if column in rows:
            coverage[label] = rows[column].gt(0).groupby(rows.sector).mean()
    # CORE LOGIC: STEP 6 — Define fixed coverage and par-quantity diagnostic slices.
    # Input: rows 7/8: has_quote=[1,0], QUANTITY=[100000,1000000].
    # Output: All ids=[7,8], Covered=[7], No quote=[8], <=100K=[7], >=1MM=[8].
    masks = {'All validation': pd.Series(True, index=rows.index)}
    if 'bcq_has_quote' in rows:
        masks.update({'No quote': rows.bcq_has_quote.eq(0), 'Covered': rows.bcq_has_quote.gt(0)})
    if 'QUANTITY' in rows:
        q = pd.to_numeric(rows.QUANTITY, errors='coerce')
        masks.update({'<=100K par': q.le(100_000), '>=1MM par': q.ge(1_000_000)})
    # CORE LOGIC: STEP 7 — Summarize the fixed slices and leave-one-date sensitivity.
    # Input: 03-19 row 7 delta=-1, 03-20 row 8 delta=+0.5; each date has one target.
    # Output: overall delta=-0.25; removing each date gives range [-1,+0.5]; n=2, date_n=2.
    # Trick: The leave-one-date range reuses fixed predictions; it is a sensitivity diagnostic, not a confidence interval.
    slices = pd.DataFrame([{'slice': name, 'n': int(mask.sum()), 'delta_mae': rows.loc[mask, 'loss_delta'].mean()}
                           for name, mask in masks.items()]).set_index('slice')
    all_daily = rows.groupby('day').loss_delta.mean()
    leave_one = [rows.loc[rows.day.ne(day), 'loss_delta'].mean() for day in all_daily.index] if len(all_daily) > 1 else []
    return dict(selected=selected, reference=reference, summary=summary, daily=daily, daily_n=daily_n,
                coverage=coverage, slices=slices, n=len(rows), date_n=len(all_daily),
                delta=float(rows.loss_delta.mean()), day_equal_delta=float(all_daily.mean()),
                leave_one_date_range=(min(leave_one), max(leave_one)) if leave_one else (np.nan, np.nan))


# PLOTTING LOGIC: Define sector_figure; calling is explicit, not triggered by this declaration.
def sector_figure(result):
    # PLOTTING LOGIC: Arrange already-computed sector means, support, date matrix and P95 into one PNG-ready figure.
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
    # Trick: the symmetric color bound is a display scale, not a fitted uncertainty band; missing date cells stay blank.
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


# CACHEING LOGIC: Define load_step5_metadata; calling is explicit, not triggered by this declaration.
def load_step5_metadata(folder='outputs/quote_quality_step5'):
    """Read only experiment metadata, including a persisted test choice lock.

    No feature/prediction Parquet is opened or hashed. A frame-only checkpoint
    can restore its lock even when completed validation is not available.
    """
    # CACHEING LOGIC: Read only the pointer/experiment manifest and its checksum; no feature or test Parquet is opened.
    folder, manifest_hash = _saved_validation_location(folder)
    manifest_path = folder / 'experiment.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('No saved Step5 checkpoint/export was found. Keep any live feature frame and predictions; this viewer does not train.')
    if manifest_hash is not None and _file_sha256(manifest_path) != manifest_hash:
        raise ValueError('Saved Step5 experiment.json differs from its latest.json checksum')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict):
        raise ValueError('Saved Step5 experiment.json must contain an experiment object')
    return manifest, folder.resolve()


# SETUP LOGIC: Define load_saved_validation; calling is explicit, not triggered by this declaration.
def load_saved_validation(folder='outputs/quote_quality_step5'):
    # CORE LOGIC: STEP 1 — Require a completed validation in the selected checkpoint.
    # Input: selected manifest has validation_available=False and only model_features.parquet.
    # Output: ValueError before reading Parquet; an older leftover prediction file is not reused.
    has_pointer = (Path(folder) / 'latest.json').is_file()
    manifest, folder = load_step5_metadata(folder)
    if manifest.get('validation_available') is False:
        raise ValueError('Features were saved, but this checkpoint/export has no completed validation. Any older prediction file is not reusable.')
    # CACHEING LOGIC: Require and verify only frame/validation checksums, never test-prediction checksums.
    checksums = manifest.get('validation_files_sha256', {})
    if not isinstance(checksums, dict):
        raise ValueError('Saved Step5 validation file checksums must be a filename mapping')
    if has_pointer or 'checkpoint_schema' in manifest:
        required = {'model_features.parquet', 'validation_predictions.parquet'}
        if manifest.get('checkpoint_schema') != 1 or not required.issubset(checksums):
            raise ValueError('Saved Step5 validation checkpoint is missing required file checksums')
        all_checksums = manifest.get('files_sha256', {})
        if not isinstance(all_checksums, dict):
            raise ValueError('Saved Step5 snapshot file checksums must be a filename mapping')
        if any(all_checksums.get(name) != checksums[name] for name in required):
            raise ValueError('Saved Step5 validation checkpoint file checksums disagree')
    # Trick: the accepted checksum filenames are limited to model_features and validation_predictions; test files are not read or hashed.
    for filename, expected in checksums.items():
        if filename not in ['model_features.parquet', 'validation_predictions.parquet']:
            raise ValueError('Invalid validation manifest filename')
        if not (folder / filename).is_file():
            raise FileNotFoundError('Saved Step5 validation file is missing: ' + filename)
        if _file_sha256(folder / filename) != expected:
            raise ValueError('Saved validation file checksum differs from its manifest: ' + filename)
    if not (folder / 'model_features.parquet').is_file():
        raise FileNotFoundError('Saved Step5 full feature frame is missing from this checkpoint/export.')
    if not (folder / 'validation_predictions.parquet').is_file():
        raise FileNotFoundError('No saved completed validation predictions were found in this checkpoint/export.')
    frame = pd.read_parquet(folder / 'model_features.parquet')
    predictions = pd.read_parquet(folder / 'validation_predictions.parquet')
    # CORE LOGIC: STEP 2 — Revalidate loaded predictions and their original validation dates.
    # Input: loaded row 7 has matching time/truth; actual Validation date=2026-03-19 and manifest says 2026-03-19.
    # Output: Return frame/predictions/manifest; date=2026-03-20 in the manifest rejects.
    rows, _ = validation_rows(frame, predictions)
    expected_dates = manifest.get('split_dates', {}).get('Validation')
    if expected_dates is not None and sorted(pd.to_datetime(rows.time).dt.strftime('%Y-%m-%d').unique()) != expected_dates:
        raise ValueError('Saved validation dates differ from experiment.json')
    return frame, predictions, manifest


# PLOTTING LOGIC: Define show_validation_sectors; calling is explicit, not triggered by this declaration.
def show_validation_sectors(frame=None, predictions=None, folder='outputs/quote_quality_step5', selected='Quote levels'):
    """Safe add-on for an old live kernel: only reads existing results and draws."""
    # CACHEING LOGIC: Use live completed objects or recover existing validation; neither branch trains a model.
    if frame is None or predictions is None:
        frame, predictions, _ = load_saved_validation(folder)
    # CORE LOGIC: STEP 1 — Compute the checked same-row sector comparison.
    # Input: selected=Quote levels, Base errors=[1,3], Quote levels errors=[1,2] in Energy.
    # Output: Energy MAE delta=-0.5 bps; the result has the existing two target ids.
    result = sector_diagnostics(frame, predictions, selected)
    # PLOTTING LOGIC: Draw and save all sector panels at 180 dpi, then display the saved PNG.
    fig = sector_figure(result)
    output = Path(folder); output.mkdir(parents=True, exist_ok=True)
    slug = selected.lower().replace(' ', '_')
    path = output / f'validation_sector_{slug}.png'
    fig.savefig(path, dpi=180)
    from IPython.display import display, Image
    display(Image(filename=str(path)))
    return result


# SETUP LOGIC: Define trade_rule_diagnostics; calling is explicit, not triggered by this declaration.
def trade_rule_diagnostics(frame, quote_start, quote_end):
    """Rule effects at actual eligible model target queries, inside file ET dates.

    This reads feature values only. It does not inspect test predictions or losses.
    Historical target dates outside the quote file are counted separately.
    """
    # SETUP LOGIC: Use the shared ET conversion without changing source feature columns.
    from quote_quality_core import to_ny_datetime
    # CORE LOGIC: STEP 1 — Validate the predeclared quote file date range.
    # Input: quote_start=2026-03-01 00:00 ET, quote_end=2026-04-01 23:59 ET; target at 2026-03-19 10:00 ET.
    # Output: day=[2026-03-19 00:00 ET], start=03-01 00:00 ET, end=04-01 00:00 ET; reversed endpoints reject.
    day = to_ny_datetime(frame.time).dt.normalize()
    start, end = to_ny_datetime(pd.Series([quote_start, quote_end])).dt.normalize()
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError('Need actual quote file start and end timestamps')
    # CORE LOGIC: STEP 2 — Separate actual targets inside file dates and retain target sector/date.
    # Input: targets on 2026-02-27 and 2026-03-19, file dates 03-01 through 04-01; SECTOR=Energy for 03-19.
    # Output: inside=[False,True]; rows has one Energy/03-19 target, summary.Energy n=1 and dates=1; the 02-27 row is outside.
    inside = day.between(start, end)
    rows = frame.loc[inside].copy()
    if rows.empty:
        raise ValueError('No eligible model targets within the quote file dates')
    rows['sector'] = rows.SECTOR.astype('string').str.strip().replace('', pd.NA).fillna('Unknown')
    rows['day'] = day.loc[inside]
    summary = rows.groupby('sector', observed=True).agg(n=('row_id', 'size'), dates=('day', 'nunique'))
    effect_rows, impact_rows = [], []
    # CORE LOGIC: STEP 3 — Determine side availability, fresh coverage and peer support.
    # Input: bid n_dealers=4, n_fresh_dealers=0, n_peer_supported=0, equal center=100 bps.
    # Output: available=True, fresh=False, supported=False; the target remains in the diagnostic.
    for side in ['bid', 'ask']:
        prefix = 'bcq_' + side + '_'
        equal = rows[prefix + 'center_equal']
        available = rows[prefix + 'n_dealers'].gt(0)
        fresh = rows[prefix + 'n_fresh_dealers'].gt(0)
        supported = rows[prefix + 'n_peer_supported'].gt(0) & available
        # CORE LOGIC: STEP 4 — Aggregate coverage costs and age metadata within each sector.
        # Input: Energy has two covered bid targets, one loses all fresh dealers; message ages=[20,40], unknown-change fractions=[0,1].
        # Output: bid_covered_n=2, bid_lost_n=1, median message age=30min, mean unknown-change fraction=0.5.
        summary[side + '_covered_n'] = available.groupby(rows.sector).sum()
        summary[side + '_lost_n'] = (available & ~fresh).groupby(rows.sector).sum()
        summary[side + '_peer_supported_n'] = supported.groupby(rows.sector).sum()
        summary[side + '_message_age'] = rows[prefix + 'median_message_age_min'].groupby(rows.sector).median()
        summary[side + '_change_age'] = rows[prefix + 'median_change_age_min'].groupby(rows.sector).median()
        summary[side + '_unknown_change_fraction'] = rows[prefix + 'unknown_change_age_fraction'].groupby(rows.sector).mean()
        # CORE LOGIC: STEP 5 — Initialize per-target impact identities and the four fixed rule comparisons.
        # Input: row 7 cusip=X, day=2026-03-19 ET, side=bid.
        # Output: local=[{'cusip':'X','day':'2026-03-19 ET','side':'bid'}], changes=[]; the loop labels are Decay, Max age, Clip, Downweight.
        local = rows[['cusip', 'day']].copy()
        local['side'] = side
        changes = []
        for label, field in [('Decay', 'center_decay'), ('Max age', 'center_max_age'), ('Clip', 'center_candidate_clip'), ('Downweight', 'center_dealer_downweight')]:
            # CORE LOGIC: STEP 6 — Measure rule shifts on the same target and require clipping/downweight support.
            # Input: equal=100, decay=102, clip=100 bps, peer_supported=False.
            # Output: Decay absolute shift=2 bps; Clip shift=NaN/unassessed rather than an assessed zero.
            # Trick: Insufficient peers make numerical fallback equal to baseline, but that zero does not validate the cleaning rule.
            delta = (rows[prefix + field] - equal).abs()
            if label in ['Clip', 'Downweight']:
                delta = delta.where(supported)  # fallback is unassessed, not zero effect
            changes.append(delta)
            for sector, values in delta.groupby(rows.sector):
                values = values.dropna()
                effect_rows.append(dict(side=side, rule=label, sector=sector, n=len(values),
                                        median=values.median(), p95=values.quantile(.95)))
        # CORE LOGIC: STEP 7 — Retain per-target maximum rule effect and dealer/fresh slot counts.
        # Input: four comparable rule shifts=[2,1,3,NaN], n_dealers=4, n_fresh_dealers=2.
        # Output: effect=3 bps, covered=1, dealer_slots=4, fresh_slots=2.
        local['effect'] = pd.concat(changes, axis=1).max(axis=1)
        local['covered'] = available.astype(int)
        local['supported'] = supported.astype(int)
        local['dealer_slots'] = rows[prefix + 'n_dealers']
        local['fresh_slots'] = rows[prefix + 'n_fresh_dealers']
        # CORE LOGIC: STEP 8 — Summarize actual queries into bond/side/day case impacts.
        # Input: X/bid/03-19 has two effects=[2,4], dealer slots=[4,4], fresh slots=[2,4].
        # Output: impact_queries=2, max_center_effect_bps=4, mean_center_effect_bps=3, baseline_dealer_slots=8, fresh_dealer_slots=6.
        impacts = local.groupby(['cusip', 'side', 'day'], observed=True).agg(
            impact_queries=('effect', 'size'), peer_supported_queries=('supported', 'sum'),
            max_center_effect_bps=('effect', 'max'), mean_center_effect_bps=('effect', 'mean'),
            baseline_dealer_slots=('dealer_slots', 'sum'), fresh_dealer_slots=('fresh_slots', 'sum')).reset_index()
        # CORE LOGIC: STEP 9 — Compute age-cutoff losses and attach all-target coverage totals.
        # Input: X/bid/day baseline_slots=8 and fresh_slots=6; Energy quote flags=[1,0].
        # Output: max_age_lost_slots=2, coverage_loss_fraction=0.25; Energy quote_n=1/2 targets.
        # Trick: The denominator is baseline dealer-query slots, not raw message rows; zero baseline gives unknown loss fraction.
        impacts['max_age_lost_slots'] = impacts.baseline_dealer_slots - impacts.fresh_dealer_slots
        impacts['coverage_loss_fraction'] = (impacts.max_age_lost_slots / impacts.baseline_dealer_slots).where(impacts.baseline_dealer_slots.gt(0))
        impact_rows.append(impacts)
    for label, field in [('quote', 'bcq_has_quote'), ('pair', 'bcq_n_pair'), ('size_pair', 'bcq_n_size_time_pair')]:
        summary[label + '_n'] = rows[field].gt(0).groupby(rows.sector).sum()
    return dict(summary=summary, effects=pd.DataFrame(effect_rows), impacts=pd.concat(impact_rows, ignore_index=True),
                n=len(rows), outside_file_date_n=int((~inside).sum()), start=start, end=end)


# PLOTTING LOGIC: Define trade_rule_figure; calling is explicit, not triggered by this declaration.
def trade_rule_figure(result):
    # PLOTTING LOGIC: Create the six-panel diagnostic figure and order sectors by target count.
    summary = result['summary'].sort_values('n', ascending=False)
    fig = Figure(figsize=(18, max(12, 7 + len(summary) * .3)), facecolor='white')
    a, b, c, d, e, f = fig.subplots(2, 3).flat
    fig.subplots_adjust(left=.12, right=.96, top=.86, bottom=.13, wspace=.6, hspace=.52)
    fig.suptitle('Global / SECTOR | quote rules at actual eligible trade queries\nFeature diagnostics only; no fitting or test-loss inspection', fontsize=16)
    fig.text(.5, .9, f"{result['start']:%Y-%m-%d}–{result['end']:%Y-%m-%d} ET file dates | n={result['n']:,} targets | {result['outside_file_date_n']:,} historical targets outside file dates reported separately", ha='center', fontsize=10)
    # CORE LOGIC: STEP 1 — Combine disjoint sector counts into global target support.
    # Input: sector counts n=[2,3], quote_n=[1,3], pair_n=[0,2], size_pair_n=[0,1].
    # Output: Global counts=[5 all targets,4 quoted,2 fresh-paired,1 size-time-paired].
    totals = summary.sum(numeric_only=True)
    counts = [totals['n'], totals.quote_n, totals.pair_n, totals.size_pair_n]
    # PLOTTING LOGIC: Draw global/sector support bars using the prepared counts.
    bars = a.bar(['All targets', 'Any quote', 'Fresh pair', 'Size-time pair'], counts, color=['#BBBBBB', '#277F8E', '#C9563D', '#8560A5'])
    a.bar_label(bars, fmt='%.0f', fontsize=8); a.tick_params(axis='x', rotation=20)
    a.set_title('Query coverage; one target row per vote'); a.set_ylabel('Targets; no coverage filtering')
    names = summary.index; y = np.arange(len(names))
    # CORE LOGIC: STEP 2 — Compute quote coverage on all sector target trades.
    # Input: Energy n=2, quote_n=1; Utilities n=3, quote_n=3.
    # Output: Plotted coverage fractions are Energy=0.5 and Utilities=1; no-quote targets stay in the denominator.
    # Trick: The fraction is calculated inside the bar argument; it is research arithmetic, not just styling.
    b.barh(y, summary.quote_n / summary.n, color='#277F8E')
    # PLOTTING LOGIC: Format the computed sector coverage bars.
    b.set_yticks(y, [f'{name} n={int(summary.loc[name,"n"]):,}' for name in names]); b.invert_yaxis(); b.set_xlim(0, 1.05)
    b.set_title('Any quote / all sector targets'); b.set_xlabel('Fraction')
    # CORE LOGIC: STEP 3 — Prepare explicitly sector-level effect summaries rather than pooled quantiles.
    # Input: bid Decay sector medians=[1,9] bps and P95=[2,10] with n=[100,1].
    # Output: median_sector=5, p95_sector=6 and n=101; these are not pooled target quantiles.
    # Trick: Median-of-sector summaries give sectors equal votes; labels state this rather than hiding a changed denominator.
    effects = result['effects']
    pooled = effects.groupby(['side', 'rule'], sort=False).agg(n=('n', 'sum'), median_sector=('median', 'median'), p95_sector=('p95', 'median'))
    # Sector medians are explicitly labelled; not a disguised pooled quantile.
    # PLOTTING LOGIC: Render the labelled sector-median effect bars.
    labels = [f'{side} {rule}\nn={int(r.n):,}' for (side, rule), r in pooled.iterrows()]
    x = np.arange(len(pooled))
    c.bar(x - .18, pooled.median_sector, .36, label='Median of sector medians', color='#277F8E')
    c.bar(x + .18, pooled.p95_sector, .36, label='Median of sector P95', color='#C9563D')
    c.set_xticks(x, labels, rotation=50, fontsize=7); c.set_ylabel('|rule − equal center| bps'); c.set_title('Effect among comparable targets'); c.legend(fontsize=7)
    # CORE LOGIC: STEP 4 — Display cutoff/support fractions with side-covered targets as denominator.
    # Input: bid covered=100, lost=20, peer_supported=30; ask covered=0.
    # Output: bid lost fraction=0.2, supported fraction=0.3; ask fractions are NaN, not zero.
    # Trick: Arithmetic is embedded in barh arguments here; those fractions are research quantities even though drawn in a plot call.
    for side, offset, color in [('bid', -.16, '#277F8E'), ('ask', .16, '#C9563D')]:
        covered = summary[side + '_covered_n']
        d.barh(y + offset, (summary[side + '_lost_n'] / covered).where(covered.gt(0)), .3, color=color, label=side)
        f.barh(y + offset, (summary[side + '_peer_supported_n'] / covered).where(covered.gt(0)), .3, color=color, label=side)
        e.scatter(summary[side + '_message_age'], y + offset, color=color, marker='o', label=side + ' message')
        e.scatter(summary[side + '_change_age'], y + offset, color=color, marker='x', label=side + ' change')
    # PLOTTING LOGIC: Format the prepared fractions, age points and explanatory footnotes without refitting.
    for ax in [d, e, f]:
        ax.set_yticks(y, names); ax.invert_yaxis(); ax.legend(fontsize=7)
    d.set_title('Max age loses side / covered side targets'); d.set_xlabel('Fraction; NaN = no baseline coverage')
    e.set_title('Sector median ages at trade query'); e.set_xlabel('Minutes; unknown change ages excluded')
    f.set_title('Any supported dealer / side-covered targets'); f.set_xlabel('Unsupported clip/downweight is unassessed')
    fig.text(.05, .06, 'Decisions: measure cutoff coverage cost; retain age/support/ambiguity before adopting a rule. Paired information remains a supplement to single sides.', fontsize=10)
    fig.text(.05, .035, 'Each effect uses the same bond, side and target time. No different bonds are averaged into a price. Feature diagnostics do not establish predictive gains.', fontsize=10)
    return fig


# PLOTTING LOGIC: Define show_trade_rule_effects; calling is explicit, not triggered by this declaration.
def show_trade_rule_effects(frame, quote_start, quote_end, folder='outputs/quote_quality_step5'):
    # CORE LOGIC: STEP 1 — Compute actual-query rule diagnostics from the existing feature frame.
    # Input: one in-file target at 2026-03-19, baseline bid center=100 and decay center=102 bps.
    # Output: The rule effect result records a 2bps bid decay shift; no predictive-gain claim is produced.
    result = trade_rule_diagnostics(frame, quote_start, quote_end)
    # PLOTTING LOGIC: Save and display the complete six-panel PNG at 180 dpi.
    path = Path(folder) / 'trade_query_rule_effects.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    trade_rule_figure(result).savefig(path, dpi=180)
    from IPython.display import display, Image
    display(Image(filename=str(path)))
    return result
