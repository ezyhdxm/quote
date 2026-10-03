"""One predeclared validation fit per block, against saved Age decay predictions.

No test fitting interface and no production-pipeline imports. Completed sidecars
and validation runs live separately from the original Step5 snapshot and lock.
"""
# SETUP LOGIC: Import scalar/table/model interfaces; no dataset is loaded.
from pathlib import Path
from uuid import uuid4
import hashlib
import json
import os
import tempfile

import numpy as np
import pandas as pd

from quote_quality_core import BASE_FEATURES, BASE_CAT_FEATURES, model_versions, to_ny_datetime
from quote_quality_saved import validation_rows, _file_sha256

# CONFIGURATION LOGIC: Keep independent output directories and the two allowed candidate names.
DEFAULT_FOLDER = 'outputs/quote_quality_incremental'
BLOCK_NAMES = {'Direction': 'Age decay + Direction', 'Issuer': 'Age decay + Issuer'}
_PARAM_KEYS = {'objective', 'boosting_type', 'n_estimators', 'learning_rate',
               'num_leaves', 'max_bin', 'n_jobs', 'random_state'}
_SOURCE_METADATA = ['base_features', 'target', 'anchor', 'error_multiplier',
    'quote_spread_unit', 'allow_exact_quotes', 'age_min', 'sync_min', 'lgb_params',
    'validation_versions', 'model_columns', 'quote_file_dates', 'input_files',
    'event_cache_key', 'settings_origin']


# CACHEING LOGIC: Define _json; calling is explicit, not triggered by this declaration.
def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


# CACHEING LOGIC: Define _digest; calling is explicit, not triggered by this declaration.
def _digest(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


# SETUP LOGIC: Define _positions; calling is explicit, not triggered by this declaration.
def _positions(frame):
    # CORE LOGIC: STEP 1 — Require source keys and unique original target identities.
    # Input: columns=['row_id','cusip','time','split'], row_id=[7,7].
    # Output: ValueError for duplicate row 7; no training starts.
    if not {'row_id', 'cusip', 'time', 'split'}.issubset(frame):
        raise ValueError('The original Step5 frame needs row_id, cusip, time and split')
    if frame.columns.duplicated().any() or frame.row_id.duplicated().any():
        raise ValueError('Original frame row_id must be unique')
    # CORE LOGIC: STEP 2 — Select and order only original Train/Validation rows.
    # Input: row_id=[9,7,8], split=['Test','Train','Validation'].
    # Output: positions=[1,2]; ids=[7,8]; row 9 is excluded.
    # Trick: Stable row_id ordering makes reordered input recover the same validation cohort.
    positions = np.flatnonzero(frame['split'].isin(['Train', 'Validation']).to_numpy())
    if not len(positions):
        raise ValueError('No original Train/Validation rows are available')
    ids = frame.row_id.iloc[positions]
    if ids.isna().any():
        raise ValueError('Original Train/Validation row_id cannot be missing')
    return positions[np.argsort(ids.to_numpy(), kind='stable')]


# CACHEING LOGIC: Define _frame_hash; calling is explicit, not triggered by this declaration.
def _frame_hash(frame, positions=None):
    """Stream columns to avoid copying a wide 600K-row source frame for hashing."""
    # CACHEING LOGIC: Hash scalar columns one at a time, including dtype and row count.
    positions = np.arange(len(frame)) if positions is None else positions
    # Trick: streaming one column avoids copying a wide 600,000-row frame; index labels are deliberately excluded.
    digest = hashlib.sha256()
    for column in sorted(frame.columns):
        values = frame[column].iloc[positions]
        digest.update(_json([column, str(values.dtype)]).encode('utf-8'))
        try:
            digest.update(pd.util.hash_pandas_object(values, index=False).to_numpy().tobytes())
        except TypeError as error:
            raise ValueError(f'Cannot fingerprint non-scalar source column {column}') from error
    digest.update(str(len(positions)).encode('ascii'))
    return digest.hexdigest()


# CACHEING LOGIC: Define frame_fingerprint; calling is explicit, not triggered by this declaration.
def frame_fingerprint(frame):
    """Bind complete original Train/Validation content; never hash test values."""
    return _frame_hash(frame, _positions(frame))


# SETUP LOGIC: Define _metadata_identity; calling is explicit, not triggered by this declaration.
def _metadata_identity(metadata):
    # CORE LOGIC: STEP 1 — Verify the fixed existing prediction task.
    # Input: metadata.base_features omits COUPON; target='D_BM_SPREAD', anchor='PREV_BM_SPREAD', error_multiplier=100.
    # Output: ValueError for the altered BASE list; an error_multiplier=1 also rejects before budgeting.
    if not isinstance(metadata, dict):
        raise ValueError('Original Step5 experiment metadata is required')
    fixed = {'base_features': BASE_FEATURES, 'target': 'D_BM_SPREAD',
             'anchor': 'PREV_BM_SPREAD', 'error_multiplier': 100}
    if any(metadata.get(name) != value for name, value in fixed.items()):
        raise ValueError('Original BASE14, target, anchor and bps metadata must match')
    # CORE LOGIC: STEP 2 — Require the recorded DART/MAE budget instead of inventing it.
    # Input: params={'objective':'mae','boosting_type':'dart','n_estimators':400,'learning_rate':0.2,'num_leaves':127,'max_bin':511,'n_jobs':8,'random_state':2026}.
    # Output: These eight required budget keys pass; missing n_estimators is rejected.
    params = metadata.get('lgb_params')
    if not isinstance(params, dict) or not _PARAM_KEYS.issubset(params):
        raise ValueError('Original lgb_params/model budget is incomplete; do not infer it')
    if (params.get('objective') != 'mae' or params.get('boosting_type') != 'dart'
            or not isinstance(params.get('n_estimators'), int) or params['n_estimators'] < 1):
        raise ValueError('Need the original positive DART/MAE training budget')
    # CORE LOGIC: STEP 3 — Reject unverifiable provenance and require original Age decay columns.
    # Input: settings_origin='legacy_live_settings_at_capture', model_columns={'Age decay':['D_CPP_BM_SPREAD']}; old budget=400 trees.
    # Output: ValueError: after-fit capture cannot prove the original training budget.
    # Trick: A present parameter dictionary is not proof that those parameters were used by the old fit.
    if metadata.get('settings_origin') == 'legacy_live_settings_at_capture':
        raise ValueError('Legacy settings captured after fitting cannot prove the original run budget')
    if metadata.get('settings_origin') != 'captured_before_fitting':
        raise ValueError('Original settings provenance is unverified; no fitting is permitted')
    if not isinstance(metadata.get('model_columns'), dict) or 'Age decay' not in metadata['model_columns']:
        raise ValueError('Original Age decay feature specification is required')
    return {key: metadata.get(key) for key in _SOURCE_METADATA}


# SETUP LOGIC: Define _guard_fit; calling is explicit, not triggered by this declaration.
def _guard_fit(metadata):
    # CORE LOGIC: STEP 1 — Block new validation after locking, exposure, or unknown test status.
    # Input: locked_choice='Age decay', test_available=False, saved_lock_guard=False.
    # Output: ValueError before construction of any new model; absent test_available also rejects.
    if (not isinstance(metadata, dict) or 'locked_choice' not in metadata or 'test_available' not in metadata
            or metadata.get('locked_choice') is not None or metadata.get('test_available')
            or metadata.get('saved_lock_guard') or metadata.get('test_exposed')):
        raise ValueError('Test/choice is locked, exposed, or its status is unknown; validation fitting is forbidden')


# SETUP LOGIC: Define _feature_lists; calling is explicit, not triggered by this declaration.
def _feature_lists():
    # Lazy import permits movement to use frame_fingerprint without a cycle.
    # SETUP LOGIC: Import feature names lazily so movement can call frame_fingerprint without an import cycle.
    from quote_quality_movement import DIRECTION_FEATURES, ISSUER_FEATURES
    return list(DIRECTION_FEATURES), list(ISSUER_FEATURES)


# SETUP LOGIC: Define _validate_sidecar; calling is explicit, not triggered by this declaration.
def _validate_sidecar(frame, sidecar, metadata):
    # CORE LOGIC: STEP 1 — Bind source content and quote-cache identity to the original run.
    # Input: original frame row 7 COUPON=5; sidecar was built before row 7 changed to COUPON=6; both event_cache_key='events-v1'.
    # Output: ValueError for the changed source content; absent event_cache_key also rejects even when both sides are None.
    identity = _metadata_identity(metadata)
    source_hash = frame_fingerprint(frame)
    if sidecar.attrs.get('source_frame_sha256') != source_hash:
        raise ValueError('Movement sidecar source-frame fingerprint differs')
    event_key = sidecar.attrs.get('event_cache_key')
    if (not isinstance(event_key, str) or not event_key
            or event_key != metadata.get('event_cache_key')):
        raise ValueError('Movement sidecar quote/event source key differs from the original run')
    # CORE LOGIC: STEP 2 — Match the one fixed movement window and as-of policy.
    # Input: movement_config={'lookback_min':30,'age_min':30,'allow_exact':True,'same_et_day':True}; original age_min=30, exact=True.
    # Output: The window passes; lookback_min=60 or exact=False rejects.
    config = sidecar.attrs.get('movement_config')
    if not isinstance(config, dict):
        raise ValueError('Movement sidecar needs its explicit movement_config')
    if (config.get('lookback_min') != 30 or config.get('same_et_day') is not True
            or config.get('age_min') != metadata.get('age_min')
            or config.get('allow_exact') != metadata.get('allow_exact_quotes')):
        raise ValueError('Movement config differs from the fixed 30min/original as-of policy')
    # CORE LOGIC: STEP 3 — Require only the declared additional columns and nonmissing identities.
    # Input: include_issuer=True; keys=3, Direction columns=28, Issuer columns=20, total=51 columns.
    # Output: The 51-column schema passes; a duplicated row_id=7 or extra COUPON column rejects.
    direction, issuer = _feature_lists()
    expected = ['row_id', 'cusip', 'time'] + direction + (issuer if config.get('include_issuer') else [])
    if sidecar.columns.duplicated().any() or set(sidecar.columns) != set(expected):
        raise ValueError('Movement sidecar columns do not match the declared feature schema')
    if sidecar.row_id.duplicated().any() or sidecar[['row_id', 'cusip', 'time']].isna().any().any():
        raise ValueError('Movement sidecar requires unique nonmissing row_id/cusip/time')
    # CORE LOGIC: STEP 4 — Match every original allowed row and its exact bond/time.
    # Input: original rows 7/X/2026-03-19 10:00 ET and 8/Y/10:01; sidecar has rows 7,8.
    # Output: Both rows pass; replacing row 8 by a Test row 9 or X by Z rejects.
    original = frame.iloc[_positions(frame), frame.columns.get_indexer(['row_id', 'cusip', 'time'])].set_index('row_id')
    if len(sidecar) != len(original) or set(sidecar.row_id) != set(original.index):
        raise ValueError('Movement sidecar must contain exactly original Train/Validation rows; no test rows')
    ordered = sidecar.set_index('row_id').loc[original.index]
    if (not ordered.cusip.eq(original.cusip).all()
            or not pd.to_datetime(ordered.time).eq(pd.to_datetime(original.time)).all()):
        raise ValueError('Movement sidecar cusip/time keys differ from the source frame')
    # CORE LOGIC: STEP 5 — Keep unknown numeric movement; reject infinite or malformed values.
    # Input: bcq_bid_move_mean_bps=[-2.0,NaN]; common_n=[3,0].
    # Output: Numeric -2 and unknown NaN pass; +infinity or text bad rejects.
    for column in direction + (issuer if config.get('include_issuer') else []):
        values = pd.to_numeric(ordered[column], errors='coerce')
        if np.isinf(values).any() or (ordered[column].notna() & values.isna()).any():
            raise ValueError('Movement features must be numeric or unknown, not infinite: ' + column)
    # CACHEING LOGIC: Fingerprint all sidecar values/config for strict recovery and retain the feature dictionary.
    content_hash = _frame_hash(sidecar, np.argsort(sidecar.row_id.to_numpy(), kind='stable'))
    return dict(source_frame_sha256=source_hash, original_metadata=identity,
                movement_config=config, event_cache_key=sidecar.attrs.get('event_cache_key'),
                sidecar_sha256=content_hash, feature_columns=expected[3:],
                feature_dictionary=sidecar.attrs.get('feature_dictionary', {}))


# CACHEING LOGIC: Define _source_key; calling is explicit, not triggered by this declaration.
def _source_key(source_hash, identity):
    return _digest({'source_frame_sha256': source_hash, 'original_metadata': identity})


# CACHEING LOGIC: Define _atomic_json; calling is explicit, not triggered by this declaration.
def _atomic_json(path, value):
    # CACHEING LOGIC: Write and fsync a temporary JSON, then publish it atomically; remove the temporary on failure.
    # Trick: the old pointer remains usable until replace succeeds; fsync flushes the temporary content before publication.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.publish-', suffix='.json',
                                         mode='w', encoding='utf-8', delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(_json(value)); temporary.flush(); os.fsync(temporary.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


# CACHEING LOGIC: Define _read_sidecar_pointer; calling is explicit, not triggered by this declaration.
def _read_sidecar_pointer(root):
    # CACHEING LOGIC: Resolve only an in-folder snapshot and verify manifest/Parquet hashes before reading features.
    pointer_path = root / 'latest.json'
    if not pointer_path.is_file():
        raise FileNotFoundError('No matching saved movement sidecar; build it once before fitting')
    pointer = json.loads(pointer_path.read_text(encoding='utf-8'))
    relative = pointer.get('snapshot') if isinstance(pointer, dict) else None
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('Invalid movement-sidecar snapshot pointer')
    # Trick: resolve checks symlink destinations as well as .. and absolute paths, preventing an out-of-folder snapshot.
    snapshot = (root / relative).resolve()
    if not snapshot.is_relative_to(root.resolve()) or snapshot == root.resolve():
        raise ValueError('Movement-sidecar snapshot must stay inside its folder')
    manifest_path = snapshot / 'manifest.json'
    if _file_sha256(manifest_path) != pointer.get('manifest_sha256'):
        raise ValueError('Movement sidecar manifest checksum differs')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if _file_sha256(snapshot / 'movement_features.parquet') != manifest.get('file_sha256'):
        raise ValueError('Movement sidecar Parquet checksum differs')
    return snapshot, manifest


# SETUP LOGIC: Define save_movement_sidecar; calling is explicit, not triggered by this declaration.
def save_movement_sidecar(frame, sidecar, metadata, folder=DEFAULT_FOLDER):
    """Save a unique sidecar snapshot; never touch the original Step5 pointer."""
    # CORE LOGIC: STEP 1 — Validate source/schema before saving an additional feature file.
    # Input: 50 allowed rows, 51 sidecar columns, original event key='original-events'.
    # Output: manifest.movement_config.lookback_min=30 and event_cache_key='original-events'; a mismatched row 59 rejects before writing.
    manifest = _validate_sidecar(frame, sidecar, metadata)
    # CACHEING LOGIC: Write a unique snapshot and publish its sidecar pointer; original Step5 latest.json is untouched.
    root = Path(folder).resolve() / 'sidecars' / _source_key(manifest['source_frame_sha256'], manifest['original_metadata'])
    snapshot = root / uuid4().hex
    snapshot.mkdir(parents=True, exist_ok=False)
    sidecar.to_parquet(snapshot / 'movement_features.parquet', index=False)
    manifest['file_sha256'] = _file_sha256(snapshot / 'movement_features.parquet')
    manifest['schema'] = 1
    (snapshot / 'manifest.json').write_text(_json(manifest), encoding='utf-8')
    _atomic_json(root / 'latest.json', {'snapshot': snapshot.name,
                 'manifest_sha256': _file_sha256(snapshot / 'manifest.json')})
    return snapshot


# SETUP LOGIC: Define load_movement_sidecar; calling is explicit, not triggered by this declaration.
def load_movement_sidecar(frame, metadata, folder=DEFAULT_FOLDER):
    """Recover only a sidecar bound to this frame and original experiment settings."""
    # CORE LOGIC: STEP 1 — Check the original task and obtain the allowed-frame content identity.
    # Input: frame contains 40 Train, 10 Validation and 10 Test rows with captured 400-tree parameters.
    # Output: Only the 50 allowed rows enter the lookup identity; Test target values are not hashed.
    identity, source_hash = _metadata_identity(metadata), frame_fingerprint(frame)
    # CACHEING LOGIC: Look up and verify the matching saved sidecar snapshot.
    root = Path(folder) / 'sidecars' / _source_key(source_hash, identity)
    snapshot, manifest = _read_sidecar_pointer(root)
    # CORE LOGIC: STEP 2 — Reject a saved sidecar belonging to another experiment.
    # Input: schema=1 and original_metadata.lgb_params.n_estimators=400; caller also records 400.
    # Output: Metadata passes; stored 401 versus caller 400 rejects.
    if manifest.get('schema') != 1 or manifest.get('original_metadata') != identity:
        raise ValueError('Movement sidecar source metadata differs')
    # CACHEING LOGIC: Read sidecar only and restore binding attrs from the checked companion manifest.
    sidecar = pd.read_parquet(snapshot / 'movement_features.parquet')
    sidecar.attrs = {key: manifest[key] for key in ['source_frame_sha256', 'movement_config', 'event_cache_key', 'feature_dictionary']}
    # CORE LOGIC: STEP 3 — Revalidate all restored rows, schema and content.
    # Input: restored rows=[7,8], times=[10:00,10:01 ET] and stored movement=-2,1 bps.
    # Output: Matching content returns the sidecar and snapshot path; changed movement=-9 rejects.
    # Trick: Parquet attrs are not the sole provenance check: the separate manifest and content hash are authoritative.
    verified = _validate_sidecar(frame, sidecar, metadata)
    if verified['sidecar_sha256'] != manifest.get('sidecar_sha256'):
        raise ValueError('Movement sidecar content fingerprint differs')
    return sidecar, dict(manifest, snapshot_path=str(snapshot))


# SETUP LOGIC: Define _validate_reference; calling is explicit, not triggered by this declaration.
def _validate_reference(frame, predictions, metadata):
    # CORE LOGIC: STEP 1 — Validate original predictions and ensure Age decay is present.
    # Input: saved models=['Base','Quote levels','Reliability','Age decay'], each predicts Validation rows 7 and 8.
    # Output: The common rows pass; one omitted model/row or a stale timestamp rejects.
    rows, _ = validation_rows(frame, predictions)
    names = predictions.model.drop_duplicates().tolist()
    if 'Age decay' not in names or set(names) != set(metadata.get('validation_versions', [])):
        raise ValueError('Need original saved validation versions, including Age decay')
    # CORE LOGIC: STEP 2 — Check the original training population and validation dates.
    # Input: frame has 40 Train rows; all predictions.train_n=40; Validation date=2026-03-19.
    # Output: Counts/dates pass; train_n=39 or saved date=2026-03-20 rejects.
    train_n = int(frame['split'].eq('Train').sum())
    if 'train_n' not in predictions or not predictions.train_n.eq(train_n).all():
        raise ValueError('Original prediction train_n differs from the original Train cohort')
    if metadata.get('split_dates'):
        dates = to_ny_datetime(rows.time).dt.strftime('%Y-%m-%d').unique()
        if sorted(dates) != metadata['split_dates'].get('Validation'):
            raise ValueError('Validation dates differ from the original experiment')
    # CACHEING LOGIC: Bind every saved reference value in deterministic model/row_id order.
    order = predictions.sort_values(['model', 'row_id'], kind='stable').reset_index(drop=True)
    return _frame_hash(order)


# SETUP LOGIC: Define _load_completed; calling is explicit, not triggered by this declaration.
def _load_completed(path, expected, frame):
    # CACHEING LOGIC: Read the immutable completion record before any model construction.
    manifest = json.loads((path / 'completed.json').read_text(encoding='utf-8'))
    # CORE LOGIC: STEP 1 — Require the identical completed block request.
    # Input: stored block='Direction', sidecar mean=-2 bps, n_estimators=400; request has those same values.
    # Output: The request passes; changing that movement to -3 or budget to 401 refuses another fit.
    if manifest.get('request') != expected:
        raise ValueError('This block already has a different saved request; no additional fit is permitted')
    # CACHEING LOGIC: Verify request, sidecar and validation-prediction file checksums, then read predictions only.
    for filename in ['request.json', 'movement_features.parquet', 'validation_predictions.parquet']:
        if _file_sha256(path / filename) != manifest.get('files_sha256', {}).get(filename):
            raise ValueError('Completed incremental-run checksum differs: ' + filename)
    predictions = pd.read_parquet(path / 'validation_predictions.parquet')
    # CORE LOGIC: STEP 2 — Recheck recovered validation errors/model labels.
    # Input: row 7: target=0.60, pred_spread=0.61, error_bps=1, abs_error_bps=1; model='Age decay + Direction'.
    # Output: The row passes and recovery returns model=None/reused=True; label Age decay rejects.
    validation_rows(frame, predictions)
    if not predictions.model.eq(expected['candidate']).all():
        raise ValueError('Completed incremental-run model name differs')
    return predictions, dict(manifest['run_metadata'], snapshot_path=str(path), reused=True)


# SETUP LOGIC: Define run_incremental_validation; calling is explicit, not triggered by this declaration.
def run_incremental_validation(frame, sidecar, predictions, metadata, block,
                               progress=None, folder=DEFAULT_FOLDER):
    """Fit one candidate, or recover that block's identical completed validation.

    An incomplete prior attempt is deliberately not restarted. No test targets,
    predictions or metrics are opened. The returned model is None on recovery.
    New candidates are not in core.model_versions: the old Step5 locked-test
    button therefore cannot select or evaluate them without a separate reviewed
    test adapter and persisted choice; this module exposes no such adapter.
    """
    # CORE LOGIC: STEP 1 — Allow only the two predeclared blocks and bind existing experiment inputs.
    # Input: block='Direction', metadata.locked_choice=None, test_available=False; 50 sidecar rows and 4 saved validation models.
    # Output: Direction passes; block=Test, locked choice, mismatched sidecar or reference fails before fit.
    if block not in BLOCK_NAMES:
        raise ValueError('Only Direction and Issuer validation blocks are supported')
    _guard_fit(metadata)
    sidecar_info = _validate_sidecar(frame, sidecar, metadata)
    reference_hash = _validate_reference(frame, predictions, metadata)
    # CORE LOGIC: STEP 2 — Keep exactly the existing Train/Validation sample with finite targets.
    # Input: rows 0..39 are Train, 40..49 Validation, 50..59 Test; row 40 target=0.02, anchor=0.60, actual=0.62.
    # Output: x contains rows 0..49; invalid row 40 rejects rather than being silently deleted.
    original_columns = list(dict.fromkeys(BASE_FEATURES + ['row_id', 'cusip', 'time', 'split',
        'D_BM_SPREAD', 'PREV_BM_SPREAD', 'BM_SPREAD'] + [c for c in frame if c.startswith('bcq_')]))
    x = frame.iloc[_positions(frame), frame.columns.get_indexer(original_columns)].copy()
    if not np.isfinite(x[['D_BM_SPREAD', 'PREV_BM_SPREAD', 'BM_SPREAD']]).all().all():
        raise ValueError('Original Train/Validation targets are invalid; no cohort filtering is allowed')
    # CORE LOGIC: STEP 3 — Require chronology and restore the original Age decay definition.
    # Input: 40 Train rows end 2026-03-18; 10 Validation rows begin 2026-03-19; saved Age decay has 37 columns.
    # Output: Chronology and the 37-column specification pass; overlapping times or a changed column list reject.
    train, evaluate = x['split'].eq('Train'), x['split'].eq('Validation')
    if train.sum() < 20 or not evaluate.any() or x.loc[train, 'time'].max() >= x.loc[evaluate, 'time'].min():
        raise ValueError('Need original chronological Train/Validation cohorts')
    converted, specs = model_versions(x)
    base_columns, replacements = specs['Age decay']
    if list(metadata['model_columns']['Age decay']) != base_columns:
        raise ValueError('Current Age decay columns differ from the original feature specification')
    # CORE LOGIC: STEP 4 — Select one added block and align its values by original row_id.
    # Input: block='Direction'; sidecar order=[8,7], x.row_id=[7,8], means=[1,-2] bps.
    # Output: Added block has 28 columns; aligned means become [-2,1] bps; Issuer is not combined.
    direction, issuer = _feature_lists()
    added = direction if block == 'Direction' else issuer
    if not set(added).issubset(sidecar):
        raise ValueError('Requested block was not built in this sidecar')
    ordered = sidecar.set_index('row_id').loc[x.row_id]
    # Count-only columns cannot make an entirely unknown movement block assessed.
    # CORE LOGIC: STEP 5 — Distinguish an observed zero movement from entirely unknown support.
    # Input: Train means=[0,NaN], Validation means=[-2,NaN], support counts=[1,1].
    # Output: The block is assessed; all-NaN Validation means reject even if count columns are finite.
    # Trick: Coverage counts alone do not prove a movement value was observed; zero movement with a finite mean is valid support.
    support_columns = [c for c in added if c.endswith('mean_bps')]
    numeric = ordered[added].apply(pd.to_numeric, errors='coerce')
    support = np.isfinite(numeric[support_columns]).any(axis=1).to_numpy()
    support_counts = {'Train': int(support[train.to_numpy()].sum()),
                      'Validation': int(support[evaluate.to_numpy()].sum())}
    if min(support_counts.values()) == 0:
        raise ValueError('Movement block is unassessed: all movement values are unknown in Train or Validation')
    # One block per source experiment, even if someone changes its parameters,
    # reference predictions or sidecar config. Changed requests must be reviewed,
    # not turned into additional fits through a different request-hash directory.
    # CACHEING LOGIC: Bind one immutable request per source/block so altered parameters cannot bypass the single-fit ledger.
    source_key = _source_key(sidecar_info['source_frame_sha256'],
                             {'event_cache_key': sidecar_info['event_cache_key']})
    request = dict(schema=1, block=block, candidate=BLOCK_NAMES[block], reference='Age decay',
        source_frame_sha256=sidecar_info['source_frame_sha256'], reference_predictions_sha256=reference_hash,
        original_metadata=sidecar_info['original_metadata'], movement_config=sidecar_info['movement_config'],
        event_cache_key=sidecar_info['event_cache_key'], sidecar_sha256=sidecar_info['sidecar_sha256'],
        added_features=added, model_features=base_columns + added)
    run_path = Path(folder).resolve() / 'runs' / source_key / block.lower()
    # CORE LOGIC: STEP 6 — Reuse a completed block instead of fitting it again.
    # Input: runs/source/Direction/completed.json matches the 400-tree request and 10 validation predictions.
    # Output: Return the same 10 predictions, model=None and reused=True; zero extra fits.
    if (run_path / 'completed.json').is_file():
        restored, run_metadata = _load_completed(run_path, request, frame)
        if progress is not None:
            progress('models', 1, 1, f'Recovered completed {request["candidate"]}; no fitting')
        return restored, None, run_metadata
    # CORE LOGIC: STEP 7 — Refuse automatic restart of a partial prior model attempt.
    # Input: Direction/request.json exists but completed.json does not after an interrupted fit.
    # Output: RuntimeError; no second constructor or fit is attempted.
    # Trick: Partial model fit is not recoverable; a saved sidecar is not evidence of a completed model.
    if run_path.exists():
        raise RuntimeError('This block has an incomplete prior attempt; inspect it before retrying, no automatic refit')
    # CACHEING LOGIC: Create a new request directory and preserve its sidecar before starting the sole fit.
    run_path.mkdir(parents=True, exist_ok=False)
    (run_path / 'request.json').write_text(_json(request), encoding='utf-8')
    sidecar.to_parquet(run_path / 'movement_features.parquet', index=False)
    # CORE LOGIC: STEP 8 — Apply only original Age decay replacements and append this one block.
    # Input: center_decay=123.5 bps, PREV_BM_SPREAD=0.60, Direction mean=-2 bps.
    # Output: bcq_bid_anchor_gap=63.5 bps; matrix has 37+28=65 columns; the 14 BASE values are unchanged.
    z = converted[base_columns].copy()
    for column, value in replacements.items():
        z[column] = value
    for column in added:
        z[column] = numeric[column].to_numpy()
    # CORE LOGIC: STEP 9 — Use training-only categorical vocabularies and numeric missing values.
    # Input: Train TRADE_TYPE=['B','S']; Validation TRADE_TYPE=['D']; numeric COUPON=+infinity.
    # Output: Categories remain B/S; unseen D and infinite COUPON become NaN.
    # Trick: Validation/Test category names never extend the training vocabulary.
    for column in z:
        if column in BASE_CAT_FEATURES:
            categories = pd.Index(z.loc[train, column].dropna().unique())
            z[column] = pd.Categorical(z[column], categories=categories)
        else:
            z[column] = pd.to_numeric(z[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
    # SETUP LOGIC: Import the estimator without invoking fit.
    import lightgbm as lgb
    # CORE LOGIC: STEP 10 — Instantiate one estimator using the exact original recorded parameters.
    # Input: saved n_estimators=400, learning_rate=0.2, boosting_type=dart, n_jobs=8.
    # Output: One estimator has the same 400-tree budget and 8 threads; original four estimators are not rebuilt.
    params = dict(metadata['lgb_params'])
    model = lgb.LGBMRegressor(**params)
    fit_kwargs = {}
    # UI LOGIC: Report the model stage and install an iteration-display callback.
    if progress is not None:
        progress('models', 0, 1, f'Preparing {request["candidate"]}; original Age decay is not refitted')
        progress('fit', 0, model.n_estimators, f'{request["candidate"]}: training')
        # UI LOGIC: Define report_iteration; calling is explicit, not triggered by this declaration.
        def report_iteration(env):
            done, total = env.iteration - env.begin_iteration + 1, env.end_iteration - env.begin_iteration
            if done % 10 == 0 or done == total:
                progress('fit', done, total, f'{request["candidate"]}: iteration {done:,}/{total:,}')
        report_iteration.order, report_iteration.before_iteration = 20, False
        fit_kwargs['callbacks'] = [report_iteration]
    # CORE LOGIC: STEP 11 — Fit solely on the original Train rows.
    # Input: Train row_ids=0..39 with D_BM_SPREAD=0.02; Validation=40..49; Test=50..59.
    # Output: One fit receives 40 Train labels only; Validation/Test target values do not enter training.
    model.fit(z.loc[train], x.loc[train, 'D_BM_SPREAD'], categorical_feature=BASE_CAT_FEATURES, **fit_kwargs)
    # UI LOGIC: Report prediction after training has completed.
    if progress is not None:
        progress('models', 0, 1, f'Predicting validation rows: {request["candidate"]}')
    # CORE LOGIC: STEP 12 — Predict the same validation rows and reconstruct spread/error in bps.
    # Input: row 40 model delta=0.015, anchor=0.60, actual BM_SPREAD=0.62.
    # Output: pred_spread=0.615, error_bps=-0.5, abs_error_bps=0.5, train_n=40.
    pred = model.predict(z.loc[evaluate]) + x.loc[evaluate, 'PREV_BM_SPREAD'].to_numpy()
    output_columns = ['row_id', 'time', 'TRADE_TYPE', 'QUANTITY', 'bcq_has_quote', 'bcq_n_pair', 'bcq_n_size_time_pair']
    out = x.loc[evaluate, output_columns].copy().reset_index(drop=True)
    out['model'], out['stage'] = request['candidate'], 'Validation'
    out['pred_spread'], out['error_bps'] = pred, (pred - x.loc[evaluate, 'BM_SPREAD'].to_numpy()) * 100
    out['abs_error_bps'], out['train_n'] = out.error_bps.abs(), int(train.sum())
    validation_rows(frame, out)
    # CACHEING LOGIC: Save completed new validation predictions independently from the old snapshot.
    out.to_parquet(run_path / 'validation_predictions.parquet', index=False)
    # CORE LOGIC: STEP 13 — Record support limitations and the current test-adapter boundary.
    # Input: Train has 200 rows with 1 supported movement; Validation has 10 rows with 1 supported movement.
    # Output: warnings=['Train movement support is below 1%']; train_n=200, validation_n=10, stage=Validation, locked_test_supported=False.
    # Trick: These new names are absent from core.model_versions, so the original Step5 test button cannot evaluate them.
    warnings = [f'{name} movement support is below 1%' for name, count in support_counts.items()
                if count / int(x["split"].eq(name).sum()) < .01]
    run_metadata = dict(request, support_counts=support_counts, warnings=warnings,
        train_n=int(train.sum()), validation_n=int(evaluate.sum()), stage='Validation',
        locked_test_supported=False, locked_test_limitation='Candidate is absent from original core model_versions; original Step5 test control cannot evaluate it.',
        snapshot_path=str(run_path), reused=False)
    # CACHEING LOGIC: Hash completed artifacts and atomically publish the immutable completion record.
    checksums = {name: _file_sha256(run_path / name) for name in
                 ['request.json', 'movement_features.parquet', 'validation_predictions.parquet']}
    _atomic_json(run_path / 'completed.json', dict(request=request, files_sha256=checksums, run_metadata=run_metadata))
    # UI LOGIC: Report completion only after durable publication succeeds.
    if progress is not None:
        progress('models', 1, 1, f'Finished and saved {request["candidate"]}')
    return out, model, run_metadata
