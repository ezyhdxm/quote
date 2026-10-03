"""Content-checked narrow event cache shared by research notebooks. No pickle."""
from pathlib import Path
import hashlib
import json
import tempfile
from time import perf_counter
import numpy as np
import pandas as pd
import quote_quality_core as core

SCHEMA_VERSION = 1


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _source_key(quotes):
    narrow = quotes[core.KEYS + ['spread', 'quantity']].copy()
    # Full source-row duplicate semantics remain intact even though unrelated
    # metadata does not travel through the cached event table.
    narrow['source_repeat'] = quotes.duplicated()
    rows = pd.util.hash_pandas_object(narrow, index=False).to_numpy(dtype='uint64')
    h = hashlib.sha256(np.sort(rows).tobytes())
    h.update(json.dumps([(str(c), str(narrow[c].dtype)) for c in narrow], separators=(',', ':')).encode())
    h.update(_file_hash(core.__file__).encode())
    h.update(str(SCHEMA_VERSION).encode())
    return h.hexdigest()


def prepare_quote_events(quotes, progress=None, cache_dir='outputs/quote_quality_cache'):
    """Reuse only events from the same source/duplicate semantics and core code.

    Raw rows stay in the caller. Distinct sets are JSON in Parquet; timestamps
    retain nanosecond precision and ET timezone. Set cache_dir=False to bypass.
    """
    if cache_dir is False:
        return core.prepare_quote_events(quotes, progress)
    started = perf_counter()
    if progress is not None:
        progress('events', None, None, f'Hashing {len(quotes):,} source rows and full-row repeat flags for cache identity')
    key = _source_key(quotes)
    identity_s = perf_counter() - started
    folder = Path(cache_dir)
    data_path, manifest_path = folder / (key + '.parquet'), folder / (key + '.json')
    status = 'miss'
    if data_path.is_file() and manifest_path.is_file():
        try:
            read_started = perf_counter()
            if progress is not None:
                progress('events', None, None, 'Checking cached file content and reading event Parquet')
            manifest = json.loads(manifest_path.read_text())
            if manifest['key'] != key or manifest['schema'] != SCHEMA_VERSION or manifest['sha256'] != _file_hash(data_path):
                raise ValueError('Cache identity or file content changed')
            events = pd.read_parquet(data_path)
            read_s = perf_counter() - read_started
            decode_started = perf_counter()
            if progress is not None:
                progress('events', None, None, f'Decoding exact candidate sets for {len(events):,} cached events')
            for col in ['spread_set', 'quantity_set']:
                events[col] = events[col].map(lambda value: tuple(json.loads(value)))
            events['pair_set'] = events.pair_set.map(lambda value: frozenset(tuple(pair) for pair in json.loads(value)))
            if len(events) != manifest['events']:
                raise ValueError('Cache event count changed')
            if progress is not None:
                progress('events', len(events), len(events), 'Reused checked narrow-event cache')
            return dict(events=events, unkeyed=manifest['unkeyed'], cache_key=key, cache_status='hit',
                        timings={'cache_identity_s': identity_s, 'cache_read_s': read_s,
                                 'cache_decode_s': perf_counter() - decode_started,
                                 'cache_load_s': perf_counter() - started})
        except (OSError, ValueError, KeyError, TypeError):
            status = 'invalid; rebuilt'
    result = core.prepare_quote_events(quotes, progress)
    folder.mkdir(parents=True, exist_ok=True)
    encode_started = perf_counter()
    if progress is not None:
        progress('events', None, None, f'Encoding exact candidate sets for {len(result["events"]):,} events')
    export = result['events'].copy()
    for col in ['spread_set', 'quantity_set']:
        export[col] = export[col].map(lambda values: json.dumps(list(values), separators=(',', ':')))
    export['pair_set'] = export.pair_set.map(lambda values: json.dumps(sorted(list(values), key=repr), separators=(',', ':')))
    encode_s = perf_counter() - encode_started
    write_started = perf_counter()
    if progress is not None:
        progress('events', None, None, 'Writing event Parquet and verifying its content')
    with tempfile.NamedTemporaryFile(dir=folder, suffix='.parquet', delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        export.to_parquet(temporary_path, index=False)
        digest = _file_hash(temporary_path)
        temporary_path.replace(data_path)
        manifest = dict(key=key, schema=SCHEMA_VERSION, sha256=digest,
                        events=len(export), unkeyed=result['unkeyed'])
        with tempfile.NamedTemporaryFile(dir=folder, mode='w', suffix='.json', delete=False) as temporary:
            json.dump(manifest, temporary)
            temporary_manifest = Path(temporary.name)
        temporary_manifest.replace(manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    result.update(cache_key=key, cache_status=status)
    result['timings'].update(cache_identity_s=identity_s, cache_encode_s=encode_s,
                             cache_write_s=perf_counter() - write_started)
    result['timings']['cache_total_s'] = perf_counter() - started
    if progress is not None:
        progress('events', len(export), len(export), 'Checked event cache ready')
    return result


cached_prepare_quote_events = prepare_quote_events
