"""Content-checked narrow event cache shared by research notebooks. No pickle."""
# SETUP LOGIC: 缓存基础设施；不改变事件/quote计算。
from pathlib import Path
import ast
import hashlib
import json
import tempfile
from time import perf_counter
import numpy as np
import pandas as pd
import quote_quality_core as core

# CACHEING LOGIC: 兼容25e6f3c原core的现有缓存键；两种hash必须一起匹配。
# 原字节hash保留旧文件名，AST hash确认现在的可执行结构确实相同。
# 注释/空白变更允许复用；函数体、常量、docstring或schema变更仍使身份改变。
SCHEMA_VERSION = 1
_LEGACY_CORE_SHA256 = 'aa51280245fc529fc3bd28637cb615f470da1915a3e2124658f63ebf73b0b24b'
_LEGACY_CORE_CRLF_SHA256 = 'fa06f055b24aa8fcd451cf4b6a7038d0008579243408825c36e02aad9bbc7ebb'
_LEGACY_CORE_SYNTAX_SHA256 = 'd7ff09dd42fbdd72b668f44e65378aea1254fa1544accc8037f1c849d37adf30'


# CACHEING LOGIC: 源码的结构化身份，不包含行号/注释；不读取或执行数据代码。
def _syntax_payload(node):
    # Input: ast.parse("x=1 # note") 与 ast.parse("x = 1")。
    # Output: 两者相同的[Module, {body: ...}]结构；x=2的Constant值不同。
    # Trick: Python 3.12新增空type_params；跳过空字段以保持3.10+兼容，非空必须保留。
    if isinstance(node, ast.AST):
        return [type(node).__name__, {name: _syntax_payload(value) for name, value in ast.iter_fields(node)
                 if not (name == 'type_params' and value == [])}]
    if isinstance(node, list):
        return [_syntax_payload(item) for item in node]
    return node


# CACHEING LOGIC: 对canonical AST做SHA256；函数不执行待检查的源文件。
def _core_syntax_hash(source):
    syntax = _syntax_payload(ast.parse(source))
    encoded = json.dumps(syntax, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


# CACHEING LOGIC: 语义相同的老core保留旧身份；其他逻辑版本使用新的AST身份。
def _core_identity():
    # Input: 旧core只加CORE LOGIC说明；或把某个阈值30改成60。
    # Output: 前者返回_LEGACY_CORE_SHA256，后者返回不同的ast-v1:<digest>。
    # Trick: 先核对AST再返回旧hash，绝不是无条件固定缓存键。
    source = Path(core.__file__).read_text(encoding='utf-8')
    syntax = _core_syntax_hash(source)
    if syntax == _LEGACY_CORE_SYNTAX_SHA256:
        return _LEGACY_CORE_SHA256
    return 'ast-v1:' + syntax


# CACHEING LOGIC: 流式校验实际Parquet文件字节，仍检测损坏/截断。
def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


# CACHEING LOGIC: 内容+完整重复标记+dtype+core身份+schema共同决定key。
def _source_keys(quotes):
    narrow = quotes[core.KEYS + ['spread', 'quantity']].copy()
    # Full source-row duplicate semantics remain intact even though unrelated
    # metadata does not travel through the cached event table.
    narrow['source_repeat'] = quotes.duplicated()
    rows = pd.util.hash_pandas_object(narrow, index=False).to_numpy(dtype='uint64')
    h = hashlib.sha256(np.sort(rows).tobytes())
    h.update(json.dumps([(str(c), str(narrow[c].dtype)) for c in narrow], separators=(',', ':')).encode())
    # CACHEING LOGIC: 大表仅hash一次；LF/CRLF旧源码身份只复制小hash状态。
    # Input: 当前AST等同交接core，旧Windows checkout使用CRLF。
    # Output: keys=[旧LF key,旧CRLF key]；两个key都绑定相同source/dtype/schema。
    identity = _core_identity()
    identities = [identity]
    if identity == _LEGACY_CORE_SHA256:
        identities.append(_LEGACY_CORE_CRLF_SHA256)
    keys = []
    for candidate in identities:
        digest = h.copy()
        digest.update(candidate.encode())
        digest.update(str(SCHEMA_VERSION).encode())
        keys.append(digest.hexdigest())
    return keys


# CACHEING LOGIC: 保留现有单key接口；新写入默认使用canonical LF/AST身份。
def _source_key(quotes):
    return _source_keys(quotes)[0]


# CACHEING LOGIC: 检查旧缓存、读取并精确解码；未命中才构建及原子替换文件。
def prepare_quote_events(quotes, progress=None, cache_dir='outputs/quote_quality_cache'):
    """Reuse only events from the same source/duplicate semantics and core code.

    Raw rows stay in the caller. Distinct sets are JSON in Parquet; timestamps
    retain nanosecond precision and ET timezone. Set cache_dir=False to bypass.
    """
    # CACHEING LOGIC: 显式False绕过持久缓存；默认先算source身份。
    if cache_dir is False:
        return core.prepare_quote_events(quotes, progress)
    started = perf_counter()
    if progress is not None:
        progress('events', None, None, f'Hashing {len(quotes):,} source rows and full-row repeat flags for cache identity')
    keys = _source_keys(quotes)
    identity_s = perf_counter() - started
    folder = Path(cache_dir)
    # CACHEING LOGIC: 两种旧换行cache都可恢复，随后仍校验manifest与文件内容。
    # 一个已存在的旧CRLF缓存被复用时保留其key，保证原Step5 provenance一致。
    key = next((candidate for candidate in keys
                if (folder / (candidate + '.parquet')).is_file()
                and (folder / (candidate + '.json')).is_file()), keys[0])
    data_path, manifest_path = folder / (key + '.parquet'), folder / (key + '.json')
    status = 'miss'
    # CACHEING LOGIC: manifest与文件SHA256同时通过才复用。
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
            # CACHEING LOGIC: JSON列表恢复tuple/frozenset；候选值和ns timestamp不做四舍五入。
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
    # CACHEING LOGIC: 未命中或内容损坏才调用核心聚合；记录miss/invalid状态。
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
    # FILE IO LOGIC: 临时文件写完并取hash后replace；manifest最后替换。
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
    # PROGRESS LOGIC: 分开hash/encode/write成本，完成回调不意味着模型已有增益。
    result.update(cache_key=key, cache_status=status)
    result['timings'].update(cache_identity_s=identity_s, cache_encode_s=encode_s,
                             cache_write_s=perf_counter() - write_started)
    result['timings']['cache_total_s'] = perf_counter() - started
    if progress is not None:
        progress('events', len(export), len(export), 'Checked event cache ready')
    return result


# SETUP LOGIC: 保留已有公共别名。
cached_prepare_quote_events = prepare_quote_events
