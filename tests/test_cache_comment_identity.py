"""Comments preserve existing event caches; real logic changes still invalidate."""
# TEST SETUP LOGIC: Tiny synthetic quotes and source-only mutations; never train.
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import quote_quality_cache as cache
import quote_quality_core as core

# TEST FIXTURE LOGIC: Reproduce the pre-annotation key contract independently.
def legacy_key(quotes, raw_core_hash):
    narrow = quotes[core.KEYS + ['spread', 'quantity']].copy()
    narrow['source_repeat'] = quotes.duplicated()
    rows = pd.util.hash_pandas_object(narrow, index=False).to_numpy(dtype='uint64')
    digest = hashlib.sha256(np.sort(rows).tobytes())
    digest.update(json.dumps([(str(c), str(narrow[c].dtype)) for c in narrow], separators=(',', ':')).encode())
    digest.update(raw_core_hash.encode())
    digest.update(str(cache.SCHEMA_VERSION).encode())
    return digest.hexdigest()

# TEST FIXTURE LOGIC: One genuine two-candidate event with a repeated raw row.
def quotes():
    data = pd.DataFrame(dict(firm=['D1', 'D1'], cusip=['A', 'A'], side=['bid', 'bid'],
        quote_timestamp_ET=[pd.Timestamp('2026-03-02 10:00', tz='America/New_York')] * 2,
        spread=[-1., 2.], quantity=[0, 2]))
    return pd.concat([data, data.iloc[:1]], ignore_index=True)

# TEST LOGIC: Validate behavior and actual disk reuse, not only matching helper output.
class CommentCacheIdentityChecks(unittest.TestCase):
    def test_current_core_retains_both_legacy_keys(self):
        self.assertEqual(cache._core_identity(), cache._LEGACY_CORE_SHA256)
        expected = [legacy_key(quotes(), digest) for digest in
                    [cache._LEGACY_CORE_SHA256, cache._LEGACY_CORE_CRLF_SHA256]]
        self.assertEqual(cache._source_keys(quotes()), expected)

    def test_comment_changes_reuse_a_cache_written_with_old_key(self):
        source = Path(core.__file__).read_text()
        for raw_digest in [cache._LEGACY_CORE_SHA256, cache._LEGACY_CORE_CRLF_SHA256]:
            with self.subTest(line_ending_hash=raw_digest), tempfile.TemporaryDirectory() as folder:
                # TEST FIXTURE LOGIC: Force the exact old raw-byte key at cache creation.
                old_key = legacy_key(quotes(), raw_digest)
                with patch.object(cache, '_source_keys', return_value=[old_key]):
                    first = cache.prepare_quote_events(quotes(), cache_dir=folder)
                changed_source = Path(folder) / 'core_with_comments.py'
                changed_source.write_text('# Extra reader note\n\n' + source + '\n# More explanation\n')
                with patch.object(core, '__file__', str(changed_source)), patch.object(
                        core, 'prepare_quote_events', side_effect=AssertionError('Existing event cache must be reused')):
                    second = cache.prepare_quote_events(quotes(), cache_dir=folder)
                self.assertEqual(second['cache_status'], 'hit')
                self.assertEqual(second['cache_key'], old_key)
                pd.testing.assert_frame_equal(first['events'], second['events'])

    def test_real_code_changes_invalidate_and_later_comments_stay_stable(self):
        source = Path(core.__file__).read_text()
        old_key = cache._source_key(quotes())
        with tempfile.TemporaryDirectory() as folder:
            changed_source = Path(folder) / 'changed_core.py'
            # TEST FIXTURE LOGIC: A real new assignment changes executable structure.
            changed_source.write_text(source + '\nCACHE_IDENTITY_REGRESSION_PROBE = 1\n')
            with patch.object(core, '__file__', str(changed_source)):
                key = cache._source_key(quotes())
                identity = cache._core_identity()
                changed_source.write_text('# explanatory note\n' + changed_source.read_text())
                self.assertEqual(cache._source_key(quotes()), key)
                self.assertEqual(cache._core_identity(), identity)
            self.assertNotEqual(key, old_key)
            self.assertTrue(identity.startswith('ast-v1:'))
        self.assertNotEqual(cache._core_syntax_hash('threshold = 30'), cache._core_syntax_hash('threshold = 60'))
        self.assertNotEqual(cache._core_syntax_hash('x = a + b'), cache._core_syntax_hash('x = a - b'))

    def test_whitespace_line_endings_and_new_empty_ast_field(self):
        self.assertEqual(cache._core_syntax_hash('x=1\n'), cache._core_syntax_hash('x = 1  # note\r\n'))
        tree = ast.parse('def f():\n    return 1\n')
        before = cache._syntax_payload(tree)
        function = tree.body[0]
        function._fields = tuple(field for field in function._fields if field != 'type_params') + ('type_params',)
        function.type_params = []
        self.assertEqual(cache._syntax_payload(tree), before)
        function.type_params = [ast.Name(id='T', ctx=ast.Load())]
        self.assertNotEqual(cache._syntax_payload(tree), before)

# TEST RUNNER LOGIC: Standard unittest entry point.
if __name__ == '__main__':
    unittest.main()
