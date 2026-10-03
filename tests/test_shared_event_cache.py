"""Persistent event reuse preserves exact candidate/history/duplicate semantics."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import quote_quality_core as core
import quote_quality_cache as cache


def raw():
    t = pd.Timestamp('2026-03-02 10:00', tz='America/New_York')
    q = pd.DataFrame([dict(firm='A', cusip='X', side='bid', quantity=size, spread=spread,
                           quote_timestamp_ET=t + pd.Timedelta(nanoseconds=offset), note=note)
                      for offset, spread, size, note in [(0, -1., 0, 'a'), (0, 2., 2, 'a'),
                                                         (1, np.nan, 'other', 'a'), (2, 3., None, 'a')]])
    return pd.concat([q, q.iloc[:1]], ignore_index=True)


class SharedCacheChecks(unittest.TestCase):
    def test_roundtrip_reuses_and_keeps_exact_nanosecond_sets(self):
        q = raw()
        with tempfile.TemporaryDirectory() as folder:
            first = cache.prepare_quote_events(q, cache_dir=folder)
            with patch.object(core, 'prepare_quote_events', side_effect=AssertionError('must reuse')):
                second = cache.prepare_quote_events(q, cache_dir=folder)
            pd.testing.assert_frame_equal(first['events'], second['events'])
            self.assertEqual(second['cache_status'], 'hit')
            self.assertEqual(first['events'].iloc[0].repeats, 1)
            self.assertFalse(first['events'].iloc[1].complete)
            self.assertEqual(len(second['events']), 3)

    def test_source_code_and_corrupt_cache_invalidate(self):
        q = raw()
        with tempfile.TemporaryDirectory() as folder:
            first = cache.prepare_quote_events(q, cache_dir=folder)
            changed = q.copy(); changed.loc[0, 'spread'] = 5
            second = cache.prepare_quote_events(changed, cache_dir=folder)
            self.assertNotEqual(first['cache_key'], second['cache_key'])
            file = Path(folder) / (first['cache_key'] + '.parquet')
            file.write_bytes(b'truncated')
            third = cache.prepare_quote_events(q, cache_dir=folder)
            self.assertEqual(third['cache_status'], 'invalid; rebuilt')
            pd.testing.assert_frame_equal(first['events'], third['events'])
            with patch.object(cache, 'SCHEMA_VERSION', cache.SCHEMA_VERSION + 1):
                fourth = cache.prepare_quote_events(q, cache_dir=folder)
            self.assertNotEqual(first['cache_key'], fourth['cache_key'])

    def test_metadata_that_changes_repeat_count_invalidates(self):
        q = raw()
        with tempfile.TemporaryDirectory() as folder:
            first = cache.prepare_quote_events(q, cache_dir=folder)
            same = cache.prepare_quote_events(q.assign(SECTOR='Tech'), cache_dir=folder)
            self.assertEqual(first['cache_key'], same['cache_key'])
            changed = q.copy(); changed.loc[4, 'note'] = 'different'
            second = cache.prepare_quote_events(changed, cache_dir=folder)
            self.assertNotEqual(first['cache_key'], second['cache_key'])
            self.assertEqual(second['events'].iloc[0].repeats, 0)


if __name__ == '__main__': unittest.main()
