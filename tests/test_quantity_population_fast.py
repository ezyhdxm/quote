"""Step 1 must summarize quantities without starting the history/cache pipeline."""
import time
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from quote_quality_core import prepare_quote_events
from quote_quality_population import (
    attach_quote_metadata, build_bond_universe, population_tables,
    quantity_population_tables, scope_selection, summary_html,
)
from tests.test_quality_population import fixture


HISTORY = {"continuous_event_transitions", "unchanged_pair_refresh_events",
           "changed_spread_events", "changed_condition_events"}


class FastQuantityChecks(unittest.TestCase):
    def test_descriptive_tables_match_full_path_without_history(self):
        trades, quotes = fixture()
        raw = attach_quote_metadata(quotes, build_bond_universe(trades))
        full = population_tables(trades, quotes, event_cache=prepare_quote_events(raw))
        fast = quantity_population_tables(trades, quotes)
        for scope in ["Global", "SECTOR", "Issuer", "Dealer"]:
            expected = full["tables"][scope].drop(columns=list(HISTORY)).sort_index()
            actual = fast["tables"][scope].drop(columns=list(HISTORY)).sort_index()
            pd.testing.assert_frame_equal(actual.reindex(columns=expected.columns), expected,
                                          check_dtype=False, check_index_type=False)
            self.assertTrue(fast["tables"][scope][list(HISTORY)].isna().all().all())
        self.assertIsNone(fast["event_cache"])
        text = summary_html(scope_selection(fast))
        self.assertIn("Refresh/change history was not evaluated", text)
        self.assertNotIn("Unchanged pair refresh=0", text)

    def test_same_positive_quantity_multi_exact_timestamps_and_incomplete_rows(self):
        base = pd.Timestamp("2026-03-02 10:00", tz="America/New_York")
        trades = pd.DataFrame({"CUSIP": ["A", "NO_QUOTE"], "ISSUER": ["Alpha", "Beta"],
                               "SECTOR": ["Bank", None], "EFFECTIVE_DATETIME_TS": [base, base]})
        rows = [
            (base, -1, 5), (base, 0, 5), (base, -1, 5),
            (base + pd.Timedelta(minutes=1), 10, 1), (base + pd.Timedelta(minutes=1), 20, 2),
            (base + pd.Timedelta(minutes=2), 30, 7), (base + pd.Timedelta(minutes=2), 40, 7),
            (base + pd.Timedelta(minutes=2), np.inf, 7),
            (base + pd.Timedelta(minutes=3), 50, 0), (base + pd.Timedelta(minutes=3), 60, None),
            (base + pd.Timedelta(minutes=3, nanoseconds=1), 70, -1),
        ]
        quotes = pd.DataFrame(rows, columns=["quote_timestamp_ET", "spread", "quantity"])
        quotes["cusip"], quotes["firm"], quotes["side"] = "A", "D", "bid"
        p = quantity_population_tables(trades, quotes)
        g = p["events"].sort_values("quote_timestamp_ET")
        self.assertEqual(len(g), 5)
        self.assertEqual(g.same_positive_multi.tolist(), [True, False, True, False, False])
        self.assertEqual(g.multi.tolist(), [True, True, True, True, False])
        self.assertEqual(g.complete.tolist(), [True, True, False, True, True])
        self.assertEqual(g.qclass.tolist(), ["Same positive", "Different positive", "Same positive",
                                            "Missing / other", "Missing / other"])
        s = p["tables"]["Global"].iloc[0]
        self.assertEqual((s.raw_rows, s.events, s.same_positive_multi_events, s.repeats), (11, 5, 2, 1))
        self.assertEqual((s.no_quote_bonds, s.raw_zero_rows, s.raw_missing_rows, s.raw_other_rows), (1, 1, 1, 1))
        self.assertEqual(p["raw"].spread.eq(-1).sum(), 2)

    def test_sector_unknown_conflicting_and_no_quote_input(self):
        trades, quotes = fixture()
        p = quantity_population_tables(trades, quotes.iloc[:0])
        s = p["tables"]["Global"].iloc[0]
        self.assertEqual((s.traded_bonds, s.no_quote_bonds, s.events), (4, 4, 0))
        self.assertEqual(s.unknown_quote_window_trade_days, 6)
        self.assertEqual(set(p["tables"]["SECTOR"].index), {"Energy", "Conflicting", "Unknown", "Utility"})
        self.assertEqual(p["tables"]["Dealer"].index.tolist(), ["[No observed dealer]"])
        self.assertIn("not evaluated", summary_html(scope_selection(p)))

    def test_unkeyed_quantity_rows_remain_in_raw_denominators(self):
        trades, quotes = fixture()
        p = quantity_population_tables(trades, quotes)
        self.assertEqual(p["unkeyed"], 1)
        missing = p["tables"]["Dealer"].loc["[Missing dealer]"]
        self.assertEqual((missing.raw_rows, missing.raw_other_rows, missing.events, missing.unkeyed_rows), (1, 1, 0, 1))
        self.assertEqual(missing.traded_bonds, 4)

    def test_large_integer_quantity_uses_existing_float_condition_semantics(self):
        base = pd.Timestamp("2026-03-02 10:00", tz="America/New_York")
        trades = pd.DataFrame({"CUSIP": ["A"], "ISSUER": ["Alpha"], "EFFECTIVE_DATETIME_TS": [base]})
        quotes = pd.DataFrame({"cusip": ["A", "A"], "firm": ["D", "D"], "side": ["bid", "bid"],
            "quote_timestamp_ET": [base, base], "spread": [10, 20], "quantity": [2**54, 2**54 + 1]})
        p = quantity_population_tables(trades, quotes)
        self.assertEqual(p["events"].n_quantity.iloc[0], 1)
        self.assertTrue(p["events"].same_positive_multi.iloc[0])

    def test_high_cardinality_path_never_calls_history_cache_or_per_group_summary(self):
        n = 10000
        base = pd.Timestamp("2026-03-02 10:00", tz="America/New_York")
        cusips = [f"B{i}" for i in range(n)]
        trades = pd.DataFrame({"CUSIP": cusips, "ISSUER": [f"Issuer{i}" for i in range(n)],
            "SECTOR": np.where(np.arange(n) % 2, "Bank", "Utility"), "EFFECTIVE_DATETIME_TS": base})
        quotes = pd.DataFrame({"cusip": cusips, "firm": [f"D{i % 100}" for i in range(n)],
            "side": "bid", "quote_timestamp_ET": base, "spread": np.arange(n), "quantity": 0})
        with patch("quote_quality_core.event_history", side_effect=AssertionError("no history")), \
             patch("quote_quality_cache.prepare_quote_events", side_effect=AssertionError("no cache")), \
             patch("quote_quality_cache._source_key", side_effect=AssertionError("no source hash")), \
             patch("quote_quality_population._summary_row", side_effect=AssertionError("no per-scope row loop")):
            start = time.perf_counter()
            p = quantity_population_tables(trades, quotes)
            elapsed = time.perf_counter() - start
        self.assertEqual(p["tables"]["Global"].iloc[0].events, n)
        self.assertEqual(len(p["tables"]["Issuer"]), n)
        # A generous ceiling catches accidental O(issuers x events) scans while
        # allowing slower test machines; correctness never depends on timing.
        self.assertLess(elapsed, 10)


if __name__ == "__main__":
    unittest.main()
