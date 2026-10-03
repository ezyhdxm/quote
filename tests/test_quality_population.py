"""Population denominators, reproducible cases and measured rule sensitivity."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from quote_quality_core import event_history, prepare_quote_events
from quote_quality_population import (
    build_bond_universe, population_tables, scope_selection, fixed_case_manifest,
    measure_rule_impacts, case_options,
)
from tests.test_quality_step1 import load_dashboard as load_step1
from tests.test_quality_step2 import load_dashboard as load_step2


def fixture():
    trades = pd.DataFrame({
        "CUSIP": ["A", "A", "B", "B", "C", "D", "D"],
        "ISSUER": ["Alpha", "Alpha", "Beta", "Beta", "Gamma", "Delta", "Delta"],
        "SECTOR": ["Energy", None, "Bank", "Financial", None, "Utility", "Utility"],
        "EFFECTIVE_DATETIME_TS": pd.to_datetime([
            "2026-01-02", "2026-03-02", "2026-03-02", "2026-03-02", "2026-03-02", "2026-03-02", "2026-01-02"]),
    })
    base = pd.Timestamp("2026-03-02T15:00:00Z")
    quotes = pd.DataFrame([
        ("A", "d1", "bid", base, -10, 5),
        ("A", "d1", "bid", base, 0, 5),
        ("A", "d1", "bid", base, -10, 5),
        ("A", "d1", "bid", base + pd.Timedelta(minutes=1), -10, 5),
        ("A", "d1", "bid", base + pd.Timedelta(minutes=1), 0, 5),
        ("A", "d2", "ask", base, 30, 0),
        ("B", "d2", "bid", base, 40, None),
        ("B", None, "ask", base, 50, -1),
        ("OUT", "d3", "bid", base, 999, 1),
    ], columns=["cusip", "firm", "side", "quote_timestamp_ET", "spread", "quantity"])
    quotes["quote_timestamp_ET"] = quotes["quote_timestamp_ET"].dt.tz_convert("America/New_York")
    quotes["quote_timestamp_UTC"] = quotes["quote_timestamp_ET"].dt.tz_convert("UTC").astype(str)
    return trades, quotes


class PopulationChecks(unittest.TestCase):
    def setUp(self):
        self.event_prepare_patch = patch("quote_quality_cache.prepare_quote_events", side_effect=prepare_quote_events)
        self.event_prepare_patch.start()
        self.addCleanup(self.event_prepare_patch.stop)

    def test_sector_one_nonnull_unknown_conflicting_and_no_quote_universe(self):
        trades, quotes = fixture()
        u = build_bond_universe(trades).set_index("cusip")
        self.assertEqual(u.SECTOR.to_dict(), {"A": "Energy", "B": "Conflicting", "C": "Unknown", "D": "Utility"})
        pd.testing.assert_frame_equal(build_bond_universe(trades), build_bond_universe(trades.sample(frac=1, random_state=17)))
        p = population_tables(trades, quotes)
        g = p["tables"]["Global"].loc["Global"]
        self.assertEqual((g.traded_bonds, g.quoted_bonds, g.no_quote_bonds), (4, 2, 2))
        self.assertEqual((g.raw_rows, g.events, g.dealer_bond_side_days), (8, 4, 3))
        self.assertEqual(g.multi_events, 2)
        self.assertEqual(g.same_positive_multi_events, 2)
        self.assertEqual(g.affected_bond_days, 1)
        self.assertEqual(g.unchanged_pair_refresh_events, 1)
        self.assertEqual(g.continuous_event_transitions, 1)
        self.assertEqual((g.traded_bond_days, g.quote_covered_traded_bond_days), (4, 2))
        self.assertEqual(g.outside_file_window_trade_days, 2)
        self.assertEqual(p["tables"]["SECTOR"].loc["Utility", "no_quote_bonds"], 1)
        self.assertEqual(p["tables"]["Issuer"].loc["Gamma", "events"], 0)
        self.assertEqual(p["tables"]["Dealer"].loc["d1", "traded_bonds"], 4)
        # Equal issuer and dealer estimands differ from event weights.
        self.assertAlmostEqual(g.multi_event_rate, 0.5)
        self.assertAlmostEqual(g.issuer_equal_multi_rate, (2 / 3 + 0) / 2)
        self.assertAlmostEqual(g.dealer_equal_multi_rate, (1 + 0) / 2)
        self.assertAlmostEqual(g.unit_equal_multi_rate, 1 / 3)

    def test_reused_events_and_missing_keys_are_not_relabelled_into_events(self):
        trades, quotes = fixture()
        raw = quotes.loc[quotes.cusip.isin(trades.CUSIP)]
        e = event_history(raw)
        cache = {"events": e["events"], "unkeyed": e["unkeyed"]}
        with patch("quote_quality_core.prepare_quote_events", side_effect=AssertionError("must reuse")):
            p = population_tables(trades, quotes, cache)
        self.assertIs(p["event_cache"], cache)
        self.assertEqual(p["unkeyed"], 1)
        self.assertEqual(p["tables"]["Global"].loc["Global", "unkeyed_rows"], 1)
        self.assertEqual(scope_selection(p, "Issuer", "Gamma")["groups"].shape[0], 0)

    def test_entirely_no_quote_input_keeps_universe_and_empty_case_manifest(self):
        trades, quotes = fixture()
        p = population_tables(trades, quotes.iloc[:0])
        summary = p["tables"]["Global"].loc["Global"]
        self.assertEqual((summary.traded_bonds, summary.no_quote_bonds, summary.events), (4, 4, 0))
        self.assertEqual(summary.outside_file_window_trade_days, 0)
        self.assertEqual(summary.unknown_quote_window_trade_days, 6)
        self.assertTrue(fixed_case_manifest(p).empty)
        self.assertEqual(case_options(p["case_manifest"]), [])
        dealer = scope_selection(p, "Dealer", "[No observed dealer]")
        self.assertEqual(dealer["summary"].no_quote_bonds, 4)
        state, _ = load_step2(quotes=quotes.iloc[:0], trades=trades, issuer=None)
        state["freeze_cases"]()
        self.assertIsNone(state["fixed_case_box"].value)

    def test_actual_impact_uses_same_local_queries_and_preserves_zero_negative(self):
        trades, quotes = fixture()
        quotes = quotes.iloc[[0]].copy()
        quotes["spread"], quotes["quantity"] = 100, 5
        base = quotes.quote_timestamp_ET.iloc[0]
        for dealer in ["p1", "p2", "p3"]:
            r = quotes.iloc[[0]].copy()
            r["firm"], r["spread"] = dealer, 0
            quotes = pd.concat([quotes, r], ignore_index=True)
        later = quotes.iloc[[0]].copy()
        later["quote_timestamp_ET"] = base + pd.Timedelta(hours=2)
        later["quote_timestamp_UTC"] = later.quote_timestamp_ET.dt.tz_convert("UTC").astype(str)
        quotes = pd.concat([quotes, later], ignore_index=True)
        p = population_tables(trades, quotes)
        impacts = measure_rule_impacts(p["events"])
        self.assertEqual(len(impacts), 1)
        r = impacts.iloc[0]
        self.assertEqual(r.impact_queries, 2)  # Duplicate first/middle/last times removed.
        self.assertGreater(r.max_center_effect_bps, 0)
        self.assertEqual(r.max_age_lost_slots, 3)
        self.assertGreater(r.peer_supported_queries, 0)
        self.assertEqual(p["raw"].spread.eq(0).sum(), 3)

    def test_frozen_cases_stable_hash_strata_and_no_invented_high_impact(self):
        trades, rows = [], []
        base = pd.Timestamp("2026-03-02 10:00", tz="America/New_York")
        for i in range(24):
            bond = f"B{i:02}"
            trades.append((bond, f"Issuer{i:02}", ["Energy", "Utility", "Bank"][i % 3], base))
            rows.append((bond, f"Dealer{i:02}", "bid", base, 10 + i, None if i % 4 == 0 else 0))
            if i % 2:
                rows.append((bond, f"Dealer{i:02}", "bid", base + pd.Timedelta(minutes=1), 12 + i, 5))
        t = pd.DataFrame(trades, columns=["CUSIP", "ISSUER", "SECTOR", "EFFECTIVE_DATETIME_TS"])
        q = pd.DataFrame(rows, columns=["cusip", "firm", "side", "quote_timestamp_ET", "spread", "quantity"])
        p1 = population_tables(t, q)
        impacts = measure_rule_impacts(p1["events"])
        # No changed rule center/coverage: random+typical only, never gap ranked.
        m1 = fixed_case_manifest(p1, impacts)
        self.assertEqual(m1.selection.value_counts().to_dict(), {"Random": 6, "Typical": 6})
        self.assertTrue((m1.loc[m1.selection.eq("Random"), "unknown_rate"] > 0).any())
        self.assertTrue(m1.one_sided_day.all())
        p2 = population_tables(t.sample(frac=1, random_state=12), q.sample(frac=1, random_state=7))
        m2 = fixed_case_manifest(p2, impacts)
        self.assertEqual(m1.case_id.tolist(), m2.case_id.tolist())
        self.assertEqual(case_options(m1)[0][1], m1.case_id.iloc[0])
        self.assertFalse(m1.cap_relaxed.any())
        # Only measured positive effects are eligible for high impact.
        impacts["max_center_effect_bps"] = np.arange(1, len(impacts) + 1)
        m3 = fixed_case_manifest(p1, impacts)
        high = m3.loc[m3.selection.eq("High impact")]
        self.assertEqual(len(m3), 18)
        self.assertEqual(len(high), 6)
        self.assertTrue(high.max_center_effect_bps.gt(0).all())
        self.assertEqual(m3.case_id.nunique(), 18)

    def test_global_default_apply_and_issuer_drilldown_without_reaggregating(self):
        for load, result_key, apply_name in [(load_step1, "quantity_result", "apply_quantity"),
                                             (load_step2, "step2_result", "apply_scope")]:
            state, _ = load(issuer=None)
            self.assertEqual(state[result_key]["scope"], "Global")
            events = state["quality_population"]["events"]
            if load is load_step1:
                self.assertIsNone(state["quality_population"]["event_cache"])
            state["scope_box"].value = "SECTOR"
            self.assertEqual(state[result_key]["scope"], "Global")
            state[apply_name]()
            self.assertEqual(state[result_key]["scope"], "SECTOR")
            state["issuer_box"].value = "Beta"
            self.assertEqual(state[result_key]["scope"], "Issuer")
            self.assertIs(events, state["quality_population"]["events"])

    def test_fixed_case_opens_sparse_single_side_event(self):
        state, _ = load_step2(issuer=None)
        state["freeze_cases"]()
        m = state["case_manifest"]
        sparse = m.loc[m.events.le(3)].iloc[0]
        state["fixed_case_box"].value = sparse.case_id
        state["apply_fixed_case"]()
        self.assertEqual(state["view_box"].value, "Case")
        self.assertEqual(state["case_type"].value, "All events")
        self.assertEqual(state["bond_box"].value, sparse.cusip)
        self.assertEqual(state["dealer_box"].value, sparse.firm)


if __name__ == "__main__":
    unittest.main()
