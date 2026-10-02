"""Integration checks for exact-time groups, denominators and case controls."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "quote_quality_step2.py"


def fixture():
    rows = []
    base = pd.Timestamp("2026-03-02T15:00:00Z")

    def group(i, spreads, sizes, firm="A", cusip="BOND_A"):
        for spread, size in zip(spreads, sizes):
            rows.append((cusip, firm, "bid", base + pd.Timedelta(minutes=i), spread, size, "original"))

    group(0, [-1, 0, -1], [5, 5, 5])  # Repeat and multi-spread coexist.
    group(1, [20, 30], [1, 5])
    group(2, [50, 70], [0, 0])
    group(3, [100, 101], [0, 2])
    group(4, [5, 6, 7], [None, 0, 5])
    group(5, [-5, -4], [-2, 0])
    group(6, [10, "bad"], [3, 3])
    group(7, [np.inf, np.nan], [4, 4])
    group(8, [40, 40, 40], [5, 5, 5])
    rows[-1] = (*rows[-1][:-1], "different source field")
    group(9, [8], [1])
    rows.append(("BOND_A", "A", "bid", base + pd.Timedelta(minutes=9, nanoseconds=1), 9, 1, "original"))
    for i in range(100):
        group(i, [20], [5], firm="B")
    group(0, [50], [0], firm="Z", cusip="BOND_B")
    group(0, [60], [0], firm=None, cusip="BOND_C")
    group(0, [9999], [5], cusip="OUTSIDE")
    quotes = pd.DataFrame(rows, columns=["cusip", "firm", "side", "quote_timestamp_UTC", "spread", "quantity", "extra"])
    quotes["quote_timestamp_UTC"] = quotes["quote_timestamp_UTC"].astype(str)
    trades = pd.DataFrame({"CUSIP": ["BOND_A", "BOND_B", "BOND_C"],
                           "ISSUER": ["Alpha", "Beta", "Gamma"],
                           "EFFECTIVE_DATETIME_TS": pd.to_datetime(["2026-01-02"] * 3)})
    return quotes, trades


def load_dashboard(quotes=None, trades=None):
    default_quotes, default_trades = fixture()
    quotes = default_quotes if quotes is None else quotes
    trades = default_trades if trades is None else trades

    def read(path, *args, **kwargs):
        return (trades if Path(path).name == "data_ig.parquet" else quotes).copy(deep=True)

    state = {"__name__": "step2_test"}
    with contextlib.redirect_stdout(io.StringIO()), patch("pandas.read_parquet", side_effect=read), \
            patch.object(Path, "exists", return_value=True), \
            patch("IPython.display.display") as show, patch("IPython.display.clear_output"):
        exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), state)
    return state, show


class MultiSpreadChecks(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_exact_timestamp_and_group_denominators_without_cleaning(self):
        state, _ = load_dashboard()
        result = state["step2_result"]
        g = result["groups"]
        self.assertEqual(len(g), 111)
        self.assertEqual(int(g["multi"].sum()), 6)
        self.assertEqual(result["day_count"], 2)
        self.assertAlmostEqual(result["day_balanced_rate"], (6 / 11) / 2)
        self.assertEqual(result["affected_day_rate"], 0.5)
        self.assertEqual(len(g.loc[g["firm"].eq("A")]), 11)
        self.assertEqual(int(result["raw"]["spread"].eq(-1).sum()), 2)
        self.assertEqual(int(result["raw"]["spread"].eq(0).sum()), 1)
        self.assertNotIn("OUTSIDE", state["bcq_df"]["cusip"].tolist())
        self.assertEqual(str(g["quote_timestamp_ET"].dt.tz), "America/New_York")
        self.assertEqual(g["quote_timestamp_ET"].iloc[0].hour, 10)

    def test_exact_duplicates_use_all_original_columns_and_keep_raw_rows(self):
        state, _ = load_dashboard()
        r = state["step2_result"]
        self.assertEqual(r["repeated_rows"], 2)
        a = r["groups"].loc[lambda d: d["firm"].eq("A")].reset_index(drop=True)
        self.assertEqual(a.loc[0, "rows"], 3)
        self.assertEqual(a.loc[0, "repeats"], 1)
        self.assertEqual(a.loc[8, "rows"], 3)
        self.assertEqual(a.loc[8, "repeats"], 1)
        self.assertEqual(a.loc[8, "n_spreads"], 1)

    def test_quantity_priority_zero_semantics_and_incomplete_groups(self):
        state, _ = load_dashboard()
        a = state["step2_result"]["groups"].loc[lambda d: d["firm"].eq("A")].reset_index(drop=True)
        self.assertEqual(a["qclass"].iloc[:6].tolist(), [
            "Same positive", "Different positive", "Contains zero", "Contains zero",
            "Missing / other", "Missing / other",
        ])
        self.assertTrue(a.loc[2, "all_zero"])
        self.assertFalse(a.loc[3, "all_zero"])
        self.assertEqual(a.loc[6, "n_spreads"], 1)
        self.assertEqual(a.loc[6, "bad_spreads"], 1)
        self.assertEqual(a.loc[7, "n_spreads"], 0)
        self.assertEqual(a.loc[7, "bad_spreads"], 2)
        self.assertEqual(state["step2_result"]["bad_spread_rows"], 3)
        text = " ".join(t.get_text() for t in state["current_figure"].texts)
        self.assertIn("2 keyed groups", text)
        self.assertIn("unassessed", text)

    def test_widget_case_filters_controls_and_unkeyed_issuer_clear_old_chart(self):
        state, show = load_dashboard()
        state["view_box"].value = "Case"
        self.assertEqual(state["event_box"].value, 0)
        state["case_type"].value = "Different positive"
        event = state["step2_result"]["groups"].loc[state["event_box"].value]
        self.assertEqual(event["qclass"], "Different positive")
        state["case_type"].value = "Incomplete spread"
        self.assertGreater(state["step2_result"]["groups"].loc[state["event_box"].value, "bad_spreads"], 0)
        state["case_type"].value = "Single-spread control"
        self.assertTrue(all(state["step2_result"]["groups"].loc[i, "bad_spreads"] == 0 for _, i in state["event_box"].options))
        state["issuer_box"].value = "Beta"
        state["case_type"].value = "All multi-spread"
        self.assertIsNone(state["event_box"].value)
        state["issuer_box"].value = "Gamma"
        self.assertEqual(state["step2_result"]["unkeyed_rows"], 1)
        self.assertEqual(len(state["step2_result"]["groups"]), 0)
        state["view_box"].value = "Overview"
        self.assertIn("unkeyed rows=1", " ".join(t.get_text() for t in state["current_figure"].texts))
        for call in show.call_args_list:
            self.assertFalse(any(isinstance(a, pd.DataFrame) for a in call.args))

    def test_case_preserves_all_candidates_and_paginated_exact_values(self):
        quotes, trades = fixture()
        extra = pd.concat([quotes.iloc[[0]]] * 17, ignore_index=True)
        extra["spread"] = np.arange(17)
        extra["quantity"] = np.arange(17) + 1
        state, _ = load_dashboard(extra, trades)
        state["view_box"].value = "Case"
        self.assertEqual(tuple(state["values_page"].options), (1, 2, 3))
        state["values_page"].value = 3
        texts = " ".join(t.get_text() for ax in state["current_figure"].axes for t in ax.texts)
        self.assertIn("Values 17-17 of 17", texts)
        self.assertIn("spread=16", texts)
        self.assertEqual(len(state["step2_result"]["raw"]), 17)

    def test_overview_paging_leaves_issuer_statistics_and_range_distribution_intact(self):
        quotes, trades = fixture()
        extra = pd.concat([quotes.iloc[[0]]] * 12, ignore_index=True)
        extra["firm"] = [f"Dealer {i}" for i in range(12)]
        state, _ = load_dashboard(extra, trades)
        first = state["current_figure"]
        state["page_box"].value = 2
        second = state["current_figure"]
        self.assertEqual([t.get_text() for t in first.texts], [t.get_text() for t in second.texts])
        self.assertEqual(len(second.axes[0].patches), 4)
        self.assertEqual(len(state["step2_result"]["groups"]), 12)

    def test_event_pages_expose_all_controls_and_issuer_switch_resets(self):
        state, _ = load_dashboard()
        state["view_box"].value = "Case"
        state["case_type"].value = "Single-spread control"
        state["dealer_box"].value = "B"
        first = {i for _, i in state["event_box"].options}
        self.assertEqual(len(first), 50)
        self.assertEqual(tuple(state["event_page"].options), (1, 2))
        state["event_page"].value = 2
        second = {i for _, i in state["event_box"].options}
        self.assertEqual(len(second), 50)
        self.assertFalse(first & second)
        self.assertIn(state["event_box"].value, second)
        state["issuer_box"].value = "Beta"
        self.assertEqual(state["event_page"].value, 1)
        self.assertEqual(state["dealer_box"].value, "Z")

    def test_same_price_different_quantities_and_incomplete_multi_are_distinct(self):
        quotes, trades = fixture()
        q = pd.concat([quotes.iloc[[0]]] * 8, ignore_index=True)
        q["quantity"] = [1, 5, 1, 1, 1, 1, 1, 1]
        q["spread"] = [10, 10, 20, 30, np.inf, 999, 888, 777]
        q.loc[2:4, "quote_timestamp_UTC"] = "2026-03-02T15:01:00Z"
        q.loc[5, "side"] = "ask"
        q.loc[6, "firm"] = "B"
        q.loc[7, "quote_timestamp_UTC"] = "2026-03-03T15:00:00Z"
        state, _ = load_dashboard(q, trades)
        g = state["step2_result"]["groups"]
        self.assertEqual(len(g), 5)
        first = g.iloc[0]
        self.assertEqual(first["qclass"], "Different positive")
        self.assertFalse(first["multi"])
        incomplete = g.loc[g["bad_spreads"].gt(0)].iloc[0]
        self.assertTrue(incomplete["multi"])
        self.assertEqual(incomplete["n_spreads"], 2)
        self.assertEqual(incomplete["bad_spreads"], 1)


if __name__ == "__main__":
    unittest.main()
