"""Integration checks for the raw-quantity dashboard and included data loader."""
import contextlib
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "quote_quality_step1.py"
CACHE = Path("data/pipeline/data_ig.parquet")
QUOTES = Path("data/bondcliq/quotes_pretrade_260301_260401_Wells_quotes2.parquet")


def fixture():
    """Include ambiguity and repeats without requiring any market data."""
    quantities = [1.0, 2.0, 0.0, -0.0, None, np.nan, pd.NA,
                  -1.0, np.inf, -np.inf, "bad", "", "4"]
    quotes = pd.DataFrame({
        "cusip": "BOND_A", "firm": "A", "quantity": quantities,
        "quote_timestamp_UTC": "2026-03-02T15:00:00Z",
        "spread": [-20.0, 0.0] + [60.0] * (len(quantities) - 2),
        "side": ["bid", "ask"] + ["bid"] * (len(quantities) - 2),
    })
    # The first row is deliberately duplicated, not a second independent quote.
    quotes = pd.concat([quotes, quotes.iloc[[0]]], ignore_index=True)
    extra = [
        ("BOND_A", f"DEALER_{i:02}", 10.0 + i) for i in range(10)
    ] + [
        ("BOND_A", None, 0.0),
        ("BOND_B", "B", 0.0), ("BOND_B", "B", None),
        ("BOND_B", "B", -2.0),
        ("BOND_C", None, 3.0),
        ("OUTSIDE", "EXCLUDED", 999999.0),
    ]
    more = pd.DataFrame(extra, columns=["cusip", "firm", "quantity"])
    more["quote_timestamp_UTC"] = "2026-03-03T15:00:00Z"
    more["spread"] = 65.0
    more["side"] = "ask"
    quotes = pd.concat([quotes, more], ignore_index=True)
    trades = pd.DataFrame({
        "CUSIP": ["BOND_A", "BOND_B", "BOND_C", "TRADE_ONLY"],
        "ISSUER": ["Alpha", "Beta", None, "Alpha"],
        # All trades precede the short quote window, but remain in the universe.
        "EFFECTIVE_DATETIME_TS": pd.to_datetime(["2026-01-02 10:00"] * 4),
    })
    return quotes, trades


def load_dashboard(quotes=None, trades=None, cached=True):
    """Execute all three cells with mock data reads and real widget callbacks."""
    default_quotes, default_trades = fixture()
    quotes = default_quotes if quotes is None else quotes
    trades = default_trades if trades is None else trades
    original_exists = Path.exists
    loader = Mock(return_value=trades.copy(deep=True))
    data_module = types.ModuleType("data")
    data_module.load_merged_prints = loader
    read_paths = []

    def exists(path):
        return cached if path == CACHE else original_exists(path)

    def read_parquet(path, *args, **kwargs):
        read_paths.append(path)
        if path == CACHE:
            return trades.copy(deep=True)
        if path == QUOTES:
            return quotes.copy(deep=True)
        raise AssertionError(f"Unexpected parquet path: {path}")

    namespace = {"__name__": "quantity_step1_test"}
    source = SCRIPT.read_text()
    with contextlib.redirect_stdout(io.StringIO()), \
            patch.object(Path, "exists", exists), \
            patch.object(Path, "mkdir") as mkdir, \
            patch("pandas.read_parquet", side_effect=read_parquet), \
            patch.object(pd.DataFrame, "to_parquet") as write_cache, \
            patch.dict(sys.modules, {"data": data_module}), \
            patch("IPython.display.display") as display, \
            patch("IPython.display.clear_output"):
        exec(compile(source, str(SCRIPT), "exec"), namespace)
    return namespace, {
        "loader": loader, "read_paths": read_paths, "mkdir": mkdir,
        "write_cache": write_cache, "display": display,
    }


class QuantityStep1Checks(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_raw_quantity_categories_are_exhaustive_and_do_not_infer_units(self):
        state, _ = load_dashboard()
        data = state["quantity_data"]
        first = data.iloc[:14]
        self.assertEqual(first["quantity_kind"].tolist(), [
            "Positive", "Positive", "Zero", "Zero", "Missing", "Missing",
            "Missing", "Other", "Other", "Other", "Other", "Other",
            "Positive", "Positive",
        ])
        self.assertEqual(first.loc[first["quantity_kind"].eq("Positive"),
                                   "quantity_numeric"].tolist(), [1.0, 2.0, 4.0, 1.0])
        result = state["quantity_figure"]("Alpha")
        self.assertEqual(result["counts"].loc["A"].to_dict(), {
            "Positive": 4, "Zero": 2, "Missing": 3, "Other": 5,
        })
        self.assertEqual(int(result["totals"].sum()), 25)
        bars = result["figure"].axes[0].containers
        self.assertAlmostEqual(sum(group[0].get_width() for group in bars), 100.0)
        self.assertAlmostEqual(bars[0][0].get_width(), 4 / 14 * 100)

    def test_no_deduplication_or_spread_cleaning_and_trade_universe_is_preserved(self):
        quotes, trades = fixture()
        state, _ = load_dashboard(quotes, trades)
        raw = state["bcq_df"]
        expected = quotes.loc[quotes["cusip"].isin(trades["CUSIP"])]
        self.assertEqual(len(raw), len(expected))
        self.assertEqual(int(raw["spread"].eq(-20.0).sum()), 2)
        self.assertEqual(int(raw["spread"].eq(0.0).sum()), 1)
        self.assertTrue(raw.iloc[[0, 13]].drop(columns="ISSUER").duplicated().iloc[-1])
        self.assertEqual(set(raw["cusip"]), {"BOND_A", "BOND_B", "BOND_C"})
        self.assertNotIn(999999.0, state["quantity_data"]["quantity_numeric"].tolist())
        self.assertEqual(str(raw["quote_timestamp_ET"].dt.tz), "America/New_York")
        self.assertEqual(raw["quote_timestamp_ET"].iloc[0].hour, 10)

    def test_missing_identifiers_remain_visible_and_do_not_drop_denominator(self):
        state, _ = load_dashboard()
        self.assertIn("[Missing issuer]", state["issuer_box"].options)
        alpha = state["quantity_figure"]("Alpha")
        self.assertEqual(alpha["counts"].loc["[Missing dealer]", "Zero"], 1)
        self.assertEqual(int(alpha["counts"].to_numpy().sum()), 25)
        unknown = state["quantity_figure"]("[Missing issuer]")
        self.assertEqual(int(unknown["counts"].loc["[Missing dealer]", "Positive"]), 1)
        self.assertEqual(unknown["positive_values"].tolist(), [3.0])

    def test_paging_changes_only_left_panel_and_issuer_switch_resets_page(self):
        state, calls = load_dashboard()
        first = state["quantity_result"]
        self.assertEqual(first["issuer"], "Alpha")
        self.assertEqual(tuple(state["page_box"].options), (1, 2))
        state["page_box"].value = 2
        second = state["quantity_result"]
        self.assertEqual(second["page"], 2)
        self.assertTrue(set(first["page_dealers"]).isdisjoint(second["page_dealers"]))
        pd.testing.assert_series_equal(first["positive_values"], second["positive_values"])
        pd.testing.assert_series_equal(first["totals"], second["totals"])
        first_bars = [p.get_height() for p in first["figure"].axes[1].patches]
        second_bars = [p.get_height() for p in second["figure"].axes[1].patches]
        self.assertEqual(first_bars, second_bars)
        state["issuer_box"].value = "Beta"
        beta = state["quantity_result"]
        self.assertEqual(beta["issuer"], "Beta")
        self.assertEqual(beta["page"], 1)
        self.assertEqual(tuple(state["page_box"].options), (1,))
        self.assertTrue(beta["positive_values"].empty)
        self.assertIn("No positive quantity in this issuer",
                      [text.get_text() for text in beta["figure"].axes[1].texts])
        for call in calls["display"].call_args_list:
            self.assertFalse(any(isinstance(arg, pd.DataFrame) for arg in call.args))

    def test_full_positive_range_histogram_handles_extremes_without_trimming(self):
        quotes, trades = fixture()
        quantities = [10.0 ** exponent for exponent in range(-12, 13)]
        quotes = pd.concat([quotes.iloc[[0]]] * len(quantities), ignore_index=True)
        quotes["quantity"] = quantities
        state, _ = load_dashboard(quotes, trades)
        result = state["quantity_result"]
        self.assertEqual(result["positive_values"].tolist(), quantities)
        axis = result["figure"].axes[1]
        self.assertIn("log10", axis.get_xlabel())
        self.assertAlmostEqual(sum(patch.get_height() for patch in axis.patches), 100.0)
        self.assertLessEqual(min(p.get_x() for p in axis.patches), -12)
        self.assertGreaterEqual(max(p.get_x() + p.get_width() for p in axis.patches), 12)

    def test_cache_load_uses_user_paths_without_calling_pipeline(self):
        state, calls = load_dashboard(cached=True)
        self.assertEqual(calls["read_paths"], [CACHE, QUOTES])
        calls["loader"].assert_not_called()
        calls["write_cache"].assert_not_called()
        self.assertEqual(state["PIPELINE_CSV"], Path("data/pipeline/data_pipeline.csv_20260506"))
        self.assertEqual(state["BENCHMARK_CSV"],
                         Path("data/pipeline/DailyCloseUSTBenchmarks.csv_20260506"))

    def test_fallback_calls_user_loader_parameters_and_saves_ny_trade_timestamps(self):
        state, calls = load_dashboard(cached=False)
        calls["loader"].assert_called_once_with(
            pipeline_csv=Path("data/pipeline/data_pipeline.csv_20260506"),
            benchmark_csv=Path("data/pipeline/DailyCloseUSTBenchmarks.csv_20260506"),
            liquid_trade_count_threshold=15,
            qav_threshold_ig=0.005, qav_threshold_hy=0.2,
        )
        self.assertEqual(calls["read_paths"], [QUOTES])
        calls["mkdir"].assert_called_once_with(parents=True, exist_ok=True)
        calls["write_cache"].assert_called_once_with(CACHE, index=False)
        timestamps = state["data_ig"]["EFFECTIVE_DATETIME_TS"]
        self.assertEqual(str(timestamps.dt.tz), "America/New_York")
        self.assertEqual(timestamps.iloc[0].hour, 10)
        utc = pd.Series(pd.to_datetime(["2026-01-02T15:00:00Z"]))
        converted = state["to_ny_datetime"](utc)
        self.assertEqual(converted.iloc[0].hour, 10)


if __name__ == "__main__":
    unittest.main()
