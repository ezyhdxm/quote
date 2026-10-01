"""Behavioral checks for the notebook's replay and EDA calculations."""

import contextlib
import io
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fixture():
    times = pd.date_range("2026-03-02 09:55", "2026-03-02 11:00", freq="5min")
    rows = []
    for i, cusip in enumerate(["BOND00001", "BOND00002", "BOND00003"]):
        for j, when in enumerate(times):
            for firm in ["A", "B", "C"]:
                for side, offset in [("bid", 2), ("ask", -2)]:
                    rows.append((cusip, firm, side, when, 60 + i * 10 + j + offset, 1000000))
    q = pd.DataFrame(rows, columns=["cusip", "firm", "side", "quote_timestamp_ET", "spread", "quantity"])
    t = pd.DataFrame({"CUSIP": ["BOND00001", "BOND00002", "BOND00003"],
                      "EFFECTIVE_DATETIME_TS": pd.to_datetime(["2026-02-02 10:00"] * 3),
                      "BM_SPREAD": [.6, .7, .8], "QUANTITY": 1000000, "EFF_SIDE": "D"})
    m = pd.DataFrame({"CUSIP": ["BOND00001", "BOND00002", "BOND00003"],
                      "ISSUER": "TEST", "YRS_TO_MATURITY": [1., 5., 10.]})
    return q, t, m


def replay(q, t, m, **config):
    settings = {"ISSUER": "TEST", "START": "2026-03-02 10:00", "END": "2026-03-02 11:00",
                "SHOW_PLOTS": False, "MIN_COMMON_OBS": 3, **config}
    with contextlib.redirect_stdout(io.StringIO()), patch("IPython.display.display"):
        return runpy.run_path(str(ROOT / "quote_eda.py"), init_globals={
            "bcq_df": q, "data_ig": t, "bond_info_df": m, "EDA_OVERRIDES": settings})


def snapshot(result, when, cusip="BOND00001", firm="A", side="bid", active=False):
    table = result["active" if active else "states"]
    time = pd.Timestamp(when, tz="America/New_York")
    return table.loc[table["time"].eq(time) & table["cusip"].eq(cusip)
                     & table["firm"].eq(firm) & table["side"].eq(side)]


class QuoteEDAChecks(unittest.TestCase):
    def test_trade_universe_is_independent_of_short_quote_window(self):
        q, t, m = fixture()
        extra = q.loc[q["cusip"].eq("BOND00001")].assign(cusip="NEVERTRADE")
        q = pd.concat([q, extra], ignore_index=True)
        m = pd.concat([m, pd.DataFrame({"CUSIP": ["NEVERTRADE"], "ISSUER": ["TEST"],
                                       "YRS_TO_MATURITY": [2.]})], ignore_index=True)
        t.loc[t["CUSIP"].eq("BOND00003"), "EFFECTIVE_DATETIME_TS"] = pd.Timestamp("2025-11-30 10:00")
        result = replay(q, t, m, TRADE_UNIVERSE_START="2025-12-01", TRADE_UNIVERSE_END="2026-03-01")
        self.assertEqual(set(result["analysis_cusips"]), {"BOND00001", "BOND00002"})
        self.assertEqual(set(result["q"]["cusip"]), {"BOND00001", "BOND00002"})
        self.assertTrue(result["coverage"]["n_trades"].eq(1).all())
        self.assertTrue(result["coverage"]["window_trades"].eq(0).all())
        self.assertTrue(result["coverage"]["bid_fresh_fraction"].gt(0).all())
        self.assertEqual(result["audit"]["quotes_outside_trade_universe"], len(extra) * 2)

    def test_existing_trade_cusip_prefilter_is_supported(self):
        q, t, m = fixture()
        t = t.loc[t["CUSIP"].ne("BOND00003")]
        full = replay(q, t, m)
        prefiltered = replay(q.loc[q["cusip"].isin(t["CUSIP"])], t, m)
        pd.testing.assert_frame_equal(full["side_stats"], prefiltered["side_stats"])
        self.assertEqual(set(full["analysis_cusips"]), {"BOND00001", "BOND00002"})

    def test_zero_and_negative_consensus_remain_valid_by_default(self):
        q, t, m = fixture()
        q["spread"] = (pd.Timestamp("2026-03-02 10:10") - q["quote_timestamp_ET"]).dt.total_seconds() / 300
        result = replay(q, t, m, CHANGE_LAG="10min")
        zero = snapshot(result, "2026-03-02 10:15", active=True).iloc[0]
        negative = snapshot(result, "2026-03-02 10:20", active=True).iloc[0]
        self.assertEqual(zero["spread"], 0.)
        self.assertEqual(negative["spread"], -1.)
        self.assertFalse(zero["peer_flag"])
        self.assertFalse(negative["peer_flag"])
        self.assertTrue(result["pairs"]["mid"].lt(0).any())
        when = pd.Timestamp("2026-03-02 10:15", tz="America/New_York")
        self.assertEqual(result["side_stats"].loc[(when, "BOND00001", "bid"), "median"], 0.)
        self.assertTrue(result["matched_clean"]["spread"].eq(0).any())
        self.assertTrue(result["matched_clean"]["spread_old"].eq(0).any())
        self.assertFalse(result["nonzero"]["spread"].eq(0).any())
        self.assertFalse(result["nonzero"]["spread_old"].eq(0).any())
        self.assertTrue(result["nonzero"]["spread"].lt(0).any())
        self.assertTrue((result["changes"].count() > result["changes_excluding_zero"].count()).all())

    def test_strict_event_time_and_no_trade_in_quote_window(self):
        q, t, m = fixture()
        result = replay(q, t, m)
        state = snapshot(result, "2026-03-02 10:05").iloc[0]
        self.assertEqual(state["event_time"], pd.Timestamp("2026-03-02 10:00", tz="America/New_York"))
        self.assertEqual(state["spread"], 63)
        self.assertTrue((result["states"]["event_time"] < result["states"]["time"]).all())
        self.assertTrue(result["coverage"]["window_trades"].eq(0).all())
        self.assertTrue(result["coverage"]["bid_2dealer_fraction"].gt(0).all())

    def test_future_quotes_do_not_change_earlier_states(self):
        q, t, m = fixture()
        base = replay(q, t, m)
        changed = q.copy()
        changed.loc[changed["quote_timestamp_ET"].ge(pd.Timestamp("2026-03-02 10:20")), "spread"] += 1000
        later = replay(changed, t, m)
        left = base["side_stats"].loc[lambda x: x.index.get_level_values("time") <= pd.Timestamp("2026-03-02 10:20", tz="America/New_York")]
        right = later["side_stats"].loc[left.index]
        pd.testing.assert_frame_equal(left, right)

    def test_expiry_and_invalid_latest_record(self):
        q, t, m = fixture()
        keep = ~((q["cusip"] == "BOND00001") & (q["firm"] == "A") & (q["side"] == "bid")
                 & q["quote_timestamp_ET"].gt(pd.Timestamp("2026-03-02 10:00")))
        q = q.loc[keep].copy()
        row = q.loc[(q["cusip"] == "BOND00001") & (q["firm"] == "B") & (q["side"] == "bid")].iloc[0].copy()
        row["quote_timestamp_ET"] = pd.Timestamp("2026-03-02 10:06")
        row["spread"] = np.nan
        q = q.loc[~((q["cusip"] == "BOND00001") & (q["firm"] == "B") & (q["side"] == "bid")
                    & q["quote_timestamp_ET"].ge(pd.Timestamp("2026-03-02 10:06")))]
        q = pd.concat([q, row.to_frame().T], ignore_index=True)
        result = replay(q, t, m)
        self.assertEqual(len(snapshot(result, "2026-03-02 10:35", active=True)), 0)
        invalid = snapshot(result, "2026-03-02 10:10", firm="B").iloc[0]
        self.assertTrue(pd.isna(invalid["spread"]))
        self.assertFalse(invalid["fresh"])

    def test_refresh_age_does_not_reset_value_age(self):
        q, t, m = fixture()
        mask = (q["cusip"] == "BOND00001") & (q["firm"] == "A") & (q["side"] == "bid")
        q.loc[mask, "spread"] = 62.
        result = replay(q, t, m)
        state = snapshot(result, "2026-03-02 10:30").iloc[0]
        self.assertEqual(state["age_min"], 5)
        self.assertEqual(state["value_age_min"], 35)

    def test_repeated_updates_do_not_multiply_dealer_weight(self):
        q, t, m = fixture()
        # A submits 200 unchanged refreshes; its latest side quote is still one observation.
        target = (q["cusip"] == "BOND00001") & (q["side"] == "bid")
        q.loc[target, "spread"] = q.loc[target, "firm"].map({"A": 100., "B": 62., "C": 64.})
        extras = q.loc[target & q["firm"].eq("A")].iloc[0].copy()
        new = pd.DataFrame([extras] * 200)
        new["quote_timestamp_ET"] = pd.date_range("2026-03-02 10:00:01", periods=200, freq="s")
        result = replay(pd.concat([q, new], ignore_index=True), t, m)
        stat = result["side_stats"].loc[(pd.Timestamp("2026-03-02 10:05", tz="America/New_York"), "BOND00001", "bid")]
        self.assertEqual(stat["n_before_peer_filter"], 3)
        self.assertEqual(stat["median_before_peer_filter"], 64)
        self.assertEqual(stat["n_dealers"], 2)
        self.assertEqual(stat["median"], 63)
        self.assertEqual(len(snapshot(result, "2026-03-02 10:05", active=True)), 1)
        self.assertTrue(snapshot(result, "2026-03-02 10:05", active=True)["peer_flag"].iloc[0])

    def test_pair_crossing_asynchrony_and_size_mismatch(self):
        q, t, m = fixture()
        q.loc[(q["firm"] == "A") & (q["side"] == "ask"), "spread"] += 10
        q.loc[(q["firm"] == "B") & (q["side"] == "ask"), "quantity"] = 500000
        q = q.loc[~((q["firm"] == "C") & (q["side"] == "ask")
                    & q["quote_timestamp_ET"].gt(pd.Timestamp("2026-03-02 09:55")))]
        result = replay(q, t, m)
        pairs = result["pairs"].loc[lambda x: x["time"].eq(pd.Timestamp("2026-03-02 10:10", tz="America/New_York"))]
        a, b, c = [pairs.loc[pairs["firm"].eq(f)].iloc[0] for f in ["A", "B", "C"]]
        self.assertTrue(a["crossed"])
        self.assertTrue(pd.isna(a["mid"]))
        self.assertFalse(b["eligible"])
        self.assertFalse(c["eligible"])
        self.assertEqual(len(result["active"].loc[lambda x: x["firm"].eq("A") & x["side"].eq("ask")]), 39)

    def test_replay_uses_event_time_and_sorts_input(self):
        q, t, m = fixture()
        baseline = replay(q, t, m)
        q["received_ET"] = q["quote_timestamp_ET"] + pd.Timedelta("2h")
        result = replay(q.sample(frac=1, random_state=3), t, m)
        pd.testing.assert_frame_equal(baseline["side_stats"], result["side_stats"])
        self.assertNotIn("known_time", result["q"].columns)
        self.assertEqual(snapshot(result, "2026-03-02 10:05").iloc[0]["event_time"],
                         pd.Timestamp("2026-03-02 10:00", tz="America/New_York"))

    def test_timezone_conversion_matches_naive_et(self):
        q, t, m = fixture()
        base = replay(q, t, m)
        aware = q.copy()
        aware["quote_timestamp_ET"] = aware["quote_timestamp_ET"].dt.tz_localize("America/New_York").dt.tz_convert("UTC")
        converted = replay(aware, t, m, Q_NAIVE_TZ="UTC")
        pd.testing.assert_frame_equal(base["side_stats"], converted["side_stats"])

    def test_composition_changes_do_not_create_common_moves(self):
        q, t, m = fixture()
        # Fixed dealer-specific levels. C joins at 10:25 with a high level.
        q["spread"] = q["firm"].map({"A": 60., "B": 61., "C": 90.})
        q = q.loc[~(q["firm"].eq("C") & q["quote_timestamp_ET"].lt(pd.Timestamp("2026-03-02 10:25")))]
        result = replay(q, t, m)
        self.assertEqual(result["states"]["spread_valid"].dtype, bool)
        self.assertTrue(result["changes"].stack().eq(0).all())
        self.assertTrue(result["overlap"].to_numpy().max() > 0)
        self.assertEqual(len(result["pca_variance"]), 0)

    def test_one_sided_and_missing_quantity_inputs(self):
        q, t, m = fixture()
        q = q.loc[q["side"].eq("bid")].drop(columns="quantity")
        result = replay(q, t, m)
        self.assertTrue(result["pairs"].empty)
        self.assertTrue(result["coverage"]["ask_fresh_fraction"].eq(0).all())
        self.assertTrue(result["coverage"]["paired_mid_fraction"].eq(0).all())
        self.assertTrue(result["changes"].count().gt(0).all())

    def test_zero_quantity_sentinels_preserve_comovement(self):
        q, t, m = fixture()
        minutes = (q["quote_timestamp_ET"] - q["quote_timestamp_ET"].min()).dt.total_seconds() / 60
        q["spread"] += .005 * minutes ** 2  # nonconstant aligned changes
        q["quantity"] = 0.
        result = replay(q, t, m)
        self.assertTrue(result["q"]["quantity_raw"].eq(0).all())
        self.assertTrue(result["q"]["quantity"].isna().all())
        self.assertTrue(result["pairs"]["eligible_unknown_size"].all())
        self.assertFalse(result["pairs"]["known_same_size"].any())
        self.assertTrue(result["changes"].count().ge(3).all())
        self.assertTrue(result["changes_stable_size"].isna().all().all())
        self.assertTrue(result["changes_unknown_size"].notna().any().all())
        self.assertAlmostEqual(result["correlation"].iloc[0, 1], 1.)
        self.assertGreater(len(result["pca_variance"]), 0)

    def test_unknown_size_on_one_side_is_explicitly_unverified(self):
        q, t, m = fixture()
        q.loc[q["side"].eq("ask"), "quantity"] = 0
        result = replay(q, t, m)
        self.assertTrue(result["pairs"]["size_status"].eq("unknown_one").all())
        self.assertTrue(result["pairs"]["eligible_unknown_size"].all())
        self.assertFalse(result["pairs"]["eligible_known_size"].any())
        strict = replay(q, t, m, PAIR_ALLOW_UNKNOWN_SIZE=False)
        self.assertFalse(strict["pairs"]["eligible"].any())

    def test_zero_spread_latest_does_not_revive_old_quote(self):
        q, t, m = fixture()
        q = q.loc[q["firm"].eq("A") & q["side"].eq("bid")].copy()
        target = q["cusip"].eq("BOND00001")
        row = q.loc[target].iloc[0].copy()
        row["quote_timestamp_ET"] = pd.Timestamp("2026-03-02 10:06")
        row["spread"] = 0.
        q = q.loc[~(target & q["quote_timestamp_ET"].ge(pd.Timestamp("2026-03-02 10:06")))]
        q = pd.concat([q, row.to_frame().T], ignore_index=True)
        result = replay(q, t, m, ZERO_SPREAD_IS_MISSING=True)
        latest = snapshot(result, "2026-03-02 10:10").iloc[0]
        self.assertEqual(latest["spread"], 0)
        self.assertTrue(latest["fresh"])
        self.assertFalse(latest["usable"])
        self.assertEqual(len(snapshot(result, "2026-03-02 10:10", active=True)), 0)
        when = pd.Timestamp("2026-03-02 10:10", tz="America/New_York")
        self.assertEqual(result["raw_side_stats"].loc[(when, "BOND00001", "bid"), "median"], 0)
        self.assertNotIn((when, "BOND00001", "bid"), result["side_stats"].index)
        valid_zero = replay(q, t, m)
        self.assertEqual(len(snapshot(valid_zero, "2026-03-02 10:10", active=True)), 1)

    def test_size_changes_are_sensitivity_not_main_filter(self):
        q, t, m = fixture()
        minutes = (q["quote_timestamp_ET"] - q["quote_timestamp_ET"].min()).dt.total_seconds() / 60
        q["quantity"] = 1000000 + minutes * 1000
        result = replay(q, t, m)
        self.assertTrue(result["changes"].count().gt(0).all())
        self.assertTrue(result["matched_clean"]["known_size_changed"].all())
        self.assertTrue(result["changes_stable_size"].isna().all().all())
        self.assertTrue(result["changes_unknown_size"].isna().all().all())

    def test_quote_metadata_supports_traded_bonds_with_no_window_trades(self):
        q, t, m = fixture()
        q["ISSUER"] = "TEST"
        q["YRS_TO_MATURITY"] = q["cusip"].map(m.set_index("CUSIP")["YRS_TO_MATURITY"])
        result = replay(q, t, None)
        self.assertEqual(len(result["analysis_cusips"]), 3)
        self.assertEqual(result["coverage"].loc["BOND00002", "window_trades"], 0)
        self.assertGreater(result["coverage"].loc["BOND00002", "bid_fresh_fraction"], 0)
        self.assertEqual(len(result["unmapped_quote_cusips"]), 0)

    def test_timestamp_sequence_resolves_conflicts(self):
        q, t, m = fixture()
        q["feed_sequence"] = 10
        row = q.loc[q["cusip"].eq("BOND00001") & q["firm"].eq("A") & q["side"].eq("bid")
                    & q["quote_timestamp_ET"].eq(pd.Timestamp("2026-03-02 10:00"))].iloc[0].copy()
        high, low = row.copy(), row.copy()
        high["feed_sequence"], high["spread"] = 100, 70.
        low["feed_sequence"], low["spread"] = 1, 60.
        q = pd.concat([q, high.to_frame().T, low.to_frame().T], ignore_index=True)
        result = replay(q, t, m, QUOTE_SEQUENCE_COL="feed_sequence")
        self.assertEqual(snapshot(result, "2026-03-02 10:05").iloc[0]["spread"], 70)
        self.assertGreater(result["audit"]["conflicting_timestamp_groups"], 0)


if __name__ == "__main__":
    unittest.main()
