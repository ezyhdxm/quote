# TEST SETUP LOGIC: 合成fixtures与断言；测试通过不代表真实预测增益。
"""Causality and supported same-bond/issuer direction; no model fitting."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quote_quality_core as core
import quote_quality_movement as movement
from quote_quality_incremental import frame_fingerprint

T = pd.Timestamp("2026-03-02 10:00", tz="America/New_York")


# TEST FIXTURE LOGIC: raw；仅用于复现输入或核对行为。
def raw(records):
    return pd.DataFrame([dict(quote_timestamp_ET=T + pd.Timedelta(minutes=minute),
        cusip=bond, firm=firm, side=side, spread=spread, quantity=quantity)
        for minute, bond, firm, side, spread, quantity in records])


# TEST FIXTURE LOGIC: frame；仅用于复现输入或核对行为。
def frame(records):
    return pd.DataFrame([dict(row_id=i, time=T + pd.Timedelta(minutes=minute),
        cusip=bond, ISSUER=issuer, split=split)
        for i, (minute, bond, issuer, split) in enumerate(records)])


# TEST FIXTURE LOGIC: build；仅用于复现输入或核对行为。
def build(quotes, queries, **kwargs):
    events = core.prepare_quote_events(quotes)
    events["cache_key"] = "synthetic-checked-event-cache"
    return movement.build_movement_features(queries, events, **kwargs).set_index("row_id")


# TEST FIXTURE LOGIC: direction_reference；仅用于复现输入或核对行为。
def direction_reference(events, when, side, age_min=30, allow_exact=True):
    """Independent small-fixture pandas oracle, latest state before completeness."""
    old = when - pd.Timedelta(minutes=30)
    now_states, old_states = {}, {}
    for firm, g in events.loc[events.side.eq(side)].groupby("firm", observed=True):
        for end, states in [(when, now_states), (old, old_states)]:
            eligible = g.quote_timestamp_ET.le(end) if allow_exact else g.quote_timestamp_ET.lt(end)
            eligible &= g.day.eq(when.normalize()) & (end.normalize() == when.normalize())
            h = g.loc[eligible].sort_values("quote_timestamp_ET")
            if h.empty:
                continue
            state = h.iloc[-1]
            if state.complete and np.isfinite(state.center) and (end - state.quote_timestamp_ET).total_seconds() / 60 <= age_min:
                states[firm] = state
    common = set(now_states) & set(old_states)
    result = {name: np.nan for name in movement._MOVE_SUFFIXES}
    result.update(common_n=len(common), current_n=len(now_states), lookback_n=len(old_states), guarded_common_n=0)
    if not common:
        return result
    deltas, ages, guarded, changed = [], [], [], []
    for firm in common:
        a, b = now_states[firm], old_states[firm]
        delta = a.center - b.center
        diff = a.quantity_set != b.quantity_set or a.candidate_count != b.candidate_count
        deltas.append(delta)
        changed.append(diff)
        if not diff:
            guarded.append(delta)
        ages.append(max((when - a.quote_timestamp_ET).total_seconds(),
                        (old - b.quote_timestamp_ET).total_seconds()) / 60)
    values = np.asarray(deltas)
    result.update(mean_bps=values.mean(), median_bps=np.median(values),
        guarded_mean_bps=np.mean(guarded) if guarded else np.nan,
        up_fraction=np.mean(values > 0), down_fraction=np.mean(values < 0),
        flat_fraction=np.mean(values == 0), retention=len(common) / max(len(now_states), len(old_states)),
        condition_changed_fraction=np.mean(changed), mean_support_age_min=np.mean(ages),
        max_support_age_min=max(ages), guarded_common_n=len(guarded))
    return result


# TEST LOGIC: MovementChecks；仅用于复现输入或核对行为。
class MovementChecks(unittest.TestCase):
    # TEST LOGIC: test_common_dealers_not_roster_or_quantity_selection；仅用于复现输入或核对行为。
    def test_common_dealers_not_roster_or_quantity_selection(self):
        quotes = raw([(0,"X","A","bid",-1,None),(30,"X","A","bid",3,None),
                      (30,"X","B","bid",100,0),(0,"X","C","bid",5,2),
                      (29,"X","C","bid",np.nan,2)])
        result = build(quotes, frame([(30,"X","I","Validation")]), include_issuer=False).iloc[0]
        self.assertEqual(result.bcq_bid_move_mean_bps, 4)
        self.assertEqual(result.bcq_bid_move_guarded_mean_bps, 4)
        self.assertEqual(result.bcq_bid_move_common_n, 1)
        self.assertEqual(result.bcq_bid_move_current_n, 2)
        self.assertEqual(result.bcq_bid_move_lookback_n, 2)
        self.assertEqual(result.bcq_bid_move_retention, .5)
        self.assertEqual(result.bcq_bid_move_up_fraction, 1)
        self.assertEqual(result.bcq_bid_move_condition_changed_fraction, 0)

    # TEST LOGIC: test_quantity_condition_change_does_not_become_guarded_direction；仅用于复现输入或核对行为。
    def test_quantity_condition_change_does_not_become_guarded_direction(self):
        quotes = raw([(0,"X","A","bid",100,2),(30,"X","A","bid",104,3)])
        result = build(quotes, frame([(30,"X","I","Validation")]), include_issuer=False).iloc[0]
        self.assertEqual(result.bcq_bid_move_mean_bps, 4)
        self.assertEqual(result.bcq_bid_move_condition_changed_fraction, 1)
        self.assertEqual(result.bcq_bid_move_guarded_common_n, 0)
        self.assertTrue(pd.isna(result.bcq_bid_move_guarded_mean_bps))

    # TEST LOGIC: test_latest_incomplete_blocks_old_complete_and_exact_flag；仅用于复现输入或核对行为。
    def test_latest_incomplete_blocks_old_complete_and_exact_flag(self):
        quotes = raw([(0,"X","A","bid",100,2),(29,"X","A","bid",105,2),
                      (30,"X","A","bid",np.nan,2)])
        queries = frame([(30,"X","I","Validation"),(30.5,"X","I","Validation")])
        result = build(quotes, queries, include_issuer=False)
        self.assertTrue(result.bcq_bid_move_common_n.eq(0).all())
        self.assertTrue(result.bcq_bid_move_mean_bps.isna().all())
        strict = build(quotes, queries, include_issuer=False, allow_exact=False)
        # At exact t-30, strict asof also excludes the initial observation.
        self.assertEqual(strict.loc[0,"bcq_bid_move_common_n"], 0)
        quotes.loc[0,"quote_timestamp_ET"] -= pd.Timedelta(seconds=1)
        strict = build(quotes, queries, include_issuer=False, allow_exact=False)
        self.assertEqual(strict.loc[0,"bcq_bid_move_mean_bps"], 5)
        self.assertEqual(strict.loc[1,"bcq_bid_move_common_n"], 0)

    # TEST LOGIC: test_same_et_day_and_age_boundary；仅用于复现输入或核对行为。
    def test_same_et_day_and_age_boundary(self):
        quotes = raw([(0,"X","A","bid",0,0),(30,"X","A","bid",-5,0)])
        quotes.quote_timestamp_ET += pd.Timedelta(hours=13, minutes=50)
        queries = frame([(30,"X","I","Validation")])
        queries.time += pd.Timedelta(hours=13, minutes=50)
        result = build(quotes, queries, include_issuer=False).iloc[0]
        self.assertEqual(result.bcq_bid_move_common_n, 0)  # 00:20 vs previous-day 23:50
        quotes = raw([(-30,"X","A","bid",100,2),(0,"X","A","bid",104,2)])
        queries = frame([(30,"X","I","Validation")])
        boundary = build(quotes, queries, include_issuer=False).iloc[0]
        self.assertEqual(boundary.bcq_bid_move_common_n, 1)
        self.assertEqual(boundary.bcq_bid_move_mean_bps, 0)
        queries.time += pd.Timedelta(nanoseconds=1)
        self.assertEqual(build(quotes, queries, include_issuer=False).iloc[0].bcq_bid_move_common_n, 0)

    # TEST LOGIC: test_nanosecond_asof_never_rounds_future_event_backward；仅用于复现输入或核对行为。
    def test_nanosecond_asof_never_rounds_future_event_backward(self):
        quotes = raw([(0,"X","A","bid",100,2),(30,"X","A","bid",101,2),
                      (30,"X","A","bid",200,2)])
        quotes.loc[1,"quote_timestamp_ET"] += pd.Timedelta(nanoseconds=400)
        quotes.loc[2,"quote_timestamp_ET"] += pd.Timedelta(nanoseconds=800)
        queries = frame([(30,"X","I","Validation")])
        queries.time += pd.Timedelta(nanoseconds=600)
        result = build(quotes, queries, include_issuer=False).iloc[0]
        self.assertEqual(result.bcq_bid_move_mean_bps, 1)
        self.assertAlmostEqual(result.bcq_bid_move_mean_support_age_min, 600 / (60 * 10**9))

    # TEST LOGIC: test_issuer_exclusion_equal_bond_vote_and_minimum；仅用于复现输入或核对行为。
    def test_issuer_exclusion_equal_bond_vote_and_minimum(self):
        quotes = raw([(0,bond,firm,"bid",100,2) for bond,firm in [("X","A"),("Y","A"),("Y","B"),("Z","A")]] +
                     [(30,bond,firm,"bid",100+delta,2) for bond,firm,delta in [("X","A",100),("Y","A",2),("Y","B",2),("Z","A",6)]])
        queries = frame([(0,"X","I","Train"),(0,"Y","I","Train"),(0,"Z","I","Train"),
                         (30,"X","I","Validation")])
        result = build(quotes, queries).loc[3]
        self.assertEqual(result.bcq_bid_issuer_move_mean_bps, 4)  # two bonds, not three dealers
        self.assertEqual(result.bcq_bid_issuer_move_median_bps, 4)
        self.assertEqual(result.bcq_bid_issuer_move_other_bond_n, 2)
        self.assertEqual(result.bcq_bid_issuer_move_common_dealer_n, 3)
        self.assertEqual(result.bcq_bid_issuer_move_dispersion_bps, 2)
        quotes.loc[quotes.cusip.eq("X") & quotes.quote_timestamp_ET.eq(T+pd.Timedelta(minutes=30)),"spread"] = 9999
        self.assertEqual(build(quotes, queries).loc[3].bcq_bid_issuer_move_mean_bps, 4)
        missing_mapping = queries.loc[queries.cusip.ne("Z")]
        sparse = build(quotes, missing_mapping).loc[3]
        self.assertEqual(sparse.bcq_bid_issuer_move_other_bond_n, 1)
        self.assertTrue(pd.isna(sparse.bcq_bid_issuer_move_mean_bps))
        self.assertTrue(pd.isna(sparse.bcq_bid_issuer_move_mean_support_age_min))

    # TEST LOGIC: test_mapping_prefix_no_future_backfill_and_test_excluded；仅用于复现输入或核对行为。
    def test_mapping_prefix_no_future_backfill_and_test_excluded(self):
        quotes = raw([(minute,bond,"A","bid",100 + (delta if minute else 0),2)
            for bond,delta in [("X",1),("Y",2),("Z",6)] for minute in [0,30]])
        early = frame([(0,"X","I","Train"),(30,"X","I","Validation")])
        later = frame([(40,"Y","I","Validation"),(40,"Z","I","Validation"),
                       (0,"Y","I","Test"),(0,"Z","I","Test")])
        later.row_id += 2
        combined = pd.concat([early,later],ignore_index=True)
        result = build(quotes, combined)
        self.assertEqual(result.index.tolist(), [0,1,2,3])
        self.assertEqual(result.loc[1,"bcq_bid_issuer_move_other_bond_n"], 0)
        self.assertTrue(pd.isna(result.loc[1,"bcq_bid_issuer_move_mean_bps"]))
        # The donor must be mapped at the lookback start, not just before t.
        queries = frame([(0,"X","I","Train"),(1,"Y","I","Train"),(1,"Z","I","Train"),
                         (30,"X","I","Validation")])
        self.assertEqual(build(quotes,queries).loc[3].bcq_bid_issuer_move_other_bond_n, 0)

    # TEST LOGIC: test_future_append_and_future_issuer_conflict_invariance；仅用于复现输入或核对行为。
    def test_future_append_and_future_issuer_conflict_invariance(self):
        quotes = raw([(minute,bond,"A","bid",100 + (delta if minute else 0),2)
            for bond,delta in [("X",1),("Y",2),("Z",6)] for minute in [0,30]])
        queries = frame([(0,"X","I","Train"),(0,"Y","I","Train"),(0,"Z","I","Train"),
                         (30,"X","I","Validation")])
        original = build(quotes,queries)
        more_quotes = pd.concat([quotes,raw([(60,"Y","A","bid",10000,2),(60,"Z","A","bid",-1000,2)])],ignore_index=True)
        more_queries = frame([(60,"Y","New issuer","Validation"),(60,"Z",None,"Validation")])
        more_queries.row_id += 4
        extended = build(more_quotes,pd.concat([queries,more_queries],ignore_index=True))
        pd.testing.assert_frame_equal(original, extended.loc[original.index], check_flags=False)
        # A conflict observed now excludes that donor now, without invalidating
        # its previous reliable mapping retroactively.
        conflict = frame([(0,"X","I","Train"),(0,"Y","I","Train"),(0,"Z","I","Train"),
                          (29,"Y","New issuer","Validation"),(30,"X","I","Validation")])
        self.assertEqual(build(quotes,conflict).loc[4].bcq_bid_issuer_move_other_bond_n, 1)

    # TEST LOGIC: test_block_reuse_matches_independent_reference；仅用于复现输入或核对行为。
    def test_block_reuse_matches_independent_reference(self):
        rng = np.random.default_rng(2026)
        records = []
        for side in ["bid","ask"]:
            for firm in ["A","B","C","D"]:
                for minute in [0,10,20,30,40,60]:
                    spread = rng.choice([-5.,0.,100.,110.,np.nan])
                    size = rng.choice([0.,2.,3.,np.nan])
                    records.append((minute,"X",firm,side,spread,size))
        quotes = raw(records)
        queries = frame([(minute,"X","I","Validation") for minute in [0,29,30,35,40,60,61]])
        events = core.prepare_quote_events(quotes)
        for exact in [True,False]:
            with patch.object(movement,"_QUERY_BLOCK",2):
                result = movement.build_movement_features(queries,events,include_issuer=False,allow_exact=exact)
            for row, when in enumerate(queries.time):
                for side in ["bid","ask"]:
                    expected = direction_reference(events["events"],when,side,allow_exact=exact)
                    for suffix,value in expected.items():
                        actual = result.loc[row,f"bcq_{side}_move_{suffix}"]
                        if pd.isna(value):
                            self.assertTrue(pd.isna(actual), (row,side,suffix))
                        else:
                            self.assertAlmostEqual(actual,value,msg=str((row,side,suffix)))

    # TEST LOGIC: test_empty_paths_schema_metadata_and_no_aggregation；仅用于复现输入或核对行为。
    def test_empty_paths_schema_metadata_and_no_aggregation(self):
        quotes = raw([(0,"X","A","bid",100,2)])
        queries = frame([(30,"MISSING","I","Validation"),(30,"X","I","Test")])
        cache = core.prepare_quote_events(quotes)
        cache["cache_key"] = "checked-key"
        reports = []
        with patch.object(core,"prepare_quote_events",side_effect=AssertionError("Must reuse events")), \
             patch.object(core,"event_history",side_effect=AssertionError("No aggregation")):
            result = movement.build_movement_features(queries,cache,progress=lambda *args:reports.append(args))
        self.assertEqual(result.row_id.tolist(), [0])
        self.assertEqual(result.bcq_bid_move_common_n.iloc[0],0)
        self.assertTrue(result.bcq_bid_move_mean_bps.isna().all())
        self.assertEqual(result.attrs["event_cache_key"],"checked-key")
        self.assertEqual(result.attrs["source_frame_sha256"],frame_fingerprint(queries))
        self.assertEqual(result.attrs["excluded_rows"],1)
        self.assertEqual(set(result.attrs["feature_dictionary"]),set(movement.DIRECTION_FEATURES+movement.ISSUER_FEATURES))
        self.assertEqual(reports[-1][0],"movement_features")
        empty = movement.build_movement_features(queries.loc[queries.split.eq("Test")],cache)
        self.assertTrue(empty.empty)
        self.assertEqual(list(empty.columns),["row_id","cusip","time"]+movement.DIRECTION_FEATURES+movement.ISSUER_FEATURES)
        no_events = core.prepare_quote_events(quotes.iloc[:0])
        absent = movement.build_movement_features(queries,no_events)
        self.assertTrue(absent.bcq_bid_move_common_n.eq(0).all())
        self.assertTrue(absent.bcq_bid_issuer_move_mean_bps.isna().all())

    # TEST LOGIC: test_large_issuer_block_shares_dealer_states_and_excludes_each_target；仅用于复现输入或核对行为。
    def test_large_issuer_block_shares_dealer_states_and_excludes_each_target(self):
        bonds = [f"B{i:02d}" for i in range(20)]
        quotes = raw([(minute,bond,firm,side,100 + minute*(i+1),2)
            for i,bond in enumerate(bonds) for firm in ["A","B","C"]
            for side in ["bid","ask"] for minute in [0,30]])
        records = [(0,bond,"I","Train") for bond in bonds]
        records += [(minute,bond,"I","Validation") for minute in np.linspace(31,59,600) for bond in bonds]
        queries = frame(records)
        cache = core.prepare_quote_events(quotes)
        reports = []
        with patch.object(movement,"_dealer_states",wraps=movement._dealer_states) as states, \
             patch.object(movement,"_movement_at",wraps=movement._movement_at) as at:
            result = movement.build_movement_features(queries,cache,progress=lambda *args:reports.append(args))
        # 12,020 targets share 601 times, two blocks, 20 bonds, two sides.
        self.assertEqual(states.call_count,40)
        self.assertEqual(at.call_count,80)
        self.assertTrue(all(len(call.args[1]) <= 512 for call in at.call_args_list))
        last = result.iloc[-1]
        self.assertEqual(last.bcq_bid_issuer_move_other_bond_n,19)
        self.assertEqual(last.bcq_bid_issuer_move_common_dealer_n,57)
        self.assertEqual(last.bcq_bid_issuer_move_mean_bps,30 * np.mean(np.arange(1,20)))
        self.assertTrue(any("513" in r[3] for r in reports))

    # TEST LOGIC: test_rejects_missing_cache_split_and_window_grid；仅用于复现输入或核对行为。
    def test_rejects_missing_cache_split_and_window_grid(self):
        quotes = raw([(0,"X","A","bid",100,2)])
        queries = frame([(30,"X","I","Validation")])
        cache = core.prepare_quote_events(quotes)
        for window in [0,10,60,np.nan]:
            with self.assertRaises(ValueError):
                movement.build_movement_features(queries,cache,lookback_min=window)
        with self.assertRaises(ValueError):
            movement.build_movement_features(queries,None)
        with self.assertRaises(ValueError):
            movement.build_movement_features(queries.drop(columns="split"),cache)
        with self.assertRaises(ValueError):
            movement.build_movement_features(pd.concat([queries,queries]),cache)

    # TEST LOGIC: test_original_key_and_row_order_are_retained；仅用于复现输入或核对行为。
    def test_original_key_and_row_order_are_retained(self):
        quotes = raw([(0,"X","A","bid",100,2),(30,"X","A","bid",104,2)])
        queries = frame([(30," X ","I","Validation"),(30,"X","I","Train")]).iloc[::-1]
        result = build(quotes,queries,include_issuer=False)
        self.assertEqual(result.index.tolist(),[1,0])
        self.assertEqual(result.loc[0,"cusip"]," X ")
        self.assertEqual(result.loc[0,"bcq_bid_move_common_n"],0)
        self.assertEqual(result.loc[1,"bcq_bid_move_mean_bps"],4)


if __name__ == "__main__":
    unittest.main()
