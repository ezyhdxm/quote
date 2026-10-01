"""Deterministic synthetic data for trying the notebook without private inputs."""

import numpy as np
import pandas as pd


def make_demo(seed=7):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2026-03-02 08:00", "2026-03-13 17:00", freq="5min", tz="America/New_York")
    times = times[(times.dayofweek < 5) & (times.hour >= 8) & (times.hour < 17)]
    cusips = [f"DEMO{i:05d}" for i in range(8)]
    metadata = pd.DataFrame({"CUSIP": cusips, "ISSUER": "Demo Issuer",
                             "YRS_TO_MATURITY": [.9, 1.8, 3.5, 4.5, 6.0, 7.0, 10.0, 15.0]})
    common = np.cumsum(rng.normal(0, .15, len(times)))
    quotes, trades = [], []
    for i, cusip in enumerate(cusips):
        level = 45 + 3 * metadata.loc[i, "YRS_TO_MATURITY"] + common
        for d, firm in enumerate(["Dealer A", "Dealer B", "Dealer C", "Dealer D"]):
            # One dealer refreshes much more often; its feed share should not be its weight.
            selected = np.arange(len(times))[::[1, 2, 3, 4][d]]
            if i == 1 and d > 0:
                selected = selected[::3]
            quantity = [1.0, 1.0, .5, 0.0][d]  # source units; zero-coded missing size
            for j in selected:
                noise = rng.normal(0, .3)
                for side, width in [("bid", 2.0), ("ask", -2.0)]:
                    if i == 7 and side == "ask":  # deliberate one-sided bond
                        continue
                    value = level[j] + width + .2 * d + noise
                    if i == 1 and firm == "Dealer A" and j % 67 == 0:
                        value = 0.0  # noisy placeholder-like updates; retained for audit
                    if i == 3 and firm == "Dealer C" and side == "ask" and j % 73 == 0:
                        value += 15.0  # eligible crossed pair
                    quotes.append((cusip, firm, side, times[j].tz_localize(None), value, quantity))
        # Bond 1 is quote-only. Odd bonds are much less traded than their tenor neighbor.
        count = 0 if i == 1 else (15 if i % 2 else 160)
        for j in sorted(rng.choice(len(times), size=count, replace=False)):
            side = rng.choice(["B", "S", "D"])
            spread = level[j] + {"B": 1.5, "S": -1.5, "D": 0}[side] + rng.normal(0, .7)
            trades.append((cusip, times[j].tz_localize(None), spread / 100,
                           rng.choice([.1, .5, 1., 3.]), side,
                           "Demo Issuer", metadata.loc[i, "YRS_TO_MATURITY"]))
    q = pd.DataFrame(quotes, columns=["cusip", "firm", "side", "quote_timestamp_ET", "spread", "quantity"])
    q = pd.concat([q, q.iloc[:10]], ignore_index=True)  # exact duplicate examples
    t = pd.DataFrame(trades, columns=["CUSIP", "EFFECTIVE_DATETIME_TS", "BM_SPREAD", "QUANTITY",
                                      "EFF_SIDE", "ISSUER", "YRS_TO_MATURITY"])
    return q, t, metadata
