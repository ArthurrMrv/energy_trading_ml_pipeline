"""A strategy that memorizes the next return of every row it is fitted on.

In-sample it looks like a gift: it is trading tomorrow's move. A chronological
train/test split fits only on train, so the test window has no memorized labels
and the order is zero. That gap is the point of the fixture.
"""

import polars as pl

from backend.pipeline.contract import Strategy


class PeekingStrategy(Strategy):

    def collect(self, config):
        n = int(config.get("n", 60))
        return pl.DataFrame({
            "ts": list(range(n)),
            "asset": ["A"] * n,
            "price": [100.0 + (i % 2) for i in range(n)],
        })

    def prepare(self, raw, config):
        return raw.with_columns(pl.lit(0.0).alias("feat"))

    def fit(self, prepared, config):
        ordered = prepared.sort("ts")
        future = ordered["price"].diff().shift(-1).fill_null(0.0)
        return {
            ts: (1.0 if move > 0 else -1.0 if move < 0 else 0.0)
            for ts, move in zip(ordered["ts"].to_list(), future.to_list())
        }

    def on_tick(self, ts, market, inventory, model, config):
        lookup = {str(k): float(v) for k, v in dict(model or {}).items()}
        assets = market["asset"].to_list()
        qty = []
        for asset, t in zip(assets, market["ts"].to_list()):
            target = lookup.get(str(t), 0.0)
            qty.append(target - inventory.get(asset, 0.0))
        return pl.DataFrame({"asset": assets, "qty": qty})
