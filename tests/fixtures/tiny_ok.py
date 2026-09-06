"""A minimal working strategy with no external data, for fast tests."""

import polars as pl

from backend.pipeline.contract import Strategy


class TinyStrategy(Strategy):

    def collect(self, config):
        n = int(config.get("n", 20))
        print(f"collecting {n} bars")
        return pl.DataFrame({
            "ts": list(range(n)),
            "asset": ["A"] * n,
            "price": [100.0 + i for i in range(n)],
        })

    def prepare(self, raw, config):
        return raw.with_columns(pl.col("price").diff().fill_null(0.0).alias("mom"))

    def on_tick(self, ts, market, inventory, model, config):
        assets = market["asset"].to_list()
        mom = market["mom"].to_list()
        qty = []
        for asset, m in zip(assets, mom):
            target = 1.0 if m > 0 else -1.0 if m < 0 else 0.0
            qty.append(target - inventory.get(asset, 0.0))
        return pl.DataFrame({"asset": assets, "qty": qty})
