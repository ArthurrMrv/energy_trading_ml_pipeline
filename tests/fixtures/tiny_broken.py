"""Raises in prepare(), to exercise pause / context / resume."""

import polars as pl

from backend.pipeline.contract import Strategy


class BrokenStrategy(Strategy):

    def collect(self, config):
        return pl.DataFrame({
            "ts": [0, 1, 2], "asset": ["A"] * 3, "price": [10.0, 11.0, 12.0]
        })

    def prepare(self, raw, config):
        return raw.select(pl.col("no_such_column"))

    def on_tick(self, ts, market, inventory, model, config):
        return market.select("asset", pl.lit(1.0).alias("qty"))
