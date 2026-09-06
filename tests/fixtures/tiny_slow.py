"""Prints, then takes its time. Used to prove logs stream while a stage runs."""

import time

import polars as pl

from backend.pipeline.contract import Strategy


class SlowStrategy(Strategy):

    def collect(self, config):
        print("collect: starting the slow part")
        time.sleep(float(config.get("sleep_s", 3.0)))
        print("collect: done")
        return pl.DataFrame({"ts": [0, 1], "asset": ["A", "A"],
                             "price": [100.0, 101.0]})

    def prepare(self, raw, config):
        return raw

    def on_tick(self, ts, market, inventory, model, config):
        return market.select("asset", pl.lit(1.0).alias("qty"))
