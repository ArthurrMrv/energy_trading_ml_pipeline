"""Day-ahead timing identities on a toy strategy, not a research one.

The print on day ``t`` was auctioned at noon ``t-1``. Features at ``t`` may
use prints through ``t-1`` only; the panel mark is the next print.
"""

from datetime import date, timedelta

import polars as pl
import pytest

from backend.pipeline.contract import Strategy

N, LOOKBACK = 40, 5
START = date(2015, 6, 1)
PROBE = START + timedelta(days=20)


class ToyDA(Strategy):
    """One asset. ``z`` from the lagged print; ``price`` is the next print."""

    def collect(self, config):
        return None

    def prepare(self, raw, config):
        lookback = int(config.get("lookback", LOOKBACK))
        return (
            raw.sort("ts")
            .with_columns(pl.col("price").alias("print"))
            .with_columns(pl.col("print").shift(1).alias("known"))
            .with_columns(
                pl.col("known").rolling_mean(lookback).alias("mean"),
                pl.col("known").rolling_std(lookback).alias("std"),
            )
            .with_columns(
                pl.when(pl.col("std") > 0)
                .then((pl.col("known") - pl.col("mean")) / pl.col("std"))
                .otherwise(0.0)
                .alias("z"),
                pl.col("print").shift(-1).alias("price"),
            )
            .drop_nulls()
        )

    def on_tick(self, ts, market, inventory, model, config):
        assets = market["asset"].to_list()
        z = market["z"].to_list()
        qty = [-(zi / 2.0) - inventory.get(a, 0.0) for a, zi in zip(assets, z)]
        return pl.DataFrame({"asset": assets, "qty": qty})


def panel(*, scramble_from=None):
    rows = []
    for i in range(N):
        ts = START + timedelta(days=i)
        px = 999.0 + i if scramble_from is not None and ts >= scramble_from else 40.0 + i * 0.1
        rows.append({"ts": ts, "asset": "A", "price": px})
    return pl.DataFrame(rows)


def prepared(raw):
    return ToyDA().prepare(raw, {})


def qty_at(raw, ts):
    strategy = ToyDA()
    market = prepared(raw).filter(pl.col("ts") == ts)
    return strategy.on_tick(ts, market, {}, None, {})["qty"].to_list()


class TestSameBarInvariance:
    def test_qty_at_t_ignores_same_bar_print(self):
        assert qty_at(panel(), PROBE) == qty_at(panel(scramble_from=PROBE), PROBE)

    def test_qty_at_t_plus_1_may_see_scrambled_print(self):
        nxt = PROBE + timedelta(days=1)
        assert qty_at(panel(), nxt) != qty_at(panel(scramble_from=PROBE), nxt)


class TestNextPrintMark:
    def test_price_is_the_next_print(self):
        raw = panel()
        by_ts = {row["ts"]: row["price"] for row in raw.to_dicts()}
        for row in prepared(raw).to_dicts():
            nxt = row["ts"] + timedelta(days=1)
            assert row["price"] == pytest.approx(by_ts[nxt])
