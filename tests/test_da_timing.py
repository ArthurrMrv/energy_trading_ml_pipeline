"""European day-ahead timing guarantees for the spread strategies.

Ember gold is delivery-dated: the print on day ``t`` was auctioned at noon
``t-1``. Strategies must therefore:

1. Build features at ``t`` from prints / loads through ``t-1`` only.
2. Mark the panel at the next DA print ``spread_{t+1}``.
3. (ML) Label the next *tradeable* move ``spread_{t+2} - spread_{t+1}``.

These are algebraic identities on a synthetic panel -- no gold parquet, no
seeds on the assertion, no tolerance on the clock.
"""

from datetime import date, timedelta

import polars as pl
import pytest

from backend.strategies.helpers import ml_features as feat
from backend.strategies.spread_meanrev_toFix import SpreadMeanReversion

PAIRS = [["FRA", "DEU"]]
ASSET = "FRA-DEU"
LOOKBACK = 5
N_DAYS = 80


def _dates(n=N_DAYS, start=date(2015, 6, 1)):
    return [start + timedelta(days=i) for i in range(n)]


def synthetic_prices(n=N_DAYS, *, scramble_from: date | None = None):
    """FRA/DEU daily prices. Optionally scramble prints from ``scramble_from``."""
    rows = []
    for i, ts in enumerate(_dates(n)):
        fra, deu = 40.0 + i * 0.1, 35.0 + i * 0.05
        if scramble_from is not None and ts >= scramble_from:
            fra, deu = 999.0 + i, -999.0 - i
        rows.append({"ts": ts, "country": "FRA", "price": fra})
        rows.append({"ts": ts, "country": "DEU", "price": deu})
    return pl.DataFrame(rows).sort("ts", "country")


def synthetic_loads(n=N_DAYS, *, scramble_from: date | None = None):
    """FR/DE daily loads (enough for emea_west / emea_central after intersect)."""
    rows = []
    for i, ts in enumerate(_dates(n)):
        fr, de = 50_000.0 + i, 60_000.0 + i * 2
        if scramble_from is not None and ts >= scramble_from:
            fr, de = 1.0, 2.0
        rows.append({"ts": ts, "country": "FR", "load": fr})
        rows.append({"ts": ts, "country": "DE", "load": de})
    return pl.DataFrame(rows).sort("ts", "country")


def spreads_from(prices: pl.DataFrame) -> pl.DataFrame:
    return feat.build_spreads(prices, PAIRS)


def meanrev_prepared(prices: pl.DataFrame):
    return SpreadMeanReversion().prepare(prices, {"pairs": PAIRS, "lookback": LOOKBACK})


def meanrev_qty(prices: pl.DataFrame, ts):
    strategy = SpreadMeanReversion()
    prepared = strategy.prepare(prices, {"pairs": PAIRS, "lookback": LOOKBACK})
    market = prepared.filter(pl.col("ts") == ts)
    orders = strategy.on_tick(ts, market, {}, None, {"entry_z": 2.0})
    return orders["qty"].to_list()


def ml_prepared(prices: pl.DataFrame, loads: pl.DataFrame):
    return feat.prepare_spread_features(prices, loads, PAIRS)


PROBE = _dates()[40]


class TestSameBarInvariance:
    """Scrambling spread[t] (and later) must not change the order / features at t."""

    def test_meanrev_qty_at_t_ignores_same_bar_print(self):
        assert meanrev_qty(synthetic_prices(), PROBE) == meanrev_qty(
            synthetic_prices(scramble_from=PROBE), PROBE
        )

    def test_meanrev_qty_at_t_plus_1_may_see_scrambled_print(self):
        nxt = PROBE + timedelta(days=1)
        assert meanrev_qty(synthetic_prices(), nxt) != meanrev_qty(
            synthetic_prices(scramble_from=PROBE), nxt
        )

    def test_ml_features_at_t_ignore_same_bar_print_and_load(self):
        clean = ml_prepared(synthetic_prices(), synthetic_loads())
        dirty = ml_prepared(
            synthetic_prices(scramble_from=PROBE),
            synthetic_loads(scramble_from=PROBE),
        )
        cols = feat.feature_columns(clean)
        assert cols
        assert clean.filter(pl.col("ts") == PROBE).select(cols).row(0) == (
            dirty.filter(pl.col("ts") == PROBE).select(cols).row(0)
        )


class TestNextPrintMarkAndLabel:
    """Panel price is spread[t+1]; ML label is spread[t+2] - spread[t+1]."""

    def test_meanrev_price_is_the_next_print(self):
        prices = synthetic_prices()
        prepared = meanrev_prepared(prices)
        by_ts = {row["ts"]: row["spread"] for row in spreads_from(prices).to_dicts()}
        for row in prepared.to_dicts():
            nxt = row["ts"] + timedelta(days=1)
            assert nxt in by_ts
            assert row["price"] == pytest.approx(by_ts[nxt])

    def test_ml_label_is_the_next_tradeable_move(self):
        prices = synthetic_prices()
        prepared = ml_prepared(prices, synthetic_loads())
        by_ts = {row["ts"]: row["spread"] for row in spreads_from(prices).to_dicts()}
        for row in prepared.to_dicts():
            t1 = row["ts"] + timedelta(days=1)
            t2 = row["ts"] + timedelta(days=2)
            assert t1 in by_ts and t2 in by_ts
            assert row["label"] == pytest.approx(by_ts[t2] - by_ts[t1])

    def test_ml_price_is_the_next_print(self):
        prices = synthetic_prices()
        prepared = ml_prepared(prices, synthetic_loads())
        by_ts = {row["ts"]: row["spread"] for row in spreads_from(prices).to_dicts()}
        for row in prepared.to_dicts():
            nxt = row["ts"] + timedelta(days=1)
            assert row["price"] == pytest.approx(by_ts[nxt])

    def test_ml_on_tick_sizes_from_the_stub_prediction(self):
        from backend.strategies.helpers import spread_gbm as gbm

        prepared = ml_prepared(synthetic_prices(), synthetic_loads())
        market = prepared.filter(pl.col("ts") == PROBE)

        class Stub:
            def predict(self, x):
                return [0.0] * len(x)

        features = feat.feature_columns(prepared)[:3]
        orders = gbm.on_tick(
            market, {}, {"model": Stub(), "features": features}, {"signal_scale": 5.0},
        )
        assert orders["qty"].to_list() == [0.0]
        assert orders["asset"].to_list() == [ASSET]
