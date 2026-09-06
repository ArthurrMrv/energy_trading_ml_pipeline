"""Reference strategy: mean reversion on cross-border day-ahead power spreads.

Coupled bidding zones are arbitraged by interconnectors, so the spread between
them is anchored. When it stretches, it tends to come back -- that is the whole
idea. Each pair trades as one asset whose price *is* the spread, which is why
runs of this strategy want ``return_mode: "diff"``: a spread is signed and
crosses zero, so a percentage return on it is meaningless.

European day-ahead clock (Ember gold is delivery-dated): the print on day ``t``
was auctioned at noon ``t-1``. Features therefore use only ``spread_{t-1}``;
the mark is ``spread_{t+1}`` (the next tradeable DA print). A fill at ``t``
earns from ``t`` to ``t+1``, so PnL is ``spread_{t+2} - spread_{t+1}``.

This file is an example of the contract, not part of the pipeline. Nothing in
engine.py or orchestrator.py knows it exists.

A caution on its results: the headline Sharpe is not an alpha claim. It is
flattered by the 2022 energy crisis (the FRA-DEU spread reaches 462 EUR/MWh),
by zero market impact, and by assuming fills at the next DA print. Its purpose
is to exercise the pipeline end to end with real data, not to be traded.
"""

import polars as pl

from backend.pipeline.contract import Strategy
from backend.pipeline.loader import load_module
from backend.pipeline.store import ROOT

_lake = load_module(ROOT / "backend" / "data" / "code" / "base.py")
GOLD = ROOT / _lake.DataPaths.gold_path
PRICES = GOLD / "european_wholesale_electricity_price_data_daily.parquet"

DEFAULTS = {
    "pairs": [["FRA", "DEU"], ["ESP", "PRT"], ["NLD", "BEL"]],
    "start_date": "2015-01-01",
    "end_date": "2020-12-31",
    "lookback": 30,
    "entry_z": 2.0,
}


class SpreadMeanReversion(Strategy):

    def collect(self, config: dict) -> pl.DataFrame:
        pairs = config.get("pairs", DEFAULTS["pairs"])
        codes = sorted({code for pair in pairs for code in pair})
        start = config.get("start_date", DEFAULTS["start_date"])
        end = config.get("end_date", DEFAULTS["end_date"])

        prices = (
            pl.read_parquet(PRICES)
            .select(
                pl.col("Date").str.to_date().alias("ts"),
                pl.col("ISO3 Code").alias("country"),
                pl.col("Price (EUR/MWhe)").alias("price"),
            )
            .filter(pl.col("country").is_in(codes))
            .filter(pl.col("ts").is_between(pl.lit(start).str.to_date(),
                                            pl.lit(end).str.to_date()))
            .sort("ts", "country")
        )

        found = set(prices["country"].unique().to_list())
        if missing := sorted(set(codes) - found):
            raise ValueError(
                f"no price data for {missing}; available ISO3 codes include "
                f"{sorted(found)[:8]}..."
            )
        print(f"collected {prices.height} price rows for {len(found)} zones")
        return prices

    def prepare(self, raw: pl.DataFrame, config: dict) -> pl.DataFrame:
        pairs = config.get("pairs", DEFAULTS["pairs"])
        lookback = int(config.get("lookback", DEFAULTS["lookback"]))

        wide = raw.pivot(on="country", index="ts", values="price").sort("ts").drop_nulls()

        spreads = pl.concat([
            wide.select(
                pl.col("ts"),
                pl.lit(f"{a}-{b}").alias("asset"),
                (pl.col(a) - pl.col(b)).alias("spread"),
            )
            for a, b in pairs
        ]).sort("asset", "ts")

        featured = spreads.with_columns(
            pl.col("spread").shift(1).over("asset").alias("known"),
        ).with_columns(
            pl.col("known").rolling_mean(lookback).over("asset").alias("spread_mean"),
            pl.col("known").rolling_std(lookback).over("asset").alias("spread_std"),
        ).with_columns(
            pl.when(pl.col("spread_std") > 0)
            .then((pl.col("known") - pl.col("spread_mean")) / pl.col("spread_std"))
            .otherwise(0.0)
            .alias("z"),
            pl.col("spread").shift(-1).over("asset").alias("price"),
        ).drop("known")

        prepared = featured.drop_nulls()
        print(f"prepared {prepared.height} rows across {len(pairs)} spreads "
              f"(dropped {featured.height - prepared.height} warmup rows)")
        return prepared

    def on_tick(self, ts, market, inventory, model, config: dict):
        entry_z = float(config.get("entry_z", DEFAULTS["entry_z"]))
        assets = market["asset"].to_list()
        z = market["z"].to_list()

        raise ValueError("This line is clearly an error, should be removed")
        
        qty, delta = [], []
        for asset, zi in zip(assets, z):
            target = max(-1.0, min(1.0, -(zi / entry_z)))
            qty.append(target - inventory.get(asset, 0.0))
            delta.append(1.0)
        return pl.DataFrame({"asset": assets, "qty": qty, "delta": delta})
