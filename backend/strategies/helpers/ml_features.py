"""Shared feature generation for ML spread strategies (not uploadable).

Causal only: rolling windows and lags look backward. Labels are shifted
forward and must never appear in the feature matrix. Feature *selection*
belongs in ``fit``, not here -- that is what keeps ``config.split`` honest.

European day-ahead clock (delivery-dated prints): features at ``t`` see only
information through ``t-1``; the label is the next *tradeable* move
``spread_{t+2} - spread_{t+1}``.
"""

from __future__ import annotations

import polars as pl

from backend.pipeline.loader import load_module
from backend.pipeline.store import ROOT

_lake = load_module(ROOT / "backend" / "data" / "code" / "base.py")
GOLD = ROOT / _lake.DataPaths.gold_path
PRICES = GOLD / "european_wholesale_electricity_price_data_daily.parquet"
LOADS = GOLD / "monthly_hourly_load_values.parquet"

#: Ember ISO3 ↔ ENTSO-E ISO2 for the zones we actually join.
ISO3_TO_ISO2 = {
    "FRA": "FR", "DEU": "DE", "ESP": "ES", "PRT": "PT",
    "NLD": "NL", "BEL": "BE", "AUT": "AT", "CHE": "CH",
    "CZE": "CZ", "POL": "PL", "SVK": "SK", "HUN": "HU",
    "NOR": "NO", "SWE": "SE", "FIN": "FI", "DNK": "DK",
    "ITA": "IT", "GRC": "GR", "HRV": "HR", "SVN": "SI",
    "ROU": "RO", "BGR": "BG", "EST": "EE", "LVA": "LV",
    "LTU": "LT", "GBR": "GB", "IRL": "IE",
}

#: EMEA sub-regions used as aggregated load features.
EMEA_REGIONS = {
    "emea_west": ["FR", "ES", "PT", "BE", "NL", "GB", "IE"],
    "emea_central": ["DE", "AT", "CH", "CZ", "PL", "SK", "HU"],
    "emea_nordics": ["NO", "SE", "FI", "DK"],
    "emea_se": ["IT", "GR", "HR", "SI", "RO", "BG"],
    "emea_ee": ["EE", "LV", "LT"],
}

ROLL_WINDOWS = (20, 30)
LAGS = (1, 5, 20, 30)

META_COLS = ("ts", "asset", "spread", "label", "price")


def load_prices(codes: list[str], start: str, end: str) -> pl.DataFrame:
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
        raise ValueError(f"no price data for {missing}")
    return prices


def load_daily_load(iso2_codes: list[str], start: str, end: str) -> pl.DataFrame:
    """Hourly ENTSO-E load → daily mean MW, filtered to the requested zones."""
    codes = sorted(set(iso2_codes))
    daily = (
        pl.read_parquet(LOADS)
        .select(
            pl.col("DateUTC").dt.date().alias("ts"),
            pl.col("CountryCode").alias("country"),
            pl.col("Value").alias("load"),
        )
        .filter(pl.col("country").is_in(codes))
        .filter(pl.col("ts").is_between(pl.lit(start).str.to_date(),
                                        pl.lit(end).str.to_date()))
        .group_by("ts", "country")
        .agg(pl.col("load").mean().alias("load"))
        .sort("ts", "country")
    )
    found = set(daily["country"].unique().to_list())
    if missing := sorted(set(codes) - found):
        raise ValueError(
            f"no load data for {missing}; load gold currently covers "
            f"{sorted(found)[:10]}..."
        )
    return daily


def region_membership(iso2_codes: list[str]) -> dict[str, list[str]]:
    """Intersect configured EMEA regions with the countries we actually have."""
    available = set(iso2_codes)
    return {
        name: [c for c in members if c in available]
        for name, members in EMEA_REGIONS.items()
        if any(c in available for c in members)
    }


def _roll_and_lag(frame: pl.DataFrame, col: str, *, over: str | None) -> pl.DataFrame:
    """Add rolling mean/std at 20/30 and lags 1/5/20/30 for one numeric column.

    Rolling windows use the one-bar-lagged series so the same-bar print is
    excluded. Lags are of the original series and remain causal.
    """
    exprs = []
    base = pl.col(col)
    known = base.shift(1)
    for window in ROLL_WINDOWS:
        mean = known.rolling_mean(window)
        std = known.rolling_std(window)
        if over:
            mean = mean.over(over)
            std = std.over(over)
        exprs.append(mean.alias(f"{col}_rmean_{window}"))
        exprs.append(std.alias(f"{col}_rstd_{window}"))
    for lag in LAGS:
        lagged = base.shift(lag)
        if over:
            lagged = lagged.over(over)
        exprs.append(lagged.alias(f"{col}_lag_{lag}"))
    return frame.with_columns(exprs)


def build_spreads(prices: pl.DataFrame, pairs: list[list[str]]) -> pl.DataFrame:
    wide = prices.pivot(on="country", index="ts", values="price").sort("ts").drop_nulls()
    return pl.concat([
        wide.select(
            pl.col("ts"),
            pl.lit(f"{a}-{b}").alias("asset"),
            (pl.col(a) - pl.col(b)).alias("spread"),
        )
        for a, b in pairs
    ]).sort("asset", "ts")


def build_region_loads(daily_load: pl.DataFrame, regions: dict[str, list[str]]) -> pl.DataFrame:
    """One column per EMEA region: sum of member-country daily loads."""
    wide = daily_load.pivot(on="country", index="ts", values="load").sort("ts")
    cols = [
        pl.sum_horizontal([pl.col(c) for c in members]).alias(name)
        for name, members in regions.items()
        if members and all(c in wide.columns for c in members)
    ]
    if not cols:
        raise ValueError("no EMEA region could be built from available load countries")
    return wide.select("ts", *cols)


def prepare_spread_features(
    prices: pl.DataFrame,
    daily_load: pl.DataFrame,
    pairs: list[list[str]],
) -> pl.DataFrame:
    """Long panel: one row per (ts, spread asset) with causal features + label.

    Label is the next *tradeable* DA move ``spread_{t+2} - spread_{t+1}``.
    It is kept on the frame for ``fit`` and must be excluded from the feature
    matrix. ``price`` is the next DA print (the mark the engine fills at).
    """
    spreads = build_spreads(prices, pairs)
    iso2_needed = sorted({
        ISO3_TO_ISO2[code]
        for pair in pairs for code in pair
        if code in ISO3_TO_ISO2
    })
    # Always pull the full EMEA membership that exists in the load frame so
    # regional aggregates are stable across pair configs.
    all_iso2 = sorted(set(daily_load["country"].unique().to_list()) | set(iso2_needed))
    regions = region_membership(all_iso2)
    region_loads = build_region_loads(daily_load, regions)

    # Per-leg daily load, pivoted for join onto each spread.
    leg_codes = sorted({ISO3_TO_ISO2[c] for pair in pairs for c in pair})
    leg_load = (
        daily_load.filter(pl.col("country").is_in(leg_codes))
        .pivot(on="country", index="ts", values="load")
        .sort("ts")
    )

    featured = []
    for a, b in pairs:
        a2, b2 = ISO3_TO_ISO2[a], ISO3_TO_ISO2[b]
        asset = f"{a}-{b}"
        panel = (
            spreads.filter(pl.col("asset") == asset)
            .join(region_loads, on="ts", how="inner")
            .join(leg_load.select("ts", a2, b2), on="ts", how="inner")
            .rename({a2: "load_a", b2: "load_b"})
            .with_columns(
                (pl.col("load_a") - pl.col("load_b")).alias("load_spread"),
                pl.col("ts").dt.weekday().alias("dow"),
                pl.col("ts").dt.month().alias("month"),
            )
        )
        for col in ("spread", "load_a", "load_b", "load_spread", *regions):
            if col in panel.columns:
                panel = _roll_and_lag(panel, col, over=None)
        # Raw load / region columns are features: shift so same-bar actuals
        # cannot enter the matrix. Spread stays unshifted (meta only).
        raw_feats = [
            c for c in ("load_a", "load_b", "load_spread", *regions)
            if c in panel.columns
        ]
        if raw_feats:
            panel = panel.with_columns(
                [pl.col(c).shift(1).alias(c) for c in raw_feats]
            )
        panel = panel.with_columns(
            (pl.col("spread").shift(-2) - pl.col("spread").shift(-1)).alias("label"),
            pl.col("spread").shift(-1).alias("price"),
        )
        featured.append(panel)

    out = pl.concat(featured, how="diagonal_relaxed").sort("asset", "ts")
    # Warmup / horizon: rows that cannot yet compute the longest roll or the
    # forward label are dropped once, here.
    return out.drop_nulls()


def feature_columns(prepared: pl.DataFrame) -> list[str]:
    return [c for c in prepared.columns if c not in META_COLS]


def matrix(prepared: pl.DataFrame, columns: list[str]):
    """Numpy feature matrix in a stable column order."""
    return prepared.select(columns).to_numpy()
