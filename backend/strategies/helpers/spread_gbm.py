"""Shared collect / prepare / fit / on_tick logic for the XGBoost spread strategy.

Selection runs inside ``fit`` so each ``config.split`` train (or test-rolling
refit) session re-selects on that fit window only. Not a Strategy -- do not
upload; lives under ``helpers/``.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import polars as pl
from sklearn.feature_selection import SelectFromModel

from backend.strategies.helpers import ml_features as feat

DEFAULTS = {
    # Load gold in this repo covers 2015–2017 and 2020 only (2018–2019 absent).
    "pairs": [["FRA", "DEU"], ["ESP", "PRT"], ["NLD", "BEL"]],
    "start_date": "2015-06-01",
    "end_date": "2017-12-31",
    "top_features": 40,
    "signal_scale": 5.0,
    "min_train_rows": 200,
}


def _cfg(config: dict, key: str):
    return (config or {}).get(key, DEFAULTS[key])


def collect(config: dict) -> pl.DataFrame:
    """Prices + daily loads in one long frame so checkpoints stay parquet and
    ``config.split`` can truncate on ``ts``.
    """
    pairs = _cfg(config, "pairs")
    start, end = _cfg(config, "start_date"), _cfg(config, "end_date")
    codes = sorted({c for pair in pairs for c in pair})
    iso2 = sorted({feat.ISO3_TO_ISO2[c] for c in codes if c in feat.ISO3_TO_ISO2})
    region_iso2 = sorted({c for members in feat.EMEA_REGIONS.values() for c in members})

    prices = feat.load_prices(codes, start, end).select(
        "ts", "country",
        pl.col("price").alias("value"),
        pl.lit("price").alias("kind"),
    )
    loads = feat.load_daily_load(sorted(set(iso2) | set(region_iso2)), start, end).select(
        "ts", "country",
        pl.col("load").alias("value"),
        pl.lit("load").alias("kind"),
    )
    raw = pl.concat([prices, loads]).sort("ts", "kind", "country")
    print(f"collected {prices.height} price rows, {loads.height} daily load rows "
          f"({loads['country'].n_unique()} zones)")
    return raw


def prepare(raw: pl.DataFrame, config: dict) -> pl.DataFrame:
    pairs = _cfg(config, "pairs")
    prices = (
        raw.filter(pl.col("kind") == "price")
        .select("ts", "country", pl.col("value").alias("price"))
    )
    loads = (
        raw.filter(pl.col("kind") == "load")
        .select("ts", "country", pl.col("value").alias("load"))
    )
    prepared = feat.prepare_spread_features(prices, loads, pairs)
    print(f"prepared {prepared.height} rows, "
          f"{len(feat.feature_columns(prepared))} candidate features "
          f"across {prepared['asset'].n_unique()} spreads")
    return prepared


def fit_booster(
    prepared: pl.DataFrame,
    config: dict,
    *,
    make_model: Callable[[], Any],
    name: str,
) -> dict:
    """Train ``make_model()`` after importance-based feature selection."""
    top = int(_cfg(config, "top_features"))
    min_rows = int(_cfg(config, "min_train_rows"))
    if prepared.height < min_rows:
        raise ValueError(
            f"{name} needs at least {min_rows} training rows, got {prepared.height}"
        )

    columns = feat.feature_columns(prepared)
    x = feat.matrix(prepared, columns)
    y = prepared["label"].to_numpy()

    # First pass: fit on everything to rank features.
    probe = make_model()
    probe.fit(x, y)
    selector = SelectFromModel(probe, max_features=min(top, len(columns)), prefit=True)
    mask = selector.get_support()
    selected = [col for col, keep in zip(columns, mask) if keep]
    if not selected:
        # Fallback: keep the top-importance columns by hand.
        importances = np.asarray(probe.feature_importances_, dtype=float)
        order = np.argsort(importances)[::-1][: min(top, len(columns))]
        selected = [columns[i] for i in order]

    x_sel = feat.matrix(prepared, selected)
    model = make_model()
    model.fit(x_sel, y)
    print(f"{name} fitted on {prepared.height} rows × {len(selected)} features "
          f"(from {len(columns)} candidates)")
    return {"model": model, "features": selected, "name": name}


def on_tick(market: pl.DataFrame, inventory: dict, model: dict | None, config: dict) -> pl.DataFrame:
    if not isinstance(model, dict) or "model" not in model or "features" not in model:
        raise ValueError(
            "on_tick expects the dict returned by fit() "
            "(keys: model, features); re-run from fit if the checkpoint is stale"
        )
    booster = model["model"]
    if not hasattr(booster, "predict"):
        raise TypeError(
            f"fit checkpoint's 'model' is {type(booster).__name__}, not an estimator "
            "-- the model was stringified when saved; re-run from fit after upgrading "
            "the pipeline (model dicts are pickled now)"
        )

    scale = float(_cfg(config, "signal_scale"))
    if scale <= 0:
        raise ValueError(f"signal_scale must be positive, got {scale}")
    features = model["features"]
    missing = [c for c in features if c not in market.columns]
    if missing:
        raise ValueError(f"prepared frame is missing selected features {missing}")

    pred = np.asarray(booster.predict(feat.matrix(market, features)), dtype=float)
    target = np.clip(pred / scale, -1.0, 1.0)
    assets = market["asset"].to_list()
    qty = [float(t) - float(inventory.get(a, 0.0)) for t, a in zip(target, assets)]
    return pl.DataFrame({"asset": assets, "qty": qty, "delta": [1.0] * len(assets)})
