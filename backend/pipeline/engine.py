"""Tick-by-tick simulation and evaluation. Pipeline-owned, identical for every strategy.

The engine owns inventory and fills. Previous inventory earns this tick's price
change; this tick's fill starts earning next tick. Volume caps the fill;
unfilled remainder is dropped. Unit delta comes from the strategy.
"""

from typing import Any, Mapping

import polars as pl

from backend.pipeline.contract import TAPE_COLUMNS, Strategy, validate_orders, validate_prepared

DEFAULTS = {
    "cost_bps": 1.0,
    "fee_bps": None,
    "slippage_bps": 0.0,
    "fee_per_unit": 0.0,
    "slippage_per_unit": 0.0,
    "periods_per_year": 365,
    "return_mode": "pct",
}

RETURN_MODES = ("pct", "diff")


def _cfg(config: dict, key: str) -> Any:
    return (config or {}).get(key, DEFAULTS[key])


def _cost_scalar(config: dict, key: str) -> float:
    """Resolve one cost knob from config.

    ``fee_bps`` falls back to the older ``cost_bps``, so a run configured before
    fees and slippage were split still charges exactly what it used to.
    """
    value = (config or {}).get(key)
    if value is None:
        value = _cfg(config, "cost_bps") if key == "fee_bps" else DEFAULTS[key]
    rate = float(value)
    if rate < 0:
        raise ValueError(f"{key} must not be negative, got {rate}")
    return rate


def _mode(config: dict) -> str:
    mode = _cfg(config, "return_mode")
    if mode not in RETURN_MODES:
        raise ValueError(f"return_mode must be one of {list(RETURN_MODES)}, got {mode!r}")
    return mode


def _rates(config: dict) -> dict[str, float]:
    scale = {"fee_bps": 10_000.0, "slippage_bps": 10_000.0,
             "fee_per_unit": 1.0, "slippage_per_unit": 1.0}
    return {key: _cost_scalar(config, key) / scale[key] for key in scale}


def _charge(fill: float, price: float, mode: str, rates: dict[str, float],
            row: Mapping[str, Any]) -> tuple[float, float]:
    notional = 1.0 if mode == "pct" else abs(price)
    mag = abs(fill)

    def rate(key: str) -> float:
        if key in row and row[key] is not None:
            return float(row[key]) / (10_000.0 if key.endswith("_bps") else 1.0)
        return rates[key]

    fee = mag * (rate("fee_bps") * notional + rate("fee_per_unit"))
    slip = mag * (rate("slippage_bps") * notional + rate("slippage_per_unit"))
    return fee, slip


def _empty_tape() -> pl.DataFrame:
    return pl.DataFrame({
        "ts": [], "asset": pl.Series(dtype=pl.String),
        **{c: pl.Series(dtype=pl.Float64)
           for c in TAPE_COLUMNS if c not in ("ts", "asset")},
    })


def simulate(strategy: Strategy, prepared: Any, model: Any, config: dict) -> pl.DataFrame:
    """Walk timestamps, call ``on_tick``, fill, mark. Returns the inventory tape."""
    df = validate_prepared(prepared)
    mode = _mode(config)
    if df.height == 0:
        return _empty_tape()
    if mode == "pct" and (worst := df["price"].min()) is not None and worst <= 0:
        raise ValueError(
            f"return_mode='pct' needs strictly positive prices, got {worst}; "
            "a percentage return through zero is undefined -- use "
            "return_mode='diff' to mark PnL in currency instead"
        )

    rates = _rates(config)
    has_volume = "volume" in df.columns
    df = df.sort("ts", "asset")
    n = df.height
    ts_vals = df["ts"].to_list()

    inventory: dict[str, float] = {}
    last_price: dict[str, float] = {}
    last_delta: dict[str, float] = {}
    rows: list[dict] = []

    i = 0
    while i < n:
        j = i + 1
        ts = ts_vals[i]
        while j < n and ts_vals[j] == ts:
            j += 1
        market = df.slice(i, j - i)
        known = set(market["asset"].to_list())
        snapshot = market.to_dicts()

        orders = validate_orders(
            strategy.on_tick(ts, market, dict(inventory), model, config or {}),
            known=known,
        )
        qty = dict(zip(orders["asset"].to_list(), orders["qty"].to_list()))
        deltas = {
            a: d for a, d in zip(orders["asset"].to_list(), orders["delta"].to_list())
            if d is not None
        }

        for snap in snapshot:
            asset, price = snap["asset"], float(snap["price"])
            prev = inventory.get(asset, 0.0)
            px0 = last_price.get(asset)
            if px0 is None:
                ret = 0.0
            elif mode == "pct":
                ret = price / px0 - 1.0
            else:
                ret = price - px0
            last_price[asset] = price

            volume = float(snap["volume"]) if has_volume else float("inf")
            q = float(qty.get(asset, 0.0))
            fill = (1.0 if q > 0 else -1.0 if q < 0 else 0.0) * min(abs(q), volume)
            inventory[asset] = prev + fill
            if asset in deltas:
                last_delta[asset] = deltas[asset]
            fee, slip = _charge(fill, price, mode, rates, snap)
            rows.append({
                "ts": ts, "asset": asset,
                "inventory": inventory[asset], "fill": fill,
                "price": price, "volume": volume,
                "delta": last_delta.get(asset, 1.0),
                "pnl": prev * ret, "fee": fee, "slippage": slip,
            })
        i = j

    return pl.DataFrame(rows).select(*TAPE_COLUMNS)


def to_book(tape: pl.DataFrame, config: dict) -> pl.DataFrame:
    """Collapse the per-asset tape to one row per timestamp."""
    mode = _mode(config)
    if tape.height == 0:
        return pl.DataFrame({
            "ts": [], "pnl": [], "fee": [], "slippage": [], "cost": [],
            "turnover": [], "net_exposure": [], "gross_exposure": [],
            "delta": [], "net": [], "equity": [],
        })
    book = (
        tape.group_by("ts")
        .agg(
            pl.col("pnl").sum().alias("pnl"),
            pl.col("fee").sum().alias("fee"),
            pl.col("slippage").sum().alias("slippage"),
            pl.col("fill").abs().sum().alias("turnover"),
            pl.col("inventory").sum().alias("net_exposure"),
            pl.col("inventory").abs().sum().alias("gross_exposure"),
            (pl.col("inventory") * pl.col("delta")).sum().alias("delta"),
        )
        .sort("ts")
        .with_columns((pl.col("fee") + pl.col("slippage")).alias("cost"))
        .with_columns((pl.col("pnl") - pl.col("cost")).alias("net"))
    )
    if mode == "pct":
        compounded = (1.0 + pl.col("net")).cum_prod()
        equity = pl.when(compounded.cum_min() <= 0).then(0.0).otherwise(compounded)
    else:
        equity = pl.col("net").cum_sum()
    return book.with_columns(equity.alias("equity"))


def _cost_totals(bt: pl.DataFrame) -> dict:
    total = round(float(bt["cost"].sum() or 0.0), 6) if "cost" in bt.columns else 0.0
    if "fee" not in bt.columns:
        return {"total_cost": total, "total_fees": None, "total_slippage": None}
    return {
        "total_cost": total,
        "total_fees": round(float(bt["fee"].sum() or 0.0), 6),
        "total_slippage": round(float(bt["slippage"].sum() or 0.0), 6),
    }


def _ruined_at(bt: pl.DataFrame, mode: str):
    if mode != "pct" or bt.height == 0 or "equity" not in bt.columns:
        return None
    dead = bt.sort("ts").filter(pl.col("equity") <= 0)
    return dead["ts"][0] if dead.height else None


def _exposure(bt: pl.DataFrame) -> tuple[int, float]:
    if "gross_exposure" not in bt.columns:
        return 0, 0.0
    exposed = bt.filter(pl.col("gross_exposure") > 0)
    if not exposed.height:
        return 0, 0.0
    return exposed.height, round(float((exposed["net"] > 0).mean()), 4)


def _n_trades(bt: pl.DataFrame) -> int:
    if bt.height == 0 or "turnover" not in bt.columns:
        return 0
    return int((bt["turnover"] > 0).sum())


def _terminal_inventory(tape: pl.DataFrame) -> dict[str, float]:
    if tape.height == 0:
        return {}
    last_ts = tape.sort("ts")["ts"][-1]
    last = tape.filter(pl.col("ts") == last_ts)
    return {a: float(q) for a, q in zip(last["asset"].to_list(), last["inventory"].to_list())}


def _greeks(bt: pl.DataFrame, exposed_periods: int) -> dict:
    empty = {"delta": 0.0, "gross": 0.0, "time_in_market": 0.0}
    if bt.height == 0:
        return empty
    time_in_market = round(exposed_periods / bt.height, 4)
    delta = round(float(bt["delta"].mean() or 0.0), 6) if "delta" in bt.columns else 0.0
    gross = (
        round(float(bt["gross_exposure"].mean() or 0.0), 6)
        if "gross_exposure" in bt.columns else 0.0
    )
    return {"delta": delta, "gross": gross, "time_in_market": time_in_market}


def evaluate(tape: pl.DataFrame, config: dict) -> dict:
    """Summarize a tape into JSON-serializable metrics."""
    periods_per_year = float(_cfg(config, "periods_per_year"))
    mode = _mode(config)
    bt = to_book(tape, config)
    net, equity = bt["net"], bt["equity"]

    exposed_periods, hit_rate = _exposure(bt)
    n_trades = _n_trades(bt)
    greeks = _greeks(bt, exposed_periods)
    ruined_at = _ruined_at(bt, mode)
    header = {
        "n_obs": bt.height,
        "n_trades": n_trades,
        "return_mode": mode,
        "units": "fraction" if mode == "pct" else "currency",
        "exposed_periods": exposed_periods,
        "ruined_at": ruined_at,
        "inventory": _terminal_inventory(tape),
    }

    if bt.height == 0:
        return {**header, "sharpe": 0.0, "max_drawdown": 0.0, "total_return": 0.0,
                "annual_return": 0.0, "volatility": 0.0, "avg_turnover": 0.0,
                "hit_rate": 0.0, "win_rate": 0.0, "greeks": greeks,
                **_cost_totals(bt)}

    std = net.std() or 0.0
    mean = net.mean() or 0.0
    sharpe = (mean / std) * periods_per_year**0.5 if std > 0 else 0.0

    peak = equity.cum_max()
    if ruined_at is not None:
        drawdown, total = -1.0, -1.0
    elif mode == "pct":
        drawdown = (equity / peak - 1.0).min()
        total = equity[-1] - 1.0
    else:
        drawdown = (equity - peak).min()
        total = equity[-1]

    return {
        **header,
        "sharpe": round(float(sharpe), 4),
        "max_drawdown": round(float(drawdown), 6),
        "total_return": round(float(total), 6),
        "annual_return": round(float(mean * periods_per_year), 6),
        "volatility": round(float(std * periods_per_year**0.5), 6),
        "avg_turnover": round(float(bt["turnover"].mean() or 0.0), 6),
        "hit_rate": hit_rate,
        "win_rate": hit_rate,
        "greeks": greeks,
        **_cost_totals(bt),
    }
