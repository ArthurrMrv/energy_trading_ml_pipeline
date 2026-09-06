"""Monte Carlo permutation test: is this book better than luck?

A Sharpe ratio on its own says nothing. The honest null for an event-loop
strategy is the same prices and liquidity, with the information set scrambled:
rotate or shuffle feature columns in time, re-run ``on_tick``, and see where
the real result falls.

**Prices, timestamps, volume and costs do not move.** Permuting the traded
path would invent a market that never existed. Feature columns are the
strategy's information set.

``rotate`` (default) circularly shifts features, preserving autocorrelation so
the null trades about as often as the real book. ``shuffle`` draws an i.i.d.
permutation and is the right choice only for a feature with no time structure.

Each permutation re-runs the full event loop. That is cheap on daily panels and
will hurt on true tick tapes.
"""

import random
from typing import Any, Callable

import polars as pl

from backend.pipeline.contract import COST_COLUMNS, FROZEN_COLUMNS, Strategy, validate_prepared
from backend.pipeline.engine import evaluate, simulate, to_book

METHODS = ("rotate", "shuffle")

METRICS = ("sharpe", "total_return", "annual_return", "hit_rate", "win_rate",
           "max_drawdown")

DEFAULTS = {"n": 0, "method": "rotate", "metric": "sharpe", "seed": 0}

PROGRESS_STEPS = 10
MAX_CURVES = 50
CURVE_POINTS = 240


def settings(config: dict) -> dict:
    """Read and validate the ``permutation`` block of a run config.

    Off unless asked for: the test costs ``n`` full event loops.
    """
    raw = (config or {}).get("permutation") or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config 'permutation' must be an object, got {type(raw).__name__}")

    unknown = sorted(set(raw) - set(DEFAULTS))
    if unknown:
        raise ValueError(
            f"unknown permutation option(s) {unknown}; expected {sorted(DEFAULTS)}"
        )

    options = {**DEFAULTS, **raw}

    count = options["n"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError(f"permutation 'n' must be a non-negative integer, got {count!r}")
    if options["method"] not in METHODS:
        raise ValueError(
            f"permutation 'method' must be one of {list(METHODS)}, "
            f"got {options['method']!r}"
        )
    if options["metric"] not in METRICS:
        raise ValueError(
            f"permutation 'metric' must be one of {list(METRICS)}, "
            f"got {options['metric']!r}"
        )
    if not isinstance(options["seed"], int) or isinstance(options["seed"], bool):
        raise ValueError(f"permutation 'seed' must be an integer, got {options['seed']!r}")

    return options


def enabled(config: dict) -> bool:
    return settings(config)["n"] > 0


def _feature_columns(df: pl.DataFrame) -> list[str]:
    frozen = FROZEN_COLUMNS | set(COST_COLUMNS)
    return [c for c in df.columns if c not in frozen]


def permute(prepared: Any, *, method: str, seed: int) -> pl.DataFrame:
    """Return a new frame with feature columns moved in time, per asset."""
    df = validate_prepared(prepared)
    feats = _feature_columns(df)
    if not feats:
        raise ValueError(
            "cannot permute a frame with no feature columns; "
            "add a feature or turn permutation off"
        )
    periods = df.select("ts").n_unique()
    if periods < 2:
        raise ValueError(
            f"cannot permute a panel with {periods} timestamp(s); at least 2 are needed"
        )

    length = pl.len()
    if method == "rotate":
        offset = random.Random(seed).randrange(1, periods)
        index = (pl.int_range(0, length, dtype=pl.UInt32) + offset) % length
    elif method == "shuffle":
        index = pl.int_range(0, length, dtype=pl.UInt32).shuffle(seed=seed)
    else:
        raise ValueError(f"unknown permutation method {method!r}; expected {list(METHODS)}")

    return df.with_columns([pl.col(c).gather(index).over("asset") for c in feats])


def _run(strategy: Strategy, prepared: pl.DataFrame, model: Any,
         config: dict, metric: str) -> tuple[float, list[float]]:
    tape = simulate(type(strategy)(), prepared, model, config)
    book = to_book(tape, config)
    return float(evaluate(tape, config)[metric]), book.sort("ts")["equity"].to_list()


def _thin(values: list[float], points: int = CURVE_POINTS) -> list[float]:
    """Evenly sample a curve down to ``points``, keeping both endpoints."""
    if len(values) <= points:
        return [round(float(v), 6) for v in values]
    step = (len(values) - 1) / (points - 1)
    return [round(float(values[round(i * step)]), 6) for i in range(points)]


def _summarize(null: list[float]) -> dict:
    series = pl.Series(null)
    return {
        "null_mean": round(float(series.mean()), 6),
        "null_std": round(float(series.std() or 0.0), 6),
        "null_min": round(float(series.min()), 6),
        "null_p50": round(float(series.median()), 6),
        "null_p95": round(float(series.quantile(0.95)), 6),
        "null_max": round(float(series.max()), 6),
    }


def permutation_test(
    strategy: Strategy,
    prepared: Any,
    model: Any,
    config: dict,
    *,
    tape: pl.DataFrame | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict | None:
    """Score the strategy against ``n`` permutations of its feature columns."""
    options = settings(config)
    count = options["n"]
    if count == 0:
        return None

    df = validate_prepared(prepared)
    metric = options["metric"]
    if tape is None:
        observed, observed_equity = _run(strategy, df, model, config, metric)
    else:
        observed = float(evaluate(tape, config)[metric])
        observed_equity = to_book(tape, config).sort("ts")["equity"].to_list()

    rng = random.Random(options["seed"])
    every = max(1, count // PROGRESS_STEPS)

    null: list[float] = []
    paths: list[list[float]] = []
    cls = type(strategy)
    for done in range(1, count + 1):
        permuted = permute(df, method=options["method"], seed=rng.randrange(2**31))
        score, equity = _run(cls(), permuted, model, config, metric)
        null.append(score)
        if len(paths) < MAX_CURVES:
            paths.append(_thin(equity))
        if on_progress and (done % every == 0 or done == count):
            on_progress(done, count)

    beaten = sum(1 for value in null if value >= observed)

    return {
        "metric": metric,
        "method": options["method"],
        "n": count,
        "seed": options["seed"],
        "observed": round(observed, 6),
        "p_value": round((1 + beaten) / (count + 1), 6),
        "beat_by_chance": beaten,
        **_summarize(null),
        "curves": {
            "x": [int(v) for v in _thin(list(range(len(observed_equity))))],
            "strategy": _thin(observed_equity),
            "null": paths,
        },
    }
