"""The contract between an uploaded strategy and the pipeline.

A strategy owns data, features, orders and unit delta. It does not own the
book: fills, inventory and PnL are pipeline code, written once so results
from different strategies are comparable.

The interface is an event loop. ``prepare`` is a tidy frame of
``ts, asset, price`` plus any features the strategy wants on the snapshot.
At each timestamp the engine calls ``on_tick`` with the actual filled
inventory; the strategy returns signed trades. Optional ``volume`` caps the
fill. Optional ``delta`` is the strategy's unit greek for that asset.
"""

from abc import ABC, abstractmethod
from typing import Any, Mapping

import polars as pl

STAGES = ("collect", "prepare", "fit", "simulate", "evaluate")

#: Strategy-owned stages. ``simulate`` is joint (strategy decides, engine fills).
STRATEGY_STAGES = ("collect", "prepare", "fit")

PREPARED_COLUMNS = ("ts", "asset", "price")
ORDER_COLUMNS = ("asset", "qty")
TAPE_COLUMNS = (
    "ts", "asset", "inventory", "fill", "price", "volume", "delta",
    "pnl", "fee", "slippage",
)

#: Frozen under permutation: the traded path, not the information set.
FROZEN_COLUMNS = frozenset((*PREPARED_COLUMNS, "volume"))

#: Optional per-row execution costs the prepared frame may carry. A strategy
#: that knows its own venue fees or its own liquidity puts them here, per asset
#: and per timestamp; the config default is a fallback for strategies that do
#: not.
#:
#: ``*_bps`` charge basis points of the traded notional. ``*_per_unit`` charge a
#: flat amount in currency per unit of turnover, which is the right shape for a
#: spread: basis points of a spread's *level* collapse to nothing as the spread
#: approaches zero, exactly when both of its legs still trade at full size.
COST_COLUMNS = ("fee_bps", "slippage_bps", "fee_per_unit", "slippage_per_unit")


class PanelError(ValueError):
    """A strategy returned a frame the engine cannot use."""


class Strategy(ABC):
    """Subclass this and upload the file.

    Each *stage* runs in its own subprocess, so nothing is carried on ``self``
    between collect / prepare / fit / simulate. Inside ``simulate``, ``on_tick``
    is called repeatedly on the same instance: tick-local state is allowed.
    """

    @abstractmethod
    def collect(self, config: dict) -> Any:
        """Fetch raw data. Any shape -- only this strategy reads it back."""

    @abstractmethod
    def prepare(self, raw: Any, config: dict) -> Any:
        """Clean and featurize ``raw``. Must include ``ts, asset, price``."""

    def fit(self, prepared: Any, config: dict) -> Any:
        """Train on ``prepared`` and return the model.

        Returning the model rather than assigning ``self.model`` is deliberate:
        stages are separate processes, so instance state would not survive.
        Override only if the strategy actually learns something -- the default
        no-op is correct for a rule-based strategy.
        """
        return None

    @abstractmethod
    def on_tick(
        self,
        ts: Any,
        market: pl.DataFrame,
        inventory: Mapping[str, float],
        model: Any,
        config: dict,
    ) -> Any:
        """Return trades for this timestamp: ``asset, qty``, optional ``delta``.

        ``market`` is every prepared row at ``ts`` (prices, volume, features).
        ``inventory`` is filled quantity after previous ticks, copied so
        mutating it cannot touch the engine book. ``qty`` is a signed trade,
        not a target; omitted assets do not trade. Unfilled remainder is
        dropped. ``delta`` is the unit greek of that asset; default 1.
        """


def to_polars(frame: Any, *, stage: str = "stage") -> pl.DataFrame:
    """Normalize a stage's return value to a polars DataFrame.

    Accepts polars or pandas so a strategy can use scikit-learn without
    fighting the boundary.
    """
    if isinstance(frame, pl.DataFrame):
        return frame
    if isinstance(frame, pl.LazyFrame):
        return frame.collect()
    if hasattr(frame, "to_dict") and type(frame).__module__.startswith("pandas"):
        return pl.from_pandas(frame)
    raise PanelError(
        f"{stage} returned {type(frame).__name__}; expected a polars or pandas DataFrame"
    )


def _no_nulls(df: pl.DataFrame, cols: tuple[str, ...] | list[str], *, what: str) -> None:
    present = [c for c in dict.fromkeys(cols) if c in df.columns]
    if not present:
        return
    null_counts = {c: n for c, n in zip(present, df.select(present).null_count().row(0)) if n}
    if null_counts:
        raise PanelError(f"{what} contains null values: {null_counts}")


def _non_negative(df: pl.DataFrame, cols: tuple[str, ...] | list[str], *, what: str) -> None:
    for col in cols:
        if col not in df.columns:
            continue
        if (worst := df[col].min()) is not None and worst < 0:
            raise PanelError(f"{what} column '{col}' must not be negative, got {worst}")


def validate_prepared(frame: Any) -> pl.DataFrame:
    """Check a prepare() frame. Extra columns (features) are kept."""
    df = to_polars(frame, stage="prepare")
    missing = [c for c in PREPARED_COLUMNS if c not in df.columns]
    if missing:
        raise PanelError(
            f"prepared frame is missing column(s) {missing}; "
            f"required {list(PREPARED_COLUMNS)}, got {df.columns}"
        )

    costs = [c for c in COST_COLUMNS if c in df.columns]
    numeric = ["price", *costs]
    if "volume" in df.columns:
        numeric.append("volume")
    for col in numeric:
        if not df.schema[col].is_numeric():
            raise PanelError(f"column '{col}' must be numeric, got {df.schema[col]}")

    df = df.with_columns(
        pl.col("asset").cast(pl.String),
        *[pl.col(c).cast(pl.Float64) for c in numeric],
    )
    _no_nulls(df, [*PREPARED_COLUMNS, *numeric], what="prepared frame")
    _non_negative(df, costs, what="prepared")
    if "volume" in df.columns:
        _non_negative(df, ("volume",), what="prepared")

    duplicates = df.height - df.select("ts", "asset").n_unique()
    if duplicates:
        raise PanelError(
            f"prepared frame has {duplicates} duplicate (ts, asset) row(s); "
            "each asset may appear at most once per timestamp"
        )
    return df.sort("ts", "asset")


def validate_orders(frame: Any, *, known: set[str]) -> pl.DataFrame:
    """Check an on_tick() return. Empty / None means no trade."""
    empty = pl.DataFrame({
        "asset": pl.Series(dtype=pl.String),
        "qty": pl.Series(dtype=pl.Float64),
        "delta": pl.Series(dtype=pl.Float64),
    })
    if frame is None:
        return empty
    df = to_polars(frame, stage="on_tick")
    if df.height == 0:
        return empty

    missing = [c for c in ORDER_COLUMNS if c not in df.columns]
    if missing:
        raise PanelError(
            f"orders are missing column(s) {missing}; "
            f"required {list(ORDER_COLUMNS)}, got {df.columns}"
        )
    if not df.schema["qty"].is_numeric():
        raise PanelError(f"column 'qty' must be numeric, got {df.schema['qty']}")
    has_delta = "delta" in df.columns
    if has_delta and not df.schema["delta"].is_numeric():
        raise PanelError(f"column 'delta' must be numeric, got {df.schema['delta']}")

    keep = ["asset", "qty"] + (["delta"] if has_delta else [])
    df = df.select(keep).with_columns(
        pl.col("asset").cast(pl.String),
        pl.col("qty").cast(pl.Float64),
        *([pl.col("delta").cast(pl.Float64)] if has_delta else []),
    )
    _no_nulls(df, keep, what="orders")

    unknown = sorted(set(df["asset"].to_list()) - known)
    if unknown:
        raise PanelError(f"orders reference unknown asset(s) {unknown}")
    duplicates = df.height - df.select("asset").n_unique()
    if duplicates:
        raise PanelError(
            f"orders have {duplicates} duplicate asset(s); "
            "each asset may appear at most once per tick"
        )
    if not has_delta:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("delta"))
    return df
