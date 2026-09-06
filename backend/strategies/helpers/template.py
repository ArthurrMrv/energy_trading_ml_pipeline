from abc import ABC, abstractmethod
from typing import Any, Mapping
import polars as pl


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