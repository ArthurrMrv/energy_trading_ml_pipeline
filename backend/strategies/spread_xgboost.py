"""XGBoost on EMEA load + spread features for cross-border power spreads.

Predicts the next *tradeable* day-ahead move
(``spread_{t+2} - spread_{t+1}``). Regional loads, rolling 20/30 windows and
lags are built causally in ``prepare`` (features at ``t`` see only prints /
loads through ``t-1``); feature selection runs inside ``fit`` so a
``config.split`` train session never selects on test.

Upload **this file only** (helpers under ``backend/strategies/helpers/`` are
not strategies).

Use ``return_mode: "diff"``. ``prepare`` marks ``price`` at the next DA print.
Load gold in this repo covers 2015–2017 and 2020
only — keep ``start_date`` / ``end_date`` inside a continuous window.
"""

from xgboost import XGBRegressor

from backend.pipeline.contract import Strategy
from backend.strategies.helpers import spread_gbm as gbm


def _make_model(config: dict) -> XGBRegressor:
    return XGBRegressor(
        max_depth=int(config.get("depth", 4)),
        learning_rate=float(config.get("learning_rate", 0.05)),
        n_estimators=int(config.get("iterations", 200)),
        objective="reg:squarederror",
        random_state=int(config.get("seed", 0)),
        n_jobs=1,
        verbosity=0,
    )


class SpreadXGBoost(Strategy):

    def collect(self, config: dict):
        return gbm.collect(config)

    def prepare(self, raw, config: dict):
        return gbm.prepare(raw, config)

    def fit(self, prepared, config: dict):
        return gbm.fit_booster(
            prepared, config,
            make_model=lambda: _make_model(config),
            name="xgboost",
        )

    def on_tick(self, ts, market, inventory, model, config: dict):
        return gbm.on_tick(market, inventory, model, config)
