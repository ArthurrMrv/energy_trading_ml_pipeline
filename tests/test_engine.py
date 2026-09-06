"""Engine tests: fills, inventory, same-tick MTM, costs, greeks."""

import polars as pl
import pytest

from backend.pipeline.contract import (
    PREPARED_COLUMNS, PanelError, validate_orders, validate_prepared,
)
from backend.pipeline.engine import evaluate, simulate, to_book


class ColumnTrader:
    """Orders the ``qty`` column; optional ``unit_delta`` becomes ``delta``."""

    def on_tick(self, ts, market, inventory, model, config):
        cols = [pl.col("asset"), pl.col("qty")]
        if "unit_delta" in market.columns:
            cols.append(pl.col("unit_delta").alias("delta"))
        return market.select(*cols)


def frame(rows, cols=("ts", "asset", "price", "qty")):
    types = {
        "ts": pl.Int64, "asset": pl.String, "price": pl.Float64, "qty": pl.Float64,
        "volume": pl.Float64, "fee_bps": pl.Float64, "unit_delta": pl.Float64,
    }
    return pl.DataFrame(rows, schema={c: types.get(c, pl.Float64) for c in cols}, orient="row")


def run(prepared, config=None, cls=ColumnTrader):
    cfg = {"cost_bps": 0, "return_mode": "diff", **(config or {})}
    return simulate(cls(), prepared, None, cfg)


def book(prepared, config=None):
    cfg = {"cost_bps": 0, "return_mode": "diff", **(config or {})}
    return to_book(run(prepared, cfg), cfg)


class TestValidatePrepared:
    def test_accepts_and_sorts(self):
        out = validate_prepared(frame([(1, "B", 1.0, 0.0), (0, "A", 1.0, 0.0)]))
        assert out["ts"].to_list() == [0, 1]
        assert out["asset"].to_list() == ["A", "B"]

    def test_keeps_feature_columns(self):
        out = validate_prepared(frame([(0, "A", 1.0, 0.0)]).with_columns(pl.lit(2.0).alias("z")))
        assert "z" in out.columns
        assert list(PREPARED_COLUMNS) == ["ts", "asset", "price"]

    def test_rejects_missing_column(self):
        with pytest.raises(PanelError, match="price"):
            validate_prepared(frame([(0, "A", 1.0, 0.0)]).drop("price"))

    def test_rejects_nulls(self):
        bad = frame([(0, "A", 1.0, 0.0)]).with_columns(pl.lit(None, dtype=pl.Float64).alias("price"))
        with pytest.raises(PanelError, match="null"):
            validate_prepared(bad)

    def test_rejects_duplicate_keys(self):
        with pytest.raises(PanelError, match="duplicate"):
            validate_prepared(pl.concat([frame([(0, "A", 1.0, 0.0)])] * 2))

    def test_rejects_negative_volume(self):
        with pytest.raises(PanelError, match="volume"):
            validate_prepared(frame([(0, "A", 1.0, 0.0, -1.0)],
                                    ("ts", "asset", "price", "qty", "volume")))

    def test_rejects_negative_cost_column(self):
        with pytest.raises(PanelError, match="fee_bps"):
            validate_prepared(frame([(0, "A", 1.0, 0.0, -1.0)],
                                    ("ts", "asset", "price", "qty", "fee_bps")))

    def test_accepts_pandas(self):
        out = validate_prepared(frame([(0, "A", 1.0, 0.0)]).to_pandas())
        assert isinstance(out, pl.DataFrame)


class TestValidateOrders:
    def test_none_is_no_trade(self):
        out = validate_orders(None, known={"A"})
        assert out.height == 0

    def test_rejects_unknown_asset(self):
        with pytest.raises(PanelError, match="unknown"):
            validate_orders(pl.DataFrame({"asset": ["B"], "qty": [1.0]}), known={"A"})

    def test_rejects_duplicate_asset(self):
        with pytest.raises(PanelError, match="duplicate"):
            validate_orders(
                pl.DataFrame({"asset": ["A", "A"], "qty": [1.0, 2.0]}), known={"A"},
            )


class TestFills:
    def test_volume_caps_the_fill(self):
        tape = run(frame([
            (0, "A", 100.0, 10.0, 4.0),
            (1, "A", 110.0, 0.0, 100.0),
        ], ("ts", "asset", "price", "qty", "volume")))
        assert tape.filter(pl.col("ts") == 0)["fill"].to_list() == [4.0]
        assert tape.filter(pl.col("ts") == 0)["inventory"].to_list() == [4.0]
        assert tape.filter(pl.col("ts") == 1)["inventory"].to_list() == [4.0]

    def test_missing_volume_is_a_full_fill(self):
        tape = run(frame([(0, "A", 100.0, 7.0)]))
        assert tape["fill"].to_list() == [7.0]
        assert tape["inventory"].to_list() == [7.0]

    def test_unfilled_remainder_is_dropped(self):
        tape = run(frame([
            (0, "A", 100.0, 10.0, 3.0),
            (1, "A", 100.0, 0.0, 100.0),
        ], ("ts", "asset", "price", "qty", "volume")))
        assert tape.filter(pl.col("ts") == 1)["inventory"].to_list() == [3.0]

    def test_same_tick_fill_earns_nothing(self):
        tape = run(frame([(0, "A", 100.0, 1.0), (1, "A", 110.0, 0.0)]))
        assert tape.filter(pl.col("ts") == 0)["pnl"].to_list() == [0.0]
        assert tape.filter(pl.col("ts") == 1)["pnl"].to_list() == [10.0]

    def test_costs_charge_the_fill_not_the_order(self):
        tape = run(frame([
            (0, "A", 100.0, 10.0, 4.0),
        ], ("ts", "asset", "price", "qty", "volume")),
            {"fee_per_unit": 1.0, "cost_bps": 0, "return_mode": "diff"})
        assert tape["fee"].to_list() == [4.0]


class TestBook:
    def test_two_assets_mark_independently(self):
        tape = run(frame([
            (0, "A", 100.0, 1.0),
            (0, "B", 50.0, -1.0),
            (1, "A", 110.0, 0.0),
            (1, "B", 40.0, 0.0),
        ]))
        bt = to_book(tape, {"cost_bps": 0, "return_mode": "diff"})
        # t0: no prior inv. t1: +10 on A, +10 on B (short the drop).
        assert bt.sort("ts")["pnl"].to_list() == pytest.approx([0.0, 20.0])

    def test_pct_refuses_non_positive_prices(self):
        with pytest.raises(ValueError, match="pct"):
            run(frame([(0, "A", 0.0, 1.0)]), {"return_mode": "pct", "cost_bps": 0})

    def test_diff_accepts_a_negative_price(self):
        tape = run(frame([(0, "A", -5.0, 1.0), (1, "A", 5.0, 0.0)]))
        assert tape.filter(pl.col("ts") == 1)["pnl"].to_list() == [10.0]


class TestCosts:
    def test_fee_bps_falls_back_to_cost_bps(self):
        prepared = frame([(0, "A", 100.0, 1.0), (1, "A", 100.0, 0.0)])
        a = run(prepared, {"cost_bps": 50, "return_mode": "diff"})["fee"].sum()
        b = run(prepared, {"fee_bps": 50, "cost_bps": 0, "return_mode": "diff"})["fee"].sum()
        assert a == pytest.approx(b)

    def test_fee_and_slippage_add(self):
        prepared = frame([(0, "A", 100.0, 1.0)])
        split = run(prepared, {"fee_bps": 30, "slippage_bps": 20, "return_mode": "diff"})
        together = run(prepared, {"cost_bps": 50, "return_mode": "diff"})
        assert (split["fee"] + split["slippage"]).sum() == pytest.approx(
            (together["fee"] + together["slippage"]).sum()
        )

    def test_panel_fee_beats_config(self):
        cheap = frame([(0, "A", 100.0, 1.0, 0.0)], ("ts", "asset", "price", "qty", "fee_bps"))
        dear = frame([(0, "A", 100.0, 1.0, 100.0)], ("ts", "asset", "price", "qty", "fee_bps"))
        cfg = {"fee_bps": 50, "return_mode": "diff", "cost_bps": 0}
        assert run(cheap, cfg)["fee"].sum() < run(dear, cfg)["fee"].sum()

    def test_rejects_negative_config_fee(self):
        with pytest.raises(ValueError, match="fee_bps"):
            run(frame([(0, "A", 100.0, 1.0)]), {"fee_bps": -5, "return_mode": "diff"})

    def test_zero_fill_pays_no_per_unit_fee(self):
        tape = run(frame([(0, "A", 100.0, 0.0)]),
                   {"fee_per_unit": 100.0, "cost_bps": 0, "return_mode": "diff"})
        assert tape["fee"].sum() == 0.0

    def test_diff_scales_bps_by_price(self):
        cheap = frame([(0, "A", 10.0, 1.0)])
        dear = frame([(0, "A", 100.0, 1.0)])
        cfg = {"fee_bps": 100, "return_mode": "diff", "cost_bps": 0}
        assert run(cheap, cfg)["fee"].sum() < run(dear, cfg)["fee"].sum()


class TestEvaluate:
    def test_empty_tape(self):
        metrics = evaluate(run(frame([])), {"return_mode": "diff"})
        assert metrics["n_obs"] == 0
        assert metrics["sharpe"] == 0.0
        assert metrics["inventory"] == {}

    def test_terminal_inventory(self):
        tape = run(frame([(0, "A", 100.0, 2.0), (1, "A", 101.0, 1.0)]))
        metrics = evaluate(tape, {"cost_bps": 0, "return_mode": "diff"})
        assert metrics["inventory"] == {"A": 3.0}

    def test_greeks_use_strategy_delta(self):
        tape = run(frame([
            (0, "A", 100.0, 2.0, 0.5),
            (0, "B", 100.0, -1.0, 1.0),
            (1, "A", 100.0, 0.0, 0.5),
            (1, "B", 100.0, 0.0, 1.0),
        ], ("ts", "asset", "price", "qty", "unit_delta")))
        greeks = evaluate(tape, {"cost_bps": 0, "return_mode": "diff"})["greeks"]
        assert greeks["delta"] == pytest.approx(0.0)
        assert greeks["gross"] == pytest.approx(3.0)
        assert greeks["time_in_market"] == pytest.approx(1.0)

    def test_a_flat_book_has_zero_greeks(self):
        tape = run(frame([(0, "A", 100.0, 0.0), (1, "A", 110.0, 0.0)]))
        greeks = evaluate(tape, {"cost_bps": 0, "return_mode": "diff"})["greeks"]
        assert greeks["delta"] == 0.0
        assert greeks["gross"] == 0.0
        assert greeks["time_in_market"] == 0.0

    def test_n_trades_counts_rebalances(self):
        tape = run(frame([(0, "A", 100.0, 1.0), (1, "A", 101.0, 0.0), (2, "A", 102.0, 0.0)]))
        assert evaluate(tape, {"cost_bps": 0, "return_mode": "diff"})["n_trades"] == 1

    def test_pct_wipeout_is_capped(self):
        tape = run(frame([
            (0, "A", 100.0, 2.0),
            (1, "A", 10.0, 0.0),
        ]), {"return_mode": "pct", "cost_bps": 0})
        metrics = evaluate(tape, {"return_mode": "pct", "cost_bps": 0})
        assert metrics["total_return"] == -1.0
        assert metrics["ruined_at"] is not None

    def test_costs_reduce_net(self):
        prepared = frame([(0, "A", 100.0, 1.0), (1, "A", 110.0, 0.0)])
        free = evaluate(run(prepared, {"cost_bps": 0, "return_mode": "diff"}),
                        {"cost_bps": 0, "return_mode": "diff"})
        charged = evaluate(run(prepared, {"fee_per_unit": 1.0, "cost_bps": 0, "return_mode": "diff"}),
                           {"cost_bps": 0, "return_mode": "diff"})
        assert charged["total_return"] < free["total_return"]


class MutatingTrader(ColumnTrader):
    def on_tick(self, ts, market, inventory, model, config):
        inventory["A"] = 99.0
        return super().on_tick(ts, market, inventory, model, config)


class TestInventoryIsolation:
    def test_strategy_cannot_mutate_the_engine_book(self):
        tape = run(frame([(0, "A", 100.0, 1.0), (1, "A", 110.0, 0.0)]), cls=MutatingTrader)
        assert tape.filter(pl.col("ts") == 1)["inventory"].to_list() == [1.0]
