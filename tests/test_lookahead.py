"""Lookahead guarantees for the engine, stated as exact identities.

Previous inventory earns this tick's price change; this tick's fill starts
earning next tick. An off-by-one produces a plausible equity curve that is
simply false. Nothing here is statistical.
"""

import polars as pl
import pytest

from backend.pipeline.engine import simulate, to_book

FREE = {"return_mode": "diff", "cost_bps": 0}


class TargetTrader:
    """``qty`` is a desired inventory, not a trade."""

    def on_tick(self, ts, market, inventory, model, config):
        assets = market["asset"].to_list()
        target = market["qty"].to_list()
        return pl.DataFrame({
            "asset": assets,
            "qty": [float(t) - inventory.get(a, 0.0) for a, t in zip(assets, target)],
        })


def walk(n=400, seed=7):
    price, prices = 100.0, []
    state = seed
    for _ in range(n):
        state = (1103515245 * state + 12345) % 2147483648
        price += (state / 2147483648) - 0.5
        prices.append(price)
    return prices


def prepared(prices, qtys, asset="A"):
    return pl.DataFrame({
        "ts": list(range(len(prices))),
        "asset": [asset] * len(prices),
        "price": [float(p) for p in prices],
        "qty": [float(q) for q in qtys],
    })


def returns(prices):
    return [0.0] + [prices[i] - prices[i - 1] for i in range(1, len(prices))]


def pnl(prices, qtys, **overrides):
    tape = simulate(TargetTrader(), prepared(prices, qtys), None, {**FREE, **overrides})
    return to_book(tape, {**FREE, **overrides})["pnl"].to_list()


class TestFillDoesNotEarnThisTick:
    def test_a_fill_on_the_open_prints_zero(self):
        prices = walk(50)
        qtys = [1.0] + [0.0] * 49
        assert pnl(prices, qtys)[0] == pytest.approx(0.0)

    def test_that_fill_is_paid_on_the_next_tick(self):
        prices = walk(50)
        qtys = [1.0] + [0.0] * 49
        # Flattens at t1 after earning prices[1] - prices[0].
        assert pnl(prices, qtys)[1] == pytest.approx(prices[1] - prices[0])

    def test_a_position_opened_on_the_final_bar_cannot_be_paid(self):
        prices = walk(50)
        qtys = [0.0] * 49 + [1.0]
        assert sum(pnl(prices, qtys)) == pytest.approx(0.0)


class TestKnownReturnEarnsAutocorrNotTheSquare:
    """Ordering this bar's already-printed return cannot earn ret^2.

    Fill at t, earn ret_{t+1} * fill_t. If fill_t = ret_t, that is
    autocorrelation, near zero. Earning ret_t * ret_t would be the off-by-one.
    """

    def test_trading_the_printed_return_is_not_the_sum_of_squares(self):
        prices = walk()
        rets = returns(prices)
        total = sum(pnl(prices, rets))
        cheating = sum(r * r for r in rets)
        assert abs(total) < 0.1 * cheating

    def test_trading_the_next_return_is_exactly_the_sum_of_squares(self):
        prices = walk()
        rets = returns(prices)
        foresight = rets[1:] + [0.0]
        total = sum(pnl(prices, foresight))
        # Hold ret_{t+1} into t+1: earn ret_{t+1}^2. rets[0] is 0.
        assert total == pytest.approx(sum(r * r for r in rets))


class TestPerAsset:
    def test_one_asset_does_not_bleed_into_the_next(self):
        a, b = walk(80, seed=7), [p * 50 for p in walk(80, seed=99)]
        qa, qb = returns(a), returns(b)
        together = pl.concat([prepared(a, qa, "A"), prepared(b, qb, "B")])
        tape = simulate(TargetTrader(), together, None, FREE)
        book = to_book(tape, FREE)
        only_a = sum(pnl(a, qa))
        only_b = sum(pnl(b, qb))
        assert book["pnl"].sum() == pytest.approx(only_a + only_b)
