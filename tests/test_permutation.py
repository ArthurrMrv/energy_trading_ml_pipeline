"""Monte Carlo permutation tests.

The null is the same prices and liquidity with the information set scrambled:
feature columns move, ``on_tick`` is re-run, inventory is rebuilt from fills.
"""

import json
import random

import polars as pl
import pytest

from backend.pipeline.engine import simulate, to_book
from backend.pipeline.permutation import (
    CURVE_POINTS,
    MAX_CURVES,
    permutation_test,
    permute,
    settings,
)


class FollowFeat:
    def on_tick(self, ts, market, inventory, model, config):
        assets = market["asset"].to_list()
        feat = market["feat"].to_list()
        qty = [float(f) - inventory.get(a, 0.0) for a, f in zip(assets, feat)]
        return pl.DataFrame({"asset": assets, "qty": qty})


def walk(n=120, seed=0):
    rng = random.Random(seed)
    price, prices = 100.0, []
    for _ in range(n):
        price += rng.gauss(0.0, 1.0)
        prices.append(price)
    return prices


def make_prepared(feat, prices):
    return pl.DataFrame({
        "ts": list(range(len(prices))),
        "asset": ["A"] * len(prices),
        "price": prices,
        "feat": [float(s) for s in feat],
    })


def prescient(n=120, seed=0):
    prices = walk(n, seed)
    moves = [prices[i + 1] - prices[i] for i in range(n - 1)] + [0.0]
    return make_prepared([1.0 if m > 0 else -1.0 for m in moves], prices)


def noise(n=120, seed=0):
    prices = walk(n, seed)
    rng = random.Random(seed + 1000)
    return make_prepared([rng.choice([-1.0, 0.0, 1.0]) for _ in range(n)], prices)


CONFIG = {"cost_bps": 0, "return_mode": "diff"}
STRAT = FollowFeat()


def with_test(n=50, **kwargs):
    return {**CONFIG, "permutation": {"n": n, "seed": 7, **kwargs}}


class TestSettings:
    def test_is_off_unless_asked_for(self):
        assert settings({})["n"] == 0
        assert permutation_test(STRAT, prescient(), None, CONFIG) is None

    def test_zero_permutations_is_off(self):
        assert permutation_test(STRAT, prescient(), None, with_test(0)) is None

    def test_rejects_an_unknown_method(self):
        with pytest.raises(ValueError, match="method"):
            permutation_test(STRAT, prescient(), None, with_test(method="bootstrap"))

    def test_rejects_an_unknown_metric(self):
        with pytest.raises(ValueError, match="metric"):
            permutation_test(STRAT, prescient(), None, with_test(metric="alpha"))

    def test_rejects_a_negative_count(self):
        with pytest.raises(ValueError, match="n"):
            permutation_test(STRAT, prescient(), None, with_test(-1))

    def test_rejects_a_non_integer_count(self):
        with pytest.raises(ValueError, match="n"):
            permutation_test(STRAT, prescient(), None, with_test(10.0))

    def test_rejects_a_non_integer_seed(self):
        with pytest.raises(ValueError, match="seed"):
            permutation_test(STRAT, prescient(), None, with_test(10, seed="lucky"))

    def test_rejects_a_permutation_block_that_is_not_an_object(self):
        with pytest.raises(ValueError, match="object"):
            settings({"permutation": 50})

    def test_rejects_an_unknown_option(self):
        with pytest.raises(ValueError, match="permutations"):
            settings({"permutation": {"n": 10, "permutations": 99}})


class TestPermute:
    def test_leaves_the_price_path_untouched(self):
        original = prescient()
        moved = permute(original, method="rotate", seed=3)
        assert moved["price"].to_list() == original["price"].to_list()
        assert moved["ts"].to_list() == original["ts"].to_list()
        assert moved["asset"].to_list() == original["asset"].to_list()

    def test_preserves_the_multiset_of_features(self):
        original = prescient()
        for method in ("rotate", "shuffle"):
            moved = permute(original, method=method, seed=5)
            assert sorted(moved["feat"].to_list()) == sorted(original["feat"].to_list())

    def test_actually_moves_the_feature(self):
        original = prescient()
        moved = permute(original, method="rotate", seed=3)
        assert moved["feat"].to_list() != original["feat"].to_list()

    def test_does_not_mutate_its_input(self):
        original = prescient()
        before = original["feat"].to_list()
        permute(original, method="shuffle", seed=1)
        assert original["feat"].to_list() == before

    def test_rejects_an_unknown_method(self):
        with pytest.raises(ValueError, match="method"):
            permute(prescient(), method="bootstrap", seed=0)

    def test_needs_at_least_two_timestamps(self):
        one = make_prepared([1.0], [100.0])
        with pytest.raises(ValueError, match="timestamp"):
            permute(one, method="rotate", seed=0)

    def test_rejects_a_frame_with_no_features(self):
        bare = pl.DataFrame({"ts": [0, 1], "asset": ["A", "A"], "price": [1.0, 2.0]})
        with pytest.raises(ValueError, match="feature"):
            permute(bare, method="rotate", seed=0)

    def test_rotation_preserves_turnover(self):
        prices = walk(120, seed=4)
        slow = [1.0 if (t // 20) % 2 == 0 else -1.0 for t in range(120)]
        original = make_prepared(slow, prices)

        def turnover(frame):
            tape = simulate(FollowFeat(), frame, None, CONFIG)
            return to_book(tape, CONFIG)["turnover"].sum()

        base = turnover(original)
        rotated = turnover(permute(original, method="rotate", seed=3))
        shuffled = turnover(permute(original, method="shuffle", seed=3))

        assert rotated == pytest.approx(base, abs=2.0)
        assert shuffled > 5 * base


class TestPermutationTest:
    def test_a_prescient_signal_beats_every_permutation(self):
        result = permutation_test(STRAT, prescient(), None, with_test(50))
        assert result["beat_by_chance"] == 0
        assert result["p_value"] == pytest.approx(1 / 51, abs=1e-6)
        assert result["observed"] > result["null_max"]

    def test_p_value_is_never_zero(self):
        result = permutation_test(STRAT, prescient(), None, with_test(20))
        assert result["p_value"] > 0

    def test_a_random_signal_is_not_significant(self):
        result = permutation_test(STRAT, noise(), None, with_test(50))
        assert result["p_value"] > 0.05
        assert result["null_min"] <= result["observed"] <= result["null_max"]

    def test_is_deterministic_given_a_seed(self):
        panel = noise()
        first = permutation_test(STRAT, panel, None, with_test(20))
        second = permutation_test(STRAT, panel, None, with_test(20))
        assert first == second

    def test_a_different_seed_draws_a_different_null(self):
        panel = noise()
        first = permutation_test(STRAT, panel, None, with_test(20, seed=1))
        second = permutation_test(STRAT, panel, None, with_test(20, seed=2))
        assert first["null_mean"] != second["null_mean"]

    def test_reports_what_it_did(self):
        result = permutation_test(STRAT, prescient(), None, with_test(10, method="shuffle"))
        assert result["method"] == "shuffle"
        assert result["metric"] == "sharpe"
        assert result["n"] == 10
        assert result["seed"] == 7

    def test_can_test_another_metric(self):
        result = permutation_test(STRAT, prescient(), None, with_test(10, metric="total_return"))
        assert result["metric"] == "total_return"
        assert result["observed"] > result["null_max"]

    def test_result_is_json_serializable(self):
        json.dumps(permutation_test(STRAT, prescient(), None, with_test(10)))

    def test_reports_progress(self):
        seen = []
        permutation_test(STRAT, prescient(), None, with_test(10),
                         on_progress=lambda done, total: seen.append((done, total)))
        assert seen[-1] == (10, 10)

    def test_costs_are_charged_to_the_null_too(self):
        panel = prescient()
        free = permutation_test(STRAT, panel, None, with_test(20))
        costly = permutation_test(
            STRAT, panel, None, {**with_test(20), "fee_bps": 200, "slippage_bps": 100}
        )
        assert costly["null_mean"] < free["null_mean"]


class TestCurves:
    def test_carries_the_strategy_and_the_null_paths(self):
        result = permutation_test(STRAT, prescient(), None, with_test(20))
        curves = result["curves"]
        assert len(curves["null"]) == 20
        assert len(curves["strategy"]) == len(curves["x"])

    def test_the_strategy_curve_is_the_real_equity(self):
        panel = prescient()
        tape = simulate(FollowFeat(), panel, None, CONFIG)
        equity = to_book(tape, CONFIG).sort("ts")["equity"]
        curve = permutation_test(STRAT, panel, None, with_test(5))["curves"]["strategy"]
        assert curve[0] == pytest.approx(equity[0], abs=1e-6)
        assert curve[-1] == pytest.approx(equity[-1], abs=1e-6)

    def test_long_curves_are_thinned_for_the_plot(self):
        panel = prescient(n=1000)
        curves = permutation_test(STRAT, panel, None, with_test(3))["curves"]
        assert len(curves["strategy"]) == CURVE_POINTS
        assert all(len(c) == CURVE_POINTS for c in curves["null"])

    def test_short_curves_are_left_alone(self):
        curves = permutation_test(STRAT, prescient(n=40), None, with_test(3))["curves"]
        assert len(curves["strategy"]) == 40

    def test_the_number_of_plotted_paths_is_capped(self):
        result = permutation_test(STRAT, noise(), None, with_test(MAX_CURVES + 20))
        assert result["n"] == MAX_CURVES + 20
        assert len(result["curves"]["null"]) == MAX_CURVES

    def test_x_is_the_period_index(self):
        curves = permutation_test(STRAT, prescient(n=40), None, with_test(3))["curves"]
        assert curves["x"][0] == 0
        assert curves["x"][-1] == 39

    def test_curves_are_json_serializable(self):
        json.dumps(permutation_test(STRAT, prescient(), None, with_test(5))["curves"])
