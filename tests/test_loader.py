"""Loader and runner units, exercised in-process.

The integration tests drive these through subprocesses, where coverage cannot
follow them. These cover the same code directly.
"""

import pathlib
import pickle

import polars as pl
import pytest

from backend.pipeline.contract import PanelError, Strategy, to_polars
from backend.pipeline.loader import (
    StrategyLoadError, find_strategy_class, load_module, load_strategy,
    method_source, validate_source,
)
from backend.pipeline.runner import load_artifact, main, save_artifact
from tests.conftest import FIXTURES

VALID = (
    "from backend.pipeline.contract import Strategy\n"
    "class Ok(Strategy):\n"
    "    def collect(self, config): return None\n"
    "    def prepare(self, raw, config): return raw\n"
    "    def on_tick(self, ts, market, inventory, model, config): return market\n"
)


class TestLoadStrategy:
    def test_loads_a_fixture(self):
        strategy = load_strategy(FIXTURES / "tiny_ok.py")
        assert isinstance(strategy, Strategy)
        assert type(strategy).__name__ == "TinyStrategy"

    def test_missing_file_is_reported_clearly(self, tmp_path):
        with pytest.raises(StrategyLoadError, match="not found"):
            load_strategy(tmp_path / "absent.py")

    def test_import_error_is_wrapped(self, tmp_path):
        path = tmp_path / "boom.py"
        path.write_text("raise RuntimeError('module-level explosion')\n")
        with pytest.raises(StrategyLoadError, match="failed to import"):
            load_strategy(path)

    def test_file_without_a_strategy_is_rejected(self, tmp_path):
        path = tmp_path / "plain.py"
        path.write_text("x = 1\n")
        with pytest.raises(StrategyLoadError, match="no Strategy subclass"):
            find_strategy_class(load_module(path))

    def test_abstract_subclass_names_what_is_missing(self, tmp_path):
        path = tmp_path / "partial.py"
        path.write_text(
            "from backend.pipeline.contract import Strategy\n"
            "class Partial(Strategy):\n"
            "    def collect(self, config): return None\n"
        )
        with pytest.raises(StrategyLoadError, match="prepare"):
            find_strategy_class(load_module(path))

    def test_two_subclasses_are_rejected(self, tmp_path):
        path = tmp_path / "double.py"
        path.write_text(VALID + VALID.split("\n", 1)[1].replace("Ok", "Also"))
        with pytest.raises(StrategyLoadError, match="exactly one"):
            find_strategy_class(load_module(path))

    def test_imported_strategies_are_not_mistaken_for_the_upload(self, tmp_path):
        """Importing Strategy itself must not count as defining one."""
        path = tmp_path / "importer.py"
        path.write_text(
            "from backend.pipeline.contract import Strategy\n"
            "from tests.fixtures.tiny_ok import TinyStrategy\n" + VALID.split("\n", 1)[1]
        )
        assert find_strategy_class(load_module(path)).__name__ == "Ok"


class TestStaticValidation:
    def test_accepts_valid_source(self):
        assert validate_source(VALID) == "Ok"

    def test_reports_the_syntax_error(self):
        with pytest.raises(StrategyLoadError, match="not valid Python"):
            validate_source("def broken(:\n")

    def test_recognizes_a_qualified_base_class(self):
        source = (
            "from backend.pipeline import contract\n"
            "class Q(contract.Strategy):\n"
            "    def collect(self, c): ...\n"
            "    def prepare(self, r, c): ...\n"
            "    def on_tick(self, ts, m, i, mo, c): ...\n"
        )
        assert validate_source(source) == "Q"

    def test_fit_is_optional(self):
        assert validate_source(VALID) == "Ok"

    def test_method_source_extracts_one_method(self):
        assert "def prepare" in method_source(VALID, "prepare")

    def test_method_source_returns_none_for_an_absent_method(self):
        assert method_source(VALID, "fit") is None

    def test_method_source_survives_unparseable_input(self):
        assert method_source("def broken(:\n", "prepare") is None


class _ToyModel:
    def predict(self, x):
        return [1.0]


class TestArtifacts:
    def test_dataframe_round_trips_through_parquet(self, tmp_path):
        frame = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path, summary = save_artifact(frame, tmp_path, "thing")

        assert summary == {"type": "dataframe", "rows": 2, "cols": 2,
                           "columns": ["a", "b"]}
        assert load_artifact(path).equals(frame)

    def test_dict_round_trips_through_json(self, tmp_path):
        path, summary = save_artifact({"sharpe": 1.5}, tmp_path, "metrics")
        assert summary["type"] == "dict"
        assert load_artifact(path) == {"sharpe": 1.5}

    def test_none_round_trips(self, tmp_path):
        path, summary = save_artifact(None, tmp_path, "model")
        assert summary == {"type": "none"}
        assert load_artifact(path) is None

    def test_arbitrary_object_falls_back_to_pickle(self, tmp_path):
        path, summary = save_artifact({1, 2, 3}, tmp_path, "model")
        assert summary["type"] == "set"
        assert load_artifact(path) == {1, 2, 3}

    def test_pandas_is_accepted_at_the_boundary(self, tmp_path):
        pandas_frame = pl.DataFrame({"a": [1.0, 2.0]}).to_pandas()
        path, summary = save_artifact(pandas_frame, tmp_path, "prepared")
        assert summary["type"] == "dataframe"
        assert load_artifact(path)["a"].to_list() == [1.0, 2.0]

    def test_json_uses_a_readable_fallback_for_odd_values(self, tmp_path):
        import datetime

        path, summary = save_artifact({"when": datetime.date(2026, 1, 1)}, tmp_path, "m")
        assert path.endswith(".json")
        assert load_artifact(path)["when"] == "2026-01-01"

    def test_a_dict_holding_a_model_is_pickled_not_stringified(self, tmp_path):
        """Regression: default=str used to turn estimators into dead strings."""
        path, summary = save_artifact(
            {"model": _ToyModel(), "features": ["a"]}, tmp_path, "model"
        )
        assert path.endswith(".pkl")
        assert summary.get("format") == "pickle"
        loaded = load_artifact(path)
        assert hasattr(loaded["model"], "predict")
        assert loaded["features"] == ["a"]

    def test_pickle_artifact_is_a_real_pickle(self, tmp_path):
        path, _ = save_artifact([1, 2], tmp_path, "model")
        assert pickle.loads(pathlib.Path(path).read_bytes()) == [1, 2]


class TestToPolars:
    def test_passes_polars_through(self):
        frame = pl.DataFrame({"a": [1]})
        assert to_polars(frame) is frame

    def test_collects_a_lazyframe(self):
        assert to_polars(pl.LazyFrame({"a": [1]})).height == 1

    def test_rejects_something_that_is_not_a_frame(self):
        with pytest.raises(PanelError, match="expected a polars or pandas"):
            to_polars(42, stage="collect")


class TestRunnerCli:
    def test_wrong_argument_count_returns_usage_code(self):
        assert main(["only-one"]) == 2

    def test_unknown_run_exits_nonzero(self, pipeline_home):
        assert main(["no-such-run", "collect"]) == 1

    def test_unknown_stage_exits_nonzero(self, pipeline_home, make_run):
        assert main([make_run("tiny_ok.py"), "train"]) == 1
