import importlib
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def pipeline_home(tmp_path, monkeypatch):
    """Point the store at a throwaway directory, for this test only."""
    monkeypatch.setenv("PIPELINE_HOME", str(tmp_path))
    from backend.pipeline import store

    importlib.reload(store)
    store.init_db()

    # Modules that captured store attributes at import time need re-binding.
    from backend.pipeline import context, orchestrator, runner
    for module in (runner, orchestrator, context):
        importlib.reload(module)

    yield store


@pytest.fixture()
def make_run(pipeline_home):
    from backend.pipeline.contract import STAGES

    def _make(fixture_name: str, config: dict | None = None) -> str:
        source = (FIXTURES / fixture_name).read_text()
        strategy = pipeline_home.create_strategy(fixture_name, source)
        return pipeline_home.create_run(
            strategy["id"], strategy["path"], config or {}, STAGES
        )

    return _make
