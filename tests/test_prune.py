"""Deleting a run's files deletes the run.

The run directory is the record. SQLite only indexes it, so a row whose files
are gone is a ghost -- and a ghost in the run list is worse than no list at all,
because clicking it fails in a way that looks like a bug in the pipeline.
"""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FIXTURES


@pytest.fixture()
def client(pipeline_home):
    from backend.pipeline.api import app

    with TestClient(app) as test_client:
        yield test_client


def upload(client, name="tiny_ok.py"):
    return client.post(
        "/strategies",
        files={"file": (name, (FIXTURES / name).read_bytes(), "text/x-python")},
    ).json()


def start(client, strategy_id):
    return client.post(
        "/runs", json={"strategy_id": strategy_id, "config": {"n": 5}}
    ).json()["run_id"]


def wait_for_events(store, run_id, timeout=30.0):
    """The run starts asynchronously; give it something to lose."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if events := store.get_events(run_id):
            return events
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} emitted nothing within {timeout}s")


def purge_run(store, run_id):
    import shutil

    shutil.rmtree(store.run_path(run_id))


class TestPurgedRuns:
    def test_a_purged_run_leaves_the_listing(self, client, pipeline_home):
        run_id = start(client, upload(client)["id"])
        assert run_id in [r["id"] for r in client.get("/runs").json()["runs"]]

        purge_run(pipeline_home, run_id)
        assert run_id not in [r["id"] for r in client.get("/runs").json()["runs"]]

    def test_a_purged_run_is_gone_not_broken(self, client, pipeline_home):
        """404, not a row that fails when opened."""
        run_id = start(client, upload(client)["id"])
        purge_run(pipeline_home, run_id)
        assert client.get(f"/runs/{run_id}").status_code == 404

    def test_purging_one_run_spares_the_others(self, client, pipeline_home):
        strategy_id = upload(client)["id"]
        doomed, kept = start(client, strategy_id), start(client, strategy_id)

        purge_run(pipeline_home, doomed)
        listed = [r["id"] for r in client.get("/runs").json()["runs"]]
        assert kept in listed and doomed not in listed

    def test_the_steps_and_events_go_too(self, client, pipeline_home):
        """Otherwise every purge leaks rows that nothing will ever read again."""
        run_id = start(client, upload(client)["id"])
        wait_for_events(pipeline_home, run_id)

        purge_run(pipeline_home, run_id)
        client.get("/runs")

        assert pipeline_home.get_steps(run_id) == []
        assert pipeline_home.get_events(run_id) == []

    def test_listing_does_not_resurrect_the_directory(self, client, pipeline_home):
        """``run_dir`` creates on read. If the listing used it, every purged run
        would come back as an empty folder."""
        run_id = start(client, upload(client)["id"])
        purge_run(pipeline_home, run_id)

        client.get("/runs")
        assert not pipeline_home.run_path(run_id).exists()


class TestPurgedStrategies:
    def test_a_deleted_strategy_file_leaves_the_listing(self, client, pipeline_home):
        strategy = upload(client)
        assert client.get("/strategies").json()["strategies"]

        Path(strategy["path"]).unlink()
        assert client.get("/strategies").json() == {"strategies": []}

    def test_a_deleted_strategy_is_gone_not_broken(self, client, pipeline_home):
        strategy = upload(client)
        Path(strategy["path"]).unlink()
        assert client.get(f"/strategies/{strategy['id']}").status_code == 404

    def test_purging_strategies_spares_their_finished_runs(self, client, pipeline_home):
        """The run kept its own checkpoints; it is still a result worth reading."""
        strategy = upload(client)
        run_id = start(client, strategy["id"])

        Path(strategy["path"]).unlink()
        client.get("/strategies")

        assert run_id in [r["id"] for r in client.get("/runs").json()["runs"]]


class TestPrune:
    def test_reports_what_it_removed(self, client, pipeline_home):
        strategy = upload(client)
        run_id = start(client, strategy["id"])

        purge_run(pipeline_home, run_id)
        Path(strategy["path"]).unlink()

        removed = pipeline_home.prune()
        assert removed == {"runs": [run_id], "strategies": [strategy["id"]]}

    def test_is_a_no_op_when_nothing_was_deleted(self, client, pipeline_home):
        start(client, upload(client)["id"])
        assert pipeline_home.prune() == {"runs": [], "strategies": []}
