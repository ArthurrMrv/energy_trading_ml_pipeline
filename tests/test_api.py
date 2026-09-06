"""API surface tests, driven through FastAPI's TestClient."""

import time

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FIXTURES


@pytest.fixture()
def client(pipeline_home):
    from backend.pipeline.api import app

    with TestClient(app) as test_client:
        yield test_client


def upload(client, fixture_name="tiny_ok.py"):
    return client.post(
        "/strategies",
        files={"file": (fixture_name, (FIXTURES / fixture_name).read_bytes(),
                        "text/x-python")},
    )


def upload_source(client, name, source):
    return client.post(
        "/strategies", files={"file": (name, source.encode(), "text/x-python")}
    )


def wait_for_finish(client, run_id, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/runs/{run_id}").json()
        if body["status"] in ("completed", "paused"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


class TestUploadValidation:
    def test_accepts_a_valid_strategy(self, client):
        response = upload(client)
        assert response.status_code == 201
        assert response.json()["class_name"] == "TinyStrategy"

    def test_rejects_a_non_python_file(self, client):
        assert upload_source(client, "notes.txt", "hello").status_code == 400

    def test_rejects_invalid_python(self, client):
        response = upload_source(client, "bad.py", "def oops(:\n")
        assert response.status_code == 422
        assert "not valid Python" in response.json()["detail"]

    def test_rejects_a_file_with_no_strategy(self, client):
        response = upload_source(client, "empty.py", "x = 1\n")
        assert response.status_code == 422
        assert "no Strategy subclass" in response.json()["detail"]

    def test_rejects_a_strategy_missing_required_methods(self, client):
        source = (
            "from backend.pipeline.contract import Strategy\n"
            "class Half(Strategy):\n"
            "    def collect(self, config): return None\n"
        )
        response = upload_source(client, "half.py", source)
        assert response.status_code == 422
        assert "does not implement" in response.json()["detail"]

    def test_rejects_two_strategies_in_one_file(self, client):
        source = (
            "from backend.pipeline.contract import Strategy\n"
            "class A(Strategy):\n"
            "    def collect(self, c): ...\n"
            "    def prepare(self, r, c): ...\n"
            "    def on_tick(self, p, m, c): ...\n"
            "class B(Strategy):\n"
            "    def collect(self, c): ...\n"
            "    def prepare(self, r, c): ...\n"
            "    def on_tick(self, p, m, c): ...\n"
        )
        response = upload_source(client, "two.py", source)
        assert response.status_code == 422
        assert "exactly one" in response.json()["detail"]

    def test_validation_does_not_execute_the_upload(self, client, tmp_path):
        """A file is parsed, never imported. Module-level code must not run."""
        marker = tmp_path / "executed.txt"
        source = (
            "import pathlib\n"
            f"pathlib.Path({str(marker)!r}).write_text('ran')\n"
            "from backend.pipeline.contract import Strategy\n"
            "class Sneaky(Strategy):\n"
            "    def collect(self, c): ...\n"
            "    def prepare(self, r, c): ...\n"
            "    def on_tick(self, p, m, c): ...\n"
        )
        assert upload_source(client, "sneaky.py", source).status_code == 201
        assert not marker.exists()

    def test_source_is_readable_back(self, client):
        strategy_id = upload(client).json()["id"]
        body = client.get(f"/strategies/{strategy_id}").json()
        assert "class TinyStrategy" in body["source"]


class TestNotFound:
    def test_unknown_strategy(self, client):
        assert client.get("/strategies/nope").status_code == 404

    def test_unknown_run(self, client):
        assert client.get("/runs/nope").status_code == 404

    def test_run_against_unknown_strategy(self, client):
        response = client.post("/runs", json={"strategy_id": "nope", "config": {}})
        assert response.status_code == 404

    def test_resume_with_an_unknown_stage(self, client):
        strategy_id = upload(client).json()["id"]
        run_id = client.post(
            "/runs", json={"strategy_id": strategy_id, "config": {}}
        ).json()["run_id"]
        wait_for_finish(client, run_id)

        response = client.post(
            f"/runs/{run_id}/resume", json={"from_stage": "train"}
        )
        assert response.status_code == 400
        assert "unknown stage" in response.json()["detail"]


class TestRunLifecycle:
    @pytest.fixture()
    def finished(self, client):
        strategy_id = upload(client).json()["id"]
        run_id = client.post(
            "/runs", json={"strategy_id": strategy_id, "config": {"n": 12}}
        ).json()["run_id"]
        return run_id, wait_for_finish(client, run_id)

    def test_run_completes_through_the_api(self, finished):
        _, body = finished
        assert body["status"] == "completed"
        assert [s["status"] for s in body["steps"]] == ["completed"] * 5

    def test_steps_are_returned_in_pipeline_order(self, finished):
        _, body = finished
        assert [s["name"] for s in body["steps"]] == [
            "collect", "prepare", "fit", "simulate", "evaluate"
        ]

    def test_context_endpoint_serves_the_agent_hook(self, client, finished):
        run_id, _ = finished
        ctx = client.get(f"/runs/{run_id}/steps/prepare/context").json()
        assert ctx["status"] == "completed"
        assert ctx["inputs"]["collect"]["shape"] == [12, 3]
        assert "def prepare" in ctx["source"]

    def test_context_rejects_an_unknown_stage(self, client, finished):
        run_id, _ = finished
        assert client.get(f"/runs/{run_id}/steps/train/context").status_code == 400

    def test_artifacts_are_downloadable(self, client, finished):
        run_id, _ = finished
        response = client.get(f"/runs/{run_id}/artifacts/metrics.json")
        assert response.status_code == 200
        assert response.json()["n_obs"] == 12

    def test_unknown_artifact_is_404(self, client, finished):
        run_id, _ = finished
        assert client.get(f"/runs/{run_id}/artifacts/ghost.parquet").status_code == 404

    def test_artifact_path_cannot_escape_the_run_directory(self, client, finished):
        run_id, _ = finished
        response = client.get(f"/runs/{run_id}/artifacts/..%2Fpipeline.db")
        assert response.status_code == 404

    def test_runs_are_listed(self, client, finished):
        run_id, _ = finished
        assert run_id in [r["id"] for r in client.get("/runs").json()["runs"]]

    def test_events_are_readable_over_http(self, client, finished):
        run_id, _ = finished
        events = client.get(f"/runs/{run_id}/events").json()["events"]
        assert events[0]["event"] == "run_started"
        assert events[-1]["event"] == "run_completed"


class TestWebSocket:
    def test_stream_replays_history_for_a_late_subscriber(self, client):
        strategy_id = upload(client).json()["id"]
        run_id = client.post(
            "/runs", json={"strategy_id": strategy_id, "config": {"n": 5}}
        ).json()["run_id"]
        wait_for_finish(client, run_id)

        with client.websocket_connect(f"/runs/{run_id}/events") as socket:
            first = socket.receive_json()
            assert first["event"] == "run_started"
            events = [first]
            while events[-1]["event"] != "run_completed":
                events.append(socket.receive_json())
            assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)

    def test_stream_rejects_an_unknown_run(self, client):
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/runs/nope/events") as socket:
                socket.receive_json()


class TestShutdown:
    def test_a_parked_debug_stage_does_not_survive_shutdown(self, pipeline_home):
        """A debug stage waits for a client forever. If shutdown does not cancel
        it, the server cannot exit and the debug port stays held."""
        import socket

        from backend.pipeline import orchestrator
        from backend.pipeline.api import app

        with TestClient(app) as client:
            strategy_id = upload(client).json()["id"]
            run_id = client.post(
                "/runs", json={"strategy_id": strategy_id, "config": {}}
            ).json()["run_id"]
            wait_for_finish(client, run_id)

            client.post(
                f"/runs/{run_id}/resume",
                json={"from_stage": "collect", "debug": True},
            )
            port = None
            deadline = time.time() + 30
            while time.time() < deadline and port is None:
                ready = [e for e in pipeline_home.get_events(run_id)
                         if e["event"] == "debug_ready"]
                port = ready[0]["payload"]["port"] if ready else None
                time.sleep(0.05)
            assert port, "debug stage never opened a port"
            assert orchestrator._port_is_open(port)

        # The context manager exiting runs lifespan shutdown.
        deadline = time.time() + 15
        while time.time() < deadline and orchestrator._port_is_open(port):
            time.sleep(0.1)
        assert not orchestrator._port_is_open(port), "debug stage outlived the server"

    def test_shutdown_is_clean_with_no_active_runs(self, pipeline_home):
        from backend.pipeline.api import app

        with TestClient(app) as client:
            assert client.get("/runs").status_code == 200


class TestStrategyListing:
    def test_lists_uploaded_strategies_newest_first(self, client):
        first = upload(client, "tiny_ok.py").json()["id"]
        second = upload(client, "tiny_broken.py").json()["id"]

        strategies = client.get("/strategies").json()["strategies"]
        ids = [s["id"] for s in strategies]

        assert ids[:2] == [second, first]
        assert strategies[0]["filename"] == "tiny_broken.py"
        assert strategies[0]["class_name"] == "BrokenStrategy"

    def test_is_empty_before_any_upload(self, client):
        assert client.get("/strategies").json() == {"strategies": []}

    def test_honours_the_limit(self, client):
        for _ in range(3):
            upload(client)
        assert len(client.get("/strategies?limit=2").json()["strategies"]) == 2


class TestFrontend:
    """The UI is served by this app, so its mount is part of the API contract."""

    def test_serves_the_app_shell_at_root(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "js/app.js" in response.text

    def test_serves_the_module_tree(self, client):
        for asset in ("/js/app.js", "/js/api.js", "/js/render.js", "/js/chart.js",
                      "/js/esc.js", "/js/logs.js", "/js/assist.js", "/js/md.js",
                      "/style.css"):
            assert client.get(asset).status_code == 200, asset

    def test_static_mount_does_not_shadow_the_api(self, client):
        """A Mount('/') matches every path, so registration order is load-bearing:
        mounted too early it would serve index.html in place of every route."""
        listing = client.get("/runs")
        assert listing.status_code == 200
        assert listing.json() == {"runs": []}

        missing = client.get("/runs/nope")
        assert missing.status_code == 404
        assert missing.json()["detail"] == "unknown run nope"
