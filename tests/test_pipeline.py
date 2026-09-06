"""End-to-end tests: real subprocesses, real checkpoints, real SQLite."""

import asyncio
import contextlib
import json
from pathlib import Path

import pytest

from tests.conftest import FIXTURES


def run(coro):
    return asyncio.run(coro)


def steps_by_name(store, run_id):
    return {s["name"]: s for s in store.get_steps(run_id)}


class TestHappyPath:
    def test_full_run_completes_every_stage(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator
        from backend.pipeline.contract import STAGES

        run_id = make_run("tiny_ok.py")
        assert run(orchestrator.run_pipeline(run_id)) == "completed"

        steps = steps_by_name(pipeline_home, run_id)
        assert [steps[s]["status"] for s in STAGES] == ["completed"] * len(STAGES)

    def test_every_stage_leaves_a_checkpoint_on_disk(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py")
        run(orchestrator.run_pipeline(run_id))

        written = {p.name for p in pipeline_home.run_dir(run_id).iterdir()}
        assert written == {"raw.parquet", "prepared.parquet", "model.pkl",
                           "tape.parquet", "metrics.json"}

    def test_metrics_are_well_formed(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py")
        run(orchestrator.run_pipeline(run_id))

        metrics = json.loads(
            (pipeline_home.run_dir(run_id) / "metrics.json").read_text()
        )
        assert metrics["n_obs"] == 20
        assert {"sharpe", "max_drawdown", "total_return", "units"} <= set(metrics)

    def test_researcher_print_output_is_captured_as_logs(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py", {"n": 7})
        run(orchestrator.run_pipeline(run_id))

        logs = [e["payload"]["line"] for e in pipeline_home.get_events(run_id)
                if e["event"] == "log"]
        assert "collecting 7 bars" in logs

    def test_run_completed_event_carries_the_metrics(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py")
        run(orchestrator.run_pipeline(run_id))

        final = [e for e in pipeline_home.get_events(run_id)
                 if e["event"] == "run_completed"]
        assert final and final[0]["payload"]["metrics"]["n_obs"] == 20


class TestFailurePauses:
    @pytest.fixture()
    def broken_run(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_broken.py")
        assert run(orchestrator.run_pipeline(run_id)) == "paused"
        return run_id

    def test_run_is_paused_not_failed(self, pipeline_home, broken_run):
        assert pipeline_home.get_run(broken_run)["status"] == "paused"

    def test_stages_before_the_failure_still_completed(self, pipeline_home, broken_run):
        steps = steps_by_name(pipeline_home, broken_run)
        assert steps["collect"]["status"] == "completed"
        assert steps["prepare"]["status"] == "failed"
        assert steps["fit"]["status"] == "pending"

    def test_the_completed_stage_kept_its_checkpoint(self, pipeline_home, broken_run):
        assert (pipeline_home.run_dir(broken_run) / "raw.parquet").exists()

    def test_traceback_is_recorded(self, pipeline_home, broken_run):
        step = pipeline_home.get_step(broken_run, "prepare")
        assert "no_such_column" in step["error"]
        assert "Traceback" in step["traceback"]

    def test_downstream_stages_never_ran(self, pipeline_home, broken_run):
        assert not (pipeline_home.run_dir(broken_run) / "prepared.parquet").exists()


class TestDebugContext:
    def test_context_describes_the_failure_and_its_input(self, pipeline_home, make_run):
        from backend.pipeline import context, orchestrator

        run_id = make_run("tiny_broken.py")
        run(orchestrator.run_pipeline(run_id))

        ctx = context.build_context(run_id, "prepare")
        assert ctx["status"] == "failed"
        assert "no_such_column" in ctx["error"]
        assert "no_such_column" in ctx["traceback"]
        # The agent hook must show what the stage was handed, not just what broke.
        assert ctx["inputs"]["collect"]["shape"] == [3, 3]
        assert ctx["inputs"]["collect"]["schema"]["price"] == "Float64"
        assert len(ctx["inputs"]["collect"]["head"]) == 3

    def test_context_includes_the_source_that_ran(self, pipeline_home, make_run):
        from backend.pipeline import context, orchestrator

        run_id = make_run("tiny_broken.py")
        run(orchestrator.run_pipeline(run_id))

        assert "no_such_column" in context.build_context(run_id, "prepare")["source"]

    def test_context_for_a_pipeline_owned_stage_reads_engine_source(
        self, pipeline_home, make_run
    ):
        from backend.pipeline import context, orchestrator

        run_id = make_run("tiny_ok.py")
        run(orchestrator.run_pipeline(run_id))

        assert "def on_tick" in context.build_context(run_id, "simulate")["source"]

    def test_context_never_unpickles_user_objects(self, pipeline_home, make_run):
        from backend.pipeline import context, orchestrator

        run_id = make_run("tiny_ok.py")
        run(orchestrator.run_pipeline(run_id))

        model = context.build_context(run_id, "simulate")["inputs"]["fit"]
        assert model["type"] == "pickle"
        assert "content" not in model

    def test_unknown_stage_is_rejected(self, pipeline_home, make_run):
        from backend.pipeline import context

        with pytest.raises(ValueError, match="unknown stage"):
            context.build_context(make_run("tiny_ok.py"), "nonsense")


class TestResume:
    def test_resume_reuses_the_upstream_checkpoint(self, pipeline_home, make_run):
        """The point of checkpointing: fixing prepare() must not re-run collect()."""
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_broken.py")
        run(orchestrator.run_pipeline(run_id))

        raw = pipeline_home.run_dir(run_id) / "raw.parquet"
        untouched = raw.stat().st_mtime_ns

        # The researcher fixes the file in place, as they would in an editor.
        strategy_path = Path(pipeline_home.get_run(run_id)["strategy_path"])
        strategy_path.write_text(
            (FIXTURES / "tiny_broken.py").read_text().replace(
                'raw.select(pl.col("no_such_column"))', "raw"
            )
        )

        assert run(orchestrator.resume(run_id, "prepare")) == "completed"
        assert raw.stat().st_mtime_ns == untouched

    def test_resume_clears_the_previous_failure(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_broken.py")
        run(orchestrator.run_pipeline(run_id))

        strategy_path = Path(pipeline_home.get_run(run_id)["strategy_path"])
        strategy_path.write_text(
            (FIXTURES / "tiny_broken.py").read_text().replace(
                'raw.select(pl.col("no_such_column"))', "raw"
            )
        )
        run(orchestrator.resume(run_id, "prepare"))

        step = pipeline_home.get_step(run_id, "prepare")
        assert step["status"] == "completed"
        assert step["error"] is None and step["traceback"] is None

    def test_resume_from_a_stage_with_no_upstream_checkpoint_fails_clearly(
        self, pipeline_home, make_run
    ):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py")
        assert run(orchestrator.resume(run_id, "simulate")) == "paused"
        assert "no checkpoint" in pipeline_home.get_step(run_id, "simulate")["error"]

    def test_remaining_stages_rejects_an_unknown_stage(self):
        from backend.pipeline import orchestrator

        with pytest.raises(ValueError, match="unknown stage"):
            orchestrator.remaining_stages("train")

    def test_remaining_stages_starts_where_asked(self):
        from backend.pipeline import orchestrator

        assert orchestrator.remaining_stages("fit") == [
            "fit", "simulate", "evaluate"
        ]


class TestTimeout:
    def test_a_hanging_stage_is_killed(self, pipeline_home, tmp_path):
        from backend.pipeline import orchestrator
        from backend.pipeline.contract import STAGES

        source = (
            "import time\n"
            "import polars as pl\n"
            "from backend.pipeline.contract import Strategy\n"
            "class Hanger(Strategy):\n"
            "    def collect(self, config):\n"
            "        time.sleep(60)\n"
            "    def prepare(self, raw, config): return raw\n"
            "    def on_tick(self, ts, market, inventory, model, config): return prepared\n"
        )
        strategy = pipeline_home.create_strategy("hanger.py", source)
        run_id = pipeline_home.create_run(
            strategy["id"], strategy["path"], {"stage_timeout_s": 2}, STAGES
        )

        assert run(orchestrator.run_pipeline(run_id)) == "paused"
        assert "timeout" in pipeline_home.get_step(run_id, "collect")["error"]


class TestEventStream:
    def test_events_are_ordered_and_cover_the_run(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py")
        run(orchestrator.run_pipeline(run_id))

        events = pipeline_home.get_events(run_id)
        assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)
        names = [e["event"] for e in events]
        assert names[0] == "run_started" and names[-1] == "run_completed"
        assert names.count("start") == 5 and names.count("done") == 5

    def test_after_seq_returns_only_newer_events(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py")
        run(orchestrator.run_pipeline(run_id))

        events = pipeline_home.get_events(run_id)
        midpoint = events[len(events) // 2]["seq"]
        assert all(
            e["seq"] > midpoint for e in pipeline_home.get_events(run_id, midpoint)
        )

    def test_subscribers_receive_events_live(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py")

        async def scenario():
            queue = orchestrator.hub.subscribe(run_id)
            await orchestrator.run_pipeline(run_id)
            received = []
            while not queue.empty():
                received.append(queue.get_nowait())
            return received

        received = run(scenario())
        assert [e["event"] for e in received][0] == "run_started"
        assert any(e["event"] == "run_completed" for e in received)


class TestDebugAttach:
    def test_command_is_plain_without_debug(self):
        from backend.pipeline import orchestrator

        command = orchestrator._build_command("r1", "prepare", None)
        assert "debugpy" not in command
        assert command[-3:] == ["backend.pipeline.runner", "r1", "prepare"]

    def test_command_waits_for_a_client_with_debug(self):
        from backend.pipeline import orchestrator

        command = orchestrator._build_command("r1", "prepare", 5679)
        assert "--wait-for-client" in command
        assert "127.0.0.1:5679" in command

    def test_free_port_is_bindable(self):
        import socket

        from backend.pipeline import orchestrator

        port = orchestrator.free_port()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))

    def test_debug_stage_blocks_until_a_client_attaches(self, pipeline_home, make_run):
        """The whole point of the debug path: the stage must hold, with its port
        open, so a human has time to attach before any code runs."""
        import socket

        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py")

        async def scenario():
            task = asyncio.create_task(
                orchestrator.resume(run_id, "collect", debug=True)
            )
            port = None
            for _ in range(200):
                await asyncio.sleep(0.05)
                ready = [e for e in pipeline_home.get_events(run_id)
                         if e["event"] == "debug_ready"]
                if ready:
                    port = ready[0]["payload"]["port"]
                    break
            assert port, "stage never reported a debug port"

            # The adapter is listening...
            with socket.create_connection(("127.0.0.1", port), timeout=5):
                pass
            # ...and the stage is still parked, not racing ahead.
            await asyncio.sleep(1.0)
            status = pipeline_home.get_step(run_id, "collect")["status"]

            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            return status

        assert run(scenario()) == "running"
        assert not (pipeline_home.run_dir(run_id) / "raw.parquet").exists()


class TestPermutationInThePipeline:
    """The permutation test runs inside the ``evaluate`` stage, in its own
    subprocess, and lands in ``metrics.json`` alongside everything else."""

    def metrics(self, store, run_id):
        return json.loads((store.run_dir(run_id) / "metrics.json").read_text())

    def test_metrics_carry_the_permutation_result(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py", {"permutation": {"n": 10, "seed": 1}})
        assert run(orchestrator.run_pipeline(run_id)) == "completed"

        block = self.metrics(pipeline_home, run_id)["permutation"]
        assert block["n"] == 10
        assert block["method"] == "rotate"
        assert block["metric"] == "sharpe"
        assert 0 < block["p_value"] <= 1

    def test_it_is_absent_unless_asked_for(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py")
        run(orchestrator.run_pipeline(run_id))
        assert "permutation" not in self.metrics(pipeline_home, run_id)

    def test_progress_streams_to_the_log(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py", {"permutation": {"n": 10}})
        run(orchestrator.run_pipeline(run_id))

        logs = [e["payload"]["line"] for e in pipeline_home.get_events(run_id)
                if e["event"] == "log"]
        assert "permutation 10/10" in logs

    def test_a_bad_permutation_config_pauses_at_evaluate(self, pipeline_home, make_run):
        """Rejected at the stage boundary, so the run pauses and every earlier
        checkpoint survives -- fix the config, resume from evaluate."""
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py", {"permutation": {"n": 5, "method": "bootstrap"}})
        assert run(orchestrator.run_pipeline(run_id)) == "paused"

        steps = steps_by_name(pipeline_home, run_id)
        assert steps["simulate"]["status"] == "completed"
        assert steps["evaluate"]["status"] == "failed"
        assert "method" in steps["evaluate"]["error"]

    def test_the_equity_paths_go_to_their_own_artifact(self, pipeline_home, make_run):
        """Inlining 50 curves would bury the eight numbers anyone reads."""
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py", {"permutation": {"n": 5, "seed": 1}})
        run(orchestrator.run_pipeline(run_id))

        assert "curves" not in self.metrics(pipeline_home, run_id)["permutation"]

        curves = json.loads(
            (pipeline_home.run_dir(run_id) / "permutation.json").read_text()
        )["curves"]
        assert len(curves["null"]) == 5
        assert len(curves["strategy"]) == len(curves["x"]) == 20

    def test_the_chart_data_is_downloadable(self, pipeline_home, make_run):
        """The UI fetches it through the ordinary artifact endpoint."""
        from fastapi.testclient import TestClient

        from backend.pipeline import orchestrator
        from backend.pipeline.api import app

        run_id = make_run("tiny_ok.py", {"permutation": {"n": 5, "seed": 1}})
        run(orchestrator.run_pipeline(run_id))

        with TestClient(app) as client:
            body = client.get(f"/runs/{run_id}/artifacts/permutation.json")
        assert body.status_code == 200
        assert body.json()["curves"]["x"][0] == 0


class TestSplitInThePipeline:
    """Chronological split runs in fit/simulate; metrics.split summarises sessions."""

    def metrics(self, store, run_id):
        return json.loads((store.run_dir(run_id) / "metrics.json").read_text())

    def test_metrics_carry_the_split_result(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py", {
            "n": 40,
            "split": {"enabled": True, "test_frac": 0.25, "purge": 0},
        })
        assert run(orchestrator.run_pipeline(run_id)) == "completed"

        block = self.metrics(pipeline_home, run_id)["split"]
        assert block["n_train"] + block["n_test"] == 40
        assert block["n_train_sessions"] == 1
        assert block["n_test_sessions"] == 1
        assert block["train_end"] < block["test_start_ts"]

    def test_it_is_absent_unless_asked_for(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py")
        run(orchestrator.run_pipeline(run_id))
        assert "split" not in self.metrics(pipeline_home, run_id)

    def test_progress_streams_to_the_log(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py", {
            "n": 40,
            "split": {
                "enabled": True, "test_frac": 0.4, "purge": 0,
                "test": {"mode": "rolling", "chunk_size": 5},
            },
        })
        run(orchestrator.run_pipeline(run_id))

        logs = [e["payload"]["line"] for e in pipeline_home.get_events(run_id)
                if e["event"] == "log"]
        assert any(line.startswith("test session ") for line in logs)

    def test_a_bad_split_config_pauses_at_fit(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py", {
            "split": {"enabled": True, "test_frac": 0.3, "train": {"mode": "bogus"}},
        })
        assert run(orchestrator.run_pipeline(run_id)) == "paused"

        steps = steps_by_name(pipeline_home, run_id)
        assert steps["prepare"]["status"] == "completed"
        assert steps["fit"]["status"] == "failed"
        assert "mode" in steps["fit"]["error"]

    def test_split_and_sessions_artifacts_exist(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py", {
            "n": 40,
            "split": {"enabled": True, "test_frac": 0.25, "purge": 0},
        })
        run(orchestrator.run_pipeline(run_id))

        root = pipeline_home.run_dir(run_id)
        assert (root / "split.json").exists()
        sessions = json.loads((root / "sessions.json").read_text())
        assert len(sessions["train"]) == 1
        assert len(sessions["test"]) == 1

    def test_the_artifacts_are_downloadable(self, pipeline_home, make_run):
        from fastapi.testclient import TestClient

        from backend.pipeline import orchestrator
        from backend.pipeline.api import app

        run_id = make_run("tiny_ok.py", {
            "n": 40,
            "split": {"enabled": True, "test_frac": 0.25, "purge": 0},
        })
        run(orchestrator.run_pipeline(run_id))

        with TestClient(app) as client:
            body = client.get(f"/runs/{run_id}/artifacts/split.json")
        assert body.status_code == 200
        assert body.json()["n_test"] > 0

    def test_train_and_test_rolling_aggregate_sessions(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_ok.py", {
            "n": 50,
            "split": {
                "enabled": True, "test_frac": 0.4, "purge": 0,
                "train": {"mode": "rolling", "chunk_size": 10},
                "test": {"mode": "rolling", "chunk_size": 5},
            },
        })
        assert run(orchestrator.run_pipeline(run_id)) == "completed"
        block = self.metrics(pipeline_home, run_id)["split"]
        assert block["n_train_sessions"] > 1
        assert block["n_test_sessions"] > 1
        assert len(block["test_sessions"]) == block["n_test_sessions"]

    def test_a_peeking_strategy_does_not_generalize_on_test(self, pipeline_home, make_run):
        from backend.pipeline import orchestrator

        run_id = make_run("tiny_peek.py", {
            "n": 60, "return_mode": "diff", "cost_bps": 0,
            "split": {"enabled": True, "test_frac": 0.3, "purge": 1},
        })
        assert run(orchestrator.run_pipeline(run_id)) == "completed"

        metrics = self.metrics(pipeline_home, run_id)
        # OOS book: peek has no memorized labels on test → near-flat / weak.
        assert metrics["sharpe"] < 1.0
        assert metrics["split"]["n_test"] > 0


class TestLiveLogs:
    """A researcher's print must arrive while the stage is still running.

    Python block-buffers stdout when it is a pipe, so without ``-u`` on the
    child every print sits in an 8KB buffer until the process exits: the log
    pane stays empty for the whole stage, then fills all at once. That is
    indistinguishable from a hung pipeline.
    """

    def test_the_child_runs_unbuffered(self):
        from backend.pipeline import orchestrator

        command = orchestrator._build_command("run", "collect", None)
        assert "-u" in command

    def test_debug_mode_is_unbuffered_too(self):
        from backend.pipeline import orchestrator

        command = orchestrator._build_command("run", "collect", 5678)
        assert command.index("-u") < command.index("-m")

    def test_a_print_arrives_before_the_stage_ends(self, pipeline_home, make_run):
        """The stage sleeps 3s after printing. Buffered, the line would only
        show up once the stage finished."""
        import time

        from backend.pipeline import orchestrator

        run_id = make_run("tiny_slow.py", {"sleep_s": 3.0})

        async def watch():
            queue = orchestrator.hub.subscribe(run_id)
            task = asyncio.create_task(orchestrator.run_pipeline(run_id))
            started = time.perf_counter()
            try:
                while True:
                    event = await asyncio.wait_for(queue.get(), timeout=10)
                    if event["event"] == "log" and "slow part" in event["payload"]["line"]:
                        return time.perf_counter() - started
                    if event["event"] in ("run_completed", "run_paused"):
                        raise AssertionError("stage ended before its log arrived")
            finally:
                orchestrator.hub.unsubscribe(run_id, queue)
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        assert run(watch()) < 2.0
