"""In-process runner coverage. Subprocess e2e tests never reach these lines."""

import pytest

from backend.pipeline.contract import STAGES
from backend.pipeline.runner import load_artifact, main, require, run_stage


def stamp(store, run_id, stage, path, summary):
    store.update_step(
        run_id, stage, status="completed", artifact=path, summary_json=summary,
    )


class TestRunStage:
    def test_five_stages_complete_in_process(self, pipeline_home, make_run):
        run_id = make_run("tiny_ok.py", {"n": 8})
        for stage in STAGES:
            path, summary = run_stage(run_id, stage)
            stamp(pipeline_home, run_id, stage, path, summary)

        metrics = load_artifact(
            pipeline_home.get_step(run_id, "evaluate")["artifact"]
        )
        assert "sharpe" in metrics

    def test_split_once_writes_session_metrics(self, pipeline_home, make_run):
        run_id = make_run("tiny_ok.py", {
            "n": 20,
            "split": {"enabled": True, "test_frac": 0.3, "purge": 0},
        })
        for stage in STAGES:
            path, summary = run_stage(run_id, stage)
            stamp(pipeline_home, run_id, stage, path, summary)

        metrics = load_artifact(
            pipeline_home.get_step(run_id, "evaluate")["artifact"]
        )
        assert metrics["split"]["n_train_sessions"] == 1
        assert metrics["split"]["n_test_sessions"] == 1

    def test_rolling_split_scores_every_session_without_leaking(
        self, pipeline_home, make_run,
    ):
        """The invariant the rolling path exists to hold: every session fits
        strictly before the window it scores, purge included."""
        run_id = make_run("tiny_ok.py", {
            "n": 40,
            "split": {
                "enabled": True, "test_frac": 0.3, "purge": 1,
                "train": {"mode": "rolling", "chunk_size": 10},
                "test": {"mode": "rolling", "chunk_size": 4},
            },
        })
        for stage in STAGES:
            path, summary = run_stage(run_id, stage)
            stamp(pipeline_home, run_id, stage, path, summary)

        split = load_artifact(
            pipeline_home.get_step(run_id, "evaluate")["artifact"]
        )["split"]

        assert split["n_train_sessions"] > 1
        assert split["n_test_sessions"] > 1
        assert split["train_end"] < split["test_start_ts"]
        for session in split["test_sessions"]:
            assert session["fit_end"] < session["score_start"]

    def test_require_without_a_checkpoint_is_clear(self, pipeline_home, make_run):
        run_id = make_run("tiny_ok.py")
        with pytest.raises(FileNotFoundError, match="no checkpoint"):
            require(run_id, "collect")

    def test_main_collect_exits_zero(self, pipeline_home, make_run):
        assert main([make_run("tiny_ok.py", {"n": 8}), "collect"]) == 0
