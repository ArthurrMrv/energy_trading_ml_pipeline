"""Runs exactly one stage, in its own process.

Invoked as ``python -m backend.pipeline.runner <run_id> <stage>``. It reads the
previous stages' checkpoints from disk, runs one stage, writes its own
checkpoint and exits. Nothing is held in memory between stages, which is what
makes a single stage independently re-runnable -- with or without a debugger
attached to this process.

Progress is reported on stdout as sentinel-prefixed JSON. Anything the
researcher prints themselves passes through untouched and is captured as a log
line, so ``print`` debugging keeps working.
"""

import json
import pickle
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import polars as pl

from backend.pipeline import permutation, split, store
from backend.pipeline.contract import STAGES, to_polars, validate_prepared
from backend.pipeline.engine import evaluate, simulate
from backend.pipeline.loader import load_strategy

SENTINEL = "@@PIPE@@"


def _json_default(obj: Any) -> Any:
    """Dates become ISO strings; anything else must not be silently stringified.

    ``default=str`` used to turn a fitted model into ``\"<CatBoostRegressor...>\"``
    and the next stage would load a string where it expected ``.predict``.
    """
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


def emit(stage: str, event: str, **payload: Any) -> None:
    print(SENTINEL + json.dumps({"stage": stage, "event": event, "payload": payload}),
          flush=True)


def save_artifact(obj: Any, directory: Path, name: str) -> tuple[str, dict]:
    """Persist a stage result, picking the format from what it actually is.

    DataFrames go to parquet so they stay inspectable from any tool. Plain
    JSON-serializable dicts (metrics) stay JSON. Anything else -- including a
    dict that holds a fitted model -- is pickled, so ``fit`` → ``simulate`` can
    round-trip a real estimator.
    """
    if obj is None:
        path = directory / f"{name}.pkl"
        path.write_bytes(pickle.dumps(None))
        return str(path), {"type": "none"}

    if isinstance(obj, dict):
        try:
            text = json.dumps(obj, indent=2, default=_json_default)
        except TypeError:
            path = directory / f"{name}.pkl"
            path.write_bytes(pickle.dumps(obj))
            return str(path), {"type": "dict", "keys": sorted(map(str, obj)),
                               "format": "pickle"}
        path = directory / f"{name}.json"
        path.write_text(text)
        return str(path), {"type": "dict", "keys": sorted(obj)}

    try:
        frame = to_polars(obj)
    except Exception:
        path = directory / f"{name}.pkl"
        path.write_bytes(pickle.dumps(obj))
        return str(path), {"type": type(obj).__name__}

    path = directory / f"{name}.parquet"
    frame.write_parquet(path)
    return str(path), {
        "type": "dataframe", "rows": frame.height, "cols": frame.width,
        "columns": frame.columns,
    }


def load_artifact(path: str) -> Any:
    suffix = Path(path).suffix
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".json":
        return json.loads(Path(path).read_text())
    return pickle.loads(Path(path).read_bytes())


def require(run_id: str, stage: str) -> Any:
    """Load an upstream checkpoint, failing with a resumable message if absent."""
    step = store.get_step(run_id, stage)
    if not step or not step.get("artifact"):
        raise FileNotFoundError(
            f"stage '{stage}' has no checkpoint for run {run_id}; "
            f"re-run from '{stage}' first"
        )
    return load_artifact(step["artifact"])


def _fit_split(strategy, prepared: pl.DataFrame, config: dict, directory: Path) -> Any:
    """Train session(s) on train only; write split.json + sessions train rows."""
    options = split.settings(config)
    train_ts, test_ts = split.cut(
        split.timestamps(prepared),
        test_frac=options["test_frac"],
        test_start=options["test_start"],
        purge=options["purge"],
    )
    meta = split.summary(train_ts, test_ts, options)
    (directory / "split.json").write_text(json.dumps(meta, default=str, indent=2))

    specs = split.train_sessions(
        train_ts, mode=options["train"]["mode"], chunk_size=options["train"]["chunk_size"],
    )
    train_rows, sessions = [], []
    for i, spec in enumerate(specs, start=1):
        print(f"train session {i}/{len(specs)}", flush=True)
        model = strategy.fit(split.in_ts(prepared, spec["fit_ts"]), config)
        sessions.append({"id": spec["id"], "model": model})
        train_rows.append({
            "id": spec["id"],
            "n_fit": len(spec["fit_ts"]),
            "fit_end": split.stamp(spec["fit_ts"][-1]),
        })

    (directory / "sessions.json").write_text(json.dumps({
        "train": train_rows, "test": [],
    }, default=str, indent=2))

    return {
        "_split": True,
        "chosen": sessions[-1]["model"],
        "sessions": sessions,
        "train_ts": train_ts,
        "test_ts": test_ts,
        "options": {
            "purge": options["purge"],
            "train": options["train"],
            "test": options["test"],
        },
    }


def _simulate_split(make_strategy, prepared: pl.DataFrame, pack: dict, config: dict,
                    directory: Path) -> pl.DataFrame:
    """Test session(s): empty book each chunk, new strategy instance each chunk."""
    options = pack["options"]
    train_ts, test_ts = pack["train_ts"], pack["test_ts"]
    specs = split.make_test_sessions(
        train_ts, test_ts,
        mode=options["test"]["mode"],
        chunk_size=options["test"]["chunk_size"],
        purge=options["purge"],
    )

    tapes = []
    test_rows = []
    for i, spec in enumerate(specs, start=1):
        print(f"test session {i}/{len(specs)}", flush=True)
        if options["test"]["mode"] == "once":
            model = pack["chosen"]
        else:
            model = make_strategy().fit(split.in_ts(prepared, spec["fit_ts"]), config)
        part = simulate(
            make_strategy(), split.in_ts(prepared, spec["score_ts"]), model, config,
        )
        tapes.append(part)
        test_rows.append({
            "id": spec["id"],
            "n_fit": len(spec["fit_ts"]),
            "n_score": len(spec["score_ts"]),
            "fit_end": split.stamp(spec["fit_ts"][-1]),
            "score_start": split.stamp(spec["score_ts"][0]),
            "score_end": split.stamp(spec["score_ts"][-1]),
            "score_ts": [split.stamp(t) for t in spec["score_ts"]],
        })

    sessions_path = directory / "sessions.json"
    train_rows = []
    if sessions_path.exists():
        train_rows = json.loads(sessions_path.read_text()).get("train", [])
    sessions_path.write_text(json.dumps({
        "train": train_rows, "test": test_rows,
    }, default=str, indent=2))

    if len(tapes) == 1:
        return tapes[0]
    return pl.concat(tapes).sort("ts", "asset")


def _session_metrics(tape: pl.DataFrame, config: dict, directory: Path) -> dict | None:
    """Attach per-session evaluate() numbers; primary tape metrics stay top-level."""
    split_path = directory / "split.json"
    sessions_path = directory / "sessions.json"
    if not split_path.exists() or not sessions_path.exists():
        return None
    meta = json.loads(split_path.read_text())
    sessions = json.loads(sessions_path.read_text())
    test_out = []
    for row in sessions.get("test", []):
        keys = [str(t) for t in row.get("score_ts", [])]
        slim = {k: v for k, v in row.items() if k != "score_ts"}
        chunk = tape.filter(pl.col("ts").cast(pl.Utf8).is_in(keys)) if keys else tape.clear()
        if chunk.height == 0:
            test_out.append({**slim, "sharpe": None})
            continue
        m = evaluate(chunk, config)
        test_out.append({
            **slim,
            "sharpe": m.get("sharpe"),
            "total_return": m.get("total_return"),
            "n_obs": m.get("n_obs"),
        })
    sessions_path.write_text(json.dumps({
        "train": sessions.get("train", []), "test": test_out,
    }, default=str, indent=2))
    return {
        **meta,
        "n_train_sessions": len(sessions.get("train", [])),
        "n_test_sessions": len(test_out),
        "train_sessions": sessions.get("train", []),
        "test_sessions": test_out,
    }


def run_stage(run_id: str, stage: str) -> tuple[str, dict]:
    run = store.get_run(run_id)
    if run is None:
        raise LookupError(f"unknown run {run_id}")

    config = run["config"]
    directory = store.run_dir(run_id)

    if stage == "evaluate":
        tape = require(run_id, "simulate")
        metrics = evaluate(tape, config)
        if permutation.enabled(config):
            prepared = validate_prepared(
                to_polars(require(run_id, "prepare"), stage="prepare")
            )
            model = require(run_id, "fit")
            sim_model = model["chosen"] if split.is_pack(model) else model
            sim_prepared = (
                split.in_ts(prepared, model["test_ts"]) if split.is_pack(model)
                else prepared
            )
            result = permutation.permutation_test(
                load_strategy(run["strategy_path"]), sim_prepared, sim_model, config,
                tape=tape,
                on_progress=lambda done, total: print(
                    f"permutation {done}/{total}", flush=True
                ),
            )
            (directory / "permutation.json").write_text(json.dumps(result))
            metrics = {**metrics, "permutation": {
                key: value for key, value in result.items() if key != "curves"
            }}
        split_block = _session_metrics(tape, config, directory)
        if split_block is not None:
            metrics = {**metrics, "split": split_block}
        return save_artifact(metrics, directory, "metrics")

    make_strategy = lambda: load_strategy(run["strategy_path"])
    strategy = make_strategy()

    if stage == "collect":
        return save_artifact(strategy.collect(config), directory, "raw")
    if stage == "prepare":
        return save_artifact(
            strategy.prepare(require(run_id, "collect"), config), directory, "prepared"
        )
    if stage == "fit":
        prepared = require(run_id, "prepare")
        if split.enabled(config):
            return save_artifact(
                _fit_split(strategy, to_polars(prepared, stage="prepare"), config, directory),
                directory, "model",
            )
        return save_artifact(strategy.fit(prepared, config), directory, "model")
    if stage == "simulate":
        prepared = validate_prepared(
            to_polars(require(run_id, "prepare"), stage="prepare")
        )
        model = require(run_id, "fit")
        if split.is_pack(model):
            tape = _simulate_split(make_strategy, prepared, model, config, directory)
        else:
            tape = simulate(strategy, prepared, model, config)
        return save_artifact(tape, directory, "tape")

    raise ValueError(f"unknown stage '{stage}'; expected one of {list(STAGES)}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m backend.pipeline.runner <run_id> <stage>",
              file=sys.stderr)
        return 2

    run_id, stage = argv
    emit(stage, "start")
    started = time.perf_counter()
    try:
        artifact, summary = run_stage(run_id, stage)
    except Exception as exc:
        emit(stage, "error", error=f"{type(exc).__name__}: {exc}",
             traceback=traceback.format_exc())
        return 1

    emit(stage, "done", artifact=artifact, summary=summary,
         ms=round((time.perf_counter() - started) * 1000, 1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
