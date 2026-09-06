"""The debug context for a single stage, as JSON.

This is the agent hook. It assembles everything a human or an LLM needs to
explain why a stage behaved the way it did -- the code that ran, the config it
ran under, the shape and content of what it was handed, the traceback, and the
logs -- and stops there. Nothing in this module calls a model.

Wiring an agent up later is one function that POSTs this payload somewhere; no
restructuring is needed, because the context is already captured and stored.
"""

import inspect
import json
from pathlib import Path
from typing import Any

import polars as pl

from backend.pipeline import engine, store
from backend.pipeline.loader import method_source

#: Which upstream checkpoints a stage reads, and which method implements it.
STAGE_INPUTS = {
    "collect": (),
    "prepare": ("collect",),
    "fit": ("prepare",),
    "simulate": ("prepare", "fit"),
    "evaluate": ("simulate", "prepare"),
}

STAGE_METHODS = {
    "collect": "collect", "prepare": "prepare", "fit": "fit",
    "simulate": "on_tick",
}

HEAD_ROWS = 5
LOG_LINES = 50


def describe_artifact(path: str | None) -> dict | None:
    """Summarize a checkpoint without trusting its contents.

    Parquet and JSON are read; pickles are only measured. Unpickling a model
    would execute code from the uploaded strategy inside the API process, and
    describing a file is never worth that.
    """
    if not path:
        return None
    file = Path(path)
    if not file.exists():
        return {"path": path, "exists": False}

    base = {"path": path, "exists": True, "bytes": file.stat().st_size}

    try:
        if file.suffix == ".parquet":
            frame = pl.read_parquet(file)
            head = frame.head(HEAD_ROWS)
            return base | {
                "type": "dataframe",
                "shape": [frame.height, frame.width],
                "schema": {c: str(t) for c, t in frame.schema.items()},
                "null_counts": {
                    c: n for c, n in zip(frame.columns, frame.null_count().row(0)) if n
                },
                "head": json.loads(head.write_json()),
            }
        if file.suffix == ".json":
            return base | {"type": "json", "content": json.loads(file.read_text())}
        return base | {"type": "pickle", "note": "not loaded: unpickling runs user code"}
    except Exception as exc:
        # A corrupt checkpoint must not take down the whole context payload --
        # especially when the researcher opened it to read a stage error.
        return base | {"type": "error", "error": f"{type(exc).__name__}: {exc}"}


def stage_source(run: dict, stage: str) -> str | None:
    """The code that ran for this stage."""
    if stage == "evaluate":
        return inspect.getsource(engine.evaluate)
    path = Path(run["strategy_path"])
    if not path.exists():
        return inspect.getsource(engine.simulate) if stage == "simulate" else None
    try:
        src = method_source(path.read_text(), STAGE_METHODS[stage])
        if src:
            return src
    except Exception:
        pass
    if stage == "simulate":
        return inspect.getsource(engine.simulate)
    return None


def build_context(run_id: str, stage: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise LookupError(f"unknown run {run_id}")
    if stage not in STAGE_INPUTS:
        raise ValueError(f"unknown stage '{stage}'; expected {sorted(STAGE_INPUTS)}")

    step = store.get_step(run_id, stage) or {}
    steps = {s["name"]: s for s in store.get_steps(run_id)}

    logs = [
        event["payload"].get("line", "")
        for event in store.get_events(run_id)
        if event["event"] == "log" and event["stage"] == stage
    ]

    return {
        "run_id": run_id,
        "stage": stage,
        "status": step.get("status"),
        "run_status": run["status"],
        "config": run["config"],
        "error": step.get("error"),
        "traceback": step.get("traceback"),
        "source": stage_source(run, stage),
        "inputs": {
            name: describe_artifact(steps.get(name, {}).get("artifact"))
            for name in STAGE_INPUTS[stage]
        },
        "output": describe_artifact(step.get("artifact")),
        "logs": logs[-LOG_LINES:],
        "debug_port": step.get("debug_port"),
    }
