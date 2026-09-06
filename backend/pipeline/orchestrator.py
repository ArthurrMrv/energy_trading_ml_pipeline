"""Sequences stages, streams their progress, and pauses where it breaks.

One subprocess per stage. The orchestrator never imports the uploaded file, so
a strategy that segfaults or hangs takes down its own stage and nothing else.

It is the single writer to SQLite: stage subprocesses only report on stdout.
That keeps concurrent writers to one and the event ordering unambiguous.
"""

import asyncio
import contextlib
import json
import socket
import sys
from typing import Any, Iterable

from backend.pipeline import store
from backend.pipeline.contract import STAGES
from backend.pipeline.runner import SENTINEL

DEFAULT_STAGE_TIMEOUT_S = 900


class Hub:
    """In-process fan-out of run events to WebSocket subscribers."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue]] = {}

    def subscribe(self, run_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.setdefault(run_id, set()).add(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        listeners = self._subscribers.get(run_id)
        if listeners:
            listeners.discard(queue)
            if not listeners:
                self._subscribers.pop(run_id, None)

    def publish(self, run_id: str, record: dict) -> None:
        for queue in self._subscribers.get(run_id, ()):
            queue.put_nowait(record)


hub = Hub()


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _port_is_open(port: int) -> bool:
    """Passive check: if the port can no longer be bound, something is listening.

    Deliberately does not connect -- an incidental connection to a debug adapter
    that is waiting for a client risks being mistaken for that client.
    """
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return True
        return False


async def _announce_debug_port(run_id: str, stage: str, port: int,
                               timeout: float = 30.0) -> None:
    """Report the port only once the adapter is really accepting connections.

    debugpy takes a moment to bind. Announcing on spawn would hand out a port
    that refuses the first attach attempt.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if _port_is_open(port):
            _record(run_id, stage, "debug_ready", port=port,
                    hint=f"attach a debugger to 127.0.0.1:{port} to start this stage")
            return
        await asyncio.sleep(0.05)
    _record(run_id, stage, "debug_unavailable", port=port,
            error=f"debug adapter did not open port {port} within {timeout}s")


async def _cancel(*tasks: asyncio.Task | None) -> None:
    for task in tasks:
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


def _record(run_id: str, stage: str | None, event: str, **payload: Any) -> dict:
    """Persist an event and push it to any live listener."""
    saved = store.append_event(run_id, stage, event, payload)
    hub.publish(run_id, saved)
    return saved


def _build_command(run_id: str, stage: str, debug_port: int | None) -> list[str]:
    # -u matters more than it looks. The child's stdout is a pipe, so Python
    # block-buffers it: without this, a researcher's print() sits in an 8KB
    # buffer until the stage exits and the log pane shows nothing while the
    # stage is actually running. The runner's own protocol lines flush
    # explicitly; the researcher's prints cannot be expected to.
    command = [sys.executable, "-u"]
    if debug_port is not None:
        # Opt-in only. Putting --wait-for-client on the default path would hang
        # every run behind a debugger that is usually never going to attach.
        command += ["-m", "debugpy", "--listen", f"127.0.0.1:{debug_port}",
                    "--wait-for-client"]
    return command + ["-m", "backend.pipeline.runner", run_id, stage]


async def _consume(run_id: str, stage: str, stream: asyncio.StreamReader) -> dict:
    """Split the child's stdout into protocol events and plain log lines."""
    outcome: dict = {}
    async for raw in stream:
        line = raw.decode(errors="replace").rstrip("\n")
        if not line:
            continue
        if line.startswith(SENTINEL):
            try:
                message = json.loads(line[len(SENTINEL):])
            except json.JSONDecodeError:
                _record(run_id, stage, "log", line=line)
                continue
            payload = message.get("payload", {})
            _record(run_id, stage, message["event"], **payload)
            if message["event"] in ("done", "error"):
                outcome = {"event": message["event"], **payload}
        else:
            _record(run_id, stage, "log", line=line)
    return outcome


async def run_stage(run_id: str, stage: str, *, debug: bool = False,
                    timeout: int | None = None) -> bool:
    """Run one stage to completion. Returns True if it succeeded."""
    run = store.get_run(run_id)
    if run is None:
        raise LookupError(f"unknown run {run_id}")
    timeout = timeout or int(run["config"].get("stage_timeout_s", DEFAULT_STAGE_TIMEOUT_S))

    port = free_port() if debug else None
    store.update_step(run_id, stage, status="running", started_at=store.now(),
                      debug_port=port)

    process = await asyncio.create_subprocess_exec(
        *_build_command(run_id, stage, port),
        cwd=str(store.ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    consumer = asyncio.create_task(_consume(run_id, stage, process.stdout))
    announcer = (
        asyncio.create_task(_announce_debug_port(run_id, stage, port)) if port else None
    )
    try:
        # A debugger attach is a human waiting; do not race it against a clock.
        await asyncio.wait_for(process.wait(), None if debug else timeout)
    except asyncio.CancelledError:
        # Shutting the orchestrator down must not leave a stage subprocess
        # running -- especially a debug stage, which waits indefinitely.
        process.kill()
        await process.wait()
        await _cancel(consumer, announcer)
        store.update_step(run_id, stage, status="failed", ended_at=store.now(),
                          error=f"stage '{stage}' was cancelled")
        raise
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        await _cancel(consumer, announcer)
        message = f"stage '{stage}' exceeded its {timeout}s timeout and was killed"
        store.update_step(run_id, stage, status="failed", ended_at=store.now(),
                          error=message)
        _record(run_id, stage, "error", error=message)
        return False

    outcome = await consumer
    await _cancel(announcer)

    if process.returncode == 0 and outcome.get("event") == "done":
        store.update_step(run_id, stage, status="completed", ended_at=store.now(),
                          artifact=outcome.get("artifact"),
                          summary_json=outcome.get("summary"))
        return True

    error = outcome.get("error") or f"stage exited with code {process.returncode}"
    store.update_step(run_id, stage, status="failed", ended_at=store.now(),
                      error=error, traceback=outcome.get("traceback"))
    return False


async def run_pipeline(run_id: str, *, stages: Iterable[str] = STAGES,
                       debug_stage: str | None = None) -> str:
    """Run the given stages in order, stopping at the first failure.

    A failed run is left ``paused`` rather than ``failed``: every completed
    stage kept its checkpoint, so the researcher fixes the file and resumes
    from the broken stage instead of starting over.
    """
    stages = list(stages)
    store.set_run_status(run_id, "running")
    _record(run_id, None, "run_started", stages=stages)

    for stage in stages:
        ok = await run_stage(run_id, stage, debug=(stage == debug_stage))
        if not ok:
            store.set_run_status(run_id, "paused")
            step = store.get_step(run_id, stage) or {}
            _record(run_id, stage, "run_paused", failed_stage=stage,
                    error=step.get("error"),
                    traceback=step.get("traceback"),
                    resume=f"POST /runs/{run_id}/resume with from_stage='{stage}'")
            return "paused"

    store.set_run_status(run_id, "completed")
    _record(run_id, None, "run_completed", metrics=_final_metrics(run_id))
    return "completed"


def _final_metrics(run_id: str) -> dict | None:
    step = store.get_step(run_id, "evaluate")
    if not step or not step.get("artifact"):
        return None
    from backend.pipeline.runner import load_artifact

    return load_artifact(step["artifact"])


def remaining_stages(from_stage: str) -> list[str]:
    if from_stage not in STAGES:
        raise ValueError(f"unknown stage '{from_stage}'; expected one of {list(STAGES)}")
    return list(STAGES[STAGES.index(from_stage):])


async def resume(run_id: str, from_stage: str, *, debug: bool = False) -> str:
    """Re-run from ``from_stage``, reusing every checkpoint before it."""
    stages = remaining_stages(from_stage)
    store.reset_steps(run_id, stages)
    _record(run_id, from_stage, "run_resumed", from_stage=from_stage, debug=debug)
    return await run_pipeline(run_id, stages=stages,
                              debug_stage=from_stage if debug else None)
