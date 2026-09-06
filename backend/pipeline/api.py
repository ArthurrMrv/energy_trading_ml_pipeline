"""HTTP and WebSocket surface.

Single user by design: no auth, no tenancy. The API process holds no strategy
code -- it validates uploads statically and hands execution to subprocesses.
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.pipeline import assist, context, orchestrator, store
from backend.pipeline.contract import STAGES
from backend.pipeline.loader import StrategyLoadError, validate_source

async def _cancel_active_runs() -> None:
    """Stop in-flight runs on shutdown.

    A debug stage waits for a client indefinitely, so without this the server
    cannot exit and its subprocess keeps holding the debug port. Cancelling the
    task lets the orchestrator kill the child it owns.
    """
    for task in list(_active.values()):
        if not task.done():
            task.cancel()
    if _active:
        await asyncio.gather(*_active.values(), return_exceptions=True)
    _active.clear()


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    store.prune()
    try:
        yield
    finally:
        await _cancel_active_runs()


app = FastAPI(title="Research Pipeline", version="0.1.0", lifespan=lifespan)

#: Runs currently executing in this process, so a run cannot be started twice.
_active: dict[str, asyncio.Task] = {}


class RunRequest(BaseModel):
    strategy_id: str
    config: dict = Field(default_factory=dict)


class ResumeRequest(BaseModel):
    from_stage: str
    debug: bool = False


class AssistRequest(BaseModel):
    run_id: str
    stage: str
    messages: list[dict] = Field(default_factory=list)


def _require_run(run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"unknown run {run_id}")
    return run


def _launch(run_id: str, coro) -> None:
    if not (task := _active.get(run_id)) or task.done():
        _active[run_id] = asyncio.create_task(coro)
        return
    raise HTTPException(409, f"run {run_id} is already executing")


# --- strategies -------------------------------------------------------------

@app.post("/strategies", status_code=201)
async def upload_strategy(file: UploadFile) -> dict:
    if not (file.filename or "").endswith(".py"):
        raise HTTPException(400, "expected a .py file")
    source = (await file.read()).decode("utf-8", errors="replace")
    try:
        class_name = validate_source(source)
    except StrategyLoadError as exc:
        raise HTTPException(422, str(exc)) from exc
    return store.create_strategy(file.filename, source, class_name)


@app.get("/strategies")
def list_strategies(limit: int = 50) -> dict:
    store.prune()
    return {"strategies": store.list_strategies(limit)}


@app.get("/strategies/{strategy_id}")
def get_strategy(strategy_id: str) -> dict:
    strategy = store.get_strategy(strategy_id)
    if strategy is None:
        raise HTTPException(404, f"unknown strategy {strategy_id}")
    return strategy | {"source": Path(strategy["path"]).read_text()}


# --- runs -------------------------------------------------------------------

@app.post("/runs", status_code=201)
async def create_run(request: RunRequest) -> dict:
    strategy = store.get_strategy(request.strategy_id)
    if strategy is None:
        raise HTTPException(404, f"unknown strategy {request.strategy_id}")

    run_id = store.create_run(strategy["id"], strategy["path"], request.config, STAGES)
    _launch(run_id, orchestrator.run_pipeline(run_id))
    return {"run_id": run_id, "status": "running", "stages": list(STAGES)}


@app.get("/runs")
def list_runs(limit: int = 50) -> dict:
    # The listings are where a hand-deleted run gets noticed, so this is where
    # the index catches up with the filesystem.
    store.prune()
    return {"runs": store.list_runs(limit)}


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    run = _require_run(run_id)
    steps = sorted(store.get_steps(run_id), key=lambda s: STAGES.index(s["name"]))
    return run | {"steps": steps}


@app.post("/runs/{run_id}/resume")
async def resume_run(run_id: str, request: ResumeRequest) -> dict:
    _require_run(run_id)
    try:
        stages = orchestrator.remaining_stages(request.from_stage)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    _launch(run_id, orchestrator.resume(run_id, request.from_stage, debug=request.debug))
    response = {"run_id": run_id, "status": "running", "stages": stages}
    if request.debug:
        response["note"] = (
            f"stage '{request.from_stage}' will wait for a debugger; poll "
            f"GET /runs/{run_id} for its debug_port, then attach to 127.0.0.1:<port>"
        )
    return response


@app.get("/runs/{run_id}/steps/{stage}/context")
def get_context(run_id: str, stage: str) -> dict:
    """Everything needed to diagnose a stage, as JSON. No model is called."""
    _require_run(run_id)
    try:
        return context.build_context(run_id, stage)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# --- assist (DeepSeek proxy; key from request only) -------------------------

def _assist_http(exc: Exception) -> HTTPException:
    if isinstance(exc, PermissionError):
        return HTTPException(401, str(exc))
    if isinstance(exc, LookupError):
        return HTTPException(404, str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(400, str(exc))
    return HTTPException(502, str(exc))


@app.post("/assist/chat")
async def assist_chat(
    request: AssistRequest,
    authorization: str | None = Header(default=None),
) -> dict:
    """Explain a stage failure. Authorization: Bearer <DeepSeek key>."""
    _require_run(request.run_id)
    try:
        return await assist.chat(
            request.run_id, request.stage, request.messages, authorization,
        )
    except Exception as exc:
        raise _assist_http(exc) from exc


@app.post("/assist/fix", status_code=201)
async def assist_fix(
    request: AssistRequest,
    authorization: str | None = Header(default=None),
) -> dict:
    """Minimal strategy rewrite. Authorization: Bearer <DeepSeek key>."""
    _require_run(request.run_id)
    try:
        return await assist.fix(
            request.run_id, request.stage, request.messages, authorization,
        )
    except Exception as exc:
        raise _assist_http(exc) from exc


@app.get("/runs/{run_id}/artifacts/{name}")
def get_artifact(run_id: str, name: str) -> FileResponse:
    _require_run(run_id)
    # Resolve and confine: never serve a path that escapes the run directory.
    directory = store.run_dir(run_id).resolve()
    path = (directory / name).resolve()
    if not path.is_relative_to(directory) or not path.is_file():
        raise HTTPException(404, f"no artifact '{name}' for run {run_id}")
    return FileResponse(path, filename=path.name)


@app.get("/runs/{run_id}/events")
def get_events(run_id: str, after: int = 0) -> dict:
    _require_run(run_id)
    return {"events": store.get_events(run_id, after)}


@app.websocket("/runs/{run_id}/events")
async def stream_events(websocket: WebSocket, run_id: str) -> None:
    """Replay what already happened, then tail. A client that connects late
    still sees the whole run rather than joining mid-story."""
    await websocket.accept()
    if store.get_run(run_id) is None:
        await websocket.close(code=4004, reason=f"unknown run {run_id}")
        return

    queue = orchestrator.hub.subscribe(run_id)
    try:
        seen = 0
        for event in store.get_events(run_id):
            await websocket.send_json(event)
            seen = event["seq"]

        while True:
            event = await queue.get()
            if event["seq"] > seen:  # skip anything the replay already sent
                await websocket.send_json(event)
                seen = event["seq"]
    except WebSocketDisconnect:
        pass
    finally:
        orchestrator.hub.unsubscribe(run_id, queue)


# --- ui ---------------------------------------------------------------------
# Registered last, deliberately. A Mount("/") matches every path and Starlette
# resolves routes in registration order, so mounting this any earlier would
# serve index.html in place of the whole API.
app.mount("/", StaticFiles(directory=store.ROOT / "frontend", html=True), name="ui")
