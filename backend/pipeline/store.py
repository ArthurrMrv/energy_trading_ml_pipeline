"""SQLite metadata plus the on-disk run directory.

Deliberately plain ``sqlite3``: the whole schema is three tables and there is
nothing here an ORM would make shorter.

Read helpers return new dicts rather than live cursors, so callers cannot
mutate stored state by accident.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HOME = Path(os.environ.get("PIPELINE_HOME", ROOT / "runs"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies (
    id TEXT PRIMARY KEY, filename TEXT NOT NULL,
    class_name TEXT, path TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, strategy_id TEXT NOT NULL, strategy_path TEXT NOT NULL,
    config_json TEXT NOT NULL, status TEXT NOT NULL,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS steps (
    run_id TEXT NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL,
    started_at TEXT, ended_at TEXT, error TEXT, traceback TEXT,
    artifact TEXT, summary_json TEXT, debug_port INTEGER,
    PRIMARY KEY (run_id, name)
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, ts TEXT NOT NULL,
    stage TEXT, event TEXT NOT NULL, payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, seq);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def db_path() -> Path:
    HOME.mkdir(parents=True, exist_ok=True)
    return HOME / "pipeline.db"


@contextmanager
def connect():
    """Open a connection, commit on success, and always close it.

    sqlite3's own context manager commits but leaves the connection open, which
    leaks a handle per call -- and this is called on every event of every stage.
    """
    conn = sqlite3.connect(db_path(), timeout=30.0)
    conn.row_factory = sqlite3.Row
    # The API process and every stage subprocess touch this file concurrently.
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)


def run_path(run_id: str) -> Path:
    """Where a run's files live, without creating anything.

    Distinct from ``run_dir`` on purpose: anything that only *reads* must use
    this, or listing the runs would recreate every directory a researcher just
    deleted.
    """
    return HOME / run_id


def run_dir(run_id: str) -> Path:
    path = run_path(run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def strategy_dir() -> Path:
    path = HOME / "strategies"
    path.mkdir(parents=True, exist_ok=True)
    return path


# --- strategies -------------------------------------------------------------

def create_strategy(filename: str, source: str, class_name: str | None = None) -> dict:
    strategy_id = uuid.uuid4().hex[:12]
    path = strategy_dir() / f"{strategy_id}.py"
    path.write_text(source)
    row = {
        "id": strategy_id, "filename": filename, "class_name": class_name,
        "path": str(path), "created_at": now(),
    }
    with connect() as conn:
        conn.execute(
            "INSERT INTO strategies VALUES (:id,:filename,:class_name,:path,:created_at)",
            row,
        )
    return row


def get_strategy(strategy_id: str) -> dict | None:
    """``None`` for a strategy whose file has been deleted, same as for one that
    never existed -- the file is the strategy, this table only indexes it."""
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM strategies WHERE id=?", (strategy_id,)
        ).fetchone()
    if not row or not Path(row["path"]).exists():
        return None
    return dict(row)


def list_strategies(limit: int = 50) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id,filename,class_name,created_at,path FROM strategies "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
        ).fetchall()
    return [{k: v for k, v in dict(r).items() if k != "path"}
            for r in rows if Path(r["path"]).exists()]


# --- runs -------------------------------------------------------------------

def create_run(strategy_id: str, strategy_path: str, config: dict,
               stages: tuple[str, ...]) -> str:
    run_id = uuid.uuid4().hex[:12]
    stamp = now()
    with connect() as conn:
        conn.execute(
            "INSERT INTO runs VALUES (?,?,?,?,?,?,?)",
            (run_id, strategy_id, strategy_path, json.dumps(config),
             "pending", stamp, stamp),
        )
        conn.executemany(
            "INSERT INTO steps (run_id,name,status) VALUES (?,?,'pending')",
            [(run_id, stage) for stage in stages],
        )
    run_dir(run_id)
    return run_id


def get_run(run_id: str) -> dict | None:
    """``None`` for a run whose directory has been deleted. Its checkpoints were
    the result; without them there is nothing left to show."""
    with connect() as conn:
        row = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    if not row or not run_path(run_id).exists():
        return None
    run = dict(row)
    run["config"] = json.loads(run.pop("config_json"))
    return run


def list_runs(limit: int = 50) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id,strategy_id,status,created_at,updated_at FROM runs "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows if run_path(r["id"]).exists()]


def prune() -> dict[str, list[str]]:
    """Reconcile the index with the filesystem, and report what went.

    Deleting a run directory is how a researcher deletes a run -- there is no
    other command for it. Left alone, the row survives and the UI lists a run
    that fails when opened, which reads as a bug in the pipeline rather than as
    the deletion it actually was.
    """
    with connect() as conn:
        runs = [row["id"] for row in conn.execute("SELECT id FROM runs").fetchall()
                if not run_path(row["id"]).exists()]
        strategies = [
            row["id"]
            for row in conn.execute("SELECT id,path FROM strategies").fetchall()
            if not Path(row["path"]).exists()
        ]

        gone = [(run_id,) for run_id in runs]
        # Events and steps hang off the run and are unreadable without it.
        conn.executemany("DELETE FROM events WHERE run_id=?", gone)
        conn.executemany("DELETE FROM steps WHERE run_id=?", gone)
        conn.executemany("DELETE FROM runs WHERE id=?", gone)
        conn.executemany("DELETE FROM strategies WHERE id=?",
                         [(sid,) for sid in strategies])

    return {"runs": runs, "strategies": strategies}


def set_run_status(run_id: str, status: str) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE runs SET status=?, updated_at=? WHERE id=?",
            (status, now(), run_id),
        )


# --- steps ------------------------------------------------------------------

_STEP_FIELDS = ("status", "started_at", "ended_at", "error", "traceback",
                "artifact", "summary_json", "debug_port")


def update_step(run_id: str, name: str, **fields: Any) -> None:
    unknown = set(fields) - set(_STEP_FIELDS)
    if unknown:
        raise ValueError(f"unknown step field(s): {sorted(unknown)}")
    if "summary_json" in fields and not isinstance(fields["summary_json"], (str, type(None))):
        fields["summary_json"] = json.dumps(fields["summary_json"])
    assignments = ", ".join(f"{k}=?" for k in fields)
    with connect() as conn:
        conn.execute(
            f"UPDATE steps SET {assignments} WHERE run_id=? AND name=?",
            (*fields.values(), run_id, name),
        )


def reset_steps(run_id: str, names: list[str]) -> None:
    """Clear prior results so a resumed stage does not report stale state."""
    with connect() as conn:
        conn.executemany(
            "UPDATE steps SET status='pending', started_at=NULL, ended_at=NULL, "
            "error=NULL, traceback=NULL, artifact=NULL, summary_json=NULL, "
            "debug_port=NULL WHERE run_id=? AND name=?",
            [(run_id, name) for name in names],
        )


def get_steps(run_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM steps WHERE run_id=?", (run_id,)
        ).fetchall()
    steps = []
    for row in rows:
        step = dict(row)
        step["summary"] = json.loads(step.pop("summary_json") or "null")
        steps.append(step)
    return steps


def get_step(run_id: str, name: str) -> dict | None:
    return next((s for s in get_steps(run_id) if s["name"] == name), None)


# --- events -----------------------------------------------------------------

def append_event(run_id: str, stage: str | None, event: str, payload: dict) -> dict:
    record = {"run_id": run_id, "ts": now(), "stage": stage,
              "event": event, "payload": payload}
    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO events (run_id,ts,stage,event,payload_json) VALUES (?,?,?,?,?)",
            (run_id, record["ts"], stage, event, json.dumps(payload)),
        )
        record["seq"] = cursor.lastrowid
    return record


def get_events(run_id: str, after_seq: int = 0) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM events WHERE run_id=? AND seq>? ORDER BY seq",
            (run_id, after_seq),
        ).fetchall()
    events = []
    for row in rows:
        record = dict(row)
        record["payload"] = json.loads(record.pop("payload_json"))
        events.append(record)
    return events
