"""Chronological train/test split with optional rolling sessions.

Split first. Train (fit / HPO) only sees train. Test scores held-out data,
optionally updating weights on an expanding history. Session results are kept
for every session, not only the last.
"""

from typing import Any

import polars as pl

MODES = ("once", "rolling")
SIDE_KEYS = {"mode", "chunk_size"}
TOP_KEYS = {"enabled", "test_frac", "test_start", "purge", "train", "test"}

SIDE_DEFAULTS = {"mode": "once", "chunk_size": 60}
DEFAULTS = {
    "enabled": True,
    "test_frac": 0.3,
    "test_start": None,
    "purge": None,
    "train": dict(SIDE_DEFAULTS),
    "test": dict(SIDE_DEFAULTS),
}


def _side(raw: Any, label: str) -> dict:
    if raw is None:
        return dict(SIDE_DEFAULTS)
    if not isinstance(raw, dict):
        raise ValueError(f"split '{label}' must be an object, got {type(raw).__name__}")
    unknown = sorted(set(raw) - SIDE_KEYS)
    if unknown:
        raise ValueError(f"unknown split.{label} option(s) {unknown}; expected {sorted(SIDE_KEYS)}")
    out = {**SIDE_DEFAULTS, **raw}
    if out["mode"] not in MODES:
        raise ValueError(f"split.{label}.mode must be one of {list(MODES)}, got {out['mode']!r}")
    size = out["chunk_size"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError(f"split.{label}.chunk_size must be an integer >= 1, got {size!r}")
    return out


def settings(config: dict) -> dict:
    """Validate ``config['split']``. Absent / null → disabled (full-sample path)."""
    raw = (config or {}).get("split", None)
    if raw is None:
        return {**DEFAULTS, "enabled": False, "train": dict(SIDE_DEFAULTS),
                "test": dict(SIDE_DEFAULTS)}
    if not isinstance(raw, dict):
        raise ValueError(f"config 'split' must be an object, got {type(raw).__name__}")
    unknown = sorted(set(raw) - TOP_KEYS)
    if unknown:
        raise ValueError(f"unknown split option(s) {unknown}; expected {sorted(TOP_KEYS)}")

    options = {
        **DEFAULTS,
        **{k: v for k, v in raw.items() if k not in ("train", "test")},
        "train": _side(raw.get("train"), "train"),
        "test": _side(raw.get("test"), "test"),
    }
    if not isinstance(options["enabled"], bool):
        raise ValueError(f"split 'enabled' must be a bool, got {options['enabled']!r}")

    frac = options["test_frac"]
    if not isinstance(frac, (int, float)) or isinstance(frac, bool) or not (0 < float(frac) < 1):
        raise ValueError(f"split 'test_frac' must be in (0, 1), got {frac!r}")
    options["test_frac"] = float(frac)

    purge = options["purge"]
    if purge is None:
        options["purge"] = 1
    elif not isinstance(purge, int) or isinstance(purge, bool) or purge < 0:
        raise ValueError(f"split 'purge' must be a non-negative integer, got {purge!r}")

    return options


def enabled(config: dict) -> bool:
    return settings(config)["enabled"]


def stamp(value: Any) -> Any:
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return iso()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def cut(
    timestamps: list,
    *,
    test_frac: float,
    test_start: Any = None,
    purge: int = 0,
) -> tuple[list, list]:
    """Chronological train / test cut. Train always ends before test begins."""
    stamps = list(timestamps)
    n = len(stamps)
    if n < 2:
        raise ValueError(f"split needs at least 2 timestamps, got {n}")

    if test_start is None:
        n_test = max(1, int(round(n * test_frac)))
        n_test = min(n_test, n - 1)
        idx = n - n_test
    else:
        try:
            idx = next(i for i, ts in enumerate(stamps) if ts >= test_start)
        except StopIteration:
            raise ValueError(f"split test_start {test_start!r} is after every timestamp") from None
        if idx < 1:
            raise ValueError(f"split test_start {test_start!r} leaves no training timestamps")

    train_stop = idx - purge
    if train_stop < 1:
        raise ValueError(
            f"purge={purge} leaves no training timestamps "
            f"({n} timestamps, cut at {idx})"
        )
    train_ts, test_ts = stamps[:train_stop], stamps[idx:]
    if not test_ts:
        raise ValueError("split produced an empty test window")
    if train_ts[-1] >= test_ts[0]:
        raise ValueError(
            f"split leaked: train_end={train_ts[-1]!r} >= test_start={test_ts[0]!r}"
        )
    return train_ts, test_ts


def chunk(timestamps: list, size: int) -> list[list]:
    """Fixed-length chunks; the last chunk may be shorter."""
    if size < 1:
        raise ValueError(f"chunk size must be >= 1, got {size}")
    stamps = list(timestamps)
    if not stamps:
        raise ValueError("cannot chunk an empty timestamp list")
    return [stamps[i:i + size] for i in range(0, len(stamps), size)]


def train_sessions(train_ts: list, *, mode: str, chunk_size: int) -> list[dict]:
    """Expanding train prefixes when rolling; one session when once."""
    if mode == "once":
        return [{"id": 0, "fit_ts": list(train_ts)}]
    parts = chunk(train_ts, chunk_size)
    out, seen = [], []
    for i, part in enumerate(parts):
        seen.extend(part)
        out.append({"id": i, "fit_ts": list(seen)})
    return out


def make_test_sessions(
    train_ts: list,
    test_ts: list,
    *,
    mode: str,
    chunk_size: int,
    purge: int = 0,
) -> list[dict]:
    """Score the whole test once, or expanding fit + per-chunk score."""
    if mode == "once":
        return [{"id": 0, "fit_ts": list(train_ts), "score_ts": list(test_ts)}]
    parts = chunk(test_ts, chunk_size)
    out, seen = [], []
    for i, part in enumerate(parts):
        base = list(train_ts) + seen
        # Cut already purged train vs first test bar. Later sessions need a fresh
        # purge so labels at the fit tail cannot straddle into the scored chunk.
        fit_ts = base if i == 0 else _purged(base, purge)
        if not fit_ts:
            raise ValueError(f"test session {i}: purge emptied the fit window")
        out.append({"id": i, "fit_ts": fit_ts, "score_ts": list(part)})
        seen.extend(part)
    return out


def _purged(timestamps: list, purge: int) -> list:
    if purge <= 0:
        return list(timestamps)
    if len(timestamps) <= purge:
        return []
    return list(timestamps[:-purge])


def in_ts(frame: pl.DataFrame, stamps: list) -> pl.DataFrame:
    """Filter rows whose ``ts`` is in ``stamps``.

    Compares as strings so Date columns survive ISO round-trips and mixed
    stamp types without ``is_in`` type errors.
    """
    if not stamps:
        return frame.clear()
    flat: list = []
    for value in stamps:
        if isinstance(value, (list, tuple)):
            flat.extend(value)
        else:
            flat.append(value)
    keys = [str(value) for value in flat]
    return frame.filter(pl.col("ts").cast(pl.Utf8).is_in(keys))


def timestamps(frame: pl.DataFrame) -> list:
    if "ts" not in frame.columns:
        raise ValueError("split needs a 'ts' column on prepare")
    return frame.select("ts").unique().sort("ts")["ts"].to_list()


def summary(train_ts: list, test_ts: list, options: dict) -> dict:
    return {
        "enabled": True,
        "test_frac": options["test_frac"],
        "test_start": stamp(options["test_start"]),
        "purge": options["purge"],
        "train": dict(options["train"]),
        "test": dict(options["test"]),
        "train_end": stamp(train_ts[-1]),
        "test_start_ts": stamp(test_ts[0]),
        "test_end": stamp(test_ts[-1]),
        "n_train": len(train_ts),
        "n_test": len(test_ts),
    }


def is_pack(obj: Any) -> bool:
    return isinstance(obj, dict) and obj.get("_split") is True
