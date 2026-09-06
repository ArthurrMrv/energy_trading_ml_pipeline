"""DeepSeek-backed error assistant: explain stage failures; optional strategy fix.

The API key is supplied per request (browser localStorage → Authorization header).
Nothing here persists keys or chat history.

Fixes are minimal line patches: the model returns JSON with line numbers and a
replacement snippet; we splice that into a copy of the strategy file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

from backend.pipeline import context, store
from backend.pipeline.contract import STRATEGY_STAGES
from backend.pipeline.loader import StrategyLoadError, validate_source

#: Failures here can be patched in the uploaded strategy. ``simulate`` is
#: included because that is where ``on_tick`` runs.
FIXABLE_STAGES = (*STRATEGY_STAGES, "simulate")

DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"
MAX_TB = 4000
MAX_SRC = 6000
MAX_LOG = 2000

SYSTEM_CHAT = """\
You help a researcher debug a quantitative trading research pipeline.
Explain the stage failure in plain English: what broke, why, and what to try next.
Use Markdown (short paragraphs, bullet lists, fenced code when useful).
Be concise. If the bug is in pipeline code (backend/pipeline/*), say so clearly —
do not invent a strategy workaround unless one truly exists in the uploaded strategy.
Never invent APIs or files that are not in the context.
"""

SYSTEM_FIX = """\
You fix uploaded trading strategies for this research pipeline with the SMALLEST
possible edit.

Return ONLY a JSON object (no markdown fence, no prose) with exactly these keys:
{
  "exact_lines_to_replace": [<1-based line numbers, contiguous>],
  "new_code_patch": "<exact text that replaces those lines, including newlines>"
}

Rules:
- exact_lines_to_replace is a non-empty list of consecutive integers (e.g. [16] or [16,17,18]).
- new_code_patch is the exact replacement for that span (may be fewer or more lines).
- Change as little as possible. Do not rewrite style or add features.
- The numbered strategy file in the context is authoritative for line numbers.
If the failure is in pipeline code (not the strategy), return exactly:
{"pipeline_bug": "<one sentence>"}
"""


def _bearer(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise PermissionError("missing DeepSeek API key (Authorization: Bearer …)")
    key = authorization.split(" ", 1)[1].strip()
    if not key:
        raise PermissionError("missing DeepSeek API key (Authorization: Bearer …)")
    return key


def _clip(text: str | None, limit: int) -> str:
    if not text:
        return ""
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "\n…[truncated]"


def _context_blob(payload: dict) -> str:
    """Compact JSON the model can read; drop bulky artifact heads when huge."""
    slim = {
        "stage": payload.get("stage"),
        "status": payload.get("status"),
        "error": _clip(payload.get("error"), 800),
        "traceback": _clip(payload.get("traceback"), MAX_TB),
        "source": _clip(payload.get("source"), MAX_SRC),
        "config": payload.get("config"),
        "logs": _clip("\n".join(payload.get("logs") or []), MAX_LOG),
        "inputs": {},
    }
    for name, artifact in (payload.get("inputs") or {}).items():
        if not isinstance(artifact, dict):
            slim["inputs"][name] = artifact
            continue
        slim["inputs"][name] = {
            k: artifact[k]
            for k in ("type", "shape", "schema", "error", "note")
            if k in artifact
        }
    return json.dumps(slim, default=str, indent=2)


def _numbered(source: str) -> str:
    lines = source.splitlines()
    width = max(3, len(str(len(lines))))
    return "\n".join(f"{i:>{width}}|{line}" for i, line in enumerate(lines, start=1))


async def _deepseek(key: str, system: str, messages: list[dict]) -> str:
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, *messages],
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            DEEPSEEK_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
    if response.status_code >= 400:
        detail = response.text[:500]
        raise RuntimeError(f"DeepSeek HTTP {response.status_code}: {detail}")
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("DeepSeek response missing message content") from exc


def _user_messages(messages: list[dict], ctx_blob: str) -> list[dict]:
    """Prepend stage context once; keep only role/content from the client."""
    cleaned = []
    for item in messages or []:
        role = item.get("role")
        content = item.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        cleaned.append({"role": role, "content": content[:4000]})
    preface = {
        "role": "user",
        "content": "Stage context (JSON):\n" + ctx_blob,
    }
    if cleaned and cleaned[0]["role"] == "user":
        return [preface, *cleaned]
    return [preface, *cleaned] if cleaned else [
        preface,
        {"role": "user", "content": "Explain this failure in plain English."},
    ]


async def chat(run_id: str, stage: str, messages: list[dict],
               authorization: str | None) -> dict:
    key = _bearer(authorization)
    payload = context.build_context(run_id, stage)
    reply = await _deepseek(
        key, SYSTEM_CHAT, _user_messages(messages, _context_blob(payload)),
    )
    return {"reply": reply}


_JSON_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _parse_patch(reply: str) -> dict:
    text = reply.strip()
    if text.startswith("PIPELINE_BUG:"):
        raise ValueError(text)
    match = _JSON_FENCE.search(text)
    if match:
        text = match.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model did not return valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("model JSON must be an object")
    if "pipeline_bug" in data:
        raise ValueError(f"PIPELINE_BUG: {data['pipeline_bug']}")
    return data


def apply_patch(source: str, patch: dict) -> str:
    """Replace contiguous 1-based lines with ``new_code_patch``."""
    nums = patch.get("exact_lines_to_replace")
    replacement = patch.get("new_code_patch")
    if not isinstance(nums, list) or not nums:
        raise ValueError("exact_lines_to_replace must be a non-empty list of line numbers")
    if not all(isinstance(n, int) and not isinstance(n, bool) and n >= 1 for n in nums):
        raise ValueError("exact_lines_to_replace must be positive integers (1-based)")
    ordered = sorted(nums)
    if ordered != list(range(ordered[0], ordered[0] + len(ordered))):
        raise ValueError("exact_lines_to_replace must be a contiguous range")
    if not isinstance(replacement, str):
        raise ValueError("new_code_patch must be a string")

    lines = source.splitlines(keepends=True)
    start, end = ordered[0] - 1, ordered[-1]  # slice end exclusive
    if end > len(lines):
        raise ValueError(
            f"exact_lines_to_replace ends at {ordered[-1]} but file has {len(lines)} lines"
        )
    new_lines = replacement.splitlines(keepends=True)
    if replacement and not replacement.endswith("\n") and new_lines:
        # Preserve file shape when the patch is a mid-file chunk without a trailing newline.
        if end < len(lines) and lines[end - 1].endswith("\n") and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
    return "".join(lines[:start] + new_lines + lines[end:])


async def fix(run_id: str, stage: str, messages: list[dict],
              authorization: str | None) -> dict:
    if stage not in FIXABLE_STAGES:
        raise ValueError(
            f"fix is only available for strategy stages {list(FIXABLE_STAGES)}; "
            f"got '{stage}'"
        )
    key = _bearer(authorization)
    run = store.get_run(run_id)
    if run is None:
        raise LookupError(f"unknown run {run_id}")
    payload = context.build_context(run_id, stage)
    path = Path(run["strategy_path"])
    full_source = path.read_text() if path.exists() else payload.get("source") or ""
    if not full_source.strip():
        raise ValueError("no strategy source available to patch")

    blob = _context_blob(payload)
    blob_with_file = (
        blob
        + "\n\nNumbered strategy file (line|code):\n"
        + _clip(_numbered(full_source), MAX_SRC * 2)
    )

    fix_messages = list(messages or [])
    if not any(m.get("role") == "user" for m in fix_messages):
        fix_messages = [{"role": "user", "content":
                         "Return a minimal JSON line patch for this strategy."}]
    reply = await _deepseek(
        key, SYSTEM_FIX, _user_messages(fix_messages, blob_with_file),
    )
    patch = _parse_patch(reply)
    source = apply_patch(full_source, patch)
    try:
        class_name = validate_source(source)
    except StrategyLoadError as exc:
        raise ValueError(f"patched source failed validation: {exc}") from exc

    stem = Path(path.name).stem if path.name else "strategy"
    stem = re.sub(r"_fix(?:_\d+)?$", "", stem)
    filename = f"{stem}_fix.py"
    row = store.create_strategy(filename, source, class_name)
    return {**row, "patch": {
        "exact_lines_to_replace": sorted(patch["exact_lines_to_replace"]),
        "new_code_patch": patch["new_code_patch"],
    }}
