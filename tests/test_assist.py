"""DeepSeek assist proxy: key from request only; strategy-only line patches."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.pipeline.assist import apply_patch
from tests.conftest import FIXTURES


@pytest.fixture()
def client(pipeline_home):
    from backend.pipeline.api import app

    with TestClient(app) as test_client:
        yield test_client


def _upload(client, fixture="tiny_ok.py"):
    return client.post(
        "/strategies",
        files={"file": (fixture, (FIXTURES / fixture).read_bytes(), "text/x-python")},
    ).json()


def _failed_run(client, pipeline_home):
    """Upload tiny_broken, run until paused at prepare, return ids."""
    strategy = _upload(client, "tiny_broken.py")
    run_id = client.post(
        "/runs", json={"strategy_id": strategy["id"], "config": {}},
    ).json()["run_id"]

    import time
    deadline = time.time() + 30
    while time.time() < deadline:
        body = client.get(f"/runs/{run_id}").json()
        if body["status"] == "paused":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("run did not pause")
    return strategy, run_id


BROKEN = (FIXTURES / "tiny_broken.py").read_text()
# Line 16 in tiny_broken.py is the bad select.
PATCH_REPLY = json.dumps({
    "exact_lines_to_replace": [16],
    "new_code_patch": "        return raw\n",
})


class TestApplyPatch:
    def test_replaces_a_single_line(self):
        out = apply_patch(BROKEN, {
            "exact_lines_to_replace": [16],
            "new_code_patch": "        return raw\n",
        })
        assert "no_such_column" not in out
        assert "return raw" in out
        assert out.count("\n") == BROKEN.count("\n")

    def test_rejects_non_contiguous_lines(self):
        with pytest.raises(ValueError, match="contiguous"):
            apply_patch(BROKEN, {
                "exact_lines_to_replace": [10, 12],
                "new_code_patch": "x\n",
            })


class TestAssistAuth:
    def test_chat_requires_bearer(self, client, pipeline_home):
        strategy = _upload(client)
        run_id = client.post(
            "/runs", json={"strategy_id": strategy["id"], "config": {}},
        ).json()["run_id"]
        response = client.post(
            "/assist/chat",
            json={"run_id": run_id, "stage": "collect", "messages": []},
        )
        assert response.status_code == 401


class TestAssistChat:
    def test_proxies_to_deepseek(self, client, pipeline_home):
        _, run_id = _failed_run(client, pipeline_home)

        async def fake_deepseek(key, system, messages):
            assert key == "sk-test"
            return "The prepare stage failed because a column is missing."

        with patch("backend.pipeline.assist._deepseek", new=AsyncMock(side_effect=fake_deepseek)):
            response = client.post(
                "/assist/chat",
                headers={"Authorization": "Bearer sk-test"},
                json={"run_id": run_id, "stage": "prepare", "messages": []},
            )
        assert response.status_code == 200
        assert "missing" in response.json()["reply"].lower()


class TestAssistFix:
    def test_refuses_pipeline_stages(self, client, pipeline_home):
        strategy = _upload(client)
        run_id = client.post(
            "/runs", json={"strategy_id": strategy["id"], "config": {}},
        ).json()["run_id"]
        import time
        deadline = time.time() + 30
        while time.time() < deadline:
            if client.get(f"/runs/{run_id}").json()["status"] in ("completed", "paused"):
                break
            time.sleep(0.05)

        response = client.post(
            "/assist/fix",
            headers={"Authorization": "Bearer sk-test"},
            json={"run_id": run_id, "stage": "evaluate", "messages": []},
        )
        assert response.status_code == 400
        assert "strategy stages" in response.json()["detail"]

    def test_applies_json_line_patch(self, client, pipeline_home):
        _, run_id = _failed_run(client, pipeline_home)

        async def fake_deepseek(key, system, messages):
            assert "exact_lines_to_replace" in system
            return PATCH_REPLY

        with patch("backend.pipeline.assist._deepseek", new=AsyncMock(side_effect=fake_deepseek)):
            response = client.post(
                "/assist/fix",
                headers={"Authorization": "Bearer sk-test"},
                json={"run_id": run_id, "stage": "prepare", "messages": []},
            )
        assert response.status_code == 201
        body = response.json()
        assert body["filename"].endswith("_fix.py")
        assert body["class_name"] == "BrokenStrategy"
        fixed = Path(body["path"]).read_text()
        assert "no_such_column" not in fixed
        assert body["patch"]["exact_lines_to_replace"] == [16]

    def test_pipeline_bug_refusal_is_400(self, client, pipeline_home):
        _, run_id = _failed_run(client, pipeline_home)

        async def fake_deepseek(key, system, messages):
            return json.dumps({"pipeline_bug": "Date/String mismatch in split.in_ts"})

        with patch("backend.pipeline.assist._deepseek", new=AsyncMock(side_effect=fake_deepseek)):
            response = client.post(
                "/assist/fix",
                headers={"Authorization": "Bearer sk-test"},
                json={"run_id": run_id, "stage": "prepare", "messages": []},
            )
        assert response.status_code == 400
        assert "PIPELINE_BUG" in response.json()["detail"]
