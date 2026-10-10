"""The Harbor integration's parts, without Docker (see tests/harbor_smoke/smoke.py)."""

import asyncio
import base64
import json
import os
import re
import shutil
import tomllib

import pytest

from assistanteval import run_task
from assistanteval.harbor import export, verify
from assistanteval.harbor.targets import HermesTarget
from assistanteval.harbor.user import admin_call, task_of

TASK = run_task.ROOT / "tasks/kingsway-reschedule-crew.json"


@pytest.fixture
def exported(tmp_path):
    wheel = tmp_path / "assistanteval-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    return export.export(TASK, tmp_path / "out", wheel)


def test_the_instruction_carries_the_task_and_fixtures_to_the_simulated_user(exported):
    payload = task_of((exported / "instruction.md").read_text())
    assert payload["task"] == json.loads(TASK.read_text())
    assert set(payload["fixtures"]) == {
        c["fixture"] for c in payload["task"]["environment"]["connectors"]
    }
    config = tomllib.loads((exported / "task.toml").read_text())
    assert config["environment"]["mcp_servers"] == [
        {
            "name": "calendar",
            "transport": "streamable-http",
            "url": "http://services:8000/calendar/mcp",
        }
    ]
    assert (
        config["verifier"]["env"]["ASSISTANTEVAL_JUDGE_MODEL"]
        == "${ASSISTANTEVAL_JUDGE_MODEL:-}"
    )
    assert (
        "ENV TZ=America/Vancouver" in (exported / "environment/Dockerfile").read_text()
    )
    for image in ("environment/services", "tests"):
        assert (exported / image / "assistanteval-0.1.0-py3-none-any.whl").is_file()


def test_admin_calls_go_through_curl_with_the_token():
    commands = []

    def sh(command):
        commands.append(command)
        return 0, '{"session_id": "s1", "key": "k"}\n200'

    call = admin_call(sh, "secret")
    assert call("POST", "/sessions", {"service": "calendar", "fixture": {}}) == {
        "session_id": "s1",
        "key": "k",
    }
    sent = re.search(r"echo (\S+) \| base64 -d", commands[0]).group(1)
    assert json.loads(base64.b64decode(sent)) == {"service": "calendar", "fixture": {}}
    assert "'Authorization: Bearer secret'" in commands[0]
    assert commands[0].endswith("http://services:8000/_admin/sessions")
    with pytest.raises(RuntimeError, match="HTTP 401"):
        admin_call(lambda command: (0, '{"detail": "no"}\n401'), "t")(
            "GET", "/sessions/x", None
        )


@pytest.mark.parametrize(
    "model_name, model, provider",
    [
        # An OpenRouter id by default; a native provider when its key is set.
        ("qwen/qwen3.6-35b", "qwen/qwen3.6-35b", "openrouter"),
        ("anthropic/claude-sonnet-5", "claude-sonnet-5", "anthropic"),
        ("openrouter/qwen/qwen3.6-35b", "qwen/qwen3.6-35b", "openrouter"),
    ],
)
def test_hermes_picks_its_provider_from_the_model_name(
    tmp_path, monkeypatch, model_name, model, provider
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "an-key")
    target = HermesTarget(logs_dir=tmp_path, model_name=model_name)
    assert target._provider()[1:] == (model, provider)
    key = f"{provider.upper()}_API_KEY"
    assert target.acp_env()[key] == os.environ[key]


def test_hermes_runs_as_an_acp_server_with_the_tasks_mcp_servers(tmp_path, monkeypatch):
    from harbor.models.task.config import MCPServerConfig

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    target = HermesTarget(
        logs_dir=tmp_path,
        model_name="qwen/qwen3.6-35b",
        mcp_servers=[
            MCPServerConfig(
                name="calendar",
                transport="streamable-http",
                url="http://services:8000/calendar/mcp",
            )
        ],
    )
    commands, uploads = [], []

    async def exec_as_agent(environment, command, env=None, **kwargs):
        commands.append(command)

    monkeypatch.setattr(target, "exec_as_agent", exec_as_agent)

    class Environment:
        async def exec(self, command, user=None):
            commands.append(command)

        async def upload_file(self, source, target):
            uploads.append((source.name, target))

    asyncio.run(target.acp_install(Environment()))
    assert uploads == [("acp_resume_shim.py", "/opt/assistanteval/acp_resume_shim.py")]
    assert (
        "provider: openrouter" in commands[0]
        and "model: qwen/qwen3.6-35b" in commands[0]
    )
    assert "mcp_discovery_timeout: 30" in commands[0]
    assert "url: http://services:8000/calendar/mcp" in commands[1]
    assert "sync_venv(['acp']" in commands[2]  # the ACP extra
    # acpx starts Hermes through the shim.
    command = target.acp_command()
    assert "/opt/assistanteval/acp_resume_shim.py" in command
    assert command[-1].endswith("exec hermes acp")


def test_hermes_installs_into_the_home_it_runs_with(tmp_path, monkeypatch):
    target = HermesTarget(logs_dir=tmp_path, model_name="qwen/qwen3.6-35b")
    installs = []

    async def record(environment, command, env=None, **kwargs):
        installs.append(env)

    async def no_deps(environment, packages):
        pass

    monkeypatch.setattr(target, "exec_as_agent", record)
    monkeypatch.setattr(target, "ensure_system_dependencies", no_deps)
    asyncio.run(target.install(None))
    # Installed into the home Hermes runs with, or it has no MCP SDK there.
    assert installs == [{"HERMES_HOME": "/tmp/hermes"}]
    assert target.get_version_command().endswith("hermes --version")


def test_a_trial_with_no_run_record_is_invalid(tmp_path):
    assert verify.main([str(tmp_path / "missing"), str(tmp_path / "out")]) == 0
    rewards = json.loads((tmp_path / "out/reward.json").read_text())
    assert rewards["reward"] == 0 and rewards["invalid"] == 1


def test_only_runnable_tasks_export(tmp_path):
    with pytest.raises(ValueError, match="not runnable"):
        export.export(
            run_task.ROOT / "tasks/courier-date-clear.json",
            tmp_path,
            tmp_path / "w.whl",
        )


def test_export_reads_fixtures_from_the_tasks_own_catalog(tmp_path):
    """Not from the installed package: a downloaded catalog works anywhere."""
    elsewhere = tmp_path / "catalog" / "tasks"
    shutil.copytree(
        TASK.parent / "fixtures" / "kingsway-reschedule-crew",
        elsewhere / "fixtures" / "kingsway-reschedule-crew",
    )
    shutil.copy(TASK, elsewhere / TASK.name)
    calendar = elsewhere / "fixtures/kingsway-reschedule-crew/calendar.json"
    calendar.write_text(
        calendar.read_text().replace("Kingsway", "Kingsway (this catalog)")
    )
    wheel = tmp_path / "assistanteval-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    out = export.export(elsewhere / TASK.name, tmp_path / "out", wheel)
    assert "Kingsway (this catalog)" in (out / "instruction.md").read_text()
