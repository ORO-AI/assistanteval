"""AssistantEval's simulated user as a Harbor user agent.

    harbor run -p <exported task> --agent <target> --model <model> \\
        --user-agent assistanteval.harbor.user:SimulatedUserAgent --bridge acp

Harbor gives this agent the task's instruction, which holds the AssistantEval
task and its fixtures (the assistant never sees it), and the trial's
environment. It runs the task with `run_task.run_task`, as the local runner
does, with:

- the services: the task's ``services`` container, through its admin API
  (`assistanteval.remote`), called with curl from the agent container;
- the assistant: the target harness over ACP (`AcpChat`), through the acpx
  that Harbor's bridge configured;
- the simulated user: on the Harbor host, with the model roles set in the
  host's environment (`assistanteval.endpoints`).

It then uploads the run record to ``RECORD_DIR`` for the verifier.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import secrets
import shlex
import tempfile
from contextlib import contextmanager
from pathlib import Path

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from assistanteval import run_task
from assistanteval.assistants.acp import AcpChat
from assistanteval.connectors import ServiceSession
from assistanteval.harbor import RECORD_DIR, SERVICES_URL, TASK_MARK
from assistanteval.remote import ADMIN, RemoteSessions

TASK_BLOCK = re.compile(re.escape(TASK_MARK) + r"\s*```json\n(.*?)\n```", re.S)


def task_of(instruction: str) -> dict:
    """The task and its fixtures, from the instruction `export` wrote."""
    found = TASK_BLOCK.search(instruction)
    if found is None:
        raise ValueError("The instruction holds no AssistantEval task")
    return json.loads(found.group(1))


class SimulatedUserAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "assistanteval-user"

    def version(self) -> str:
        return "1"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        loop = asyncio.get_running_loop()

        def sh(command: str) -> tuple[int, str]:
            done = asyncio.run_coroutine_threadsafe(environment.exec(command), loop)
            result = done.result()
            return result.return_code, result.stdout or ""

        payload = task_of(instruction)
        token = secrets.token_urlsafe(32)
        admin = admin_call(sh, token)
        # Named for the trial: the record's run id, unique across a job's trials.
        record_dir = self.logs_dir / self.logs_dir.parent.name
        with tempfile.TemporaryDirectory() as root:
            for path, fixture in payload["fixtures"].items():
                (Path(root) / path).parent.mkdir(parents=True, exist_ok=True)
                (Path(root) / path).write_text(json.dumps(fixture))
            await asyncio.to_thread(admin, "POST", "/claim", {"token": token})
            sessions = RemoteSessions(admin)
            record = await asyncio.to_thread(
                run_task.run_task,
                payload["task"],
                "acp",
                acp_assistant(sh),
                record_dir,
                services_from=lambda name, fixture, kind: ServiceSession(
                    name, fixture, sessions, f"{SERVICES_URL}/{name}", kind
                ),
                manifest={"task_sha256": payload["task_sha256"]},
                root=Path(root),
            )
        await environment.exec(f"rm -rf {RECORD_DIR} && mkdir -p {RECORD_DIR}")
        await environment.upload_dir(record_dir, RECORD_DIR)
        context.metadata = {"end_reason": record.result["end_reason"]}


def admin_call(sh, token: str):
    """`RemoteSessions`' call: one admin request, sent with curl where the
    services container is reachable."""
    auth = shlex.quote(f"Authorization: Bearer {token}")

    def call(method: str, path: str, body: dict | None):
        data = base64.b64encode(json.dumps(body or {}).encode()).decode()
        code, out = sh(
            f"echo {data} | base64 -d | curl -sS -X {method} -H {auth} "
            "-H 'Content-Type: application/json' --data-binary @- "
            f"-w '\\n%{{http_code}}' {SERVICES_URL}{ADMIN}{path}"
        )
        text, _, status = out.rpartition("\n")
        if code != 0 or not status.startswith("2"):
            raise RuntimeError(f"services admin {method} {path}: HTTP {status or code}")
        return json.loads(text) if text.strip() else None

    return call


def acp_assistant(sh):
    """The runner's assistant factory for the target harness."""

    @contextmanager
    def open_chat(account, evidence_dir: Path):
        chat = AcpChat(sh)
        try:
            yield chat
        finally:
            chat.close()
            evidence_dir.mkdir(parents=True, exist_ok=True)
            (evidence_dir / "acp_events.json").write_text(json.dumps(chat.events))

    open_chat.connector_kind = "mcp_bearer"
    return open_chat
