"""Write AssistantEval tasks as Harbor tasks.

    assistanteval harbor-export tasks/ harbor-tasks/ [--wheel dist/assistanteval-*.whl]

Each task becomes a directory Harbor runs with AssistantEval's simulated user
(`assistanteval.harbor.user`) against any ACP target harness:

- ``instruction.md``: the task and its fixtures, for the simulated user only.
- ``environment/``: the harness's container (``main``, in the task's time
  zone) and a ``services`` container serving every service with an admin API
  (`assistanteval.remote`); ``task.toml`` lists each task service as an MCP
  server, which Harbor registers with the harness.
- ``tests/``: a separate verifier image that grades the run record the user
  agent uploads, with the JUDGE, READER and CHECKER roles passed from the host.

Only runnable tasks export; a folder's others are skipped. Both images
install AssistantEval from the wheel, built from this checkout
unless ``--wheel`` names one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from assistanteval import endpoints
from assistanteval.harbor import RECORD_DIR, SERVICES_URL, TASK_MARK
from assistanteval.task import load as load_task

ROOT = Path(__file__).resolve().parents[2]
AGENT_TIMEOUT_SECONDS = 90 * 60  # past the runner's whole-run deadline
VERIFIER_TIMEOUT_SECONDS = 30 * 60

MAIN_DOCKERFILE = """\
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \\
    ca-certificates curl git procps tzdata && rm -rf /var/lib/apt/lists/*
# The assistant's clock reads the world's time zone.
ENV TZ={timezone}
WORKDIR /workspace
"""

SERVICES_DOCKERFILE = """\
FROM python:3.12-slim
COPY {wheel} /tmp/
RUN pip install --no-cache-dir /tmp/{wheel}
EXPOSE 8000
CMD ["assistanteval", "serve", "--host", "0.0.0.0", "--port", "8000"]
"""

COMPOSE = """\
services:
  main:
    depends_on:
      services:
        condition: service_healthy
  services:
    build:
      context: ./services
    expose:
      - "8000"
    healthcheck:
      test: ["CMD", "python", "-c", "import socket; socket.create_connection(('localhost', 8000), 2)"]
      interval: 2s
      timeout: 5s
      retries: 30
"""

VERIFIER_DOCKERFILE = """\
FROM python:3.12-slim
COPY {wheel} /tmp/
RUN pip install --no-cache-dir /tmp/{wheel}
COPY test.sh /tests/test.sh
"""

TEST_SH = f"""\
#!/bin/bash
mkdir -p /logs/verifier
assistanteval harbor-verify {RECORD_DIR} /logs/verifier
"""

INSTRUCTION = """\
# {id}

An AssistantEval task. Run it with AssistantEval's simulated user, which plays
the user and keeps what only the user knows:

    harbor run -p <this task> --agent <target> --model <model> \\
        --user-agent assistanteval.harbor.user:SimulatedUserAgent --bridge acp

The block below is for the simulated user. The assistant under test never sees it.

{mark}
```json
{payload}
```
"""


def task_toml(task: dict) -> str:
    servers = "".join(
        f'\n[[environment.mcp_servers]]\nname = "{c["service"]}"\n'
        f'transport = "streamable-http"\nurl = "{SERVICES_URL}/{c["service"]}/mcp"\n'
        for c in task["environment"]["connectors"]
    )
    judges = "".join(
        f'{name} = "${{{name}:-}}"\n'
        for role in ("JUDGE", "READER", "CHECKER")
        for name in (f"ASSISTANTEVAL_{role}_{s}" for s in endpoints.SETTINGS)
    )
    return f"""\
schema_version = "1.3"

[task]
name = "assistanteval/{task["id"]}"
keywords = ["assistanteval", "{task["category"]}"]

[metadata]
category = "{task["category"]}"
twin = "{(task.get("twin") or {}).get("id", "")}"

[agent]
timeout_sec = {AGENT_TIMEOUT_SECONDS}

[verifier]
timeout_sec = {VERIFIER_TIMEOUT_SECONDS}
environment_mode = "separate"

[verifier.env]
{judges}
[environment]
{servers}"""


def export(path: Path, out: Path, wheel: Path, loaded=None) -> Path:
    """Write one task as a Harbor task directory under `out`."""
    raw = path.read_bytes()
    task = (loaded or load_task(path)).dump()
    if task["status"] != "runnable":
        raise ValueError(f"{task['id']} is not runnable: {task['environment']['note']}")
    target = out / task["id"]
    shutil.rmtree(target, ignore_errors=True)
    (target / "environment/services").mkdir(parents=True)
    (target / "tests").mkdir()
    fixtures = {
        c["fixture"]: json.loads((path.resolve().parents[1] / c["fixture"]).read_text())
        for c in task["environment"]["connectors"]
    }
    payload = {
        "task": json.loads(raw),
        "fixtures": fixtures,
        "task_sha256": hashlib.sha256(raw).hexdigest(),
    }
    (target / "instruction.md").write_text(
        INSTRUCTION.format(
            id=task["id"], mark=TASK_MARK, payload=json.dumps(payload, indent=1)
        )
    )
    (target / "task.toml").write_text(task_toml(task))
    env = target / "environment"
    (env / "Dockerfile").write_text(
        MAIN_DOCKERFILE.format(timezone=task["environment"]["timezone"])
    )
    (env / "docker-compose.yaml").write_text(COMPOSE)
    for folder, dockerfile in (
        (env / "services", SERVICES_DOCKERFILE),
        (target / "tests", VERIFIER_DOCKERFILE),
    ):
        shutil.copy(wheel, folder / wheel.name)
        (folder / "Dockerfile").write_text(dockerfile.format(wheel=wheel.name))
    (target / "tests/test.sh").write_text(TEST_SH)
    (target / "tests/test.sh").chmod(0o755)
    return target


def build_wheel(out: Path) -> Path:
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out), str(ROOT)],
        check=True,
        capture_output=True,
    )
    return next(out.glob("assistanteval-*.whl"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tasks", type=Path, help="a task file or a folder of them")
    parser.add_argument("out", type=Path)
    parser.add_argument("--wheel", type=Path, help="default: built from this checkout")
    args = parser.parse_args(argv)
    paths = sorted(args.tasks.glob("*.json")) if args.tasks.is_dir() else [args.tasks]
    with tempfile.TemporaryDirectory() as scratch:
        wheel = args.wheel or build_wheel(Path(scratch))
        for path in paths:
            if (task := load_task(path)).status != "runnable" and args.tasks.is_dir():
                print(f"skipped {path.stem}: not runnable", file=sys.stderr)
                continue
            print(export(path, args.out, wheel, task))
    return 0
