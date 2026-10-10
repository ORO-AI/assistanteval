import json
import os
import time
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from assistanteval import connectors, dates, world

SERVICES_URL = "http://services.test"
TASKS = Path(__file__).resolve().parents[1] / "tasks"
ANCHOR = datetime.fromisoformat("2026-09-29T09:00:00-07:00")  # for task fixtures


@pytest.fixture(autouse=True)
def keep_timezone(monkeypatch):
    """The runner sets TZ for the product's clock; keep it out of other tests."""
    monkeypatch.setenv("TZ", os.environ.get("TZ", "UTC"))
    yield
    monkeypatch.undo()
    time.tzset()


def catalog(root: Path) -> Path:
    """A task catalog at root: its tasks/ folder, sharing the repo's fixtures,
    which task files name relative to root."""
    (root / "tasks").mkdir(parents=True, exist_ok=True)
    (root / "tasks" / "fixtures").symlink_to(TASKS / "fixtures")
    return root / "tasks"


def render_task(task: dict, anchor: datetime) -> dict:
    """The task with every date placeholder filled in at `anchor`."""
    named = dates.resolve(task["environment"].get("dates", {}), anchor)
    return dates._render_task(task, named)


def task_fixture(task: str, service: str) -> dict:
    """A task fixture as a run hands it to its service: rendered at ANCHOR."""
    template = json.loads((TASKS / f"{task}.json").read_bytes())
    return json.loads(dates.render(template, TASKS.parent, ANCHOR).fixtures[service])


def session(tmp_path, service: str, state: dict):
    """A session on `state`: a call that invokes a tool as the engine does, and
    a read of the stored session."""
    sessions = world.Sessions(world.DirBackend(tmp_path))
    sid, _ = sessions.create(service, state)

    def call(tool: str, **args):
        status, out = world.invoke(
            sessions, sid, world.SERVICES[service].ops[tool], args
        )
        return status, out.model_dump(
            mode="json", by_alias=True
        ) if status == 200 else out

    return call, lambda: sessions.read(sid)


def in_process_services(store):
    """Services on a local session directory, served in process: the runner's
    ``services_from``, and an HTTP client that reaches them at SERVICES_URL."""
    sessions = world.Sessions(world.DirBackend(store))

    def services_from(name, fixture, connector_kind):
        return connectors.ServiceSession(
            name, fixture, sessions, f"{SERVICES_URL}/{name}", connector_kind
        )

    return services_from, TestClient(world.asgi(sessions), base_url=SERVICES_URL)
