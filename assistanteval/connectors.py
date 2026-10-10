"""Open a task connector's service by name, for `assistanteval.run_task`.

Every service runs in this process, on one local HTTP server started on first
use: service ``s`` serves REST at ``<base>/s`` and MCP at ``<base>/s/mcp``.

- ``ASSISTANTEVAL_SERVICES_HOST`` / ``_PORT``: where the server listens
  (default 127.0.0.1 and a free port).
- ``ASSISTANTEVAL_SERVICES_URL``: the base URL the assistant is given, when it
  reaches this machine by another name, such as
  ``http://host.docker.internal:8765`` from a container (set the host and port
  to match).
- ``ASSISTANTEVAL_SERVICES_STORE``: the session directory (default: a new
  temporary one).

A caller with services elsewhere passes its own ``services_from`` to
``run_task``, returning objects with this module's `ServiceSession` methods.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from assistanteval import world
from assistanteval.contracts import EpisodeStepError
from assistanteval.run_record import Request


@dataclass(frozen=True)
class ApiKeyConnector:
    kind: ClassVar[str] = "openapi_api_key"
    name: str
    url: str  # the REST API's base URL; its OpenAPI schema is <url>/openapi.json
    key: str = field(repr=False)


@dataclass(frozen=True)
class McpBearerConnector:
    kind: ClassVar[str] = "mcp_bearer"
    name: str
    url: str  # the streamable HTTP MCP endpoint
    key: str = field(repr=False)


@dataclass(frozen=True)
class Prepared:
    connector: ApiKeyConnector | McpBearerConnector = field(repr=False)
    session_id: str


class ServiceSession:
    """One service's session for one run: prepare it, read its state and
    requests, and revoke its key."""

    def __init__(
        self,
        name: str,
        fixture: Path,
        sessions: world.Sessions,
        url: str,
        connector_kind: str,
    ):
        if connector_kind not in (ApiKeyConnector.kind, McpBearerConnector.kind):
            raise ValueError(f"{name} has no {connector_kind} connector")
        self.name, self.fixture, self.sessions = name, fixture, sessions
        self.url, self.connector_kind = url.rstrip("/"), connector_kind
        self.session_id = self.key = None

    def prepare(self) -> Prepared:
        self.session_id, self.key = self.sessions.create(
            self.name, json.loads(self.fixture.read_bytes())
        )
        title = world.SERVICES[self.name].title
        if self.connector_kind == McpBearerConnector.kind:
            connector = McpBearerConnector(title, f"{self.url}/mcp", self.key)
        else:
            connector = ApiKeyConnector(title, self.url, self.key)
        return Prepared(connector, self.session_id)

    def state(self) -> object:
        """This run's full service state now, as JSON data."""
        return self.sessions.read(self.session_id)["state"]

    def request_log(self) -> tuple[list[Request], bool]:
        """Every request, in order, with its operation's kind and plain words.
        The log is written with the state, so it is always complete."""
        fields = ("operation", "at_ms", "input", "output", "status", "error", "kind")
        log = self.sessions.read(self.session_id)["log"]
        return [Request(self.name, *(entry[f] for f in fields)) for entry in log], True

    def release(self) -> None:
        if self.session_id is not None:
            self.sessions.revoke(self.session_id)
            if self.sessions.resolve(self.key, self.name) is not None:
                raise EpisodeStepError(f"{self.name}_revocation_unverified")


class LocalServices:
    """Every service on one HTTP server in a background thread."""

    def __init__(self, store: Path, host: str = "127.0.0.1", port: int = 0, url=None):
        import uvicorn

        self.sessions = world.Sessions(world.DirBackend(store))
        config = uvicorn.Config(
            world.asgi(self.sessions), host=host, port=port, log_level="warning"
        )
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            if not self.thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError(f"The services could not listen on {host}:{port}")
            time.sleep(0.05)
        port = self.server.servers[0].sockets[0].getsockname()[1]  # port 0 picks one
        self.url = (url or f"http://{host}:{port}").rstrip("/")

    def close(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=10)


_local: LocalServices | None = None


def local_services() -> LocalServices:
    """This process's local services, started on first use."""
    global _local
    if _local is None:
        store = os.environ.get("ASSISTANTEVAL_SERVICES_STORE") or tempfile.mkdtemp(
            prefix="assistanteval-services-"
        )
        _local = LocalServices(
            Path(store),
            os.environ.get("ASSISTANTEVAL_SERVICES_HOST", "127.0.0.1"),
            int(os.environ.get("ASSISTANTEVAL_SERVICES_PORT", "0")),
            os.environ.get("ASSISTANTEVAL_SERVICES_URL"),
        )
    return _local


def open_service(name: str, fixture: Path, connector_kind: str) -> ServiceSession:
    """The service on this process's local server, offered over `connector_kind`."""
    if name not in world.SERVICES:
        raise ValueError(
            f"No service named {name} in assistanteval.services: "
            "the task needs a service the runner cannot open yet"
        )
    local = local_services()
    return ServiceSession(
        name, fixture, local.sessions, f"{local.url}/{name}", connector_kind
    )
