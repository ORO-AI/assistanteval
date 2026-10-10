"""Services in another process, opened and read over HTTP.

`assistanteval serve` runs every service behind an admin API, for a runner that
cannot share the services' process: a Harbor user agent, for example, reaches
its task's services container only over the task's network.

- ``POST /_admin/claim {"token"}``: the first claim sets the admin token (a
  non-empty string); every later claim is refused. Every other admin call
  needs the token as its bearer key, so an assistant on the same network
  cannot open, read or revoke sessions.
- ``POST /_admin/sessions {"service", "fixture"}`` opens a session;
  ``GET`` and ``DELETE /_admin/sessions/<id>`` read and revoke it;
  ``POST /_admin/resolve {"key", "service"}`` names the session a key opens.
- A service request with no bearer key uses the service's latest session, for
  harnesses whose MCP configuration cannot carry a key (a Harbor task's MCP
  servers have no header field). The task's private network isolates the run.
  Before a service's first session, keyless MCP still lists its tools, for a
  harness that starts before the run; a tool call then reports it not ready.

`RemoteSessions` is the runner's side: `world.Sessions`' four methods over the
admin API, so `connectors.ServiceSession` works unchanged.
"""

from __future__ import annotations

import argparse
import hmac
import json
import tempfile
from pathlib import Path

from assistanteval import world

ADMIN = "/_admin"


class AdminApp:
    """The services' ASGI app, with the admin API and keyless requests."""

    def __init__(self, sessions: world.Sessions):
        self.sessions = sessions
        self.services = world.asgi(sessions)
        self.latest: dict[str, str] = {}  # service -> the key of its latest session
        self.token: str | None = None  # the claimed admin token

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return
        path, method = scope["path"], scope["method"]
        headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
        service = path.lstrip("/").partition("/")[0]
        keyless = "authorization" not in headers
        if path.startswith(ADMIN + "/"):
            auth = headers.get("authorization", "")
            reply = self.admin_reply(method, path, auth, await world.read_body(receive))
            await world.respond(send, *reply)
            return
        early = keyless and service in world.SERVICES and service not in self.latest
        if early and path == f"/{service}/mcp":
            reply = self.early_mcp(service, method, await world.read_body(receive))
            await world.respond(send, *reply)
            return
        if keyless and service in self.latest:
            bearer = f"Bearer {self.latest[service]}".encode()
            scope = {
                **scope,
                "headers": [*scope["headers"], (b"authorization", bearer)],
            }
        await self.services(scope, receive, send)

    def admin_reply(self, method: str, path: str, auth: str, body: bytes):
        request = _json(body or b"{}")
        try:
            status, out = self.admin(
                method,
                path.removeprefix(ADMIN),
                auth,
                request if isinstance(request, dict) else {},
            )
        except (KeyError, TypeError, ValueError):  # a malformed request
            status, out = 400, {"detail": "malformed admin request"}
        payload = b"" if out is None else json.dumps(out).encode()
        return status, {"content-type": "application/json"}, payload

    def admin(self, method: str, path: str, auth: str, body: dict):
        if path == "/claim" and method == "POST":
            if self.token:
                return 409, {"detail": "already claimed"}
            if not isinstance(token := body["token"], str) or not token:
                raise ValueError("the token must be a non-empty string")
            self.token = token
            return 204, None
        if not self.token or not hmac.compare_digest(auth, f"Bearer {self.token}"):
            return 401, {"detail": "admin token required"}
        if path == "/sessions" and method == "POST":
            sid, key = self.sessions.create(body["service"], body["fixture"])
            self.latest[body["service"]] = key
            return 200, {"session_id": sid, "key": key}
        if path == "/resolve" and method == "POST":
            sid = self.sessions.resolve(body["key"], body["service"])
            return 200, {"session_id": sid}
        sid = path.removeprefix("/sessions/")
        if sid == path:
            return 404, {"detail": "Not Found"}
        if method == "GET":
            return 200, self.sessions.read(sid)
        if method == "DELETE":
            self.sessions.revoke(sid)
            return 204, None
        return 405, {"detail": "Method Not Allowed"}

    def early_mcp(self, service: str, method: str, body: bytes):
        """MCP for a service whose session is not open yet. A harness may
        start before the run (Harbor's bridge starts it at setup) and parks
        a server that refuses its discovery, so the tools are listed now;
        only a call needs the session."""
        call = _json(body)
        if (
            method == "POST"
            and isinstance(call, dict)
            and call.get("method") == "tools/call"
        ):
            result = world.tool_error(f"{service} is not ready yet")
            reply = {"jsonrpc": "2.0", "id": call.get("id"), "result": result}
            return 200, {"content-type": "application/json"}, json.dumps(reply).encode()
        return world._mcp(self.sessions, None, world.SERVICES[service], method, body)


def _json(body: bytes):
    """The body's JSON, or None if it is not JSON."""
    try:
        return json.loads(body)
    except ValueError:
        return None


class RemoteSessions:
    """`world.Sessions`' methods over the admin API. ``call(method, path,
    body)`` sends one admin request and returns its JSON (None when empty),
    raising on an error status."""

    def __init__(self, call):
        self.call = call

    def create(self, service_name: str, fixture: dict) -> tuple[str, str]:
        out = self.call(
            "POST", "/sessions", {"service": service_name, "fixture": fixture}
        )
        return out["session_id"], out["key"]

    def read(self, sid: str) -> dict:
        return self.call("GET", f"/sessions/{sid}", None)

    def revoke(self, sid: str) -> None:
        self.call("DELETE", f"/sessions/{sid}", None)

    def resolve(self, key: str, service_name: str) -> str | None:
        out = self.call("POST", "/resolve", {"key": key, "service": service_name})
        return out["session_id"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve every service, with an admin API."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--store", type=Path, help="session directory (default: a new temporary one)"
    )
    args = parser.parse_args(argv)
    import uvicorn

    store = args.store or Path(tempfile.mkdtemp(prefix="assistanteval-services-"))
    sessions = world.Sessions(world.DirBackend(store))
    uvicorn.run(AdminApp(sessions), host=args.host, port=args.port, log_level="warning")
    return 0
