"""Services in another process: the admin API and the runner's client for it."""

import json

import pytest
from fastapi.testclient import TestClient

from assistanteval import world
from assistanteval.connectors import ServiceSession
from assistanteval.remote import AdminApp, RemoteSessions
from tests.test_services import CASES, rendered

URL = "http://services:8000"


@pytest.fixture
def served(tmp_path):
    return TestClient(
        AdminApp(world.Sessions(world.DirBackend(tmp_path))), base_url=URL
    )


def admin_call(client, token):
    def call(method, path, body):
        response = client.request(
            method,
            f"/_admin{path}",
            content=None if body is None else json.dumps(body),
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        return response.json() if response.content else None

    return call


def test_the_first_claim_sets_the_admin_token_and_later_claims_are_refused(served):
    assert served.post("/_admin/claim", json={"token": "runner"}).status_code == 204
    assert served.post("/_admin/claim", json={"token": "assistant"}).status_code == 409
    fixture = json.loads(CASES["contacts"].read_text())
    body = {"service": "contacts", "fixture": fixture}
    for auth in ({}, {"Authorization": "Bearer assistant"}):
        assert (
            served.post("/_admin/sessions", json=body, headers=auth).status_code == 401
        )
    assert served.get("/_admin/sessions/x").status_code == 401


@pytest.mark.parametrize("token", ["", None, 5])
def test_a_claim_without_a_token_is_malformed_and_claims_nothing(served, token):
    assert served.post("/_admin/claim", json={"token": token}).status_code == 400
    assert served.post("/_admin/claim", json={"token": "runner"}).status_code == 204


def test_a_service_session_runs_over_the_admin_api(served):
    """`ServiceSession`, as the runner uses it, with sessions in another process."""
    served.post("/_admin/claim", json={"token": "runner"})
    sessions = RemoteSessions(admin_call(served, "runner"))
    service = ServiceSession(
        "marketplace",
        CASES["marketplace"],
        sessions,
        f"{URL}/marketplace",
        "openapi_api_key",
    )
    connector = service.prepare().connector
    start = service.state()
    # A harness that cannot send a key reaches the service's latest session.
    keyless = served.get("/marketplace/v1/listings", params={"thread_id": "missing"})
    assert keyless.json() == {"listings": []}
    keyed = served.get(
        "/marketplace/v1/listings",
        params={"thread_id": "missing"},
        headers={"Authorization": f"Bearer {connector.key}"},
    )
    assert keyed.json() == {"listings": []}
    requests, complete = service.request_log()
    assert complete and [r.operation for r in requests] == ["marketplace.read"] * 2
    assert service.state() == start
    service.release()  # revokes, and proves the key no longer opens it
    assert served.get("/marketplace/v1/listings").status_code == 401


def test_a_keyless_request_to_a_service_with_no_session_is_refused(served):
    assert served.get("/contacts/v1/contacts").status_code == 401


def test_a_harness_started_before_the_run_can_list_the_tools(served, tmp_path):
    """Harbor starts the harness before the services have a session; one that
    found no tools would park the server for the whole run."""

    def mcp(method, **params):
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        return served.post("/calendar/mcp", json=body).json()

    assert mcp("initialize", protocolVersion="2025-06-18")["result"]["serverInfo"]
    names = {t["name"] for t in mcp("tools/list")["result"]["tools"]}
    assert "search_events" in names
    early = mcp("tools/call", name="search_events", arguments={"query": "x"})
    assert early["result"]["isError"]
    served.post("/_admin/claim", json={"token": "runner"})
    fixture = json.loads(rendered(CASES["calendar"], tmp_path).read_text())
    open_ = admin_call(served, "runner")
    open_("POST", "/sessions", {"service": "calendar", "fixture": fixture})
    later = mcp("tools/call", name="search_events", arguments={"query": "x"})
    assert not later["result"].get("isError")


@pytest.mark.parametrize("body", [b"not json", b"[1, 2]"])
def test_malformed_bodies_get_a_protocol_error_not_a_crash(served, body):
    """Keyless MCP before any session, as a probing harness sends it."""
    mcp = served.post("/calendar/mcp", content=body)
    assert mcp.status_code == 400 and mcp.json()["error"]["message"] == "Parse error"
    served.post("/_admin/claim", json={"token": "runner"})
    for path, admin in (
        ("/_admin/claim", body),
        ("/_admin/sessions", b'{"service": "calendar"}'),
    ):
        response = served.post(
            path, content=admin, headers={"Authorization": "Bearer runner"}
        )
        assert response.status_code in (400, 409), (path, response.status_code)
