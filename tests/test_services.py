"""Services as the task runner opens them, and the codes records may carry."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from assistanteval import world
from assistanteval.contracts import EpisodeStepError, ResetEvidence
from assistanteval import connectors
from assistanteval.connectors import (
    ApiKeyConnector,
    LocalServices,
    McpBearerConnector,
    ServiceSession,
    open_service,
)
from tests.conftest import SERVICES_URL, task_fixture

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = sorted((ROOT / "tasks/fixtures").glob("*/*.json")) + [
    ROOT / "tests/fixtures/marketplace.json"
]
CASES = {path.stem: path for path in FIXTURES if path.stem in world.SERVICES}


def rendered(path: Path, folder: Path) -> Path:
    """A fixture as a run hands it to its service: rendered at ANCHOR."""
    if "{{" not in path.read_text():
        return path
    out = folder / f"rendered-{path.name}"
    out.write_text(json.dumps(task_fixture(path.parent.name, path.stem)))
    return out


@pytest.mark.parametrize(
    "make",
    [lambda: ResetEvidence(True, "SECRET key"), lambda: EpisodeStepError("SECRET key")],
)
def test_record_codes_reject_secret_shaped_text(make):
    with pytest.raises(ValueError, match="Evidence codes"):
        make()


def test_every_service_has_a_task_fixture():
    assert set(CASES) == set(world.SERVICES)


@pytest.fixture
def local(request, tmp_path):
    """Open a service (calendar unless parametrized) on a local session directory."""
    name = getattr(request, "param", "calendar")
    sessions = world.Sessions(world.DirBackend(tmp_path / "sessions"))
    fixture = rendered(CASES[name], tmp_path)
    return lambda kind="openapi_api_key", fixture=fixture: ServiceSession(
        name, fixture, sessions, f"{SERVICES_URL}/{name}", kind
    )


def test_sessions_are_isolated_and_revoked(local):
    first, second = local(), local()
    a, b = first.prepare(), second.prepare()
    assert isinstance(a.connector, ApiKeyConnector)
    assert a.connector.url == f"{SERVICES_URL}/{first.name}"
    assert a.connector.name == world.SERVICES[first.name].title
    assert a.session_id != b.session_id
    assert first.sessions.resolve(a.connector.key, first.name) == a.session_id
    assert first.sessions.resolve(a.connector.key, "other") is None
    assert first.state() == second.state() and isinstance(first.state(), dict)
    assert first.request_log() == ([], True)
    first.release()
    assert first.sessions.resolve(a.connector.key, first.name) is None
    assert second.sessions.resolve(b.connector.key, second.name) == b.session_id
    second.release()


def test_release_rejects_an_unrevoked_key(local, monkeypatch):
    service = local()
    service.prepare()
    monkeypatch.setattr(service.sessions, "revoke", lambda session_id: None)
    with pytest.raises(EpisodeStepError, match=f"{service.name}_revocation_unverified"):
        service.release()


@pytest.mark.parametrize("local", sorted(CASES), indirect=True)
def test_prepare_rejects_a_malformed_fixture(local, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"unexpected": True}))
    with pytest.raises(ValueError):
        local(fixture=bad).prepare()


def test_connectors_are_rest_or_mcp_and_mcp_works_locally(local):
    with pytest.raises(ValueError, match="no carrier_pigeon connector"):
        local("carrier_pigeon")
    service = local("mcp_bearer")
    prepared = service.prepare()
    assert isinstance(prepared.connector, McpBearerConnector)
    assert prepared.connector.url == f"{SERVICES_URL}/{service.name}/mcp"
    status, _, body = world.handle(
        service.sessions,
        "POST",
        f"/{service.name}/mcp",
        "",
        {"authorization": f"Bearer {prepared.connector.key}"},
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).encode(),
        host="services.test",
    )
    assert status == 200 and json.loads(body)["result"]["tools"]
    service.release()


def test_the_runner_reads_requests_in_the_record_shape(tmp_path):
    sessions = world.Sessions(world.DirBackend(tmp_path))
    service = ServiceSession(
        "marketplace",
        CASES["marketplace"],
        sessions,
        f"{SERVICES_URL}/marketplace",
        "openapi_api_key",
    )
    connector = service.prepare().connector
    key = connector.key
    TestClient(world.asgi(service.sessions), base_url=connector.url).get(
        "/v1/listings",
        params={"thread_id": "missing"},
        headers={"Authorization": f"Bearer {key}"},
    )
    (request,), complete = service.request_log()
    assert complete
    assert (request.service, request.operation, request.status) == (
        "marketplace",
        "marketplace.read",
        "ok",
    )
    assert (request.input, request.output, request.error) == (
        {"thread_id": "missing"},
        {"listings": []},
        None,
    )
    assert request.at_ms > 0


def test_open_service_refuses_what_it_cannot_open(tmp_path):
    with pytest.raises(ValueError, match="No service named"):
        open_service("nonexistent", tmp_path / "x.json", "openapi_api_key")


@pytest.fixture
def served(tmp_path):
    local = LocalServices(tmp_path / "sessions")
    yield local
    local.close()


def test_open_service_serves_every_service_on_one_local_server(
    served, monkeypatch, tmp_path
):
    import httpx

    monkeypatch.setattr(connectors, "_local", served)
    service = open_service("marketplace", CASES["marketplace"], "openapi_api_key")
    connector = service.prepare().connector
    assert connector.url == f"{served.url}/marketplace"
    assert connector.url.startswith("http://127.0.0.1:")
    schema = httpx.get(f"{connector.url}/openapi.json").json()
    assert schema["paths"]
    response = httpx.get(
        f"{connector.url}/v1/listings",
        params={"thread_id": "missing"},
        headers={"Authorization": f"Bearer {connector.key}"},
    )
    assert response.json() == {"listings": []}
    # A slash the route does not take redirects within the service's prefix.
    redirect = httpx.get(
        f"{connector.url}/v1/listings/",
        headers={"Authorization": f"Bearer {connector.key}"},
    )
    assert redirect.headers["location"] == f"{connector.url}/v1/listings"
    service.release()
    assert (
        httpx.get(
            f"{connector.url}/v1/listings",
            headers={"Authorization": f"Bearer {connector.key}"},
        ).status_code
        == 401
    )


def test_the_url_given_to_the_assistant_can_differ_from_where_it_listens(tmp_path):
    local = LocalServices(tmp_path, url="http://host.docker.internal:8765/")
    try:
        assert local.url == "http://host.docker.internal:8765"
    finally:
        local.close()


def test_an_mcp_client_lists_tools_over_http(served, tmp_path):
    """What an agent harness does: the official MCP client, streamable HTTP,
    the run's bearer key."""
    import asyncio

    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    service = ServiceSession(
        "contacts",
        rendered(CASES["contacts"], tmp_path),
        served.sessions,
        f"{served.url}/contacts",
        "mcp_bearer",
    )
    connector = service.prepare().connector

    async def run():
        async with (
            httpx2.AsyncClient(
                headers={"Authorization": f"Bearer {connector.key}"}
            ) as http,
            streamable_http_client(connector.url, http_client=http) as (r, w),
            ClientSession(r, w) as session,
        ):
            await session.initialize()
            return (await session.list_tools()).tools

    assert asyncio.run(run())
    service.release()
