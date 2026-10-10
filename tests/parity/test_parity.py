"""Replay the parity scripts on `assistanteval.world` and compare with the goldens.

The goldens were recorded from an earlier implementation of the services,
built on FastAPI and the MCP SDK. Every difference allowed here is listed,
with its reason:

1. Request log inputs of refused calls: the engine logs the arguments as
   sent. The goldens log them as FastAPI or the MCP SDK parsed them, so
   they can carry null defaults and a datetime printed with a space
   (``2026-10-01 08:00:00-07:00``). Only the grader's judges see these, as
   the attempted request.
2. Request log inputs of successful MCP calls: in the goldens the MCP SDK
   fills every omitted argument with its null default before logging; the
   engine logs what was sent, as over REST. Non-null values are the same.
3. MCP calls the SDK refuses before they reach the service (a wrong type,
   a missing field) are logged as refused, like the same call over REST;
   the goldens do not log them. The assistant sees the same error either
   way.
4. Log entries carry the operation's ``kind`` (the grader's approval class).
5. Calendar and email search match every word of the query in any field
   (for email, the sender or recipient too); in the goldens they match the
   query as one phrase. The goldens' tool and parameter descriptions say
   so; no recorded call's result differs.

The ``amazon`` goldens are the exception: `record_new.py` recorded them from
`assistanteval.world` itself, as there is no earlier implementation with the
same API. They pin the service's behaviour against later change; they prove
no parity.
"""

from __future__ import annotations

import asyncio
import json
import re

import httpx2
import pytest
from fastapi.testclient import TestClient
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from assistanteval import world
from tests.parity import scripts
from tests.parity.normalise import mcp_result, rest_response, tool_list

CASES = scripts.cases()
SERVICES = sorted({service for service, _ in CASES})


def golden(name: str) -> dict:
    return json.loads((scripts.GOLDEN / name).read_text())


def open_session(tmp_path, service: str, fixture: str):
    sessions = world.Sessions(world.DirBackend(tmp_path / "sessions"))
    sid, key = sessions.create(
        service, json.loads((scripts.ROOT / fixture).read_text())
    )
    return sessions, sid, key


def log_of(sessions, sid: str, service: str) -> list[dict]:
    keep = ("operation", "input", "output", "status", "error")
    return [
        {"service": service, **{k: e[k] for k in keep}}
        for e in sessions.read(sid)["log"]
    ]


def loose(value):
    """A logged input without nulls, datetimes written with a T (differences 1, 2)."""
    if isinstance(value, dict):
        return {k: loose(v) for k, v in value.items() if v is not None}
    if isinstance(value, str) and re.fullmatch(
        r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d[+-]\d\d:\d\d", value
    ):
        return value.replace(" ", "T")
    return value


def replay_rest(tmp_path, service: str, fixture: str) -> dict:
    sessions, sid, key = open_session(tmp_path, service, fixture)
    client = TestClient(
        world.asgi(sessions),
        base_url=f"http://{service}.parity.test",
        follow_redirects=False,
    )
    auth = {"Authorization": f"Bearer {key}"}
    calls, outputs, state = [], [], sessions.read(sid)["state"]
    start = state
    for step in scripts.steps(service, fixture):
        if step.get("only") == "mcp":
            continue
        request = scripts.rest_request(service, step, outputs)
        response = client.request(
            request["method"],
            request["path"],
            params=request.get("query") or None,
            headers=auth
            | ({"Content-Type": "application/json"} if "body" in request else {}),
            **({"json": request["json"]} if "json" in request else {}),
            **({"content": request["body"]} if "body" in request else {}),
        )
        result = rest_response(response)
        outputs.append(result["body"] if response.status_code < 300 else None)
        now = sessions.read(sid)["state"]
        calls.append(
            {"request": request, **result, "state": now if now != state else None}
        )
        state = now
    read = scripts.rest_request(service, scripts.call(scripts.first_read(service)), [])[
        "path"
    ]
    wrong = rest_response(client.get(read, headers={"Authorization": "Bearer wrong"}))
    missing = rest_response(client.get(read))
    sessions.revoke(sid)
    return {
        "start_state": start,
        "calls": calls,
        "log": log_of(sessions, sid, service),
        "wrong_key": wrong,
        "no_key": missing,
        "revoked": rest_response(client.get(read, headers=auth)),
    }


SDK_REFUSED = re.compile(r"validation errors? for|Unknown tool")


def replay_mcp(tmp_path, service: str, fixture: str) -> dict:
    sessions, sid, key = open_session(tmp_path, service, fixture)
    calls, outputs, sizes = [], [], []
    start = sessions.read(sid)["state"]

    async def run():
        transport = httpx2.ASGITransport(app=world.asgi(sessions))
        async with (
            httpx2.AsyncClient(
                transport=transport, headers={"Authorization": f"Bearer {key}"}
            ) as http,
            streamable_http_client(
                f"http://{service}-mcp.parity.test/mcp", http_client=http
            ) as (r, w),
            ClientSession(r, w) as session,
        ):
            await session.initialize()
            listed = tool_list(await session.list_tools())
            state = start
            for step in scripts.steps(service, fixture):
                if step.get("only") == "rest":
                    continue
                args = scripts.resolve(step["args"], outputs)
                result = mcp_result(await session.call_tool(step["tool"], args))
                outputs.append(result.get("structuredContent"))
                now = sessions.read(sid)["state"]
                calls.append(
                    {
                        "tool": step["tool"],
                        "args": args,
                        "result": result,
                        "state": now if now != state else None,
                    }
                )
                sizes.append(len(sessions.read(sid)["log"]))
                state = now
            return listed

    tools = asyncio.run(run())
    sessions.revoke(sid)
    status, headers, _ = wire(
        sessions, service, key, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    )
    return {
        "start_state": start,
        "tools": tools,
        "calls": calls,
        "sizes": sizes,
        "log": log_of(sessions, sid, service),
        "revoked": (status, headers),
    }


def wire(sessions, service: str, key: str | None, body, headers=None, path="/mcp"):
    return world.handle(
        sessions,
        "POST",
        path,
        "",
        {
            "accept": "application/json, text/event-stream",
            "content-type": "application/json",
            **({"authorization": f"Bearer {key}"} if key else {}),
            **(headers or {}),
        },
        json.dumps(body).encode(),
        host=f"{service}-mcp.assistanteval.example.test",
    )


def as_recorded(response) -> dict:
    status, headers, payload = response
    headers = {k.lower(): v for k, v in headers.items()}
    keep = ("content-type", "www-authenticate", "allow", "mcp-session-id")
    try:
        body = json.loads(payload) if payload else None
    except ValueError:
        body = payload.decode()
    return {
        "status": status,
        "headers": {k: headers[k] for k in keep if k in headers},
        "body": body,
    }


@pytest.mark.parametrize(
    "service, fixture", CASES, ids=[scripts.case_id(*c) for c in CASES]
)
def test_rest_matches_the_golden(service, fixture, tmp_path):
    old = golden(f"{scripts.case_id(service, fixture)}.json")["rest"]
    new = replay_rest(tmp_path, service, fixture)
    assert new["start_state"] == old["start_state"]
    for number, (was, now) in enumerate(zip(old["calls"], new["calls"], strict=True)):
        assert now == was, f"call {number}: {was['request']}"
    for key in ("wrong_key", "no_key", "revoked"):
        assert new[key] == old[key]
    assert len(new["log"]) == len(old["log"])
    for was, now in zip(old["log"], new["log"]):
        if was["status"] == "ok":
            assert now == was
        else:  # difference 1
            assert now | {"input": loose(now["input"])} == was | {
                "input": loose(was["input"])
            }


@pytest.mark.parametrize(
    "service, fixture", CASES, ids=[scripts.case_id(*c) for c in CASES]
)
def test_mcp_matches_the_golden(service, fixture, tmp_path):
    old = golden(f"{scripts.case_id(service, fixture)}.json")["mcp"]
    new = replay_mcp(tmp_path, service, fixture)
    assert new["start_state"] == old["start_state"]
    assert new["tools"] == old["tools"]
    for number, (was, now) in enumerate(zip(old["calls"], new["calls"], strict=True)):
        assert now == was, f"call {number}: {was['tool']} {was['args']}"
    assert new["revoked"] == (401, {"WWW-Authenticate": "Bearer"})
    # Difference 3: calls the SDK refuses before invoke are logged.
    sdk_refused = [
        SDK_REFUSED.search(json.dumps(c["result"])) is not None for c in old["calls"]
    ]
    logged, before = [], 0
    for size, refused_early in zip(new["sizes"], sdk_refused):
        entries = new["log"][before:size]
        before = size
        assert len(entries) == 1
        if refused_early:
            assert entries[0]["status"] == "refused"
        else:
            logged += entries
    assert len(logged) == len(old["log"])
    for was, now in zip(old["log"], logged):  # differences 1 and 2
        sent = was["input"]
        if (
            was["status"] != "ok"
        ):  # the SDK's defaults, e.g. "subject": "", were never sent
            sent = {k: v for k, v in sent.items() if k in now["input"]}
        assert now | {"input": loose(now["input"])} == was | {"input": loose(sent)}


@pytest.mark.parametrize("service", SERVICES)
def test_openapi_matches_the_golden(service, tmp_path):
    sessions = world.Sessions(world.DirBackend(tmp_path))
    client = TestClient(world.asgi(sessions), base_url=f"http://{service}.parity.test")
    assert client.get("/openapi.json").json() == golden(f"{service}.openapi.json")


def replay_wire(tmp_path, service: str) -> dict:
    fixture = next(f for s, f in CASES if s == service)
    sessions, _, key = open_session(tmp_path, service, fixture)
    version = {"mcp-protocol-version": "2025-06-18"}
    new = {}
    for request in scripts.wire_requests(service):
        headers = {} if request["body"].get("method") == "initialize" else version
        new[request["name"]] = as_recorded(
            wire(
                sessions, service, key, request["body"], request.get("headers", headers)
            )
        )
    tools_list = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    new["no key"] = as_recorded(wire(sessions, service, None, tools_list))
    new["wrong key"] = as_recorded(
        wire(sessions, service, "wrong", tools_list, version)
    )
    new["other path"] = as_recorded(
        wire(sessions, service, key, tools_list, version, path="/other")
    )
    return new


@pytest.mark.parametrize("service", SERVICES)
def test_mcp_wire_matches_the_golden(service, tmp_path):
    old = golden(f"{service}.mcp-wire.json")
    new = replay_wire(tmp_path, service)
    assert set(new) == set(old)
    for name in old:
        assert new[name] == old[name], name
