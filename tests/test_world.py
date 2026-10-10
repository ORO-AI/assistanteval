"""The service engine: sessions, the request log, and the HTTP handler.

What each service answers is pinned by tests/parity; these check what the
engine guarantees for every service.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from assistanteval import world
from tests.conftest import task_fixture

ROOT = Path(__file__).resolve().parents[1]
CONTACTS = {"contacts": [{"id": "alex", "name": "Alex Rowan", "phone": "A-0142"}]}
MARKETPLACE = json.loads((ROOT / "tests/fixtures/marketplace.json").read_text())
REPLY = {"thread_id": "thr-011", "message": "Still available."}


@pytest.fixture
def sessions(tmp_path):
    return world.Sessions(world.DirBackend(tmp_path / "sessions"))


def rest(sessions, service):
    return TestClient(
        world.asgi(sessions),
        base_url=f"https://{service}.example.test",
        follow_redirects=False,
    )


def mcp(sessions, service, key, method, params=None, **extra):
    body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
    status, headers, payload = world.handle(
        sessions,
        "POST",
        "/mcp",
        "",
        {"authorization": f"Bearer {key}", **extra},
        json.dumps(body).encode(),
        host=f"{service}-mcp.example.test",
    )
    return status, json.loads(payload) if payload else None


def test_sessions_are_isolated_by_key_and_service(sessions):
    a_sid, a_key = sessions.create("contacts", CONTACTS)
    b_sid, b_key = sessions.create(
        "contacts", {"contacts": [{**CONTACTS["contacts"][0], "phone": "B-0186"}]}
    )
    client = rest(sessions, "contacts")

    def phone(key):
        response = client.get(
            "/v1/contacts",
            params={"name": "Alex"},
            headers={"Authorization": f"Bearer {key}"},
        )
        return (
            response.json()["contacts"][0]["phone"]
            if response.status_code == 200
            else response.status_code
        )

    assert (phone(a_key), phone(b_key)) == ("A-0142", "B-0186")
    assert sessions.resolve(a_key, "email") is None
    assert (
        rest(sessions, "email")
        .get(
            "/v1/messages",
            params={"query": "x"},
            headers={"Authorization": f"Bearer {a_key}"},
        )
        .status_code
        == 401
    )
    assert [len(sessions.read(s)["log"]) for s in (a_sid, b_sid)] == [1, 1]
    stored = json.dumps([sessions.read(a_sid), sessions.read(b_sid)])
    assert a_key not in stored and b_key not in stored
    sessions.revoke(a_sid)
    assert (phone(a_key), phone(b_key)) == (401, "B-0186")
    assert sessions.resolve(a_sid + ".forged", "contacts") is None


def test_an_expired_session_refuses_its_key(tmp_path):
    sessions = world.Sessions(world.DirBackend(tmp_path), ttl_seconds=-1)
    _, key = sessions.create("contacts", CONTACTS)
    assert sessions.resolve(key, "contacts") is None


def test_a_log_entry_is_the_run_records_request_and_a_refusal_changes_nothing(sessions):
    sid, key = sessions.create("marketplace", MARKETPLACE)
    client = rest(sessions, "marketplace")
    auth = {"Authorization": f"Bearer {key}"}
    assert (
        client.post(
            "/v1/threads/missing/messages", json={"message": "Hi"}, headers=auth
        ).status_code
        == 404
    )
    assert sessions.read(sid)["state"] == sessions.read(sid)["start_state"]
    assert (
        client.post(
            "/v1/threads/thr-011/messages",
            json={"message": REPLY["message"]},
            headers=auth,
        ).status_code
        == 201
    )
    first, second = sessions.read(sid)["log"]
    assert set(first) == {
        "operation",
        "kind",
        "at_ms",
        "input",
        "output",
        "status",
        "error",
    }
    assert (first["status"], first["error"], first["output"]) == (
        "refused",
        "Marketplace thread not found",
        None,
    )
    assert (second["operation"], second["kind"], second["status"]) == (
        "marketplace.reply",
        "visible_to_others",
        "ok",
    )
    assert second["input"] == REPLY
    assert first["at_ms"] <= second["at_ms"]


def test_a_path_parameter_wins_over_the_same_field_in_the_body(sessions):
    sid, key = sessions.create("marketplace", MARKETPLACE)
    reply = rest(sessions, "marketplace").post(
        "/v1/threads/thr-011/messages",
        json=REPLY | {"thread_id": "thr-012"},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert reply.status_code == 201 and reply.json()["thread"]["id"] == "thr-011"
    assert sessions.read(sid)["log"][-1]["input"] == REPLY


def test_rest_and_mcp_log_the_same_request(tmp_path):
    logs = []
    for transport in ("rest", "mcp"):
        sessions = world.Sessions(world.DirBackend(tmp_path / transport))
        sid, key = sessions.create("marketplace", MARKETPLACE)
        if transport == "rest":
            rest(sessions, "marketplace").post(
                "/v1/threads/thr-011/messages",
                json={"message": REPLY["message"]},
                headers={"Authorization": f"Bearer {key}"},
            )
        else:
            mcp(
                sessions,
                "marketplace",
                key,
                "tools/call",
                {"name": "reply_to_thread", "arguments": REPLY},
            )
        logs.append(
            [
                {k: v for k, v in e.items() if k != "at_ms"}
                for e in sessions.read(sid)["log"]
            ]
        )
    assert logs[0] == logs[1] and len(logs[0]) == 1


def test_a_broken_state_fails_closed_without_leaking_it(sessions, caplog):
    sid, key = sessions.create("contacts", CONTACTS)
    sessions.update(
        sid, lambda s: s.update(state={"contacts": [{"id": "SECRET-VALUE"}]})
    )
    response = rest(sessions, "contacts").get(
        "/v1/contacts",
        params={"name": "Alex"},
        headers={"Authorization": f"Bearer {key}"},
    )
    assert (response.status_code, response.json()) == (
        503,
        {"detail": "Contacts fixture unavailable"},
    )
    status, body = mcp(
        sessions,
        "contacts",
        key,
        "tools/call",
        {"name": "find_contacts", "arguments": {"name": "Alex"}},
    )
    assert body["result"]["content"][0]["text"].endswith("Contacts fixture unavailable")
    entries = sessions.read(sid)["log"]
    assert [(e["status"], e["error"]) for e in entries] == [
        ("error", "Contacts fixture unavailable")
    ] * 2
    assert (
        "SECRET"
        not in response.text + json.dumps(body) + json.dumps(entries) + caplog.text
    )


def test_a_session_revoked_mid_request_is_unauthorized(sessions, monkeypatch):
    sid, key = sessions.create("flights", task_fixture("flight-before-ben", "flights"))
    update = sessions.update

    def revoke_first(session_id, change):
        update(session_id, lambda s: s.update(revoked=True))
        return update(session_id, change)

    monkeypatch.setattr(sessions, "update", revoke_first)
    response = rest(sessions, "flights").post(
        "/v1/changes/priya-return-1045", headers={"Authorization": f"Bearer {key}"}
    )
    assert (response.status_code, response.headers["www-authenticate"]) == (
        401,
        "Bearer",
    )
    assert (
        mcp(sessions, "flights", key, "tools/call", {"name": "list_alternatives"})[0]
        == 401
    )
    assert sessions.read(sid)["log"] == []


def test_a_stale_write_conflicts_and_update_retries(sessions):
    sid, _ = sessions.create("contacts", CONTACTS)
    session, etag = sessions.backend.get(sid)
    sessions.revoke(sid)
    with pytest.raises(world.Conflict):
        sessions.backend.put(sid, session, etag)
    with pytest.raises(world.Conflict):
        sessions.backend.put(sid, session, None)  # creating an existing session
    calls = []

    def change(s):
        calls.append(1)
        if len(calls) == 1:  # someone else writes between this read and write
            other, other_etag = sessions.backend.get(sid)
            sessions.backend.put(
                sid, other | {"expires_at": other["expires_at"] + 1}, other_etag
            )
        s["log"].append("x")

    sessions.update(sid, change)
    assert len(calls) == 2 and sessions.read(sid)["log"] == ["x"]


MALFORMED_MCP = (
    b"{not json",
    b"[1]",
    json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "find_contacts", "arguments": ["Alex"]},
        }
    ).encode(),
)


def test_malformed_mcp_requests_are_refused_without_a_call(sessions):
    sid, key = sessions.create("contacts", CONTACTS)
    client = TestClient(
        world.asgi(sessions), base_url="https://contacts-mcp.example.test"
    )
    auth = {"Authorization": f"Bearer {key}"}
    for body in MALFORMED_MCP:
        assert client.post("/mcp", content=body, headers=auth).status_code == 400
    assert client.get("/mcp", headers=auth).status_code == 405
    assert sessions.read(sid)["log"] == []


def test_malformed_mcp_requests_are_refused_under_python_dash_o(tmp_path):
    """`python -O` strips asserts: the request checks must not be asserts."""
    code = """if True:
        import sys
        from tests.test_world import CONTACTS, MALFORMED_MCP, world
        sessions = world.Sessions(world.DirBackend(sys.argv[1]))
        _, key = sessions.create("contacts", CONTACTS)
        for body in MALFORMED_MCP:
            status, _, payload = world.handle(
                sessions, "POST", "/mcp", "", {"authorization": f"Bearer {key}"},
                body, host="contacts-mcp.example.test",
            )
            print(status, payload.decode())
        """
    run = subprocess.run(
        [sys.executable, "-O", "-c", code, str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    error = {"code": -32700, "message": "Parse error"}
    refused = {"jsonrpc": "2.0", "id": "server-error", "error": error}
    lines = [line.split(" ", 1) for line in run.stdout.splitlines()]
    assert [(s, json.loads(body)) for s, body in lines] == [("400", refused)] * 3


def test_local_path_prefix_routing_serves_rest_and_mcp(sessions):
    sid, key = sessions.create("contacts", CONTACTS)
    client = TestClient(world.asgi(sessions))
    auth = {"Authorization": f"Bearer {key}"}
    assert (
        client.get(
            "/contacts/v1/contacts", params={"name": "Alex"}, headers=auth
        ).status_code
        == 200
    )
    listed = client.post(
        "/contacts/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers=auth,
    )
    assert [t["name"] for t in listed.json()["result"]["tools"]] == ["find_contacts"]
    assert client.get("/nothing/v1").status_code == 404


def test_every_operation_declares_the_graders_kind_and_plain_words():
    for svc in world.SERVICES.values():
        for op in svc.ops.values():
            assert op.name.startswith(f"{svc.name}.")
            assert op.kind in (
                world.READ,
                world.PRIVATE_NEW,
                world.PRIVATE_CHANGE,
                world.VISIBLE,
                world.MONEY,
            )
            # Every change has plain words for wherever it is named, such as a
            # task's simulated user approving it ("go ahead and add it to the
            # cart", never "amazon.cart_add"). Whether it needs a yes is graded
            # by its kind, not by these words.
            assert bool(op.plain) == (op.kind != world.READ), op.name


def test_sessions_outlive_the_runs_hard_deadline(tmp_path):
    """A connector must not start refusing a run that is still inside its
    deadline."""
    from assistanteval import run_task

    sessions = world.Sessions(world.DirBackend(tmp_path))
    assert sessions.ttl > run_task.RUN_DEADLINE_SECONDS
