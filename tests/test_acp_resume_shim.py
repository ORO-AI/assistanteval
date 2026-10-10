"""The resume shim, around an agent that forgets empty sessions as Hermes does."""

import json
import subprocess
import sys
from pathlib import Path

from assistanteval.harbor import acp_resume_shim

FORGETFUL = """
import json, sys
sessions, count = {"kept": []}, 0  # "kept" was saved by an earlier process
def write(m): sys.stdout.write(json.dumps(m) + "\\n"); sys.stdout.flush()
for line in sys.stdin:
    m = json.loads(line); method, p = m.get("method"), m.get("params") or {}
    if method == "session/new":
        count += 1; sessions[f"new-{count}"] = []; result = {"sessionId": f"new-{count}", "models": {}}
    elif method == "session/load":
        result = {"models": {}} if p["sessionId"] in sessions else {}  # as Hermes answers
    elif method == "session/prompt":
        if p["sessionId"] not in sessions:
            result = {"stopReason": "refusal"}
        else:
            sessions[p["sessionId"]].append(p["prompt"][0]["text"])
            update = {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "heard " + str(len(sessions[p["sessionId"]]))}}
            write({"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": p["sessionId"], "update": update}})
            result = {"stopReason": "end_turn"}
    else:
        result = {}
    write({"jsonrpc": "2.0", "id": m["id"], "result": result})
"""


def converse(tmp_path: Path, messages: list[dict]) -> list[dict]:
    agent = tmp_path / "agent.py"
    agent.write_text(FORGETFUL)
    shim = subprocess.run(
        [sys.executable, acp_resume_shim.__file__, sys.executable, str(agent)],
        input="".join(json.dumps(m) + "\n" for m in messages),
        capture_output=True,
        text=True,
        timeout=20,
    )
    return [json.loads(line) for line in shim.stdout.splitlines()]


def request(id_, method, **params):
    return {"jsonrpc": "2.0", "id": id_, "method": method, "params": params}


def prompt(id_, sid, text):
    return request(
        id_, "session/prompt", sessionId=sid, prompt=[{"type": "text", "text": text}]
    )


def test_resuming_a_forgotten_session_opens_a_new_one_under_the_clients_id(tmp_path):
    out = converse(
        tmp_path,
        [
            request(1, "initialize", protocolVersion=1),
            request(
                2, "session/resume", sessionId="from-setup", cwd="/", mcpServers=[]
            ),
            prompt(3, "from-setup", "move kingsway"),
            prompt(4, "from-setup", "any update?"),
        ],
    )
    replies = {m["id"]: m for m in out if "id" in m}
    assert replies[2]["result"] == {"models": {}}  # resume-shaped, no other session id
    assert replies[3]["result"] == replies[4]["result"] == {"stopReason": "end_turn"}
    updates = [m["params"] for m in out if m.get("method") == "session/update"]
    assert [u["sessionId"] for u in updates] == ["from-setup", "from-setup"]
    assert [u["update"]["content"]["text"] for u in updates] == ["heard 1", "heard 2"]


def test_a_session_the_agent_still_has_is_loaded(tmp_path):
    out = converse(
        tmp_path,
        [
            request(1, "session/resume", sessionId="kept", cwd="/", mcpServers=[]),
            prompt(2, "kept", "hello"),
        ],
    )
    replies = {m["id"]: m for m in out if "id" in m}
    assert replies[1]["result"] == {"models": {}}
    assert replies[2]["result"] == {"stopReason": "end_turn"}
    assert [m["params"]["sessionId"] for m in out if m.get("method")] == ["kept"]
