"""Make an ACP agent's sessions resumable by acpx when the agent forgets them.

    python3 acp_resume_shim.py <agent command...>

acpx creates a session with one agent process (Harbor's bridge runs ``acpx
sessions ensure``) and resumes it from a new process at the first prompt.
Hermes keeps an empty session only in memory, and resuming an id it does not
know starts a session under a different id, so every prompt to the original id
is refused. This shim sits on the agent's stdio: it answers a
``session/resume`` by loading the session, or by opening a new one when the
agent does not have it, and from then on translates between the two ids. Every
other message passes through unchanged. Standard library only, so it runs in
any task container. With ``ACP_SHIM_STDERR`` set, the agent's stderr (where
Hermes logs MCP connections) is appended to that file instead.
"""

import json
import os
import subprocess
import sys
import threading

lock = threading.Lock()
alias: dict[str, str] = {}  # the client's session id -> the agent's
pending: dict[str, tuple] = {}  # a shim request id -> (step, client request)
answered = threading.Event()  # a resume's answer has gone back to the client


def send(stream, message: dict) -> None:
    with lock:
        stream.write(json.dumps(message) + "\n")
        stream.flush()


def renamed(message: dict, names: dict[str, str]) -> dict:
    params = message.get("params")
    if isinstance(params, dict) and params.get("sessionId") in names:
        message = {
            **message,
            "params": {**params, "sessionId": names[params["sessionId"]]},
        }
    return message


def from_client(agent) -> None:
    for line in sys.stdin:
        message = json.loads(line)
        if message.get("method") == "session/resume" and "id" in message:
            params = message["params"]
            ask = f"shim-load-{message['id']}"
            pending[ask] = ("load", message)
            answered.clear()
            send(
                agent.stdin,
                {
                    "jsonrpc": "2.0",
                    "id": ask,
                    "method": "session/load",
                    "params": params,
                },
            )
            answered.wait()  # what follows may name the session: map it first
            continue
        send(agent.stdin, renamed(message, alias))
    agent.stdin.close()


def from_agent(agent) -> None:
    back = {}  # the agent's session id -> the client's
    for line in agent.stdout:
        message = json.loads(line)
        step = pending.pop(message.get("id"), None) if "method" not in message else None
        if step is None:
            send(sys.stdout, renamed(message, back))
            continue
        kind, request = step
        sid = request["params"]["sessionId"]
        # Loaded: the result names the session's state.
        if kind == "load" and message.get("result"):
            reply = message["result"]
        elif kind == "load":  # an error, null or {}: the agent does not have it
            ask = f"shim-new-{request['id']}"
            pending[ask] = ("new", request)
            params = {k: v for k, v in request["params"].items() if k != "sessionId"}
            send(
                agent.stdin,
                {
                    "jsonrpc": "2.0",
                    "id": ask,
                    "method": "session/new",
                    "params": params,
                },
            )
            continue
        elif "error" in message:
            send(
                sys.stdout,
                {"jsonrpc": "2.0", "id": request["id"], "error": message["error"]},
            )
            answered.set()
            continue
        else:
            alias[sid] = message["result"]["sessionId"]
            back[alias[sid]] = sid
            reply = {k: v for k, v in message["result"].items() if k != "sessionId"}
        send(sys.stdout, {"jsonrpc": "2.0", "id": request["id"], "result": reply})
        answered.set()


def main() -> int:
    log = os.environ.get("ACP_SHIM_STDERR")
    agent = subprocess.Popen(
        sys.argv[1:],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=open(log, "a") if log else None,
        text=True,
    )
    threading.Thread(target=from_client, args=(agent,), daemon=True).start()
    from_agent(agent)
    return agent.wait()


if __name__ == "__main__":
    sys.exit(main())
