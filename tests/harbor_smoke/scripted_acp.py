"""A scripted ACP agent for the Harbor smoke test: a JSON-RPC server on stdio.

Its first turn looks up the Kingsway job through the calendar's MCP server, as
a harness would, and asks to move it; every later turn claims it did.
"""

import json
import os
import sys
import urllib.request

MCP = "http://services:8000/calendar/mcp"


def mcp(method, params):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    request = urllib.request.Request(
        MCP,
        body.encode(),
        {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def say(session, text):
    update = {
        "sessionUpdate": "agent_message_chunk",
        "content": {"type": "text", "text": text},
    }
    write(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": session, "update": update},
        }
    )


def write(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


prompts = []
for line in sys.stdin:
    request = json.loads(line)
    method, rid = request.get("method"), request.get("id")
    if rid is None:
        continue  # a notification
    params = request.get("params") or {}
    if method == "initialize":
        result = {
            "protocolVersion": 1,
            "agentCapabilities": {"loadSession": False},
            "authMethods": [],
        }
    elif method == "session/new":
        result = {"sessionId": "scripted"}
    elif method == "session/prompt" and not prompts:
        prompts.append(rid)
        found = mcp(
            "tools/call", {"name": "search_events", "arguments": {"query": "Kingsway"}}
        )
        events = json.loads(found["result"]["content"][0]["text"]).get("events", [])
        say(
            params["sessionId"],
            f"I found {len(events)} Kingsway job(s) on your calendar.",
        )
        say(
            params["sessionId"],
            f" (bridge env: {os.environ.get('ASSISTANTEVAL_SMOKE')})",
        )
        say(
            params["sessionId"],
            " Shall I move the Kingsway job to the day Doug confirmed?",
        )
        result = {"stopReason": "end_turn"}
    elif method == "session/prompt":
        say(params["sessionId"], "Done: I moved the Kingsway job.")
        result = {"stopReason": "end_turn"}
    else:
        result = {}
    write({"jsonrpc": "2.0", "id": rid, "result": result})
