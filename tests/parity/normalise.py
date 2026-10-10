"""The shapes the parity goldens store, shared by the recorder and the replay.

Only genuinely random values are dropped: request times. Keys and session
ids never appear in a response.
"""

from __future__ import annotations

import json

KEPT_HEADERS = ("location", "www-authenticate", "allow")


def rest_response(response) -> dict:
    try:
        body = response.json()
    except ValueError:
        body = response.text or None
    headers = {k: response.headers[k] for k in KEPT_HEADERS if k in response.headers}
    if "location" in headers:  # the test client's own host
        headers["location"] = headers["location"].split("://", 1)[-1].partition("/")[2]
    return {"status": response.status_code, "headers": headers, "body": body}


def mcp_result(result) -> dict:
    """A CallToolResult as the client parsed it. A JSON text block is parsed,
    so the text's indentation does not matter."""
    dumped = result.model_dump(mode="json", by_alias=True, exclude_none=True)
    for block in dumped.get("content", []):
        if block.get("type") == "text":
            try:
                block["json"] = json.loads(block["text"])
                del block["text"]
            except ValueError:
                pass
    return dumped


def tool_list(listed) -> list[dict]:
    return [
        tool.model_dump(mode="json", by_alias=True, exclude_none=True)
        for tool in listed.tools
    ]
