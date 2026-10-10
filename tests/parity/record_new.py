"""Record goldens for a service with no earlier implementation to record.

The other services' goldens were recorded from an earlier implementation.
`amazon` has none with the same API, so its goldens come from
`assistanteval.world` itself: they pin its current behaviour so a later
change shows up call by call, and prove no parity with anything else.
Re-record only on purpose, after reading the diff::

    uv run python -m tests.parity.record_new amazon
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from assistanteval import world
from tests.parity import scripts
from tests.parity.test_parity import (
    CASES,
    SDK_REFUSED,
    replay_mcp,
    replay_rest,
    replay_wire,
)

NEW_SERVICES = {"amazon"}


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def main(services: list[str]) -> None:
    if not set(services) <= NEW_SERVICES:
        raise SystemExit(
            f"Only {sorted(NEW_SERVICES)} are recorded from assistanteval.world"
        )
    root = Path(tempfile.mkdtemp())
    try:
        for service in services:
            for s, fixture in CASES:
                if s != service:
                    continue
                case = scripts.case_id(service, fixture)
                rest = replay_rest(root / case / "rest", service, fixture) | {
                    "complete": True
                }
                mcp = replay_mcp(root / case / "mcp", service, fixture)
                # As the other goldens do, leave out calls a client refused before the service.
                refused = [
                    SDK_REFUSED.search(json.dumps(c["result"])) is not None
                    for c in mcp["calls"]
                ]
                log = [
                    e for e, early in zip(mcp["log"], refused, strict=True) if not early
                ]
                mcp = {
                    "start_state": mcp["start_state"],
                    "tools": mcp["tools"],
                    "calls": mcp["calls"],
                    "log": log,
                    "complete": True,
                    "revoked": {"status": mcp["revoked"][0]},
                }
                golden = {
                    "service": service,
                    "fixture": fixture,
                    "rest": rest,
                    "mcp": mcp,
                }
                write(scripts.GOLDEN / f"{case}.json", golden)
                print(case)
            write(
                scripts.GOLDEN / f"{service}.mcp-wire.json",
                replay_wire(root / f"{service}-wire", service),
            )
            client = TestClient(
                world.asgi(world.Sessions(world.DirBackend(root / "openapi"))),
                base_url=f"http://{service}.parity.test",
            )
            write(
                scripts.GOLDEN / f"{service}.openapi.json",
                client.get("/openapi.json").json(),
            )
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(NEW_SERVICES))
