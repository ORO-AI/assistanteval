"""The run record: everything saved from one run, in one file.

    runs/<run_id>/run.json   the rendered task, what ran, every turn, click and
                             setup message, each service's requests and start
                             and end state, the simulated user's decisions, and
                             how the run ended

docs/run-record.md is the contract. The runner writes records with save(); the
grader reads them with load().
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from assistanteval import world

UNKNOWN = "unknown"  # an operation no service declares: treated as needing a yes


def canonical(value: object) -> bytes:
    """Same serialization the services use when they hash a payload."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def describe(operation: str) -> tuple[str, str]:
    """(kind, plain words) as the operation's service declares them."""
    service = world.SERVICES.get(operation.partition(".")[0])
    op = (
        next((o for o in service.ops.values() if o.name == operation), None)
        if service
        else None
    )
    return (op.kind, op.plain) if op else (UNKNOWN, "")


def changes_world(operation: str) -> bool:
    return describe(operation)[0] != world.READ


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    text: str
    at_ms: int | None  # when the message was first shown
    nudge: bool = False
    edited_at_ms: int | None = None  # when the product last rewrote it, if it did


@dataclass
class Request:
    service: str
    operation: str
    at_ms: int
    input: object
    output: object = None  # may be None for a refused or failed request
    status: str = "ok"  # "ok", "refused", or "error"
    error: str | None = None
    kind: str | None = None  # from the operation's declaration when not given
    plain: str | None = None

    def __post_init__(self):
        kind, plain = describe(self.operation)
        self.kind = self.kind or kind
        self.plain = plain if self.plain is None else self.plain

    def label(self) -> str:
        return f"{self.operation} {json.dumps(self.input, sort_keys=True)}"


@dataclass
class Service:
    name: str
    start_state: object = None
    end_state: object = None
    requests: list[Request] = field(default_factory=list)


@dataclass
class RunRecord:
    run_id: str
    task_id: str
    product: str
    turns: list[Turn]
    services: list[Service]
    result: dict
    permission_clicks: list[dict] = field(default_factory=list)
    simulated_user: dict | None = None
    setup: list[dict] = field(
        default_factory=list
    )  # connector setup messages and replies: not part of the task conversation
    manifest: dict = field(default_factory=dict)
    task: dict | None = None  # the task the run used, rendered at its anchor

    @property
    def requests(self) -> list[Request]:
        return sorted(
            (r for s in self.services for r in s.requests), key=lambda r: r.at_ms
        )


def save(record: RunRecord, folder: Path) -> Path:
    """Write <folder>/run.json, whole or not at all."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    written = folder / "run.json.tmp"
    written.write_text(json.dumps(asdict(record), indent=1, sort_keys=True) + "\n")
    written.replace(folder / "run.json")
    return folder


def _known(cls, data: dict):
    """Build a dataclass from the fields it knows; extra keys are ignored."""
    return cls(**{f.name: data[f.name] for f in fields(cls) if f.name in data})


def _service(data: dict) -> Service:
    requests = [_known(Request, r) for r in data.get("requests", [])]
    return Service(
        data["name"], data.get("start_state"), data.get("end_state"), requests
    )


def load(folder: Path) -> RunRecord:
    """The record in <folder>/run.json."""
    data = json.loads((Path(folder) / "run.json").read_text())
    record = _known(RunRecord, data)
    record.turns = [_known(Turn, t) for t in record.turns]
    record.services = [_service(s) for s in record.services]
    return record
