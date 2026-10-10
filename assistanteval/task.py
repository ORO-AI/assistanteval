"""A task file as a model: load it and check it.

    task = load(Path("tasks/<id>.json"))

The model checks that a task is complete and consistent with itself. load()
also checks it against this checkout: its operations are declared by its
services, and its fixtures, rendered at a sample anchor (assistanteval.dates),
load in them. The runner renders the task at the run's start; the record keeps
the rendered task. tasks/README.md describes the fields.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, model_validator

from assistanteval import dates, world

_STAMP = re.compile(r"\d{4}-\d\d-\d\dT[\d:]+(?:[+-]\d\d:\d\d|Z)")
# The local wall time load() renders a task's dates at, to check its fixtures.
SAMPLE_ANCHOR = "2026-09-29T09:00"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Connector(_Model):
    service: str
    fixture: str  # a path from the repository root
    operations: list[str]
    start_state_sha256: str | None = None  # unused: the record keeps the start state


class WorldFact(_Model):
    id: str
    service: str
    value: str


class Environment(_Model):
    connectors: list[Connector]
    timezone: str
    # Named dates, as rules relative to the run's start (assistanteval.dates).
    dates: dict[str, str | dict[str, int | bool]] = {}
    # A fixed run window: allowed in a record's embedded task so such
    # records grade, and refused by load() in a task file.
    run_between: tuple[str, str] | None = None
    facts: list[WorldFact]
    assistant_memory: Literal["none"]
    note: str | None = None


class Fact(_Model):
    id: str
    value: str
    reveal_when: str
    needed_before: list[str] = []


class Approval(_Model):
    operation: str
    answer: Literal["yes", "no"]
    only_if: str | None = None
    if_not: str = Field(min_length=1)  # the user's reason, so the assistant can revise


class SimulatedUser(_Model):
    persona: str
    voice: str
    goal: str
    facts: list[Fact]
    approvals: list[Approval]
    otherwise: str


class EndState(_Model):
    service: str
    unchanged: bool = False
    path: str | None = None
    equals: Any = None
    contains: Any = None
    matches: str | None = None  # a case-insensitive regular expression

    @model_validator(mode="after")
    def _one_check(self):
        if self.unchanged == bool({"path"} & self.model_fields_set):
            raise ValueError("an end state is `unchanged` or has a `path`")
        checks = {"equals", "contains", "matches"} & self.model_fields_set
        if self.path and len(checks) != 1:
            raise ValueError("a path end state has `equals`, `contains` or `matches`")
        if self.matches is not None:
            try:
                re.compile(self.matches)
            except re.error as exc:
                raise ValueError(f"`matches` is not a regular expression: {exc}")
        return self


class Question(_Model):
    id: str
    text: str
    expect: bool = True
    criteria: dict[str, str] | None = None


class Rules(_Model):
    forbidden_operations: list[str]
    required_operations: list[str]
    end_state: list[EndState]
    questions: list[Question]
    # Operations the assistant should do without asking first (a twin where
    # the user already said go ahead): asking to approve one fails the task.
    no_approval_request: list[str] = []


class Task(_Model):
    id: str = Field(pattern=r"[a-z0-9][a-z0-9-]*")
    title: str
    category: Literal["permission", "clarification", "proactivity", "recovery"]
    status: Literal["runnable", "needs_environment"]
    environment: Environment
    opening_message: str  # the only field the assistant sees
    key_fact: str
    good_behavior: str
    observed: dict
    simulated_user: SimulatedUser
    rules: Rules
    twin: dict | None

    @model_validator(mode="after")
    def _consistent(self):
        env, user = self.environment, self.simulated_user
        services = {c.service for c in env.connectors}
        available = {op for c in env.connectors for op in c.operations}
        # A rule about an action the assistant cannot take proves nothing.
        checked = {a.operation for a in user.approvals}
        checked |= {op for f in user.facts for op in f.needed_before}
        checked |= {*self.rules.forbidden_operations, *self.rules.required_operations}
        if checked - available:
            raise ValueError(f"not in any connector: {sorted(checked - available)}")
        if {r.service for r in self.rules.end_state} - services or {
            f.service for f in env.facts
        } - services:
            raise ValueError("an end state or world fact names a service not connected")
        # Each fact lives in one place: the world (look it up) or the user (ask).
        ids = [f.id for f in env.facts + user.facts]
        if len(ids) != len(set(ids)):
            raise ValueError("fact ids repeat")
        try:
            ZoneInfo(env.timezone)
        except (KeyError, ValueError):
            raise ValueError(f"no time zone {env.timezone}") from None
        return self

    def dump(self) -> dict:
        """The task as JSON data, with only the fields the file set."""
        return self.model_dump(mode="json", exclude_unset=True)


def load(path: Path, anchor: datetime | None = None) -> Task:
    """A task file, checked against the declared services and its fixtures.

    Its dates are templates: the task and every fixture must render at
    `anchor` (default SAMPLE_ANCHOR in the task's zone), and each rendered
    fixture must load in its service. The task is returned as written."""
    path = Path(path)
    task = Task.model_validate_json(path.read_bytes())
    if task.environment.run_between is not None:
        raise ValueError(
            "run_between is gone: a task's dates are rendered at run start"
        )
    zone = ZoneInfo(task.environment.timezone)
    anchor = anchor or datetime.fromisoformat(SAMPLE_ANCHOR).replace(tzinfo=zone)
    try:
        # Fixture paths are relative to the task folder's parent.
        rendered = dates.render(task.dump(), path.resolve().parents[1], anchor)
    except ValueError as error:
        raise ValueError(f"{path.name} does not render: {error}") from None
    Task.model_validate(rendered.task)  # still a task once rendered
    for c in task.environment.connectors:
        fixture = rendered.fixtures[c.service].decode()
        if task.status == "runnable":
            service = world.SERVICES.get(c.service)
            if service is None:
                raise ValueError(f"runnable, but no service declares {c.service}")
            unknown = set(c.operations) - {op.name for op in service.ops.values()}
            if unknown:
                raise ValueError(f"{c.service} declares no {sorted(unknown)}")
            service.state.model_validate_json(fixture)
        # The world has one time zone: every fixture time is written in it.
        for stamp in _STAMP.findall(fixture):
            moment = datetime.fromisoformat(stamp)
            if moment.utcoffset() != moment.astimezone(zone).utcoffset():
                raise ValueError(f"{c.fixture}: {stamp} is not in {zone}")
    return task
