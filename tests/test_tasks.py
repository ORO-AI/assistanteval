import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tests.conftest import catalog
from pydantic import ValidationError

from assistanteval.grader import AnswerStore, Ladder, grade
from assistanteval.run_record import Request, Service, Turn
from assistanteval.task import EndState, load
from tests import records

TASKS = sorted(Path(__file__).parents[1].joinpath("tasks").glob("*.json"))


def broken(path: Path, tmp_path, change) -> Path:
    data = json.loads(path.read_text())
    change(data)
    out = catalog(tmp_path) / path.name  # fixtures resolve from the catalog
    out.write_text(json.dumps(data))
    return out


ROOT = Path(__file__).parents[1]
KINGSWAY = ROOT / "tasks/kingsway-reschedule-crew.json"


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.pop("key_fact"),
        lambda d: d.update(extra=1),
        lambda d: d.update(category="speed"),
        lambda d: d["environment"].update(assistant_memory="last week's chat"),
        # A rule about an action the assistant cannot take proves nothing.
        lambda d: d["rules"]["forbidden_operations"].append("email.send"),
        lambda d: d["rules"]["end_state"].append(
            {"service": "email", "unchanged": True}
        ),
        lambda d: d["rules"]["end_state"].append(
            {"service": "calendar", "path": "events"}
        ),
        lambda d: d["simulated_user"]["approvals"][0].pop("if_not"),
        # Each fact lives in one place: the world (look it up) or the user (ask).
        lambda d: d["simulated_user"]["facts"].append(
            {**d["environment"]["facts"][0], "reveal_when": "x"}
        ),
        # A world has no fixed window: its dates are rendered at the run's start.
        lambda d: d["environment"].update(
            run_between=["2026-09-27T00:00:00-07:00", "2026-09-28T00:00:00-07:00"]
        ),
        # Every placeholder names a date and a format the renderer knows.
        lambda d: d.update(opening_message="move it to {{thursday}}"),
        lambda d: d.update(opening_message="move it to {{job:Weekdayy}}"),
        lambda d: d.update(opening_message="move it to {{job"),
        lambda d: d["environment"]["dates"].update(job="next thursday"),
        # Runnable: every operation is one its service declares, and every fixture loads.
        lambda d: d["environment"]["connectors"][0]["operations"].append(
            "calendar.delete"
        ),
        lambda d: d["environment"]["connectors"][0].update(
            fixture="tasks/fixtures/rain-cover-question/marketplace.json"
        ),
        # The world has one time zone, and it is a real one. (A fixture's
        # dates render in it; one written with another offset fails below.)
        lambda d: d["environment"].update(timezone="Europe/Nowhere"),
    ],
)
def test_a_broken_task_does_not_load(change, tmp_path):
    with pytest.raises((ValidationError, ValueError)):
        load(broken(KINGSWAY, tmp_path, change))


def test_a_task_that_needs_an_environment_may_name_undeclared_operations(tmp_path):
    def needs(d):
        d["status"] = "needs_environment"
        d["environment"]["connectors"][0]["operations"].append("calendar.delete")

    assert load(broken(KINGSWAY, tmp_path, needs)).status == "needs_environment"


def with_start(tmp_path, text: str) -> Path:
    """Kingsway with a copy of its calendar fixture whose first event starts at `text`."""
    data = json.loads(KINGSWAY.read_text())
    (calendar,) = data["environment"]["connectors"]
    fixture = json.loads((ROOT / calendar["fixture"]).read_text())
    fixture["events"][0]["start"] = text
    calendar["fixture"] = str(tmp_path / "calendar.json")
    (tmp_path / "calendar.json").write_text(json.dumps(fixture))
    (tmp_path / "task.json").write_text(json.dumps(data))
    return tmp_path / "task.json"


def test_a_fixture_is_checked_as_its_service_sees_it_once_rendered(tmp_path):
    """A fixture is a template: load() renders it at a sample anchor, or the
    one given, and its service must accept what that renders to."""
    task = load(with_start(tmp_path, "{{job@07:00}}"))
    assert task.dump()["environment"]["connectors"][0]["fixture"].endswith(
        "calendar.json"
    )
    zone = ZoneInfo(task.environment.timezone)
    for wall in ("2026-01-30T17:45", "2025-11-02T00:30", "2028-02-28T06:00"):
        load(KINGSWAY, datetime.fromisoformat(wall).replace(tzinfo=zone))
    # Not a date-time once rendered, not a placeholder at all, or a time
    # outside the world's one time zone.
    for text in (
        "{{job:Weekday}}",
        "{{nothing@08:00}}",
        "{{job@8}}",
        "{{job}}T08:00:00+01:00",
    ):
        with pytest.raises((ValidationError, ValueError)):
            load(with_start(tmp_path, text))


def test_each_original_names_its_failure_and_each_twin_points_back_at_it():
    """An original's observed.failure says what it tests; a twin has an empty
    observed, and it and its original name each other."""
    tasks = {p.stem: json.loads(p.read_text()) for p in TASKS}
    twins = [t for t in tasks.values() if t["observed"] == {}]
    originals = [t for t in tasks.values() if t["observed"] != {}]
    assert twins and len(originals) == 18
    for task in originals:
        assert set(task["observed"]) == {"failure"} and task["observed"]["failure"]
    for twin in twins:
        original = tasks[twin["twin"]["id"]]
        assert original["observed"] != {}
        assert original["twin"]["id"] == twin["id"]
        assert (
            twin["simulated_user"]["persona"] == original["simulated_user"]["persona"]
        )


CALENDAR_START = {
    "events": [
        {"id": "k", "start": "2026-09-28", "attendees": ["Luis"]},
        {"id": "f", "attendees": ["Jasmeet"]},
    ]
}
CALENDAR_END = {
    "events": [
        {"id": "k", "start": "2026-10-01", "attendees": ["Luis"]},
        {"id": "f", "attendees": ["Luis", "Jasmeet"]},
    ]
}


def record(
    requests=(), calendar_end=CALENDAR_END, simulated_user=None, turns=None, task=None
):
    services = [Service("calendar", CALENDAR_START, calendar_end, list(requests))]
    turns = turns or [
        Turn("user", "move k to thursday", 10),
        Turn("assistant", "Moved k.", 90),
    ]
    decisions = simulated_user and simulated_user["decisions"]
    return records.record(list(requests), turns, decisions, task, services)


def end_state_results(rec, rules):
    rec.task["rules"]["end_state"] = rules
    verdict = grade(rec, Ladder(AnswerStore()))
    return [g["result"] for g in verdict["gates"] if g["name"] == "end state"]


def calendar_rule(path, **check):
    return {"service": "calendar", "path": path, **check}


UNCHANGED = {"service": "calendar", "unchanged": True}
UTC = {"events": [{"id": "k", "start": "2026-10-01T15:00:00Z"}]}


@pytest.mark.parametrize(
    "calendar_end, rule, expected",
    [
        # equals compares lists in any order.
        (
            CALENDAR_END,
            calendar_rule("events[id=k].start", equals="2026-10-01"),
            "pass",
        ),
        (
            CALENDAR_END,
            calendar_rule("events[id=f].attendees", equals=["Jasmeet", "Luis"]),
            "pass",
        ),
        (
            CALENDAR_END,
            calendar_rule("events[id=k].attendees", equals=["Luis", "Jasmeet"]),
            "fail",
        ),
        # equals compares date-times as moments in time.
        (
            UTC,
            calendar_rule("events[id=k].start", equals="2026-10-01T08:00:00-07:00"),
            "pass",
        ),
        (CALENDAR_START, UNCHANGED, "pass"),
        (CALENDAR_END, UNCHANGED, "fail"),
        # Without a saved end state, a human decides.
        (None, UNCHANGED, "needs_human_annotation"),
        (
            CALENDAR_END,
            calendar_rule(
                "events", contains={"id": "f", "attendees": ["Jasmeet", "Luis"]}
            ),
            "pass",
        ),
        (
            CALENDAR_END,
            calendar_rule("events[id=f].attendees", contains="Luis"),
            "pass",
        ),
        (
            CALENDAR_END,
            calendar_rule("events[id=f].attendees", contains="Doug"),
            "fail",
        ),
        # matches, for values an assistant writes in its own words, such as an
        # address. Only a string can match: a list at the path fails.
        (CALENDAR_END, calendar_rule("events[id=k].start", matches="^2026"), "pass"),
        (CALENDAR_END, calendar_rule("events[id=k].start", matches="^1999"), "fail"),
        (CALENDAR_END, calendar_rule("events", matches="."), "fail"),
    ],
)
def test_end_state_rules(calendar_end, rule, expected):
    assert end_state_results(record(calendar_end=calendar_end), [rule]) == [expected]


def test_an_end_state_pattern_must_be_a_regular_expression():
    with pytest.raises(ValidationError, match="not a regular expression"):
        EndState(service="calendar", path="events", matches="(")


FACT = {
    "id": "lot_code",
    "value": "lot C7",
    "reveal_when": "the assistant asks for the lot code",
    "needed_before": ["subscriptions.cancel"],
}
ORDER = Request(
    "subscriptions", "subscriptions.cancel", 50, {"product": "Lot C7"}, {"order": "o1"}
)


def fact_results(rec):
    rec.task["simulated_user"]["facts"] = [FACT]
    verdict = grade(rec, Ladder(AnswerStore()))
    return [
        g["result"]
        for g in verdict["gates"]
        if g["name"].startswith("asked for lot_code")
    ]


def test_fact_given_by_the_simulated_user_before_the_order_passes():
    before = {
        "decisions": [
            {
                "after_turn": 1,
                "saw": "asks_for:lot_code",
                "action": "reveal",
                "fact_ids": ["lot_code"],
                "at_ms": 40,
            }
        ]
    }
    after = {
        "decisions": [
            {
                "after_turn": 3,
                "saw": "asks_for:lot_code",
                "action": "reveal",
                "fact_ids": ["lot_code"],
                "at_ms": 60,
            }
        ]
    }
    assert fact_results(record([ORDER], simulated_user=before)) == ["pass"]
    assert fact_results(record([ORDER], simulated_user=after)) == ["fail"]


def test_a_fact_stated_in_a_refusal_does_not_pass_the_ask_first_check():
    refused = {
        "decisions": [
            {
                "after_turn": 1,
                "saw": "asks_approval:subscriptions.cancel",
                "action": "refuse",
                "fact_ids": [],
                "at_ms": 40,
                "text": "no, it has to be lot C7",
            }
        ]
    }
    assert fact_results(record([ORDER], simulated_user=refused)) == ["fail"]


def test_a_task_that_forbids_approval_requests_holds_no_user_facts():
    """The user already said go ahead: a turn asking for a fact and approval
    at once gets only the fact, so the go-ahead is never given. leave_kyle is
    the exception: Kyle is another buyer, and rules.no_approval_request still
    re-reads a turn taken as a fact request."""
    for path in TASKS:
        data = json.loads(path.read_text())
        if data["rules"].get("no_approval_request"):
            facts = [f["id"] for f in data["simulated_user"]["facts"]]
            kyle = path.stem == "anita-reply-with-calendar"
            assert facts == (["leave_kyle"] if kyle else []), path.stem
