"""Every task, rendered at many anchors, is a coherent world its services accept.

A task's dates are expressions rendered at the run's start (assistanteval.dates), so a
task must read correctly whatever day it runs: across a month end, a year
end, a leap day and both DST changes, at any hour.
"""

import calendar
import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from assistanteval import dates, world
from assistanteval.grader.grade import validity
from assistanteval.run_record import RunRecord, Service, Turn
from assistanteval.task import Task, load as load_task
from assistanteval.connectors import open_service

ROOT = Path(__file__).parents[1]
TASKS = sorted(ROOT.joinpath("tasks").glob("*.json"))
# Local wall times, in each task's own zone.
ANCHORS = [
    "2026-09-29T09:00",  # a Tuesday morning
    "2026-10-04T03:10",  # a Sunday, in the middle of the night
    "2026-01-30T17:45",  # a Friday before a month end
    "2026-12-30T23:50",  # the year end, minutes before midnight
    # Historical Vancouver transitions: newer tzdb cancels the 2026 fall-back.
    "2025-10-31T09:00",  # a Friday: next week's dates are past the fall DST change
    "2025-11-02T00:30",  # the fall DST change's own day, before it
    "2025-03-07T20:00",  # a Friday: next week's dates are past the spring change
    "2027-02-10T09:00",  # a 28-day February
    "2028-02-28T06:00",  # the day before a leap day
]


@pytest.mark.parametrize(
    "before, after",
    [
        ("2025-10-31T09:00", "2025-11-03T09:00"),
        ("2025-03-07T20:00", "2025-03-10T20:00"),
    ],
)
def test_dst_anchors_cross_a_real_vancouver_transition(before, after):
    zone = ZoneInfo("America/Vancouver")
    assert before in ANCHORS
    assert datetime.fromisoformat(before).replace(tzinfo=zone).utcoffset() != (
        datetime.fromisoformat(after).replace(tzinfo=zone).utcoffset()
    )


WEEKDAYS = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
MONTHS = list(calendar.month_name[1:])
MONTH = "|".join(MONTHS)
INSTANT = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def anchor(task: dict, wall: str) -> datetime:
    zone = ZoneInfo(task["environment"]["timezone"])
    return datetime.fromisoformat(wall).replace(tzinfo=zone)


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)


def check_coherent(task: dict, rendered: dates.World) -> None:
    """What must hold of any task's world at any anchor."""
    now = rendered.anchor
    zone = now.tzinfo
    fixtures = {name: json.loads(raw) for name, raw in rendered.fixtures.items()}
    texts = list(strings({k: v for k, v in rendered.task.items() if k != "observed"}))
    texts += list(strings(fixtures))

    for text in texts:
        # Every date-time carries the offset its zone has at that moment.
        for stamp in INSTANT.findall(text):
            moment = datetime.fromisoformat(stamp)
            assert moment.utcoffset() == moment.astimezone(zone).utcoffset(), stamp
        # A weekday named with its date is that date's weekday.
        for weekday, month, day in re.findall(
            rf"({WEEKDAYS}),? ({MONTH}) (\d+)\b", text
        ):
            assert near(now, month, int(day)).strftime("%A") == weekday, text
        for weekday, day, month in re.findall(
            rf"({WEEKDAYS}) the (\d+)(?:st|nd|rd|th), ({MONTH})", text
        ):
            assert near(now, month, int(day)).strftime("%A") == weekday, text

    # Things already happened are in the past; things planned are ahead.
    for event in fixtures.get("calendar", {}).get("events", []):
        start, end = map(datetime.fromisoformat, (event["start"], event["end"]))
        assert now < start < end, event
    market = fixtures.get("marketplace")
    if market:
        assert datetime.fromisoformat(market["now"]) == now
        for listing in market["my_listings"]:
            for thread in listing["threads"]:
                for message in thread["messages"]:
                    assert datetime.fromisoformat(message["date"]) < now, message
    flights = fixtures.get("flights")
    if flights:
        for flight in [flights["current"], *flights["alternatives"]]:
            assert datetime.fromisoformat(flight["departure"]) > now, flight
    for charge in fixtures.get("subscriptions", {}).get("charges", []):
        for key in ("last_charged", "last_used"):
            if charge[key]:
                assert date.fromisoformat(charge[key]) < now.date(), charge
    for rule in rendered.task["rules"]["end_state"]:
        for text in strings(rule.get("equals")):
            if INSTANT.fullmatch(text):
                assert datetime.fromisoformat(text) > now, rule

    # A weekday named without its date must name one day only: one of the
    # next six. A weekday a week or more away needs its date beside it.
    named = dates.resolve(task["environment"].get("dates", {}), now)
    for template in strings({k: v for k, v in task.items() if k != "observed"}):
        for match in dates.PLACEHOLDER.finditer(template):
            if match.group(4) not in {"Weekday", "weekday", "Wkd", "wkd"}:
                continue
            day = dates.evaluate(named, *match.groups()[:3])
            day = day.date() if isinstance(day, datetime) else day
            head = re.escape("{{" + match.group(1) + match.group(2))
            dated = re.search(
                r"\d+(st|nd|rd|th)\b|" + head + r"(@[\d:]+)?:(day|nth|Month\}\} \d+)",
                template,
            )
            if not dated:
                assert 1 <= (day - now.date()).days <= 6, (match.group(0), template)


def near(now: datetime, month: str, day: int) -> date:
    """The date with this month and day closest to `now`."""
    number = MONTHS.index(month) + 1
    candidates = [
        date(now.year + y, number, day)
        for y in (-1, 0, 1)
        if day <= calendar.monthrange(now.year + y, number)[1]
    ]
    return min(candidates, key=lambda d: abs(d - now.date()))


# A date written out instead of rendered: 2026-09-28, September, Sept 28.
FIXED_DATE = re.compile(
    r"\d{4}-\d\d-\d\d|\b(January|February|March|April|June|July|August|September"
    r"|October|November|December|May \d|(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct"
    r"|Nov|Dec)\.? \d)"
)


@pytest.mark.parametrize("path", TASKS, ids=lambda p: p.stem)
def test_task_file_is_complete_and_its_world_opens_as_the_grader_expects(
    path, tmp_path
):
    # The model's checks, the declared operations and the fixtures.
    loaded = load_task(path)
    assert loaded.id == path.stem
    task = load(path)
    assert loaded.dump() == task  # the task as written: its dates are templates
    # The assistant reads the real clock, so no date is fixed: each is a rule
    # in environment.dates, rendered at the run's start.
    assert "run_between" not in task["environment"]
    assert all("start_state_sha256" not in c for c in task["environment"]["connectors"])
    templates = [json.dumps({k: v for k, v in task.items() if k != "observed"})]
    templates += [Path(c.fixture).read_text() for c in loaded.environment.connectors]
    for text in templates:
        assert not FIXED_DATE.search(text), FIXED_DATE.search(text).group(0)

    moment = anchor(task, ANCHORS[0])
    rendered = dates.render(task, ROOT, moment)
    started = []
    for connector in task["environment"]["connectors"]:
        name = connector["service"]
        if name not in world.SERVICES:
            continue  # a service no module declares yet (a needs_environment task)
        # The service, opened as the runner opens it, takes the rendered fixture.
        fixture = tmp_path / f"{name}.json"
        fixture.write_bytes(rendered.fixtures[name])
        service = open_service(name, fixture, "openapi_api_key")
        service.prepare()
        started.append(Service(name, service.state()))
        service.release()
    # The record holds the task and each fixture rendered at the anchor, as
    # the runner saves them, and the grader finds each service started from them.
    opening = Turn("user", "", int(moment.timestamp() * 1000) + 1000)
    record = RunRecord("r", task["id"], "p", [opening], started, {})
    record.task = rendered.task
    fixtures = {k: json.loads(v) for k, v in rendered.fixtures.items()}
    record.manifest = {"world": rendered.manifest() | {"fixture_data": fixtures}}
    checks = validity(record, Task.model_validate(record.task))
    results = {c.name: c.result for c in checks if "world" in c.name}
    assert results == {
        "services started from the task's world": "pass",
        "run started while its world was current": "pass",
    }


@pytest.mark.parametrize("path", TASKS, ids=lambda p: p.stem)
def test_task_is_coherent_on_every_day_of_a_year(path):
    task = load(path)
    first = anchor(task, "2026-09-01T11:20")
    moments = [anchor(task, wall) for wall in ANCHORS]
    for days in range(366):
        moment = first + timedelta(days=days, hours=(days * 7) % 24)
        moments.append(moment.replace(tzinfo=None).replace(tzinfo=first.tzinfo))
    for moment in moments:
        rendered = dates.render(task, ROOT, moment)
        check_coherent(task, rendered)
        Task.model_validate(rendered.task)  # still a task once rendered
        for name, raw in rendered.fixtures.items():
            if name in world.SERVICES:  # its service takes the rendered fixture
                world.SERVICES[name].state.model_validate_json(raw)


def test_courier_trap_holds_wherever_it_is_anchored():
    """The 28th the user means is not the weekday they name; next month's is."""
    task = load(ROOT / "tasks" / "courier-date-month.json")
    first = anchor(task, "2026-01-01T10:00")
    for days in range(4 * 366):
        now = first + timedelta(days=days)
        rendered = dates.render(task, ROOT, now)
        named = dates.resolve(task["environment"]["dates"], rendered.anchor)
        pickup, decoy, birthday = named["pickup"], named["decoy"], named["birthday"]
        assert pickup.day == decoy.day == 28 and now.date() < pickup < birthday
        assert (decoy.year * 12 + decoy.month) - (pickup.year * 12 + pickup.month) == 1
        assert pickup.weekday() != decoy.weekday()
        opening = rendered.task["opening_message"]
        assert opening.startswith(f"Set up the courier pickup for {decoy:%a} the 28th")
        assert (
            f"{pickup:%A} the 28th, {pickup:%B}"
            in rendered.task["simulated_user"]["facts"][0]["value"]
        )
        # The pickup is this month's 28th unless that is today or past, or
        # this is a 28-day February, whose 28th shares March's weekday.
        feb = now.month == 2 and pickup.month == 3
        assert (pickup.month == now.month) == (now.day < 28 and not feb)
