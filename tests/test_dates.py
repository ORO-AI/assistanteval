"""Anchored task worlds: date rules, formats, and rendering a task and its fixtures."""

import json
from datetime import date, datetime

import pytest

from assistanteval import dates
from tests.conftest import render_task

ZONE = "America/Vancouver"


def at(value: str) -> datetime:
    return dates.parse_anchor(value, ZONE)


def resolved(rules: dict, anchor: str = "2026-09-29T09:00:00-07:00") -> dict:
    return dates.resolve(rules, at(anchor))


def test_anchor_is_the_run_start_in_the_task_zone_to_the_minute():
    anchor = dates.anchor_at(1_790_000_123_456, ZONE)
    assert anchor.isoformat() == "2026-09-21T07:15:00-07:00"
    assert (
        # Historical winter offset; 2026 Vancouver rules changed in newer tzdb.
        dates.anchor_at(1_763_467_200_000, ZONE).utcoffset().total_seconds()
        == -8 * 3600
    )


def test_named_dates_are_expressions_on_the_anchor_or_an_earlier_date():
    found = resolved(
        {
            "pickup": "today+2d",
            "job": "today+1wd",
            "moved": "job+3wd",
            "back": "moved-2wd",
            "tiled": "today-3mo",
            "wrote": "now-80min",
            "call": "today+2d@14:30",
        },
        "2026-09-25T09:00:00-07:00",  # a Friday
    )
    assert found["today"] == date(2026, 9, 25)
    assert found["pickup"] == date(2026, 9, 27)
    assert found["job"] == date(2026, 9, 28)  # over the weekend to Monday
    assert found["moved"] == date(2026, 10, 1)
    assert found["back"] == date(2026, 9, 29)
    assert found["tiled"] == date(2026, 6, 25)
    assert found["wrote"].isoformat() == "2026-09-25T07:40:00-07:00"
    assert found["call"].isoformat() == "2026-09-27T14:30:00-07:00"


def test_months_keep_the_day_or_take_the_last_day_of_a_shorter_month():
    assert resolved({"d": "today+1mo"}, "2026-01-31T09:00:00-08:00")["d"] == date(
        2026, 2, 28
    )


def test_a_time_of_day_takes_the_offset_in_force_on_that_day():
    found = resolved({"d": "today+3d@18:00"}, "2025-10-31T09:00:00-07:00")
    assert found["d"].isoformat() == "2025-11-03T18:00:00-08:00"
    # A wall time the spring change skips still lands on a real moment.
    found = resolved({"d": "today+1d@02:30"}, "2025-03-08T09:00:00-08:00")
    assert found["d"].isoformat() == "2025-03-09T03:30:00-07:00"


@pytest.mark.parametrize(
    "anchor, expected",
    [
        ("2026-09-24T09:00:00-07:00", date(2026, 9, 28)),  # this month's 28th
        ("2026-09-28T09:00:00-07:00", date(2026, 10, 28)),  # today is the 28th
        ("2026-12-30T09:00:00-08:00", date(2027, 1, 28)),  # over the year end
        ("2027-02-10T09:00:00-08:00", date(2027, 3, 28)),  # a 28-day February
        ("2028-02-10T09:00:00-08:00", date(2028, 2, 28)),  # a leap February
    ],
)
def test_a_monthday_whose_weekday_differs_from_the_next_month(anchor, expected):
    found = resolved(
        {
            "d": {"monthday": 28, "weekday_unlike_next_month": True},
            "next": "d+1mo",
        },
        anchor,
    )
    assert found["d"] == expected
    assert found["next"].day == 28
    assert found["next"].weekday() != found["d"].weekday()


def test_placeholders_and_formats():
    task = {
        "environment": {
            "timezone": ZONE,
            "dates": {"pickup": "today+2d", "wrote": "now-80min"},
            "connectors": [],
        },
        "opening_message": "{{pickup:weekday}} at 6, {{pickup:wkd}} the {{pickup:nth}}",
        "facts": [
            "{{pickup:Weekday}}, {{pickup:Month}} {{pickup:day}}, {{pickup:year}}",
            "{{pickup}} {{pickup@18:00}} {{wrote}} {{now}} {{today:Wkd}}",
            "{{ pickup:month }} {{pickup:date}} {{wrote@07:00:iso}}",
        ],
    }
    rendered = render_task(task, at("2026-09-29T09:00:00-07:00"))
    assert rendered["opening_message"] == "thursday at 6, thu the 1st"
    assert rendered["facts"] == [
        "Thursday, October 1, 2026",
        "2026-10-01 2026-10-01T18:00:00-07:00 2026-09-29T07:40:00-07:00 "
        "2026-09-29T09:00:00-07:00 Tue",
        "october 2026-10-01 2026-09-29T07:00:00-07:00",
    ]


@pytest.mark.parametrize(
    "rules, text",
    [
        ({}, "{{nope}}"),
        ({"d": "today+1d"}, "{{d:shout}}"),
        ({"d": "later+1d", "later": "today+2d"}, "{{d}}"),  # used before it is named
        ({"d": "tomorrow"}, "{{d}}"),
        ({"d": "today+1dayz"}, "{{d}}"),
        ({"d": {"days": 1}}, "{{d}}"),  # an object rule is only a monthday
        ({"d": {"monthday": 28, "from": "today"}}, "{{d}}"),
        ({"d": 3}, "{{d}}"),
        ({"now": "today+1d"}, "{{now}}"),
        ({"d": "today+1d"}, "{{ d : month }}"),  # never left unrendered
        ({"d": "today+1d"}, "{{d@8am}}"),
        ({}, "{{today+5min}}"),  # minutes need an instant
        ({}, "{{today:time}}"),  # a day has no time of day
        ({}, "{{today+3x}}"),  # unknown unit: never left unrendered
        ({}, "{{today+3wd@9am}}"),
        ({}, "{{later+1d}}"),  # no such date
    ],
)
def test_a_bad_rule_or_placeholder_is_an_error(rules, text):
    task = {
        "environment": {"timezone": ZONE, "dates": rules, "connectors": []},
        "opening_message": text,
    }
    with pytest.raises(ValueError):
        render_task(task, at("2026-09-29T09:00:00-07:00"))


def test_render_keeps_the_observations_the_rules_and_the_field_order(tmp_path):
    fixture = tmp_path / "tasks" / "fixtures" / "t" / "calendar.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(
        json.dumps({"events": [{"id": "e-{{d}}", "start": "{{d@08:00}}"}]})
    )
    task = {
        "id": "t",
        "environment": {
            "connectors": [
                {"service": "calendar", "fixture": "tasks/fixtures/t/calendar.json"}
            ],
            "timezone": ZONE,
            "dates": {"d": "today+1d"},
        },
        "opening_message": "move it to {{d:weekday}}",
        "observed": {"quote": "moved to {{d}} on Sept 24"},
    }
    rendered = dates.render(task, tmp_path, at("2026-09-29T09:00:00-07:00"))
    assert rendered.task["opening_message"] == "move it to wednesday"
    assert rendered.task["observed"] == task["observed"]
    assert rendered.task["environment"]["dates"] == {"d": "today+1d"}
    assert list(rendered.task) == list(task)
    assert json.loads(rendered.fixtures["calendar"]) == {
        "events": [{"id": "e-2026-09-30", "start": "2026-09-30T08:00:00-07:00"}]
    }
    manifest = rendered.manifest()
    assert manifest["anchor"] == "2026-09-29T09:00:00-07:00"
    assert len(manifest["fixtures"]["calendar"]) == 64
    # The same anchor renders the same world, byte for byte.
    again = dates.render(task, tmp_path, at("2026-09-29T16:00:00+00:00"))
    assert again.manifest() == manifest


def render_text(text: str, anchor: str = "2026-09-29T09:00:00-07:00", rules=None):
    task = {
        "environment": {"timezone": ZONE, "dates": rules or {}, "connectors": []},
        "opening_message": text,
    }
    return render_task(task, at(anchor))["opening_message"]


@pytest.mark.parametrize(
    "text, expected",
    [
        # A day: offsets from today, then a time of day or a format.
        ("{{today+3wd}}", "2026-10-02"),
        ("{{today+3wd@09:00}}", "2026-10-02T09:00:00-07:00"),
        (
            "Reach out to Sam on {{today+3wd:Weekday}} at 9",
            "Reach out to Sam on Friday at 9",
        ),
        ("{{today-37d@09:15}}", "2026-08-23T09:15:00-07:00"),
        ("{{today-1mo}}", "2026-08-29"),
        ("{{today+1mo-2d:long}}", "Tuesday, October 27"),  # applied left to right
        ("{{today+0d}}", "2026-09-29"),
        # An instant keeps its time of day across day offsets.
        ("{{now-80min}}", "2026-09-29T07:40:00-07:00"),
        ("{{now+2d}}", "2026-10-01T09:00:00-07:00"),
        ("{{now-1d-30min}}", "2026-09-28T08:30:00-07:00"),
        # From a named date.
        ("{{job+2wd:Weekday}}", "Monday"),
        # Long and time formats.
        ("{{today+3wd@09:00:long}}", "Friday, October 2 at 9:00 am"),
        ("{{today+3wd@18:05:long}}", "Friday, October 2 at 6:05 pm"),
        ("{{today+3wd@00:30:time}}", "12:30 am"),
        ("{{now:time}}", "9:00 am"),
        ("{{job:long}}", "Thursday, October 1"),
    ],
)
def test_inline_offsets_and_formats(text, expected):
    assert render_text(text, rules={"job": "today+2d"}) == expected


def test_an_inline_offset_on_an_instant_keeps_the_wall_clock_across_dst():
    # Historical transition: 2025-11-02 ends DST in Vancouver.
    # Use 2025: Vancouver no longer falls back in November 2026 in newer tzdb.
    assert render_text("{{now+2d}}", "2025-10-31T09:00:00-07:00") == (
        "2025-11-02T09:00:00-08:00"
    )
    # Minutes are real time: 180 minutes spans the repeated hour, whose second
    # pass is in PST.
    assert render_text("{{now+180min}}", "2025-11-02T00:30:00-07:00") == (
        "2025-11-02T02:30:00-08:00"
    )


def test_inline_placeholders_render_in_fixtures_with_no_named_dates(tmp_path):
    fixture = tmp_path / "tasks" / "fixtures" / "t" / "calendar.json"
    fixture.parent.mkdir(parents=True)
    events = [
        {"id": f"standup-{n}", "start": f"{{{{today-{n}wd@09:00}}}}"}
        for n in range(1, 101)
    ]
    fixture.write_text(json.dumps({"events": events}))
    task = {
        "id": "t",
        "environment": {
            "connectors": [
                {"service": "calendar", "fixture": "tasks/fixtures/t/calendar.json"}
            ],
            "timezone": ZONE,
        },
    }
    rendered = dates.render(task, tmp_path, at("2026-09-29T09:00:00-07:00"))
    starts = [e["start"] for e in json.loads(rendered.fixtures["calendar"])["events"]]
    assert starts[0] == "2026-09-28T09:00:00-07:00"  # the workday before Tuesday
    assert starts[1] == "2026-09-25T09:00:00-07:00"  # over the weekend
    assert len(set(starts)) == 100
    assert all(datetime.fromisoformat(s).weekday() < 5 for s in starts)
