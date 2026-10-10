"""Anchor a task's world to the moment a run starts.

The assistants under test read the real clock, so a task never holds a fixed
date. Every string in a task and its fixtures says where a date goes with a
placeholder: a date expression relative to the anchor (the run's start, in
the task's time zone), and optionally a format. At run start the runner
renders the task and its fixtures at the anchor, and the run record keeps the
rendered task and the anchor, so the grader sees the same world.
tasks/README.md documents the syntax.

    "{{today+3wd@09:00}}"                ->  2026-10-02T09:00:00-07:00
    "Call Sam {{today+3wd@09:00:long}}"  ->  Call Sam Friday, October 2 at 9:00 am
    "{{now-80min}}"                      ->  2026-09-29T07:40:00-07:00

A date the story mentions in several places is named once in
``environment.dates``, with the same expressions, and used as ``{{name}}``:

    "dates": {
      "pickup": "today+2d",                the day after tomorrow
      "moved": "job+3wd",                  three workdays after the date named job
      "courier": {"monthday": 28, "weekday_unlike_next_month": true}
    }
    "{{pickup:Weekday}} at 6"  ->  Thursday at 6
"""

from __future__ import annotations

import calendar
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from assistanteval.run_record import sha256

# A date expression: a base (now, today or a named date), any offsets applied
# left to right (+3wd, -80min), then an optional time of day. A placeholder is
# an expression and an optional format: {{today+3wd@09:00:long}}.
EXPRESSION = r"(\w+)((?:[+-]\d+(?:wd|mo|min|d))*)(?:@(\d{1,2}:\d{2}))?"
OFFSET = re.compile(r"([+-]\d+)(wd|mo|min|d)")
PLACEHOLDER = re.compile(r"\{\{\s*" + EXPRESSION + r"(?::(\w+))?\s*\}\}")
NAMED = re.compile(EXPRESSION)
# The one named rule an expression cannot say (courier-date-month's trap).
MONTHDAY_KEYS = {"monthday", "weekday_unlike_next_month"}
# Task fields never rendered: the observations are history.
KEEP = {"observed"}


def anchor_at(ms: int, timezone: str) -> datetime:
    """The world's anchor for a run starting at `ms`: that moment in the
    task's time zone, to the minute."""
    moment = datetime.fromtimestamp(ms / 1000, ZoneInfo(timezone))
    return moment.replace(second=0, microsecond=0)


def parse_anchor(value: str, timezone: str) -> datetime:
    moment = datetime.fromisoformat(value)
    if moment.tzinfo is None:
        raise ValueError(f"anchor {value} has no UTC offset")
    return moment.astimezone(ZoneInfo(timezone))


def _local(day: date, at: time, zone: ZoneInfo) -> datetime:
    # Through UTC, so a wall time the DST change skips lands on a real moment.
    return datetime.combine(day, at, zone).astimezone(UTC).astimezone(zone)


def _clock(value: str) -> time:
    hours, minutes = value.split(":")
    return time(int(hours), int(minutes))


def _add_months(day: date, months: int) -> date:
    month = day.month - 1 + months
    year, month = day.year + month // 12, month % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def _add_workdays(day: date, count: int) -> date:
    step = 1 if count > 0 else -1
    for _ in range(abs(count)):
        day += timedelta(days=step)
        while day.weekday() >= 5:
            day += timedelta(days=step)
    return day


def _monthday(after: date, monthday: int, unlike_next: bool) -> date:
    """The first `monthday` of a month strictly after `after`. With
    `unlike_next`, only one whose weekday differs from the same day of the
    next month (a 28-day February never qualifies)."""
    year, month = after.year, after.month
    for _ in range(24):
        if monthday <= calendar.monthrange(year, month)[1]:
            day = date(year, month, monthday)
            following = _add_months(day, 1)
            if day > after and (
                not unlike_next
                or (following.day == monthday and following.weekday() != day.weekday())
            ):
                return day
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    raise ValueError(f"no month has a {monthday} that fits")


def _ordinal(n: int) -> str:
    suffix = (
        "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    )
    return f"{n}{suffix}"


def _clock_text(moment: datetime) -> str:
    hour = moment.hour % 12 or 12
    return f"{hour}:{moment.minute:02d} {'am' if moment.hour < 12 else 'pm'}"


def _long(value: date | datetime) -> str:
    text = f"{value:%A}, {value:%B} {value.day}"
    return f"{text} at {_clock_text(value)}" if isinstance(value, datetime) else text


def _time(value: date | datetime) -> str:
    if not isinstance(value, datetime):
        raise ValueError("a day has no time of day: add @HH:MM")
    return _clock_text(value)


FORMATS = {
    "date": lambda d: d.strftime("%Y-%m-%d"),
    "Weekday": lambda d: d.strftime("%A"),
    "weekday": lambda d: d.strftime("%A").lower(),
    "Wkd": lambda d: d.strftime("%a"),
    "wkd": lambda d: d.strftime("%a").lower(),
    "Month": lambda d: d.strftime("%B"),
    "month": lambda d: d.strftime("%B").lower(),
    "day": lambda d: str(d.day),
    "nth": lambda d: _ordinal(d.day),
    "year": lambda d: str(d.year),
    "long": _long,  # Thursday, October 1 (at 9:00 am, for an instant)
    "time": _time,  # 9:00 am
}


def _day(value: date | datetime) -> date:
    return value.date() if isinstance(value, datetime) else value


def evaluate(
    dates: dict, name: str, offsets: str = "", at: str | None = None
) -> date | datetime:
    """The day or instant an expression stands for. A day stays a day; an
    instant stays an instant, keeping its wall-clock time across day, workday
    and month offsets (so "now+2d" is the same time of day two days later),
    while minutes count real time."""
    if name not in dates:
        raise ValueError(f"no date named {name}")
    zone = dates["now"].tzinfo
    value = dates[name]
    for amount, unit in OFFSET.findall(offsets):
        amount = int(amount)
        if unit == "min":
            if not isinstance(value, datetime):
                raise ValueError("minutes count from an instant, such as now")
            value = (value.astimezone(UTC) + timedelta(minutes=amount)).astimezone(zone)
            continue
        day = _day(value)
        if unit == "d":
            day += timedelta(days=amount)
        elif unit == "wd":
            day = _add_workdays(day, amount)
        else:
            day = _add_months(day, amount)
        value = _local(day, value.time(), zone) if isinstance(value, datetime) else day
    if at is not None:
        value = _local(_day(value), _clock(at), zone)
    return value


def resolve(rules: dict, anchor: datetime) -> dict[str, date | datetime]:
    """Every named date, in the anchor's zone. "now" is the anchor itself and
    "today" its day; each rule may use the names before it."""
    known: dict[str, date | datetime] = {"now": anchor, "today": anchor.date()}
    for name, rule in rules.items():
        if name in known:
            raise ValueError(f"date {name} is predefined")
        if isinstance(rule, dict):
            if set(rule) - MONTHDAY_KEYS or "monthday" not in rule:
                raise ValueError(f"date {name}: an object rule takes {MONTHDAY_KEYS}")
            known[name] = _monthday(
                anchor.date(),
                rule["monthday"],
                rule.get("weekday_unlike_next_month", False),
            )
            continue
        match = NAMED.fullmatch(rule) if isinstance(rule, str) else None
        if match is None:
            raise ValueError(f"date {name}: {rule!r} is not a date expression")
        try:
            known[name] = evaluate(known, *match.groups())
        except ValueError as error:
            raise ValueError(f"date {name}: {error}") from None
    return known


def _format(dates: dict, match: re.Match) -> str:
    try:
        value = evaluate(dates, *match.groups()[:3])
        style = match.group(4)
        if style is None or style == "iso":
            if isinstance(value, datetime):
                return value.isoformat(timespec="seconds")
            return value.isoformat()
        if style not in FORMATS:
            raise ValueError(f"no format {style}")
        return FORMATS[style](value)
    except ValueError as error:
        raise ValueError(f"{match.group(0)}: {error}") from None


def _render(value: object, dates: dict) -> object:
    if isinstance(value, str):
        text = PLACEHOLDER.sub(lambda m: _format(dates, m), value)
        if "{{" in text or "}}" in text:
            raise ValueError(f"a malformed date placeholder in {value!r}")
        return text
    if isinstance(value, list):
        return [_render(v, dates) for v in value]
    if isinstance(value, dict):
        return {k: _render(v, dates) for k, v in value.items()}
    return value


def _fixture_bytes(data: object) -> bytes:
    return json.dumps(data, indent=2, ensure_ascii=False).encode() + b"\n"


@dataclass(frozen=True)
class World:
    """A task and its fixtures rendered at one anchor."""

    anchor: datetime
    task: dict
    fixtures: dict[str, bytes]  # service -> rendered fixture file

    def manifest(self) -> dict:
        """What the run record keeps: enough to render this world again and
        prove it is the same one."""
        return {
            "anchor": self.anchor.isoformat(timespec="seconds"),
            "fixtures": {name: sha256(raw) for name, raw in self.fixtures.items()},
        }


def _render_task(task: dict, dates: dict) -> dict:
    return {k: v if k in KEEP else _render(v, dates) for k, v in task.items()}


def render(task: dict, root: Path, anchor: datetime) -> World:
    """The task and every fixture it names, rendered at `anchor` in the task's
    time zone. Fixture paths are relative to `root`, the repository."""
    anchor = anchor.astimezone(ZoneInfo(task["environment"]["timezone"]))
    dates = resolve(task["environment"].get("dates", {}), anchor)
    fixtures = {
        connector["service"]: _fixture_bytes(
            _render(json.loads((root / connector["fixture"]).read_bytes()), dates)
        )
        for connector in task["environment"]["connectors"]
    }
    return World(anchor, _render_task(task, dates), fixtures)
