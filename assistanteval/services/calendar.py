"""Calendar: find, change and create the user's events."""

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from assistanteval.services import Text, find, matches, new_id
from assistanteval.world import PRIVATE_CHANGE, READ, VISIBLE, Refused, service


class CalendarEventCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    start: AwareDatetime
    end: AwareDatetime
    location: str
    description: str
    attendees: list[str]

    @model_validator(mode="after")
    def valid_interval(self):
        if self.start >= self.end:
            raise ValueError("Event must end after it starts")
        return self


class CalendarEvent(CalendarEventCreate):
    id: str


class Calendar(BaseModel):
    events: list[CalendarEvent]

    @model_validator(mode="after")
    def unique_event_ids(self):
        if len({event.id for event in self.events}) != len(self.events):
            raise ValueError("Calendar event IDs must be unique")
        return self


class SearchParams(BaseModel):
    query: Text | None = Field(
        default=None,
        description="Words to find in the event's title, location, description or attendees",
    )
    start: AwareDatetime | None = Field(
        default=None, description="Range start, with a timezone offset"
    )
    end: AwareDatetime | None = Field(
        default=None, description="Range end, with a timezone offset"
    )

    @model_validator(mode="after")
    def query_or_range(self):
        if (
            (self.start is None) != (self.end is None)
            or (self.query is None and self.start is None)
            or (self.start is not None and self.end <= self.start)
        ):
            raise ValueError("Provide a query or a valid time range")
        return self


class UpdateParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    start: AwareDatetime | None = None
    end: AwareDatetime | None = None
    location: str | None = None
    description: str | None = None
    attendees: list[str] | None = None
    event_id: str = Field(description="The event to change")


def _search_input(p: SearchParams) -> dict:
    logged = {"query": p.query} if p.query is not None else {}
    if p.start is not None:
        logged.update(start=p.start.isoformat(), end=p.end.isoformat())
    return logged


calendar = service("calendar", "Calendar", Calendar)


@calendar.op(
    "search",
    "GET /v1/events",
    tool="search_events",
    params=SearchParams,
    output=Calendar,
    kind=READ,
    description="Find calendar events by words in their title, location, description or attendees, or within a time range.",
    invalid="Provide a query or a valid time range",
    logged=_search_input,
)
def search_events(state: Calendar, p: SearchParams):
    return Calendar(
        events=[
            e
            for e in state.events
            if matches(p.query, e.title, e.location, e.description, *e.attendees)
            and (p.start is None or (e.start < p.end and p.start < e.end))
        ]
    )


@calendar.op(
    "update",
    "PATCH /v1/events/{event_id}",
    tool="update_event",
    params=UpdateParams,
    output=CalendarEvent,
    kind=PRIVATE_CHANGE,
    plain="change or move a calendar event",
    description="Change an existing event's title, time, location, description or "
    "attendees. Attendees see the change.",
    invalid="Invalid event update",
    logged=lambda p: p.model_dump(mode="json", exclude_unset=True),
)
def update_event(state: Calendar, p: UpdateParams):
    # Only the non-null fields sent change; an empty change is refused first.
    changes = p.model_dump(exclude_none=True, exclude={"event_id"})
    if not changes:
        raise Refused("Invalid event update", 422)
    event = find(state.events, p.event_id, "Event not found")
    try:
        updated = CalendarEvent.model_validate({**event.model_dump(), **changes})
    except ValidationError:
        raise Refused("Invalid event update", 422) from None
    state.events[state.events.index(event)] = updated
    return updated


@calendar.op(
    "create",
    "POST /v1/events",
    tool="create_event",
    params=CalendarEventCreate,
    output=CalendarEvent,
    kind=VISIBLE,
    plain="add a calendar event",
    status=201,
    description="Create a calendar event. Attendees see it.",
    invalid="Invalid event",
)
def create_event(state: Calendar, p: CalendarEventCreate):
    event_id = new_id("created", (e.id for e in state.events), len(state.events) + 1)
    event = CalendarEvent(id=event_id, **p.model_dump())
    state.events.append(event)
    return event
