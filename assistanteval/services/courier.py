"""Courier: book parcel pickups, and change a booking detail by detail.

Only the date is required, so a pickup can be booked before every address
is known. Changing a booked pickup by its id sets only the details given;
every other detail keeps its stored value.
"""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from assistanteval.services import NoParams, Text, find, new_id
from assistanteval.world import READ, VISIBLE, service


class PickupFields(BaseModel):
    """Every pickup detail optional: null means "not given"."""

    model_config = ConfigDict(extra="forbid")

    date: dt.date | None = Field(default=None, description="Pickup date, YYYY-MM-DD")
    time: str | None = Field(
        default=None,
        pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
        description="Pickup time, 24-hour HH:MM",
    )
    pickup_address: Text | None = Field(
        default=None, description="Where the courier collects the parcel"
    )
    recipient: Text | None = Field(default=None, description="Who receives it")
    delivery_address: Text | None = Field(
        default=None, description="Where the parcel is delivered"
    )
    note: Text | None = Field(
        default=None, description="Note or gift message sent with the parcel"
    )


class Pickup(PickupFields):
    date: dt.date = Field(description="Pickup date, YYYY-MM-DD")
    id: str
    status: Literal["scheduled"] = "scheduled"


class Courier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pickups: list[Pickup] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids(self):
        if len({pickup.id for pickup in self.pickups}) != len(self.pickups):
            raise ValueError("Pickup IDs must be unique")
        return self


class SaveParams(PickupFields):
    pickup_id: str | None = Field(
        default=None,
        description="Change this booked pickup: only the details given change, "
        "omitted (or null) ones are kept. Omit to book a new pickup",
    )

    @model_validator(mode="after")
    def booking_has_a_date(self):
        if self.pickup_id is None and self.date is None:
            raise ValueError("A new pickup needs a date")
        return self


courier = service("courier", "Courier", Courier)


@courier.op(
    "list",
    "GET /v1/pickups",
    tool="list_pickups",
    params=NoParams,
    output=Courier,
    kind=READ,
    description="List the user's booked courier pickups.",
)
def list_pickups(state: Courier, p: NoParams):
    return state


@courier.op(
    "save",
    "POST /v1/pickups",
    tool="save_pickup",
    params=SaveParams,
    output=Pickup,
    kind=VISIBLE,
    plain="book a courier pickup",
    description="Book a courier pickup (a date is required), or change a booked one by its "
    "id: only the details given change, the rest are kept. The courier company acts on it.",
    invalid="Invalid pickup: a new booking needs a date (YYYY-MM-DD); a time, if given, is HH:MM",
    logged=lambda p: p.model_dump(mode="json", exclude_none=True),
)
def save_pickup(state: Courier, p: SaveParams):
    """Book a new pickup, or change only the given details of an existing one."""
    given = p.model_dump(exclude={"pickup_id"}, exclude_none=True)
    if p.pickup_id is None:
        pickup = Pickup(id=new_id("pickup", (x.id for x in state.pickups)), **given)
        state.pickups.append(pickup)
        return pickup
    before = find(state.pickups, p.pickup_id, "Pickup not found")
    pickup = Pickup.model_validate({**before.model_dump(), **given, "id": p.pickup_id})
    state.pickups[state.pickups.index(before)] = pickup
    return pickup
