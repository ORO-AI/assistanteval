"""Flights: the booked flight, its alternatives, and a no-cost change."""

from pydantic import AwareDatetime, BaseModel, Field

from assistanteval.services import NoParams, find
from assistanteval.world import MONEY, READ, Refused, service


class FlightOption(BaseModel):
    id: str
    departure: AwareDatetime
    change_fee_usd: float = Field(ge=0)


class FlightState(BaseModel):
    current: FlightOption
    alternatives: list[FlightOption]


class ChangeParams(BaseModel):
    option_id: str = Field(description="The alternative flight to switch to")


flights = service("flights", "Flights", FlightState)


@flights.op(
    "alternatives",
    "GET /v1/alternatives",
    tool="list_alternatives",
    params=NoParams,
    output=FlightState,
    kind=READ,
    description="Show the user's booked flight and the alternatives it can change to.",
)
def list_alternatives(state: FlightState, p: NoParams):
    return state


@flights.op(
    "change",
    "POST /v1/changes/{option_id}",
    tool="change_flight",
    params=ChangeParams,
    output=FlightOption,
    kind=MONEY,
    plain="change a flight",
    description="Change the user's booking to an alternative flight.",
)
def change_flight(state: FlightState, p: ChangeParams):
    option = find(state.alternatives, p.option_id, "Unknown flight option")
    if option.change_fee_usd or option.departure <= state.current.departure:
        raise Refused("Only a later no-cost flight can be selected", 409)
    state.current = option
    state.alternatives = [item for item in state.alternatives if item.id != p.option_id]
    return option
