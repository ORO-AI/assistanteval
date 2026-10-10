"""Subscriptions: the user's recurring charges, and cancelling one."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from assistanteval.services import NoParams
from assistanteval.world import MONEY, READ, Refused, service


class Subscription(BaseModel):
    vendor: str = Field(min_length=1)
    amount_usd: float = Field(ge=0)
    cadence: Literal["monthly", "yearly"]
    last_charged: date
    last_used: date | None = None
    cancel_url: str | None = None
    cancelled: bool = False


class Subscriptions(BaseModel):
    charges: list[Subscription]


class CancelParams(BaseModel):
    vendor: str = Field(description="The vendor whose subscription to cancel")


subscriptions = service("subscriptions", "Subscriptions", Subscriptions)


@subscriptions.op(
    "list",
    "GET /v1/subscriptions",
    tool="list_subscriptions",
    params=NoParams,
    output=Subscriptions,
    kind=READ,
    description="List the user's subscriptions and their recurring charges.",
)
def list_subscriptions(state: Subscriptions, p: NoParams):
    return state


@subscriptions.op(
    "cancel",
    "POST /v1/subscriptions/{vendor}/cancel",
    tool="cancel_subscription",
    params=CancelParams,
    output=Subscription,
    kind=MONEY,
    plain="cancel a subscription",
    description="Cancel one of the user's subscriptions.",
)
def cancel_subscription(state: Subscriptions, p: CancelParams):
    match = next(
        (c for c in state.charges if c.vendor.casefold() == p.vendor.casefold()), None
    )
    if match is None:
        raise Refused("Unknown subscription", 404)
    if match.cancelled:
        raise Refused("Subscription already cancelled", 409)
    match.cancelled = True
    return match
