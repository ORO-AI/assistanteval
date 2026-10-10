"""Marketplace: the user's listings and buyer threads, and replies to buyers."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from assistanteval.services import Text
from assistanteval.world import READ, VISIBLE, Refused, service


class Message(BaseModel):
    sender: Literal["buyer", "me"] = Field(alias="from")
    text: str
    date: datetime


class Thread(BaseModel):
    id: str
    buyer: str
    messages: list[Message]


class Listing(BaseModel):
    id: str
    title: str
    asking_usd: int
    status: str
    threads: list[Thread]


class Marketplace(BaseModel):
    now: datetime
    my_listings: list[Listing]


class Listings(BaseModel):
    listings: list[Listing]


class ReplyResult(BaseModel):
    listing_id: str
    thread: Thread


class ReadParams(BaseModel):
    thread_id: str | None = Field(
        default=None, description="Only this buyer thread; omit for every listing"
    )


class ReplyParams(BaseModel):
    message: Text
    thread_id: str = Field(description="The buyer thread to reply in")


marketplace = service("marketplace", "Marketplace", Marketplace)


@marketplace.op(
    "read",
    "GET /v1/listings",
    tool="read_listings",
    params=ReadParams,
    output=Listings,
    kind=READ,
    description="Read the user's marketplace listings and their buyer threads.",
)
def read_listings(state: Marketplace, p: ReadParams):
    if p.thread_id is None:
        return Listings(listings=state.my_listings)
    listings = [
        listing.model_copy(
            update={"threads": [t for t in listing.threads if t.id == p.thread_id]}
        )
        for listing in state.my_listings
    ]
    return Listings(listings=[listing for listing in listings if listing.threads])


@marketplace.op(
    "reply",
    "POST /v1/threads/{thread_id}/messages",
    tool="reply_to_thread",
    params=ReplyParams,
    output=ReplyResult,
    kind=VISIBLE,
    plain="send a reply to a buyer",
    status=201,
    description="Send a message to a buyer in one of the user's listing threads.",
)
def reply_to_thread(state: Marketplace, p: ReplyParams):
    for listing in state.my_listings:
        for thread in listing.threads:
            if thread.id == p.thread_id:
                thread.messages.append(
                    Message.model_validate(
                        {"from": "me", "text": p.message, "date": state.now}
                    )
                )
                return ReplyResult(listing_id=listing.id, thread=thread)
    raise Refused("Marketplace thread not found", 404)
