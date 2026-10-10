"""Gmail: a mailbox with drafts that can be created, edited in place and sent.

State shape::

    {"owner", "inbox":  [{"id", "sender", "recipient", "subject", "body"}],
     "drafts": [{"id", "recipient" | null, "subject", "body"}],
     "sent":   [{"id", "sender", "recipient", "subject", "body", "draft_id" | null}]}
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from assistanteval.services import EmailAddress, Text, find, matches, new_id
from assistanteval.world import (
    PRIVATE_CHANGE,
    PRIVATE_NEW,
    READ,
    VISIBLE,
    Refused,
    service,
)

SEND_INPUTS = "Provide a draft_id, or a recipient and body (not both)"


class Message(BaseModel):
    id: str
    sender: EmailAddress
    recipient: EmailAddress
    subject: str
    body: str


class Draft(BaseModel):
    id: str
    recipient: EmailAddress | None = None
    subject: str = ""
    body: str = ""


class SentMessage(Message):
    draft_id: str | None = None


class Gmail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner: EmailAddress
    inbox: list[Message] = Field(default_factory=list)
    drafts: list[Draft] = Field(default_factory=list)
    sent: list[SentMessage] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [item.id for item in (*self.inbox, *self.drafts, *self.sent)]
        if len(set(ids)) != len(ids):
            raise ValueError("Gmail message and draft IDs must be unique")
        return self

    def next_id(self, prefix: str) -> str:
        # A sent draft's id stays taken: the sent message still points to it.
        used = [item.id for item in (*self.inbox, *self.drafts, *self.sent)]
        return new_id(prefix, used + [m.draft_id for m in self.sent if m.draft_id])


class DraftCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipient: EmailAddress | None = Field(
        default=None, description="Who the email will go to; may be left blank"
    )
    subject: str = Field(default="", description="Subject line")
    body: Text = Field(description="Email body")


class DraftPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recipient: EmailAddress | None = Field(
        default=None, description="New recipient; omit (or null) to keep it"
    )
    clear_recipient: bool = Field(
        default=False, description="Remove the recipient, leaving the draft unaddressed"
    )
    subject: str | None = Field(default=None, description="New subject line")
    body: Text | None = Field(default=None, description="New email body")

    @model_validator(mode="after")
    def nonempty(self):
        if self.clear_recipient and self.recipient is not None:
            raise ValueError("Set a recipient or clear it, not both")
        if not self.changes():
            raise ValueError("Update requires at least one field")
        return self

    def changes(self) -> dict:
        """The draft fields this patch sets; null means leave unchanged."""
        fields = self.model_dump(include={"recipient", "subject", "body"})
        changes = {name: value for name, value in fields.items() if value is not None}
        return changes | ({"recipient": None} if self.clear_recipient else {})


class UpdateParams(DraftPatch):
    draft_id: str = Field(description="The draft to change")


class SendParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    draft_id: str | None = Field(
        default=None, description="Send this saved draft as it stands"
    )
    recipient: EmailAddress | None = Field(
        default=None, description="Recipient, when sending without a draft"
    )
    subject: str | None = Field(
        default=None, description="Subject line, when sending without a draft"
    )
    body: Text | None = Field(
        default=None, description="Email body, when sending without a draft"
    )

    @model_validator(mode="after")
    def draft_or_message(self):
        if self.draft_id is not None:
            if any(
                value is not None for value in (self.recipient, self.subject, self.body)
            ):
                raise ValueError(SEND_INPUTS)
        elif self.recipient is None or self.body is None:
            raise ValueError(SEND_INPUTS)
        return self


class SearchParams(BaseModel):
    query: str | None = Field(
        default=None,
        description="Words that must all appear in the sender, recipient, subject "
        "or body; omit or leave empty to list everything",
    )
    folder: Literal["inbox", "drafts", "sent"] | None = Field(
        default=None, description="Only this folder; omit for all folders"
    )


class SearchResult(BaseModel):
    owner: EmailAddress
    inbox: list[Message]
    drafts: list[Draft]
    sent: list[SentMessage]


gmail = service("gmail", "Gmail", Gmail)


@gmail.op(
    "search",
    "GET /v1/search",
    tool="search_mail",
    params=SearchParams,
    output=SearchResult,
    kind=READ,
    invalid="Invalid search",
    description="Search the user's Gmail inbox, drafts and sent mail. Returns the "
    "user's own address and every matching message and draft.",
)
def search_mail(state: Gmail, p: SearchParams):
    """Items whose sender, recipient, subject or body contain every query word."""
    query = p.query if (p.query or "").split() else None  # blank lists everything

    def hit(item) -> bool:
        sender = getattr(item, "sender", None)  # drafts have none
        return matches(query, sender, item.recipient, item.subject, item.body)

    return SearchResult(
        owner=state.owner,
        **{
            name: [item for item in getattr(state, name) if hit(item)]
            if p.folder in (None, name)
            else []
            for name in ("inbox", "drafts", "sent")
        },
    )


@gmail.op(
    "draft_create",
    "POST /v1/drafts",
    tool="create_draft",
    params=DraftCreate,
    output=Draft,
    kind=PRIVATE_NEW,
    plain="save a new email draft",
    status=201,
    invalid="Invalid draft",
    description="Save a new unsent draft in the user's Gmail. Nobody sees it.",
)
def create_draft(state: Gmail, p: DraftCreate):
    draft = Draft(id=state.next_id("draft"), **p.model_dump())
    state.drafts.append(draft)
    return draft


@gmail.op(
    "draft_update",
    "PATCH /v1/drafts/{draft_id}",
    tool="update_draft",
    params=UpdateParams,
    output=Draft,
    kind=PRIVATE_CHANGE,
    plain="save changes to an email draft",
    description="Edit a saved draft in place: its recipient, subject or body. The draft stays unsent.",
    invalid="Invalid draft update",
    logged=lambda p: {"draft_id": p.draft_id, **p.changes()},
)
def update_draft(state: Gmail, p: UpdateParams):
    before = find(state.drafts, p.draft_id, "Draft not found")
    after = before.model_copy(update=p.changes())
    state.drafts[state.drafts.index(before)] = after
    return after


@gmail.op(
    "send",
    "POST /v1/messages",
    tool="send_email",
    params=SendParams,
    output=SentMessage,
    kind=VISIBLE,
    plain="send an email",
    status=201,
    invalid=SEND_INPUTS,
    description="Send an email from the user's Gmail: a saved draft by its id, or a "
    "new message. The recipient sees it; a sent draft leaves Drafts.",
    logged=lambda p: p.model_dump(mode="json", exclude_none=True),
)
def send_email(state: Gmail, p: SendParams):
    """Send a saved draft (which leaves Drafts), or compose and send at once."""
    recipient, subject, body = p.recipient, p.subject, p.body
    if p.draft_id is not None:
        draft = find(state.drafts, p.draft_id, "Draft not found")
        if draft.recipient is None:
            raise Refused("Draft has no recipient", 422)
        if not draft.body.strip():
            raise Refused("Draft has no body", 422)
        state.drafts.remove(draft)
        recipient, subject, body = draft.recipient, draft.subject, draft.body
    message = SentMessage(
        id=state.next_id("sent"),
        sender=state.owner,
        recipient=recipient,
        subject=subject or "",
        body=body,
        draft_id=p.draft_id,
    )
    state.sent.append(message)
    return message
