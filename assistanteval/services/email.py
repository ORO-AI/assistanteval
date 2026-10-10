"""Email: search a mailbox, and send only from and to the allowed addresses."""

from pydantic import BaseModel, Field

from assistanteval.services import EmailAddress, Text, matches, new_id
from assistanteval.world import READ, VISIBLE, Refused, service


class EmailMessage(BaseModel):
    id: str
    sender: str
    subject: str
    body: str


class SendRequest(BaseModel):
    sender: EmailAddress
    recipient: EmailAddress
    subject: Text
    body: Text


class SentMessage(SendRequest):
    id: str


class Mailbox(BaseModel):
    messages: list[EmailMessage]
    sent: list[SentMessage] = Field(default_factory=list)
    allowed_senders: list[EmailAddress] = Field(default_factory=list)
    allowed_recipients: list[EmailAddress] = Field(default_factory=list)


class SearchParams(BaseModel):
    query: Text = Field(
        description="Words to find in a message's sender, recipient, subject or body"
    )


email = service("email", "Email", Mailbox)


@email.op(
    "search",
    "GET /v1/messages",
    tool="search_messages",
    params=SearchParams,
    output=Mailbox,
    kind=READ,
    description="Search the user's mailbox for messages containing every one of the words.",
)
def search_messages(state: Mailbox, p: SearchParams):
    return state.model_copy(
        update={
            "messages": [
                m
                for m in state.messages
                if matches(p.query, m.sender, m.subject, m.body)
            ],
            "sent": [
                m
                for m in state.sent
                if matches(p.query, m.sender, m.recipient, m.subject, m.body)
            ],
        }
    )


@email.op(
    "send",
    "POST /v1/messages",
    tool="send_message",
    params=SendRequest,
    output=SentMessage,
    kind=VISIBLE,
    plain="send an email",
    status=201,
    description="Send an email from the user's address. The recipient sees it.",
)
def send_message(state: Mailbox, p: SendRequest):
    if (
        p.sender not in state.allowed_senders
        or p.recipient not in state.allowed_recipients
    ):
        raise Refused("Sender or recipient is not allowed", 422)
    used = [m.id for m in (*state.messages, *state.sent)]
    sent = SentMessage(id=new_id("sent", used, len(state.sent) + 1), **p.model_dump())
    state.sent.append(sent)
    return sent
