"""Contacts: a read-only name lookup."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from assistanteval.services import matches
from assistanteval.world import READ, service


class Contact(BaseModel):
    id: str
    name: str
    phone: str
    address: str | None = None


class ContactBook(BaseModel):
    contacts: list[Contact]


class FindParams(BaseModel):
    name: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=100),
        Field(description="All or part of a contact's name"),
    ]


contacts = service("contacts", "Contacts", ContactBook, "Read-only synthetic contacts.")


@contacts.op(
    "lookup",
    "GET /v1/contacts",
    tool="find_contacts",
    params=FindParams,
    output=ContactBook,
    kind=READ,
    description="Find contacts by all or part of a name.",
    exclude_none=True,
)
def find_contacts(state: ContactBook, p: FindParams):
    return ContactBook(contacts=[c for c in state.contacts if matches(p.name, c.name)])
