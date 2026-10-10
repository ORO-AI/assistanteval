"""The synthetic services: each module declares one on `assistanteval.world`.

The helpers below are defined before the modules are imported, since each
module imports them from here.
"""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from assistanteval.world import Refused

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
EmailAddress = Annotated[
    str, Field(pattern=r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
]


class NoParams(BaseModel):
    """Inputs of an operation that takes none."""


def find(items, item_id: str, message: str):
    """The item with this id, or a 404 refusal."""
    for item in items:
        if item.id == item_id:
            return item
    raise Refused(message, 404)


def new_id(prefix: str, used, start: int = 1) -> str:
    """The first free "<prefix>-0001"-style id, counting from `start`."""
    used, number = set(used), start
    while f"{prefix}-{number:04d}" in used:
        number += 1
    return f"{prefix}-{number:04d}"


def matches(query: str | None, *fields) -> bool:
    """Every word of the query appears somewhere in the fields, ignoring case.

    Word by word, not as one phrase: a model searching "Wednesday cover shift"
    must find the message that asks "Could you cover Wednesday?"."""
    if query is None:
        return True
    words, text = query.casefold().split(), " ".join(f for f in fields if f).casefold()
    return bool(words) and all(word in text for word in words)  # no words, no match


from assistanteval.services import (  # noqa: E402,F401 - importing registers them
    amazon,
    calendar,
    contacts,
    courier,
    email,
    flights,
    gmail,
    marketplace,
    subscriptions,
)
