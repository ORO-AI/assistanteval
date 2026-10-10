"""Transport-free contracts between the task runner and its assistant adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, Sequence


_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")


def _safe_code(value: str) -> None:
    if not _CODE.fullmatch(value):
        raise ValueError("Evidence codes must be short lowercase identifiers")


@dataclass(frozen=True)
class ResetEvidence:
    verified: bool
    method: str  # Fixed, credential-free evidence code.
    # False when the reset is verified but leaves some state an earlier run
    # could have touched, such as a machine the assistant shares across chats.
    isolated: bool = True

    def __post_init__(self) -> None:
        _safe_code(self.method)


class EpisodeStepError(Exception):
    """A step the assistant refused or failed, as a fixed, credential-free code."""

    def __init__(self, code: str):
        _safe_code(code)
        super().__init__(code)


class ConnectorDescriptor(Protocol):
    kind: str


@dataclass(frozen=True)
class Message:
    """One assistant message as the assistant shows it now.

    ``id`` stays the same while the message is shown; a new ``text``
    under the same ``id`` is the assistant rewriting or still writing it.
    """

    id: str
    text: str


class Assistant(Protocol):
    """One assistant, driven by ``assistanteval.run_task``.

    Its factory, ``open_chat(account, evidence_dir)`` in any module, is a
    context manager that starts the assistant (raising if it cannot) and
    cleans up the run's connectors on exit; the runner passes ``account=None``.
    The factory names the connector transport it uses as ``connector_kind``,
    and may name what the run measured (model, harness version) with a
    ``manifest()`` function.
    """

    def reset(self) -> ResetEvidence: ...

    # Connect every service, then wait until the assistant stops posting about
    # setup. Returns what it posted meanwhile.
    def connect(self, connectors: Sequence[ConnectorDescriptor]) -> list[str]: ...

    # Deliver one user message, confirming delivery even when the same text
    # was sent before. Raises EpisodeStepError("prompt_delivery_unconfirmed")
    # when the message may have arrived but delivery was not confirmed: the
    # runner records it as sent, so the run is not retried as fresh.
    def send(self, text: str) -> None: ...

    # Every assistant message since the first send, and each permission
    # prompt the adapter answered since the last read, as {"label", "detail"}:
    # the button or outcome, and what the prompt asked. Never waits.
    def read(self) -> tuple[list[Message], list[dict]]: ...

    # Optional, read with getattr: ``answering``, set by read() together with
    # the messages it returns, is True when the answer was still coming at that
    # read (a tool loop still running, a "working" indicator). The runner holds
    # the user meanwhile, so it must clear whenever the adapter stops.


class Stop(Exception):
    """A stop signal, the run's deadline, or a host's reason to end the run now
    (such as a lost account lease): the runner saves the record, with `code`
    as its error, and releases the services."""

    def __init__(self, code: str = "stopped"):
        super().__init__(code)
        self.code = code
