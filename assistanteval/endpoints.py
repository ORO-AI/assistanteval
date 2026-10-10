"""The models AssistantEval calls, other than the assistant under test.

Each role is set on its own, so any OpenAI-compatible API can fill it: a hosted
provider, OpenRouter, or a local vLLM or Ollama server. For a role ``R``:

- ``ASSISTANTEVAL_R_MODEL``: the model id. Unset leaves the role empty.
- ``ASSISTANTEVAL_R_BASE_URL``: the API's base URL (default: the OpenAI client's,
  which honours ``OPENAI_BASE_URL``).
- ``ASSISTANTEVAL_R_API_KEY``: its key (default: ``OPENAI_API_KEY``).
- ``ASSISTANTEVAL_R_API``: ``openai`` (default), or ``jev`` for TypeSafe's Jev,
  which only the READER role can use.
- ``ASSISTANTEVAL_R_EXTRA_BODY``: JSON added to every request, for options a
  provider needs, such as ``{"reasoning": {"effort": "low"}}``.

The roles:

- ``READER``: the first judge. It reads each assistant turn for the
  simulated user, and answers the grader's questions first.
- ``JUDGE``: the second judge. It answers what the reader is unsure about, and
  the simulated user's approval rules. Required for a live run or live grading.
  Set READER and JUDGE to different models: the grader re-checks some
  of the judge's answers with the reader alone.
- ``WRITER``: rewords the simulated user's fixed replies in the persona's
  voice. Unset, the fixed replies are sent as they are.
- ``BASELINE``: the model in the model baseline assistant.
- ``CHECKER``: optional, a model other than the JUDGE. The grader re-checks the
  simulated user's approval-rule decisions, which the judge made, with the
  reader; when the reader is unsure, the checker answers before a person does.
  It also answers whether a result holds despite the simulated user's
  misreadings, which otherwise waits for a person.

Records keep each role's model, API and base URL, never its key.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

ROLES = ("READER", "JUDGE", "WRITER", "BASELINE", "CHECKER")
SETTINGS = ("MODEL", "BASE_URL", "API_KEY", "API", "EXTRA_BODY")


@dataclass(frozen=True)
class Endpoint:
    model: str
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    api: str = "openai"
    extra_body: dict | None = None

    def client(self, **options):
        """An OpenAI client for this endpoint."""
        from openai import OpenAI

        # A local server often needs no key, but the client insists on one.
        key = self.api_key or os.environ.get("OPENAI_API_KEY") or "unused"
        return OpenAI(base_url=self.base_url, api_key=key, **options)

    @property
    def url(self) -> str | None:
        """The base URL requests go to, when it is not the API's default."""
        if self.api == "openai":
            return self.base_url or os.environ.get("OPENAI_BASE_URL") or None
        return self.base_url

    @property
    def identity(self) -> str:
        """What saved answers are keyed by: the model, and the server and
        request options when they are not the default, so a judge asked
        elsewhere or otherwise never reuses another's answers."""
        extra = json.dumps(self.extra_body, sort_keys=True) if self.extra_body else ""
        return " ".join(filter(None, (self.model, self.url and f"@ {self.url}", extra)))

    def describe(self) -> dict:
        """What a record keeps about the endpoint: never its key."""
        return {"model": self.model, "api": self.api, "base_url": self.url}


def endpoint(role: str) -> Endpoint | None:
    """The role's endpoint from the environment, or None if its model is unset."""

    def setting(name: str) -> str | None:
        return os.environ.get(f"ASSISTANTEVAL_{role}_{name}", "").strip() or None

    if (model := setting("MODEL")) is None:
        return None
    api, extra = setting("API") or "openai", setting("EXTRA_BODY")
    if api not in ("openai", "jev") or (api == "jev" and role != "READER"):
        raise ValueError(
            f"ASSISTANTEVAL_{role}_API must be openai, or jev for READER only"
        )
    return Endpoint(
        model, setting("BASE_URL"), setting("API_KEY"), api, extra and json.loads(extra)
    )


def recorded() -> dict:
    """Every set role's endpoint, for a run's manifest."""
    return {
        role.lower(): found.describe() for role in ROLES if (found := endpoint(role))
    }


def required(role: str) -> Endpoint:
    found = endpoint(role)
    if found is None:
        raise ValueError(
            f"Set ASSISTANTEVAL_{role}_MODEL (and its _BASE_URL and _API_KEY "
            "for a non-OpenAI API); see assistanteval/endpoints.py"
        )
    return found
