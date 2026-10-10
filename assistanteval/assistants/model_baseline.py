"""The model baseline: a chosen model in a minimal tool loop, run like a product.

`ModelBaselineChat` and `open_chat` are the task runner's adapter. The model is
any model behind an OpenAI-compatible chat API, set as the BASELINE role
(``ASSISTANTEVAL_BASELINE_MODEL``, ``_BASE_URL`` and ``_API_KEY``; see
`assistanteval.endpoints`): OpenRouter, a provider's own API, or a local vLLM or
Ollama server. It sees each task connector as tools generated from the
service's ``/openapi.json``, the description a product's custom API connector
gets, and nothing else: no memory, no product features. So a run measures the
model and this small harness, which the manifest pins.

There is no product account: each run starts a new chat with an empty
history, so the reset is proven by the chat being new. The model's loop runs
on a worker thread, so ``read`` never waits on it.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import httpx

from assistanteval.contracts import EpisodeStepError, Message, ResetEvidence
from assistanteval.endpoints import required

HARNESS_VERSION = "3"  # 2: a null query argument is omitted; 3: the neutral prompt only
MAX_STEPS = 40  # model calls for one user message
MAX_RESULT_CHARS = 20_000  # of one tool result, as the model sees it
MODEL_TIMEOUT_SECONDS = 300
TOOL_TIMEOUT_SECONDS = 60
# The role and the tools only, as a product's assistant starts. Records keep
# its name, "neutral", and its hash in the manifest.
SYSTEM_PROMPT = (
    "You are the user's personal assistant. You can act on their accounts "
    "through the tools provided."
)


def openapi_tools(service: str, schema: dict) -> list[dict]:
    """One function tool per operation, its path, query and body inputs as
    one object. Each tool keeps its ``route``: the method, path and where
    each input goes."""
    refs = schema.get("components", {}).get("schemas", {})

    def resolve(node, seen=()):
        if isinstance(node, dict):
            if "$ref" in node:
                name = node["$ref"].rsplit("/", 1)[-1]
                if name in seen:  # recursive schema: leave it open
                    return {}
                return resolve(refs[name], (*seen, name))
            return {key: resolve(value, seen) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(value, seen) for value in node]
        return node

    prefix = re.sub(r"\W+", "_", service).strip("_").lower()
    tools = []
    for path, methods in schema.get("paths", {}).items():
        for method, operation in methods.items():
            properties, required, where = {}, [], {}
            for parameter in operation.get("parameters", []):
                name = parameter["name"]
                spec = resolve(parameter.get("schema", {}))
                if parameter.get("description"):
                    spec = {**spec, "description": parameter["description"]}
                properties[name], where[name] = spec, parameter["in"]
                if parameter.get("required"):
                    required.append(name)
            content = operation.get("requestBody", {}).get("content", {})
            if "application/json" in content:
                body = resolve(content["application/json"]["schema"])
                for name, spec in body.get("properties", {}).items():
                    properties[name], where[name] = spec, "body"
                required += body.get("required", [])
            tools.append(
                {
                    "name": f"{prefix}__{_operation_name(operation, path, method)}",
                    "description": operation.get("summary")
                    or operation.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                    "route": (method.upper(), path, where),
                }
            )
    return tools


def _operation_name(operation: dict, path: str, method: str) -> str:
    """The operation's own name: FastAPI appends the path and method to it."""
    name = operation.get("operationId") or f"{method}{path}"
    suffix = re.sub(r"\W", "_", path) + f"_{method.lower()}"
    if name.endswith(suffix) and len(name) > len(suffix):
        name = name[: -len(suffix)]
    return re.sub(r"\W", "_", name)


def chat_model(endpoint):
    """``(messages, tools) -> (assistant message, usage)`` over the endpoint's
    OpenAI-compatible API. The message is kept whole, reasoning included,
    so providers that need it back on the next call get it."""
    client = endpoint.client(timeout=MODEL_TIMEOUT_SECONDS, max_retries=5)

    def call(messages: list[dict], tools: list[dict]) -> tuple[dict, dict]:
        response = client.chat.completions.create(
            model=endpoint.model,
            messages=messages,
            tools=[{"type": "function", "function": tool} for tool in tools] or None,
            extra_body=endpoint.extra_body,
        )
        if not response.choices:
            raise EpisodeStepError("model_returned_no_choice")
        message = response.choices[0].message.model_dump(exclude_none=True)
        usage = response.usage.model_dump(exclude_none=True) if response.usage else {}
        return message, usage

    return call


class ModelBaselineChat:
    """A model and its tool loop behind the four-method `Assistant` contract.

    `send` adds the user's message and, if the loop is idle, starts it on a
    thread: model, then any tool calls, until the model answers without
    calling a tool. A message sent meanwhile joins the next model call. Each
    piece of text the model writes is one message, never edited, so its
    position is its id.
    """

    def __init__(self, model, http, *, now=None):  # an httpx.Client
        self.model = model
        self.http = http
        self.now = now or (lambda: datetime.now().astimezone())
        self.tools: dict[str, tuple[dict, str, str]] = {}  # name -> tool, url, key
        self.history: list[dict] = []  # what the model sees, bar the system prompt
        self.usage: dict[str, int] = {}
        self.steps = 0
        self.step_limits = 0  # user messages whose loop hit MAX_STEPS
        self.error: str | None = None
        self._messages: list[Message] = []
        self._pending: list[str] = []  # sent, not yet shown to the model
        self._lock = threading.Lock()
        self._loop: threading.Thread | None = None
        self._looping = False  # cleared, under the lock, only as the loop ends
        self.answering = False  # whether the loop ran when read() took its messages
        self._closed = False

    def reset(self) -> ResetEvidence:
        fresh = not (self.history or self._messages or self.tools or self._loop)
        return ResetEvidence(fresh, "new_chat_empty_history")

    def connect(self, connectors) -> list[str]:
        for connector in connectors:
            if connector.kind != "openapi_api_key":
                raise EpisodeStepError("connector_kind_unsupported")
            response = self.http.get(f"{connector.url}/openapi.json")
            if response.status_code != 200:
                raise EpisodeStepError("openapi_unavailable")
            for tool in openapi_tools(connector.name, response.json()):
                if len(tool["name"]) > 64 or tool["name"] in self.tools:
                    raise EpisodeStepError("tool_name_unusable")
                self.tools[tool["name"]] = (tool, connector.url, connector.key)
        return []  # nothing is said while connecting

    def send(self, text: str) -> None:
        with self._lock:
            if self.error:
                raise EpisodeStepError(self.error)
            self._pending.append(text)
            if self._looping:
                return  # the running loop takes it at its next model call
            self._looping = True
            self._loop = threading.Thread(target=self._run, daemon=True)
            self._loop.start()

    def read(self) -> tuple[list[Message], list[dict]]:
        with self._lock:
            if self.error:
                raise EpisodeStepError(self.error)
            # With the messages: while the loop runs, its latest text may not be its answer.
            self.answering = self._looping
            return list(self._messages), []  # nothing asks for permission

    def close(self) -> None:
        with self._lock:
            self._pending.clear()
            loop, self._closed = self._loop, True
        if loop:
            loop.join(timeout=MODEL_TIMEOUT_SECONDS + 10)

    def _run(self) -> None:
        """Answer every pending message, then stop."""
        try:
            while self._take_pending():
                self._answer()
        except Exception as error:  # noqa: BLE001 - its text may carry a key
            # Its type and HTTP status only: enough to tell a rate limit from
            # a bad request, with nothing that could carry a key.
            status = getattr(error, "status_code", None) or getattr(
                getattr(error, "response", None), "status_code", None
            )
            print(
                f"Model loop failed: {type(error).__name__} {status or ''}", flush=True
            )
            with self._lock:
                self._looping = False
                self.error = (
                    str(error)
                    if isinstance(error, EpisodeStepError)
                    else "model_loop_failed"
                )

    def _take_pending(self) -> bool:
        with self._lock:
            if self._closed or not self._pending:
                self._looping = False
                return False
            self.history += [{"role": "user", "content": t} for t in self._pending]
            self._pending.clear()
            return True

    def _answer(self) -> None:
        specs = [
            {key: tool[key] for key in ("name", "description", "parameters")}
            for tool, _, _ in self.tools.values()
        ]
        for _ in range(MAX_STEPS):
            if self._closed:
                return
            if self._pending:  # the user spoke again: the model sees it now
                self._take_pending()
            message, usage = self.model(
                [{"role": "system", "content": self._system()}, *self.history], specs
            )
            self.steps += 1
            for key, value in usage.items():
                if isinstance(value, int):
                    self.usage[key] = self.usage.get(key, 0) + value
            message["role"] = "assistant"
            self.history.append(message)
            if text := (message.get("content") or "").strip():
                with self._lock:
                    self._messages.append(Message(str(len(self._messages)), text))
            calls = message.get("tool_calls") or []
            if not calls:
                return
            for call in calls:
                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": self._call(call["function"]),
                    }
                )
        # The model's own failure to finish, not a setup error: it goes quiet
        # and the runner nudges, as with a product that stalls.
        self.step_limits += 1

    def _system(self) -> str:
        now = self.now()
        return (
            f"{SYSTEM_PROMPT}\n\nThe user's current local time is "
            f"{now.strftime('%A, %B %-d, %Y, %-I:%M %p')} ({now.tzname()})."
        )

    def _call(self, function: dict) -> str:
        """Run one tool call against its service; the model sees the HTTP
        status and body, as a developer calling the API would."""
        name = function.get("name", "")
        if name not in self.tools:
            return json.dumps({"error": f"No tool named {name}"})
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except ValueError:
            return json.dumps({"error": "Arguments are not valid JSON"})
        if not isinstance(arguments, dict):
            return json.dumps({"error": "Arguments must be a JSON object"})
        tool, url, key = self.tools[name]
        method, path, where = tool["route"]
        for field in re.findall(r"{(\w+)}", path):
            if field not in arguments:
                return json.dumps({"error": f"Missing required argument {field}"})
            value = str(arguments[field])
            path = path.replace("{" + field + "}", quote(value, safe=""))
        # A query string cannot say null: a null argument is left out, as omitted.
        query = {
            k: v
            for k, v in arguments.items()
            if where.get(k) == "query" and v is not None
        }
        body = {k: v for k, v in arguments.items() if where.get(k) == "body"}
        try:
            response = self.http.request(
                method,
                f"{url}{path}",
                params=query or None,
                json=body if body or method in ("POST", "PUT", "PATCH") else None,
                headers={"Authorization": f"Bearer {key}"},
            )
        except httpx.HTTPError:
            return json.dumps({"error": "The service could not be reached"})
        return json.dumps(
            {"status": response.status_code, "body": response.text[:MAX_RESULT_CHARS]}
        )


def manifest() -> dict:
    """What a run of this assistant measured, for the run manifest."""
    endpoint = required("BASELINE")
    return {
        "model_baseline": {
            "model": endpoint.model,
            "base_url": endpoint.base_url,
            "harness_version": HARNESS_VERSION,
            "system_prompt": "neutral",
            "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
            "max_steps": MAX_STEPS,
            "tools_from": "openapi",
        }
    }


@contextmanager
def open_chat(account, evidence_dir: Path, *, model=None, http=None):
    """A new chat with the configured model; it needs no account."""
    if model is None:
        model = chat_model(required("BASELINE"))
    client = http or httpx.Client(timeout=TOOL_TIMEOUT_SECONDS)
    chat = ModelBaselineChat(model, client)
    try:
        yield chat
    finally:
        chat.close()
        if http is None:
            client.close()
        evidence_dir = Path(evidence_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        # The loop's full working: every message and tool result the model
        # saw, and the tokens it used.
        trace = {
            "steps": chat.steps,
            "step_limits": chat.step_limits,
            "usage": chat.usage,
            "error": chat.error,
            "tools": sorted(chat.tools),
            "history": chat.history,
        }
        (evidence_dir / "model_baseline_trace.json").write_text(
            json.dumps(trace, indent=1, default=str)
        )


open_chat.connector_kind = "openapi_api_key"
open_chat.manifest = manifest
