"""Every synthetic service as typed JSON state plus declared operations.

A service is a pydantic state model and one function per operation, which
mutates the state and returns the output. This module does the rest:
sessions (one JSON object each, in a local directory, or any backend with
the same get and put), validation,
the request log, and one HTTP handler for REST, OpenAPI and MCP. A call is
validated, applied and logged in one compare-and-swap write, so a failed
call changes nothing. Only pydantic is needed.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, Callable
from urllib.parse import parse_qsl, unquote

from pydantic import BaseModel, ConfigDict, ValidationError, create_model
from pydantic.json_schema import models_json_schema

READ, PRIVATE_NEW, PRIVATE_CHANGE, VISIBLE, MONEY = (
    "read",
    "private_new",
    "private_change",
    "visible_to_others",
    "money",
)
log = logging.getLogger("assistanteval.world")


class Refused(Exception):
    """A refusal the assistant sees, with its HTTP status."""

    def __init__(self, message: str, status: int = 403):
        super().__init__(message)
        self.status = status


def inactive() -> Refused:
    return Refused("Invalid API key", 401)


@dataclass(frozen=True)
class Op:
    service: str
    name: str  # "calendar.update": the record's and the grader's name
    tool: str  # the MCP tool; with the path, the OpenAPI operationId
    method: str
    path: str  # "{field}" placeholders come from params
    params: type[BaseModel]
    output: type[BaseModel]
    kind: str  # READ ... MONEY: the grader's approval and reporting rules
    description: str
    fn: Callable[[Any, Any], Any]  # (state, params) -> output; mutates state
    plain: str = ""  # "change or move a calendar event": the change in the user's words
    status: int = 200  # REST success status
    invalid: str = "Invalid request"  # the message for rejected inputs
    exclude_none: bool = False  # REST body and logged output drop nulls
    logged: Callable[[Any], dict] | None = None  # the logged input; default every param

    @cached_property
    def route(self) -> re.Pattern:
        return re.compile(re.sub(r"{(\w+)}", r"(?P<\1>[^/]+)", self.path))

    @cached_property
    def fields(self) -> dict:
        """Inputs outside the path: the query for GET, else the JSON body."""
        in_path = re.findall(r"{(\w+)}", self.path)
        return {n: f for n, f in self.params.model_fields.items() if n not in in_path}

    @cached_property
    def checked(self) -> type[BaseModel]:
        """The body checked before the call, as FastAPI checks it: the fields, not the validators."""
        fields = {n: (f.annotation, f) for n, f in self.fields.items()}
        return create_model(
            f"{self.params.__name__}Body", __config__=self.params.model_config, **fields
        )

    @cached_property
    def arguments(self) -> type[BaseModel]:
        """The arguments checked before the call, as the MCP SDK checks them: every field, no validators."""
        fields = {
            n: (Annotated[f.annotation, f], ... if f.is_required() else f.default)
            for n, f in self.params.model_fields.items()
        }
        return create_model(
            f"{self.tool}Arguments",
            __config__=ConfigDict(arbitrary_types_allowed=True),
            **fields,
        )


@dataclass
class Service:
    name: str
    title: str
    state: type[BaseModel]
    description: str | None = None
    ops: dict[str, Op] = field(default_factory=dict)

    def op(self, name: str, http: str, **declared):
        """Declare the decorated function as operation `name` at "METHOD /path"."""
        method, path = http.split(" ", 1)

        def register(fn):
            op = Op(
                self.name,
                f"{self.name}.{name}",
                method=method,
                path=path,
                fn=fn,
                **declared,
            )
            self.ops[op.tool] = op
            return fn

        return register


SERVICES: dict[str, Service] = {}


def service(
    name: str, title: str, state: type[BaseModel], description: str | None = None
) -> Service:
    SERVICES[name] = Service(name, title, state, description)
    return SERVICES[name]


def dump(model: BaseModel) -> dict:
    return model.model_dump(mode="json", by_alias=True)


# Sessions: {service, key_sha256, fixture_sha256, start_state, state, log,
# revoked, expires_at}, updated by compare-and-swap on a two-method backend.


class Conflict(Exception):
    """The session changed since it was read."""


class DirBackend:
    """Local and test sessions: one file each, a lock for the swap."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def get(self, sid: str) -> tuple[dict, str] | None:
        if not (path := self.root / f"{sid}.json").exists():
            return None
        raw = path.read_bytes()
        return json.loads(raw), hashlib.sha256(raw).hexdigest()

    def put(self, sid: str, obj: dict, etag: str | None) -> None:
        with open(self.root / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if ((current := self.get(sid)) and current[1]) != etag:
                raise Conflict(sid)
            (tmp := self.root / f"{sid}.tmp").write_bytes(
                json.dumps(obj, sort_keys=True).encode()
            )
            tmp.replace(self.root / f"{sid}.json")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _active(session: dict) -> bool:
    return (
        not session.get("revoked", True) and session.get("expires_at", 0) > time.time()
    )


class Sessions:
    def __init__(self, backend, ttl_seconds: int = 3 * 3600):  # past any run's deadline
        self.backend, self.ttl = backend, ttl_seconds

    def create(self, service_name: str, fixture: dict) -> tuple[str, str]:
        """A new session on this fixture, and its key. A bad fixture raises."""
        state = dump(SERVICES[service_name].state.model_validate(fixture))
        sid = secrets.token_hex(16)
        key = f"{sid}.{secrets.token_urlsafe(32)}"
        self.backend.put(
            sid,
            {
                "service": service_name,
                "key_sha256": _sha(key),
                "fixture_sha256": _sha(json.dumps(fixture, sort_keys=True)),
                "start_state": state,
                "state": state,
                "log": [],
                "revoked": False,
                "expires_at": time.time() + self.ttl,
            },
            None,
        )
        return sid, key

    def resolve(self, key: str, service_name: str) -> str | None:
        """The live session this key opens for this service, or None."""
        sid = key.partition(".")[0]
        if not re.fullmatch(r"[0-9a-f]{32}", sid) or not (
            found := self.backend.get(sid)
        ):
            return None
        s = found[0]
        ok = s.get("service") == service_name and hmac.compare_digest(
            s.get("key_sha256", ""), _sha(key)
        )
        return sid if ok and _active(s) else None

    def read(self, sid: str) -> dict:
        return self.backend.get(sid)[0]

    def update(self, sid: str, change: Callable[[dict], Any]) -> Any:
        for _ in range(5):
            session, etag = self.backend.get(sid)
            result = change(session)
            try:
                self.backend.put(sid, session, etag)
                return result
            except Conflict:
                continue
        raise RuntimeError("session busy")

    def revoke(self, sid: str) -> None:
        self.update(sid, lambda s: s.update(revoked=True))


def invoke(
    sessions: Sessions, sid: str, op: Op, args: dict, rejected=None
) -> tuple[int, Any]:
    """Apply one call and log it with the state change, in one write.

    Returns 200 and the output model, or the status and error detail. A call
    the transport already refused (``rejected``, its 422 detail) is only
    logged. A session no longer live raises `inactive()` and logs nothing.
    """

    def apply(session: dict) -> tuple[int, Any]:
        if not _active(session):
            raise inactive()
        entry = {
            "operation": op.name,
            "kind": op.kind,
            "at_ms": int(time.time() * 1000),
            "input": args,
            "output": None,
            "status": "ok",
            "error": None,
        }
        try:
            if rejected is not None:
                raise Refused(op.invalid, 422)
            try:
                params = op.params.model_validate(args)
            except ValidationError:
                raise Refused(op.invalid, 422) from None
            state = SERVICES[op.service].state.model_validate(session["state"])
            output = op.output.model_validate(op.fn(state, params))
            session["state"] = dump(state)
            entry["input"] = (
                op.logged(params) if op.logged else params.model_dump(mode="json")
            )
            entry["output"] = output.model_dump(
                mode="json", by_alias=True, exclude_none=op.exclude_none
            )
            result = 200, output
        except Refused as exc:
            entry |= {"status": "refused", "error": str(exc)}
            result = exc.status, str(exc) if rejected is None else rejected
        except Exception as exc:  # noqa: BLE001 - the harness's fault; its text may hold fixture data
            log.error("%s_failed: %s", op.name, type(exc).__name__)
            detail = f"{SERVICES[op.service].title} fixture unavailable"
            entry |= {"status": "error", "error": detail}
            result = 503, detail
        session["log"].append(entry)
        return result

    return sessions.update(sid, apply)


# One HTTP entry point for REST, OpenAPI and MCP. ----------------------------

Response = tuple[int, dict, bytes]
BEARER = {"WWW-Authenticate": "Bearer"}


def _json(status: int, body, headers=None) -> Response:
    data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
    return status, {"content-type": "application/json", **(headers or {})}, data


def handle(
    sessions, method, path, query, headers, body, host="", scheme="https"
) -> Response:
    """Serve one request. The host label routes: <service>.<domain> is REST,
    <service>-mcp.<domain>/mcp is MCP. Under any other host, /<service>/...
    and /<service>/mcp do."""
    label = host.partition(".")[0]
    name, mcp = label.removesuffix("-mcp"), label.endswith("-mcp")
    origin = f"{scheme}://{host}"
    if name not in SERVICES:
        name, _, rest = path.lstrip("/").partition("/")
        path, mcp, origin = "/" + rest, rest == "mcp", f"{origin}/{name}"
    if (svc := SERVICES.get(name)) is None:
        return 404, {}, b""
    scheme_, _, key = {k.lower(): v for k, v in headers.items()}.get(
        "authorization", ""
    ).partition(" ")
    sid = sessions.resolve(key, name) if scheme_.lower() == "bearer" and key else None
    try:
        if not mcp:
            return _rest(sessions, sid, svc, method, path, query, body, origin)
        if sid is None:
            raise inactive()
        if path != "/mcp":
            return 404, {"content-type": "text/plain; charset=utf-8"}, b"Not Found"
        return _mcp(sessions, sid, svc, method, body)
    except Refused as exc:  # the key opens no live session
        return (401, BEARER, b"") if mcp else _json(401, {"detail": str(exc)}, BEARER)


def _rest(sessions, sid, svc: Service, method, path, query, body, origin) -> Response:
    if path == "/openapi.json":
        return _json(200, openapi(svc))
    hits = [(op, m) for op in svc.ops.values() if (m := op.route.fullmatch(path))]
    if not hits:
        if path.endswith("/") and any(
            op.route.fullmatch(path.rstrip("/")) for op in svc.ops.values()
        ):
            return (
                307,
                {
                    "location": origin
                    + path.rstrip("/")
                    + (f"?{query}" if query else "")
                },
                b"",
            )
        return _json(404, {"detail": "Not Found"})
    op, found = next(((op, m) for op, m in hits if op.method == method), (None, None))
    if op is None:  # Starlette names the first route's method
        return _json(
            405, {"detail": "Method Not Allowed"}, {"allow": hits[0][0].method}
        )
    if sid is None:
        raise inactive()
    args, errors = _rest_inputs(
        op, {k: unquote(v) for k, v in found.groupdict().items()}, query, body
    )
    status, out = invoke(sessions, sid, op, args, errors or None)
    if status != 200:
        return _json(status, {"detail": out})
    return _json(
        op.status,
        out.model_dump(mode="json", by_alias=True, exclude_none=op.exclude_none),
    )


def _rest_inputs(op: Op, args: dict, query: str, body: bytes) -> tuple[dict, list]:
    """The call's inputs, and FastAPI's 422 detail if its field checks refuse them."""
    if op.method == "GET":
        data = dict(parse_qsl(query, keep_blank_values=True))
        args, where, data = (
            args | data,
            "query",
            {n: v for n, v in data.items() if n in op.fields},
        )
    elif not op.fields:
        return args, []
    else:
        where = "body"
        try:
            data = json.loads(body) if body else None
        except ValueError as exc:
            return args, [
                {
                    "type": "json_invalid",
                    "loc": ["body", exc.pos],
                    "msg": "JSON decode error",
                    "input": {},
                    "ctx": {"error": exc.msg},
                }
            ]
        if data is None:
            return args, [
                {
                    "type": "missing",
                    "loc": ["body"],
                    "msg": "Field required",
                    "input": None,
                }
            ]
        args = (data if isinstance(data, dict) else {}) | args  # the path wins
    try:
        op.checked.model_validate(data, from_attributes=True)
        return args, []
    except ValidationError as exc:
        return args, [_fastapi_error(e, where) for e in exc.errors(include_url=False)]


def _fastapi_error(error: dict, where: str) -> dict:
    ctx = {
        k: {} if isinstance(v, Exception) else v
        for k, v in error.get("ctx", {}).items()
    }
    missing_query = (
        where == "query" and error["type"] == "missing"
    )  # FastAPI gives no input for it
    return {
        "type": error["type"],
        "loc": [where, *error["loc"]],
        "msg": error["msg"],
        "input": None if missing_query else error["input"],
        **({"ctx": ctx} if ctx else {}),
    }


_REF = "#/components/schemas/{model}"
_ERRORS = {  # FastAPI's schemas for its 422 body
    "HTTPValidationError": {
        "properties": {
            "detail": {
                "items": {"$ref": _REF.format(model="ValidationError")},
                "title": "Detail",
                "type": "array",
            }
        },
        "title": "HTTPValidationError",
        "type": "object",
    },
    "ValidationError": {
        "properties": {
            "ctx": {"title": "Context", "type": "object"},
            "input": {"title": "Input"},
            "loc": {
                "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                "title": "Location",
                "type": "array",
            },
            "msg": {"title": "Message", "type": "string"},
            "type": {"title": "Error Type", "type": "string"},
        },
        "required": ["loc", "msg", "type"],
        "title": "ValidationError",
        "type": "object",
    },
}


def openapi(svc: Service) -> dict:
    """The REST API as FastAPI describes it: the same paths, ids and schemas."""
    ops = list(svc.ops.values())
    bodies = {op.tool: op.checked for op in ops if op.method != "GET" and op.fields}
    models = [(m, "validation") for m in bodies.values()] + [
        (op.output, "serialization") for op in ops
    ]
    refs, defs = models_json_schema(models, ref_template=_REF)
    paths: dict = {}
    for op in ops:
        spec = _operation(op, refs, bodies.get(op.tool))
        paths.setdefault(op.path, {})[op.method.lower()] = spec
    info = {"title": f"{svc.title} API", "version": "0.1.0"}
    return _no_null_defaults(
        {
            "openapi": "3.1.0",
            "info": info
            | ({"description": svc.description} if svc.description else {}),
            "paths": paths,
            "components": {
                "schemas": defs.get("$defs", {}) | _ERRORS,
                "securitySchemes": {"HTTPBearer": {"type": "http", "scheme": "bearer"}},
            },
        }
    )


def _content(schema: dict) -> dict:
    return {"application/json": {"schema": schema}}


def _operation(op: Op, refs: dict, body: type[BaseModel] | None) -> dict:
    """One operation: its path and query parameters, body and responses."""
    schema = op.params.model_json_schema()["properties"]
    parameters = [
        {
            "name": n,
            "in": "path",
            "required": True,
            "schema": {k: schema[n][k] for k in ("title", "type")},
        }
        for n in sorted(set(op.params.model_fields) - set(op.fields))
    ]
    for n, f in op.fields.items() if op.method == "GET" else ():
        query = {
            "name": n,
            "in": "query",
            "required": f.is_required(),
            "schema": schema[n],
        }
        parameters.append(
            query | ({"description": f.description} if f.description else {})
        )
    ok = {
        "description": "Successful Response",
        "content": _content(refs[(op.output, "serialization")]),
    }
    spec = {
        "summary": op.description,
        "responses": {str(op.status): ok},
        "security": [{"HTTPBearer": []}],
        "operationId": re.sub(r"\W", "_", op.tool + op.path) + f"_{op.method.lower()}",
    }
    if parameters:
        spec["parameters"] = parameters
    if body:
        spec["requestBody"] = {
            "content": _content(refs[(body, "validation")]),
            "required": True,
        }
    if parameters or body:
        invalid = _content({"$ref": _REF.format(model="HTTPValidationError")})
        spec["responses"]["422"] = {
            "description": "Validation Error",
            "content": invalid,
        }
    return spec


def _no_null_defaults(node):
    """Without "default": null, as FastAPI writes it."""
    if isinstance(node, dict):
        return {
            k: _no_null_defaults(v)
            for k, v in node.items()
            if not (k == "default" and v is None)
        }
    return [_no_null_defaults(v) for v in node] if isinstance(node, list) else node


# MCP: stateless streamable HTTP with JSON responses, as the MCP SDK serves it.

HANDSHAKE = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")
MODERN = "2026-07-28"  # no handshake; the version rides in each request's _meta
LISTS = {  # method -> its result's key
    "ping": None,
    "tools/list": "tools",
    "prompts/list": "prompts",
    "resources/list": "resources",
    "resources/templates/list": "resourceTemplates",
}


def tools(svc: Service) -> list[dict]:
    listed = []
    for op in svc.ops.values():
        schema = op.arguments.model_json_schema(by_alias=True)
        if op.params.model_config.get("extra") == "forbid":
            schema["additionalProperties"] = False
        read = op.kind == READ
        hints = {
            "readOnlyHint": read,
            "destructiveHint": not read,
            "idempotentHint": read,
            "openWorldHint": False,
        }
        listed.append(
            {
                "name": op.tool,
                "description": op.description,
                "inputSchema": schema,
                "outputSchema": op.output.model_json_schema(),
                "annotations": hints,
            }
        )
    return listed


def tool_error(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def call_tool(sessions: Sessions, sid: str, op: Op, args: dict) -> dict:
    """Check the fields as the MCP SDK does, then invoke (which runs the op's own checks)."""
    for name, value in args.items():  # the SDK parses JSON sent inside a string
        if (
            (f := op.arguments.model_fields.get(name))
            and f.annotation is not str
            and isinstance(value, str)
        ):
            try:
                parsed = json.loads(value)
                args = args | (
                    {} if isinstance(parsed, str | int | float) else {name: parsed}
                )
            except ValueError:
                pass
    try:
        op.arguments.model_validate(args)
        status, out = invoke(sessions, sid, op, args)
    except ValidationError as exc:
        status, out = invoke(sessions, sid, op, args, rejected=str(exc))
    if status != 200:
        return tool_error(f"Error executing tool {op.tool}: {out}")
    text = [{"type": "text", "text": out.model_dump_json(indent=2, by_alias=True)}]
    return {"content": text, "structuredContent": dump(out), "isError": False}


def _message(body: bytes) -> dict | None:
    """The JSON-RPC message, or None if the SDK would not parse it."""
    try:
        msg = json.loads(body)
    except ValueError:
        return None
    ok = (
        isinstance(msg, dict)
        and isinstance(msg.get("method"), str)
        and isinstance(params := msg.get("params") or {}, dict)
        and isinstance(params.get("arguments") or {}, dict)
    )
    return msg if ok else None


def _mcp(
    sessions: Sessions, sid: str, svc: Service, method: str, body: bytes
) -> Response:
    def error(code: int, message: str, id_="server-error", **extra) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": id_,
            "error": {"code": code, "message": message, **extra},
        }

    if method != "POST":
        return _json(405, error(-32000, "Method Not Allowed"), {"allow": "POST"})
    if (msg := _message(body)) is None:
        return _json(400, error(-32700, "Parse error"))
    params = msg.get("params") or {}
    if "id" not in msg:  # a notification, e.g. notifications/initialized
        return 202, {"content-type": "application/json"}, b""
    rpc = msg["method"]
    modern = (params.get("_meta") or {}).get(
        "io.modelcontextprotocol/protocolVersion"
    ) == MODERN
    info, cache = (
        {"name": svc.title, "version": ""},
        {"cacheScope": "private", "ttlMs": 0} if modern else {},
    )
    if rpc == "initialize":
        version = params.get("protocolVersion")
        result = {
            "protocolVersion": version if version in HANDSHAKE else HANDSHAKE[-1],
            "serverInfo": info,
            "capabilities": {
                "experimental": {},
                "prompts": {"listChanged": False},
                "resources": {"listChanged": False, "subscribe": False},
                "tools": {"listChanged": False},
            },
        }
    elif rpc == "server/discover" and modern:
        result = {
            "supportedVersions": [MODERN],
            **cache,
            "capabilities": {
                "prompts": {"listChanged": True},
                "resources": {"listChanged": True, "subscribe": True},
                "tools": {"listChanged": True},
            },
        }
    elif rpc in LISTS:
        key = LISTS[rpc]
        result = {key: tools(svc) if key == "tools" else [], **cache} if key else {}
    elif rpc == "tools/call":
        if op := svc.ops.get(params.get("name")):
            result = call_tool(sessions, sid, op, params.get("arguments") or {})
        else:
            result = tool_error(f"Unknown tool: {params.get('name')}")
    else:
        return _json(200, error(-32601, "Method not found", msg["id"], data=rpc))
    if modern:
        result |= {
            "_meta": {"io.modelcontextprotocol/serverInfo": info},
            "resultType": "complete",
        }
    return _json(200, {"jsonrpc": "2.0", "id": msg["id"], "result": result})


# Entry points. ---------------------------------------------------------------


def asgi(sessions: Sessions):
    """The same handler as an ASGI app, for tests and local serving."""

    async def app(scope, receive, send):
        body = await read_body(receive)
        headers = {k.decode(): v.decode() for k, v in scope["headers"]}
        reply = handle(
            sessions,
            scope["method"],
            scope.get("raw_path", scope["path"].encode()).decode(),
            scope["query_string"].decode(),
            headers,
            body,
            host=headers.get("host", ""),
            scheme=scope["scheme"],
        )
        await respond(send, *reply)

    return app


async def read_body(receive) -> bytes:
    body, more = b"", True
    while more:
        message = await receive()
        body += message.get("body", b"")
        more = message.get("more_body", False)
    return body


async def respond(send, status: int, headers: dict, payload: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
        }
    )
    await send({"type": "http.response.body", "body": payload})


from assistanteval import services as _services  # noqa: E402,F401 - registers every service
