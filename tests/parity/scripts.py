"""Scripted calls that pin every service's behaviour, whatever serves it.

A recorder ran these against an earlier implementation of the services
(FastAPI routes and the MCP SDK) and saved what came back in `golden/`. One
fixture per service is kept (two for gmail, whose script branches on the draft).
`test_parity.py` replays them against `assistanteval.world` and compares.
The `amazon` goldens were recorded from `assistanteval.world` itself, by
`record_new.py`.

Task fixtures are date templates rendered at each run's start
(`assistanteval.dates`), so the fixed-date fixtures the goldens were recorded
on are kept in `fixtures/` (`fixtures/<task>/<service>.json`; the two gmail
fixtures are written out as the state they load as, which the goldens record).
`tests/test_task_worlds.py` checks that every task fixture, rendered at many
anchors, loads in its service.

A script is a list of steps. Each step names an MCP tool and its arguments;
the REST call is derived from the tool's route below (path fields from the
arguments, the rest as the query for GET and the JSON body otherwise). A
step may override the REST call (``rest``) or run over one transport only
(``only``). An argument ``{"$ref": [i, "id"]}`` is field ``id`` of step
``i``'s output.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = Path(__file__).resolve().parent / "golden"

ROUTES = {
    "amazon": {
        "search_products": ("GET", "/v1/products"),
        "list_addresses": ("GET", "/v1/addresses"),
        "list_payment_cards": ("GET", "/v1/payment-cards"),
        "show_cart": ("GET", "/v1/cart"),
        "list_orders": ("GET", "/v1/orders"),
        "add_to_cart": ("POST", "/v1/cart/{product_id}"),
        "place_order": ("POST", "/v1/orders"),
    },
    "calendar": {
        "search_events": ("GET", "/v1/events"),
        "update_event": ("PATCH", "/v1/events/{event_id}"),
        "create_event": ("POST", "/v1/events"),
    },
    "contacts": {"find_contacts": ("GET", "/v1/contacts")},
    "courier": {
        "list_pickups": ("GET", "/v1/pickups"),
        "save_pickup": ("POST", "/v1/pickups"),
    },
    "email": {
        "search_messages": ("GET", "/v1/messages"),
        "send_message": ("POST", "/v1/messages"),
    },
    "flights": {
        "list_alternatives": ("GET", "/v1/alternatives"),
        "change_flight": ("POST", "/v1/changes/{option_id}"),
    },
    "gmail": {
        "search_mail": ("GET", "/v1/search"),
        "create_draft": ("POST", "/v1/drafts"),
        "update_draft": ("PATCH", "/v1/drafts/{draft_id}"),
        "send_email": ("POST", "/v1/messages"),
    },
    "marketplace": {
        "read_listings": ("GET", "/v1/listings"),
        "reply_to_thread": ("POST", "/v1/threads/{thread_id}/messages"),
    },
    "subscriptions": {
        "list_subscriptions": ("GET", "/v1/subscriptions"),
        "cancel_subscription": ("POST", "/v1/subscriptions/{vendor}/cancel"),
    },
}


def cases() -> list[tuple[str, str]]:
    """(service, fixture path) for every fixture the goldens were recorded on."""
    paths = sorted((ROOT / "tests/parity/fixtures").glob("*/*.json"))
    return [
        (path.stem, str(path.relative_to(ROOT)))
        for path in paths
        if path.stem in ROUTES
    ]


def case_id(service: str, fixture: str) -> str:
    return f"{service}--{Path(fixture).parent.name}"


def ref(step: int, key: str) -> dict:
    return {"$ref": [step, key]}


def resolve(value, outputs: list):
    if isinstance(value, dict) and set(value) == {"$ref"}:
        step, key = value["$ref"]
        return outputs[step][key]
    if isinstance(value, dict):
        return {k: resolve(v, outputs) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, outputs) for v in value]
    return value


def rest_request(service: str, step: dict, outputs: list) -> dict:
    """The REST call for a step: method, path, query, and json or raw body."""
    if "rest" in step:
        return resolve(step["rest"], outputs)
    method, path = ROUTES[service][step["tool"]]
    args = resolve(step["args"], outputs)
    for name in [p.strip("{}") for p in path.split("/") if p.startswith("{")]:
        path = path.replace("{" + name + "}", quote(str(args.pop(name)), safe=""))
    if method == "GET":
        return {"method": method, "path": path, "query": args}
    return {"method": method, "path": path, "json": args}


def call(tool: str, **args) -> dict:
    return {"tool": tool, "args": args}


def rest_only(**request) -> dict:
    return {"only": "rest", "rest": {"query": {}, **request}}


# Per-service scripts. -------------------------------------------------------


def amazon(fx: dict) -> list[dict]:
    first = fx["products"][0]
    kind = first.get("item", first.get("product_type"))
    toronto = {
        "name": "Jordan",
        "street_address": "88 Larkhaven Road",
        "city": "Toronto",
        "state": "ON",
        "country": "Canada",
        "zip_code": "M6J 3C3",
    }
    return [
        call("search_products", query=first["name"].split()[0]),
        call("search_products", query=first["name"].upper()),
        call("search_products"),
        call("search_products", query="zzzz-no-match"),
        call("search_products", query="", product_type=kind.upper(), max_price=30),
        call("search_products", page_limit=0),
        call("search_products", page_limit="x"),
        call("search_products", colour="red"),
        call("list_addresses"),
        call("list_payment_cards"),
        call("show_cart"),
        call("list_orders"),
        call("place_order"),
        call("add_to_cart", product_id=1, quantity=0),
        call("add_to_cart", product_id=999),
        call("add_to_cart", product_id="x"),
        call("add_to_cart", product_id=2),
        call("add_to_cart", product_id=2, quantity=2, gift_wrap=True),
        call("show_cart"),
        call("place_order", payment_card_id=9),
        call("place_order", address_id=9),
        call("place_order", address_id=1, ship_to=toronto),
        call("place_order", ship_to={"city": "Toronto"}),
        call("place_order", ship_to="88 Larkhaven Road"),
        call("place_order", ship_to=toronto),
        call("list_orders"),
        call("add_to_cart", product_id=1, quantity=1000),
        call("add_to_cart", product_id=3, quantity=3),
        call("place_order", address_id=1, payment_card_id=1),
        call("show_cart"),
        call("list_orders"),
    ]


def calendar(fx: dict) -> list[dict]:
    ev = fx["events"][0]
    word = ev["title"].split()[0]
    e = ev["id"]
    later = {"start": "2026-10-01T08:00:00-07:00", "end": "2026-10-01T12:00:00-07:00"}
    new = {
        "title": "Site visit",
        "start": "2026-10-03T09:00:00-07:00",
        "end": "2026-10-03T10:00:00-07:00",
        "location": "Main St",
        "description": "Measure the kitchen",
        "attendees": ["Luis"],
    }
    return [
        call("search_events", query=word),
        call("search_events", query=word.upper()),
        call("search_events", start=ev["start"], end=ev["end"]),
        call("search_events"),
        call("search_events", start=ev["start"]),
        call("search_events", start=ev["end"], end=ev["start"]),
        call("search_events", query=""),
        call("search_events", start="not-a-date", end=ev["end"]),
        call("search_events", start="2026-09-28T08:00:00", end="2026-09-28T09:00:00"),
        call("search_events", query="zzzz-no-match"),
        call("update_event", event_id=e, title="Moved job"),
        call("update_event", event_id=e, **later),
        call("update_event", event_id=e, start="2026-10-01T13:00:00-07:00"),
        call("update_event", event_id=e),
        call("update_event", event_id=e, title=None),
        call("update_event", event_id=e, colour="red"),
        call("update_event", event_id=e, attendees="Luis"),
        call("update_event", event_id=e, start="2026-10-01T08:00:00"),
        call("update_event", event_id="nope", title="x"),
        call("update_event", event_id=e, attendees=[]),
        call("update_event", event_id=e, title="Moved job"),
        call("create_event", **new),
        call("create_event", **{k: v for k, v in new.items() if k != "location"}),
        call("create_event", **new | {"end": new["start"]}),
        call("create_event", **new | {"colour": "red"}),
        call("create_event", **new | {"attendees": "Luis"}),
        call("create_event", **new | {"title": "Second visit"}),
        call("search_events", query="visit"),
    ]


def contacts(fx: dict) -> list[dict]:
    name = fx["contacts"][0]["name"]
    return [
        call("find_contacts", name=name.split()[0][:3]),
        call("find_contacts", name=name.lower()),
        call("find_contacts", name="zzzz"),
        call("find_contacts", name=""),
        call("find_contacts", name="x" * 101),
        call("find_contacts"),
    ]


def courier(fx: dict) -> list[dict]:
    full = {
        "date": "2026-09-29",
        "time": "09:30",
        "pickup_address": "1 Main St",
        "recipient": "Jo",
        "delivery_address": "2 High St",
        "note": "Gift",
    }
    a = 2  # step index of the first booking
    return [
        call("list_pickups"),
        call("save_pickup"),
        call("save_pickup", date="2026-09-28"),
        call("save_pickup", **full),
        call("save_pickup", pickup_id=ref(a, "id"), note="Fragile"),
        call("save_pickup", pickup_id=ref(a, "id"), time="14:00", date="2026-09-30"),
        call("save_pickup", pickup_id=ref(a, "id")),
        call("save_pickup", pickup_id="nope", note="x"),
        call("save_pickup", date="2026-09-28", time="25:00"),
        call("save_pickup", date="2026-13-01"),
        call("save_pickup", date="2026-09-28", courier="Speedy"),
        call("save_pickup", pickup_id=ref(a, "id"), recipient="   "),
        call("save_pickup", date=5),
        call("save_pickup", pickup_id=ref(a, "id"), note=None),
        call("save_pickup", pickup_id=ref(a, "id"), recipient="  Sam  "),
        call("list_pickups"),
    ]


def email(fx: dict) -> list[dict]:
    word = fx["messages"][0]["subject"].split()[0]
    sender = (fx.get("allowed_senders") or ["me@example.com"])[0]
    to = (fx.get("allowed_recipients") or ["them@example.com"])[0]
    mail = {"sender": sender, "recipient": to, "subject": "Re: plans", "body": "Hello"}
    return [
        call("search_messages", query=word),
        call("search_messages", query=word.lower()),
        call("search_messages", query=""),
        call("search_messages"),
        call("send_message", **mail),
        call("send_message", **mail | {"sender": "someone@else.example"}),
        call("send_message", **mail | {"recipient": "someone@else.example"}),
        call("send_message", **mail | {"sender": "not-an-email"}),
        call("send_message", **mail | {"subject": "   "}),
        call("send_message", **mail | {"cc": "x@example.com", "body": "  Again  "}),
        call("send_message", **{k: v for k, v in mail.items() if k != "body"}),
        call("send_message", **mail | {"body": 5}),
        call("search_messages", query="hello"),
    ]


def flights(fx: dict) -> list[dict]:
    current = fx["current"]
    alts = fx["alternatives"]
    free = next(
        a
        for a in alts
        if not a["change_fee_usd"] and a["departure"] > current["departure"]
    )
    paid = next(a for a in alts if a["change_fee_usd"])
    return [
        call("list_alternatives"),
        call("change_flight", option_id="nope"),
        call("change_flight", option_id=paid["id"]),
        call("change_flight", option_id=free["id"], reason="earlier meeting"),
        call("change_flight", option_id=free["id"]),
        call("list_alternatives"),
    ]


def gmail(fx: dict) -> list[dict]:
    word = (fx["drafts"] or fx["inbox"])[0]["subject"].split()[0]
    d1, d2 = 5, 9  # step indexes of the two drafts created below
    steps = [
        call("search_mail"),
        call("search_mail", query=word.lower()),
        call("search_mail", folder="drafts"),
        call("search_mail", folder="trash"),
        call("search_mail", query=""),
        call(
            "create_draft", recipient="jo@example.com", subject="Hi", body="Hello there"
        ),
        call("create_draft", body="  "),
        call("create_draft", body="x", recipient="not-an-email"),
        call("create_draft", body="x", cc="a@example.com"),
        call("create_draft", body="Only a body"),
        call("update_draft", draft_id=ref(d1, "id"), subject="New subject"),
        call("update_draft", draft_id=ref(d1, "id")),
        call(
            "update_draft",
            draft_id=ref(d1, "id"),
            recipient="a@example.com",
            clear_recipient=True,
        ),
        call("update_draft", draft_id=ref(d1, "id"), clear_recipient=True),
        call("update_draft", draft_id="nope", subject="x"),
        call("update_draft", draft_id=ref(d1, "id"), body=""),
        call("send_email", draft_id=ref(d1, "id")),
        call("update_draft", draft_id=ref(d1, "id"), recipient="jo@example.com"),
        call("send_email", draft_id=ref(d1, "id")),
        call("send_email", draft_id=ref(d1, "id")),
        call("send_email", recipient="jo@example.com", body="Direct"),
        call("send_email", recipient="jo@example.com"),
        call("send_email", draft_id=ref(d2, "id"), body="x"),
        call("send_email", recipient="jo@example.com", body="x", cc="y@example.com"),
        call("send_email", draft_id="nope"),
        call("send_email", recipient="bad", body="x"),
        call("create_draft", body="Third"),
        call("update_draft", draft_id="draft-0001", colour="red"),
    ]
    if fx["drafts"]:
        steps += [
            call("update_draft", draft_id="draft-0001", body="Happy birthday!"),
            call("send_email", draft_id="draft-0001"),
        ]
    return steps + [call("search_mail", folder="sent")]


def marketplace(fx: dict) -> list[dict]:
    t = fx["my_listings"][0]["threads"][0]["id"]
    return [
        call("read_listings"),
        call("read_listings", thread_id=t),
        call("read_listings", thread_id="nope"),
        call("reply_to_thread", thread_id=t, message="Hi, still available."),
        call("reply_to_thread", thread_id=t, message="   "),
        call("reply_to_thread", thread_id="nope", message="Hi"),
        call("reply_to_thread", thread_id=t, message="  Hi again  ", extra="x"),
        call("reply_to_thread", thread_id=t, message=5),
        call("reply_to_thread", thread_id=t),
        call("read_listings", thread_id=t),
    ]


def subscriptions(fx: dict) -> list[dict]:
    vendors = [c["vendor"] for c in fx["charges"]]
    spaced = next((v for v in vendors if " " in v), vendors[-1])
    return [
        call("list_subscriptions"),
        call("cancel_subscription", vendor=vendors[0]),
        call("cancel_subscription", vendor=vendors[0]),
        call("cancel_subscription", vendor=vendors[0].upper()),
        call("cancel_subscription", vendor="Nope"),
        call("cancel_subscription", vendor=spaced.lower()),
        call("list_subscriptions"),
    ]


SCRIPTS = {
    "amazon": amazon,
    "calendar": calendar,
    "contacts": contacts,
    "courier": courier,
    "email": email,
    "flights": flights,
    "gmail": gmail,
    "marketplace": marketplace,
    "subscriptions": subscriptions,
}


def rest_extras(service: str) -> list[dict]:
    """REST-only calls: transport errors a client may meet."""
    routes = ROUTES[service]
    read = next(path for method, path in routes.values() if method == "GET")
    extras = [
        rest_only(method="GET", path="/v1/nope"),
        rest_only(method="DELETE", path=read),
        rest_only(method="GET", path=read + "/"),
    ]
    writes = [(m, p) for m, p in routes.values() if m != "GET"]
    if writes:
        method, path = sorted(writes, key=lambda w: "{" in w[1])[0]
        path = path.replace("{", "").replace("}", "")
        extras += [
            rest_only(method=method, path=path, body="{not json"),
            rest_only(method=method, path=path, body="[1, 2]"),
        ]
    return extras


def steps(service: str, fixture: str) -> list[dict]:
    fx = json.loads((ROOT / fixture).read_text())
    return SCRIPTS[service](fx) + rest_extras(service)


def first_read(service: str) -> str:
    """A read tool, for the revoked-key checks."""
    return next(
        tool for tool, (method, _) in ROUTES[service].items() if method == "GET"
    )


# MCP requests sent as raw JSON-RPC, to pin the wire format a client sees.

HANDSHAKE_VERSIONS = (
    "2024-11-05",
    "2025-03-26",
    "2025-06-18",
    "2025-11-25",
    "2099-01-01",
)


def wire_requests(service: str) -> list[dict]:
    """(name, body, extra headers) for the raw MCP exchanges."""
    read = first_read(service)
    rpc = lambda method, params=None, id_=1: {  # noqa: E731
        "jsonrpc": "2.0",
        "id": id_,
        "method": method,
        **({"params": params} if params is not None else {}),
    }
    init = lambda v: rpc(  # noqa: E731
        "initialize",
        {
            "protocolVersion": v,
            "capabilities": {},
            "clientInfo": {"name": "parity", "version": "1"},
        },
    )
    requests = [
        {"name": f"initialize {v}", "body": init(v)} for v in HANDSHAKE_VERSIONS
    ]
    requests += [
        {
            "name": "initialized",
            "body": {"jsonrpc": "2.0", "method": "notifications/initialized"},
        },
        {"name": "ping", "body": rpc("ping")},
        {"name": "tools/list", "body": rpc("tools/list")},
        {"name": "tools/list string id", "body": rpc("tools/list", {}, "abc")},
        {"name": "prompts/list", "body": rpc("prompts/list")},
        {"name": "resources/list", "body": rpc("resources/list")},
        {"name": "resources/templates/list", "body": rpc("resources/templates/list")},
        {"name": "unknown method", "body": rpc("foo/bar")},
        {
            "name": "unknown tool",
            "body": rpc("tools/call", {"name": "nope", "arguments": {}}),
        },
        {
            "name": "read tool",
            "body": rpc("tools/call", {"name": read, "arguments": {}}),
        },
        {"name": "read tool no arguments", "body": rpc("tools/call", {"name": read})},
    ]
    # The stateless 2026-07-28 envelope: no handshake, the version in _meta.
    meta = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {"name": "parity", "version": "1"},
    }
    modern = {"mcp-protocol-version": "2026-07-28"}
    requests += [
        {
            "name": "modern tools/list",
            "body": rpc("tools/list", {"_meta": meta}),
            "headers": modern | {"mcp-method": "tools/list"},
        },
        {
            "name": "modern read tool",
            "body": rpc("tools/call", {"name": read, "arguments": {}, "_meta": meta}),
            "headers": modern | {"mcp-method": "tools/call", "mcp-name": read},
        },
        {
            "name": "modern discover",
            "body": rpc("server/discover", {"_meta": meta}),
            "headers": modern | {"mcp-method": "server/discover"},
        },
    ]
    return requests
