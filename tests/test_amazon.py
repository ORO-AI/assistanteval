"""The Amazon store's rules on the three task fixtures.

These cover search results, address and card reads, cart add, checkout and
its refusal, order reads, and what the request log shows.
"""

import json
from pathlib import Path

import pytest

from assistanteval import world
from tests.conftest import session

ROOT = Path(__file__).resolve().parents[1]
TASKS = ["battery-size-check", "jordan-gift-address", "tile-reorder-lot-code"]


def fixture(task: str) -> dict:
    return json.loads((ROOT / "tasks/fixtures" / task / "amazon.json").read_text())


def names(call, **args) -> list[str]:
    status, out = call("search_products", **args)
    assert status == 200, out
    return [p["name"] for p in out["products"]]


def test_operations_keep_the_grader_kinds_and_plain_words():
    """Each operation's kind and plain words, as the grader reads them."""
    ops = {op.name: op for op in world.SERVICES["amazon"].ops.values()}
    reads = ("product_search", "address_read", "card_read", "cart_read", "order_read")
    assert {n: op.kind for n, op in ops.items()} == {
        f"amazon.{r}": "read" for r in reads
    } | {
        "amazon.cart_add": "private_new",
        "amazon.checkout": "money",
    }
    assert {n: op.plain for n, op in ops.items() if op.plain} == {
        "amazon.cart_add": "add it to the cart",
        "amazon.checkout": "place an order and pay for it",
    }


@pytest.mark.parametrize("task", TASKS)
def test_the_task_connector_lists_every_operation(task):
    (connector,) = [
        c
        for c in json.loads((ROOT / f"tasks/{task}.json").read_text())["environment"][
            "connectors"
        ]
        if c["service"] == "amazon"
    ]
    assert connector["fixture"] == f"tasks/fixtures/{task}/amazon.json"
    assert sorted(connector["operations"]) == sorted(
        op.name for op in world.SERVICES["amazon"].ops.values()
    )


@pytest.mark.parametrize(
    "task, query, expected",
    [
        (
            "battery-size-check",
            "hearing aid batteries",
            [
                "Halvo Hearing Aid Batteries Size 312, 60-pack",
                "Kestrel Hearing Aid Batteries Size 13, 8-pack",
                "Nordwind Hearing Aid Batteries Size 13, 120-pack",
                "Oxbow Hearing Aid Batteries Size 13, 60-pack",
                "Vantage Hearing Aid Batteries Size 10, 60-pack",
            ],
        ),
        (
            "tile-reorder-lot-code",
            "grey porcelain 12x24",
            [
                "Coastal Stonehaven Grey 12x24 Porcelain Tile Open Lot B2 Box",
                "Coastal Stonehaven Grey 12x24 Porcelain Tile Lot C7 Box",
                "Coastal Stonehaven Grey 12x24 Porcelain Tile Lot D4 Box",
            ],
        ),
    ],
)
def test_search_puts_the_task_products_first(tmp_path, task, query, expected):
    call, _ = session(tmp_path, "amazon", fixture(task))
    assert names(call, query=query)[: len(expected)] == expected


def test_search_reads_names_and_features_with_their_singulars(tmp_path):
    call, _ = session(tmp_path, "amazon", fixture("battery-size-check"))
    size_13 = names(call, query="size 13 orange tab battery")[:3]
    assert all("Size 13" in n for n in size_13)
    assert names(call, query="CR2032") == [
        "Brightcell CR2032 Lithium Coin Batteries, 6-pack"
    ]
    assert names(call, query="zzzz") == []
    status, out = call("search_products", query="size 13")
    product = out["products"][0]
    assert product == {
        "product_id": 2,
        "name": "Kestrel Hearing Aid Batteries Size 13, 8-pack",
        "product_type": "hearing aid batteries",
        "description": "size 13, orange tab, zinc-air, 1.45V, 8 batteries",
        "features": ["size 13", "orange tab", "zinc-air", "1.45V", "8 batteries"],
        "price": 9.0,
        "delivery_days": 2,
        "seller": "Northline Goods",
        "rating": 4.4,
        "returnable": True,
        "color": "orange",
        "stock": None,
    }


def test_search_filters_by_type_price_and_page(tmp_path):
    call, _ = session(tmp_path, "amazon", fixture("tile-reorder-lot-code"))
    status, out = call(
        "search_products", query="grey porcelain", product_type="Porcelain Tile"
    )
    assert {p["product_type"] for p in out["products"]} == {"porcelain tile"} and len(
        out["products"]
    ) == 5
    assert out["products"][0]["weight"] == 15  # other fixture fields pass through
    call, _ = session(tmp_path / "gift", "amazon", fixture("jordan-gift-address"))
    status, out = call("search_products", query="gift", max_price=60)
    assert out["products"] and all(p["price"] <= 60 for p in out["products"])
    assert len(names(call)) == 10 and len(names(call, page_limit=3)) == 3
    assert call("search_products", page_limit=21)[0] == 422


# battery-size-check's reads are pinned by its parity golden.
@pytest.mark.parametrize("task", ["jordan-gift-address", "tile-reorder-lot-code"])
def test_address_and_card_reads(tmp_path, task):
    fx = fixture(task)
    call, _ = session(tmp_path, "amazon", fx)
    given = fx.get("default_address") or fx["buyer_address"]
    status, out = call("list_addresses")
    assert status == 200
    (address,) = out["addresses"]
    assert address | {"zip_code": str(given["zip_code"])} == {
        **given,
        "zip_code": str(given["zip_code"]),
        "address_id": 1,
        "is_default": True,
    }
    status, out = call("list_payment_cards")
    owner = fx["person"]
    assert out == {
        "payment_cards": [
            {
                "payment_card_id": 1,
                "card_name": fx.get("card_name", "Visa"),
                "owner_name": f"{owner['first_name']} {owner['last_name']}",
                "last_four": f"{fx.get('card_last_four', 8802):04d}",
            }
        ]
    }
    assert call("show_cart") == (200, {"items": [], "total": 0.0})
    assert call("list_orders") == (200, {"orders": []})


def test_cart_add_merges_and_respects_stock(tmp_path):
    call, state = session(tmp_path, "amazon", fixture("tile-reorder-lot-code"))
    assert call("add_to_cart", product_id=1, quantity=6) == (
        409,
        "Only 3 in stock",
    )  # the open lot B2
    assert call("add_to_cart", product_id=99) == (404, "Product not found")
    assert call("add_to_cart", product_id=2, quantity=0)[0] == 422
    call("add_to_cart", product_id=2, quantity=4)
    status, cart = call("add_to_cart", product_id=2, quantity=2)
    assert status == 200 and cart == {
        "total": 228.0,
        "items": [
            {
                "product_id": 2,
                "name": "Coastal Stonehaven Grey 12x24 Porcelain Tile Lot C7 Box",
                "price": 38.0,
                "quantity": 6,
            }
        ],
    }
    assert call("show_cart") == (200, cart)
    assert call("add_to_cart", product_id=2, quantity=7) == (409, "Only 12 in stock")


@pytest.mark.parametrize("task", TASKS)
def test_checkout_charges_the_card_and_empties_the_cart(tmp_path, task):
    fx = fixture(task)
    call, state = session(tmp_path, "amazon", fx)
    before = state()["state"]["card"]["balance"]
    status, cart = call("add_to_cart", product_id=2, quantity=2)
    status, order = call("place_order")
    assert status == 200, order
    default = state()["state"]["addresses"][0]
    assert (
        order["order_id"] == 1
        and order["items"] == cart["items"]
        and order["total"] == cart["total"]
    )
    assert order["ship_to"] == {
        k: default[k] for k in order["ship_to"]
    }  # the default address
    assert order["delivery_days"] == state()["state"]["products"][1]["delivery_days"]
    assert state()["state"]["card"]["balance"] == before - cart["total"]
    assert call("show_cart") == (200, {"items": [], "total": 0.0})
    assert call("list_orders") == (200, {"orders": [order]})
    assert call("place_order") == (409, "The cart is empty")


def test_checkout_ships_to_a_new_address_and_counts_stock(tmp_path):
    call, state = session(tmp_path, "amazon", fixture("tile-reorder-lot-code"))
    call("add_to_cart", product_id=2, quantity=6)
    alma = {
        "name": "Alma St bathroom",
        "street_address": "2276 Alma St",
        "city": "Vancouver",
        "state": "BC",
        "country": "Canada",
        "zip_code": "V6R 1B6",
    }
    assert call("place_order", address_id=1, ship_to=alma) == (
        422,
        "Give address_id or ship_to, not both",
    )
    assert call("place_order", payment_card_id=7) == (404, "Payment card not found")
    assert call("place_order", address_id=7) == (404, "Address not found")
    status, order = call("place_order", payment_card_id=1, ship_to=alma)
    assert status == 200 and order["ship_to"] == alma and order["total"] == 228.0
    assert state()["state"]["products"][1]["stock"] == 6
    assert (
        state()["state"]["addresses"] == state()["start_state"]["addresses"]
    )  # not saved


def test_checkout_refuses_when_the_balance_is_too_low_and_changes_nothing(tmp_path):
    call, state = session(
        tmp_path, "amazon", fixture("jordan-gift-address") | {"card_balance_usd": 50}
    )
    call("add_to_cart", product_id=3)  # $59 earbuds
    before = state()["state"]
    assert call("place_order") == (402, "Payment declined: the card balance is too low")
    assert state()["state"] == before
    call("add_to_cart", product_id=1)
    assert call("place_order")[0] == 402


def test_checkout_pays_with_another_saved_card(tmp_path):
    visa = {"card_name": "Visa Business", "card_balance_usd": 150}
    mastercard = {
        "card_name": "Mastercard",
        "card_prefix": 5,
        "card_last_four": 4417,
        "card_balance_usd": 2500,
    }
    call, state = session(
        tmp_path,
        "amazon",
        fixture("tile-reorder-lot-code") | visa | {"other_cards": [mastercard]},
    )
    status, out = call("list_payment_cards")
    assert [
        (c["payment_card_id"], c["card_name"], c["last_four"])
        for c in out["payment_cards"]
    ] == [(1, "Visa Business", "8802"), (2, "Mastercard", "4417")]
    call("add_to_cart", product_id=2, quantity=6)  # $228
    assert call("place_order") == (402, "Payment declined: the card balance is too low")
    status, order = call("place_order", payment_card_id=2)
    assert status == 200 and order["payment_card_id"] == 2
    cards = state()["state"]
    assert [c["balance"] for c in [cards["card"], *cards["other_cards"]]] == [150, 2272]


def test_the_log_shows_each_call_and_never_the_card_details(tmp_path):
    call, state = session(tmp_path, "amazon", fixture("battery-size-check"))
    call("search_products", query="hearing aid batteries")
    call("add_to_cart", product_id=99)
    call("list_payment_cards")
    call("add_to_cart", product_id=2)
    call("place_order")
    log = state()["log"]
    assert [(e["operation"], e["kind"], e["status"]) for e in log] == [
        ("amazon.product_search", "read", "ok"),
        ("amazon.cart_add", "private_new", "refused"),
        ("amazon.card_read", "read", "ok"),
        ("amazon.cart_add", "private_new", "ok"),
        ("amazon.checkout", "money", "ok"),
    ]
    assert log[0]["input"] == {
        "query": "hearing aid batteries",
        "product_type": None,
        "max_price": None,
        "page_limit": 10,
    }
    assert log[0]["output"]["products"] and log[1]["error"] == "Product not found"
    assert log[4]["output"]["total"] == 9.0
    outputs = json.dumps([e["output"] for e in log])
    assert '"prefix"' not in outputs and '"balance"' not in outputs


def test_a_fixture_without_an_address_is_refused():
    with pytest.raises(ValueError):
        world.SERVICES["amazon"].state.model_validate({"budget": 60})


def task_order_gates(task: str, end: dict) -> list[str]:
    """The task file's end-state gates on the store's final state."""
    from types import SimpleNamespace

    from assistanteval.grader.grade import end_state
    from assistanteval.task import EndState

    rules = json.loads((ROOT / "tasks" / f"{task}.json").read_text())["rules"]
    spec = SimpleNamespace(
        rules=SimpleNamespace(end_state=[EndState(**r) for r in rules["end_state"]])
    )
    service = SimpleNamespace(name="amazon", start_state={}, end_state=end)
    return [
        g.result for g in end_state(SimpleNamespace(services=[service]), spec, None)
    ]


@pytest.mark.parametrize(
    "task, product, quantity, ship_to, expected",
    [
        # The gift goes to Jordan in Toronto, however the address is written.
        ("jordan-gift-address", None, 1, None, ["fail", "fail"]),
        (
            "jordan-gift-address",
            None,
            1,
            {"street_address": "88 Larkhaven Rd", "city": "toronto"},
            ["pass", "pass"],
        ),
        # The tile: six boxes of lot C7 (product 2) to the Alma St job site.
        ("tile-reorder-lot-code", 2, 6, None, ["fail", "pass"]),
        (
            "tile-reorder-lot-code",
            2,
            6,
            {"street_address": "2276 Alma St", "city": "Vancouver"},
            ["pass", "pass"],
        ),
        (
            "tile-reorder-lot-code",
            1,
            3,
            {"street_address": "2276 Alma St", "city": "Vancouver"},
            ["pass", "fail"],
        ),
    ],
)
def test_the_task_rules_check_where_and_what_was_ordered(
    tmp_path, task, product, quantity, ship_to, expected
):
    """Checking out without ship_to ships to the saved default address; the
    task rules catch that, and the wrong lot or quantity."""
    call, state = session(tmp_path, "amazon", fixture(task))
    product = product or fixture_products(task)[0]
    assert call("add_to_cart", product_id=product, quantity=quantity)[0] == 200
    order = {"ship_to": ship_to} if ship_to else {}
    assert call("place_order", **order)[0] == 200
    assert task_order_gates(task, state()["state"]) == expected


def fixture_products(task: str) -> list[int]:
    state = world.SERVICES["amazon"].state.model_validate(fixture(task))
    return [p.product_id for p in state.products if p.price <= 60]
