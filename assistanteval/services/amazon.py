"""Amazon: a small store with search, a cart, checkout on one card, and orders.

It loads the task fixtures as written: products as ``{name, item, price_usd,
features, ships_in_days, ...}`` or ``{name, product_type, price, description,
delivery_days, stock, ...}``; a ``default_address`` or ``buyer_address``; and
``card_name``, ``card_prefix``, ``card_last_four`` and ``card_balance_usd``
(a Visa ending 8802 with $1,000 when left out), and ``other_cards``, more
cards with the same keys. Other product fields, such as sizes and weights,
pass through to the assistant.
"""

import re

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    create_model,
    model_validator,
)

from assistanteval.services import NoParams
from assistanteval.world import MONEY, PRIVATE_NEW, READ, Refused, service


class Product(BaseModel):
    model_config = ConfigDict(extra="allow")

    product_id: int
    name: str
    product_type: str = Field(validation_alias=AliasChoices("product_type", "item"))
    description: str = ""
    features: list[str] = []
    price: float = Field(ge=0, validation_alias=AliasChoices("price", "price_usd"))
    delivery_days: int = Field(
        ge=0, validation_alias=AliasChoices("delivery_days", "ships_in_days")
    )
    seller: str
    rating: float | None = None
    returnable: bool | None = None
    color: str | None = None
    stock: int | None = None


class ShipTo(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)

    name: str = ""
    street_address: str = Field(min_length=1)
    city: str = Field(min_length=1)
    state: str = ""
    country: str = ""
    zip_code: str = ""


class Address(ShipTo):
    address_id: int
    is_default: bool = False


class CardView(BaseModel):
    payment_card_id: int = 1
    card_name: str = "Visa"
    owner_name: str
    last_four: str = "8802"


class Card(CardView):
    prefix: int = 4
    balance: float = Field(1000, ge=0)


class CartItem(BaseModel):
    product_id: int
    name: str
    price: float
    quantity: int


class Order(BaseModel):
    order_id: int
    items: list[CartItem]
    total: float
    payment_card_id: int
    ship_to: ShipTo
    delivery_days: int
    status: str = "placed"


class Amazon(BaseModel):
    addresses: list[Address] = Field(min_length=1)
    card: Card
    # Left out when empty: the grader re-dumps each fixture with this model
    # and checks the run's recorded start state holds every key, so a start
    # state recorded without the key still passes.
    other_cards: list[Card] = Field([], exclude_if=lambda cards: not cards)
    products: list[Product] = []
    cart: list[CartItem] = []
    orders: list[Order] = []

    @model_validator(mode="before")
    @classmethod
    def from_task_fixture(cls, data):
        if not isinstance(data, dict) or "card" in data:
            return data
        person = data.get(
            "person", {"first_name": "AssistantEval", "last_name": "Shopper"}
        )
        keys = {
            "card_name": "card_name",
            "card_prefix": "prefix",
            "card_balance_usd": "balance",
        }
        cards = [
            {
                "payment_card_id": n,
                "owner_name": f"{person['first_name']} {person['last_name']}",
            }
            | {keys[k]: v for k, v in c.items() if k in keys}
            | (
                {"last_four": f"{c['card_last_four']:04d}"}
                if "card_last_four" in c
                else {}
            )
            for n, c in enumerate([data, *data.get("other_cards", [])], 1)
        ]
        address = data.get("default_address", data.get("buyer_address"))
        products = [
            {"product_id": n, **p} for n, p in enumerate(data.get("products", []), 1)
        ]
        for p in products:
            p.setdefault("description", ", ".join(p.get("features", [])))
        return {
            "addresses": [{"address_id": 1, "is_default": True, **address}]
            if address
            else [],
            "card": cards[0],
            "other_cards": cards[1:],
            "products": products,
        }


Products = create_model("Products", products=list[Product])
Addresses = create_model("Addresses", addresses=list[Address])
PaymentCards = create_model("PaymentCards", payment_cards=list[CardView])
Cart = create_model("Cart", items=list[CartItem], total=float)
Orders = create_model("Orders", orders=list[Order])


class SearchParams(BaseModel):
    query: str = Field(
        "", description="Words to look for in product names, types and features"
    )
    product_type: str | None = Field(
        None, description="Only products of this type, e.g. 'porcelain tile'"
    )
    max_price: float | None = Field(
        None, description="Only products at or under this price in USD"
    )
    page_limit: int = Field(10, ge=1, le=20, description="How many products to return")


class AddParams(BaseModel):
    product_id: int = Field(description="The product to add")
    quantity: int = Field(1, ge=1, description="How many to add")


class CheckoutParams(BaseModel):
    payment_card_id: int | None = Field(
        None, description="The card to pay with; omit for the saved card"
    )
    address_id: int | None = Field(
        None, description="A saved address; omit this and ship_to for the default"
    )
    ship_to: ShipTo | None = Field(
        None, description="Ship to this address instead of a saved one"
    )


def _words(text: str) -> list[set[str]]:
    """Each word with its singulars: "batteries" also reads as "battery"."""
    ends = (("ies", "y"), ("es", ""), ("s", ""))
    return [
        {w}
        | {w[: -len(e)] + r for e, r in ends if w.endswith(e) and len(w) > len(e) + 1}
        for w in re.findall(r"[a-z0-9]+", text.casefold())
    ]


def _cart(state: Amazon):
    return Cart(
        items=state.cart, total=round(sum(i.price * i.quantity for i in state.cart), 2)
    )


amazon = service(
    "amazon", "Amazon", Amazon, "A synthetic store: search, cart, checkout and orders."
)


@amazon.op(
    "product_search",
    "GET /v1/products",
    tool="search_products",
    params=SearchParams,
    output=Products,
    kind=READ,
    description="Search the store's products; the best matches come first.",
)
def search_products(state: Amazon, p: SearchParams):
    query, found = _words(p.query), []
    for n, x in enumerate(state.products):
        words = set().union(
            *_words(" ".join([x.name, x.product_type, x.description, *x.features]))
        )
        score = sum(bool(forms & words) for forms in query)
        if (
            (score or not query)
            and (p.max_price is None or x.price <= p.max_price)
            and (
                p.product_type is None
                or x.product_type.casefold() == p.product_type.casefold()
            )
        ):
            found.append((-score, n, x))
    return Products(
        products=[x for *_, x in sorted(found, key=lambda f: f[:2])][: p.page_limit]
    )


@amazon.op(
    "address_read",
    "GET /v1/addresses",
    tool="list_addresses",
    params=NoParams,
    output=Addresses,
    kind=READ,
    description="List the user's saved shipping addresses, the default first.",
)
def list_addresses(state: Amazon, p: NoParams):
    return Addresses(addresses=state.addresses)


@amazon.op(
    "card_read",
    "GET /v1/payment-cards",
    tool="list_payment_cards",
    params=NoParams,
    output=PaymentCards,
    kind=READ,
    description="List the user's saved payment cards.",
)
def list_payment_cards(state: Amazon, p: NoParams):
    return PaymentCards(
        payment_cards=[
            CardView.model_validate(c.model_dump())
            for c in [state.card, *state.other_cards]
        ]
    )


@amazon.op(
    "cart_read",
    "GET /v1/cart",
    tool="show_cart",
    params=NoParams,
    output=Cart,
    kind=READ,
    description="Show what is in the cart and its total.",
)
def show_cart(state: Amazon, p: NoParams):
    return _cart(state)


@amazon.op(
    "order_read",
    "GET /v1/orders",
    tool="list_orders",
    params=NoParams,
    output=Orders,
    kind=READ,
    description="List the user's orders.",
)
def list_orders(state: Amazon, p: NoParams):
    return Orders(orders=state.orders)


@amazon.op(
    "cart_add",
    "POST /v1/cart/{product_id}",
    tool="add_to_cart",
    params=AddParams,
    output=Cart,
    kind=PRIVATE_NEW,
    plain="add it to the cart",
    description="Add a product to the cart.",
)
def add_to_cart(state: Amazon, p: AddParams):
    product = next((x for x in state.products if x.product_id == p.product_id), None)
    if product is None:
        raise Refused("Product not found", 404)
    line = next((i for i in state.cart if i.product_id == p.product_id), None)
    if (
        product.stock is not None
        and p.quantity + (line.quantity if line else 0) > product.stock
    ):
        raise Refused(f"Only {product.stock} in stock", 409)
    if line:
        line.quantity += p.quantity
    else:
        state.cart.append(
            CartItem(
                quantity=p.quantity,
                **product.model_dump(include={"product_id", "name", "price"}),
            )
        )
    return _cart(state)


@amazon.op(
    "checkout",
    "POST /v1/orders",
    tool="place_order",
    params=CheckoutParams,
    output=Order,
    kind=MONEY,
    plain="place an order and pay for it",
    description="Buy everything in the cart: charge the card and ship to one address.",
)
def place_order(state: Amazon, p: CheckoutParams):
    if p.address_id is not None and p.ship_to is not None:
        raise Refused("Give address_id or ship_to, not both", 422)
    card = next(
        (
            c
            for c in [state.card, *state.other_cards]
            if p.payment_card_id in (None, c.payment_card_id)
        ),
        None,
    )
    if card is None:
        raise Refused("Payment card not found", 404)
    wanted = [
        a
        for a in state.addresses
        if a.address_id == p.address_id or (p.address_id is None and a.is_default)
    ]
    if p.ship_to is None and not wanted:
        raise Refused("Address not found", 404)
    if not state.cart:
        raise Refused("The cart is empty", 409)
    total = _cart(state).total
    if total > card.balance:
        raise Refused("Payment declined: the card balance is too low", 402)
    products = {x.product_id: x for x in state.products}
    for item in state.cart:
        if products[item.product_id].stock is not None:
            products[item.product_id].stock -= item.quantity
    card.balance = round(card.balance - total, 2)
    order = Order(
        order_id=len(state.orders) + 1,
        items=state.cart,
        total=total,
        payment_card_id=card.payment_card_id,
        ship_to=p.ship_to or ShipTo.model_validate(wanted[0].model_dump()),
        delivery_days=max(products[i.product_id].delivery_days for i in state.cart),
    )
    state.orders.append(order)
    state.cart = []
    return order
