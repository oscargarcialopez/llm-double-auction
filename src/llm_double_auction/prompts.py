"""The two prompt framings. Both state the rules, the common prior and the objective; neither
states the solution.

`informed-v1` uses the vocabulary of the game. `shopper-v1` is the same game told the way a
shopping or selling assistant would meet it, without game-theory words. Any change to a
template must bump its version: the cache keys on it, and stored prompts carry their hash.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

INFORMED_BUYER = """\
You are an AI agent bidding on behalf of a buyer in a sealed-bid double auction.

The rules, which both sides know:
- One item is for sale. The buyer (your client) values it at V euros. The seller's cost of \
selling it is C euros.
- Each side knows only its own number. V and C are independent, and each is uniformly \
distributed between 0 and 100 euros; this is common knowledge.
- You submit a bid b for the buyer. At the same time, an AI agent acting for the seller \
submits an ask s. Neither side sees the other's number before submitting.
- If b >= s, the trade happens at the price p = (b + s) / 2: the buyer pays p and gets the \
item, so your client's payoff is V - p. If b < s, there is no trade and your client's \
payoff is 0.
- The seller's agent knows these same rules and wants to maximize its own client's expected \
payoff.
- This is a single, one-shot auction: no negotiation and no second round.

Your client's value: V = {value:.2f} euros.

Your objective: maximize your client's expected payoff. Your client is risk-neutral.

Answer with your bid in euros."""

INFORMED_SELLER = """\
You are an AI agent bidding on behalf of a seller in a sealed-bid double auction.

The rules, which both sides know:
- One item is for sale. The seller (your client) has a cost of C euros for selling it. The \
buyer values it at V euros.
- Each side knows only its own number. V and C are independent, and each is uniformly \
distributed between 0 and 100 euros; this is common knowledge.
- You submit an ask s for the seller. At the same time, an AI agent acting for the buyer \
submits a bid b. Neither side sees the other's number before submitting.
- If b >= s, the trade happens at the price p = (b + s) / 2: the seller receives p and gives \
up the item, so your client's payoff is p - C. If b < s, there is no trade and your \
client's payoff is 0.
- The buyer's agent knows these same rules and wants to maximize its own client's expected \
payoff.
- This is a single, one-shot auction: no negotiation and no second round.

Your client's cost: C = {value:.2f} euros.

Your objective: maximize your client's expected payoff. Your client is risk-neutral.

Answer with your ask in euros."""

SHOPPER_BUYER = """\
You are a shopping assistant that buys things for your user on a second-hand marketplace.

Your user wants a used camera lens. It is worth {value:.2f} € to them: that is the most \
they would be willing to pay for it.

This marketplace closes deals with a single round of sealed offers:
- You submit one price offer for your user. At the same moment, the seller's assistant \
submits an asking price. Neither of you sees the other's number first.
- If your offer is at least the asking price, the deal goes through at the midpoint of the \
two numbers. If it is lower, there is no deal.
- There is no second round and no counter-offer.

What you know about the seller: the least they would accept is private. Judging by similar \
listings, it is equally likely to be anywhere between 0 € and 100 €. The seller's assistant \
is in the same position: all it knows is that your user's maximum is equally likely to be \
anywhere between 0 € and 100 €.

Your goal is to get your user the best deal on average: what the lens is worth to them \
minus what they pay, counting no deal as zero.

Reply with your offer in euros."""

SHOPPER_SELLER = """\
You are a sales assistant that sells things for your user on a second-hand marketplace.

Your user is selling a used camera lens. Keeping it is worth {value:.2f} € to them: below \
that price they would rather not sell.

This marketplace closes deals with a single round of sealed offers:
- You submit one asking price for your user. At the same moment, the buyer's assistant \
submits a price offer. Neither of you sees the other's number first.
- If the buyer's offer is at least your asking price, the deal goes through at the midpoint \
of the two numbers. If it is lower, there is no deal.
- There is no second round and no counter-offer.

What you know about the buyer: the most they would pay is private. Judging by similar \
listings, it is equally likely to be anywhere between 0 € and 100 €. The buyer's assistant \
is in the same position: all it knows is that your user's minimum is equally likely to be \
anywhere between 0 € and 100 €.

Your goal is to get your user the best deal on average: the price they receive minus what \
the lens is worth to them, counting no deal as zero.

Reply with your asking price in euros."""


# The same informed framing with a different price rule, p = (b + 2s)/3 (k = 1/3). Its linear
# equilibrium is b(v) = 3v/4 + 100/12, s(c) = 3c/5 + 100/3: not the famous k = 1/2 formula,
# so a model that only recalls the textbook case gives itself away.
INFORMED_BUYER_K13 = INFORMED_BUYER.replace(
    "the trade happens at the price p = (b + s) / 2:",
    "the trade happens at the price p = (b + 2s) / 3:",
)
INFORMED_SELLER_K13 = INFORMED_SELLER.replace(
    "the trade happens at the price p = (b + s) / 2:",
    "the trade happens at the price p = (b + 2s) / 3:",
)


@dataclass(frozen=True)
class Framing:
    buyer: str
    seller: str
    buyer_field: str
    seller_field: str


FRAMINGS: dict[str, Framing] = {
    "informed-v1": Framing(INFORMED_BUYER, INFORMED_SELLER, "bid_eur", "ask_eur"),
    "shopper-v1": Framing(SHOPPER_BUYER, SHOPPER_SELLER, "offer_eur", "asking_price_eur"),
    "informed-k13-v1": Framing(INFORMED_BUYER_K13, INFORMED_SELLER_K13, "bid_eur", "ask_eur"),
}


def render(prompt_version: str, role: str, value: float) -> tuple[str, str]:
    """The prompt text and the name of the single numeric field the reply must carry."""
    framing = FRAMINGS[prompt_version]
    if role == "buyer":
        return framing.buyer.format(value=value), framing.buyer_field
    if role == "seller":
        return framing.seller.format(value=value), framing.seller_field
    raise ValueError(f"unknown role: {role}")


def response_schema(field: str) -> dict:
    return {
        "type": "object",
        "properties": {field: {"type": "number"}},
        "required": [field],
    }


def template_hash(prompt_version: str, role: str) -> str:
    framing = FRAMINGS[prompt_version]
    text = framing.buyer if role == "buyer" else framing.seller
    return hashlib.sha256(text.encode()).hexdigest()[:12]
