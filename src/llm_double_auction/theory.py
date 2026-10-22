"""Theory of the split-the-difference (k = 1/2) double auction with uniform private values.

The buyer's value v and the seller's cost c are independent draws from U[0, H]. Each side
submits a sealed number (bid b, ask s); if b >= s they trade at p = k*b + (1 - k)*s. The
headline mechanism is k = 1/2, split the difference; k = 1/3 is the control that is not in
the textbooks.

Textbook results live on [0, 1]; everything scales linearly, so H is explicit everywhere
and the experiment uses H = 100 euros.
"""

from __future__ import annotations

import numpy as np

H = 100.0

# Linear Bayesian-Nash equilibrium (Chatterjee & Samuelson, 1983) and the efficiency it
# attains, which Myerson & Satterthwaite (1983) showed is the best any incentive-compatible,
# individually rational mechanism can do with uniform values.
EQ_EFFICIENCY = 27 / 32


def eq_buyer_bid(v, h: float = H, k: float = 0.5):
    """b(v) = v/(1+k) + h*k(1-k)/(2(1+k)); for k = 1/2, 2v/3 + h/12."""
    return np.asarray(v, dtype=float) / (1.0 + k) + h * k * (1.0 - k) / (2.0 * (1.0 + k))


def eq_seller_ask(c, h: float = H, k: float = 0.5):
    """s(c) = c/(2-k) + h(1-k)/2; for k = 1/2, 2c/3 + h/4."""
    return np.asarray(c, dtype=float) / (2.0 - k) + h * (1.0 - k) / 2.0


def eq_trade_gap(h: float = H) -> float:
    """In the linear equilibrium, trade happens iff v - c >= h/4."""
    return h / 4.0


def first_best_gains(h: float = H) -> float:
    """E[(v - c)+] for independent U[0, h] draws: h/6."""
    return h / 6.0


def eq_gains(h: float = H) -> float:
    """E[(v - c) 1{v - c >= h/4}] = 9h/64, from the triangular density of v - c."""
    return 9.0 * h / 64.0


def eq_surplus_each(h: float = H) -> float:
    """Expected payoff of each side in the linear equilibrium: the 9h/64 splits evenly."""
    return 9.0 * h / 128.0


# --- Best responses to a linear opponent (closed form) ------------------------------------


def seller_br_to_linear_buyer(c, alpha: float, beta: float, h: float = H, k: float = 0.5):
    """Best ask against a buyer who bids b(v) = alpha*v + beta, v ~ U[0, h], alpha > 0.

    Bids are uniform on [beta, beta + alpha*h]. The first-order condition of
    E[(k*b + (1-k)*a - c) 1{b >= a}] is (a - c) = (1-k)(b_max - a), so
    a* = (c + (1-k) b_max)/(2-k), clipped to the bid range; for k = 1/2, (2c + b_max)/3.
    """
    b_min, b_max = beta, beta + alpha * h
    a = (np.asarray(c, dtype=float) + (1.0 - k) * b_max) / (2.0 - k)
    return np.clip(a, b_min, b_max)


def buyer_br_to_linear_seller(v, gamma: float, delta: float, h: float = H, k: float = 0.5):
    """Best bid against a seller who asks s(c) = gamma*c + delta, c ~ U[0, h], gamma > 0.

    Asks are uniform on [delta, delta + gamma*h]; the first-order condition is
    (v - b) = k (b - s_min), so b* = (v + k s_min)/(1+k), clipped to the ask range.
    """
    s_min, s_max = delta, delta + gamma * h
    b = (np.asarray(v, dtype=float) + k * s_min) / (1.0 + k)
    return np.clip(b, s_min, s_max)


# --- Best responses to an empirical opponent ----------------------------------------------
#
# Against a finite sample of opponent numbers the expected payoff is piecewise linear, and
# its maximum always sits on one of the sample values: for the seller the payoff rises with
# the ask until the ask reaches the next bid, and for the buyer it falls with the bid as soon
# as the bid passes an ask. So enumerating the sample values (plus "do not trade") is exact.


def seller_best_response_empirical(c, bids, k: float = 0.5):
    """Best ask, and its expected payoff, for each cost in `c` against the sample `bids`.

    Payoff of ask a: mean_i 1{b_i >= a} (k b_i + (1-k) a - c). Returns (asks, payoffs); when
    no ask earns a positive payoff the seller prices itself out (ask = +inf, payoff 0).
    """
    c = np.atleast_1d(np.asarray(c, dtype=float))
    b = np.sort(np.asarray(bids, dtype=float))
    n = b.size
    cand, first = np.unique(b, return_index=True)
    suffix = np.concatenate([np.cumsum(b[::-1])[::-1], [0.0]])
    m = n - first  # bids >= candidate
    total = suffix[first]  # their sum
    payoff = (k * total + (1.0 - k) * m * cand)[None, :] / n - c[:, None] * m[None, :] / n
    j = np.argmax(payoff, axis=1)
    best = payoff[np.arange(c.size), j]
    asks = np.where(best > 0.0, cand[j], np.inf)
    return asks, np.maximum(best, 0.0)


def buyer_best_response_empirical(v, asks, k: float = 0.5):
    """Best bid, and its expected payoff, for each value in `v` against the sample `asks`.

    Payoff of bid b: mean_j 1{b >= s_j} (v - k b - (1-k) s_j). Returns (bids, payoffs); when
    no bid earns a positive payoff the buyer stays out (bid = -inf, payoff 0).
    """
    v = np.atleast_1d(np.asarray(v, dtype=float))
    s = np.sort(np.asarray(asks, dtype=float))
    n = s.size
    cand = np.unique(s)
    m = np.searchsorted(s, cand, side="right")  # asks <= candidate
    prefix = np.concatenate([[0.0], np.cumsum(s)])
    total = prefix[m]
    payoff = (v[:, None] * m[None, :] - (k * m * cand + (1.0 - k) * total)[None, :]) / n
    j = np.argmax(payoff, axis=1)
    best = payoff[np.arange(v.size), j]
    bids = np.where(best > 0.0, cand[j], -np.inf)
    return bids, np.maximum(best, 0.0)


def seller_payoff_against(asks, c, bids, values, k: float = 0.5, per: str = "own"):
    """Mean payoffs when sellers with costs `c` post `asks` against a buyer sample.

    Returns (seller payoff, buyer payoff). With per="own", each is averaged over the buyer
    sample, one entry per cost; with per="opponent", averaged over the costs, one entry per
    buyer. `values` are the buyers' values behind `bids`.
    """
    a = np.asarray(asks, dtype=float)[:, None]
    c = np.asarray(c, dtype=float)[:, None]
    b = np.asarray(bids, dtype=float)[None, :]
    v = np.asarray(values, dtype=float)[None, :]
    trade = b >= a
    price = np.where(trade, k * b + (1.0 - k) * np.where(np.isfinite(a), a, 0.0), 0.0)
    axis = 1 if per == "own" else 0
    seller = np.where(trade, price - c, 0.0).mean(axis=axis)
    buyer = np.where(trade, v - price, 0.0).mean(axis=axis)
    return seller, buyer


def buyer_payoff_against(bids, v, asks, costs, k: float = 0.5, per: str = "own"):
    """Mean payoffs when buyers with values `v` post `bids` against a seller sample.

    Returns (buyer payoff, seller payoff), averaged over the seller sample (per="own", one
    entry per value) or over the values (per="opponent", one entry per seller).
    """
    b = np.asarray(bids, dtype=float)[:, None]
    v = np.asarray(v, dtype=float)[:, None]
    s = np.asarray(asks, dtype=float)[None, :]
    c = np.asarray(costs, dtype=float)[None, :]
    trade = b >= s
    price = np.where(trade, k * np.where(np.isfinite(b), b, 0.0) + (1.0 - k) * s, 0.0)
    axis = 1 if per == "own" else 0
    buyer = np.where(trade, v - price, 0.0).mean(axis=axis)
    seller = np.where(trade, price - c, 0.0).mean(axis=axis)
    return buyer, seller


def pair_outcomes(v, b, c, s, k: float = 0.5) -> dict[str, float]:
    """Every buyer call against every seller call: the sealed bids make the pairing valid.

    Returns mean gains from trade, the first-best benchmark on the same draws, the two kinds
    of loss (efficient trades missed, inefficient trades made) and each side's payoff.
    """
    v = np.asarray(v, dtype=float)[:, None]
    b = np.asarray(b, dtype=float)[:, None]
    c = np.asarray(c, dtype=float)[None, :]
    s = np.asarray(s, dtype=float)[None, :]
    trade = b >= s
    gap = v - c
    price = k * b + (1.0 - k) * s
    first_best = np.maximum(gap, 0.0).mean()
    return {
        "gains": np.where(trade, gap, 0.0).mean(),
        "first_best": first_best,
        "missed": np.where(~trade, np.maximum(gap, 0.0), 0.0).mean(),
        "bad": np.where(trade, np.maximum(-gap, 0.0), 0.0).mean(),
        "buyer": np.where(trade, v - price, 0.0).mean(),
        "seller": np.where(trade, price - c, 0.0).mean(),
        "trade_rate": trade.mean(),
    }


def eq_efficiency(k: float = 0.5, h: float = H) -> float:
    """Share of the first-best gains the linear equilibrium of the k-double auction captures.

    27/32 at k = 1/2 (the Myerson-Satterthwaite optimum); lower for any other k.
    """
    from scipy import integrate

    def gains(c):
        # trade iff b(v) >= s(c)  <=>  v >= (s(c) - h k(1-k)/(2(1+k))) (1+k)
        v_min = (eq_seller_ask(c, h, k) - h * k * (1 - k) / (2 * (1 + k))) * (1 + k)
        v_min = min(max(v_min, 0.0), h)
        return ((h - c) ** 2 - (v_min - c) ** 2) / 2.0  # integral of (v - c) over [v_min, h]

    val, _ = integrate.quad(gains, 0.0, h, limit=200)
    return (val / h**2) / first_best_gains(h)
