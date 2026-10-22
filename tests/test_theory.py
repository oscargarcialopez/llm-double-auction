"""The closed forms in theory.py, checked against numerical integration and optimisation."""

import numpy as np
import pytest
from scipy import integrate, optimize

from llm_double_auction import theory as th

H = th.H


def midpoints(n: int, h: float = H) -> np.ndarray:
    return (np.arange(n) + 0.5) * h / n


def test_first_best_gains_is_h_over_6():
    val, _ = integrate.dblquad(lambda v, c: max(v - c, 0.0), 0, H, 0, H)
    assert val / H**2 == pytest.approx(th.first_best_gains(), rel=1e-6)
    assert th.first_best_gains() == pytest.approx(100 / 6)


def test_equilibrium_gains_and_the_27_over_32_ratio():
    gap = th.eq_trade_gap()
    # trade region: v >= c + h/4
    val, _ = integrate.dblquad(lambda v, c: v - c, 0, H - gap, lambda c: c + gap, H)
    assert val / H**2 == pytest.approx(th.eq_gains(), rel=1e-6)
    assert th.eq_gains() / th.first_best_gains() == pytest.approx(27 / 32)
    assert abs(th.EQ_EFFICIENCY - 0.84375) < 1e-12


def test_equilibrium_surplus_splits_evenly():
    gap = th.eq_trade_gap()

    def price(v, c):
        return (th.eq_buyer_bid(v) + th.eq_seller_ask(c)) / 2

    buyer, _ = integrate.dblquad(lambda v, c: v - price(v, c), 0, H - gap, lambda c: c + gap, H)
    seller, _ = integrate.dblquad(lambda v, c: price(v, c) - c, 0, H - gap, lambda c: c + gap, H)
    assert buyer / H**2 == pytest.approx(th.eq_surplus_each(), rel=1e-6)
    assert seller / H**2 == pytest.approx(th.eq_surplus_each(), rel=1e-6)


def test_trade_condition_of_the_linear_equilibrium():
    v = np.array([30.0, 50.0, 80.0, 99.0])
    c = v - th.eq_trade_gap()
    assert np.allclose(th.eq_buyer_bid(v), th.eq_seller_ask(c))  # exactly at the margin


def test_equilibrium_strategies_are_mutual_best_responses():
    c = np.linspace(0, 75, 31)  # above 75 no equilibrium bid reaches the ask: no trade either way
    br = th.seller_br_to_linear_buyer(c, alpha=2 / 3, beta=H / 12)
    assert np.allclose(br, th.eq_seller_ask(c))
    v = np.linspace(25, 100, 31)
    br = th.buyer_br_to_linear_seller(v, gamma=2 / 3, delta=H / 4)
    assert np.allclose(br, th.eq_buyer_bid(v))


def seller_payoff_vs_linear_buyer(a, c, alpha, beta, k=0.5):
    def integrand(v):
        b = alpha * v + beta
        return (k * b + (1 - k) * a - c) if b >= a else 0.0

    val, _ = integrate.quad(integrand, 0, H, points=[(a - beta) / alpha], limit=200)
    return val / H


def buyer_payoff_vs_linear_seller(b, v, gamma, delta, k=0.5):
    def integrand(c):
        s = gamma * c + delta
        return (v - k * b - (1 - k) * s) if b >= s else 0.0

    val, _ = integrate.quad(integrand, 0, H, points=[(b - delta) / gamma], limit=200)
    return val / H


@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (0.9, 3.0), (0.5, 20.0)])
@pytest.mark.parametrize("c", [0.0, 10.0, 35.0])
def test_seller_best_response_formula_matches_numeric_optimum(alpha, beta, c):
    b_max = alpha * H + beta
    res = optimize.minimize_scalar(
        lambda a: -seller_payoff_vs_linear_buyer(a, c, alpha, beta),
        bounds=(beta, b_max),
        method="bounded",
        options={"xatol": 1e-6},
    )
    assert th.seller_br_to_linear_buyer(c, alpha, beta) == pytest.approx(res.x, abs=1e-3)


@pytest.mark.parametrize("gamma,delta", [(1.0, 0.0), (0.8, 10.0), (0.5, 30.0)])
@pytest.mark.parametrize("v", [60.0, 85.0, 100.0])
def test_buyer_best_response_formula_matches_numeric_optimum(gamma, delta, v):
    s_max = gamma * H + delta
    res = optimize.minimize_scalar(
        lambda b: -buyer_payoff_vs_linear_seller(b, v, gamma, delta),
        bounds=(delta, s_max),
        method="bounded",
        options={"xatol": 1e-6},
    )
    assert th.buyer_br_to_linear_seller(v, gamma, delta) == pytest.approx(res.x, abs=1e-3)


def test_truthful_opponent_best_responses():
    # Against a truthful buyer (b = v) the best ask is (2c + 100)/3: shade up, never at cost.
    assert th.seller_br_to_linear_buyer(0.0, 1.0, 0.0) == pytest.approx(100 / 3)
    assert th.buyer_br_to_linear_seller(100.0, 1.0, 0.0) == pytest.approx(200 / 3)


def test_empirical_best_responses_recover_the_equilibrium():
    v = midpoints(20_000)
    c = np.linspace(0, 70, 15)
    asks, _ = th.seller_best_response_empirical(c, th.eq_buyer_bid(v))
    assert np.allclose(asks, th.eq_seller_ask(c), atol=0.05)
    costs = midpoints(20_000)
    v_grid = np.linspace(30, 100, 15)
    bids, _ = th.buyer_best_response_empirical(v_grid, th.eq_seller_ask(costs))
    assert np.allclose(bids, th.eq_buyer_bid(v_grid), atol=0.05)


def test_empirical_best_response_is_exact_on_a_small_sample():
    bids = np.array([10.0, 20.0, 20.0, 50.0])
    # brute force over a fine grid of asks
    grid = np.linspace(0, 60, 60_001)
    for c in (0.0, 5.0, 15.0, 40.0, 60.0):
        payoffs = [np.mean(np.where(bids >= a, (bids + a) / 2 - c, 0.0)) for a in grid]
        asks, best = th.seller_best_response_empirical(c, bids)
        assert best[0] == pytest.approx(max(max(payoffs), 0.0), abs=1e-6)
    asks_ = np.array([10.0, 20.0, 20.0, 50.0])
    for val in (5.0, 25.0, 45.0, 90.0):
        payoffs = [np.mean(np.where(b >= asks_, val - (b + asks_) / 2, 0.0)) for b in grid]
        _, best = th.buyer_best_response_empirical(val, asks_)
        assert best[0] == pytest.approx(max(max(payoffs), 0.0), abs=1e-6)


def test_no_profitable_trade_means_staying_out():
    asks, best = th.seller_best_response_empirical(np.array([95.0]), np.array([10.0, 50.0]))
    assert np.isinf(asks[0]) and best[0] == 0.0
    bids, best = th.buyer_best_response_empirical(np.array([5.0]), np.array([10.0, 50.0]))
    assert np.isneginf(bids[0]) and best[0] == 0.0


def test_pair_outcomes_truthful_is_fully_efficient():
    x = midpoints(400)
    out = th.pair_outcomes(x, x, x, x)
    assert out["gains"] == pytest.approx(out["first_best"])
    assert out["missed"] == pytest.approx(0.0) and out["bad"] == pytest.approx(0.0)
    assert out["first_best"] == pytest.approx(th.first_best_gains(), rel=1e-4)


def test_pair_outcomes_equilibrium_reaches_27_over_32():
    v = midpoints(1_000)
    out = th.pair_outcomes(v, th.eq_buyer_bid(v), v, th.eq_seller_ask(v))
    assert out["gains"] / out["first_best"] == pytest.approx(27 / 32, abs=2e-3)
    assert out["buyer"] == pytest.approx(th.eq_surplus_each(), rel=5e-3)
    assert out["seller"] == pytest.approx(th.eq_surplus_each(), rel=5e-3)


def test_pair_outcomes_accounting_identities():
    rng = np.random.default_rng(0)
    v, c = rng.uniform(0, H, 300), rng.uniform(0, H, 300)
    b, s = v + rng.normal(0, 15, 300), c + rng.normal(0, 15, 300)  # noisy, sometimes irrational
    out = th.pair_outcomes(v, b, c, s)
    assert out["first_best"] - out["missed"] - out["bad"] == pytest.approx(out["gains"])
    assert out["buyer"] + out["seller"] == pytest.approx(out["gains"])
    assert out["bad"] > 0


def test_payoff_against_samples_matches_pair_outcomes():
    rng = np.random.default_rng(1)
    v, c = rng.uniform(0, H, 50), rng.uniform(0, H, 60)
    b, s = th.eq_buyer_bid(v), th.eq_seller_ask(c)
    seller, buyer = th.seller_payoff_against(s, c, b, v)
    out = th.pair_outcomes(v, b, c, s)
    assert seller.mean() == pytest.approx(out["seller"])
    assert buyer.mean() == pytest.approx(out["buyer"])
    buyer2, seller2 = th.buyer_payoff_against(b, v, s, c)
    assert buyer2.mean() == pytest.approx(out["buyer"])
    assert seller2.mean() == pytest.approx(out["seller"])


# --- k = 1/3: the price rule whose equilibrium is not the textbook one ---------------------

K13 = 1 / 3


def test_k13_equilibrium_formulas():
    v = np.linspace(0, 100, 11)
    assert np.allclose(th.eq_buyer_bid(v, k=K13), 3 * v / 4 + H / 12)
    assert np.allclose(th.eq_seller_ask(v, k=K13), 3 * v / 5 + H / 3)


def test_k13_equilibrium_strategies_are_mutual_best_responses():
    b_max = 3 * H / 4 + H / 12
    c = np.linspace(0, b_max, 21)
    br = th.seller_br_to_linear_buyer(c, alpha=3 / 4, beta=H / 12, k=K13)
    assert np.allclose(br, th.eq_seller_ask(c, k=K13))
    v = np.linspace(H / 3, H, 21)
    br = th.buyer_br_to_linear_seller(v, gamma=3 / 5, delta=H / 3, k=K13)
    assert np.allclose(br, th.eq_buyer_bid(v, k=K13))


@pytest.mark.parametrize("c", [5.0, 30.0])
def test_k13_seller_best_response_matches_numeric_optimum(c):
    alpha, beta = 0.8, 5.0
    res = optimize.minimize_scalar(
        lambda a: -seller_payoff_vs_linear_buyer(a, c, alpha, beta, k=K13),
        bounds=(beta, alpha * H + beta),
        method="bounded",
        options={"xatol": 1e-6},
    )
    assert th.seller_br_to_linear_buyer(c, alpha, beta, k=K13) == pytest.approx(res.x, abs=1e-3)


@pytest.mark.parametrize("v", [60.0, 95.0])
def test_k13_buyer_best_response_matches_numeric_optimum(v):
    gamma, delta = 0.7, 12.0
    res = optimize.minimize_scalar(
        lambda b: -buyer_payoff_vs_linear_seller(b, v, gamma, delta, k=K13),
        bounds=(delta, gamma * H + delta),
        method="bounded",
        options={"xatol": 1e-6},
    )
    assert th.buyer_br_to_linear_seller(v, gamma, delta, k=K13) == pytest.approx(res.x, abs=1e-3)


def test_k13_empirical_best_responses_recover_the_equilibrium():
    v = midpoints(20_000)
    c = np.linspace(0, 70, 15)
    asks, _ = th.seller_best_response_empirical(c, th.eq_buyer_bid(v, k=K13), k=K13)
    assert np.allclose(asks, th.eq_seller_ask(c, k=K13), atol=0.05)
    v_grid = np.linspace(40, 100, 13)
    bids, _ = th.buyer_best_response_empirical(v_grid, th.eq_seller_ask(v, k=K13), k=K13)
    assert np.allclose(bids, th.eq_buyer_bid(v_grid, k=K13), atol=0.05)


def test_equilibrium_efficiency_by_k():
    assert th.eq_efficiency(0.5) == pytest.approx(27 / 32, rel=1e-6)
    eff13 = th.eq_efficiency(K13)
    v = midpoints(1_000)
    out = th.pair_outcomes(v, th.eq_buyer_bid(v, k=K13), v, th.eq_seller_ask(v, k=K13), k=K13)
    assert out["gains"] / out["first_best"] == pytest.approx(eff13, abs=2e-3)
    assert eff13 < 27 / 32  # the Myerson-Satterthwaite optimum is only reached at k = 1/2
