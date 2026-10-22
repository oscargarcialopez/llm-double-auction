"""The analysis on synthetic agents whose answers we know."""

import numpy as np
import pytest

from llm_double_auction import analysis as an
from llm_double_auction import theory as th
from llm_double_auction.design import STRATA, draws


def synthetic(role: str, strategy, replicates: int = 10) -> an.Side:
    grid = draws(role)[:replicates]
    stratum = np.tile(np.arange(STRATA), replicates)
    replicate = np.repeat(np.arange(replicates), STRATA)
    value = grid.ravel()
    return an.Side(value, strategy(value), stratum, replicate, [""] * value.size)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_strat_resample_keeps_strata_sizes(rng):
    stratum = np.repeat(np.arange(5), 4)
    idx = an.strat_resample(rng, stratum)
    assert np.array_equal(np.bincount(stratum[idx]), np.bincount(stratum))


def test_wilson_interval_contains_the_proportion():
    lo, hi = an.wilson(3, 100)
    assert lo < 0.03 < hi and lo > 0


def test_equilibrium_agents_score_the_frontier(rng):
    buyer = synthetic("buyer", th.eq_buyer_bid)
    seller = synthetic("seller", th.eq_seller_ask)
    eff = an.efficiency(buyer, seller, 0.5, rng, n_boot=200)
    assert eff["efficiency"] == pytest.approx(eff["eq_efficiency_same_draws"])
    assert eff["efficiency"] == pytest.approx(27 / 32, abs=0.03)
    assert eff["bad_share"] == 0.0
    side = an.describe_side("buyer", buyer, 0.5, rng, n_boot=100)
    assert side["slope"] == pytest.approx(2 / 3)
    assert side["intercept"] == pytest.approx(100 / 12)
    assert side["share_at_eq_to_the_cent"] == 1.0
    assert side["ir_violations"] > 0  # the equilibrium bids above value when v < 25...
    assert side["ir_binding"] == 0  # ...but never where it could trade
    seller_side = an.describe_side("seller", seller, 0.5, rng, n_boot=100)
    assert seller_side["ir_violations"] > 0 and seller_side["ir_binding"] == 0


def test_truthful_agents_are_efficient_but_exploitable(rng):
    buyer = synthetic("buyer", lambda v: v)
    seller = synthetic("seller", lambda c: c)
    eff = an.efficiency(buyer, seller, 0.5, rng, n_boot=100)
    assert eff["efficiency"] == pytest.approx(1.0)
    ex, in_sample = an.exploit_buyer(buyer, 0.5)
    # A seller best-responding to a truthful buyer earns far more than the 9h/128
    # equilibrium payoff, and more than playing the equilibrium against the same buyer.
    assert ex["rival_br"].mean() > th.eq_surplus_each() + 3
    assert ex["rival_br"].mean() > ex["rival_eq"].mean()
    assert in_sample >= ex["rival_br"].mean()  # the oracle rival does at least as well


def test_equilibrium_agents_are_not_exploitable(rng):
    buyer = synthetic("buyer", th.eq_buyer_bid)
    seller = synthetic("seller", th.eq_seller_ask)
    for ex, _ in (an.exploit_buyer(buyer, 0.5), an.exploit_seller(seller, 0.5)):
        # best-responding gains at most sampling noise over playing the equilibrium
        assert ex["rival_br"].mean() - ex["rival_eq"].mean() < 0.4
        assert ex["rival_eq"].mean() == pytest.approx(th.eq_surplus_each(), abs=0.6)


def test_exploitability_intervals_contain_their_estimates(rng):
    # Noisy over-shading sellers, like the cheap model's: the out-of-fold estimate must sit
    # inside its own interval (refitting inside the bootstrap broke this).
    noise = np.random.default_rng(3)
    buyer = synthetic("buyer", th.eq_buyer_bid)
    seller = synthetic("seller", lambda c: 40 + 0.55 * c + noise.normal(0, 5, c.size))
    ex = an.exploitability(buyer, seller, 0.5, rng, n_boot=300)
    for entry in ex.values():
        for key in ("rival_br", "rival_eq", "br_gain_over_eq_play"):
            lo, hi = entry[f"{key}_ci"]
            assert lo <= entry[key] <= hi
        assert entry["rival_br_in_sample"] >= entry["rival_br"] - 0.5
    assert ex["vs_model_seller"]["br_gain_over_eq_play_ci"][0] > 0  # over-asking is exploitable


def test_eq_payoffs_match_closed_form():
    eq = an.eq_payoffs(0.5)
    assert eq["buyer"] == pytest.approx(th.eq_surplus_each(), rel=2e-3)
    assert eq["seller"] == pytest.approx(th.eq_surplus_each(), rel=2e-3)
    eq13 = an.eq_payoffs(1 / 3)
    assert eq13["gains"] / eq13["first_best"] == pytest.approx(th.eq_efficiency(1 / 3), abs=1e-3)


def test_recall_of_the_textbook_formula_is_detected(rng):
    # An agent playing k = 1/2 in a k = 1/3 auction is at the k = 1/2 formula, not at eq.
    buyer = synthetic("buyer", lambda v: th.eq_buyer_bid(v, k=0.5))
    side = an.describe_side("buyer", buyer, 1 / 3, rng, n_boot=50)
    assert side["share_at_k_half_formula"] == 1.0
    assert side["share_at_eq_to_the_cent"] < 0.2
