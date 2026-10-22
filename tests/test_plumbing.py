"""Parsing replies, the cache, costs, prompts and the design: no API calls."""

import numpy as np
import pytest

from llm_double_auction import prompts
from llm_double_auction.agent import parse_amount
from llm_double_auction.cache import CallCache, call_key
from llm_double_auction.costs import call_cost
from llm_double_auction.design import CONDITIONS, STRATA, draws, plan


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"bid_eur": 42.5}', 42.5),
        ('```json\n{"bid_eur": 7}\n```', 7.0),
        ('{"bid_eur": "about 31,25 euros"}', 31.25),
        ('{"other": 3}', None),
        ("not json", None),
        ('{"bid_eur": true}', None),
        ("", None),
    ],
)
def test_parse_amount(text, expected):
    assert parse_amount(text, "bid_eur") == expected


def test_cache_is_idempotent_and_persistent(tmp_path):
    path = tmp_path / "calls.jsonl"
    cache = CallCache(path)
    key = call_key("m", "p-v1", "low", "buyer", 12.345, 0)
    assert key == "m|p-v1|low|buyer|12.35|0"
    cache.put({"key": key, "cost_usd": 0.5})
    cache.put({"key": key, "cost_usd": 0.5})  # second write ignored
    again = CallCache(path)
    assert len(again) == 1 and key in again
    assert again.total_cost() == pytest.approx(0.5)


def test_call_cost_counts_thinking_as_output():
    usage = {"prompt_token_count": 1_000_000, "candidates_token_count": 0,
             "thoughts_token_count": 1_000_000}
    assert call_cost("gemini-3.8-flash", usage) == pytest.approx(0.75 + 3.75)


def test_prompts_state_rules_and_prior_but_no_solution():
    for version in prompts.FRAMINGS:
        for role in ("buyer", "seller"):
            text, field = prompts.render(version, role, 37.5)
            assert "37.50" in text
            assert "between 0" in text and "100" in text
            for leak in ("2/3", "1/12", "Chatterjee", "equilibrium", "Nash"):
                assert leak not in text
            assert prompts.response_schema(field)["required"] == [field]


def test_shopper_prompt_avoids_game_theory_vocabulary():
    for role in ("buyer", "seller"):
        text, _ = prompts.render("shopper-v1", role, 50.0)
        for word in ("auction", "risk-neutral", "common knowledge", "uniformly", "payoff"):
            assert word not in text.lower()


def test_draws_are_stratified_and_shared_across_conditions():
    for role in ("buyer", "seller"):
        grid = draws(role)
        width = 100 / STRATA
        lo = np.arange(STRATA) * width
        assert np.all(grid >= lo - 1e-9) and np.all(grid <= lo + width + 1e-9)
        assert np.array_equal(grid, draws(role))
    a, b = plan(CONDITIONS["A"]), plan(CONDITIONS["B"])
    assert set(b) <= set(a)  # B uses the first replicates of the same draws
