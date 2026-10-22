"""What the calls cost, from usage_metadata and the published prices.

Gemini API paid tier, standard (not batch) text prices in USD per million tokens, from
https://ai.google.dev/gemini-api/docs/pricing ("Last updated 2026-10-19 UTC"). Thinking
tokens are billed as output. gemini-3.8-flash doubles both prices on 2027-01-01.
"""

from __future__ import annotations

PRICES_USD_PER_M = {
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
}

BUDGET_STOP_USD = 3.50


def tokens(usage: dict) -> tuple[int, int]:
    """(input, billed output) tokens; billed output = answer + thinking."""
    prompt = usage.get("prompt_token_count") or 0
    answer = usage.get("candidates_token_count") or 0
    thoughts = usage.get("thoughts_token_count") or 0
    return prompt, answer + thoughts


def call_cost(model: str, usage: dict) -> float:
    price = PRICES_USD_PER_M[model]
    prompt, output = tokens(usage)
    return (prompt * price["input"] + output * price["output"]) / 1e6
