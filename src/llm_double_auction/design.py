"""Experimental design: the three conditions and the stratified draws of values and costs.

Values come from stratified uniform sampling: [0, 100) is cut into 20 strata of 5 euros and
replicate r of stratum j is one uniform draw inside it, rounded to cents. The draws depend
only on (role, stratum, replicate), so every condition sees the same numbers (common random
numbers), and pairing all buyer calls with all seller calls estimates the continuous
expectations without the grid artefacts a fixed set of midpoints would add.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .theory import H

STRATA = 20
MAX_REPLICATES = 20
SEED = 20261022
ROLES = ("buyer", "seller")


@dataclass(frozen=True)
class Condition:
    name: str
    label: str
    model: str
    thinking_level: str
    prompt_version: str
    replicates: int
    include_thoughts: bool = False
    k: float = 0.5  # price = k*b + (1 - k)*s


# A 2x2: model (Flash-Lite at minimal thinking, Flash at low thinking) x framing (the game
# named in its own vocabulary, or the same game told as a marketplace task), plus E. D and E
# were added after the pilot showed Flash reciting the textbook equilibrium in both framings:
# E changes the price rule to k = 1/3, whose equilibrium is not the famous one, to tell
# recall from reasoning. Replicates per stratum follow the pilot's cost per call: Flash-Lite
# is ~$0.0001 a call, Flash ~$0.003 on the textbook game and ~$0.01 when it has to derive.
CONDITIONS: dict[str, Condition] = {
    "A": Condition(
        "A", "Flash-Lite, informed prompt", "gemini-3.5-flash-lite", "minimal", "informed-v1", 10
    ),
    "B": Condition(
        "B", "Flash, informed prompt", "gemini-3.8-flash", "low", "informed-v1", 4,
        include_thoughts=True,
    ),
    "C": Condition(
        "C", "Flash-Lite, shopping-assistant prompt", "gemini-3.5-flash-lite", "minimal",
        "shopper-v1", 10,
    ),
    "D": Condition(
        "D", "Flash, shopping-assistant prompt", "gemini-3.8-flash", "low", "shopper-v1", 4,
        include_thoughts=True,
    ),
    "E": Condition(
        "E", "Flash, informed prompt, k = 1/3", "gemini-3.8-flash", "low", "informed-k13-v1",
        3, include_thoughts=True, k=1 / 3,
    ),
}


def draws(role: str) -> np.ndarray:
    """Matrix [replicate, stratum] of values (buyer) or costs (seller), in euros."""
    rng = np.random.default_rng([SEED, ROLES.index(role)])
    u = rng.uniform(size=(MAX_REPLICATES, STRATA))
    width = H / STRATA
    return np.round((np.arange(STRATA)[None, :] + u) * width, 2)


def plan(condition: Condition, roles=ROLES, strata=None, replicates=None):
    """(role, stratum, replicate, value) for every call the condition needs."""
    strata = range(STRATA) if strata is None else strata
    replicates = range(condition.replicates) if replicates is None else replicates
    out = []
    for role in roles:
        grid = draws(role)
        for r in replicates:
            for j in strata:
                out.append((role, j, r, float(grid[r, j])))
    return out
