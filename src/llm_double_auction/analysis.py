"""From the cached calls to every number in the README, the figures and the post.

Each condition is analysed on its own: how the model bids as a function of its value, how
much of the possible gains a pair of such agents realises, and how much a rival who plays
the best response against it can take. Uncertainty comes from a stratified bootstrap over
calls (strata are the 5-euro value bands of the design).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import prompts
from . import theory as th
from .costs import PRICES_USD_PER_M, tokens
from .design import CONDITIONS, STRATA, Condition

RECOGNITION_TERMS = ("chatterjee", "samuelson")
DERIVATION = re.compile(r"\bfocs?\b|first[- ]order|derivative|differentiat")
CENT = 0.011  # "to the cent": within rounding of the equilibrium bid


@dataclass
class Side:
    value: np.ndarray
    amount: np.ndarray
    stratum: np.ndarray
    replicate: np.ndarray
    thoughts: list[str]


def load(cache_path: Path) -> tuple[dict, list[dict]]:
    recs = [json.loads(line) for line in Path(cache_path).read_text().splitlines() if line]
    for r in recs:
        expected = prompts.template_hash(r["prompt_version"], r["role"])
        if r["prompt_hash"] != expected:
            raise SystemExit(f"prompt drift in {r['key']}: bump the prompt version")
    by_cond: dict[str, dict[str, list[dict]]] = {}
    for r in recs:
        by_cond.setdefault(r["condition"], {}).setdefault(r["role"], []).append(r)
    return by_cond, recs


def side(records: list[dict]) -> Side:
    ok = [r for r in records if r["valid"]]
    ok.sort(key=lambda r: (r["stratum"], r["replicate"]))
    return Side(
        value=np.array([r["value"] for r in ok]),
        amount=np.array([r["amount"] for r in ok]),
        stratum=np.array([r["stratum"] for r in ok]),
        replicate=np.array([r["replicate"] for r in ok]),
        thoughts=[r.get("thoughts") or "" for r in ok],
    )


def strat_resample(rng: np.random.Generator, stratum: np.ndarray) -> np.ndarray:
    """Indices of a bootstrap sample drawn with replacement inside each stratum."""
    out = []
    for j in np.unique(stratum):
        idx = np.flatnonzero(stratum == j)
        out.append(rng.choice(idx, size=idx.size, replace=True))
    return np.concatenate(out)


def ci(samples, level: float = 0.95) -> list[float]:
    lo, hi = np.percentile(samples, [50 * (1 - level), 50 * (1 + level)])
    return [float(lo), float(hi)]


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return [float(centre - half), float(centre + half)]


def linfit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept), float(slope)


def eq_strategy(role: str, x, k: float):
    return th.eq_buyer_bid(x, k=k) if role == "buyer" else th.eq_seller_ask(x, k=k)


def trade_region(role: str, x, k: float):
    """Values that can trade against an equilibrium opponent: a buyer whose equilibrium bid
    reaches the lowest equilibrium ask, a seller whose ask is below the highest bid. Outside
    it, what an agent bids never changes its payoff against the equilibrium."""
    if role == "buyer":
        return th.eq_buyer_bid(x, k=k) >= th.eq_seller_ask(0.0, k=k)
    return th.eq_seller_ask(x, k=k) <= th.eq_buyer_bid(th.H, k=k)


def fit_with_ci(x, y, stratum, rng, n_boot: int) -> dict:
    intercept, slope = linfit(x, y)
    boots = []
    for _ in range(n_boot):
        idx = strat_resample(rng, stratum)
        boots.append(linfit(x[idx], y[idx]))
    boots = np.array(boots)
    resid = y - (intercept + slope * x)
    return {
        "intercept": intercept,
        "intercept_ci": ci(boots[:, 0]),
        "slope": slope,
        "slope_ci": ci(boots[:, 1]),
        "resid_sd": float(resid.std(ddof=2)) if x.size > 2 else float("nan"),
    }


def describe_side(role: str, sd: Side, k: float, rng, n_boot: int) -> dict:
    """Bidding behaviour of one role: line fit, per-stratum means, rationality, recall."""
    x, y = sd.value, sd.amount
    full = fit_with_ci(x, y, sd.stratum, rng, n_boot)
    rel = trade_region(role, x, k)
    relevant = fit_with_ci(x[rel], y[rel], sd.stratum[rel], rng, n_boot)
    eq = eq_strategy(role, x, k)
    eq_half = eq_strategy(role, x, 0.5)
    # Bidding above your value (asking below your cost) is weakly dominated, yet the linear
    # equilibrium itself does it for buyers with v < h(1-k)/2... who never trade. What matters
    # is whether the dominated number could trade against an equilibrium opponent.
    irr = (y > x) if role == "buyer" else (y < x)
    if role == "buyer":
        binding = irr & (y >= th.eq_seller_ask(0.0, k=k))
    else:
        binding = irr & (y <= th.eq_buyer_bid(th.H, k=k))
    strata = []
    for j in np.unique(sd.stratum):
        m = sd.stratum == j
        means = [y[m][rng.integers(0, m.sum(), m.sum())].mean() for _ in range(n_boot)]
        strata.append({
            "stratum": int(j),
            "n": int(m.sum()),
            "value_mean": float(x[m].mean()),
            "amount_mean": float(y[m].mean()),
            "amount_ci": ci(means),
            "amount_sd": float(y[m].std(ddof=1)) if m.sum() > 1 else 0.0,
        })
    thoughts = [t.lower() for t in sd.thoughts]
    with_thoughts = [t for t in thoughts if t]
    recognised = sum(any(term in t for term in RECOGNITION_TERMS) for t in with_thoughts)
    derived = sum(bool(DERIVATION.search(t)) for t in with_thoughts)
    eq_slope = float(1 / (1 + k)) if role == "buyer" else float(1 / (2 - k))
    eq_int = float(eq_strategy(role, 0.0, k))
    at_eq = np.abs(y - eq) <= CENT
    return {
        "n": int(x.size),
        **full,
        "eq_intercept": eq_int,
        "eq_slope": eq_slope,
        "n_trade_region": int(rel.sum()),
        "fit_trade_region": relevant,
        "mean_abs_dev_from_eq": float(np.abs(y - eq).mean()),
        "mean_abs_dev_from_eq_trade_region": float(np.abs(y - eq)[rel].mean()),
        "share_at_eq_to_the_cent": float(at_eq.mean()),
        "share_at_eq_trade_region": float(at_eq[rel].mean()),
        "share_at_k_half_formula": float(np.mean(np.abs(y - eq_half) <= CENT)),
        "share_at_k_half_formula_trade_region": float(np.mean(np.abs(y - eq_half)[rel] <= CENT)),
        "ir_violations": int(irr.sum()),
        "ir_violation_share": float(irr.mean()),
        "ir_violation_ci": wilson(int(irr.sum()), int(x.size)),
        "ir_binding": int(binding.sum()),
        "ir_binding_share": float(binding.mean()),
        "ir_binding_ci": wilson(int(binding.sum()), int(x.size)),
        "out_of_range": int(np.sum((y < 0) | (y > th.H))),
        "thought_summaries": len(with_thoughts),
        "recognised_by_name": recognised,
        "derivation_terms": derived,
        "strata": strata,
    }


def efficiency(buyer: Side, seller: Side, k: float, rng, n_boot: int) -> dict:
    """All buyer calls against all seller calls, with a two-sample stratified bootstrap."""

    def summarise(vb, b, cs, s):
        o = th.pair_outcomes(vb, b, cs, s, k=k)
        fb = o["first_best"]
        return {
            "efficiency": o["gains"] / fb,
            "missed_share": o["missed"] / fb,
            "bad_share": o["bad"] / fb,
            "buyer_payoff": o["buyer"],
            "seller_payoff": o["seller"],
            "gains": o["gains"],
            "first_best": fb,
            "trade_rate": o["trade_rate"],
        }

    point = summarise(buyer.value, buyer.amount, seller.value, seller.amount)
    reps = []
    for _ in range(n_boot):
        ib = strat_resample(rng, buyer.stratum)
        js = strat_resample(rng, seller.stratum)
        reps.append(summarise(buyer.value[ib], buyer.amount[ib], seller.value[js],
                              seller.amount[js]))
    out = {key: float(val) for key, val in point.items()}
    for key in point:
        out[f"{key}_ci"] = ci([r[key] for r in reps])
    same_sample = th.pair_outcomes(
        buyer.value, th.eq_buyer_bid(buyer.value, k=k),
        seller.value, th.eq_seller_ask(seller.value, k=k), k=k,
    )
    out["eq_efficiency_same_draws"] = float(same_sample["gains"] / same_sample["first_best"])
    out["eq_trade_rate_same_draws"] = float(same_sample["trade_rate"])
    out["eq_efficiency_theory"] = float(th.eq_efficiency(k))
    return out


COST_GRID = (np.arange(400) + 0.5) * th.H / 400  # the rival's own value or cost, integrated


def folds(sd: Side) -> list[np.ndarray]:
    """Two halves split by replicate parity: every stratum appears in both."""
    a = np.flatnonzero(sd.replicate % 2 == 0)
    b = np.flatnonzero(sd.replicate % 2 == 1)
    return [a, b] if a.size and b.size else [np.arange(sd.value.size)] * 2


EXPLOIT_KEYS = ("rival_br", "model_vs_br", "rival_eq", "model_vs_eq")


def exploit_buyer(buyer: Side, k: float) -> tuple[dict[str, np.ndarray], float]:
    """What a seller who best-responds to the model's bids earns, per model call.

    The rival's ask for each of its costs is fitted on one half of the calls and scored on
    the other (cross-fitting keeps the optimisation from grading its own homework), so every
    call gets an out-of-fold payoff, averaged over the rival's cost. The same for a rival
    playing the equilibrium. Also returns the in-sample best response's payoff, an
    optimistic bound: a rival who knew the model's whole bid distribution.
    """
    v, b = buyer.value, buyer.amount
    out = {key: np.empty(v.size) for key in EXPLOIT_KEYS}
    eq_asks = th.eq_seller_ask(COST_GRID, k=k)
    fa, fb = folds(buyer)
    for train, test in ((fa, fb), (fb, fa)):
        asks, _ = th.seller_best_response_empirical(COST_GRID, b[train], k=k)
        out["rival_br"][test], out["model_vs_br"][test] = th.seller_payoff_against(
            asks, COST_GRID, b[test], v[test], k=k, per="opponent")
        out["rival_eq"][test], out["model_vs_eq"][test] = th.seller_payoff_against(
            eq_asks, COST_GRID, b[test], v[test], k=k, per="opponent")
    _, in_sample = th.seller_best_response_empirical(COST_GRID, b, k=k)
    return out, float(in_sample.mean())


def exploit_seller(seller: Side, k: float) -> tuple[dict[str, np.ndarray], float]:
    """What a buyer who best-responds to the model's asks earns, cross-fitted the same way."""
    c, s = seller.value, seller.amount
    out = {key: np.empty(c.size) for key in EXPLOIT_KEYS}
    eq_bids = th.eq_buyer_bid(COST_GRID, k=k)
    fa, fb = folds(seller)
    for train, test in ((fa, fb), (fb, fa)):
        bids, _ = th.buyer_best_response_empirical(COST_GRID, s[train], k=k)
        out["rival_br"][test], out["model_vs_br"][test] = th.buyer_payoff_against(
            bids, COST_GRID, s[test], c[test], k=k, per="opponent")
        out["rival_eq"][test], out["model_vs_eq"][test] = th.buyer_payoff_against(
            eq_bids, COST_GRID, s[test], c[test], k=k, per="opponent")
    _, in_sample = th.buyer_best_response_empirical(COST_GRID, s, k=k)
    return out, float(in_sample.mean())


def exploitability(buyer: Side, seller: Side, k: float, rng, n_boot: int) -> dict:
    """Means of the out-of-fold payoffs, with CIs that hold the fitted strategies fixed and
    resample (within strata) the calls they are scored on. Refitting inside each bootstrap
    sample would not do: duplicated calls make the fit overfit, which biases the intervals
    down."""
    out = {}
    for name, sd, fn in (("vs_model_buyer", buyer, exploit_buyer),
                         ("vs_model_seller", seller, exploit_seller)):
        per_call, in_sample = fn(sd, k)
        per_call["br_gain_over_eq_play"] = per_call["rival_br"] - per_call["rival_eq"]
        idx = [strat_resample(rng, sd.stratum) for _ in range(n_boot)]
        entry: dict = {}
        for key, vals in per_call.items():
            entry[key] = float(vals.mean())
            entry[f"{key}_ci"] = ci([vals[i].mean() for i in idx])
        entry["rival_br_in_sample"] = in_sample
        entry["br_gain_in_sample"] = in_sample - entry["rival_eq"]
        out[name] = entry
    return out


def eq_payoffs(k: float) -> dict:
    """Each side's expected payoff when both play the linear equilibrium (fine grid)."""
    g = (np.arange(1500) + 0.5) * th.H / 1500
    o = th.pair_outcomes(g, th.eq_buyer_bid(g, k=k), g, th.eq_seller_ask(g, k=k), k=k)
    return {"buyer": float(o["buyer"]), "seller": float(o["seller"]),
            "gains": float(o["gains"]), "first_best": float(o["first_best"])}


def midpoint_grid_benchmark(strata: int = STRATA) -> dict:
    """Why the design draws values inside the bands instead of using their midpoints: on a
    midpoint grid many pairs sit exactly on the equilibrium's trade boundary v - c = h/4, and
    whether they trade (b >= s up to floating-point rounding) moves the benchmark itself."""
    g = (np.arange(strata) + 0.5) * th.H / strata
    d = g[:, None] - g[None, :]
    first_best = np.maximum(d, 0.0).mean()
    gap = th.eq_trade_gap()
    on_boundary = np.isclose(d, gap)
    return {
        "pairs": int(d.size),
        "pairs_on_boundary": int(on_boundary.sum()),
        "eq_efficiency_if_boundary_pairs_skip": float(
            np.where((d > gap) & ~on_boundary, d, 0.0).mean() / first_best),
        "eq_efficiency_if_boundary_pairs_trade": float(
            np.where((d > gap) | on_boundary, d, 0.0).mean() / first_best),
    }


def cost_table(recs: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for r in recs:
        e = out.setdefault(r["condition"], {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                            "thinking_tokens": 0, "usd": 0.0,
                                            "models": set()})
        p, o = tokens(r["usage"])
        e["calls"] += 1
        e["input_tokens"] += p
        e["output_tokens"] += o
        e["thinking_tokens"] += r["usage"].get("thoughts_token_count") or 0
        e["usd"] += r["cost_usd"]
        e["models"].add(r["model_version"] or r["model"])
    for e in out.values():
        e["models"] = sorted(e["models"])
        e["usd_per_call"] = e["usd"] / e["calls"]
    return out


def analyze(cache_path: Path, out_dir: Path, n_boot: int = 2000, seed: int = 7) -> dict:
    by_cond, recs = load(cache_path)
    rng = np.random.default_rng(seed)
    summary: dict = {
        "theory": {
            "ms_frontier": th.EQ_EFFICIENCY,
            "first_best_gains": th.first_best_gains(),
            "eq_gains_k_half": th.eq_gains(),
            "eq_payoff_each_k_half": th.eq_surplus_each(),
            "eq_k_third": eq_payoffs(1 / 3) | {"efficiency": th.eq_efficiency(1 / 3)},
            "midpoint_grid": midpoint_grid_benchmark(),
        },
        "prices_usd_per_m": PRICES_USD_PER_M,
        "conditions": {},
    }
    costs = cost_table(recs)
    summary["cost"] = {
        "total_usd": float(sum(e["usd"] for e in costs.values())),
        "calls": int(sum(e["calls"] for e in costs.values())),
        "retried_calls": int(sum(r["attempts"] > 1 for r in recs)),
        "invalid_replies": int(sum(not r["valid"] for r in recs)),
        "by_condition": costs,
    }
    for name, cond in CONDITIONS.items():
        roles = by_cond.get(name)
        if not roles or "buyer" not in roles or "seller" not in roles:
            continue
        buyer, seller = side(roles["buyer"]), side(roles["seller"])
        n_bad = sum(not r["valid"] for rs in roles.values() for r in rs)
        summary["conditions"][name] = describe_condition(cond, buyer, seller, n_bad, rng, n_boot)
        print(f"{name}: efficiency {summary['conditions'][name]['efficiency']['efficiency']:.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=list) + "\n")
    write_calls_csv(recs, out_dir / "calls.csv")
    write_strata_csv(summary, out_dir / "strata.csv")
    print(f"-> {out_dir / 'summary.json'}")
    return summary


def describe_condition(cond: Condition, buyer: Side, seller: Side, n_bad: int, rng,
                       n_boot: int) -> dict:
    eq = eq_payoffs(cond.k)
    return {
        "label": cond.label,
        "model": cond.model,
        "thinking_level": cond.thinking_level,
        "prompt_version": cond.prompt_version,
        "k": cond.k,
        "invalid_replies": n_bad,
        "buyer": describe_side("buyer", buyer, cond.k, rng, n_boot),
        "seller": describe_side("seller", seller, cond.k, rng, n_boot),
        "efficiency": efficiency(buyer, seller, cond.k, rng, n_boot),
        "eq_payoffs": eq,
        "exploitability": exploitability(buyer, seller, cond.k, rng, n_boot),
    }


def write_calls_csv(recs: list[dict], path: Path) -> None:
    cols = ["condition", "model", "model_version", "prompt_version", "thinking_level", "role",
            "stratum", "replicate", "value", "amount", "valid", "input_tokens",
            "output_tokens", "thinking_tokens", "cost_usd", "latency_s"]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in sorted(recs, key=lambda r: (r["condition"], r["role"], r["stratum"],
                                             r["replicate"])):
            p, o = tokens(r["usage"])
            w.writerow([r["condition"], r["model"], r["model_version"], r["prompt_version"],
                        r["thinking_level"], r["role"], r["stratum"], r["replicate"],
                        r["value"], r["amount"], r["valid"], p, o,
                        r["usage"].get("thoughts_token_count") or 0,
                        round(r["cost_usd"], 6), r["latency_s"]])


def write_strata_csv(summary: dict, path: Path) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "role", "stratum", "n", "value_mean", "amount_mean",
                    "ci_low", "ci_high", "amount_sd"])
        for name, c in summary["conditions"].items():
            for role in ("buyer", "seller"):
                for s in c[role]["strata"]:
                    w.writerow([name, role, s["stratum"], s["n"], round(s["value_mean"], 3),
                                round(s["amount_mean"], 3), round(s["amount_ci"][0], 3),
                                round(s["amount_ci"][1], 3), round(s["amount_sd"], 3)])
