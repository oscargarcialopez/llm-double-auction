"""Every number the README, the carousel and the post may quote, formatted once from
results/summary.json, and a checker that flags any number in a text that is not one of them.

The carousel takes its numbers from here directly; prose written by hand is checked with
`auction check FILE...`.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import theory as th
from .design import CONDITIONS, SEED, STRATA
from .style import pct

# Numbers that are not results: model versions (3.5, 3.8), the value range and the theory's
# thresholds, slopes, intercepts and fractions, prices per million tokens and the spend stop,
# publication years, design sizes (bootstrap draws, cost grid), the confidence level and the
# HTTP codes of the retried calls.
CONSTANTS = {
    "3.5", "3.8",
    "100", "25", "75", "50", "0.67", "8.3", "33.3", "0.75", "0.60", "27", "32", "64", "12",
    "0.30", "2.50", "3.75", "1.50", "7.50", "3.50",  # (0.75 is also a price)
    "1983", "2026", "2027",
    "2,000", "400", "95", "500", "503",
    str(STRATA), str(SEED),
}


def signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}".replace("-", "−")


def euros(x: float) -> str:
    return f"{x:.2f} €"


def build(summary: dict) -> dict[str, str]:
    """Named, formatted facts. Percentages carry one decimal; euros two."""
    f: dict[str, str] = {}
    theory = summary["theory"]
    f["frontier"] = pct(theory["ms_frontier"])
    f["frontier_exact"] = pct(theory["ms_frontier"], 3)
    f["lost_at_frontier"] = pct(1 - theory["ms_frontier"])
    f["first_best_eur"] = euros(theory["first_best_gains"])
    f["eq_gains_eur"] = euros(theory["eq_gains_k_half"])
    f["eq_each_eur"] = euros(theory["eq_payoff_each_k_half"])
    f["k13_eff"] = pct(theory["eq_k_third"]["efficiency"])
    f["k13_buyer_eur"] = euros(theory["eq_k_third"]["buyer"])
    f["k13_seller_eur"] = euros(theory["eq_k_third"]["seller"])
    grid = theory["midpoint_grid"]
    f["grid_pairs_on_boundary"] = f"{grid['pairs_on_boundary']} of {grid['pairs']}"
    f["grid_eq_eff_skip"] = pct(grid["eq_efficiency_if_boundary_pairs_skip"])
    f["grid_eq_eff_trade"] = pct(grid["eq_efficiency_if_boundary_pairs_trade"])
    cost = summary["cost"]
    f["calls"] = f"{cost['calls']:,}"
    f["retried_calls"] = str(cost["retried_calls"])
    f["invalid_replies"] = str(cost["invalid_replies"])
    f["cost_usd"] = f"${cost['total_usd']:.2f}"
    for name, e in cost["by_condition"].items():
        f[f"{name}.calls"] = f"{e['calls']:,}"
        f[f"{name}.cost_usd"] = f"${e['usd']:.2f}"
        f[f"{name}.usd_per_call"] = f"${e['usd_per_call']:.5f}"
        f[f"{name}.thinking_per_call"] = f"{e['thinking_tokens'] / e['calls']:,.0f}"
    flash_rel = flash_rel_n = flash_named = flash_thoughts = flash_derived = 0
    for name, c in summary["conditions"].items():
        e = c["efficiency"]
        f[f"{name}.eff"] = pct(e["efficiency"])
        lo, hi = e["efficiency_ci"]
        f[f"{name}.eff_ci"] = f"[{100 * lo:.1f}, {100 * hi:.1f}]"
        f[f"{name}.eq_same_draws"] = pct(e["eq_efficiency_same_draws"])
        f[f"{name}.missed"] = pct(e["missed_share"])
        f[f"{name}.bad"] = pct(e["bad_share"], 2)
        f[f"{name}.trade_rate"] = pct(e["trade_rate"])
        f[f"{name}.eq_trade_rate"] = pct(e["eq_trade_rate_same_draws"])
        f[f"{name}.buyer_eur"] = euros(e["buyer_payoff"])
        f[f"{name}.seller_eur"] = euros(e["seller_payoff"])
        f[f"{name}.gains_eur"] = euros(e["gains"])
        for role in ("buyer", "seller"):
            d = c[role]
            p = f"{name}.{role}"
            f[f"{p}.n"] = str(d["n"])
            fit = d["fit_trade_region"]
            f[f"{p}.slope"] = f"{fit['slope']:.2f}"
            f[f"{p}.intercept"] = f"{fit['intercept']:.1f}"
            f[f"{p}.slope_ci"] = f"[{fit['slope_ci'][0]:.2f}, {fit['slope_ci'][1]:.2f}]"
            f[f"{p}.intercept_ci"] = (f"[{fit['intercept_ci'][0]:.1f}, "
                                      f"{fit['intercept_ci'][1]:.1f}]")
            f[f"{p}.slope_all"] = f"{d['slope']:.2f}"
            f[f"{p}.intercept_all"] = f"{d['intercept']:.1f}"
            n_rel = d["n_trade_region"]
            at_eq = round(d["share_at_eq_trade_region"] * n_rel)
            f[f"{p}.n_trade_region"] = str(n_rel)
            f[f"{p}.at_eq_trade_region"] = f"{at_eq} of {n_rel}"
            f[f"{p}.at_eq_trade_region_pct"] = pct(d["share_at_eq_trade_region"])
            f[f"{p}.at_eq_all_pct"] = pct(d["share_at_eq_to_the_cent"])
            f[f"{p}.at_eq_all"] = f"{round(d['share_at_eq_to_the_cent'] * d['n'])} of {d['n']}"
            f[f"{p}.at_k_half"] = f"{round(d['share_at_k_half_formula'] * d['n'])} of {d['n']}"
            f[f"{p}.ir_violations"] = f"{d['ir_violations']} of {d['n']}"
            f[f"{p}.ir_violation_pct"] = pct(d["ir_violation_share"])
            f[f"{p}.ir_binding"] = f"{d['ir_binding']} of {d['n']}"
            f[f"{p}.ir_binding_pct"] = pct(d["ir_binding_share"])
            lo, hi = d["ir_binding_ci"]
            f[f"{p}.ir_binding_ci"] = f"[{100 * max(lo, 0):.1f}, {100 * hi:.1f}]"
            f[f"{p}.named"] = f"{d['recognised_by_name']} of {d['thought_summaries']}"
            f[f"{p}.derivation_terms"] = f"{d['derivation_terms']} of {d['thought_summaries']}"
            f[f"{p}.mad_trade_region"] = euros(d["mean_abs_dev_from_eq_trade_region"])
            if c["model"] == "gemini-3.8-flash" and c["k"] == 0.5:
                flash_rel += at_eq
                flash_rel_n += n_rel
                flash_named += d["recognised_by_name"]
                flash_thoughts += d["thought_summaries"]
                flash_derived += d["derivation_terms"]
            x = c["exploitability"]["vs_model_" + role]
            q = f"{name}.exploit_{role}"
            f[f"{q}.gain"] = signed(x["br_gain_over_eq_play"])
            glo, ghi = x["br_gain_over_eq_play_ci"]
            f[f"{q}.gain_ci"] = f"[{signed(glo)}, {signed(ghi)}]"
            f[f"{q}.gain_in_sample"] = signed(x["br_gain_in_sample"])
            f[f"{q}.rival_eq_eur"] = euros(x["rival_eq"])
            f[f"{q}.rival_br_eur"] = euros(x["rival_br"])
            rel = round(100 * x["br_gain_over_eq_play"] / x["rival_eq"])
            f[f"{q}.gain_rel"] = signed(rel, 0) + "%" if rel else "0%"
            f[f"{q}.model_vs_br_eur"] = euros(x["model_vs_br"])
            f[f"{q}.model_vs_eq_eur"] = euros(x["model_vs_eq"])
    f["flash_k_half.at_eq_trade_region"] = f"{flash_rel} of {flash_rel_n}"
    f["flash_k_half.named"] = f"{flash_named} of {flash_thoughts}"
    f["flash_k_half.derivation_terms"] = f"{flash_derived} of {flash_thoughts}"
    e13 = summary["conditions"].get("E")
    if e13:
        rel = sum(round(e13[r]["share_at_eq_trade_region"] * e13[r]["n_trade_region"])
                  for r in ("buyer", "seller"))
        rel_n = sum(e13[r]["n_trade_region"] for r in ("buyer", "seller"))
        half = sum(round(e13[r]["share_at_k_half_formula"] * e13[r]["n"])
                   for r in ("buyer", "seller"))
        n = sum(e13[r]["n"] for r in ("buyer", "seller"))
        f["E.at_eq_trade_region"] = f"{rel} of {rel_n}"
        f["E.at_k_half"] = f"{half} of {n}"
        n_thoughts = sum(e13[r]["thought_summaries"] for r in ("buyer", "seller"))
        f["E.named"] = (f"{sum(e13[r]['recognised_by_name'] for r in ('buyer', 'seller'))}"
                        f" of {n_thoughts}")
        f["E.derivation_terms"] = (
            f"{sum(e13[r]['derivation_terms'] for r in ('buyer', 'seller'))} of {n_thoughts}")
    for name, cond in CONDITIONS.items():
        f[f"{name}.replicates"] = str(cond.replicates)
    f["H"] = f"{th.H:.0f}"
    return f


NUMBER = re.compile(r"(?<![\w.])[+\-−]?\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?%?")


def numbers_in(text: str) -> list[str]:
    """Numeric tokens, without sign, currency, percent or thousands separators. Code spans,
    links, dates, dotted version numbers and a trailing References section are not claims."""
    text = re.split(r"^#+ References\s*$", text, flags=re.M)[0]
    text = re.sub(r"^```.*?^```", " ", text, flags=re.S | re.M)
    text = re.sub(r"`[^`\n]*`", " ", text)
    text = re.sub(r"(?<![\d,.])(\d+),(\d{1,2})(?!\d)", r"\1.\2", text)  # 84,2 (Spanish)
    text = re.sub(r"\(https?://[^)]*\)|https?://\S+", " ", text)
    text = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", " ", text)
    text = re.sub(r"\b\d+\.\d+\.\d+\b", " ", text)
    return [(m.group(1) + (m.group(2) or "")).replace(",", "") for m in NUMBER.finditer(text)]


def allowed(facts: dict[str, str]) -> set[str]:
    ok = {n for v in facts.values() for n in numbers_in(v)}
    ok |= {c.replace(",", "") for c in CONSTANTS}
    ok |= {str(i) for i in range(11)}
    return ok


def check(text: str, facts: dict[str, str]) -> list[str]:
    ok = allowed(facts)
    return sorted({n for n in numbers_in(text) if n not in ok}, key=float)


def check_files(paths: list[Path], summary: dict) -> int:
    facts = build(summary)
    bad = 0
    for path in paths:
        text = Path(path).read_text()
        missing = check(text, facts)
        status = "ok" if not missing else "NOT IN RESULTS: " + ", ".join(missing)
        print(f"{path}: {len(numbers_in(text))} numbers; {status}")
        bad += len(missing)
    return bad
