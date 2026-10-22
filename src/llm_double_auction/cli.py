"""auction models | pilot | run | cost | analyze | figures | carousel | facts | check"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from .agent import ROOT, QuotaExhausted, ask, make_client
from .cache import CallCache, call_key
from .costs import BUDGET_STOP_USD
from .design import CONDITIONS, plan

CACHE_PATH = ROOT / "results" / "raw" / "calls.jsonl"

PILOT = {  # (condition, strata, replicate 0): 20 calls, all reused by the main run
    "A": [3, 10, 17],
    "B": [5, 15],
    "C": [5, 15],
    "D": [5, 15],
    "E": [10],
}


def cmd_models(_args) -> None:
    client = make_client()
    rows = []
    for m in client.models.list():
        if "generateContent" in (m.supported_actions or []) and "flash" in m.name:
            rows.append({"name": m.name, "version": m.version, "display_name": m.display_name})
    out = ROOT / "results" / "models.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    for r in rows:
        print(f"{r['name']:45s} {r['version']}")
    print(f"-> {out.relative_to(ROOT)}")


def execute(calls, cache: CallCache, workers: int) -> None:
    """Run (condition, role, stratum, replicate, value) calls not yet cached, in parallel,
    stopping before the spend passes BUDGET_STOP_USD."""
    client = make_client()
    todo = []
    for cond, role, j, r, value in calls:
        key = call_key(cond.model, cond.prompt_version, cond.thinking_level, role, value, r)
        if key not in cache:
            todo.append((cond, role, j, r, value))
    print(f"{len(calls)} planned, {len(calls) - len(todo)} cached, {len(todo)} to call; "
          f"spent so far ${cache.total_cost():.4f}")
    if not todo:
        return
    pending = set()
    queue = list(todo)
    done = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while queue or pending:
            while queue and len(pending) < workers:
                if cache.total_cost() >= BUDGET_STOP_USD:
                    raise SystemExit(f"budget stop: ${cache.total_cost():.4f} spent")
                cond, role, j, r, value = queue.pop(0)
                pending.add(pool.submit(ask, client, cond, role, j, r, value))
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                try:
                    cache.put(fut.result())
                    done += 1
                except QuotaExhausted as err:
                    for p in pending:
                        p.cancel()
                    raise SystemExit(f"daily quota exhausted after {done} calls: {err}") from err
                except Exception as err:  # keep going; the call stays uncached and reruns
                    failed += 1
                    print(f"  call failed: {type(err).__name__}: {str(err)[:300]}")
            if done and done % 50 == 0:
                print(f"  {done}/{len(todo)} done, ${cache.total_cost():.4f}")
    print(f"done {done}, failed {failed}; spent ${cache.total_cost():.4f}")


def cmd_pilot(args) -> None:
    cache = CallCache(CACHE_PATH)
    calls = []
    for name, strata in PILOT.items():
        cond = CONDITIONS[name]
        for role, j, r, value in plan(cond, strata=strata, replicates=[0]):
            calls.append((cond, role, j, r, value))
    execute(calls, cache, workers=args.workers)
    print_cost_table(cache)


def cmd_run(args) -> None:
    cache = CallCache(CACHE_PATH)
    names = args.condition or list(CONDITIONS)
    calls = []
    for name in names:
        cond = CONDITIONS[name]
        calls += [(cond, role, j, r, value) for role, j, r, value in plan(cond)]
    projected = project_cost(calls, cache)
    print(f"projected total spend ${projected:.3f} (stop at ${BUDGET_STOP_USD:.2f})")
    if projected > BUDGET_STOP_USD:
        raise SystemExit("projected spend is over the stop: shrink the design first")
    execute(calls, cache, workers=args.workers)
    print_cost_table(cache)


def project_cost(calls, cache: CallCache) -> float:
    """Spend so far plus, for uncached calls, the mean cost of cached calls of that
    condition and role (the pilot first), with a 25% margin."""
    spent = cache.total_cost()
    by_kind: dict[tuple, list[float]] = {}
    for rec in cache.records.values():
        by_kind.setdefault((rec["condition"], rec["role"]), []).append(rec["cost_usd"])
    extra = 0.0
    for cond, role, _j, r, value in calls:
        key = call_key(cond.model, cond.prompt_version, cond.thinking_level, role, value, r)
        if key in cache:
            continue
        seen = by_kind.get((cond.name, role))
        if not seen:
            raise SystemExit(f"no pilot calls for condition {cond.name}/{role}: run the pilot")
        extra += 1.25 * sum(seen) / len(seen)  # 25% margin
    return spent + extra


def print_cost_table(cache: CallCache) -> None:
    from .costs import tokens

    agg: dict[tuple, dict] = {}
    for rec in cache.records.values():
        a = agg.setdefault((rec["condition"], rec["role"]), {"n": 0, "in": 0, "out": 0,
                                                             "think": 0, "usd": 0.0, "lat": 0.0})
        p, o = tokens(rec["usage"])
        a["n"] += 1
        a["in"] += p
        a["out"] += o
        a["think"] += rec["usage"].get("thoughts_token_count") or 0
        a["usd"] += rec["cost_usd"]
        a["lat"] += rec["latency_s"]
    print(f"{'cond':4s} {'role':6s} {'n':>4s} {'in/call':>8s} {'out/call':>9s} "
          f"{'think/call':>10s} {'$/call':>9s} {'s/call':>7s} {'$ total':>8s}")
    for (c, r), a in sorted(agg.items()):
        n = a["n"]
        print(f"{c:4s} {r:6s} {n:4d} {a['in'] / n:8.0f} {a['out'] / n:9.0f} "
              f"{a['think'] / n:10.0f} {a['usd'] / n:9.5f} {a['lat'] / n:7.2f} {a['usd']:8.4f}")
    print(f"total ${cache.total_cost():.4f} over {len(cache)} calls")


def cmd_cost(_args) -> None:
    print_cost_table(CallCache(CACHE_PATH))


def cmd_analyze(args) -> None:
    from .analysis import analyze

    analyze(CACHE_PATH, ROOT / "results", n_boot=args.boot)


def cmd_figures(_args) -> None:
    from .figures import make_figures

    make_figures(ROOT / "results", ROOT / "figures")


def cmd_carousel(_args) -> None:
    from .carousel import make_carousel

    make_carousel(ROOT / "results", ROOT / "carousel")


def cmd_facts(_args) -> None:
    from .facts import build
    from .style import load_summary

    for key, value in build(load_summary(ROOT / "results")).items():
        print(f"{key:40s} {value}")


def cmd_check(args) -> None:
    from .facts import check_files
    from .style import load_summary

    if check_files(args.files, load_summary(ROOT / "results")):
        raise SystemExit(1)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="auction")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models").set_defaults(func=cmd_models)
    p = sub.add_parser("pilot")
    p.add_argument("--workers", type=int, default=14)
    p.set_defaults(func=cmd_pilot)
    p = sub.add_parser("run")
    p.add_argument("--condition", nargs="*", choices=list(CONDITIONS))
    p.add_argument("--workers", type=int, default=8)
    p.set_defaults(func=cmd_run)
    sub.add_parser("cost").set_defaults(func=cmd_cost)
    p = sub.add_parser("analyze")
    p.add_argument("--boot", type=int, default=2000)
    p.set_defaults(func=cmd_analyze)
    sub.add_parser("figures").set_defaults(func=cmd_figures)
    sub.add_parser("carousel").set_defaults(func=cmd_carousel)
    sub.add_parser("facts").set_defaults(func=cmd_facts)
    p = sub.add_parser("check", help="flag numbers in a text that are not in the results")
    p.add_argument("files", nargs="+")
    p.set_defaults(func=cmd_check)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
