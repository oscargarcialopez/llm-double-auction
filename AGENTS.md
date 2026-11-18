# AGENTS.md

## What this is

Can LLM agents trading on someone's behalf reach the Myerson–Satterthwaite bound? Gemini agents
bid on both sides of a sealed-bid double auction; the code measures how much of the available
gains they realise, how they bid and how much a best-responding rival could take from them. Every
model call is cached, so the analysis, figures, README numbers and carousel rebuild offline.
Theory, design, results and limits are in [README.md](README.md): link to it, don't restate it.

## Layout

- `src/llm_double_auction/`: the package. `cli.py` is the `auction` CLI; `agent.py` makes the API
  call; `prompts.py`, `design.py` and `theory.py` hold the prompts, the conditions and draws, and
  the closed forms; `analysis.py` turns the cache into results; `facts.py` formats every quotable
  number and checks texts against them; `figures.py`, `carousel.py`, `style.py` and `fonts/` draw.
- `tests/`: pytest, no API calls.
- `results/raw/calls.jsonl`: the call cache, one JSON line per API call, append-only.
- `results/summary.json`, `calls.csv`, `strata.csv`: derived from the cache by `make analyze`.
- `figures/` and `carousel/` (PDF, PNG slides, `slides.txt`): generated, and committed.
- `Makefile`: the interface. `pyproject.toml` and `uv.lock`: Python >= 3.12, locked dependencies.

## Commands

Offline, no key needed:

```bash
make install        # uv sync: .venv with every dependency, dev tools included
make check          # lint + test + check-numbers: what CI runs
make lint           # ruff on src and tests
make test           # the whole pytest suite
make check-numbers  # every number in README.md and carousel/slides.txt is in the results
make analyze        # cache -> results/summary.json, calls.csv, strata.csv
make figures        # README figures from the results
make carousel       # carousel PDF, PNG previews and slides.txt, from the results
make results        # analyze + figures + carousel: everything downstream of the cache
make cost           # tokens and dollars per condition, from the cache
make clean          # delete derived outputs, committed ones too; keeps the cache
uv run pytest tests/<file>.py::<test> -q   # a single test
uv run auction facts                       # every number a text may quote
uv run auction check FILE                  # flag numbers in FILE (a post, say) not in the results
```

These call the Gemini API with the key in `.env` (`pilot` and `run` are paid):

```bash
make models   # list the models the key can call -> results/models.json
make pilot    # a few calls per condition, reused by the main run
make run      # every call of the design not yet cached, up to the spend stop in costs.py
```

## Rules

- **Paid API calls are cached.** Never run `make models`, `make pilot`, `make run`, their
  `auction` subcommands or anything else that calls a model without the owner's explicit OK.
  `make results` rebuilds everything from the cache at zero cost, with no key.
- **Every number comes from the results.** Numbers in the README, the carousel and posts come
  from `results/summary.json` (formatted in `facts.py`) and are verified by `make check-numbers`
  (`uv run auction check FILE` for a post). If one is wrong, fix the code or data behind it and
  rerun `make results`: never change it in the prose alone, nor add it to `facts.CONSTANTS`.
- **The carousel is generated.** Its PDF, PNGs and `slides.txt` come from `make carousel`, as
  `figures/` comes from `make figures`. Never edit them by hand: change `carousel.py` or
  `figures.py` and regenerate. Never edit `results/raw/calls.jsonl` by hand either.
- **No secrets in git.** Never commit secrets, `.env` (it holds the key; `.env.example` is the
  template) or `*:Zone.Identifier` files. Check `git status` before every commit.
- **English only:** code, comments, docs and commit messages.

## Definition of done

```bash
make check   # ruff, pytest and the number check: must pass, exactly as in CI
```

If the change touches the analysis, the figures or the carousel, run `make results` first and
commit the regenerated files with the change.

## Commits

Conventional Commits, as in the history: `feat(carousel): ...`, `fix: ...`, `build: ...`,
`chore: ...`. Header at most 60 characters.
