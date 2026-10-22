PY := .venv/bin/python
AUCTION := .venv/bin/auction

.PHONY: install models pilot run cost analyze figures carousel results check-numbers test lint check clean

install:            ## create the venv and install everything
	uv sync

models:             ## list the Flash models the key can call (writes results/models.json)
	$(AUCTION) models

pilot:              ## 20 calls across the five conditions; all reused by the main run
	$(AUCTION) pilot

run:                ## every call of the design not yet cached (stops at $$3.50 spent)
	$(AUCTION) run --workers 12

cost:               ## tokens and dollars per condition, from usage_metadata
	$(AUCTION) cost

analyze:            ## cached calls -> results/summary.json, calls.csv, strata.csv
	$(AUCTION) analyze

figures:            ## README figures from the results files
	$(AUCTION) figures

carousel:           ## the 4:5 PDF carousel and PNG previews
	$(AUCTION) carousel

results: analyze figures carousel  ## everything downstream of the cache, no API calls

carousel/slides.txt:
	$(AUCTION) carousel

check-numbers: carousel/slides.txt  ## every number in the README and on the slides comes from the results
	$(AUCTION) check README.md carousel/slides.txt

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

check: lint test check-numbers  ## what CI would run

clean:              ## drop derived artefacts, keep the call cache
	rm -rf figures carousel results/summary.json results/calls.csv results/strata.csv
