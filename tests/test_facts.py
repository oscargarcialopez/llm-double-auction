"""The number checker that keeps the README, the carousel and the post honest."""

import json
from pathlib import Path

import pytest

from llm_double_auction import facts as fx

SUMMARY = Path(__file__).resolve().parents[1] / "results" / "summary.json"


def test_numbers_in_reads_claims_not_code_dates_or_versions():
    text = (
        "Flash realised 84.2% [83.8, 84.6] of 1,240 calls for $2.00 (−0.07 €).\n"
        "`gemini-3.8-flash` on 2026-10-19 with google-genai 2.32.0\n"
        "```bash\nuv sync  # Python 3.12\n```\n"
        "## References\n- Journal 29(2), 265–281.\n"
    )
    assert fx.numbers_in(text) == ["84.2", "83.8", "84.6", "1240", "2.00", "0.07"]


def test_numbers_in_reads_spanish_decimals():
    assert fx.numbers_in("pide 0,30c + 51,2 en 1240 llamadas, 2,00 $") == [
        "0.30", "51.2", "1240", "2.00"]


def test_check_flags_numbers_missing_from_the_facts():
    facts = {"eff": "84.2%", "ci": "[83.8, 84.6]"}
    assert fx.check("84.2% [83.8, 84.6], k = 1/3", facts) == []
    assert fx.check("84.3% of the gains", facts) == ["84.3"]


@pytest.mark.skipif(not SUMMARY.exists(), reason="no results yet")
def test_facts_from_the_committed_results():
    f = fx.build(json.loads(SUMMARY.read_text()))
    assert f["frontier"] == "84.4%"
    assert f["k13_eff"] == "83.3%"
    assert f["calls"] == "1,240"
    assert f["E.at_k_half"].startswith("0 of ")
