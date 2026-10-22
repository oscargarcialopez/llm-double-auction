"""One sealed bid from Gemini: build the prompt, call the model, keep everything it returns."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from . import prompts
from .cache import call_key
from .costs import call_cost
from .design import Condition

ROOT = Path(__file__).resolve().parents[2]


class QuotaExhausted(RuntimeError):
    """A daily quota ran out: retrying today is pointless."""


def make_client() -> genai.Client:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY is not set (copy .env.example to .env)")
    return genai.Client(api_key=key)


def parse_amount(text: str, field: str) -> float | None:
    """The number in the reply's JSON field; None when there is no usable number."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
    try:
        value = json.loads(cleaned).get(field)
    except (json.JSONDecodeError, AttributeError):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:[.,]\d+)?", value)
        return float(match.group().replace(",", ".")) if match else None
    return None


def quota_info(err: errors.APIError) -> dict:
    """The quota violation and retry delay inside a 429, if the API sent them."""
    info: dict = {}
    body = err.details.get("error", err.details) if isinstance(err.details, dict) else {}
    for d in body.get("details", []) or []:
        kind = d.get("@type", "")
        if kind.endswith("QuotaFailure"):
            info["violations"] = [
                {k: v.get(k) for k in ("quotaMetric", "quotaId", "quotaValue")}
                for v in d.get("violations", [])
            ]
        elif kind.endswith("RetryInfo"):
            delay = d.get("retryDelay", "")
            if delay.endswith("s"):
                info["retry_s"] = float(delay[:-1])
    return info


def _config(cond: Condition, field: str) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_json_schema=prompts.response_schema(field),
        thinking_config=types.ThinkingConfig(
            thinking_level=cond.thinking_level, include_thoughts=cond.include_thoughts
        ),
    )


def ask(client: genai.Client, cond: Condition, role: str, stratum: int, replicate: int,
        value: float, max_tries: int = 8, log=print) -> dict:
    """Call the model once (with retries on rate limits and server errors) and return the
    cache record."""
    text, field = prompts.render(cond.prompt_version, role, value)
    config = _config(cond, field)
    for attempt in range(1, max_tries + 1):
        t0 = time.monotonic()
        try:
            resp = client.models.generate_content(model=cond.model, contents=text, config=config)
            break
        except errors.APIError as err:
            if err.code == 429:
                info = quota_info(err)
                ids = " ".join(v.get("quotaId") or "" for v in info.get("violations", []))
                if "PerDay" in ids:
                    raise QuotaExhausted(f"{cond.model}: {info}") from err
                wait = info.get("retry_s", min(60.0, 2.0**attempt))
                log(f"  429 on {cond.model} ({ids or 'no quota id'}); waiting {wait:.0f}s")
            elif err.code >= 500:
                wait = min(60.0, 2.0**attempt)
                log(f"  {err.code} on {cond.model}; waiting {wait:.0f}s")
            else:
                raise
            if attempt == max_tries:
                raise
            time.sleep(wait)
    latency = time.monotonic() - t0

    cand = resp.candidates[0] if resp.candidates else None
    parts = (cand.content.parts if cand and cand.content else None) or []
    answer = "".join(p.text for p in parts if p.text and not p.thought)
    thoughts = "".join(p.text for p in parts if p.text and p.thought)
    meta = resp.usage_metadata
    usage = meta.model_dump(mode="json", exclude_none=True) if meta else {}
    amount = parse_amount(answer, field)
    return {
        "key": call_key(cond.model, cond.prompt_version, cond.thinking_level, role, value,
                        replicate),
        "condition": cond.name,
        "model": cond.model,
        "model_version": resp.model_version,
        "prompt_version": cond.prompt_version,
        "prompt_hash": prompts.template_hash(cond.prompt_version, role),
        "thinking_level": cond.thinking_level,
        "role": role,
        "stratum": stratum,
        "replicate": replicate,
        "value": value,
        "field": field,
        "raw_text": answer,
        "thoughts": thoughts,
        "amount": amount,
        "valid": amount is not None,
        "finish_reason": str(cand.finish_reason) if cand else None,
        "usage": usage,
        "cost_usd": call_cost(cond.model, usage),
        "latency_s": round(latency, 3),
        "attempts": attempt,
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
    }
