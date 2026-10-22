"""Append-only JSONL cache of every call: the raw reply, usage_metadata and what was parsed.

A call is identified by (model, prompt version, thinking level, role, value, replicate).
Re-running skips what is already stored, so an interrupted run resumes where it stopped and
the analysis never needs the API.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path


def call_key(model: str, prompt_version: str, thinking_level: str, role: str, value: float,
             replicate: int) -> str:
    return f"{model}|{prompt_version}|{thinking_level}|{role}|{value:.2f}|{replicate}"


class CallCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.records: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.records[rec["key"]] = rec

    def __contains__(self, key: str) -> bool:
        return key in self.records

    def __len__(self) -> int:
        return len(self.records)

    def get(self, key: str) -> dict | None:
        return self.records.get(key)

    def put(self, record: dict) -> None:
        with self._lock:
            if record["key"] in self.records:
                return
            with self.path.open("a") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()
            self.records[record["key"]] = record

    def total_cost(self) -> float:
        return sum(r.get("cost_usd", 0.0) for r in self.records.values())
