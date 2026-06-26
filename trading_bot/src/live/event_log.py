"""Structured JSONL event log (SPEC 9). The disk side effect lives here.

One JSON object per line in ``data/logs/<run_id>.jsonl``; append-only, never
edited. This log is the source of truth for live reporting and comparison, so
any field comparison or post-hoc debugging needs must be logged when it happens.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

_LOG_DIR = Path("data/logs")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _to_iso(ts) -> str:
    if isinstance(ts, str):
        return ts
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def make_logger(run_id: str, instrument, log_dir: Path = _LOG_DIR) -> tuple[Callable[[str, dict], dict], Path]:
    """Return ``(log, path)``. ``log(event_type, fields)`` appends one line and
    returns the written event. Common fields (SPEC 9.1) are added automatically."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{run_id}.jsonl"

    def log(event_type: str, fields: dict) -> dict:
        event = {
            "ts": _utc_now_iso(),
            "exchange": instrument.exchange,
            "instrument": instrument.symbol,
            "type": event_type,
            "run_id": run_id,
            **fields,
        }
        if "bar_ts" in event and event["bar_ts"] is not None:
            event["bar_ts"] = _to_iso(event["bar_ts"])
        with open(path, "a") as f:
            f.write(json.dumps(event) + "\n")
        return event

    return log, path
