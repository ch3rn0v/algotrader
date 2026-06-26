"""Signal -> order translation and idempotent order ids (SPEC 8.4).

Every market order carries a deterministic ``client_order_id`` combining the
decision identity with a retry-attempt counter:

    client_order_id = sha1(instrument || signal_timestamp || target_position || retry_attempt)[:16]

Transient failures retry the *same* ``retry_attempt`` (identical id -> T-Bank
deduplicates). Hard rejections increment ``retry_attempt`` (fresh id -> avoids
being silently deduplicated against the rejected attempt). At most 2 retries.
"""
from __future__ import annotations

import hashlib

from src.core.types import Instrument

MAX_RETRIES = 2  # 3 attempts total: retry_attempt in {0, 1, 2}


def client_order_id(instrument: Instrument, signal_timestamp, target_position: float, retry_attempt: int) -> str:
    """Deterministic 16-hex-char order id for a run-process order."""
    payload = f"{instrument.figi}|{_iso(signal_timestamp)}|{target_position}|{retry_attempt}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def flat_client_order_id(instrument: Instrument, flat_run_id: str, retry_attempt: int) -> str:
    """Order id for the ``flat`` subcommand. Intentionally distinct from the run
    derivation (no signal_timestamp) so it never collides with an in-progress
    ``run`` (SPEC 13)."""
    payload = f"flat|{instrument.figi}|{flat_run_id}|{retry_attempt}"
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def order_delta(target_position: float, intended_position: float) -> float:
    """Signed quantity (lots) needed to move from intended to target."""
    return target_position - intended_position


def _iso(ts) -> str:
    if isinstance(ts, str):
        return ts
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")
