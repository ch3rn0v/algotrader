"""Shared data contracts (SPEC 4.1, 4.3, 6.3, 8.2).

This project is strictly functional: no business-logic classes. The only
classes here are a ``NamedTuple`` and ``TypedDict`` schemas, all library bases.
State flows through plain dicts and these typed views over them.
"""
from __future__ import annotations

from datetime import datetime
from typing import NamedTuple, Optional, TypedDict


class Instrument(NamedTuple):
    """Instrument reference passed wherever an instrument is identified."""

    exchange: str
    symbol: str
    figi: str
    currency: str
    lot_size: int  # number of base units (shares) per lot


class Candle(TypedDict):
    """A normalized candle. ``timestamp`` is bar-open in UTC (SPEC 4.1)."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


# Canonical observation-time columns (known at candle close).
OBSERVATION_COLUMNS: tuple[str, ...] = (
    "exchange",
    "instrument",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "signal",
    "target_position",
    "triggering_signal",
)

# Canonical execution-time columns (caused by the *previous* bar's signal).
EXECUTION_COLUMNS: tuple[str, ...] = (
    "trade_qty_from_prev",
    "fill_price_from_prev",
    "fees_from_prev",
    "position",
    "cash",
    "equity",
)

# Full canonical row schema (SPEC 4.1), in order.
CANONICAL_COLUMNS: tuple[str, ...] = OBSERVATION_COLUMNS + EXECUTION_COLUMNS


class StrategyState(TypedDict):
    """Running state of the signal state machine (SPEC 6.3)."""

    last_signal: int  # -1 / 0 / +1
    gate: str  # "open" | "blocked"
    bars_since_exit: int
    crossed_mid_since_exit: bool
    prev_close: Optional[float]  # previous in-session close, for the mid-cross test


class BotState(TypedDict):
    """Live bot's own intended state, the source of truth for decisions (SPEC 8.2)."""

    intended_position: float
    cash: float
    last_signal: int
    gate: str
    bars_since_exit: int
    crossed_mid_since_exit: bool
    prev_close: Optional[float]
    pending_order_id: Optional[str]
