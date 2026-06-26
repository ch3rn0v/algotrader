"""Pure functions over the bot's own ``BotState`` (SPEC 8.2).

The bot maintains its own intended position as the source of truth for execution
decisions; the broker is consulted as a sanity check, never trusted to drive
decisions. These functions are pure: the runner persists state via the event log.
"""
from __future__ import annotations

from typing import Optional

from src.core.types import BotState


def initial_state(initial_cash: float) -> BotState:
    return {
        "intended_position": 0.0,
        "cash": float(initial_cash),
        "last_signal": 0,
        "gate": "open",
        "bars_since_exit": 0,
        "crossed_mid_since_exit": False,
        "prev_close": None,
        "pending_order_id": None,
    }


def expected_broker_position(state: BotState, qty_remaining_in_flight: float) -> float:
    """Broker hasn't yet seen the not-yet-filled portion of in-flight orders
    (SPEC 8.2)."""
    return state["intended_position"] - qty_remaining_in_flight


def is_discrepant(state: BotState, broker_position: float, qty_remaining_in_flight: float, eps: float = 1e-9) -> bool:
    return abs(broker_position - expected_broker_position(state, qty_remaining_in_flight)) > eps


def apply_fill(state: BotState, filled_qty: float, fill_price: float, fee: float, lot_size: int) -> BotState:
    """Update intended position and cash after a fill."""
    return {
        **state,
        "intended_position": state["intended_position"] + filled_qty,
        "cash": state["cash"] - filled_qty * fill_price * lot_size - fee,
    }


def mark_equity(state: BotState, close: float) -> float:
    """Mark-to-market equity at a bar close (SPEC 8.5)."""
    return state["cash"] + state["intended_position"] * close


def merge_strategy_state(state: BotState, signal: int, strat_state: dict, pending_order_id: Optional[str]) -> BotState:
    """Fold the strategy state machine's output back into the bot state."""
    return {
        **state,
        "last_signal": signal,
        "gate": strat_state["gate"],
        "bars_since_exit": strat_state["bars_since_exit"],
        "crossed_mid_since_exit": strat_state["crossed_mid_since_exit"],
        "prev_close": strat_state["prev_close"],
        "pending_order_id": pending_order_id,
    }


def to_strategy_state(state: BotState) -> dict:
    """Project the strategy-machine sub-state out of the bot state."""
    return {
        "last_signal": state["last_signal"],
        "gate": state["gate"],
        "bars_since_exit": state["bars_since_exit"],
        "crossed_mid_since_exit": state["crossed_mid_since_exit"],
        "prev_close": state["prev_close"],
    }
