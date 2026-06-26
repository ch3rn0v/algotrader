"""Signal generation: a deterministic state machine (SPEC 6.3, 6.4).

Trend-filtered mean reversion. A slow/fast EWMA cross is the regime filter;
Bollinger-Band touches are entry triggers. Reversion trades are taken only in
the direction the regime allows. A re-entry gate enforces a cooldown plus a
``bb_mid`` re-cross after each exit.

Everything here is pure. The same ``step`` function drives both the backtest
engine (vectorized scan) and the live runner (one call per closed bar), which
guarantees live/backtest parity.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.core.types import StrategyState

REGIMES = ("up", "down")
BANDS = ("at_lower", "at_upper", "inside")
GATES = ("open", "blocked")


def _build_transition_table() -> dict[tuple[int, str, str, str], int]:
    """The complete (prev_signal, regime, band, gate) -> new_signal table.

    This dict literally transcribes SPEC 6.3 with the "any" rows expanded over
    both gate values. Every one of the 3*2*3*2 = 36 combinations has an entry,
    so ``generate_signals`` never falls through to an implicit "hold previous".
    """
    table: dict[tuple[int, str, str, str], int] = {}
    for gate in GATES:
        entry = gate == "open"
        # prev_signal == 0: gated entries only; never fade the trend.
        table[(0, "up", "at_lower", gate)] = 1 if entry else 0
        table[(0, "up", "at_upper", gate)] = 0
        table[(0, "up", "inside", gate)] = 0
        table[(0, "down", "at_upper", gate)] = -1 if entry else 0
        table[(0, "down", "at_lower", gate)] = 0
        table[(0, "down", "inside", gate)] = 0
        # prev_signal == +1 (long): hold in uptrend; exit on target or flip.
        table[(1, "up", "at_upper", gate)] = 0
        table[(1, "up", "at_lower", gate)] = 1
        table[(1, "up", "inside", gate)] = 1
        table[(1, "down", "at_lower", gate)] = 0
        table[(1, "down", "at_upper", gate)] = 0
        table[(1, "down", "inside", gate)] = 0
        # prev_signal == -1 (short): mirror image.
        table[(-1, "down", "at_lower", gate)] = 0
        table[(-1, "down", "at_upper", gate)] = -1
        table[(-1, "down", "inside", gate)] = -1
        table[(-1, "up", "at_upper", gate)] = 0
        table[(-1, "up", "at_lower", gate)] = 0
        table[(-1, "up", "inside", gate)] = 0
    return table


TRANSITION_TABLE = _build_transition_table()


def initial_state() -> StrategyState:
    """Fresh strategy state: flat, gate open, no exit history."""
    return {
        "last_signal": 0,
        "gate": "open",
        "bars_since_exit": 0,
        "crossed_mid_since_exit": False,
        "prev_close": None,
    }


def classify_regime(fast_ewm: float, slow_ewm: float) -> str:
    """``up`` iff fast EWMA strictly exceeds slow EWMA; ties map to ``down``.

    NaN (warmup) comparisons are False, so warmup bars classify as ``down`` and
    produce no entry (band is also NaN -> ``inside``)."""
    return "up" if fast_ewm > slow_ewm else "down"


def classify_band(close: float, bb_lo: float, bb_up: float) -> str:
    """``at_lower`` / ``at_upper`` / ``inside`` relative to the lagged bands."""
    if close <= bb_lo:
        return "at_lower"
    if close >= bb_up:
        return "at_upper"
    return "inside"


def _advance_gate(
    state: StrategyState,
    prev_signal: int,
    new_signal: int,
    close: float,
    bb_mid: float,
    cooldown_bars: int,
) -> tuple[str, int, bool]:
    """Return the updated (gate, bars_since_exit, crossed_mid) after a bar.

    Run *after* the signal is computed. An exit (non-zero -> 0) blocks the gate
    and resets counters. While blocked, each subsequent bar increments the bar
    counter and tests for a ``bb_mid`` re-cross; the gate re-opens only when both
    conditions hold. The mid-cross uses the same lagged ``bb_mid`` as the entry
    trigger (SPEC 6.3)."""
    gate = state["gate"]
    bars_since_exit = state["bars_since_exit"]
    crossed = state["crossed_mid_since_exit"]
    prev_close = state["prev_close"]

    exited = prev_signal != 0 and new_signal == 0
    if exited:
        return "blocked", 0, False

    if gate == "blocked":
        bars_since_exit += 1
        if prev_close is not None and not np.isnan(bb_mid):
            if (prev_close - bb_mid) * (close - bb_mid) < 0:
                crossed = True
        if bars_since_exit >= cooldown_bars and crossed:
            gate = "open"
    return gate, bars_since_exit, crossed


def step(
    state: StrategyState,
    *,
    close: float,
    fast_ewm: float,
    slow_ewm: float,
    bb_lo: float,
    bb_up: float,
    bb_mid: float,
    in_session: bool,
    params: dict,
) -> tuple[int, float, Optional[str], Optional[str], StrategyState]:
    """Compute (signal, target_position, regime, band, new_state) for one bar.

    Out-of-session bars produce ``signal = 0`` and ``target_position = 0`` and
    leave the strategy state untouched, so the cooldown counts trading bars only
    and the held position carries across the overnight gap. This mirrors the live
    runner, which skips signal computation entirely outside the session (SPEC
    8.3 step 2)."""
    if not in_session:
        return 0, 0.0, None, None, state

    regime = classify_regime(fast_ewm, slow_ewm)
    band = classify_band(close, bb_lo, bb_up)
    prev_signal = state["last_signal"]
    new_signal = TRANSITION_TABLE[(prev_signal, regime, band, state["gate"])]
    target_position = float(new_signal * params["max_position_lots"])

    gate, bars_since_exit, crossed = _advance_gate(
        state, prev_signal, new_signal, close, bb_mid, params["entry_cooldown_bars"]
    )
    new_state: StrategyState = {
        "last_signal": new_signal,
        "gate": gate,
        "bars_since_exit": bars_since_exit,
        "crossed_mid_since_exit": crossed,
        "prev_close": close,
    }
    return new_signal, target_position, regime, band, new_state


def generate_signals(
    indicators_df: pd.DataFrame,
    params: dict,
    in_session: "pd.Series | np.ndarray",
    init_state: Optional[StrategyState] = None,
) -> pd.DataFrame:
    """Scan ``indicators_df`` row by row, appending ``signal`` and
    ``target_position`` columns.

    Sequential by nature (the gate is path-dependent), but reads only
    observation-time data: no future leakage. ``params`` must include the
    strategy parameters and ``max_position_lots`` (from the risk config)."""
    close = indicators_df["close"].to_numpy()
    fast = indicators_df["fast_ewm"].to_numpy()
    slow = indicators_df["slow_ewm"].to_numpy()
    bb_lo = indicators_df["bb_lo"].to_numpy()
    bb_up = indicators_df["bb_up"].to_numpy()
    bb_mid = indicators_df["bb_mid"].to_numpy()
    sess = np.asarray(in_session, dtype=bool)

    n = len(indicators_df)
    signal_out = np.zeros(n, dtype=int)
    target_out = np.zeros(n, dtype=float)
    state = init_state if init_state is not None else initial_state()

    for i in range(n):
        signal_out[i], target_out[i], _, _, state = step(
            state,
            close=close[i],
            fast_ewm=fast[i],
            slow_ewm=slow[i],
            bb_lo=bb_lo[i],
            bb_up=bb_up[i],
            bb_mid=bb_mid[i],
            in_session=bool(sess[i]),
            params=params,
        )

    out = indicators_df.copy()
    out["signal"] = signal_out
    out["target_position"] = target_out
    return out
