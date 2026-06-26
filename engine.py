"""Backtest engine (SPEC 10.2). Produces the canonical DataFrame (SPEC 4.1).

Pipeline:
1. Vectorized prep: indicators, session mask, per-row causal ``sigma_daily``.
2. Vectorized signal: ``generate_signals`` (sequential scan, observation-only).
3. Sequential execution: a fill on bar *t* is caused by the target decided on
   the previous in-session bar, simulated against bar *t*'s 1-min sub-candles.
4. Carry ``position`` and ``cash`` forward; mark ``equity`` at each bar's close.

The row loop is not a v1 bottleneck (tens of thousands of rows/year) and keeps
the lookahead boundary visible. ``triggering_signal`` is written as
``signal.shift(1)`` to match the schema and reconstruction exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.indicators import compute_indicators
from src.core.session import is_main_session
from src.core.strategy import generate_signals
from src.core.types import CANONICAL_COLUMNS, Instrument
from src.backtest.simulator import compute_sigma_daily, simulate_fill


def _attach_sigma_daily(df15: pd.DataFrame, candles_1day: pd.DataFrame, window_days: int) -> pd.Series:
    """Map a strictly-prior daily sigma onto each 15-min bar via backward
    ``merge_asof`` with ``allow_exact_matches=False`` (no same-day leakage)."""
    sigma = compute_sigma_daily(candles_1day, window_days).rename(columns={"date": "sig_date"})
    bars = pd.DataFrame({"timestamp": df15["timestamp"].to_numpy()})
    bars["bar_date"] = pd.to_datetime(bars["timestamp"], utc=True).dt.normalize()
    bars = bars.sort_values("bar_date")
    merged = pd.merge_asof(
        bars,
        sigma.sort_values("sig_date"),
        left_on="bar_date",
        right_on="sig_date",
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.sort_index()  # restore original row order
    return merged["sigma_daily"].to_numpy()


def _group_subcandles(candles_1m: pd.DataFrame) -> dict:
    """Group 1-min candles by their parent 15-min bar-open timestamp."""
    m = candles_1m.sort_values("timestamp").copy()
    m["bar"] = pd.to_datetime(m["timestamp"], utc=True).dt.floor("15min")
    return {bar: g.reset_index(drop=True) for bar, g in m.groupby("bar")}


def run_backtest(
    candles_15m: pd.DataFrame,
    candles_1m: pd.DataFrame,
    candles_1day: pd.DataFrame,
    instrument: Instrument,
    config: dict,
) -> pd.DataFrame:
    """Replay ``candles_15m`` and return the canonical DataFrame."""
    strat = config["strategy"]
    risk = config["risk"]
    bt = config["backtest"]
    params = {**strat, "max_position_lots": risk["max_position_lots"]}

    df = candles_15m.sort_values("timestamp").reset_index(drop=True).copy()
    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = timestamps

    # 1. Vectorized prep.
    df = compute_indicators(df, params)
    in_session = timestamps.dt.tz_convert("UTC").map(is_main_session).to_numpy()
    sigma_daily = _attach_sigma_daily(df, candles_1day, bt["sigma_daily_window_days"])
    df["sigma_daily"] = sigma_daily

    # 2. Vectorized signal (forced flat out of session inside the scan).
    df = generate_signals(df, params, in_session)

    warmup_bars = max(strat["slow_ewm_span"], strat["bb_period"])
    warmup_invalid = np.arange(len(df)) < warmup_bars
    warmup_invalid = warmup_invalid | np.isnan(sigma_daily)
    df["warmup_invalid"] = warmup_invalid

    # 3 + 4. Sequential execution and accounting.
    subcandles = _group_subcandles(candles_1m) if len(candles_1m) else {}
    lot = instrument.lot_size
    n = len(df)
    close = df["close"].to_numpy()
    signal = df["signal"].to_numpy()
    target = df["target_position"].to_numpy()
    bar_open = df["timestamp"].to_numpy()

    trade_qty = np.zeros(n)
    fill_price = np.full(n, np.nan)
    fees = np.zeros(n)
    position_arr = np.zeros(n)
    cash_arr = np.zeros(n)
    equity_arr = np.zeros(n)

    position = 0.0
    cash = float(bt["initial_cash"])
    # The target decided on the previous in-session bar; carries across the gap.
    pending_target = 0.0

    for i in range(n):
        if in_session[i]:
            if pending_target != position:
                window = subcandles.get(pd.Timestamp(bar_open[i]))
                if window is not None:
                    delta = pending_target - position
                    filled, price, fee = simulate_fill(delta, window, sigma_daily[i], bt, lot)
                    if filled != 0.0:
                        trade_qty[i] = filled
                        fill_price[i] = price
                        fees[i] = fee
                        position += filled
                        cash -= filled * price * lot + fee
            pending_target = float(target[i])
        position_arr[i] = position
        cash_arr[i] = cash
        equity_arr[i] = cash + position * close[i]

    df["trade_qty_from_prev"] = trade_qty
    df["fill_price_from_prev"] = fill_price
    df["fees_from_prev"] = fees
    df["position"] = position_arr
    df["cash"] = cash_arr
    df["equity"] = equity_arr
    df["triggering_signal"] = df["signal"].shift(1).fillna(0).astype(int)

    extra = ["sigma_daily", "warmup_invalid"]
    return df[[*CANONICAL_COLUMNS, *extra]].reset_index(drop=True)
