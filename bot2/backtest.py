"""Bollinger Band mean reversion backtest.

Strategy:
  - Active only during a configurable intraday session (default 12:00–15:00 MSK = 09:00–12:00 UTC).
  - Enter long when close < lower band; enter short when close > upper band.
  - Only enter when the market is ranging: BB width is below its own rolling median.
  - Exit at the middle band (mean reversion target) or after `time_stop_bars` (time stop).
  - Flatten at the last bar of each day regardless.
  - Signals execute at the *next* bar's open.
"""
import numpy as np
import pandas as pd


def run_backtest(
    candles: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    time_stop_bars: int = 12,
    width_lookback: int = 50,
    session_start_utc: int = 9,
    session_end_utc: int = 12,
    position_size: int = 1,
    initial_cash: float = 100_000.0,
) -> dict:
    df = candles.copy().reset_index(drop=True)

    mid   = df["close"].rolling(bb_period).mean()
    std   = df["close"].rolling(bb_period).std()
    upper = mid + bb_std * std
    lower = mid - bb_std * std
    width = (upper - lower) / mid

    # Ranging filter: only trade when BB width is below its own recent median
    ranging = width < width.rolling(width_lookback).median()

    ts_dt      = pd.to_datetime(df["timestamp"])
    hour       = ts_dt.dt.hour
    in_session = (hour >= session_start_utc) & (hour < session_end_utc)

    date         = ts_dt.dt.date.to_numpy()
    is_last_bar  = np.empty(len(df), dtype=bool)
    is_last_bar[:-1] = date[:-1] != date[1:]
    is_last_bar[-1]  = True

    closes     = df["close"].to_numpy()
    opens      = df["open"].to_numpy()
    timestamps = df["timestamp"].to_numpy()
    mid_a      = mid.to_numpy()
    upper_a    = upper.to_numpy()
    lower_a    = lower.to_numpy()
    ranging_a  = ranging.to_numpy()
    session_a  = in_session.to_numpy()

    position     = 0
    cash         = float(initial_cash)
    peak_exposure  = 0.0
    desired      = 0
    entry_bar    = 0
    equity_rows  = []
    trade_rows   = []

    for i in range(len(df)):
        # Execute prior bar's desired at this bar's open
        if i > 0:
            delta = desired - position
            if delta != 0:
                cash    -= delta * opens[i]
                position += delta
                if position != 0:
                    entry_bar = i
                peak_exposure = max(peak_exposure, abs(position) * opens[i])
                trade_rows.append({"timestamp": timestamps[i], "qty": delta, "price": opens[i]})

        equity_rows.append({
            "timestamp": timestamps[i],
            "equity":    cash + position * closes[i],
            "position":  position,
        })

        # Compute desired for next bar
        if np.isnan(mid_a[i]) or is_last_bar[i]:
            desired = 0
        elif position != 0:
            bars_held       = i - entry_bar
            hit_midband     = (position > 0 and closes[i] >= mid_a[i]) or \
                              (position < 0 and closes[i] <= mid_a[i])
            if hit_midband or bars_held >= time_stop_bars:
                desired = 0
            else:
                desired = position  # hold
        elif session_a[i] and ranging_a[i]:
            if closes[i] < lower_a[i]:
                desired = position_size
            elif closes[i] > upper_a[i]:
                desired = -position_size
            else:
                desired = 0
        else:
            desired = 0

    return {
        "equity":       pd.DataFrame(equity_rows),
        "trades":       pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(columns=["timestamp", "qty", "price"]),
        "peak_exposure":  peak_exposure,
    }
