"""EWMA crossover backtest.

Strategy:
  - Go long `position_size` lots when fast EWMA crosses above slow EWMA.
  - Go short `position_size` lots when fast crosses below.
  - Flatten at the last bar of each trading day.
  - Signals execute at the *next* bar's open.

Usage:
    from backtest import run_backtest

    results = run_backtest(candles, fast_span=20, slow_span=50)
    print(results["trades"])
    results["equity"]["equity"].plot()
"""
import numpy as np
import pandas as pd


def run_backtest(
    candles: pd.DataFrame,
    fast_span: int = 20,
    slow_span: int = 50,
    position_size: int = 1,
    initial_cash: float = 100_000.0,
) -> dict:
    df = candles.copy().reset_index(drop=True)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["fast"] = df["close"].ewm(span=fast_span, adjust=False).mean()
    df["slow"] = df["close"].ewm(span=slow_span, adjust=False).mean()

    is_last_bar_of_day = df["date"] != df["date"].shift(-1)
    df["desired"] = np.where(df["fast"] > df["slow"], position_size, -position_size)
    df.loc[is_last_bar_of_day, "desired"] = 0

    desired   = df["desired"].to_numpy(dtype=int)
    opens     = df["open"].to_numpy()
    closes    = df["close"].to_numpy()
    timestamps = df["timestamp"].to_numpy()

    position = 0
    cash = float(initial_cash)
    equity_rows = []
    trade_rows = []

    for i in range(len(df)):
        if i > 0:
            delta = desired[i - 1] - position
            if delta != 0:
                cash -= delta * opens[i]
                position += delta
                trade_rows.append({
                    "timestamp": timestamps[i],
                    "qty": delta,
                    "price": opens[i],
                })

        equity_rows.append({
            "timestamp": timestamps[i],
            "equity": cash + position * closes[i],
            "position": position,
        })

    return {
        "equity": pd.DataFrame(equity_rows),
        "trades": pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(columns=["timestamp", "qty", "price"]),
    }
