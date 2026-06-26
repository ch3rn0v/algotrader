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
    position_size: float = 1.0,
    initial_cash: float = 100_000.0,
) -> dict:
    df = candles.copy().reset_index(drop=True)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    df["fast"] = df["close"].ewm(span=fast_span, adjust=False).mean()
    df["slow"] = df["close"].ewm(span=slow_span, adjust=False).mean()

    # +position_size when fast > slow, -position_size otherwise; flatten EOD
    is_last_bar_of_day = df["date"] != df["date"].shift(-1)
    df["desired"] = np.where(df["fast"] > df["slow"], position_size, -position_size)
    df.loc[is_last_bar_of_day, "desired"] = 0.0

    position = 0.0
    cash = float(initial_cash)
    equity_rows = []
    trade_rows = []

    for i in range(len(df)):
        row = df.iloc[i]

        # Execute prior bar's desired position at this bar's open
        if i > 0:
            delta = df.at[i - 1, "desired"] - position
            if abs(delta) > 1e-9:
                cash -= delta * row["open"]
                position += delta
                trade_rows.append({
                    "timestamp": row["timestamp"],
                    "qty": delta,
                    "price": row["open"],
                })

        equity_rows.append({
            "timestamp": row["timestamp"],
            "equity": cash + position * row["close"],
            "position": position,
        })

    return {
        "equity": pd.DataFrame(equity_rows),
        "trades": pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(columns=["timestamp", "qty", "price"]),
    }
