"""Bollinger Band mean reversion backtest.

Strategy:
  - Active only during a configurable intraday session (default 12:00–15:00 MSK = 09:00–12:00 UTC).
  - Enter long when close < lower band; enter short when close > upper band.
  - Only enter when the market is ranging: BB width is below its own rolling median.
  - Exit at the middle band (mean reversion target), after `time_stop_bars`, or at session end.
  - Flatten at the last bar of each day regardless.
  - Decision uses the *previous* bar's completed data; execution is at the *current* bar's open.

headless=True returns a flat stats dict instead of equity/trades DataFrames.
Use this for grid search / optimisation loops.
"""

import numpy as np
import pandas as pd


def run_backtest(
    candles: pd.DataFrame,
    bb_alpha: float = 0.1,
    bb_std: float = 1.5,
    time_stop_bars: int = 16,
    width_alpha: float = 0.2,
    session_start_utc: int = 9,
    session_end_utc: int = 16,
    position_size: int = 1,
    initial_cash: float = 100_000.0,
    headless: bool = False,
    # Market-taker fee per side (applied to each fill's notional).
    # T-Bank Trader tariff: 0.05%.
    # Source: https://www.tbank.ru/invest/tariffs/; verify before live use.
    fee_rate: float = 0.0005,
    # Optional array of model predictions aligned with candles.
    # predictions[i] is the predicted close[i]/close[i-1] for bar i.
    # When provided, only enter long if pred > pred_long_threshold and only enter short if pred < pred_short_threshold.
    # When None, the plain BB signal is used with no model filter.
    predictions: np.ndarray | None = None,
    pred_long_threshold: float = 1.002,
    pred_short_threshold: float = 0.998,
) -> dict:
    df = candles.reset_index(drop=True)
    n = len(df)

    mid = df["close"].ewm(alpha=bb_alpha).mean()
    std = df["close"].ewm(alpha=bb_alpha).std()
    upper = mid + bb_std * std
    lower = mid - bb_std * std
    width = (upper - lower) / mid

    trending = width > width.ewm(alpha=width_alpha).mean()
    ts_dt = pd.to_datetime(df["timestamp"])
    in_session = (ts_dt.dt.hour >= session_start_utc) & (ts_dt.dt.hour < session_end_utc)

    date = ts_dt.dt.date.to_numpy()
    is_last_bar = np.empty(n, dtype=bool)
    is_last_bar[:-1] = date[:-1] != date[1:]
    is_last_bar[-1] = True

    session_a = in_session.to_numpy()

    # Last in-session bar: session is True now and False on the next bar (or last bar overall).
    is_session_end = np.empty(n, dtype=bool)
    is_session_end[:-1] = session_a[:-1] & ~session_a[1:]
    is_session_end[-1] = False

    closes = df["close"].to_numpy()
    opens = df["open"].to_numpy()
    timestamps = df["timestamp"].to_numpy()
    mid_a = mid.to_numpy()
    upper_a = upper.to_numpy()
    lower_a = lower.to_numpy()
    trending_a = trending.to_numpy()

    # --- Simulate ---
    position = 0
    cash = float(initial_cash)
    desired = 0
    entry_bar = 0
    peak_exposure = 0.0
    total_fees = 0.0
    trade_rows = []
    equity = np.empty(n)
    pos_hist = np.empty(n, dtype=np.int64)

    for i in range(n):
        # Decide desired position using the previous bar's completed data.
        # (i == 0 has no previous bar; desired stays 0.)
        if i > 0:
            p = i - 1
            if np.isnan(mid_a[p]) or is_last_bar[p] or is_session_end[p]:
                desired = 0
            elif position != 0:
                bars_held = i - entry_bar
                hit_midband = (position > 0 and closes[p] >= mid_a[p]) or (position < 0 and closes[p] <= mid_a[p])
                desired = 0 if hit_midband or bars_held >= time_stop_bars else position
            elif session_a[p] and trending_a[p]:
                if predictions is not None:
                    pred = float(predictions[i])
                    want_long = closes[p] < lower_a[p] and pred > pred_long_threshold
                    want_short = closes[p] > upper_a[p] and pred < pred_short_threshold
                else:
                    want_long = closes[p] < lower_a[p]
                    want_short = closes[p] > upper_a[p]
                desired = position_size if want_long else -position_size if want_short else 0
            else:
                desired = 0

        # Execute at this bar's open.
        delta = desired - position
        if delta != 0:
            notional = abs(delta) * opens[i]
            fee = notional * fee_rate
            total_fees += fee
            cash -= delta * opens[i] + fee
            position += delta
            if position != 0:
                entry_bar = i
            peak_exposure = max(peak_exposure, abs(position) * opens[i])
            trade_rows.append({"timestamp": timestamps[i], "qty": delta, "price": opens[i], "fee": fee})

        pos_hist[i] = position
        equity[i] = cash + position * closes[i]

    trades = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(columns=["timestamp", "qty", "price", "fee"])

    if not headless:
        return {
            "equity": pd.DataFrame({"timestamp": timestamps, "equity": equity, "position": pos_hist}),
            "trades": trades,
            "total_fees": total_fees,
            "peak_exposure": peak_exposure,
        }

    # --- Stats (headless) ---
    pnl = equity[-1] - initial_cash
    t0 = pd.to_datetime(timestamps[0], utc=True)
    t1 = pd.to_datetime(timestamps[-1], utc=True)
    n_days = max((t1 - t0).days, 1)

    n_entries = int(np.sum((pos_hist[1:] != 0) & (pos_hist[:-1] == 0)))
    total_bars_held = int(np.sum(pos_hist != 0))
    avg_bars_held = total_bars_held / n_entries if n_entries > 0 else 0.0
    turnover = float((trades["qty"].abs() * trades["price"]).sum())

    # Daily equity: seed with t=0 so the first day's return isn't dropped.
    daily_eq = np.concatenate(([initial_cash], equity[is_last_bar]))

    if peak_exposure > 0 and len(daily_eq) >= 2:
        daily_ret = np.diff(daily_eq) / peak_exposure
        mean_r = float(daily_ret.mean())
        std_r = float(daily_ret.std())
        sharpe = mean_r / std_r * np.sqrt(252) if std_r > 0 else 0.0
        neg = daily_ret[daily_ret < 0]
        s_std = float(neg.std()) if len(neg) > 1 else 0.0
        sortino = mean_r / s_std * np.sqrt(252) if s_std > 0 else 0.0
        base = 1 + pnl / peak_exposure
        cagr = base ** (365 / n_days) - 1 if base > 0 else -1.0
        running_max = np.maximum.accumulate(np.maximum(equity, initial_cash))
        max_dd = float(min((equity - running_max).min(), 0.0)) / peak_exposure
    else:
        sharpe = sortino = cagr = max_dd = 0.0

    return {
        "pnl": round(pnl, 2),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_dd": round(max_dd, 4),
        "cagr": round(cagr, 4),
        "n_trades": len(trades),
        "avg_bars_held": round(avg_bars_held, 2),
        "turnover": round(turnover, 2),
        "total_fees": round(total_fees, 2),
        "peak_exposure": round(peak_exposure, 2),
    }
