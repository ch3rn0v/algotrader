"""EWM-trend mean reversion backtest.

Strategy:
  - Trend = EWM of close with `trend_alpha`.
  - Trade only in a "quiet" market, where mean reversion is more probable:
      * the trend is not too obvious: |trend % change over `slope_window` bars|
        is below `max_slope_pct`;
      * traded volume is low: EWM of volume is below `vol_ratio_max` x its
        rolling mean over `vol_window` bars;
      * both conditions have held for at least `min_quiet_bars` consecutive bars.
  - Enter when price has strayed too far from the trend: long when close is
    `entry_pct`% below the trend, short when `entry_pct`% above it.
  - Exit near the trend (within `exit_pct`%), after `time_stop_bars`, or at
    session end. Flatten at the last bar of each day regardless.
  - Active only during a configurable intraday session (default 12:00-19:00 MSK
    = 09:00-16:00 UTC).
  - Decision uses the *previous* bar's completed data; execution is at the
    *current* bar's open.

headless=True returns a flat stats dict instead of equity/trades DataFrames.
Use this for grid search / optimisation loops.
"""

import numpy as np
import pandas as pd


def run_backtest(
    candles: pd.DataFrame,
    trend_alpha: float = 0.02,
    entry_pct: float = 0.3,
    exit_pct: float = 0.05,
    max_slope_pct: float = 0.05,
    slope_window: int = 15,
    vol_alpha: float = 0.05,
    vol_window: int = 240,
    vol_ratio_max: float = 0.9,
    min_quiet_bars: int = 10,
    time_stop_bars: int = 60,
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
    # When None, the plain deviation signal is used with no model filter.
    predictions: np.ndarray | None = None,
    pred_long_threshold: float = 1.002,
    pred_short_threshold: float = 0.998,
) -> dict:
    df = candles.reset_index(drop=True)
    n = len(df)

    trend = df["close"].ewm(alpha=trend_alpha).mean()
    dev_pct = (df["close"] / trend - 1.0) * 100.0

    # Quiet-market filters.
    slope_pct = (trend / trend.shift(slope_window) - 1.0).abs() * 100.0
    flat_trend = slope_pct < max_slope_pct
    # Mean vs mean: a median baseline sits far below the EWM on skewed 1min
    # volumes, making "low volume" unreachable.
    vol_fast = df["volume"].ewm(alpha=vol_alpha).mean()
    vol_base = df["volume"].rolling(vol_window).mean()
    low_vol = vol_fast < vol_ratio_max * vol_base
    quiet = (flat_trend & low_vol).to_numpy()

    # quiet_run[i] = number of consecutive quiet bars ending at i.
    q = quiet.astype(np.int64)
    cs = np.cumsum(q)
    quiet_run = cs - np.maximum.accumulate(np.where(q == 0, cs, 0))

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
    dev_a = dev_pct.to_numpy()

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
            if np.isnan(dev_a[p]) or is_last_bar[p] or is_session_end[p]:
                desired = 0
            elif position != 0:
                bars_held = i - entry_bar
                near_trend = (position > 0 and dev_a[p] >= -exit_pct) or (position < 0 and dev_a[p] <= exit_pct)
                desired = 0 if near_trend or bars_held >= time_stop_bars else position
            elif session_a[p] and quiet_run[p] >= min_quiet_bars:
                if predictions is not None:
                    pred = float(predictions[i])
                    want_long = dev_a[p] <= -entry_pct and pred > pred_long_threshold
                    want_short = dev_a[p] >= entry_pct and pred < pred_short_threshold
                else:
                    want_long = dev_a[p] <= -entry_pct
                    want_short = dev_a[p] >= entry_pct
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
    pct_bars_in_pos = total_bars_held / n * 100 if n > 0 else 0.0
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
        "pct_bars_in_pos": round(pct_bars_in_pos, 2),
        "turnover": round(turnover, 2),
        "total_fees": round(total_fees, 2),
        "peak_exposure": round(peak_exposure, 2),
    }
