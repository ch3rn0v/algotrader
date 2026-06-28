"""Bollinger Band mean reversion backtest.

Strategy:
  - Active only during a configurable intraday session (default 12:00–15:00 MSK = 09:00–12:00 UTC).
  - Enter long when close < lower band; enter short when close > upper band.
  - Only enter when the market is ranging: BB width is below its own rolling median.
  - Exit at the middle band (mean reversion target) or after `time_stop_bars` (time stop).
  - Flatten at the last bar of each day regardless.
  - Signals execute at the *next* bar's open.

headless=True skips per-bar list accumulation; returns a flat stats dict instead of
equity/trades DataFrames. Use this for grid search / optimisation loops.
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
    headless: bool = False,
    # Estimated market-taker fee per side (applied to each fill's notional).
    # T-Bank Trader tariff: 0.04% broker + ~0.01% MOEX exchange taker fee = 0.05%.
    # Source: tbank.ru/invest/tariffs (as of 2025); verify before live use.
    fee_rate: float = 0.0005,
) -> dict:
    df = candles.copy().reset_index(drop=True)

    mid   = df["close"].rolling(bb_period).mean()
    std   = df["close"].rolling(bb_period).std()
    upper = mid + bb_std * std
    lower = mid - bb_std * std
    width = (upper - lower) / mid

    ranging    = width < width.rolling(width_lookback).median()
    ts_dt      = pd.to_datetime(df["timestamp"])
    in_session = (ts_dt.dt.hour >= session_start_utc) & (ts_dt.dt.hour < session_end_utc)

    date        = ts_dt.dt.date.to_numpy()
    is_last_bar = np.empty(len(df), dtype=bool)
    is_last_bar[:-1] = date[:-1] != date[1:]
    is_last_bar[-1]  = True

    closes    = df["close"].to_numpy()
    opens     = df["open"].to_numpy()
    timestamps = df["timestamp"].to_numpy()
    mid_a     = mid.to_numpy()
    upper_a   = upper.to_numpy()
    lower_a   = lower.to_numpy()
    ranging_a = ranging.to_numpy()
    session_a = in_session.to_numpy()

    position      = 0
    cash          = float(initial_cash)
    peak_exposure = 0.0
    desired       = 0
    entry_bar     = 0

    if headless:
        running_max_eq  = float(initial_cash)
        max_dd_abs      = 0.0
        n_trades        = 0
        n_entries       = 0
        total_bars_held = 0
        turnover        = 0.0
        total_fees      = 0.0
        daily_eq_vals   = []
        _prev_date      = None
        _last_eq        = float(initial_cash)
    else:
        equity_rows = []
        trade_rows  = []
        total_fees  = 0.0

    for i in range(len(df)):
        # Execute prior bar's signal at this bar's open
        if i > 0:
            delta = desired - position
            if delta != 0:
                prev_pos   = position
                notional   = abs(delta) * opens[i]
                fee        = notional * fee_rate
                cash      -= delta * opens[i] + fee
                total_fees += fee
                position  += delta
                if position != 0:
                    entry_bar = i
                peak_exposure = max(peak_exposure, abs(position) * opens[i])
                if headless:
                    if prev_pos == 0 and position != 0:
                        n_entries += 1
                    n_trades += 1
                    turnover += notional
                else:
                    trade_rows.append({"timestamp": timestamps[i], "qty": delta, "price": opens[i], "fee": fee})

        eq_val = cash + position * closes[i]

        if headless:
            bar_date = date[i]
            if bar_date != _prev_date:
                if _prev_date is not None:
                    daily_eq_vals.append(_last_eq)
                _prev_date = bar_date
            _last_eq = eq_val
            if eq_val > running_max_eq:
                running_max_eq = eq_val
            dd = eq_val - running_max_eq
            if dd < max_dd_abs:
                max_dd_abs = dd
            if position != 0:
                total_bars_held += 1
        else:
            equity_rows.append({"timestamp": timestamps[i], "equity": eq_val, "position": position})

        # Signal for next bar
        if np.isnan(mid_a[i]) or is_last_bar[i]:
            desired = 0
        elif position != 0:
            bars_held   = i - entry_bar
            hit_midband = (position > 0 and closes[i] >= mid_a[i]) or \
                          (position < 0 and closes[i] <= mid_a[i])
            if hit_midband or bars_held >= time_stop_bars:
                desired = 0
            else:
                desired = position
        elif session_a[i] and ranging_a[i]:
            if closes[i] < lower_a[i]:
                desired = position_size
            elif closes[i] > upper_a[i]:
                desired = -position_size
            else:
                desired = 0
        else:
            desired = 0

    # --- Return ---
    if headless:
        daily_eq_vals.append(_last_eq)
        pnl    = _last_eq - initial_cash
        t0     = pd.to_datetime(timestamps[0],  utc=True)
        t1     = pd.to_datetime(timestamps[-1], utc=True)
        n_days = max((t1 - t0).days, 1)

        if peak_exposure > 0 and len(daily_eq_vals) >= 2:
            daily_pnl = np.diff(np.array(daily_eq_vals, dtype=float))
            daily_ret = daily_pnl / peak_exposure
            mean_r    = float(daily_ret.mean())
            std_r     = float(daily_ret.std())
            sharpe    = mean_r / std_r * np.sqrt(252) if std_r > 0 else 0.0
            neg       = daily_ret[daily_ret < 0]
            s_std     = float(neg.std()) if len(neg) > 1 else 0.0
            sortino   = mean_r / s_std * np.sqrt(252) if s_std > 0 else 0.0
            cagr      = (1 + pnl / peak_exposure) ** (365 / n_days) - 1
            max_dd    = max_dd_abs / peak_exposure
        else:
            sharpe = sortino = cagr = max_dd = 0.0

        avg_bars_held = total_bars_held / n_entries if n_entries > 0 else 0.0

        return {
            "pnl":           round(pnl, 2),
            "sharpe":        round(sharpe, 4),
            "sortino":       round(sortino, 4),
            "max_dd":        round(max_dd, 4),
            "cagr":          round(cagr, 4),
            "n_trades":      n_trades,
            "avg_bars_held": round(avg_bars_held, 2),
            "turnover":      round(turnover, 2),
            "total_fees":    round(total_fees, 2),
            "peak_exposure": round(peak_exposure, 2),
        }
    else:
        return {
            "equity":       pd.DataFrame(equity_rows),
            "trades":       pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(columns=["timestamp", "qty", "price", "fee"]),
            "total_fees":   total_fees,
            "peak_exposure": peak_exposure,
        }
