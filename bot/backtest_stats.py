"""Shared performance-stat computation for the backtests.

Kept separate so the z-score and bounds strategies report identical metrics.
"""
import numpy as np
import pandas as pd


def compute_stats(
    equity: np.ndarray,
    pos_hist: np.ndarray,
    trades: pd.DataFrame,
    timestamps: np.ndarray,
    is_last_bar: np.ndarray,
    peak_exposure: float,
    total_fees: float,
    initial_cash: float,
) -> dict:
    """Summarize an equity/position path. Returns/drawdown are scaled by
    peak_exposure so they read as returns on capital-at-risk, not on the
    (arbitrary) initial cash."""
    n = len(equity)
    pnl = equity[-1] - initial_cash
    t0 = pd.to_datetime(timestamps[0], utc=True)
    t1 = pd.to_datetime(timestamps[-1], utc=True)
    n_days = max((t1 - t0).days, 1)

    n_entries = int(np.sum((pos_hist[1:] != 0) & (pos_hist[:-1] == 0)))
    total_bars_held = int(np.sum(pos_hist != 0))
    avg_bars_held = total_bars_held / n_entries if n_entries > 0 else 0.0
    pct_bars_in_pos = total_bars_held / n * 100 if n > 0 else 0.0
    turnover = float((trades["qty"].abs() * trades["price"]).sum()) if len(trades) else 0.0

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
