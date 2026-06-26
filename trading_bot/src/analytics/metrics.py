"""Metrics over the canonical DataFrame (SPEC 11.1). Pure.

Sharpe/Sortino are computed on *daily*-aggregated equity (last equity per
session day), with simple daily returns and the daily simple risk-free rate
subtracted, then annualized by sqrt(252). 15-min returns are autocorrelated and
intraday-seasonal; annualizing at that frequency inflates the ratio. The
``warmup_invalid`` prefix is excluded from every computation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS = 252


def _evaluation_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "warmup_invalid" in df.columns:
        df = df.loc[~df["warmup_invalid"].astype(bool)]
    return df.sort_values("timestamp").reset_index(drop=True)


def _daily_equity(df: pd.DataFrame) -> pd.Series:
    ts = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index(ts)["equity"].groupby(lambda t: t.normalize()).last()


def _max_drawdown(equity: pd.Series) -> tuple[float, int]:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0
    # Longest run of consecutive days below a prior peak.
    below = (drawdown < 0).to_numpy()
    longest = current = 0
    for flag in below:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return max_dd, int(longest)


def _trade_stats(df: pd.DataFrame, lot_size: int) -> dict:
    trades = df.loc[df["trade_qty_from_prev"] != 0.0]
    notional = (trades["trade_qty_from_prev"].abs() * trades["fill_price_from_prev"] * lot_size).sum()
    total_fees = float(df["fees_from_prev"].fillna(0).sum())

    # Holding spans and realized PnL per round trip, derived from position changes.
    position = df["position"].to_numpy()
    holding_bars: list[int] = []
    trade_pnls: list[float] = []
    entry_idx = None
    for i in range(len(df)):
        prev = position[i - 1] if i > 0 else 0.0
        if prev == 0.0 and position[i] != 0.0:
            entry_idx = i
        elif prev != 0.0 and position[i] == 0.0 and entry_idx is not None:
            holding_bars.append(i - entry_idx)
            segment = df.iloc[entry_idx : i + 1]
            trade_pnls.append(float(segment["equity"].iloc[-1] - segment["equity"].iloc[0]))
            entry_idx = None

    wins = [p for p in trade_pnls if p > 0]
    win_rate = len(wins) / len(trade_pnls) if trade_pnls else 0.0
    avg_holding = float(np.mean(holding_bars)) if holding_bars else 0.0
    losing_streak = current = 0
    for p in trade_pnls:
        current = current + 1 if p < 0 else 0
        losing_streak = max(losing_streak, current)

    return {
        "trade_count": int(len(trades)),
        "win_rate": float(win_rate),
        "avg_holding_bars": avg_holding,
        "avg_trade_size_lots": float(trades["trade_qty_from_prev"].abs().mean()) if len(trades) else 0.0,
        "total_notional": float(notional),
        "total_fees": total_fees,
        "longest_losing_streak_trades": int(losing_streak),
    }


def compute_metrics(df: pd.DataFrame, instrument, config: dict, *, run_id: str, split: str) -> dict:
    """Return the flat ``metrics.json`` object (SPEC 11.1)."""
    rows = _evaluation_rows(df)
    daily_equity = _daily_equity(rows)
    initial_cash = float(config["backtest"]["initial_cash"])
    final_equity = float(daily_equity.iloc[-1]) if len(daily_equity) else initial_cash
    total_return = final_equity / initial_cash - 1.0

    session_days = int(len(daily_equity))
    years = session_days / _TRADING_DAYS if session_days else 0.0
    cagr = (final_equity / initial_cash) ** (1 / years) - 1.0 if years > 0 else 0.0

    daily_ret = daily_equity.pct_change().dropna()
    rf_annual = float(config["metrics"]["risk_free_annual"])
    rf_daily = (1 + rf_annual) ** (1 / _TRADING_DAYS) - 1.0
    excess = daily_ret - rf_daily
    std = float(excess.std())
    sharpe = float(excess.mean() / std * np.sqrt(_TRADING_DAYS)) if std > 0 else 0.0
    downside = excess[excess < 0]
    dstd = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = float(excess.mean() / dstd * np.sqrt(_TRADING_DAYS)) if dstd > 0 else 0.0

    max_dd, dd_days = _max_drawdown(daily_equity)
    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    trade = _trade_stats(rows, instrument.lot_size)
    total_commission_bps = trade["total_fees"] / trade["total_notional"] * 10_000 if trade["total_notional"] else 0.0
    start_date = pd.Timestamp(rows["timestamp"].iloc[0]).date().isoformat() if len(rows) else ""
    end_date = pd.Timestamp(rows["timestamp"].iloc[-1]).date().isoformat() if len(rows) else ""

    return {
        "run_id": run_id,
        "split": split,
        "start_date": start_date,
        "end_date": end_date,
        "session_days": session_days,
        "initial_cash": initial_cash,
        "final_equity": final_equity,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": float(calmar),
        "max_drawdown": float(max_dd),
        "max_drawdown_duration_days": dd_days,
        "longest_losing_streak_trades": trade["longest_losing_streak_trades"],
        "trade_count": trade["trade_count"],
        "win_rate": trade["win_rate"],
        "avg_holding_bars": trade["avg_holding_bars"],
        "avg_trade_size_lots": trade["avg_trade_size_lots"],
        "total_notional": trade["total_notional"],
        "total_fees": trade["total_fees"],
        "total_commission_bps": float(total_commission_bps),
    }
