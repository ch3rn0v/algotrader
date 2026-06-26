"""Compute backtest stats and render the four-panel chart."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(
    candles: pd.DataFrame,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    peak_exposure: float,
    symbol: str = "",
    timeframe: str = "",
    path: Path = Path("result.png"),
) -> None:
    equity = equity.copy()
    equity["timestamp"] = pd.to_datetime(equity["timestamp"])

    eq  = equity["equity"]
    ts  = equity["timestamp"]
    pos = equity["position"].to_numpy()

    def _annualised(series, downside_only=False):
        s = series[series < 0] if downside_only else series
        std = s.std()
        return (series.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    # --- Strategy stats ---
    pnl    = eq.iloc[-1] - eq.iloc[0]
    n_days = max((ts.iloc[-1] - ts.iloc[0]).days, 1)

    if peak_exposure > 0:
        ret_pct = pnl / peak_exposure * 100
        cagr    = (1 + pnl / peak_exposure) ** (365 / n_days) - 1
        daily_eq  = equity.set_index("timestamp")["equity"].resample("D").last().ffill().dropna()
        daily_ret = daily_eq.diff().dropna() / peak_exposure
        running_max = eq.cummax()
        max_dd      = (eq - running_max).min() / peak_exposure
    else:
        ret_pct = cagr = max_dd = 0.0
        daily_ret = pd.Series(dtype=float)

    sharpe  = _annualised(daily_ret)
    sortino = _annualised(daily_ret, downside_only=True)

    gross_profit  = daily_ret[daily_ret > 0].sum()
    gross_loss    = abs(daily_ret[daily_ret < 0].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    entries       = int(np.sum((pos[1:] != 0) & (pos[:-1] == 0)))
    holding_bars  = int(np.sum(pos != 0))
    avg_bars_held = (holding_bars / entries) if entries > 0 else 0.0

    total_lots = int(trades["qty"].abs().sum()) if not trades.empty else 0
    total_rub  = (trades["qty"].abs() * trades["price"]).sum() if not trades.empty else 0.0

    # --- Buy & Hold stats (from raw candles) ---
    can = candles.copy()
    can["ts_naive"] = pd.to_datetime(can["timestamp"], utc=True).dt.tz_convert(None)
    can["date"]     = can["ts_naive"].dt.normalize()
    price_ts = can["ts_naive"]
    close    = can["close"]

    bh_n_days  = max((price_ts.iloc[-1] - price_ts.iloc[0]).days, 1)
    bh_pnl_pct = (close.iloc[-1] / close.iloc[0] - 1) * 100
    bh_cagr    = (close.iloc[-1] / close.iloc[0]) ** (365 / bh_n_days) - 1

    daily_close  = can.groupby("date")["close"].last()
    bh_daily_ret = daily_close.pct_change().dropna()
    bh_sharpe    = _annualised(bh_daily_ret)
    bh_sortino   = _annualised(bh_daily_ret, downside_only=True)
    bh_run_max   = daily_close.cummax()
    bh_max_dd    = ((daily_close - bh_run_max) / bh_run_max).min()

    daily_vol = can.groupby("date")["volume"].sum().reset_index()

    # --- Titles ---
    label = f"{symbol} {timeframe}".strip()
    title_bh = (
        f"{label} — Buy & Hold\n"
        f"PnL: {bh_pnl_pct:+.2f}%"
        f" | Sharpe: {bh_sharpe:.2f}"
        f" | Sortino: {bh_sortino:.2f}"
        f" | Max DD: {bh_max_dd*100:.2f}%"
        f" | CAGR: {bh_cagr*100:.2f}%"
    )
    title_equity = (
        f"{label} — BB mean reversion\n"
        f"PnL: {pnl:+,.0f} ({ret_pct:+.2f}%)"
        f" | Sharpe: {sharpe:.2f}"
        f" | Sortino: {sortino:.2f}"
        f" | Max DD: {max_dd*100:.2f}%"
        f" | CAGR: {cagr*100:.2f}%"
        f" | PF: {profit_factor:.2f}"
    )
    title_pos = (
        f"Position\n"
        f"N trades: {len(trades):,} | Avg bars held: {avg_bars_held:.1f}"
    )
    title_vol = (
        f"Volume\n"
        f"N lots traded: {total_lots:,} | Turnover, rub.: {total_rub:,.0f}"
    )

    fig, axes = plt.subplots(
        4, 1, figsize=(12, 12), sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 1, 1]},
    )
    ax0, ax1, ax2, ax3 = axes

    # Panel 0: asset price + daily market volume
    ax0_vol = ax0.twinx()
    ax0_vol.bar(daily_vol["date"], daily_vol["volume"], width=0.8, color="gray", alpha=0.25, zorder=1)
    ax0_vol.set_ylabel("Market volume (shares/day)")
    ax0.plot(price_ts, close, color="steelblue", lw=0.8, zorder=2)
    ax0.set_ylabel("Price (RUB)")
    ax0.set_title(title_bh, loc="left")
    ax0.margins(0)
    ax0.grid(True, lw=0.4)

    # Panel 1: strategy equity curve
    ax1.plot(ts, eq, color="steelblue", lw=1.0)
    ax1.set_ylabel("Equity (RUB)")
    ax1.set_title(title_equity, loc="left")
    ax1.margins(0)
    ax1.grid(True, lw=0.4)

    # Panel 2: position
    ax2.step(ts, equity["position"], where="post")
    ax2.axhline(0, color="gray", lw=0.7)
    ax2.set_ylabel("Position (lots)")
    ax2.set_title(title_pos, loc="left")
    ax2.margins(0)
    ax2.grid(True, lw=0.4)

    # Panel 3: strategy daily trading volume
    ax3_rub = ax3.twinx()
    if not trades.empty:
        t = trades.copy()
        t["date"]     = pd.to_datetime(t["timestamp"], utc=True).dt.tz_convert(None).dt.normalize()
        t["vol_lots"] = t["qty"].abs()
        t["vol_rub"]  = t["vol_lots"] * t["price"]
        dv = t.groupby("date")[["vol_lots", "vol_rub"]].sum().reset_index()
        ax3.bar(dv["date"], dv["vol_lots"], width=0.8, color="steelblue", label="lots")
        ax3_rub.bar(dv["date"], dv["vol_rub"], width=0.8, color="tomato", alpha=0.5, label="rubles")
    ax3.set_ylabel("Volume (lots)")
    ax3_rub.set_ylabel("Volume (RUB)")
    ax3.set_title(title_vol, loc="left")
    ax3.margins(0)
    ax3.grid(True, lw=0.4)
    handles = [plt.Line2D([0], [0], color="steelblue", lw=4),
               plt.Line2D([0], [0], color="tomato", lw=4, alpha=0.5)]
    ax3.legend(handles, ["lots", "rubles"], loc="upper left")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Chart saved to {path}")
