"""Compute backtest stats and render the three-panel chart."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_results(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    symbol: str = "",
    timeframe: str = "",
    path: Path = Path("result.png"),
) -> None:
    equity = equity.copy()
    equity["timestamp"] = pd.to_datetime(equity["timestamp"])

    eq  = equity["equity"]
    ts  = equity["timestamp"]
    pos = equity["position"].to_numpy()

    initial = eq.iloc[0]
    final   = eq.iloc[-1]
    pnl     = final - initial
    ret_pct = (final / initial - 1) * 100

    n_days = max((ts.iloc[-1] - ts.iloc[0]).days, 1)
    cagr   = (final / initial) ** (365 / n_days) - 1

    daily_eq  = equity.set_index("timestamp")["equity"].resample("D").last().ffill().dropna()
    daily_ret = daily_eq.pct_change().dropna()

    def _annualised(series, downside_only=False):
        s = series[series < 0] if downside_only else series
        std = s.std()
        return (series.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    sharpe  = _annualised(daily_ret)
    sortino = _annualised(daily_ret, downside_only=True)

    running_max   = eq.cummax()
    max_dd        = ((eq - running_max) / running_max).min()
    gross_profit  = daily_ret[daily_ret > 0].sum()
    gross_loss    = abs(daily_ret[daily_ret < 0].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # Avg bars held per trade: count bars per continuous non-zero position segment
    entries       = int(np.sum((pos[1:] != 0) & (pos[:-1] == 0)))
    holding_bars  = int(np.sum(pos != 0))
    avg_bars_held = (holding_bars / entries) if entries > 0 else 0.0

    total_lots = int(trades["qty"].abs().sum()) if not trades.empty else 0
    total_rub  = (trades["qty"].abs() * trades["price"]).sum() if not trades.empty else 0.0

    label = f"{symbol} {timeframe}".strip()
    title_equity = (
        f"{label} — EWMA crossover\n"
        f"PnL: {pnl:+,.0f} ({ret_pct:+.1f}%)"
        f" | Sharpe: {sharpe:.2f}"
        f" | Sortino: {sortino:.2f}"
        f" | Max DD: {max_dd*100:.1f}%"
        f" | CAGR: {cagr*100:.1f}%"
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

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    ax1.plot(ts, eq)
    ax1.set_ylabel("Equity")
    ax1.set_title(title_equity, loc="left")
    ax1.margins(0)
    ax1.grid(True, lw=0.4)

    ax2.step(ts, equity["position"], where="post")
    ax2.axhline(0, color="gray", lw=0.7)
    ax2.set_ylabel("Position (lots)")
    ax2.set_title(title_pos, loc="left")
    ax2.margins(0)
    ax2.grid(True, lw=0.4)

    ax3_rub = ax3.twinx()
    if not trades.empty:
        t = trades.copy()
        t["date"]     = pd.to_datetime(t["timestamp"]).dt.tz_localize(None).dt.normalize()
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
