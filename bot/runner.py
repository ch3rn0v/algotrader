from datetime import datetime, timezone
from pathlib import Path

from candles import get_candles  # also loads bot/.env
from backtest_mean_rev_bb import run_backtest
from charts import plot_results

FIGI      = "BBG0047315Y7"  # SBERP
TIMEFRAME = "5min"
FROM      = datetime(2025, 1, 1, tzinfo=timezone.utc)
TO        = datetime(2026, 1, 1, tzinfo=timezone.utc)

candles = get_candles(FIGI, TIMEFRAME, FROM, TO)
print(f"Candles loaded: {len(candles)}")

results       = run_backtest(candles, bb_period=20, bb_std=2.0, time_stop_bars=12,
                             session_start_utc=9, session_end_utc=12)
equity        = results["equity"]
trades        = results["trades"]
peak_exposure = results["peak_exposure"]

pnl = equity["equity"].iloc[-1] - equity["equity"].iloc[0]
print(f"Trades:         {len(trades)}")
print(f"Total P&L:      {pnl:+,.2f}")
print(f"Peak exposure:  {peak_exposure:,.2f}")
print(f"Return:         {pnl/peak_exposure*100:+.2f}%" if peak_exposure > 0 else "Return: n/a")

out_dir = Path(__file__).parent / "outputs" / "backtest"
out_dir.mkdir(parents=True, exist_ok=True)
plot_results(candles, equity, trades, peak_exposure, symbol="SBERP", timeframe=TIMEFRAME,
             path=out_dir / "result.png")
