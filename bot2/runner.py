import os
from datetime import datetime, timezone
from pathlib import Path

from candles import get_candles
from backtest import run_backtest
from charts import plot_results

# Load .env from trading_bot/
_env = Path(__file__).parent.parent / "trading_bot" / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

FIGI      = "BBG0047315Y7"  # SBERP
TIMEFRAME = "5min"
FROM      = datetime(2025, 1, 1, tzinfo=timezone.utc)
TO        = datetime(2026, 1, 1, tzinfo=timezone.utc)

candles = get_candles(FIGI, TIMEFRAME, FROM, TO)
print(f"Candles loaded: {len(candles)}")

results = run_backtest(candles, fast_span=20, slow_span=50)
equity  = results["equity"]
trades  = results["trades"]

print(f"Trades: {len(trades)}")
print(f"Final equity: {equity['equity'].iloc[-1]:,.2f}")

plot_results(equity, trades, symbol="SBERP", timeframe=TIMEFRAME,
             path=Path(__file__).parent / "result.png")
