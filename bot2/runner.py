import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from candles import get_candles
from backtest import run_backtest

# Load .env from trading_bot/
_env = Path(__file__).parent.parent / "trading_bot" / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

FIGI = "BBG0047315Y7"  # SBERP
TIMEFRAME = "15min"
FROM = datetime(2025, 1, 1, tzinfo=timezone.utc)
TO = datetime(2026, 1, 1, tzinfo=timezone.utc)

candles = get_candles(FIGI, TIMEFRAME, FROM, TO)
print(f"Candles loaded: {len(candles)}")

results = run_backtest(candles, fast_span=20, slow_span=50)
equity = results["equity"]
trades = results["trades"]

initial = equity["equity"].iloc[0]
final = equity["equity"].iloc[-1]
ret_pct = (final / initial - 1) * 100

print(f"Trades:       {len(trades)}")
print(f"Final equity: {final:,.2f}")
print(f"Return:       {ret_pct:.2f}%")
if not trades.empty:
    print(trades.to_string(index=False))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax1.plot(equity["timestamp"], equity["equity"])
ax1.set_ylabel("Equity")
ax1.set_title(f"SBERP EWMA crossover — {TIMEFRAME}")
ax2.step(equity["timestamp"], equity["position"], where="post")
ax2.set_ylabel("Position (lots)")
ax2.axhline(0, color="gray", lw=0.7)
fig.tight_layout()
fig.savefig(Path(__file__).parent / "result.png", dpi=120)
print("Chart saved to result.png")
