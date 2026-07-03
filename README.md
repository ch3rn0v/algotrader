# algotrader

A Python algorithmic trading system for MOEX via the T-Bank (ex-Tinkoff) Invest API.
Includes a Bollinger Band mean-reversion backtest, a grid optimizer, and a LightGBM price-prediction model.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/RussianInvestments/invest-python.git --no-deps

# For Mac OS: `brew install libomp`
```

Create `bot/.env` with your T-Bank token (never commit this file):

```
TBANK_TOKEN=t.your_token_here
```

The token is loaded automatically at import time by `config.py`.

## Files

| File | Purpose |
|---|---|
| `bot/config.py` | Shared config: assets, timeframes, date range, paths, `.env` loading |
| `bot/candles.py` | Fetch OHLCV candles from T-Bank API with local CSV cache (`bot/cache/`) |
| `bot/features.py` | Feature engineering for the model (no-lookahead multi-timeframe merge) |
| `bot/model.py` | Load the latest trained model and build predictions aligned to a candle series |
| `bot/backtest_mean_rev_bb.py` | Bollinger Band mean-reversion backtest; supports headless mode for grid search |
| `bot/charts.py` | Four-panel chart: asset price+volume, equity curve, position, trading volume |
| `bot/runner.py` | Run a single backtest and save chart to `outputs/backtest/` |
| `bot/optimizer.py` | Parallel grid search over BB parameters; saves CSV + HTML to `outputs/optimizer/` |
| `bot/train_lgbm.py` | Train a LightGBM model to predict the next 5-min close; saves model to `outputs/models/` |
| `bot/diagnostics.py` | Price/BB/signals/equity chart for a specific time window; saves to `outputs/diagnostics/` |
| `bot/find_figi.py` | Look up instrument FIGIs by ticker via the T-Bank API |

## Usage

All commands run from the repo root with the venv activated.

**Single backtest:**
```bash
python3 bot/runner.py
```

**Grid optimizer:**
```bash
python3 bot/optimizer.py
python3 bot/optimizer.py --figi BBG004730N88 --timeframe 15min --from 2024-01-01 --to 2025-01-01
python3 bot/optimizer.py --jobs 4 --out outputs/my_run
```

**Train LightGBM model:**
```bash
python3 bot/train_lgbm.py
```

**Look up a FIGI:**
```bash
python3 bot/find_figi.py SBERP
python3 bot/find_figi.py PLZL
python3 bot/find_figi.py IMOEX
```

## Strategy

Bollinger Band mean reversion, targeting the choppy MOEX midday session (12:00–15:00 MSK = 09:00–12:00 UTC):

- Enter long when close < lower band; enter short when close > upper band
- Only enter when BB width is below its rolling median (ranging market)
- Exit at the midband, after `time_stop_bars`, or at session end
- Flatten all positions at end of day
- Execution: signal on close, fill at next bar's open (market taker)
- Fee: 0.05% per side (T-Bank Trader tariff + MOEX exchange fee — verify before live use)

## No-lookahead discipline

- Backtest: signals computed on bar close, executed at next bar's open
- LightGBM features: primary 5m series uses close[t] to predict close[t+1]; all other series (other assets or higher timeframes) are merged on bar-end timestamp so only completed bars are visible at each decision point

## Outputs

All outputs are git-ignored.

| Path | Contents |
|---|---|
| `bot/cache/` | Candle CSV cache, one file per FIGI+timeframe |
| `bot/outputs/backtest/` | `result.png` from `runner.py` |
| `bot/outputs/optimizer/` | `results_<ticker>_<tf>.{csv,html}` from `optimizer.py` |
| `bot/outputs/models/` | `lgbm_<timestamp>.txt` + `lgbm_<timestamp>_meta.json` from `train_lgbm.py` |
| `bot/outputs/diagnostics/` | `diagnostics_<from>_<to>.png` from `diagnostics.py` |
