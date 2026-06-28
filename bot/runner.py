"""Model-guided backtest runner.

Workflow:
  1. Load the latest LightGBM model and its metadata.
  2. Load candles for all assets/timeframes used during training.
  3. Build features and generate a prediction for every primary 5m bar.
  4. Isolate the test period (bars after the training cutoff).
  5. Run the Bollinger Band backtest with model predictions as a directional
     filter on new entries: only enter long if pred > 1.0, short if pred < 1.0.
  6. Report results and save chart.
"""

import json
from pathlib import Path

import lightgbm as lgbm
import numpy as np
import pandas as pd

from backtest_mean_rev_bb import run_backtest
from candles import get_candles
from charts import plot_results
from train_lgbm import ASSETS, FROM, MODEL_DIR, PRIMARY_ASSET, PRIMARY_TF, TIMEFRAMES, TO, build_features

# ---------------------------------------------------------------------------
# Backtest parameters
# ---------------------------------------------------------------------------

BB_PERIOD = 20
BB_STD = 2.0
TIME_STOP_BARS = 12
SESSION_START_UTC = 9
SESSION_END_UTC = 12

OUT_DIR = Path(__file__).parent / "outputs" / "backtest"

# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------

meta_files = sorted(MODEL_DIR.glob("lgbm_*_meta.json"))
if not meta_files:
    raise FileNotFoundError(f"No model metadata found in {MODEL_DIR}. Run train_lgbm.py first.")

meta_path = meta_files[-1]
model_path = MODEL_DIR / meta_path.name.replace("_meta.json", ".txt")
meta = json.loads(meta_path.read_text())
feature_cols = meta["feature_cols"]
train_end_ts = pd.Timestamp(meta["train_end_ts"])

model = lgbm.Booster(model_file=str(model_path))
print(f"Model:         {model_path.name}")
print(f"Train cutoff:  {train_end_ts}")

# ---------------------------------------------------------------------------
# 2. Load candles for all assets and timeframes
# ---------------------------------------------------------------------------

print("\nLoading candles...")
all_candles = {}
for asset in ASSETS:
    for tf in TIMEFRAMES:
        df = get_candles(ASSETS[asset], tf, FROM, TO)
        all_candles[(asset, tf)] = df
        print(f"  {asset} {tf}: {len(df)} bars")

# ---------------------------------------------------------------------------
# 3. Build features and generate predictions
# ---------------------------------------------------------------------------

print("\nBuilding features...")
features = build_features(all_candles)

null_cols = [c for c in features.columns if features[c].isna().all()]
if null_cols:
    print(f"  Dropping {len(null_cols)} entirely-null columns: {null_cols}")
    features = features.drop(columns=null_cols)

missing_cols = [c for c in feature_cols if c not in features.columns]
if missing_cols:
    raise RuntimeError(
        f"Feature columns from training are missing in current data: {missing_cols}. "
        f"Ensure all training assets have data in the date range."
    )

# features rows are aligned 1-to-1 with primary candles (build_features uses a left join).
primary = all_candles[(PRIMARY_ASSET, PRIMARY_TF)].sort_values("timestamp").reset_index(drop=True)
assert len(features) == len(primary), "Feature matrix row count does not match primary candles."

# LightGBM handles NaN in input (warmup rows) via its missing-value branches.
predictions = model.predict(features[feature_cols])
print(f"Predictions:   {len(predictions)} bars, mean={predictions.mean():.5f}")

# ---------------------------------------------------------------------------
# 4. Isolate test period
# ---------------------------------------------------------------------------

test_mask = primary["timestamp"] > train_end_ts
test_candles = primary[test_mask].reset_index(drop=True)
test_predictions = predictions[test_mask.to_numpy()]

if len(test_candles) == 0:
    raise RuntimeError("Test period is empty. Check train_end_ts in the model metadata.")

print(f"\nTest period:   {len(test_candles)} bars")
print(f"  from: {test_candles['timestamp'].iloc[0]}")
print(f"  to:   {test_candles['timestamp'].iloc[-1]}")

# ---------------------------------------------------------------------------
# 5. Run backtest
# ---------------------------------------------------------------------------

results = run_backtest(
    test_candles,
    bb_period=BB_PERIOD,
    bb_std=BB_STD,
    time_stop_bars=TIME_STOP_BARS,
    session_start_utc=SESSION_START_UTC,
    session_end_utc=SESSION_END_UTC,
    predictions=test_predictions,
)

# ---------------------------------------------------------------------------
# 6. Report
# ---------------------------------------------------------------------------

equity = results["equity"]
trades = results["trades"]
peak_exposure = results["peak_exposure"]

pnl = equity["equity"].iloc[-1] - equity["equity"].iloc[0]
print(f"\nTrades:        {len(trades)}")
print(f"Total P&L:     {pnl:+,.2f}")
print(f"Peak exposure: {peak_exposure:,.2f}")
if peak_exposure > 0:
    print(f"Return:        {pnl / peak_exposure * 100:+.2f}%")

OUT_DIR.mkdir(parents=True, exist_ok=True)
plot_results(
    test_candles, equity, trades, peak_exposure,
    symbol="SBERP (test)", timeframe="5min",
    path=OUT_DIR / "result.png",
)
