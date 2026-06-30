"""Model-guided backtest runner.

Workflow:
  1. Load best params from the latest optimizer results CSV.
  2. Load the latest LightGBM model (no retraining).
  3. Load candles, build features, generate predictions.
  4. Isolate the test period (bars after train cutoff) and run the backtest.
  5. Report results and save chart.

Usage:
    python3 bot/runner.py
"""

import json
from pathlib import Path

import lightgbm as lgbm
import pandas as pd

from backtest_mean_rev_bb import run_backtest
from candles import get_candles
from charts import plot_results
from optimizer import RESULT_COLS
from train_lgbm import ASSETS, FROM, MODEL_DIR, PRIMARY_ASSET, PRIMARY_TF, TIMEFRAMES, TO, build_features

OPTIMIZER_OUT = Path(__file__).parent / "outputs" / "optimizer"
OUT_DIR = Path(__file__).parent / "outputs" / "backtest"

# ---------------------------------------------------------------------------
# 1. Load best params from latest optimizer run
# ---------------------------------------------------------------------------

figi = ASSETS[PRIMARY_ASSET]
csv_path = OPTIMIZER_OUT / f"results_{figi}_{PRIMARY_TF}.csv"
if not csv_path.exists():
    raise FileNotFoundError(f"No optimizer results at {csv_path}. Run optimizer.py first.")

opt_df = pd.read_csv(csv_path)  # already sorted by sortino descending
best = opt_df.iloc[0]
bt_params = {k: best[k] for k in opt_df.columns if k not in RESULT_COLS}

print(f"Best params (sortino={best['sortino']:.4f}):")
for k, v in bt_params.items():
    print(f"  {k}: {v}")

# ---------------------------------------------------------------------------
# 2. Load model
# ---------------------------------------------------------------------------

meta_files = sorted(MODEL_DIR.glob("lgbm_*_meta.json"))
if not meta_files:
    raise FileNotFoundError(f"No model found in {MODEL_DIR}. Run train_lgbm.py first.")

meta_path = meta_files[-1]
model_path = MODEL_DIR / meta_path.name.replace("_meta.json", ".txt")
meta = json.loads(meta_path.read_text())
feature_cols = meta["feature_cols"]
train_end_ts = pd.Timestamp(meta["train_end_ts"])

model = lgbm.Booster(model_file=str(model_path))
print(f"\nModel:        {model_path.name}")
print(f"Train cutoff: {train_end_ts}")

# ---------------------------------------------------------------------------
# 3. Load candles
# ---------------------------------------------------------------------------

print("\nLoading candles...")
all_candles = {}
for asset in ASSETS:
    for tf in TIMEFRAMES:
        df = get_candles(ASSETS[asset], tf, FROM, TO)
        all_candles[(asset, tf)] = df
        print(f"  {asset} {tf}: {len(df)} bars")

# ---------------------------------------------------------------------------
# 4. Build features and predict
# ---------------------------------------------------------------------------

print("\nBuilding features...")
features = build_features(all_candles)

null_cols = [c for c in features.columns if features[c].isna().all()]
if null_cols:
    features = features.drop(columns=null_cols)

missing_cols = [c for c in feature_cols if c not in features.columns]
if missing_cols:
    raise RuntimeError(f"Feature columns missing from current data: {missing_cols}")

primary = all_candles[(PRIMARY_ASSET, PRIMARY_TF)].sort_values("timestamp").reset_index(drop=True)
predictions = model.predict(features[feature_cols])
print(f"Predictions:  {len(predictions)} bars, mean={predictions.mean():.5f}")

# ---------------------------------------------------------------------------
# 5. Run backtest on test period
# ---------------------------------------------------------------------------

bt_mask = primary["timestamp"] > train_end_ts
bt_candles = primary[bt_mask].reset_index(drop=True)
bt_predictions = predictions[bt_mask.to_numpy()]

if len(bt_candles) == 0:
    raise RuntimeError("Test period is empty. Check train_end_ts in model metadata.")

print(f"\nTest period: {len(bt_candles)} bars")
print(f"  from: {bt_candles['timestamp'].iloc[0]}")
print(f"  to:   {bt_candles['timestamp'].iloc[-1]}")

result = run_backtest(bt_candles, predictions=bt_predictions, **bt_params)

equity = result["equity"]
trades = result["trades"]
peak_exposure = result["peak_exposure"]

pnl = equity["equity"].iloc[-1] - equity["equity"].iloc[0]
print(f"\nTrades:        {len(trades)}")
print(f"Total P&L:     {pnl:+,.2f}")
print(f"Peak exposure: {peak_exposure:,.2f}")
if peak_exposure > 0:
    print(f"Return:        {pnl / peak_exposure * 100:+.2f}%")

# ---------------------------------------------------------------------------
# 6. Save chart
# ---------------------------------------------------------------------------

OUT_DIR.mkdir(parents=True, exist_ok=True)
plot_results(
    bt_candles, equity, trades, peak_exposure,
    symbol=f"{PRIMARY_ASSET} (test)", timeframe=PRIMARY_TF,
    path=OUT_DIR / "result.png",
)
