"""Model-guided backtest runner.

Workflow:
  1. Load best params from the latest optimizer results CSV.
  2. Load the latest LightGBM model (no retraining).
  3. Load candles, build features, generate predictions.
  4. Isolate the test period (bars after train cutoff) and run the backtest.
  5. Report results and save chart.

Usage:
    python3 bot/runner.py                          # best params from optimizer
    python3 bot/runner.py --bb-alpha 0.05          # override specific params
    python3 bot/runner.py --bb-std 2.0 --session-end-utc 14
"""

import argparse
import json
from datetime import datetime, timezone
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
# Args — all optional; any provided value overrides the optimizer's best
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Run backtest with best optimizer params")
parser.add_argument("--bb-alpha",             type=float)
parser.add_argument("--bb-std",               type=float)
parser.add_argument("--time-stop-bars",        type=int)
parser.add_argument("--width-alpha",           type=float)
parser.add_argument("--session-start-utc",     type=int)
parser.add_argument("--session-end-utc",       type=int)
parser.add_argument("--position-size",         type=int)
parser.add_argument("--pred-long-threshold",   type=float)
parser.add_argument("--pred-short-threshold",  type=float)
args = parser.parse_args()

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

overrides = {k: v for k, v in vars(args).items() if v is not None}
bt_params.update(overrides)

source = "optimizer" if not overrides else f"optimizer + CLI overrides: {list(overrides.keys())}"
print(f"Params ({source}, sortino={best['sortino']:.4f}):")
for k, v in bt_params.items():
    marker = " *" if k in overrides else ""
    print(f"  {k}: {v}{marker}")

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

ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
OUT_DIR.mkdir(parents=True, exist_ok=True)
plot_results(
    bt_candles, equity, trades, peak_exposure,
    symbol=f"{PRIMARY_ASSET} (test)", timeframe=PRIMARY_TF,
    path=OUT_DIR / f"result_{ts}.png",
)
